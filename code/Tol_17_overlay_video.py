# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N17 — Mask-overlay time-lapse, one video per day
# MAGIC
# MAGIC **Why this is the deciding artifact.** Every metric in this study shares one blind spot: it
# MAGIC cannot tell "tracked the right calf" from "tracked the wrong calf smoothly". Coverage %, link
# MAGIC IoU and chain continuity all look perfect when SAM3 holds a neighbour the whole time. That
# MAGIC failure has already occurred once here (the Tol2 drinking-tank track, 100 % coverage, wrong
# MAGIC object), and three separate embedding results in this project have since turned out to be
# MAGIC measurement artifacts. Only looking at the masks settles it.
# MAGIC
# MAGIC **Time-lapse, not real time.** A day is ~18 h; at 2 s of wall-clock per output frame and 25 fps
# MAGIC playback that becomes **~22 min** — 50x faster, watchable end to end, and identity switches are
# MAGIC *more* visible compressed than in real time because a colour jumping between animals stands out.
# MAGIC At 640x360 a day lands around 60-90 MB, so all seven download comfortably.
# MAGIC
# MAGIC Each calf gets a fixed colour, drawn as a translucent mask fill plus a solid outline, with the
# MAGIC wall-clock stamped on every frame so any switch can be reported back as a timestamp.
# MAGIC
# MAGIC ⚠ Writes the mp4 on `/local_disk0` and copies it at the end — UC Volumes reject seek-on-write
# MAGIC and every mp4 muxer rewinds to patch its header on close.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "opencv-python-headless==4.11.0.86"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, re, json, gzip, glob, time, shutil
from datetime import datetime, timedelta
import numpy as np, cv2

dbutils.widgets.text("day", "20240422")
dbutils.widgets.text("track_root", "Tracking_v4_reid")
dbutils.widgets.text("out_w", "640")
dbutils.widgets.text("sample_sec", "2.0")
dbutils.widgets.text("out_fps", "25")
dbutils.widgets.text("max_clips", "0")
dbutils.widgets.text("ident_map", "")
dbutils.widgets.text("from_hhmm", "")   # render only clips starting at/after HHMM (boundary cuts)
dbutils.widgets.text("to_hhmm", "")     # ... and before HHMM

DAY        = dbutils.widgets.get("day").strip()
TRACK_ROOT = dbutils.widgets.get("track_root").strip()
OUT_W      = int(dbutils.widgets.get("out_w") or 640)
SAMPLE_SEC = float(dbutils.widgets.get("sample_sec") or 2.0)
OUT_FPS    = int(dbutils.widgets.get("out_fps") or 25)
MAX_CLIPS  = int(dbutils.widgets.get("max_clips") or 0)
# Maps THIS day's slot -> the week's global identity, chained from the Tol_16 night-gap
# permutations. Colour is then keyed to the animal, not to the day's arbitrary bootstrap order,
# so the seven videos concatenate into one week. The colouring IS the pipeline's cross-day claim:
# if a colour lands on a different animal after a night gap, that boundary's match was wrong.
_im = dbutils.widgets.get("ident_map").strip()
IDENT_MAP = {int(k): int(v) for k, v in json.loads(_im).items()} if _im else {i: i for i in range(4)}
assert sorted(IDENT_MAP.values()) == list(range(4)), f"ident_map must be a permutation, got {IDENT_MAP}"

# Optional GT relabel: path to a JSON with time-interval rules per slot, snapped to segment
# boundaries ({"rules": {"0": [[from_iso|null, to_iso|null, ident], ...], ...}}). Used to render
# a verified-ground-truth overlay: if the rules are right, every colour stays on ONE animal all
# day — the visual confirmation loop for GT construction.
dbutils.widgets.text("relabel_json", "")
dbutils.widgets.text("out_name", "")   # override the output filename (keeps earlier renders from being clobbered)
RELABEL = None
_rj = dbutils.widgets.get("relabel_json").strip()
if _rj:
    with open(_rj) as f: RELABEL = json.load(f)["rules"]
    for _k in RELABEL:
        for _r in RELABEL[_k]:
            _r[0] = datetime.fromisoformat(_r[0]) if _r[0] else datetime.min
            _r[1] = datetime.fromisoformat(_r[1]) if _r[1] else datetime.max
    print(f"GT relabel ACTIVE from {_rj}")

def ident_at(slot, wall):
    if RELABEL is not None:
        for _a, _b, _ident in RELABEL.get(str(slot), []):
            if _a <= wall < _b: return int(_ident)
        return None        # GT relabel active but no rule covers this instant = UNASSIGNED (grey "?")
    return IDENT_MAP.get(slot, slot)

# Exclusion windows: during a merge episode two masks can share one animal, so GT identity is
# UNDEFINED there — render contested masks gray with "?" (no false claim) and scoring skips them.
EXCLUSIONS = []
if _rj:
    with open(_rj) as f: _gt_all = json.load(f)
    for _e in _gt_all.get("exclusions", []):
        EXCLUSIONS.append((datetime.fromisoformat(_e["from"]), datetime.fromisoformat(_e["to"]),
                           set(int(s) for s in _e["slots"])))
    print(f"{len(EXCLUSIONS)} exclusion windows loaded")

def excluded_at(slot, wall):
    return any(_a <= wall < _b and slot in _ss for _a, _b, _ss in EXCLUSIONS)

# Hidden intervals (Tol_30 --noncalf, 2026-09-29): a track whose crops look like none of the four calves (e.g. the 04-23
# 15:38-15:49 farm worker, v5 seg 1_012) is not an animal, so NOTHING is drawn for that slot there: no mask, no outline, no
# label (an exclusion would still draw it grey "?"). Closed interval [from, to] widened by HIDE_TOL_S on both ends, because a
# segment's t_end_wall is its LAST frame (the rules' half-open [a, b) would leave that frame grey) and the tracker's wall
# clock differs from this renderer's clip_start + t_sec by < 1 ms; 0.1 s is half the 5-fps frame spacing, so it can never
# reach a frame of the slot's neighbouring segment. A relabel JSON without a "hidden" key changes nothing below.
HIDE_TOL_S = 0.1
HIDDEN = []
if _rj and "hidden" in _gt_all:
    for _h in _gt_all["hidden"]:
        HIDDEN.append((datetime.fromisoformat(_h["from"]) - timedelta(seconds=HIDE_TOL_S),
                       datetime.fromisoformat(_h["to"]) + timedelta(seconds=HIDE_TOL_S), int(_h["slot"])))
    print(f"{len(HIDDEN)} hidden (non-calf) intervals loaded: drawn as NO mask")

def hidden_at(slot, wall):
    return any(_a <= wall <= _b and slot == _s for _a, _b, _s in HIDDEN)

VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
VIDEO_DIR = "/dbfs/mnt/playbehavior/Tol/Videos"
SRC = osp.join(TOL, TRACK_ROOT, DAY)
OUTDIR = osp.join(TOL, "Overlay_Videos"); os.makedirs(OUTDIR, exist_ok=True)
LOCAL = "/local_disk0/tmp/n17"; os.makedirs(LOCAL, exist_ok=True)
N_CALVES = 4
# BGR, chosen to stay distinguishable in greyscale IR footage as well as daylight
COLORS = [(60, 60, 255), (60, 220, 60), (255, 160, 40), (40, 220, 255)]
NAMES  = ["calf_0", "calf_1", "calf_2", "calf_3"]
print(f"{DAY} | {TRACK_ROOT} | {OUT_W}px | 1 frame / {SAMPLE_SEC}s -> {OUT_FPS} fps "
      f"({SAMPLE_SEC*OUT_FPS:.0f}x speed)")

# COMMAND ----------

RXT = re.compile(r'-(\d{8})-(\d{6})-')
def clip_start(v):
    m = RXT.search(v.replace("_", "-"))
    return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")

def rle_decode(r):
    h, w = r["size"]
    c = np.asarray(r["counts"], dtype=np.int64)
    vals = np.zeros(len(c), bool); vals[1::2] = True
    return np.repeat(vals, c).reshape((h, w), order="F")

files = sorted(glob.glob(osp.join(SRC, "masks_*.json.gz")))
assert files, f"no masks in {SRC}"
# STREAM one clip at a time — loading a full 18-h day (33 clips, ~320k frames of RLE python
# objects) upfront OOM-killed the D8ds driver on 04-23. clip_start parses the filename, which
# carries the same date-time pattern as the "clip" field.
clips = sorted(((clip_start(osp.basename(gz)), gz) for gz in files), key=lambda z: z[0])
_fh = dbutils.widgets.get("from_hhmm").strip(); _th = dbutils.widgets.get("to_hhmm").strip()
if _fh: clips = [c for c in clips if c[0].hour*60 + c[0].minute >= int(_fh[:2])*60 + int(_fh[2:])]
if _th: clips = [c for c in clips if c[0].hour*60 + c[0].minute <  int(_th[:2])*60 + int(_th[2:])]
if MAX_CLIPS: clips = clips[:MAX_CLIPS]
assert clips, "no clips in the requested window"
print(f"{len(clips)} clips  {clips[0][0]:%H:%M} -> {clips[-1][0]:%H:%M}")

# COMMAND ----------

out_path = osp.join(LOCAL, f"{DAY}.mp4")
writer, OUT_H = None, None
n_out = n_missing = 0
t0 = time.time()

for ci, (w0, gzp) in enumerate(clips):
    with gzip.open(gzp, "rt") as f: d = json.load(f)
    # pick the stored records to render: one every SAMPLE_SEC of wall time
    picks, last_t = [], None
    for r in d["frames"]:
        if last_t is None or r["t_sec"] - last_t >= SAMPLE_SEC:
            picks.append(r); last_t = r["t_sec"]
    if not picks: continue

    dst = osp.join(LOCAL, "v.mp4")
    if osp.exists(dst): os.remove(dst)
    ok = False
    for attempt in range(3):
        try:
            dbutils.fs.cp("dbfs:" + osp.join(VIDEO_DIR, d["clip"]).replace("/dbfs", ""), "file:" + dst)
            ok = True; break
        except Exception as e:
            print(f"  stage retry {attempt+1} for {d['clip']}: {e}")
            time.sleep(5)
    if not ok:
        print(f"  ⚠ could not stage {d['clip']} — SKIPPED (video will jump here)")
        continue

    cap = cv2.VideoCapture(dst)
    for r in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame_idx"])
        got, fr = cap.read()
        if not got:
            n_missing += 1; continue
        H, W = fr.shape[:2]
        if writer is None:
            OUT_H = int(round(OUT_W * H / W)) // 2 * 2
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     OUT_FPS, (OUT_W, OUT_H))
            assert writer.isOpened(), "VideoWriter failed to open"
            print(f"  writing {OUT_W}x{OUT_H}")
        small = cv2.resize(fr, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        wall = w0 + timedelta(seconds=r["t_sec"])
        for i in range(N_CALVES):
            rl = r["masks"].get(str(i))
            if not rl: continue
            if HIDDEN and hidden_at(i, wall): continue            # non-calf track (Tol_30 --noncalf): draw nothing
            g = None if excluded_at(i, wall) else ident_at(i, wall)   # slot -> identity (GT rules or week ident_map)
            if g is None:
                col, lab = (150, 150, 150), "?"   # GT undefined: merge episode, or no reviewed rule (unassigned)
            else:
                col, lab = COLORS[g], str(g)
            m = rle_decode(rl)
            # nearest-neighbour keeps the mask crisp; INTER_AREA would blur it into a grey halo
            ms = cv2.resize(m.astype(np.uint8), (OUT_W, OUT_H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            if not ms.any(): continue
            small[ms] = (0.55*small[ms] + 0.45*np.array(col, np.float32)).astype(np.uint8)
            cnts, _ = cv2.findContours(ms.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(small, cnts, -1, col, 2)
            ys, xs = np.where(ms)
            cv2.putText(small, lab, (int(xs.mean())-6, int(ys.mean())+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(small, lab, (int(xs.mean())-6, int(ys.mean())+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        wall = w0 + timedelta(seconds=r["t_sec"])
        cv2.rectangle(small, (0, 0), (OUT_W, 22), (0, 0, 0), -1)
        cv2.putText(small, f"{DAY}  {wall:%H:%M:%S}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for i in range(N_CALVES):
            cv2.putText(small, NAMES[i], (OUT_W-260+i*64, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS[i], 1, cv2.LINE_AA)
        writer.write(small); n_out += 1
    cap.release(); os.remove(dst)
    del d
    if (ci+1) % 5 == 0:
        print(f"  {ci+1}/{len(clips)} clips | {n_out} frames | {(time.time()-t0)/60:.1f} min")

if writer is not None: writer.release()
print(f"\n{n_out} frames ({n_out/OUT_FPS/60:.1f} min of video) in {(time.time()-t0)/60:.1f} min"
      f" | {n_missing} frames unreadable")

# COMMAND ----------

assert writer is not None and osp.exists(out_path), "no video produced"
mb = osp.getsize(out_path)/1e6
_wsuf = f"_w{_fh or '0000'}-{_th or '2400'}" if (_fh or _th) else ""
_on = dbutils.widgets.get("out_name").strip()
final = osp.join(OUTDIR, _on if _on.endswith(".mp4") else
                 (f"{_on}.mp4" if _on else f"{DAY}_{TRACK_ROOT}{'_gt' if RELABEL is not None else ''}{_wsuf}.mp4"))
shutil.copy(out_path, final); os.remove(out_path)
print(f"saved {final}  ({mb:.1f} MB)")
print(f"\ncolours: " + "  ".join(f"{NAMES[i]}=BGR{COLORS[i]}" for i in range(N_CALVES)))
print("A colour that jumps to a different animal is an identity switch — note the stamped time.")
