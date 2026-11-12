# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N32 — regenerate DENSE canonical crops from stored masks (no re-tracking, CPU only)
# MAGIC
# MAGIC **Why.** The identity layer re-IDs off the tracker's saved canonical crops, written once per minute
# MAGIC (`CROP_EVERY_SEC = 60`). Measured 2026-09-15: raising the re-ID *frequency* below 3 min makes things
# MAGIC worse at every feature density, but raising crop *density* — more exemplars behind each 3-min
# MAGIC decision — is a clear win: on 04-22 it lifted agreement with the verified chain 97.7 % → 100.0 %,
# MAGIC dropped review flags 36 % → 5.6 %, and turned the 21:00 exchange (which `Tol_15_v2` could only flag at
# MAGIC margin 0.0075) into an automatic, unflagged call at margin 0.063.
# MAGIC
# MAGIC That test used `observations.npz` (the tracker's own 10-s embeddings) — but those are WHOLE-crop
# MAGIC vectors, and on the fragmented days they are over-confident: 04-23 fell to 78.7 % with 1,658 silent
# MAGIC errors vs 90.9 % and **zero** silent for the frozen 3-part matcher. So we want the *density* of the
# MAGIC observations with the *feature* of the frozen matcher. This notebook produces exactly that.
# MAGIC
# MAGIC **What it does.** Streams each clip, decodes only the frames we need, applies `canonical_crop` to the
# MAGIC already-stored per-frame RLE masks, and writes `crops_dense/<seg_id>/HHMMSS.jpg` next to the day's
# MAGIC tracking output. No SAM3, no student, no GPU, no re-tracking — the masks are already on disk.
# MAGIC Feed the result to `Tol_28_build_bands` (`crops_subdir=crops_dense`) to get a dense `crop_bands3`.
# MAGIC
# MAGIC **⚠ CROP-DEFINITION CONTRACT.** `canonical_crop` below is copied VERBATIM from
# MAGIC `Tol_03_track_v4.py`. Do not "improve" it. `Tol_09`'s variant is NOT equivalent — it gates the PCA
# MAGIC rotation on an elongation ratio (`sqrt(λmax/λmin) >= 1.5`) that the tracker does not apply, so round
# MAGIC masks come out unrotated there and rotated here. Mixing the two between bank and query is a silent
# MAGIC correctness bug. The self-check cell at the end proves the new crops reproduce the
# MAGIC tracker's own crops at the timestamps where both exist.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "Pillow==11.1.0" "opencv-python-headless==4.11.0.86"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, glob, gzip, json, re, time, shutil
from datetime import datetime, timedelta
import numpy as np, cv2
from PIL import Image

dbutils.widgets.text("day", "20240423")
dbutils.widgets.text("track_root", "Tracking_v4v2")
dbutils.widgets.text("crop_every_sec", "15")      # 60 -> 15 gives 4x the exemplars per decision
dbutils.widgets.text("out_subdir", "crops_dense")
dbutils.widgets.text("max_clips", "0")            # 0 = all; small value for a costing probe
dbutils.widgets.text("selfcheck", "true")
dbutils.widgets.text("tracker_crop_sec", "60")   # CROP_EVERY_SEC the tracker ran with (its clock, replicated)
dbutils.widgets.text("anchors_only", "false")    # repair pass: write ONLY the tracker-clock frames (~190/day)

DAY        = dbutils.widgets.get("day").strip()
TRACK_ROOT = dbutils.widgets.get("track_root").strip()
EVERY      = float(dbutils.widgets.get("crop_every_sec") or 15)
OUT_SUB    = dbutils.widgets.get("out_subdir").strip()
MAX_CLIPS  = int(dbutils.widgets.get("max_clips") or 0)
SELFCHECK  = (dbutils.widgets.get("selfcheck") or "true").strip().lower() == "true"
TRK_EVERY  = float(dbutils.widgets.get("tracker_crop_sec") or 60)
ANCHORS_ONLY = (dbutils.widgets.get("anchors_only") or "false").strip().lower() == "true"

VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
VIDEO_DIR = "/dbfs/mnt/playbehavior/Tol/Videos"
SRC = osp.join(TOL, TRACK_ROOT, DAY)
OUT = osp.join(SRC, OUT_SUB)
LOCAL = "/local_disk0/tmp/n32"; os.makedirs(LOCAL, exist_ok=True)
N_CALVES = 4
print(f"{DAY} | {TRACK_ROOT} | 1 crop / {EVERY:.0f}s / slot -> {OUT}"
      + (f"  [ANCHORS ONLY: replicating the tracker's {TRK_EVERY:.0f}s clock]" if ANCHORS_ONLY else ""))

# COMMAND ----------

# MAGIC %md ## canonical_crop — VERBATIM from Tol_03_track_v4.py (INV-9). Do not modify.

# COMMAND ----------

MIN_MASK_PX = 8000
FILL = (124, 116, 104)

def m_bbox(m):
    ys, xs = np.where(m)
    if len(xs) == 0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def canonical_crop(pil, mask, pad=10, out_size=224):
    """Pose- and exposure-normalised masked crop — the frozen re-ID input (DESIGN INV-9)."""
    if mask is None or mask.sum() < MIN_MASK_PX: return None
    ys, xs = np.where(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32); pts -= pts.mean(0)
    try:
        w, v = np.linalg.eigh(np.cov(pts.T))
        ax = v[:, int(np.argmax(w))]
        angle = float(np.degrees(np.arctan2(ax[1], ax[0])))
    except Exception:
        angle = 0.0
    arr = np.array(pil)
    fg = np.zeros(arr.shape[:2], np.uint8); fg[mask] = 255
    masked = np.empty_like(arr); masked[:] = np.array(FILL, arr.dtype); masked[mask] = arr[mask]
    rot = 90.0 - angle
    im_r = Image.fromarray(masked).rotate(rot, resample=Image.BILINEAR, expand=True,
                                          fillcolor=tuple(FILL))
    fg_r = Image.fromarray(fg).rotate(rot, resample=Image.NEAREST, expand=True, fillcolor=0)
    fga = np.array(fg_r) > 127
    bb = m_bbox(fga)
    if bb is None: return None
    a = np.array(im_r).astype(np.float32)
    sel = fga
    if sel.sum() > 50:
        vals = a[sel]
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        if hi > lo + 1e-3:
            a[sel] = np.clip((a[sel]-lo)*255.0/(hi-lo), 0, 255)
    a[~sel] = np.array(FILL, np.float32)
    H, W = a.shape[:2]
    crop = a[max(0, bb[1]-pad):min(H, bb[3]+pad+1), max(0, bb[0]-pad):min(W, bb[2]+pad+1)]
    return Image.fromarray(crop.astype(np.uint8)).resize((out_size, out_size), Image.BILINEAR)

def rle_decode(r):
    h, w = r["size"]
    c = np.asarray(r["counts"], dtype=np.int64)
    vals = np.zeros(len(c), bool); vals[1::2] = True
    return np.repeat(vals, c).reshape((h, w), order="F")

RXT = re.compile(r'-(\d{8})-(\d{6})-')
def clip_start(v):
    m = RXT.search(v.replace("_", "-"))
    return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")

# COMMAND ----------

# MAGIC %md ## Stream clips, decode only the needed frames, write crops

# COMMAND ----------

# slot -> [(t_start, t_end, seg_id)] : the stored mask rows carry ONLY frame_idx/t_sec/masks, so the
# segment a crop belongs to has to come from segments.json by (slot, wall). Getting this wrong would file
# crops under the wrong segment and silently corrupt every exemplar pool.
_segj = json.load(open(osp.join(SRC, "segments.json")))
SEG_BY_SLOT = {}
for _r in _segj["segments"]:
    SEG_BY_SLOT.setdefault(int(_r["slot"]), []).append(
        (datetime.fromisoformat(_r["t_start_wall"]), datetime.fromisoformat(_r["t_end_wall"]), _r["seg_id"]))
for _k in SEG_BY_SLOT: SEG_BY_SLOT[_k].sort()
print(f"{len(_segj['segments'])} segments indexed over {len(SEG_BY_SLOT)} slots")

def seg_at(slot, wall):
    for a, b, sid in SEG_BY_SLOT.get(slot, []):
        if a <= wall < b: return sid
    return None

files = sorted(glob.glob(osp.join(SRC, "masks_*.json.gz")))
assert files, f"no masks in {SRC}"
clips = sorted(((clip_start(osp.basename(g)), g) for g in files), key=lambda z: z[0])
if MAX_CLIPS: clips = clips[:MAX_CLIPS]
os.makedirs(OUT, exist_ok=True)
print(f"{len(clips)} clips  {clips[0][0]:%H:%M} -> {clips[-1][0]:%H:%M}")

t0 = time.time(); n_written = n_skipped = n_missing = 0
stage_s = decode_s = crop_s = 0.0
for ci, (w0, gzp) in enumerate(clips):
    with gzip.open(gzp, "rt") as f: d = json.load(f)
    # SUPERSET RULE, done by REPLICATING THE TRACKER'S CROP CLOCK rather than by matching wall-seconds.
    # At 5 fps a wall-second holds FIVE stored rows; the tracker cropped ONE of them (its clock is
    # next_crop = 0.0 then t_sec + CROP_EVERY_SEC, advanced on every processed frame — Tol_03_track_v4.py
    # :1130, :1333). Matching on the second alone picked a different frame inside the same second, which the
    # self-check caught as a 9.11 mean pixel difference. Replaying the clock over the stored rows (which ARE
    # the frames the tracker processed, in order, with the same t_sec) reproduces its picks EXACTLY.
    picks, n_anchor = [], 0
    next_anchor, next_infill = 0.0, 0.0
    for r in d["frames"]:
        t = r["t_sec"]
        is_anchor = t >= next_anchor
        if is_anchor: next_anchor = t + TRK_EVERY
        want_infill = (not ANCHORS_ONLY) and t >= next_infill
        if want_infill: next_infill = t + EVERY
        if not r.get("masks"): continue          # clock advances regardless, exactly as the tracker's does
        if is_anchor or want_infill:
            picks.append(r); n_anchor += int(is_anchor)
    if not picks: continue

    dst = osp.join(LOCAL, "v.mp4")
    if osp.exists(dst): os.remove(dst)
    ts = time.time(); ok = False
    for attempt in range(3):
        try:
            dbutils.fs.cp("dbfs:" + osp.join(VIDEO_DIR, d["clip"]).replace("/dbfs", ""), "file:" + dst)
            ok = True; break
        except Exception as e:
            print(f"  stage retry {attempt+1} for {d['clip']}: {e}"); time.sleep(5)
    stage_s += time.time() - ts
    if not ok:
        print(f"  ⚠ could not stage {d['clip']} — SKIPPED"); continue

    cap = cv2.VideoCapture(dst)
    for r in picks:
        td = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame_idx"])
        got, fr = cap.read()
        decode_s += time.time() - td
        if not got: n_missing += 1; continue
        pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        wall = w0 + timedelta(seconds=r["t_sec"])
        tc = time.time()
        for s in range(N_CALVES):
            rl = r["masks"].get(str(s))
            if not rl: continue
            sid = seg_at(s, wall)
            if not sid: n_skipped += 1; continue
            cc = canonical_crop(pil, rle_decode(rl))
            if cc is None: n_skipped += 1; continue
            cd = osp.join(OUT, sid); os.makedirs(cd, exist_ok=True)
            cc.save(osp.join(cd, f"{wall:%H%M%S}.jpg"), quality=88)
            n_written += 1
        crop_s += time.time() - tc
    cap.release(); os.remove(dst); del d
    if (ci+1) % 5 == 0:
        el = (time.time()-t0)/60
        print(f"  {ci+1}/{len(clips)} clips | {n_written} crops ({n_anchor} tracker-anchor picks this clip) | {el:.1f} min "
              f"(stage {stage_s/60:.1f} / decode {decode_s/60:.1f} / crop {crop_s/60:.1f}) "
              f"| eta {el/(ci+1)*(len(clips)-ci-1):.0f} min")

el = (time.time()-t0)/60
print(f"\n{n_written} crops written in {el:.1f} min  (stage {stage_s/60:.1f} | decode {decode_s/60:.1f} | crop {crop_s/60:.1f})")
print(f"{n_skipped} skipped (no seg_id / too small), {n_missing} frames unreadable")
print(f"-> {OUT}")

# COMMAND ----------

# MAGIC %md ## Self-check: the dense crops must reproduce the tracker's own crops
# MAGIC At timestamps where both exist, the regenerated crop must be near-identical to the one the tracker
# MAGIC wrote at run time. A large difference means the crop definition drifted — STOP and fix before
# MAGIC embedding, or bank/query end up in different spaces.

# COMMAND ----------

if SELFCHECK:
    ref_dir = osp.join(SRC, "crops")
    pairs, diffs, n_ref_sampled = 0, [], 0
    # only compare inside the clips we actually processed (max_clips probes cover a prefix of the day)
    done_hh = set()
    for _w0, _g in clips:
        with gzip.open(_g, "rt") as _f: _d = json.load(_f)
        for _r in _d["frames"]:
            if _r.get("masks"): done_hh.add(f"{_w0 + timedelta(seconds=_r['t_sec']):%H%M%S}")
        del _d
    for sid_dir in sorted(glob.glob(osp.join(ref_dir, "*"))):
        sid = osp.basename(sid_dir)
        for ref_p in sorted(glob.glob(osp.join(sid_dir, "*.jpg"))):
            if osp.basename(ref_p)[:6] not in done_hh: continue
            n_ref_sampled += 1
            new_p = osp.join(OUT, sid, osp.basename(ref_p))
            if not osp.exists(new_p): continue
            a = np.array(Image.open(ref_p).convert("RGB"), np.float32)
            b = np.array(Image.open(new_p).convert("RGB"), np.float32)
            diffs.append(float(np.abs(a-b).mean())); pairs += 1
    n_ref = sum(1 for _ in glob.glob(osp.join(ref_dir, "*", "*.jpg")))
    n_new = sum(1 for _ in glob.glob(osp.join(OUT, "*", "*.jpg")))
    cov = pairs / max(n_ref_sampled, 1)
    dm = float(np.mean(diffs)) if diffs else float("inf")
    dmax = float(np.max(diffs)) if diffs else float("inf")
    print(f"self-check: {pairs}/{n_ref_sampled} sampled tracker crops reproduced ({100*cov:.0f} %) | "
          f"mean |Δpixel| {dm:.2f} | worst {dmax:.2f} | crops {n_ref} -> {n_new}")
    # HARD GATE. A silent pass here would let Tol_28 embed crops from a different definition than the frozen
    # exemplars, which is a silent correctness bug. Fail loudly instead.
    if pairs == 0:
        raise RuntimeError("self-check found NO reproduced tracker crop — the superset rule did not fire. "
                           "Do not embed these crops.")
    if cov < 0.90:
        raise RuntimeError(f"self-check reproduced only {100*cov:.0f} % of sampled tracker crops (< 90 %). "
                           "The frame-selection rule is wrong; do not embed these crops.")
    if dm >= 2.0:
        raise RuntimeError(f"self-check mean pixel difference {dm:.2f} >= 2.0 — the crop definition drifted "
                           "from the tracker's. Do not embed these crops.")
    print("PASS — dense crops reproduce the tracker's own crops and are a strict superset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next
# MAGIC 1. `Tol_28_build_bands` with `crops_subdir=crops_dense` → `crop_bands3_dense.npz` (ViT-S, ~1 min/day/4x).
# MAGIC 2. `Tol_30_periodic_identity --T 3` on the dense bands; compare against the frozen 04-22 / 04-23 GT.
# MAGIC 3. Expect: 04-22 → 100 %, flags 36 % → ~6 %; 04-23 → the parts-3 night advantage KEPT (unlike the
# MAGIC    whole-crop observation features, which lost 1,658 crops to silent errors there).
