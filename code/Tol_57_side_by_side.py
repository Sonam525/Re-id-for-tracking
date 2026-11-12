# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N57 — Before | after side-by-side review clips of the mask refinement
# MAGIC
# MAGIC `Tol_17` rendered each review window twice from the same source frames: once from
# MAGIC `Tracking_v5a10` (**before**) and once from `Tracking_v5a10_refined` (**after**, `Tol_55`'s
# MAGIC fragment → whole-calf replacement). Reviewing them as two separate files means scrubbing two
# MAGIC players in sync; this notebook stacks them horizontally into ONE small H.264 clip per window
# MAGIC (left = before, right = after) so a fragment fix is visible at a glance.
# MAGIC
# MAGIC - Transcode only — the 1.3 GB source clips are never decoded.
# MAGIC - ffmpeg comes from `imageio-ffmpeg` (static binary), exactly as in `Tol_18`; the runtime does
# MAGIC   not reliably have one on PATH. It ships no `ffprobe`, so durations are probed with
# MAGIC   `ffprobe` if present on PATH, else with `ffmpeg -i` (container header) **and** a full decode
# MAGIC   frame count (`imageio_ffmpeg.count_frames_and_secs`).
# MAGIC - Labels: a black bar is padded ABOVE the stacked video (so nothing in the overlay is occluded)
# MAGIC   and "before" / "after" are written into it with `drawtext` if the binary has it; if not, the
# MAGIC   labels are rendered with PIL to a PNG and composited with `overlay` (no freetype needed). The
# MAGIC   method actually used is printed and returned.
# MAGIC - Staging uses `dbutils.fs.cp` (never FUSE `shutil.copy` — ~30x slower on large files).
# MAGIC - Writes ONLY `<W>_refine_side_by_side_small.mp4` under `Tol/Overlay_Videos/`; inputs untouched.

# COMMAND ----------

# MAGIC %pip install imageio-ffmpeg

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, re, json, shutil, subprocess, time
import imageio_ffmpeg

dbutils.widgets.text("windows", "20240422_w1641-1715,20240423_w0830-0910,20240423_w1011-1130")
dbutils.widgets.text("out_w", "1280")      # total output width (each half = out_w/2)
dbutils.widgets.text("crf", "30")
dbutils.widgets.text("preset", "medium")
dbutils.widgets.text("bar_h", "32")        # label bar height (px, above the video)
dbutils.widgets.text("out_suffix", "_refine_side_by_side_small")

WINDOWS = [w.strip() for w in dbutils.widgets.get("windows").split(",") if w.strip()]
OUT_W   = int(dbutils.widgets.get("out_w") or 1280)
CRF     = dbutils.widgets.get("crf").strip() or "30"
PRESET  = dbutils.widgets.get("preset").strip() or "medium"
BAR_H   = int(dbutils.widgets.get("bar_h") or 32)
OUT_SUFFIX = dbutils.widgets.get("out_suffix").strip() or "_refine_side_by_side_small"
HALF_W  = (OUT_W // 2) // 2 * 2            # even
BAR_H   = BAR_H // 2 * 2

VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
OVL = osp.join(VOL, "Tol", "Overlay_Videos")
LOCAL = "/local_disk0/tmp/n57"; os.makedirs(LOCAL, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
FFPROBE = shutil.which("ffprobe")
r = subprocess.run([FF, "-hide_banner", "-version"], capture_output=True, text=True)
print(f"ffmpeg : {FF}\n         {r.stdout.splitlines()[0] if r.stdout else '?'}")
print(f"ffprobe: {FFPROBE or 'not on PATH -> ffmpeg -i header + decode count'}")
print(f"windows: {WINDOWS}\nout {OUT_W} px wide (2 x {HALF_W}) + {BAR_H} px label bar, crf {CRF}, preset {PRESET}")

# COMMAND ----------

# --- probing ---------------------------------------------------------------------------------
def probe(path):
    """Return dict(w, h, fps, dur_s, nb_frames, method). Header values + an exact decode count."""
    info = {"method": None}
    if FFPROBE:
        p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,r_frame_rate,nb_frames:format=duration",
                            "-of", "json", path], capture_output=True, text=True)
        if p.returncode == 0:
            j = json.loads(p.stdout); s = j["streams"][0]
            num, den = s["r_frame_rate"].split("/")
            info.update(w=int(s["width"]), h=int(s["height"]), fps=float(num)/float(den),
                        dur_s=float(j["format"]["duration"]), method="ffprobe")
    if info["method"] is None:
        p = subprocess.run([FF, "-hide_banner", "-i", path], capture_output=True, text=True)
        err = p.stderr
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
        dur = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3)) if m else None
        m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
        w, h = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        m = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
        fps = float(m.group(1)) if m else None
        info.update(w=w, h=h, fps=fps, dur_s=dur, method="ffmpeg -i header")
    nf, secs = imageio_ffmpeg.count_frames_and_secs(path)   # full decode -> exact
    info.update(nb_frames=int(nf), decoded_s=round(float(secs), 3))
    return info

# --- label method: drawtext if the static binary has freetype, else a PIL PNG + overlay -------
def find_font():
    cands = []
    try:
        import matplotlib
        cands += [osp.join(matplotlib.get_data_path(), "fonts", "ttf", f)
                  for f in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")]
    except Exception:
        pass
    cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
    return next((c for c in cands if osp.exists(c)), None)

FONT = find_font()
FONT_PX = max(12, int(BAR_H * 0.65))
print(f"font   : {FONT}")

def drawtext_works():
    if not FONT:
        return False
    vf = (f"drawtext=fontfile='{FONT}':text='before':fontsize=20:fontcolor=white:x=4:y=4")
    p = subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=black:s=128x64:d=0.2", "-vf", vf, "-f", "null", "-"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"drawtext test failed: {p.stderr.strip()[:300]}")
    return p.returncode == 0

LABEL_MODE = "drawtext" if drawtext_works() else None
LABEL_PNG = osp.join(LOCAL, "label_bar.png")
if LABEL_MODE is None:
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (2 * HALF_W, BAR_H), (0, 0, 0))
        d = ImageDraw.Draw(img)
        try:
            fnt = ImageFont.truetype(FONT, FONT_PX) if FONT else ImageFont.load_default()
        except Exception:
            fnt = ImageFont.load_default()
        for i, txt in enumerate(("before", "after")):
            bb = d.textbbox((0, 0), txt, font=fnt)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            d.text((i * HALF_W + (HALF_W - tw) // 2 - bb[0], (BAR_H - th) // 2 - bb[1]),
                   txt, fill=(255, 255, 255), font=fnt)
        d.line([(HALF_W, 0), (HALF_W, BAR_H)], fill=(128, 128, 128), width=2)
        img.save(LABEL_PNG)
        LABEL_MODE = "pil_png_overlay"
    except Exception as e:
        print(f"PIL label render failed ({e!r}) -> labels skipped (plain padded bar)")
        LABEL_MODE = "none"
print(f"labels : {LABEL_MODE}")

# COMMAND ----------

def build_filter(h_half):
    """[0]=before, [1]=after, optional [2]=label png. Each half scaled to HALF_W x h_half."""
    sc = f"scale={HALF_W}:{h_half}:flags=lanczos,setsar=1"
    g = [f"[0:v]{sc}[b]", f"[1:v]{sc}[a]",
         "[b][a]hstack=inputs=2:shortest=1[s]",
         # label bar on top; nothing in the overlay frame is covered
         f"[s]pad=iw:ih+{BAR_H}:0:{BAR_H}:black[p]"]
    if LABEL_MODE == "drawtext":
        common = (f"fontfile='{FONT}':fontsize={FONT_PX}:fontcolor=white:"
                  f"y=({BAR_H}-th)/2")
        g.append(f"[p]drawtext={common}:text='before':x=({HALF_W}-tw)/2,"
                 f"drawtext={common}:text='after':x={HALF_W}+({HALF_W}-tw)/2,"
                 f"drawbox=x={HALF_W-1}:y=0:w=2:h={BAR_H}:color=gray:t=fill[v]")
    elif LABEL_MODE == "pil_png_overlay":
        g.append("[p][2:v]overlay=0:0:format=auto[v]")
    else:
        g.append("[p]null[v]")
    return ";".join(g)

rows, failures = [], []
for W in WINDOWS:
    t0 = time.time()
    src_b = osp.join(OVL, f"{W}_before_refine_review.mp4")
    src_a = osp.join(OVL, f"{W}_after_refine_review.mp4")
    dst   = osp.join(OVL, f"{W}{OUT_SUFFIX}.mp4")
    if not (osp.exists(src_b) and osp.exists(src_a)):
        msg = f"{W}: input missing (before={osp.exists(src_b)}, after={osp.exists(src_a)})"
        print("  " + msg); failures.append(msg); continue
    lb, la, lo = (osp.join(LOCAL, f) for f in ("before.mp4", "after.mp4", "out.mp4"))
    for p in (lb, la, lo):
        if osp.exists(p): os.remove(p)
    dbutils.fs.cp("dbfs:" + src_b, "file:" + lb)
    dbutils.fs.cp("dbfs:" + src_a, "file:" + la)
    t_stage = time.time() - t0
    pb, pa = probe(lb), probe(la)
    print(f"\n{W}\n  staged {osp.getsize(lb)/1e6:.1f} + {osp.getsize(la)/1e6:.1f} MB in {t_stage:.0f}s")
    print(f"  before: {pb}\n  after : {pa}")
    if (pb["w"], pb["h"]) != (pa["w"], pa["h"]):
        print(f"  NOTE: input sizes differ {pb['w']}x{pb['h']} vs {pa['w']}x{pa['h']} - both scaled to the same half size")
    if pb["nb_frames"] != pa["nb_frames"]:
        print(f"  NOTE: frame counts differ ({pb['nb_frames']} vs {pa['nb_frames']}) - output ends at the shorter (shortest=1)")
    h_half = int(round(HALF_W * pb["h"] / pb["w"] / 2.0)) * 2
    cmd = [FF, "-y", "-hide_banner", "-loglevel", "error", "-i", lb, "-i", la]
    if LABEL_MODE == "pil_png_overlay":
        cmd += ["-i", LABEL_PNG]
    cmd += ["-filter_complex", build_filter(h_half), "-map", "[v]", "-an",
            "-c:v", "libx264", "-crf", CRF, "-preset", PRESET, "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", lo]
    t1 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not osp.exists(lo):
        msg = f"{W}: ffmpeg FAILED rc={r.returncode}: {r.stderr.strip()[:600]}"
        print("  " + msg); failures.append(msg); continue
    t_enc = time.time() - t1
    po = probe(lo)
    size_mb = osp.getsize(lo) / 1e6
    dbutils.fs.cp("file:" + lo, "dbfs:" + dst)
    assert osp.exists(dst) and abs(osp.getsize(dst) - osp.getsize(lo)) == 0, f"upload size mismatch for {dst}"
    row = {"window": W, "dst": dst, "size_bytes": osp.getsize(lo), "size_mb": round(size_mb, 2),
           "out": po, "before": pb, "after": pa,
           "in_mb": [round(osp.getsize(lb)/1e6, 1), round(osp.getsize(la)/1e6, 1)],
           "encode_s": round(t_enc, 1), "total_s": round(time.time() - t0, 1)}
    rows.append(row)
    print(f"  out   : {po}\n  -> {dst}\n     {size_mb:.2f} MB, encode {t_enc:.0f}s, total {time.time()-t0:.0f}s")
    for p in (lb, la, lo):
        if osp.exists(p): os.remove(p)

# COMMAND ----------

print(f'\n{"window":<22}{"MB":>8}{"WxH":>12}{"fps":>7}{"dur_s":>9}{"frames":>8}{"in before/after":>20}')
for r_ in rows:
    o = r_["out"]
    print(f'{r_["window"]:<22}{r_["size_mb"]:>8.2f}{str(o["w"])+"x"+str(o["h"]):>12}{o["fps"] or 0:>7.2f}'
          f'{o["dur_s"] or 0:>9.2f}{o["nb_frames"]:>8}{str(r_["before"]["nb_frames"])+"/"+str(r_["after"]["nb_frames"]):>20}')
print(f"label mode: {LABEL_MODE}")
if failures:
    print("FAILURES:\n  " + "\n  ".join(failures))

summary = {"label_mode": LABEL_MODE, "font": FONT, "ffmpeg": FF, "ffprobe": FFPROBE,
           "out_w": OUT_W, "crf": CRF, "preset": PRESET, "bar_h": BAR_H,
           "rows": rows, "failures": failures}
if failures:
    raise RuntimeError(json.dumps(summary)[:4000])
dbutils.notebook.exit(json.dumps(summary))
