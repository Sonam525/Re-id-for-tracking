# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N18 — Re-encode the overlay videos to H.264
# MAGIC
# MAGIC `Tol_17` writes with OpenCV's `mp4v` (MPEG-4 Part 2), which has no rate control and lands at
# MAGIC ~35 kB/frame at 640x360 — **644 MB to 1.17 GB per day, 7.3 GB for the week**. That is a
# MAGIC download problem, not a quality requirement: H.264 at CRF 30 holds the same visual detail at
# MAGIC roughly a tenth the size. Mask outlines and the timestamp survive it easily; those are high-
# MAGIC contrast edges, which is exactly what H.264 preserves best.
# MAGIC
# MAGIC Transcode only — no re-decode of the 1.3 GB source clips, so this is minutes rather than hours.
# MAGIC `imageio-ffmpeg` is used because it ships a static ffmpeg binary; the runtime does not reliably
# MAGIC have one on PATH.

# COMMAND ----------

# MAGIC %pip install imageio-ffmpeg

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, subprocess, shutil, time
import imageio_ffmpeg

dbutils.widgets.text("crf", "30")
dbutils.widgets.text("suffix", "_small")
dbutils.widgets.text("track_root", "Tracking_v4")   # source files are <day>_<track_root><name_suffix>.mp4
dbutils.widgets.text("name_suffix", "")
dbutils.widgets.text("days", "")   # comma-separated override, e.g. windowed "20240423_w1800-2400"
CRF = dbutils.widgets.get("crf").strip() or "30"
SUFFIX = dbutils.widgets.get("suffix").strip() or "_small"
TRACK_ROOT = dbutils.widgets.get("track_root").strip() or "Tracking_v4"
NAME_SUFFIX = dbutils.widgets.get("name_suffix").strip()
DAYS_OVERRIDE = [d.strip() for d in dbutils.widgets.get("days").split(",") if d.strip()]

VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
SRC = osp.join(VOL, "Tol", "Overlay_Videos")
LOCAL = "/local_disk0/tmp/n18"; os.makedirs(LOCAL, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
print(f"ffmpeg: {FF}\ncrf {CRF}")

days = DAYS_OVERRIDE or ["20240422", "20240423", "20240424", "20240425", "20240426", "20240427", "20240428"]
rows = []
for d in days:
    src = osp.join(SRC, f"{d}_{TRACK_ROOT}{NAME_SUFFIX}.mp4")
    if not osp.exists(src):
        print(f"  {d}: source missing — skipped"); continue
    a = osp.getsize(src)/1e6
    lin  = osp.join(LOCAL, "in.mp4"); lout = osp.join(LOCAL, "out.mp4")
    for p in (lin, lout):
        if osp.exists(p): os.remove(p)
    t0 = time.time()
    shutil.copy(src, lin)
    # -movflags +faststart puts the index up front so the file plays while still downloading
    cmd = [FF, "-y", "-loglevel", "error", "-i", lin, "-c:v", "libx264", "-crf", CRF,
           "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", lout]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {d}: ffmpeg FAILED\n{r.stderr[:400]}"); continue
    b = osp.getsize(lout)/1e6
    dst = osp.join(SRC, f"{d}{NAME_SUFFIX}{SUFFIX}.mp4")
    shutil.copy(lout, dst)
    os.remove(lin); os.remove(lout)
    rows.append((d, a, b))
    print(f"  {d}: {a:7.0f} MB -> {b:6.1f} MB  ({a/max(b,0.01):4.1f}x)  {(time.time()-t0)/60:.1f} min")

print(f'\n{"day":<12}{"before MB":>12}{"after MB":>11}{"ratio":>8}')
for d, a, b in rows:
    print(f"{d:<12}{a:>12.0f}{b:>11.1f}{a/max(b,0.01):>7.1f}x")
if rows:
    ta, tb = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f'{"TOTAL":<12}{ta:>12.0f}{tb:>11.1f}{ta/max(tb,0.01):>7.1f}x')
    print(f"\nweek is now {tb/1000:.2f} GB (was {ta/1000:.2f} GB)")
