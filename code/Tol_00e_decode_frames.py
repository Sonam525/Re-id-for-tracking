# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N0e — one-time 5 fps frame cache to the Unity Catalog Volume
# MAGIC
# MAGIC **Decision 2026-08-18** (reversing the earlier no-cache call): the GT-anchored
# MAGIC distillation loop will read the same days many times — teacher GT run, every improved
# MAGIC student's test run, target extraction, re-ID experiments, verification cards. Decoding
# MAGIC ~1.3 GB mp4s per pass on a GPU-rate node is the wrong place to spend; decode ONCE on a
# MAGIC cheap CPU node and store JPEGs.
# MAGIC
# MAGIC **Layout** — one uncompressed tar per clip (UC Volumes hate millions of small files, and a
# MAGIC tar of JPEGs stages at full `dbutils.fs.cp` speed):
# MAGIC ```
# MAGIC Tol/Frames_5fps/<day>/<clip_stem>.tar          # members: <frame_idx:07d>.jpg
# MAGIC Tol/Frames_5fps/<day>/<clip_stem>_index.json   # frame_idx -> t_sec, native_fps, stride
# MAGIC Tol/Frames_5fps/<day>/.done_<clip_stem>        # resume marker
# MAGIC ```
# MAGIC **Frame identity contract:** frames are sampled with the tracker's EXACT pattern
# MAGIC (`range(0, total, round(native/fps))`, seek-decode), so `frame_idx` here equals `frame_idx`
# MAGIC in every `masks_*.json.gz` — cached frames and stored masks join without re-decoding.
# MAGIC Skip-if-exists at clip granularity: re-running is free.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "opencv-python-headless==4.11.0.86"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, re, json, time, tarfile, shutil
from datetime import datetime, timedelta
import numpy as np, cv2

dbutils.widgets.text("day", "20240422")
dbutils.widgets.text("fps", "5.0")
dbutils.widgets.text("quality", "90")
dbutils.widgets.text("max_clips", "0")

DAY     = dbutils.widgets.get("day").strip()
FPS     = float(dbutils.widgets.get("fps") or 5.0)
QUALITY = int(dbutils.widgets.get("quality") or 90)
MAX_CLIPS = int(dbutils.widgets.get("max_clips") or 0)

VIDEO_DIR  = "/dbfs/mnt/playbehavior/Tol/Videos"
DBFS_VIDEO = "dbfs:/mnt/playbehavior/Tol/Videos"
VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
OUT = osp.join(VOL, "Tol", "Frames_5fps", DAY)
LOCAL = "/local_disk0/tmp/decode"
os.makedirs(OUT, exist_ok=True); os.makedirs(LOCAL, exist_ok=True)

RX = re.compile(r'^(Tol_ch\d+)[-_](\d{8})[-_](\d{6})[-_](\d{6})(?: \((\d+)\))?\.mp4$')
CORRUPT = {"Tol_ch01_20240418-113301-121506.mp4"}   # 0.7 MB truncated at source — do not retry

clips = []
for n in sorted(os.listdir(VIDEO_DIR)):
    m = RX.match(n)
    if not m or m.group(5) is not None: continue          # exclude " (1)" duplicates
    if m.group(1) != "Tol_ch01" or m.group(2) != DAY: continue
    if n in CORRUPT:
        print(f"skip known-corrupt {n}"); continue
    clips.append(n)
if MAX_CLIPS: clips = clips[:MAX_CLIPS]
assert clips, f"no clips for {DAY}"
print(f"{DAY}: {len(clips)} clips -> {OUT}")

# COMMAND ----------

t_day = time.time(); n_frames_day = 0
for ci, vname in enumerate(clips):
    stem = vname[:-4].replace(" ", "_")
    done = osp.join(OUT, f".done_{stem}")
    if osp.exists(done) and osp.exists(osp.join(OUT, f"{stem}.tar")):
        print(f"[{ci+1}/{len(clips)}] {stem}: cached — skip"); continue

    dst = osp.join(LOCAL, "cur.mp4")
    if osp.exists(dst): os.remove(dst)
    t0 = time.time()
    dbutils.fs.cp(f"{DBFS_VIDEO}/{vname}", "file:" + dst)
    t_stage = time.time() - t0

    cap = cv2.VideoCapture(dst)
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(native/FPS))
    if total <= 0:
        print(f"  ⚠ {stem}: no frame count — skipped, NOT marked done")
        cap.release(); os.remove(dst); continue

    tar_local = osp.join(LOCAL, f"{stem}.tar")
    if osp.exists(tar_local): os.remove(tar_local)
    index = []
    t1 = time.time(); n = 0
    with tarfile.open(tar_local, "w") as tf:
        for fidx in range(0, total, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, fr = cap.read()
            if not ok: continue
            ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if not ok2: continue
            info = tarfile.TarInfo(name=f"{fidx:07d}.jpg")
            info.size = len(buf)
            import io as _io
            tf.addfile(info, _io.BytesIO(buf.tobytes()))
            index.append({"frame_idx": fidx, "t_sec": round(fidx/native, 2)})
            n += 1
    cap.release(); os.remove(dst)
    t_dec = time.time() - t1

    t2 = time.time()
    dbutils.fs.cp("file:" + tar_local, "dbfs:" + osp.join(OUT, f"{stem}.tar").replace("/Volumes", "/Volumes"))
    with open(osp.join(OUT, f"{stem}_index.json"), "w") as f:
        json.dump({"clip": vname, "native_fps": native, "total_frames": total,
                   "stride": stride, "fps": FPS, "quality": QUALITY, "n": n,
                   "frames": index}, f)
    open(done, "w").close()
    mb = osp.getsize(tar_local)/1e6; os.remove(tar_local)
    n_frames_day += n
    print(f"[{ci+1}/{len(clips)}] {stem}: {n} frames, {mb:.0f} MB tar | "
          f"stage {t_stage/60:.1f}m decode {t_dec/60:.1f}m upload {(time.time()-t2)/60:.1f}m")

with open(osp.join(OUT, "manifest.json"), "w") as f:
    json.dump({"day": DAY, "fps": FPS, "quality": QUALITY, "n_clips": len(clips),
               "n_frames": n_frames_day, "built": datetime.now().isoformat()}, f)
print(f"\n✓ {DAY}: {n_frames_day} frames cached in {(time.time()-t_day)/3600:.2f} h")
