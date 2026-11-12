# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N28 — build the crop band-embedding cache for a tracked day
# MAGIC
# MAGIC The frozen matcher's features: per saved canonical crop, DINOv3-S embeddings of
# MAGIC [whole, third_top, third_mid, third_bot] × [original, 180°-rotated] → [8, 384], cached as
# MAGIC `crop_bands3.npz` in the day's track dir. Previously Tol_25 built this as a side effect of
# MAGIC GT scoring; GT-less days (everything except 04-22) need it standalone before Tol_15_v2.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, glob, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from huggingface_hub import login

dbutils.widgets.text("day", "20240423")
dbutils.widgets.text("track_root", "Tracking_v4v2")
dbutils.widgets.text("crops_subdir", "crops")      # "crops_dense" for the Tol_32 dense regeneration
dbutils.widgets.text("out_name", "")               # default: crop_bands3.npz for "crops", _dense otherwise
DAY = dbutils.widgets.get("day").strip()
CROPS_SUB = (dbutils.widgets.get("crops_subdir") or "crops").strip()
VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
SRC = osp.join(VOL, "Tol", dbutils.widgets.get("track_root").strip(), DAY)
_out = (dbutils.widgets.get("out_name") or "").strip() or (
    "crop_bands3.npz" if CROPS_SUB == "crops" else f"crop_bands3_{CROPS_SUB.replace('crops_', '')}.npz")
CACHE = osp.join(SRC, _out)
print(f"{DAY}: crops from {CROPS_SUB}/ -> {_out}")
if osp.exists(CACHE):
    print("cache already exists — nothing to do")
    dbutils.notebook.exit("cached")

login(token=dbutils.secrets.get(scope="hf", key="token").strip().strip('"').strip("'").strip())
device = torch.device("cuda")
DTYPE = torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16

from transformers import AutoImageProcessor, AutoModel
proc = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
dino = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m",
                                 torch_dtype=DTYPE).to(device).eval()

paths = sorted(glob.glob(osp.join(SRC, CROPS_SUB, "*", "*.jpg")))
assert paths, f"no crops under {osp.join(SRC, CROPS_SUB)} — run Tol_32 first for a dense subdir"
print(f"{len(paths)} crops")
def slc(img):
    a = np.array(img); H = a.shape[0]
    outs = [a, a[:H//3], a[H//3:2*H//3], a[2*H//3:]]
    return [Image.fromarray(x).resize((224, 224), Image.BILINEAR) for x in outs]

vecs, kept = [], []
t0 = time.time()
for ci, p in enumerate(paths):
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        continue
    batch = slc(img) + slc(img.rotate(180))
    px = proc(images=batch, return_tensors="pt")["pixel_values"].to(device, DTYPE)
    with torch.no_grad():
        e = F.normalize(dino(pixel_values=px).pooler_output.float(), dim=-1)
    assert torch.isfinite(e).all()
    vecs.append(e.cpu().numpy()); kept.append(p)
    if (ci+1) % 500 == 0: print(f"  {ci+1}/{len(paths)}  {(time.time()-t0)/60:.1f} min")
PE = np.stack(vecs).astype(np.float16)
tmp = "/local_disk0/tmp/b3.npz"; os.makedirs("/local_disk0/tmp", exist_ok=True)
np.savez_compressed(tmp, paths=np.array(kept), emb=PE)
import shutil; shutil.copy(tmp, CACHE); os.remove(tmp)
print(f"cached {PE.shape} in {(time.time()-t0)/60:.1f} min -> {CACHE}")
