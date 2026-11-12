# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N43 — RF-DETR-Seg fine-tune on the `Tol_42` COCO export (A100, single node)
# MAGIC
# MAGIC **Purpose.** Prove that SAM3 (`transformers==5.2.0`) and `rfdetr==1.10.1` coexist in one process,
# MAGIC measure training throughput on a seeded smoke subset and **stop** if the projected full run would
# MAGIC exceed the budget, run the full fine-tune resumably (background `last.ckpt` sync to the Volume with a
# MAGIC sha256 sidecar), choose the operating threshold on the `valid` split, and copy checkpoint + config +
# MAGIC summary to the Volume. The `label_source` widget **defaults to `tracked`** (DET-R — **the only run planned**,
# MAGIC decision 2026-09-23; empty also means `tracked`); only if the mechanism-control arm is requested later
# MAGIC is it set to `teacher_detect` / `tracked_intersection` (DET-R restricted to the frames both label sources passed),
# MAGIC which prints and records a ⚠. A `tracked` run never reads, and never requires, the teacher-detect by-product
# MAGIC (`annotations/teacher_detect/*`, `intersection_image_ids.json`, `manifest.teacher_detect`). The consumer of this notebook's
# MAGIC output is the detector evaluation in `Tol_44` (is the
# MAGIC fine-tuned detector at least as good as the SAM3 teacher `detect()` on the frames where the tracker actually
# MAGIC called it, both compared to the reviewer-verified masks of 04-22/04-23). Tol_44 reads exactly two things from here:
# MAGIC `Detector_Checkpoints/<run_tag>/checkpoint_best_total.pth` and `train_summary.json` (`op_threshold`, `resolution`,
# MAGIC `post_chain`, `reload`, `checkpoint_best_total_sha256`, `source_days`, plus `run_tag`, `dataset_tag`,
# MAGIC `dataset_manifest_sha256`, `mode` ∈ {full, resume} that Tol_44's `read_train_summary()` requires). The summary is
# MAGIC contract-checked against that key list before it is written and again after re-reading it from the Volume.
# MAGIC
# MAGIC **Inputs.** `TOL/Detector_COCO/<dataset_tag>/` — `manifest.json` (the contract; must be sealed),
# MAGIC `images.tar` (members `train/<file>.jpg`, `valid/<file>.jpg`, verbatim JPEG bytes),
# MAGIC `annotations/<label_source>/train/_annotations.coco.json`, `annotations/tracked/valid/_annotations.coco.json`,
# MAGIC `intersection_image_ids.json` (for `tracked_intersection`). Optional pre-staged pretrained weights
# MAGIC `TOL/Detector_Checkpoints/pretrained/rf-detr-seg-{small,xlarge}.pt` (rfdetr md5-checks them and skips
# MAGIC the `storage.googleapis.com` download). `facebook/sam3` via the HF token (coexistence cell only).
# MAGIC
# MAGIC **Outputs.** `TOL/Detector_Checkpoints/<run_tag>/{checkpoint_best_total.pth, checkpoint_best_ema.pth,
# MAGIC last_ema.pth, metrics.csv, training_config.json, train_summary.json, manifest_copy.json}` (final copies), and the
# MAGIC **resume state** `TOL/Detector_Checkpoints/<run_tag>/resume_state/g<NNNNNN>/{last.ckpt, checkpoint_best_ema.pth,
# MAGIC metrics.csv, *.sha256, COMPLETE.json}` — one NEW directory per background sync tick, never rewritten in place;
# MAGIC `COMPLETE.json` is written last, so a generation without it is a cut upload and is ignored; the newest two complete
# MAGIC generations are kept. (The original layout was a flat `last.ckpt` + sidecar; a flat file is overwritten by the next tick,
# MAGIC and a tick cut mid-copy then leaves NO valid resume state — the generation layout replaces it, review 2026-09-23.)
# MAGIC Also `TOL/Figures/detector_train/<run_tag>/valid_preds_{smoke,best}.png`,
# MAGIC `TOL/Metrics_rfdetr_train_<run_tag>[_smoke].json`. A generation is published only when its three files are mutually
# MAGIC consistent under the same rule `mode=resume` applies: rfdetr's `BestModelCallback` keeps its EMA high-water mark
# MAGIC inside `last.ckpt` (PTL 2.x saves rfdetr's `last` checkpoint in `on_train_epoch_end`, i.e. AFTER the validation that
# MAGIC raised the mark and wrote `checkpoint_best_ema.pth`, `trainer.py:320-333`; only a snapshot in the in-flight window —
# MAGIC `last.ckpt` still one epoch behind the `metrics.csv` best epoch — carries a lower mark, and is patched) and, if
# MAGIC `checkpoint_best_ema.pth` is missing at the end of a resumed run, silently backfills the FINAL EMA weights as
# MAGIC "best" (`best_model.py:606-617`).
# MAGIC
# MAGIC **Assumptions (each asserted).** One node, one 80 GB GPU with bf16; `manifest.sealed == true`, or the only unmet
# MAGIC seal term is `below_floor` and `allow_below_floor=true` (smoke, a failed human gate and a skipped veto
# MAGIC without Tol_42 `require_veto=false` can never be waived). **Human gate**: when the manifest records
# MAGIC `human_gate.required: false` — the reviewed fault list is the human gate — a non-empty `human_gate.unlisted_days`
# MAGIC refuses (the fault list covers only 04-18/04-19), a present gate file is re-judged from its counts (BAD counted > 3 or
# MAGIC any `wrong_object` = failed, whatever `status` says), a failed gate refuses and any other status is recorded; when `required` is true or absent (older manifests) the seal rule is re-checked from
# MAGIC the COUNTED-cell counts (`n_bad_counted`, or `reject_rate × n_counted`), not just `status`. `source_days ⊆
# MAGIC {20240418, 20240419, 20240422}` always; `20240423..28` never (no widget admits them); `20240422` only with
# MAGIC `allow_extra_days=true` AND a non-empty `manifest.allow_test_day_reason` AND the dense strides 8 s / 4 s in
# MAGIC `constants.strides` (densify first; `allow_undensified_extra_day=true` skips that order, recorded)
# MAGIC (then Tol_44 may evaluate 04-23 only);
# MAGIC every POOLED constant and the band JSON derived from `TRAIN_DAYS` only, the per-day `A_ref[d]` /
# MAGIC `ACTIVE_THR[d]` from their own source day; `teacher_detect.drop_high` is a recorded note, never a refusal (DET-C
# MAGIC is not planned; if it is trained later its DET-R partner must be `tracked_intersection`); a completed run under the
# MAGIC same `run_tag` is never overwritten without `force=true`; a resume generation whose recorded provenance or
# MAGIC train hyper-parameters differ from this run is refused; train/valid whole-clip disjoint and within
# MAGIC `manifest.split`; `categories == [{"id":1,"name":"calf","supercategory":"none"}]`; annotation sha256 equal to the
# MAGIC manifest; `resolution % 24 == 0` **before** construction (the constructor does not check, `detr.py:388-395`);
# MAGIC `mode=full|resume` refuses to run without a smoke projection whose inputs (variant, resolution, N_train, N_valid,
# MAGIC batch/accum that survived the smoke OOM policy) equal the current widgets, and re-projects the hours with the
# MAGIC CURRENT `epochs` before the stop gate. Review 2026-09-23: the seal is re-derived even when `sealed == true`
# MAGIC (an unmet term on a sealed manifest raises); the band constants Tol_44 reads (`frag_lo[dayl|dark]`,
# MAGIC `frag_lo_provenance` ∈ {derived_P1_of_U, user_override}, `over_hi`, `S_lo/S_hi[pool]`, `STAB_LO/HI`, `A_ref[d]` per
# MAGIC source day, `valid_buffer_min ≥ 10`) must be present, a derived `frag_lo` in [0.30, 0.59]; `gates` must prove G9
# MAGIC ran; valid must hold ≥ 50 night images (`allow_thin_night_valid=true` overrides, recorded); no train image lies
# MAGIC within `valid_buffer_min` of a same-day valid image.
# MAGIC
# MAGIC **What the numbers here mean.** *valid mAP is agreement with teacher-derived labels on training days;
# MAGIC it is not detector quality.* It selects the checkpoint and the operating threshold —
# MAGIC nothing else. Whether the detector may replace the teacher is decided only by `Tol_44`.
# MAGIC
# MAGIC Local self-test (no GPU / dbutils / network): `TOL_SELFTEST=1 python Tol_43_train_rfdetr_seg.py`.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm" "scipy" "rfdetr==1.10.1" "pytorch_lightning>=2.6,!=2.6.2,!=2.6.3,<3" "torchmetrics[detection]>=1.8.2,<1.9" "faster-coco-eval>=1.7.2" "pycocotools" "peft"

# COMMAND ----------

# MAGIC %pip uninstall -y opencv-python opencv-contrib-python

# COMMAND ----------

# MAGIC %pip install --no-deps --force-reinstall "opencv-python-headless==4.11.0.86"

# COMMAND ----------

# Not rfdetr[train]: it drags roboflow + torch-hungarian (never imported) and
# supervision pulls the GUI opencv-python, which must not shadow the headless build.
try:
    dbutils.library.restartPython()
except NameError:
    pass   # local / SELFTEST mode: plain python

# COMMAND ----------

import os, os.path as osp, sys, re, json, csv, time, gc, shutil, hashlib, tarfile, threading, random
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

SELFTEST_ONLY = os.environ.get("TOL_SELFTEST") == "1"     # every Volume/model/GPU cell is skipped

IS_NOTEBOOK = True
try:
    dbutils
except NameError:
    IS_NOTEBOOK = False
if not IS_NOTEBOOK:
    # local SELFTEST on Windows: a redirected/piped stdout is cp1252 and cannot encode the ⚠ markers (review 2026-09-23)
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError): pass

def P(name, default):
    # VERBATIM from Tol_03_track_v4:71-76 (env prefix renamed TOL_; workers never assume dbutils)
    if IS_NOTEBOOK:
        try: dbutils.widgets.text(name, default)
        except Exception: pass
        return (dbutils.widgets.get(name) or default)
    return os.environ.get("TOL_" + name.upper(), default)

VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
TRAIN_DAYS = {"20240418", "20240419"}                       # the only detector training days
TEST_DAYS_HARD = {f"202404{d}" for d in range(23, 29)}        # 04-23..04-28 may never train anything
TEST_DAY_SOFT = "20240422"                                    # admissible only with allow_extra_days

# ---- the incumbent detector contract, verbatim (Tol_03_track_v4:225-227) ------------------------
MIN_MASK_PX      = 8000
DET_SCORE_MIN, DET_TEXT = 0.40, "calf"
DEDUPE_IOS       = 0.50
SAM3_ID = "facebook/sam3"                                     # Tol_03:96
TOPK_DET = 6                                                  # Tol_03:1167 / :727 call detect(pil, topk=N_CALVES+2)

# ---- label source (decision 2026-09-23; not derived from any day) ------------------------
LABEL_SOURCES = ("tracked", "teacher_detect", "tracked_intersection")   # vocabulary
LABEL_SOURCE_DEFAULT = "tracked"   # DET-R on the tracked (04-18/04-19 reviewer-checked) masks is the only run planned; DET-C / the mechanism-control arm skipped

# ---- what Tol_44 (arm `rfdetr`) reads from Detector_Checkpoints/<run_tag>/ -----------------
BEST_CKPT_FILE = "checkpoint_best_total.pth"   # Tol_44's rfdetr-arm cell stages exactly this name
SUMMARY_FILE   = "train_summary.json"          # Tol_44 read_train_summary() reads exactly this name
TOL44_TS_KEYS  = ("run_tag", "dataset_tag", "dataset_manifest_sha256", "source_days", "op_threshold", "resolution",
                  "checkpoint_best_total_sha256", "mode")   # VERBATIM the key list of Tol_44 read_train_summary() (it raises on any absent; SELFTEST re-parses it)
TOL44_TS_MODES = ("full", "resume")            # Tol_44 read_train_summary() refuses any other mode, and any truthy `smoke`

# ---- RF-DETR literals (each cited) --------------------------------------------
SEG_BLOCK = 24              # patch_size 12 x num_windows 2 for every Seg tier except Nano
CATEGORIES_EXPECTED = [{"id": 1, "name": "calf", "supercategory": "none"}]   # -> num_classes 1, label 0
SMOKE_EPOCHS = 2            # smoke-run epochs
PROJ_OVERHEAD = 1.15        # x1.15 on the projected hours
T_SETUP_FALLBACK_S = 60.0   # assumed when the first step could not be observed
T_VAL_FALLBACK_FRAC = 0.25  # T_val = 0.25*T_total if evaluate() fails on a signature reason
SWEEP_TAUS = [round(0.20 + 0.05 * i, 2) for i in range(11)]   # 0.20 .. 0.70
SWEEP_PROBE_THR = 0.10      # one predict() at 0.10 per valid image, thresholds applied offline
MATCH_IOU_MIN = 0.50        # Hungarian match counts as TP at mask IoU >= 0.5
NUM_WORKERS = 12            # (NC24ads_A100_v4 has 24 vCPU; every sample decodes a full 2304x1296 JPEG)
MIN_TRAIN_BATCHES = 5       # rfdetr training/module_data.py:29 _MIN_TRAIN_BATCHES -> replacement sampler below it
FRAME_H, FRAME_W = 1296, 2304                                 # Tol native frame
PHASES4 = ("night", "dawn", "dusk", "day")                    # phase4 vocabulary; every image's phase4 is checked against it
HUMAN_GATE_BAD = ("partial", "split", "merged", "unlabelled_calf", "wrong_object")   # BAD verdicts; n_bad_by_kind keys outside BAD ∪ HUMAN_GATE_OK raise
HUMAN_GATE_OK = ("ok", "partial_minor")                                                 # vocabulary: partial_minor counts as ok
HUMAN_GATE_MAX_BAD = 3          # seal rule: count(BAD) <= 3 among COUNTED cells (n_bad_by_kind also counts the uncounted extras)
HUMAN_GATE_SHEET_N_DEFAULT = 300   # sheet_n=300 (~300 reviewed overlays): a RECORDED planned sheet below it raises; counted cells below it (per-stratum pool shortfall) are recorded ⚠
HUMAN_GATE_TAIL_ENLARGE = 50       # the tail stratum grows 100 -> 150 when bad_overlap_high, so counted cells may exceed the sheet_n WIDGET by 50*sheet_n/300
PER_DAY_CONST_RX = re.compile(r"^(A_ref|ACTIVE_THR)\[(\d{8})\]$")   # Tol_42 manifest names of the DERIVED(day) constants; everything else is pooled from TRAIN_DAYS
# The band constants Tol_44 and the G6-G9 gates read, exactly as Tol_42's manifest cell names them (Tol_42:2139-2144);
# pooled over the 04-18/04-19 admitted set U. A manifest lacking any of them cannot be scored by Tol_44 (review 2026-09-23).
REQUIRED_BAND_CONSTS = ("frag_lo[dayl]", "frag_lo[dark]", "frag_lo_provenance", "over_hi", "S_lo[dayl]", "S_lo[dark]",
                        "S_hi[dayl]", "S_hi[dark]", "STAB_LO", "STAB_HI")
FRAG_LO_SANITY = (0.30, 0.59)   # raise band (= Tol_42:190), 04-18/04-19 U; upper edge = Tol_41's 0.59 band (must be relaxed below it so the reviewer-confirmed curled night stretch stays admitted)
FRAG_LO_PROVENANCES = ("derived_P1_of_U", "user_override")   # the only two values Tol_42 derive_constants writes (Tol_42:699 / :737)
VALID_BUFFER_MIN_FLOOR = 10.0   # default / G12 (minutes; not derived from any day): below it near-duplicates straddle the train/valid split
GATES_REQUIRED = ("funnel", "curled_or_clipped_blocks", "blocks_failing_per_slot_0.75", "collapse_overrides")   # gates block: proves G9 ran
VALID_NIGHT_MIN = 50            # project choice (review 2026-09-23), valid clips of 04-18/04-19: the split puts two night clips in valid so checkpoint
                                # selection sees night (~500 night lattice frames before gating); < 50 means G9/G11 wiped them out. Override: allow_thin_night_valid
DENSE_STRIDES = (8.0, 4.0)      # decision 2026-09-23 (= Tol_42 DENSE_STRIDES): (stride_day_s, stride_other_s) chosen to try BEFORE adding 04-22; protocol value, not derived from any day
RESUME_EMA_EPS = 1e-9           # float text round-trip slack when comparing last.ckpt's _best_ema with the metrics.csv best
SYNC_SETTLE_S = 1.0             # a checkpoint is snapshotted only when size+mtime are unchanged across this many seconds
SYNC_SETTLE_TRIES = 8           # PTL rewrites last.ckpt in place every epoch; 8 x 1 s covers a ~0.5 GB local write
RESUMABLE = ("last.ckpt", "checkpoint_best_ema.pth", "metrics.csv")   # the resume state (header); each gets a .sha256 sidecar
UPLOAD_ORDER = ("metrics.csv", "checkpoint_best_ema.pth", "last.ckpt")  # small/best first; a generation is valid only once GEN_COMPLETE exists
RESUME_DIRNAME = "resume_state"   # CK_VOL/resume_state/g<NNNNNN>/: one directory per published sync tick, never overwritten in place (review 2026-09-23)
GEN_COMPLETE = "COMPLETE.json"    # written LAST in a generation: tick id, per-file sha256/epoch, the run's PROVENANCE + train hparams
RESUME_KEEP_GENERATIONS = 2       # the newest complete generation + one fallback survive each publish (~2 x 0.7 GB for seg-small)
SNAPSHOT_SET_TRIES = 6            # the three local snapshots are retaken until mutually consistent (resume_highwater_action passes)
SNAPSHOT_SET_WAIT_S = 20.0        # between attempts: CSVLogger flushes the val row after the checkpoint callbacks (FirstStepProbe note), 6 x 20 s = 2 min
RESUME_BOUND_PROV = ("dataset_tag", "dataset_manifest_sha256", "source_days", "label_source", "variant", "resolution", "seed", "run_tag",
                     "train_frameset_file_names_sha256", "train_frameset_n_full")   # PROVENANCE keys a resumed state must share with the current run
RESUME_BOUND_HPARAMS = ("epochs", "batch_size", "grad_accum_steps", "lr", "lr_encoder", "weight_decay", "ema_decay", "scale_jitter",
                        "eval_masks_head_resolution", "early_stopping_patience", "checkpoint_interval")   # train() kwargs a resumed state must share

# FROZEN whitelist of legal train() kwargs = every TrainConfig field (rfdetr config.py:1011-1270) +
# SegmentationTrainConfig fields (config.py:1616-1636) + the two kwargs _prepare_run_config pops
# before construction, `device` (detr.py:372) and `resolution` (detr.py:378). extra="forbid"
# (config.py:1009) raises on anything else; the whitelist makes that failure local and cheap.
TRAIN_KW_WHITELIST = frozenset({
    "lr", "lr_encoder", "batch_size", "grad_accum_steps", "eval_batch_size", "auto_batch_target_effective",
    "auto_batch_max_targets_per_image", "auto_batch_ema_headroom", "epochs", "resume", "ema_decay", "ema_tau",
    "lr_drop", "checkpoint_interval", "skip_best_epochs", "smooth_alpha", "warmup_epochs", "lr_vit_layer_decay",
    "lr_component_decay", "drop_path", "cls_loss_coef", "keypoint_flip_pairs", "keypoint_l1_loss_coef",
    "keypoint_findable_loss_coef", "keypoint_visible_loss_coef", "keypoint_nll_loss_coef", "keypoint_oks_sigmas",
    "dataset_file", "square_resize_div_64", "dataset_dir", "output_dir", "multi_scale", "expanded_scales",
    "do_random_resize_via_padding", "use_ema", "ema_update_interval", "eval_base_model", "eval_ema_only",
    "num_workers", "weight_decay", "amp_dtype", "best_model_metric", "early_stopping", "early_stopping_patience",
    "early_stopping_min_delta", "early_stopping_use_ema", "progress_bar", "tensorboard", "wandb", "mlflow",
    "clearml", "project", "run", "class_names", "run_test", "eval_max_dets", "eval_interval",
    "log_per_class_metrics", "eval_masks_head_resolution", "aug_config", "scale_jitter", "augmentation_backend",
    "save_dataset_grids", "notes", "accelerator", "clip_max_norm", "seed", "sync_bn", "strategy", "devices",
    "num_nodes", "fp16_eval", "lr_scheduler", "lr_scheduler_kwargs", "lr_scheduler_interval",
    "lr_scheduler_monitor", "lr_min_factor", "optimizer", "optimizer_kwargs", "dont_save_weights",
    "train_log_sync_dist", "train_log_on_step", "compact_train_metrics", "compute_train_metrics",
    "num_sanity_val_steps", "compute_val_loss", "compute_test_loss", "pin_memory", "persistent_workers",
    "prefetch_factor", "pack_targets",
    # SegmentationTrainConfig (config.py:1632-1635)
    "mask_point_sample_ratio", "mask_ce_loss_coef", "mask_dice_loss_coef",
    # popped by RFDETR.train() before TrainConfig is built (detr.py:372, :378)
    "device", "resolution",
})
# Constructor-only ModelConfig kwargs (config.py:433-527): passing any of these to train() raises.
CONSTRUCTOR_ONLY = frozenset({"num_classes", "pretrain_weights", "gradient_checkpointing", "amp", "compile",
                              "num_queries", "num_select", "group_detr", "positional_encoding_size",
                              "backbone_lora", "freeze_encoder", "mask_downsample_ratio", "fused_optimizer",
                              "postprocess_trace_alpha", "num_channels", "segmentation_head", "model_name",
                              "license"})

# COMMAND ----------

# MAGIC %md ## Pure-python helpers (no widget, no dbutils, no GPU) — exercised by the SELFTEST cell

# COMMAND ----------

# VERBATIM from Tol_40_train_label_survey:63-78
def runs_of(rle):
    c = np.asarray(rle["counts"], dtype=np.int64)
    ends = np.cumsum(c); starts = ends - c
    s, e = starts[1::2], ends[1::2]
    return s, e, int((e - s).sum())

def inter(A, B):
    sa, ea, _ = A; sb, eb, _ = B
    i = j = 0; tot = 0
    while i < len(sa) and j < len(sb):
        s = sa[i] if sa[i] > sb[j] else sb[j]
        e = ea[i] if ea[i] < eb[j] else eb[j]
        if e > s: tot += e - s
        if ea[i] < eb[j]: i += 1
        else: j += 1
    return tot

# VERBATIM from Tol_03_track_v4:280-308
def rle_encode(mask):
    h, w = mask.shape
    flat = np.asfortranarray(mask).reshape(-1, order="F")
    idx = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate(([0], idx, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0]: counts = [0] + counts
    return {"size": [h, w], "counts": counts}

def rle_decode(r):
    h, w = r["size"]
    c = np.asarray(r["counts"], dtype=np.int64)
    vals = np.zeros(len(c), bool); vals[1::2] = True
    return np.repeat(vals, c).reshape((h, w), order="F")

def m_iou(a, b):
    if a is None or b is None: return 0.0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()/u) if u else 0.0

def m_ios(a, b):
    if a is None or b is None: return 0.0
    m = min(a.sum(), b.sum())
    return float(np.logical_and(a, b).sum()/m) if m else 0.0

def m_bbox(m):
    ys, xs = np.where(m)
    if len(xs) == 0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

# ---- time / phase -----------------------------------------------------------------
RXT = re.compile(r"-(\d{8})-(\d{6})-")                        # Tol_40:80
def clip_start(v):
    m = RXT.search(v.replace("_", "-"))
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

def phase4(w):                       # night/dawn/dusk/day
    if w.hour >= 21 or w.hour < 6: return "night"
    if w.hour < 8:                 return "dawn"      # 06:00-07:59
    if w.hour == 20:               return "dusk"
    return "day"

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

# ---- rfdetr call hygiene ------------------------------------------------------------------------
def check_train_kwargs(kw):
    """Raise on any key rfdetr's TrainConfig(extra='forbid') would reject (config.py:400-424) — before the
    model is built and the dataset staged, not after. Constructor-only keys get a pointed message."""
    bad = sorted(k for k in kw if k not in TRAIN_KW_WHITELIST)
    if bad:
        ctor = [k for k in bad if k in CONSTRUCTOR_ONLY]
        raise RuntimeError(f"train() kwargs not in TrainConfig whitelist: {bad}"
                           + (f" (constructor-only ModelConfig fields: {ctor})" if ctor else ""))
    if "dataset_dir" not in kw:
        raise RuntimeError("dataset_dir is a required TrainConfig field (config.py:1062)")
    if kw.get("batch_size") == "auto":
        raise RuntimeError("batch_size='auto' is forbidden (pinned ints; the probe ignores validation memory)")
    if kw.get("amp_dtype") != "bf16":
        raise RuntimeError(f"amp_dtype must be 'bf16' (bf16 only, never fp16), got {kw.get('amp_dtype')!r}")
    return True

def multi_scale_scales(resolution, expanded_scales=False, patch_size=12, num_windows=2):
    """Pure replica of rfdetr datasets/coco.py:277-291 compute_multi_scale_scales (cross-checked at run time)."""
    base = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [(base + o) * patch_size * num_windows for o in offsets]
    return [s for s in scales if s >= patch_size * num_windows * 2]

def resolve_precision_replica(amp, amp_dtype, cuda_available, bf16_supported, accelerator="gpu"):
    """Replica of the nested _resolve_precision in rfdetr training/trainer.py:558-621 (not importable —
    it is a closure inside build_trainer). Used to state the precision the trainer will run at."""
    if not amp: return "32-true"
    if accelerator == "cpu": return "32-true"
    if cuda_available:
        if amp_dtype == "fp16": return "16-mixed"
        if amp_dtype == "bf16": return "bf16-mixed" if bf16_supported else "16-mixed"
        return "bf16-mixed" if bf16_supported else "16-mixed"     # "auto"
    return "32-true"

# ---- smoke projection -----------------------------------------------------
def project_hours(T_total, T_val, T_setup, smoke_train_n, smoke_valid_n, N_train, N_valid, epochs,
                  smoke_epochs=SMOKE_EPOCHS, overhead=PROJ_OVERHEAD):
    """ips = smoke_epochs*smoke_train_n / (T_total - smoke_epochs*T_val - T_setup);
    t_val_img = T_val / smoke_valid_n; hours = epochs*(N_train/ips + N_valid*t_val_img)*overhead/3600."""
    denom = T_total - smoke_epochs * T_val - T_setup
    if denom <= 0:
        raise RuntimeError(f"projection denominator <= 0: T_total {T_total:.1f} - {smoke_epochs}*T_val {T_val:.1f} "
                           f"- T_setup {T_setup:.1f} = {denom:.1f}; the smoke timing is inconsistent")
    ips = smoke_epochs * smoke_train_n / denom
    t_val_img = T_val / max(1, smoke_valid_n)
    hours = epochs * (N_train / ips + N_valid * t_val_img) * overhead / 3600.0
    return dict(ips=ips, t_val_img=t_val_img, hours_full=hours, denom_s=denom)

def stop_gate(hours_full, max_hours, epochs, resolution):
    """Raise with the two cheapest fallbacks; neither is auto-applied."""
    if hours_full > max_hours:
        raise RuntimeError(
            f"STOP GATE: projected full run {hours_full:.1f} h > max_hours {max_hours:.1f} h. "
            f"Cheapest fallbacks (your call, not auto-applied): (1) epochs {epochs} -> {max(1, round(epochs * 25 / 40))} "
            f"(25/40 of the budget); (2) resolution {resolution} -> {max(SEG_BLOCK * 2, resolution - 3 * SEG_BLOCK)} "
            f"(one multi-scale step down). The training set is never cut below min_train to pass this gate.")
    return True

# ---- sha256 sidecar for last.ckpt --------------------------------------------------
def write_sidecar(ckpt_path, sidecar_path, sha=None):
    sha = sha or sha256_file(ckpt_path)
    with open(sidecar_path, "w") as f: f.write(sha + "\n")
    return sha

def verify_sidecar(ckpt_path, sidecar_path):
    """Raise unless sha256(ckpt) equals the sidecar — a mismatch means the Volume copy is partial."""
    if not osp.exists(sidecar_path):
        raise RuntimeError(f"no sha256 sidecar next to {ckpt_path}: partial checkpoint on the Volume; use mode=full")
    with open(sidecar_path) as f: want = f.read().strip().split()[0]
    have = sha256_file(ckpt_path)
    if have != want:
        raise RuntimeError(f"sha256 mismatch for {ckpt_path}: have {have[:12]} want {want[:12]} — "
                           "partial checkpoint on the Volume; use mode=full")
    return have

def file_sig(path):
    """(size, mtime_ns) or None — the stability signature a snapshot is taken on."""
    if not osp.exists(path): return None
    st = os.stat(path); return (st.st_size, st.st_mtime_ns)

def stable_snapshot(src, dst, settle_s=SYNC_SETTLE_S, tries=SYNC_SETTLE_TRIES, sleep=time.sleep):
    """Copy src -> dst only when src's (size, mtime) is unchanged across `settle_s` before AND after the copy;
    PTL's ModelCheckpoint rewrites last.ckpt in place every epoch (trainer.py:322-333), so an unguarded copy can
    capture a truncated file whose sha256 would then be certified. Returns the signature copied, or None when
    src is absent. Raises after `tries` unstable attempts (the next tick retries)."""
    for _ in range(tries):
        a = file_sig(src)
        if a is None: return None
        sleep(settle_s)
        if file_sig(src) != a: continue
        shutil.copyfile(src, dst)
        if file_sig(src) == a and osp.getsize(dst) == a[0]: return a
    raise RuntimeError(f"{src} kept changing across {tries} x {settle_s:.0f} s — no stable snapshot this tick")

def validate_zip_file(path):
    """torch.save / PTL checkpoints are zip containers (torch >= 1.6): a truncated copy fails testzip() or is
    not a zip at all. Raises with the reason; returns True."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"{path} is not a complete zip container: {e}")
    if bad is not None: raise RuntimeError(f"{path}: CRC failure in member {bad!r} — truncated snapshot")
    return True

def is_zip_container(name):
    return name.endswith(".ckpt") or name.endswith(".pth")

# ---- manifest guards -------------------
def counted_bad_from_gate(hg):
    """count(BAD) among COUNTED cells only — the quantity Tol_42's seal_rule used. Tol_42.human_gate_eval increments
    n_bad_by_kind for EVERY cell (the uncounted rejected_G*/vetoed/flagged_furniture extras included) and the manifest
    carries n_counted + reject_rate (= n_bad_counted / n_counted, 4 dp) but not n_bad_counted itself, so: read
    n_bad_counted when a later Tol_42 writes it, else recover it as round(reject_rate * n_counted) — exact while
    n_counted < 5000 (4-dp error < 0.25). Raises when neither is derivable."""
    if hg.get("n_bad_counted") is not None: return int(hg["n_bad_counted"])
    if hg.get("reject_rate") is None:
        raise RuntimeError("manifest.human_gate lacks both n_bad_counted and reject_rate — the counted-only seal rule cannot be re-checked")
    n_counted = int(hg["n_counted"])
    if n_counted >= 5000: raise RuntimeError(f"manifest.human_gate.n_counted={n_counted}: reject_rate (4 dp) cannot recover n_bad_counted exactly")
    return int(round(float(hg["reject_rate"]) * n_counted))

def human_gate_check(hg, sheet_n=None):
    """Recompute the seal rule from the manifest's own verdict counts: status == 'passed' is necessary,
    not sufficient. Every n_bad_by_kind key must be in HUMAN_GATE_BAD ∪ HUMAN_GATE_OK (an unknown verdict kind raises);
    count(BAD) <= 3 is checked on the COUNTED cells (counted_bad_from_gate); wrong_object == 0 on every cell. n_counted
    must be > 0. `sheet_n` = manifest_sheet_n(man) = (value, kind) or None: a recorded planned size below
    HUMAN_GATE_SHEET_N_DEFAULT raises (~300 reviewed overlays are required); n_counted above the recorded size (+ the tail enlargement
    when the value is the widget) raises as inconsistent. A shortfall is legal — human_gate_size_notes
    returns it for the silent_fallbacks list. Raises on any violation; returns the counted BAD count."""
    if hg.get("status") != "passed":
        raise RuntimeError(f"manifest.human_gate.status = {hg.get('status')!r}, need 'passed'")
    for k in ("n_counted", "n_bad_by_kind"):
        if k not in hg: raise RuntimeError(f"manifest.human_gate lacks {k!r} — the seal cannot be re-checked")
    nb = hg["n_bad_by_kind"] or {}
    unknown = sorted(k for k in nb if k not in HUMAN_GATE_BAD and k not in HUMAN_GATE_OK)
    if unknown:
        raise RuntimeError(f"manifest.human_gate.n_bad_by_kind has verdict kinds {unknown} outside the verdict vocabulary "
                           f"{list(HUMAN_GATE_OK) + list(HUMAN_GATE_BAD)} — Tol_42's BAD set changed; this notebook cannot re-check the seal")
    if int(nb.get("wrong_object", 0)) > 0:
        raise RuntimeError(f"manifest.human_gate has wrong_object={nb.get('wrong_object')} > 0 but status 'passed' (zero allowed anywhere)")
    n_bad = counted_bad_from_gate(hg)
    if n_bad > HUMAN_GATE_MAX_BAD:
        raise RuntimeError(f"manifest.human_gate: count(BAD among counted cells)={n_bad} > {HUMAN_GATE_MAX_BAD} (by_kind over all cells {nb}) but status 'passed'")
    n_counted = int(hg["n_counted"])
    if n_counted <= 0: raise RuntimeError(f"manifest.human_gate.n_counted={n_counted}: no counted cell was reviewed")
    if sheet_n is not None:
        val, kind = int(sheet_n[0]), sheet_n[1]
        if val < HUMAN_GATE_SHEET_N_DEFAULT:
            raise RuntimeError(f"manifest records {kind}={val} < {HUMAN_GATE_SHEET_N_DEFAULT}: the human gate PLANNED fewer than the ~300 reviewed "
                               "overlays required before training (only a smoke export may, and smoke is never trained on)")
        cap = val if kind == "sheet_n_planned" else val + -(-HUMAN_GATE_TAIL_ENLARGE * val // HUMAN_GATE_SHEET_N_DEFAULT)
        if n_counted > cap: raise RuntimeError(f"manifest.human_gate.n_counted={n_counted} > {cap} allowed by the export's {kind}={val}: the counts are inconsistent")
    return n_bad

def human_gate_required(man):
    """Decision 2026-09-23: Tol_42 records human_gate.required (its require_human_gate widget, default false —
    the reviewed fault list IS the human gate). Absent => True: a manifest written before that decision was sealed under the
    full seal rule and is re-checked under it. A non-bool value raises (a string "false" must not read as truthy)."""
    r = (man.get("human_gate") or {}).get("required", True)
    if not isinstance(r, bool):
        raise RuntimeError(f"manifest.human_gate.required = {r!r} is not a bool — cannot tell whether the human gate binds")
    return r

def human_gate_optional_check(hg):
    """required=false: Tol_42 still validates a present human_gate.json, still excludes its BAD cells
    (±90 s) and a failed gate still unseals. The waiver rests on the reviewed fault list, which covers ONLY the two reviewed
    training days, so a non-empty human_gate.unlisted_days (a source day with no fault list, e.g. 04-22) refuses: such a
    day needs require_human_gate=true (Tol_42 refuses it at finalize too). The failed condition is RE-DERIVED from the
    counts whenever a gate file was present (review 2026-09-23; Tol_42.human_gate_verdict): status 'failed', OR
    count(BAD among counted cells) > HUMAN_GATE_MAX_BAD, OR any wrong_object anywhere — whatever `status` says. The size
    of the review (n_counted, sheet_n) is not a term in this mode. Returns the status for the record."""
    st = hg.get("status")
    if st == "failed":
        raise RuntimeError("manifest.human_gate.status = 'failed' — a present-but-failed human gate always unseals "
                           "whether required or not")
    unlisted = hg.get("unlisted_days") or []
    if unlisted:
        raise RuntimeError(f"manifest.human_gate.unlisted_days = {sorted(unlisted)} with human_gate.required=false: the reviewed fault list "
                           "(the human gate when required=false) covers only 20240418/20240419, so a source day without one "
                           "needs require_human_gate=true — refused (Tol_42 refuses this at finalize as well)")
    present = hg.get("present", False)
    if not isinstance(present, bool):
        raise RuntimeError(f"manifest.human_gate.present = {present!r} is not a bool — cannot tell whether a review file was validated")
    if present:
        if "n_counted" not in hg:
            raise RuntimeError("manifest.human_gate.present=true but n_counted is absent — the failed condition cannot be re-derived")
        nb = hg.get("n_bad_by_kind") or {}
        n_wrong = max(int(hg.get("n_wrong_object") or 0), int(nb.get("wrong_object", 0) or 0))
        n_bad = counted_bad_from_gate(hg)
        if n_wrong > 0 or n_bad > HUMAN_GATE_MAX_BAD:
            raise RuntimeError(f"manifest.human_gate (required=false, present) has count(BAD among counted cells)={n_bad} (max {HUMAN_GATE_MAX_BAD}) "
                               f"and wrong_object={n_wrong} (max 0) but status {st!r}: by Tol_42's own rule this gate FAILED, and a failed gate "
                               "always unseals — the manifest is inconsistent; refused")
    return st

def human_gate_size_notes(hg, sheet_n=None):
    """The non-fatal human-gate size facts (a per-stratum pool shortfall is legal and printed), as ⚠ strings the
    caller prints AND appends to silent_fallbacks: counted cells below the recorded plan, counted cells below the required ~300
    reviewed overlays, and a manifest that does not record its planned sheet size at all (then only n_counted is known)."""
    n_counted = int(hg.get("n_counted") or 0); notes = []
    if sheet_n is None:
        notes.append(f"human gate: the manifest records no sheet_n/sheet_n_planned; the planned review size cannot be re-checked here "
                     f"(Tol_42 refuses sheet_n < {HUMAN_GATE_SHEET_N_DEFAULT} outside smoke); n_counted={n_counted}")
    elif n_counted < int(sheet_n[0]):
        notes.append(f"human gate sheet shortfall: {n_counted} counted cells < {sheet_n[1]} {sheet_n[0]} (a stratum pool ran short; legal)")
    if n_counted < HUMAN_GATE_SHEET_N_DEFAULT:
        notes.append(f"human gate reviewed {n_counted} counted cells < {HUMAN_GATE_SHEET_N_DEFAULT} (~300 reviewed overlays are required)")
    return notes

def constants_provenance_check(man, allow_extra_days=False):
    """Provenance of the band constants. band_json.derived_from ⊆ TRAIN_DAYS always. POOLED constants
    (frag_lo[*], S_lo/S_hi[*], STAB_*, body_len_px, over_hi, gate literals, ...) have derived_from ⊆ TRAIN_DAYS and
    not_from ⊇ every test day 04-22..28, allow_extra_days or not (04-22 masks may be LABELS, never a source of a pooled
    constant — Tol_42 excludes non-training days from U). The DERIVED(day) constants `A_ref[d]` / `ACTIVE_THR[d]` come
    from their own day only: derived_from == [d]; d must be a source day; d == 20240422 additionally needs
    allow_extra_days; their not_from must cover the test days other than d. Raises naming every offender;
    returns the sha256 pair for provenance.
    PRESENCE + VALUE (review 2026-09-23): every REQUIRED_BAND_CONSTS name, valid_buffer_min and A_ref[d] for every source day
    must exist with a value (band constants with a non-empty derived_from); frag_lo_provenance ∈ FRAG_LO_PROVENANCES;
    a derived frag_lo[pool] lies in FRAG_LO_SANITY, a user_override one in (0, 0.59] (above 0.59 nothing
    was relaxed) and is returned as a note; frag_lo[pool] < over_hi; valid_buffer_min >= VALID_BUFFER_MIN_FLOOR."""
    bj = man["band_json"]; bad = []; notes = []
    for k in ("sha256", "derived_from"):
        if k not in bj: bad.append(f"band_json lacks {k!r}")
    extra = sorted(set(str(d) for d in bj.get("derived_from", [])) - TRAIN_DAYS)
    if extra: bad.append(f"band_json.derived_from has non-training days {extra}")
    if not bj.get("derived_from"): bad.append("band_json.derived_from is empty")
    test_days = TEST_DAYS_HARD | {TEST_DAY_SOFT}
    source_days = {str(d) for d in man.get("source_days", [])}
    cons = man["constants"]
    if not isinstance(cons, dict) or not cons: bad.append("constants block empty")
    for name, c in (cons.items() if isinstance(cons, dict) else []):
        if not isinstance(c, dict) or "derived_from" not in c or "not_from" not in c:
            bad.append(f"constants[{name!r}] lacks derived_from/not_from"); continue
        df = set(str(d) for d in c["derived_from"]); nf = set(str(d) for d in c["not_from"])
        mday = PER_DAY_CONST_RX.match(str(name))
        if mday:
            d = mday.group(2)
            if df != {d}:
                bad.append(f"constants[{name!r}].derived_from {sorted(df)} != [{d}] (a DERIVED(day) constant comes from its own day only)")
            elif d not in TRAIN_DAYS:
                if d in TEST_DAYS_HARD or d not in source_days:
                    bad.append(f"constants[{name!r}] derived from {d}, which is not a source day")
                elif not allow_extra_days:
                    bad.append(f"constants[{name!r}] derived from {d}, which is not a training day; allow_extra_days=true is required")
            required = test_days - df
        else:
            ex = sorted(df - TRAIN_DAYS)
            if ex: bad.append(f"constants[{name!r}].derived_from has non-training days {ex} (pooled constants come from TRAIN_DAYS only)")
            required = test_days
        missing = sorted(required - nf)
        if missing: bad.append(f"constants[{name!r}].not_from lacks test days {missing}")
    # ---- presence and values of the constants Tol_44 and the gates read (review 2026-09-23) ----
    cons = cons if isinstance(cons, dict) else {}
    need = list(REQUIRED_BAND_CONSTS) + ["valid_buffer_min"] + [f"A_ref[{d}]" for d in sorted(source_days)]
    absent = [k for k in need if not isinstance(cons.get(k), dict) or cons[k].get("value") is None]
    if absent: bad.append(f"constants lack required entries {absent} (Tol_44 reads frag_lo[dayl|dark]/over_hi, the gates read the rest)")
    for k in REQUIRED_BAND_CONSTS:
        if k not in absent and isinstance(cons.get(k), dict) and not cons[k].get("derived_from"):
            bad.append(f"constants[{k!r}].derived_from is empty: a band constant must name the training days it was derived from")
    def _num(k):
        c = cons.get(k)
        if k in absent or not isinstance(c, dict): return None
        try: return float(c["value"])
        except (TypeError, ValueError): bad.append(f"constants[{k!r}].value {c.get('value')!r} is not a number"); return None
    fl_prov = (cons.get("frag_lo_provenance") or {}).get("value") if "frag_lo_provenance" not in absent else None
    if fl_prov is not None and fl_prov not in FRAG_LO_PROVENANCES:
        bad.append(f"constants['frag_lo_provenance'] = {fl_prov!r} not in {FRAG_LO_PROVENANCES} (the lower edge is derived from "
                   "admitted training frames, or a recorded user_override after the derivation raised)")
    over_hi = _num("over_hi"); frag = {}
    for p in ("dayl", "dark"):
        v = _num(f"frag_lo[{p}]"); frag[p] = v
        if v is None: continue
        if fl_prov == "derived_P1_of_U" and not (FRAG_LO_SANITY[0] <= v <= FRAG_LO_SANITY[1]):
            bad.append(f"frag_lo[{p}]={v} (derived) outside the sanity band {FRAG_LO_SANITY}")
        elif fl_prov == "user_override":
            if not (0.0 < v <= FRAG_LO_SANITY[1]):
                bad.append(f"frag_lo[{p}]={v} (user_override) outside (0, {FRAG_LO_SANITY[1]}]: above {FRAG_LO_SANITY[1]} nothing was relaxed")
            else:
                notes.append(f"frag_lo[{p}]={v} is a user_override (the derivation raised in Tol_42); sanity band {FRAG_LO_SANITY} not applied")
        if over_hi is not None and v >= over_hi: bad.append(f"frag_lo[{p}]={v} >= over_hi={over_hi}: the band is empty")
    vbm = _num("valid_buffer_min")
    if vbm is not None and vbm < VALID_BUFFER_MIN_FLOOR:
        bad.append(f"valid_buffer_min={vbm} min < {VALID_BUFFER_MIN_FLOOR} (G12): near-duplicate frames may straddle the train/valid split")
    if "sha256" not in man["faults_json"]: bad.append("faults_json lacks sha256")
    if bad: raise RuntimeError("provenance violation(s) in the manifest: " + "; ".join(bad))
    return {"band_json_sha256": bj["sha256"], "faults_json_sha256": man["faults_json"]["sha256"],
            "band_json_derived_from": sorted(str(d) for d in bj["derived_from"]),
            "frag_lo": frag, "frag_lo_provenance": fl_prov, "over_hi": over_hi, "valid_buffer_min": vbm, "notes": notes}

def f9_control_check(label_source, td):
    """Teacher-detect drop check (decision 2026-09-23: no DET-C for now; the teacher_detect labels are a by-product that
    nothing downstream may require). When manifest.teacher_detect.drop_high is true a mechanism-control pair would have to train on
    identical frames (DET-R = tracked_intersection). That is RECORDED as a note, never a refusal: a `tracked` run must
    not depend on the teacher-detect stage. Returns the ⚠ notes."""
    notes = []; td = td or {}
    if not td.get("ran", False) or not td.get("drop_high", False): return notes
    frac = td.get("drop_frac")
    if label_source == "tracked":
        notes.append(f"teacher_detect.drop_high=true (drop_frac {frac}): this tracked DET-R cannot be the partner of a later mechanism control; "
                     "if DET-C is ever trained, pair it with label_source=tracked_intersection")
    elif label_source == "teacher_detect":
        notes.append(f"teacher_detect.drop_high=true (drop_frac {frac}): the paired DET-R must be trained with label_source=tracked_intersection")
    return notes

def frameset_identity(images):
    """The training frame-set identity Tol_44 can compare across run_tags: sorted full-frame file_names and image ids
    plus a sha256 of each list (crops are excluded — they inherit their parent's identity)."""
    full = [im for im in images if not im.get("crop_variant", False)]
    files = sorted(str(im["file_name"]) for im in full); ids = sorted(int(im["id"]) for im in full)
    return {"n_full": len(full), "file_names_sha256": sha256_bytes(json.dumps(files).encode("utf-8")),
            "image_ids_sha256": sha256_bytes(json.dumps(ids).encode("utf-8")), "image_ids": ids}

def refuse_overwrite(existing, force, run_tag, mode):
    """Provenance immutability: run_tag is the key Tol_44 uses to locate the evaluated checkpoint and its op_threshold,
    and it does not encode epochs/seed/batch/lr. A completed run under the same run_tag is never overwritten silently.
    Returns [] when nothing exists or force is set (the caller records what was overwritten); raises otherwise."""
    if not existing: return []
    if not force:
        raise RuntimeError(f"mode={mode} refused: run_tag {run_tag!r} already has a completed run on the Volume ({existing}); Tol_44 keys its "
                           "cached detections and op_threshold by run_tag. Use a new run_tag, or set force=true to overwrite it explicitly")
    return list(existing)

def resume_highwater_action(best_ema_in_ckpt, csv_best, csv_best_epoch, ema_epoch, ema_exists, eps=RESUME_EMA_EPS, ckpt_epoch=None):
    """How to treat BestModelCallback._best_ema inside a staged last.ckpt. Ordering (review 2026-09-23): rfdetr's `last`
    ModelCheckpoint (trainer.py:320-333) sets no save_on_train_epoch_end and validation runs every epoch
    (check_val_every_n_epoch = eval_interval = 1, val_check_interval 1.0), so PTL 2.x saves it in on_train_epoch_end —
    AFTER the epoch-end validation and after BestModelCallback.on_validation_end (best_model.py:500-574) raised _best_ema
    and wrote checkpoint_best_ema.pth. A last.ckpt of epoch e therefore already carries epoch e's mark. The one legal way
    to see mark < csv best is a snapshot taken in the IN-FLIGHT window of epoch e+1: validation of e+1 has written the EMA
    file and the metrics.csv row, but on_train_epoch_end has not yet rewritten last.ckpt (still epoch e), i.e.
    ckpt_epoch == csv_best_epoch - 1. A resumed run would then overwrite the true best with any later epoch scoring above
    the stale mark. Returns "ok" (marks agree) or "patch" (in-flight window: set every callbacks[*]._best_ema := csv_best
    before fit); raises when the triple cannot be made consistent — including mark < csv best while last.ckpt's epoch is
    NOT the one just before the csv best epoch (last.ckpt already covers the best epoch, or lags by more than one epoch)."""
    if not ema_exists:
        if (csv_best is not None and csv_best > 0.0) or (best_ema_in_ckpt is not None and best_ema_in_ckpt > 0.0):
            raise RuntimeError(f"mode=resume: checkpoint_best_ema.pth is absent on the Volume but the EMA metric already improved (metrics.csv best "
                               f"{csv_best} @ {csv_best_epoch}, last.ckpt mark {best_ema_in_ckpt}): the best checkpoint was lost with the cluster and "
                               "rfdetr would silently backfill the FINAL EMA weights as best (best_model.py:606-617); use mode=full")
        return "ok"
    if csv_best is None:
        raise RuntimeError("mode=resume: checkpoint_best_ema.pth exists but metrics.csv holds no validation row — the staged triple is inconsistent; use mode=full")
    if best_ema_in_ckpt is None:
        raise RuntimeError("mode=resume: last.ckpt carries no BestModelCallback state (_best_ema) — it is not an rfdetr training checkpoint; use mode=full")
    if ema_epoch is not None and csv_best_epoch is not None and int(ema_epoch) != int(csv_best_epoch):
        raise RuntimeError(f"mode=resume: checkpoint_best_ema.pth is epoch {ema_epoch} but the metrics.csv best is epoch {csv_best_epoch} "
                           f"({csv_best}) — the staged triple is inconsistent; use mode=full")
    if best_ema_in_ckpt > csv_best + eps:
        raise RuntimeError(f"mode=resume: last.ckpt's EMA high-water mark {best_ema_in_ckpt} exceeds the metrics.csv best {csv_best} — "
                           "metrics.csv is truncated relative to last.ckpt; use mode=full")
    if best_ema_in_ckpt >= csv_best - eps: return "ok"
    # mark < csv best: legal only in the in-flight window (last.ckpt still epoch csv_best_epoch - 1)
    if ckpt_epoch is None or csv_best_epoch is None or int(ckpt_epoch) != int(csv_best_epoch) - 1:
        raise RuntimeError(f"mode=resume: last.ckpt's EMA high-water mark {best_ema_in_ckpt} is below the metrics.csv best {csv_best} @ epoch "
                           f"{csv_best_epoch} but last.ckpt is epoch {ckpt_epoch}, not {None if csv_best_epoch is None else int(csv_best_epoch) - 1}: "
                           "PTL saves last.ckpt in on_train_epoch_end AFTER BestModelCallback raised the mark, so only a snapshot in the in-flight "
                           "window (last.ckpt one epoch behind the csv best epoch) can carry a lower mark — the staged triple is inconsistent; use mode=full")
    return "patch"

# ---- resume-state generations (review 2026-09-23: never overwrite the last verified state; one tick = one consistent set) ----
GEN_RX = re.compile(r"^g(\d{6})$")

def gen_name(n): return f"g{int(n):06d}"

def list_generations(root):
    """[(n, path, complete)] of every g<NNNNNN> directory under root, newest first; complete = GEN_COMPLETE exists."""
    if not osp.isdir(root): return []
    out = []
    for nm in os.listdir(root):
        m = GEN_RX.match(nm)
        if m: p = osp.join(root, nm); out.append((int(m.group(1)), p, osp.exists(osp.join(p, GEN_COMPLETE))))
    return sorted(out, key=lambda z: -z[0])

def snapshot_set(src_dir, stg_dir, consistency, files=RESUMABLE, required=("last.ckpt", "metrics.csv"),
                 tries=SNAPSHOT_SET_TRIES, wait_s=SNAPSHOT_SET_WAIT_S, sleep=time.sleep, snap=None):
    """Take ALL local snapshots back to back (seconds on /local_disk0), then require (a) no source changed while the set was
    taken and (b) consistency({fn: snapshot path or None}) not to raise — the SAME check mode=resume applies
    (resume_highwater_action). Retries up to `tries` with `wait_s` between; raises when no consistent set was seen (the
    tick is skipped, the previous generation stays authoritative). Returns ({fn: sig or None}, consistency result) or
    None when a required file does not exist yet."""
    snap = snap or stable_snapshot
    last_err = None
    for k in range(tries):
        if k: sleep(wait_s)
        sigs = {fn: snap(osp.join(src_dir, fn), osp.join(stg_dir, fn)) for fn in files}
        if any(sigs[fn] is None for fn in required): return None
        moved = [fn for fn in files if file_sig(osp.join(src_dir, fn)) != sigs[fn]]
        if moved: last_err = f"sources changed while the set was taken: {moved}"; continue
        paths = {fn: (osp.join(stg_dir, fn) if sigs[fn] is not None else None) for fn in files}
        try:
            return sigs, consistency(paths)
        except RuntimeError as e:
            last_err = f"inconsistent set: {e}"
    raise RuntimeError(f"no consistent snapshot set after {tries} attempts ({last_err}) — tick skipped; the previous generation stays")

class CopierError(RuntimeError):
    """The file copy inside publish_generation failed (dbutils.fs.cp or shutil). Only THIS error may switch the CkptSync daemon
    from dbutils.fs.cp to shutil (review 2026-09-23): an inconsistent snapshot set, an unstable last.ckpt or a prune failure is
    not a copier failure and must not demote every later tick to the ~30x slower FUSE copy."""

def publish_generation(stg_paths, shas, vol_root, n, record, copier=None, order=UPLOAD_ORDER):
    """Upload one snapshot set into a NEW directory vol_root/g<n>/ (never over an existing one): each file, then its sidecar,
    in `order`; GEN_COMPLETE last. Until GEN_COMPLETE exists the generation is invisible to mode=resume, so a cut upload
    can never replace the previous verified state. A failure of the copy itself is re-raised as CopierError (chained).
    Returns the generation path."""
    gdir = osp.join(vol_root, gen_name(n))
    if osp.exists(gdir): raise RuntimeError(f"generation {gdir} already exists — generations are never rewritten")
    os.makedirs(gdir)
    for fn in order:
        if stg_paths.get(fn) is None: continue
        try:
            (copier or shutil.copyfile)(stg_paths[fn], osp.join(gdir, fn))
        except Exception as e:
            raise CopierError(f"copy of {fn} into {gen_name(n)} failed via {'copier' if copier else 'shutil'}: {type(e).__name__}: {e}") from e
        write_sidecar(stg_paths[fn], osp.join(gdir, fn + ".sha256"), shas[fn])
    rec = dict(record); rec.update({"generation": n, "files": {fn: shas[fn] for fn in order if stg_paths.get(fn) is not None}})
    with open(osp.join(gdir, GEN_COMPLETE), "w") as f: json.dump(rec, f, indent=1)
    return gdir

def prune_generations(vol_root, keep=RESUME_KEEP_GENERATIONS, rm=shutil.rmtree):
    """Keep the newest `keep` COMPLETE generations; remove older ones and any incomplete directory older than the newest
    complete one (a cut upload). Never touches a directory newer than the newest complete generation. Returns removed names."""
    gens = list_generations(vol_root); comp = [g for g in gens if g[2]]
    if not comp: return []
    keep_n = {g[0] for g in comp[:keep]}; newest = comp[0][0]; removed = []
    for n, p, c in gens:
        if n in keep_n or n > newest: continue
        rm(p); removed.append(gen_name(n))
    return removed

def resume_mismatches(recorded_prov, current_prov, recorded_hp, current_hp,
                      prov_keys=RESUME_BOUND_PROV, hp_keys=RESUME_BOUND_HPARAMS):
    """[(key, recorded, current)] for every provenance / train-hparam key on which a staged resume state differs from THIS
    run (a missing recorded key counts as a mismatch). Non-empty => the state belongs to a different run: mode=full."""
    out = []
    for keys, rec, cur, pre in ((prov_keys, recorded_prov or {}, current_prov, ""), (hp_keys, recorded_hp or {}, current_hp, "train.")):
        for k in keys:
            if k not in rec or rec[k] != cur.get(k): out.append((pre + k, rec.get(k, "<absent>"), cur.get(k)))
    return out

def seal_terms_unmet(man, allow_below_floor):
    """Seal formula (as amended 2026-09-23): sealed = (not human_gate.required OR human_gate passed) AND no
    failed human gate AND not below_floor AND (sam3_veto.ran OR not require_veto) AND not smoke.
    Returns the unmet terms that are not waived. The ONLY waivable term is below_floor (allow_below_floor=true).
    sam3_veto.ran=false on an UNSEALED manifest is counted as unmet (non-waivable) unless the manifest itself records
    require_veto=false: the deliberate skip is Tol_42's require_veto=false, which seals."""
    unmet = []
    st = man["human_gate"].get("status")
    if human_gate_required(man) and st != "passed": unmet.append(f"human_gate.status={st!r} with human_gate.required=true (no waiver exists)")
    elif st == "failed": unmet.append("human_gate.status='failed' (a present-but-failed gate always unseals; no waiver exists)")
    if bool(man["counts"].get("below_floor", False)) and not allow_below_floor:
        unmet.append(f"counts.below_floor=true (train_full={man['counts'].get('train_full')}); allow_below_floor=true waives it")
    if not man["sam3_veto"].get("ran", False) and man.get("require_veto", True) is not False:
        unmet.append("sam3_veto.ran=false — G11 (the unlabelled-animal veto) never ran on these labels (no waiver exists here; "
                     "a deliberate skip is Tol_42 require_veto=false, which seals and is recorded)")
    if man.get("smoke"): unmet.append("smoke=true (no waiver exists)")
    return unmet

def manifest_sheet_n(man):
    """(value, kind) of the human-gate size the export recorded — human_gate.sheet_n_planned (preferred: the counted cells
    actually planned), human_gate.sheet_n, or a top-level widgets.sheet_n — or None when the manifest records none (the
    Tol_42 of 2026-09-23 writes only n_counted into manifest.human_gate)."""
    hg = man.get("human_gate") or {}
    if hg.get("sheet_n_planned") is not None: return int(hg["sheet_n_planned"]), "sheet_n_planned"
    if hg.get("sheet_n") is not None: return int(hg["sheet_n"]), "sheet_n"
    w = man.get("widgets") or {}
    return (int(w["sheet_n"]), "sheet_n") if w.get("sheet_n") is not None else None

def export_balance_check(man):
    """Decorrelation + quota outcome of the export, which Tol_42 prints but does not make a seal term
    (review 2026-09-23): counts.hour_share_max / hour_cap_exceeded_after_drops (the 8 % (day, hour) cap re-checked after
    the human-gate and G11 drops), counts.distinct_10min_blocks, quotas.{targets, achieved, shortfall}. Not a refusal —
    the export sealed — but every violation becomes a ⚠ note for silent_fallbacks and the record goes to the smoke JSON and
    train_summary.json. Raises only when the required keys (counts.distinct_10min_blocks, quotas.achieved/shortfall) are
    absent. Returns (record, notes)."""
    c = man.get("counts") or {}; q = man.get("quotas"); notes = []
    if "distinct_10min_blocks" not in c:
        raise RuntimeError("manifest.counts lacks distinct_10min_blocks — the decorrelation of the training set cannot be recorded")
    if not isinstance(q, dict) or "achieved" not in q or "shortfall" not in q:
        raise RuntimeError(f"manifest.quotas = {q!r} lacks achieved/shortfall — the night/dusk/dawn/active quota outcome cannot be recorded")
    cap = ((man.get("constants") or {}).get("hour_cap") or {}).get("value") if isinstance(man.get("constants"), dict) else None
    share, exceeded = c.get("hour_share_max"), c.get("hour_cap_exceeded_after_drops")
    if exceeded is None and share is not None and cap is not None: exceeded = float(share) > float(cap) + 1e-12
    if exceeded is True:
        notes.append(f"HOUR CAP exceeded in the FINAL train set: max (day, hour) share {share} > hour_cap {cap} after the human/G11 drops "
                     "(not a Tol_42 seal term) — one hour is over-represented in training")
    elif exceeded is None:
        notes.append("the export records neither counts.hour_share_max nor hour_cap_exceeded_after_drops: the hour cap cannot be re-checked")
    unmet = {}
    for k, v in (q.get("shortfall") or {}).items():
        if isinstance(v, dict): short = (v.get("need") is not None and v.get("have") is not None and float(v["have"]) < float(v["need"]))
        else:
            try: short = float(v) > 0
            except (TypeError, ValueError): raise RuntimeError(f"manifest.quotas.shortfall[{k!r}] = {v!r} is neither a {{have, need}} dict nor a number")
        if short: unmet[k] = v
    if unmet:
        notes.append(f"QUOTA UNMET in the FINAL train set: {unmet} (targets {q.get('targets')}, achieved {q.get('achieved')})")
    rec = {"hour_share_max": share, "hour_share_max_at_gate": c.get("hour_share_max_at_gate"), "hour_cap": cap, "hour_cap_exceeded_after_drops": exceeded,
           "distinct_10min_blocks": c.get("distinct_10min_blocks"), "quota_targets": q.get("targets"), "quota_achieved": q.get("achieved"),
           "quota_shortfall": q.get("shortfall"), "quota_unmet": unmet}
    return rec, notes

def extra_day_strides_check(man, allow_undensified_extra_day=False):
    """Decision 2026-09-23: when the export falls short, the FIRST step is to densify the training
    lattice to stride_day_s=8 / stride_other_s=4 (DENSE_STRIDES); adding 20240422 (which then leaves the detector test set)
    is only the second option. So a manifest whose source_days contain 20240422 must record constants['strides'] equal to
    DENSE_STRIDES, unless the widget allow_undensified_extra_day=true explicitly skips the order (returned as a ⚠ note for
    silent_fallbacks). Raises when the strides are absent/unreadable or undensified without the widget. Returns
    (strides dict or None, notes); (None/strides, []) when 20240422 is not a source day."""
    days = {str(d) for d in man.get("source_days", [])}
    st = ((man.get("constants") or {}).get("strides") or {}).get("value") if isinstance(man.get("constants"), dict) else None
    if TEST_DAY_SOFT not in days: return st, []
    try: pair = (float(st["day"]), float(st["other"]))
    except (TypeError, KeyError, ValueError):
        raise RuntimeError(f"manifest.source_days contains {TEST_DAY_SOFT} but constants['strides'].value = {st!r} does not record the training "
                           "strides — cannot prove the densify-first order")
    if pair == DENSE_STRIDES: return st, []
    msg = (f"manifest.source_days contains {TEST_DAY_SOFT} but the export used stride_day_s={pair[0]:g}, stride_other_s={pair[1]:g}: the protocol is "
           f"to DENSIFY FIRST (stride_day_s={DENSE_STRIDES[0]:g}, stride_other_s={DENSE_STRIDES[1]:g}) and to add 04-22 only if that is still short "
           "(decision 2026-09-23)")
    if not allow_undensified_extra_day:
        raise RuntimeError(msg + " — re-export with the dense strides, or set allow_undensified_extra_day=true to skip that order explicitly (recorded)")
    return st, [msg + "; skipped explicitly with allow_undensified_extra_day=true"]

def manifest_guards(man, label_source, allow_extra_days, allow_below_floor, allow_undensified_extra_day=False):
    """Every refusal raises RuntimeError with the offending value. Returns the source-day list."""
    req = ["tag", "sources", "source_days", "forbidden_eval_days", "sealed", "counts", "human_gate",
           "label_sources_available", "sha256", "sam3_veto", "smoke",
           "constants", "band_json", "faults_json", "split", "gates"]         # every key required
    miss = [k for k in req if k not in man]
    if miss: raise RuntimeError(f"manifest lacks required keys {miss}")
    days = [str(d) for d in man["source_days"]]
    if not days: raise RuntimeError("manifest.source_days is empty")
    hard = sorted(set(days) & TEST_DAYS_HARD)
    if hard: raise RuntimeError(f"manifest.source_days contains detector TEST days {hard} (the detector test days never train) — refused, no override exists")
    unknown = sorted(set(days) - (TRAIN_DAYS | {TEST_DAY_SOFT}))
    if unknown: raise RuntimeError(f"manifest.source_days {days} contains days outside TRAIN_DAYS ∪ {{{TEST_DAY_SOFT}}}: {unknown} — refused, no widget admits them")
    if TEST_DAY_SOFT in days:
        if not allow_extra_days:
            raise RuntimeError(f"manifest.source_days contains {TEST_DAY_SOFT}; set allow_extra_days=true to accept that 04-22 "
                               "leaves the detector test set")
        if not str(man.get("allow_test_day_reason", "") or "").strip():
            raise RuntimeError(f"manifest.source_days contains {TEST_DAY_SOFT} but manifest.allow_test_day_reason is empty "
                               "(04-22 leaves the test set only for a named reason)")
        extra_day_strides_check(man, allow_undensified_extra_day)                              # densify before 04-22
    if man.get("smoke"): raise RuntimeError("manifest.smoke is true — a smoke export is never trained on")
    if human_gate_required(man): human_gate_check(man["human_gate"], sheet_n=manifest_sheet_n(man))   # full seal rule
    else: human_gate_optional_check(man["human_gate"])                                                 # only 'failed' refuses
    if not isinstance(man["sealed"], bool):
        raise RuntimeError(f"manifest.sealed = {man['sealed']!r} is not a bool — cannot tell whether the export sealed")
    gates_check(man["gates"])                                                                           # G9 block collapse ran
    if man["sealed"]:
        # the seal is re-derived, never trusted (review 2026-09-23): a sealed manifest with an unmet term is inconsistent
        inconsistent = seal_terms_unmet(man, allow_below_floor=False)
        if inconsistent:
            raise RuntimeError("manifest.sealed is true but these seal terms are unmet: " + " | ".join(inconsistent)
                               + " — the manifest is inconsistent (hand-edited or a buggy export); refused, re-run Tol_42 finalize")
    else:
        reasons = seal_terms_unmet(man, allow_below_floor=False)
        if not reasons:
            raise RuntimeError("manifest.sealed is false although every seal term is met — the exporter refused to seal for a "
                               "reason this notebook cannot see; refused")
        unmet = seal_terms_unmet(man, allow_below_floor)
        if unmet:
            raise RuntimeError("manifest.sealed is false and these seal terms are unmet and not waived: " + " | ".join(unmet)
                               + " (an unsealed manifest trains only when below_floor is the sole reason and allow_below_floor=true; "
                               "otherwise re-export: densify first, then add 04-22)")
    constants_provenance_check(man, allow_extra_days)
    if label_source not in man["label_sources_available"]:
        raise RuntimeError(f"label_source {label_source!r} not in manifest.label_sources_available {man['label_sources_available']}")
    if label_source in ("teacher_detect", "tracked_intersection") and not man.get("teacher_detect", {}).get("ran", False):
        raise RuntimeError(f"label_source {label_source!r} needs manifest.teacher_detect.ran == true")
    if sorted(man["forbidden_eval_days"]) != sorted(days):
        raise RuntimeError(f"manifest.forbidden_eval_days {man['forbidden_eval_days']} != source_days {days}")
    return days

def clip_split_check(train_images, valid_images, split):
    """Train and valid are WHOLE-CLIP disjoint and equal to manifest.split. Every image carries
    clip_stem (crops inherit the parent's). Raises on a leak or a mismatch; returns (train_clips, valid_clips) sorted."""
    for nm, ims in (("train", train_images), ("valid", valid_images)):
        nocs = [im.get("file_name") for im in ims if not im.get("clip_stem")]
        if nocs: raise RuntimeError(f"{len(nocs)} {nm} images lack clip_stem, e.g. {nocs[:3]}")
    tr = {im["clip_stem"] for im in train_images}; va = {im["clip_stem"] for im in valid_images}
    leak = sorted(tr & va)
    if leak: raise RuntimeError(f"SPLIT LEAK: {len(leak)} clip(s) appear in BOTH train and valid: {leak[:5]} — the valid mAP would be leaked")
    mv, mt = set(split.get("valid_clips", [])), set(split.get("train_clips", []))
    if not mv: raise RuntimeError("manifest.split.valid_clips is empty")
    if mt & mv: raise RuntimeError(f"manifest.split lists clips in both train_clips and valid_clips: {sorted(mt & mv)[:5]}")
    if not va <= mv: raise RuntimeError(f"valid images from clips not in manifest.split.valid_clips: {sorted(va - mv)[:5]}")
    if not tr <= mt: raise RuntimeError(f"train clips not in manifest.split.train_clips: {sorted(tr - mt)[:5]}")
    if not va: raise RuntimeError("the valid split has no image at all")
    return sorted(tr), sorted(va), sorted(mv - va)

def gates_check(gates):
    """G9: the export must prove its block-level night-collapse filter ran — every GATES_REQUIRED key present
    with the right type, and the funnel ({day: {phase4: {gate: [in, out]}}}) carries a G9 entry somewhere. Raises; returns
    the record train_summary.json copies (collapse_overrides, curled/clipped count, blocks failing the per-slot 0.75 rule)."""
    if not isinstance(gates, dict): raise RuntimeError(f"manifest.gates is {type(gates).__name__}, not an object")
    miss = [k for k in GATES_REQUIRED if k not in gates]
    if miss: raise RuntimeError(f"manifest.gates lacks {miss}: the G9 block-collapse test is not proven to have run")
    if not isinstance(gates["curled_or_clipped_blocks"], list): raise RuntimeError("manifest.gates.curled_or_clipped_blocks is not a list")
    if not isinstance(gates["collapse_overrides"], list): raise RuntimeError("manifest.gates.collapse_overrides is not a list")
    bf = gates["blocks_failing_per_slot_0.75"]
    if isinstance(bf, bool) or not isinstance(bf, int) or bf < 0: raise RuntimeError(f"manifest.gates['blocks_failing_per_slot_0.75'] = {bf!r} is not a count")
    fun = gates["funnel"]
    has_g9 = isinstance(fun, dict) and any(isinstance(gs, dict) and "G9" in gs
                                           for ps in fun.values() if isinstance(ps, dict) for gs in ps.values())
    if not has_g9: raise RuntimeError("manifest.gates.funnel has no G9 entry for any day x phase4: the block-collapse gate never ran")
    return {"collapse_overrides": list(gates["collapse_overrides"]), "n_curled_or_clipped_blocks": len(gates["curled_or_clipped_blocks"]),
            "blocks_failing_per_slot_0.75": bf}

def valid_phase_check(valid_images, night_min=VALID_NIGHT_MIN, allow_thin=False):
    """Valid holds two night clips because checkpoint selection and op_threshold must see night (the Tol_44 evaluation judges the
    dark stratum). G9/G11 can empty a clip's night block. Returns (per-phase4 counts incl. 'dark' = dusk+night, notes); raises when
    night < night_min unless allow_thin (then the shortfall is a note for silent_fallbacks)."""
    c = Counter(im.get("phase4") for im in valid_images)
    counts = {ph: int(c.get(ph, 0)) for ph in PHASES4}; counts["dark"] = counts["dusk"] + counts["night"]
    notes = []
    if counts["night"] < night_min:
        msg = (f"valid holds {counts['night']} night images < {night_min} (counts {counts}): checkpoint selection and op_threshold "
               "would be tuned (almost) without night")
        if not allow_thin:
            raise RuntimeError(msg + " — re-export with other valid clips, or set allow_thin_night_valid=true to accept it (recorded)")
        notes.append(msg + " — accepted by allow_thin_night_valid=true")
    return counts, notes

def _wall(im):
    w = im.get("wall")
    if w is None: raise RuntimeError(f"image {im.get('file_name')} lacks 'wall' (audit key)")
    return w if isinstance(w, datetime) else datetime.fromisoformat(str(w))

def valid_buffer_check(train_images, valid_images, buffer_min):
    """G12 re-checked on the images themselves: min over same-day pairs of |wall_train - wall_valid| must be >= buffer_min
    minutes. G12 drops train rows within buffer_min of a valid clip's wall BOUNDS, and every valid frame lies inside them, so this is
    implied by G12 (never a false alarm) though weaker than it. Raises; returns the min gap in minutes (None when no same-day pair)."""
    import bisect
    vw = defaultdict(list)
    for im in valid_images: vw[str(im.get("day"))].append(_wall(im))
    for d in vw: vw[d].sort()
    best = None; worst = None
    for im in train_images:
        ws = vw.get(str(im.get("day")))
        if not ws: continue
        t = _wall(im); i = bisect.bisect_left(ws, t)
        gap = min(abs((ws[j] - t).total_seconds()) for j in (i - 1, i) if 0 <= j < len(ws)) / 60.0
        if best is None or gap < best: best, worst = gap, im.get("file_name")
    if best is not None and best < float(buffer_min):
        raise RuntimeError(f"train image {worst} is {best:.2f} min from a same-day valid frame < valid_buffer_min {buffer_min} (G12): "
                           "near-duplicates straddle the split")
    return best

def image_provenance_check(images, source_days, sources, split_name):
    """Test-day exclusion (and the allow_extra_days rule for 04-22) on the images actually trained on, not only the manifest's self-declared source_days
    (audit keys day/root/clip_stem, crops included): every image's day is a source day and never 04-23..28;
    its clip_stem starts on that day (recording stops overnight, so a clip never starts on the previous date); its root is
    a manifest source root for that day; when the manifest lists the source's clips, the clip is one of them. Raises
    naming the offending file_names; returns {day: n_images}."""
    sd = {str(d) for d in source_days}
    roots = defaultdict(set); clips = defaultdict(set); has_clips = set()
    for s in sources or []:
        d = str(s.get("day")); roots[d].add(str(s.get("root")))
        if s.get("clips") is not None: has_clips.add((str(s.get("root")), d)); clips[(str(s.get("root")), d)].update(s["clips"])
    bad = defaultdict(list); per_day = Counter()
    for im in images:
        fn = im.get("file_name"); d = str(im.get("day", "")); r = str(im.get("root", "")); cs = im.get("clip_stem")
        if not d or d == "None": bad["no day"].append(fn); continue
        per_day[d] += 1
        if d in TEST_DAYS_HARD: bad[f"test day {d} (F1)"].append(fn)
        if d not in sd: bad[f"day {d} not in source_days {sorted(sd)}"].append(fn)
        try:
            cd = clip_start(cs).strftime("%Y%m%d")
            if cd != d: bad[f"clip {cs} starts on {cd}, image says day {d}"].append(fn)
        except Exception:
            bad["unparseable clip_stem"].append(fn)
        if r not in roots.get(d, set()): bad[f"root {r!r} is not a manifest source root for {d} ({sorted(roots.get(d, set()))})"].append(fn)
        elif (r, d) in has_clips and cs not in clips[(r, d)]: bad[f"clip {cs} not among manifest.sources[{r},{d}].clips"].append(fn)
    if bad:
        raise RuntimeError(f"{split_name}: image provenance violates the manifest: " +
                           "; ".join(f"{k}: {len(v)} e.g. {v[:3]}" for k, v in sorted(bad.items())))
    return dict(sorted(per_day.items()))

# ---- full/resume: the smoke record must describe THIS run ---------------------------------------
SMOKE_BOUND_KEYS = ("variant", "resolution", "N_train", "N_valid", "batch_size", "grad_accum", "scale_jitter",
                    "eval_masks_head_resolution")     # ips / t_val_img were measured under exactly these

def smoke_mismatches(smk, current):
    """[(key, smoke_value, current_value)] for every SMOKE_BOUND_KEYS whose smoke value differs from the current
    widgets (a missing smoke key counts as a mismatch: the projection did not record it)."""
    out = []
    for k in SMOKE_BOUND_KEYS:
        if k not in smk or smk[k] != current[k]: out.append((k, smk.get(k, "<absent>"), current[k]))
    return out

def reproject_hours(smk, N_train, N_valid, epochs, overhead=PROJ_OVERHEAD):
    """The projection formula from the smoke's measured ips / t_val_img and the CURRENT epochs and counts."""
    return epochs * (N_train / float(smk["ips"]) + N_valid * float(smk["t_val_img"])) * overhead / 3600.0

# ---- COCO helpers --------------------------------------------------------------------------------
def filter_coco_by_ids(coco, keep_ids):
    """tracked_intersection: keep full-frame images whose id is in keep_ids plus crop-variant images whose
    parent (by parent_file_name -> full-frame id) is kept; ids are NOT renumbered (ids kept consistent)."""
    keep_ids = set(int(i) for i in keep_ids)
    fn2id = {im["file_name"]: im["id"] for im in coco["images"] if not im.get("crop_variant", False)}
    kept_img = []
    for im in coco["images"]:
        if im.get("crop_variant", False):
            pid = fn2id.get(im.get("parent_file_name"))
            if pid is not None and pid in keep_ids: kept_img.append(im)
        elif im["id"] in keep_ids:
            kept_img.append(im)
    kid = {im["id"] for im in kept_img}
    out = dict(coco)
    out["images"] = kept_img
    out["annotations"] = [a for a in coco["annotations"] if a["image_id"] in kid]
    return out

def coco_seg_to_mask(seg):
    """COCO RLE -> bool [H,W]. list counts = the project's own column-major zero-first RLE (identical to COCO
    uncompressed RLE) -> rle_decode; str/bytes counts = compressed -> pycocotools."""
    if isinstance(seg["counts"], list):
        return rle_decode(seg)
    from pycocotools import mask as maskUtils
    s = dict(seg)
    if isinstance(s["counts"], str): s["counts"] = s["counts"].encode("ascii")
    return maskUtils.decode(s).astype(bool)

def stratified_subset(images, n, seed, key="phase4"):
    """Seeded, stratified by phase4 with largest-remainder allocation; keeps phase shares."""
    if n >= len(images): return list(images)
    rng = random.Random(seed)
    by = defaultdict(list)
    for im in images: by[im.get(key, "unknown")].append(im)
    tot = len(images)
    quota = {k: n * len(v) / tot for k, v in by.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    rem = n - sum(alloc.values())
    for k in sorted(quota, key=lambda k: -(quota[k] - alloc[k]))[:rem]: alloc[k] += 1
    out = []
    for k in sorted(by):
        lst = sorted(by[k], key=lambda im: im["id"]); rng.shuffle(lst)
        out.extend(lst[:alloc[k]])
    return out

def phase_shares(images, key="phase4"):
    c = Counter(im.get(key, "unknown") for im in images); t = max(1, len(images))
    return {k: round(v / t, 4) for k, v in sorted(c.items())}

# ---- the post-chain every arm passes (mirrors Tol_03.detect():463-472 on decoded masks) --
def post_chain(masks, scores, topk=TOPK_DET):
    """masks: list of bool [H,W] already class/score-gated; returns [(mask, score)] largest first, greedy
    keep if max(IoS, IoU) < 0.50 against every kept, top-k. Byte-for-byte the filter/dedupe of Tol_03:463-472."""
    cand = []
    for m, s in zip(masks, scores):
        mm = np.asarray(m).astype(bool)
        if mm.sum() >= MIN_MASK_PX: cand.append((mm, float(s)))
    cand.sort(key=lambda z: -z[0].sum())
    keep = []
    for m, s in cand:
        if all(m_ios(m, k) < DEDUPE_IOS and m_iou(m, k) < 0.5 for k, _ in keep):
            keep.append((m, s))
    return keep[:topk] if topk else keep

def hungarian_counts(kept_masks, label_masks, iou_min=MATCH_IOU_MIN):
    """(TP, FP, FN, mean matched IoU) — Hungarian on mask IoU, a pair counts at IoU >= iou_min."""
    from scipy.optimize import linear_sum_assignment
    nk, nl = len(kept_masks), len(label_masks)
    if nk == 0 or nl == 0: return 0, nk, nl, 0.0
    M = np.zeros((nk, nl))
    for i, k in enumerate(kept_masks):
        for j, l in enumerate(label_masks): M[i, j] = m_iou(k, l)
    ri, cj = linear_sum_assignment(-M)
    ious = [float(M[i, j]) for i, j in zip(ri, cj)]
    tp = sum(1 for v in ious if v >= iou_min)
    return tp, nk - tp, nl - tp, (float(np.mean(ious)) if ious else 0.0)

def f1_of(tp, fp, fn):
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0

def op_threshold_from_sweep(acc, taus=SWEEP_TAUS):
    """acc: {tau: (tp, fp, fn)}. argmax F1, ties -> the LOWER tau (recall is what the night
    gate geometry loses; the dedupe carries the FP load). Returns (tau, {tau: f1})."""
    f1 = {t: f1_of(*acc[t]) for t in taus}
    best = max(f1.values())
    tau = min(t for t in taus if abs(f1[t] - best) < 1e-12)
    return tau, f1

def apply_threshold_and_chain(rows, tau, topk=TOPK_DET):
    """rows: [(mask, score)] from one predict() at SWEEP_PROBE_THR with class_id<num_classes already applied;
    keep score > tau (strict, mirrors postprocess.py:219-225) then the post-chain."""
    sel = [(m, s) for m, s in rows if s > tau]
    return post_chain([m for m, _ in sel], [s for _, s in sel], topk=topk)

def sweep_accumulate(acc, rows, label_masks, taus=SWEEP_TAUS):
    for t in taus:
        kept = apply_threshold_and_chain(rows, t)
        tp, fp, fn, _ = hungarian_counts([m for m, _ in kept], label_masks)
        a = acc.setdefault(t, [0, 0, 0]); a[0] += tp; a[1] += fp; a[2] += fn
    return acc

# ---- the contract Tol_44 (arm `rfdetr`) reads from train_summary.json ---------------------------
EVAL_V2_KEYS = ("run_tag", "variant", "resolution", "op_threshold", "post_chain", "reload",
                "checkpoint_best_total_sha256", "source_days", "forbidden_eval_days")   # what Tol_44 reads

def eval_v2_contract(op_threshold, resolution, variant, source_days, forbidden_eval_days, taus=SWEEP_TAUS):
    """The block Tol_44 needs to run the rfdetr arm exactly as the threshold was chosen here: predict(threshold=op_threshold)
    keeps score > τ (strict, postprocess.py:219-225 / detr.py:2714), drop class_id >= num_classes (detr.py:2757-2772),
    then the post-chain (MIN_MASK_PX, area-descending, greedy max(IoS, IoU) < 0.50, top-6). Resolution must be
    PINNED at reload: strip_checkpoint (state_dict.py:184-205) drops model_config from checkpoint_best_total.pth, so an
    unpinned from_checkpoint builds the class default (SegSmall 384). Raises on a threshold outside the swept grid."""
    if op_threshold not in taus: raise RuntimeError(f"op_threshold {op_threshold!r} is not one of the swept thresholds {taus}")
    if resolution % SEG_BLOCK: raise RuntimeError(f"resolution {resolution} is not a multiple of {SEG_BLOCK}")
    if sorted(forbidden_eval_days) != sorted(source_days):
        raise RuntimeError(f"forbidden_eval_days {forbidden_eval_days} != source_days {source_days} (they must be equal)")
    return {"op_threshold": op_threshold, "variant": variant, "resolution": int(resolution),
            "source_days": list(source_days), "forbidden_eval_days": list(forbidden_eval_days),
            "post_chain": {"threshold_rule": "predict(frame_rgb_uint8, threshold=op_threshold) keeps score > op_threshold (strict)",
                           "drop_class_id_ge_num_classes": True, "min_mask_px": MIN_MASK_PX, "order": "area-descending",
                           "dedupe": f"greedy keep iff max(IoS, IoU) < {DEDUPE_IOS} against every kept mask", "topk": TOPK_DET,
                           "source": "SPEC §1.4 = Tol_03_track_v4 detect():463-472 filter/dedupe"},
            "reload": {"call": f"rfdetr.from_checkpoint(<Detector_Checkpoints/run_tag>/checkpoint_best_total.pth, resolution={int(resolution)})",
                       "why_resolution": "strip_checkpoint drops model_config (state_dict.py:184-205); unpinned => class default resolution",
                       "assert": "model_config.num_classes == 1, class_names == ['calf'], sha256(file) == checkpoint_best_total_sha256"}}

def train_summary_contract_check(out, taus=SWEEP_TAUS):
    """Raise unless `out` (the train_summary.json dict, before it is written) carries everything Tol_44 reads: every
    EVAL_V2_KEYS and TOL44_TS_KEYS key present and non-None, mode in TOL44_TS_MODES, no truthy `smoke`, op_threshold a
    number from the swept grid (never re-tuned by Tol_44), source_days == forbidden_eval_days as sets, resolution a
    positive multiple of SEG_BLOCK. Returns the sorted key list checked."""
    keys = sorted(set(EVAL_V2_KEYS) | set(TOL44_TS_KEYS))
    miss = [k for k in keys if k not in out or out[k] is None]
    if miss: raise RuntimeError(f"{SUMMARY_FILE} would lack the keys Tol_44 reads: {miss}")
    if out["mode"] not in TOL44_TS_MODES: raise RuntimeError(f"{SUMMARY_FILE} mode={out['mode']!r}: Tol_44 accepts only {TOL44_TS_MODES}")
    if out.get("smoke"): raise RuntimeError(f"{SUMMARY_FILE} has smoke={out['smoke']!r}: Tol_44 refuses a smoke run")
    op = out["op_threshold"]
    if isinstance(op, bool) or not isinstance(op, (int, float)) or op not in taus:
        raise RuntimeError(f"{SUMMARY_FILE} op_threshold={op!r} is not a number from the swept grid {taus}")
    r = out["resolution"]
    if isinstance(r, bool) or not isinstance(r, int) or r <= 0 or r % SEG_BLOCK:
        raise RuntimeError(f"{SUMMARY_FILE} resolution={r!r} is not a positive multiple of {SEG_BLOCK}")
    if sorted(map(str, out["source_days"])) != sorted(map(str, out["forbidden_eval_days"])):
        raise RuntimeError(f"{SUMMARY_FILE} source_days {out['source_days']} != forbidden_eval_days {out['forbidden_eval_days']}")
    return keys

def resolve_label_source(raw):
    """Decision 2026-09-23: label_source defaults to `tracked` (the only run planned); an empty widget means the
    default. `teacher_detect` / `tracked_intersection` stay legal only as the explicitly requested (unplanned) mechanism-control arm and
    return a ⚠ note. Anything else raises. Returns (label_source, notes)."""
    ls = (raw or "").strip().lower() or LABEL_SOURCE_DEFAULT
    if ls not in LABEL_SOURCES: raise RuntimeError(f"label_source must be one of {'|'.join(LABEL_SOURCES)}, got {raw!r}")
    notes = [] if ls == LABEL_SOURCE_DEFAULT else [
        f"label_source={ls}: the mechanism-control arm — NOT in the current plan (decision 2026-09-23: label_source="
        f"{LABEL_SOURCE_DEFAULT} is the only run planned). Proceeding because the widget asked for it explicitly."]
    return ls, notes

def read_metrics_csv(path):
    """PTL CSVLogger metrics.csv -> list of dict rows (strings); '' for missing cells."""
    if not osp.exists(path): return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def summarize_metrics_csv(rows, key="val/segm_mAP_50_95", key50="val/segm_mAP_50"):
    """epochs_run = max(epoch)+1 among rows carrying the key; best value/epoch of `key` (the primary
    namespace mirrors the EMA track under the defaults, config.py:1079-1088; if only the explicit
    val/ema_* track is present — coco_eval.py:643-646 — it is used instead, and that is printed)."""
    if rows and not any(r.get(key, "") not in ("", None) for r in rows):
        ek, ek50 = key.replace("val/", "val/ema_"), key50.replace("val/", "val/ema_")
        if any(r.get(ek, "") not in ("", None) for r in rows):
            print(f"⚠ metrics.csv has no {key}; reading the EMA track {ek}"); key, key50 = ek, ek50
    best, best_ep, best50, epochs = -1.0, None, None, set()
    for r in rows:
        v = r.get(key, "")
        if v in ("", None): continue
        try: v = float(v); ep = int(float(r.get("epoch", "nan")))
        except ValueError: continue
        epochs.add(ep)
        if v > best: best, best_ep, best50 = v, ep, (float(r[key50]) if r.get(key50, "") not in ("", None) else None)
    return dict(epochs_run=(max(epochs) + 1) if epochs else 0, best_epoch=best_ep,
                best_val_segm_mAP_50_95=(best if best >= 0 else None), best_val_segm_mAP_50=best50)

def luminance_stretch(rgb_u8):
    """Display-only 1-99 percentile stretch for night/dusk/dawn panels. NEVER applied to training data."""
    a = rgb_u8.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 1: return rgb_u8
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

# COMMAND ----------

# MAGIC %md ## Widgets — REQUIRED ones (`dataset_tag`, `mode`) raise when empty (nothing defaults a data choice); `label_source` defaults to `tracked` (decision 2026-09-23)

# COMMAND ----------

if not SELFTEST_ONLY:
    DATASET_TAG   = P("dataset_tag", "").strip()
    LABEL_SOURCE  = P("label_source", LABEL_SOURCE_DEFAULT)      # decision 2026-09-23: defaults to tracked (not a day list, so the no-default rule does not apply)
    MODE          = P("mode", "").strip().lower()
    VARIANT       = (P("variant", "seg-small") or "seg-small").strip().lower()
    RES           = int(P("resolution", "624") or 624)
    EPOCHS        = int(P("epochs", "40") or 40)
    BS            = int(P("batch_size", "8") or 8)
    GA            = int(P("grad_accum", "2") or 2)
    MAX_HOURS     = float(P("max_hours", "20") or 20)
    SMOKE_TRAIN_N = int(P("smoke_train_n", "1000") or 1000)
    SMOKE_VALID_N = int(P("smoke_valid_n", "200") or 200)
    SCALE_JITTER  = (P("scale_jitter", "false") or "false").strip().lower() == "true"
    EVAL_HEAD_RES = (P("eval_masks_head_resolution", "false") or "false").strip().lower() == "true"
    ALLOW_EXTRA_DAYS  = (P("allow_extra_days", "false") or "false").strip().lower() == "true"
    ALLOW_BELOW_FLOOR = (P("allow_below_floor", "false") or "false").strip().lower() == "true"
    ALLOW_UNDENSIFIED_EXTRA_DAY = (P("allow_undensified_extra_day", "false") or "false").strip().lower() == "true"   # densify-first order skip (recorded)
    FORCE         = (P("force", "false") or "false").strip().lower() == "true"     # overwrite a COMPLETED run under the same run_tag (recorded in silent_fallbacks)
    ALLOW_THIN_NIGHT_VALID = (P("allow_thin_night_valid", "false") or "false").strip().lower() == "true"   # accept < VALID_NIGHT_MIN night valid images (recorded)
    RUN_TAG       = (P("run_tag", "") or "").strip()
    CKPT_SYNC_MIN = float(P("ckpt_sync_min", "30") or 30)
    SEED          = int(P("seed", "0") or 0)

    for nm, val in (("dataset_tag", DATASET_TAG), ("mode", MODE)):
        if not val: raise RuntimeError(f"widget {nm} is REQUIRED and is empty (nothing defaults a data choice)")
    LABEL_SOURCE, _ls_notes = resolve_label_source(LABEL_SOURCE)   # decision 2026-09-23: default tracked; the other two are the unplanned mechanism-control arm
    for note in _ls_notes: print(f"⚠ {note}")
    if MODE not in ("smoke", "full", "resume"):
        raise RuntimeError(f"mode must be smoke|full|resume, got {MODE!r}")
    if VARIANT not in ("seg-small", "seg-xlarge"):
        raise RuntimeError(f"variant must be seg-small|seg-xlarge, got {VARIANT!r}")
    if RES % SEG_BLOCK != 0 or RES <= 0:
        raise RuntimeError(f"resolution {RES} is not a positive multiple of {SEG_BLOCK} (patch 12 x windows 2); the constructor "
                           "would build it and train()/predict() would raise later (detr.py:388-395)")
    if BS < 1 or GA < 1: raise RuntimeError(f"batch_size {BS} / grad_accum {GA} must be >= 1")
    if not RUN_TAG: RUN_TAG = f"{DATASET_TAG}_{LABEL_SOURCE}_{VARIANT}_r{RES}"
    if not re.match(r"^[A-Za-z0-9_.-]+$", RUN_TAG): raise RuntimeError(f"run_tag {RUN_TAG!r} has unsafe characters")

    DS_VOL   = osp.join(TOL, "Detector_COCO", DATASET_TAG)
    CK_VOL   = osp.join(TOL, "Detector_Checkpoints", RUN_TAG)
    PRE_VOL  = osp.join(TOL, "Detector_Checkpoints", "pretrained")
    FIG_VOL  = osp.join(TOL, "Figures", "detector_train", RUN_TAG)
    MET_SMOKE = osp.join(TOL, f"Metrics_rfdetr_train_{RUN_TAG}_smoke.json")
    MET_FULL  = osp.join(TOL, f"Metrics_rfdetr_train_{RUN_TAG}.json")
    LOCAL    = "/local_disk0/tmp/n43"                       # removed at start (warm clusters keep stale files)
    DS_ROOT  = osp.join("/local_disk0/rfdetr_ds", DATASET_TAG)
    SMOKE_DIR = osp.join("/local_disk0/rfdetr_ds_smoke", RUN_TAG)
    OUT_DIR  = osp.join("/local_disk0/rfdetr_out", RUN_TAG)  # never the Volume: PTL overwrites last.ckpt, CSVLogger appends
    RF_HOME  = "/local_disk0/rf_home"
    os.environ["RF_HOME"] = RF_HOME                          # assets/model_weights.py:275-316 — BEFORE `import rfdetr`
    PRETRAIN_FILE = {"seg-small": "rf-detr-seg-small.pt", "seg-xlarge": "rf-detr-seg-xlarge.pt"}[VARIANT]  # registry names, model_weights.py:246-268
    shutil.rmtree(LOCAL, ignore_errors=True)
    for d in (LOCAL, RF_HOME, CK_VOL, FIG_VOL): os.makedirs(d, exist_ok=True)
    SILENT_FALLBACKS = list(_ls_notes)                        # every assumption/fallback lands here and in the metrics JSON (a non-default label_source first)
    print(f"dataset_tag={DATASET_TAG} label_source={LABEL_SOURCE} mode={MODE} variant={VARIANT} res={RES} epochs={EPOCHS} "
          f"bs={BS}x{GA} max_hours={MAX_HOURS} smoke={SMOKE_TRAIN_N}/{SMOKE_VALID_N} scale_jitter={SCALE_JITTER} "
          f"eval_head_res={EVAL_HEAD_RES} allow_extra_days={ALLOW_EXTRA_DAYS} allow_undensified_extra_day={ALLOW_UNDENSIFIED_EXTRA_DAY} "
          f"allow_below_floor={ALLOW_BELOW_FLOOR} "
          f"force={FORCE} allow_thin_night_valid={ALLOW_THIN_NIGHT_VALID} run_tag={RUN_TAG} "
          f"ckpt_sync_min={CKPT_SYNC_MIN} seed={SEED}")
    print(f"dataset {DS_VOL}\ncheckpoints {CK_VOL}\nfigures {FIG_VOL}\nmetrics {MET_SMOKE if MODE == 'smoke' else MET_FULL}")

# COMMAND ----------

# MAGIC %md ## Staging helpers, manifest guards, dataset assembly

# COMMAND ----------

if not SELFTEST_ONLY:
    def stage(src_vol, dst_local, what=""):
        """dbutils.fs.cp Volume -> /local_disk0 (~15 MB/s; FUSE copy is 30x slower), 3 retries, 5 s sleeps
        (Tol_17:155-161). Returns (seconds, MB)."""
        if osp.exists(dst_local): os.remove(dst_local)
        t0 = time.time(); ok = False
        for attempt in range(3):
            try:
                dbutils.fs.cp("dbfs:" + src_vol, "file:" + dst_local); ok = True; break
            except Exception as e:
                print(f"  stage retry {attempt+1} for {what or src_vol}: {e}"); time.sleep(5)
        if not ok: raise RuntimeError(f"could not stage {src_vol}")
        dt = time.time() - t0; mb = osp.getsize(dst_local) / 1e6
        print(f"  staged {what or osp.basename(src_vol)}: {mb:.0f} MB in {dt:.0f} s ({mb / max(dt, 1e-6):.1f} MB/s)")
        return dt, mb

    def put(src_local, dst_vol):
        """local -> Volume via dbutils.fs.cp (driver process: main thread or the CkptSync daemon thread; never a subprocess/worker)."""
        os.makedirs(osp.dirname(dst_vol), exist_ok=True)
        dbutils.fs.cp("file:" + src_local, "dbfs:" + dst_vol, True)

    T_STAGE = {}
    man_path = osp.join(DS_VOL, "manifest.json")
    if not osp.exists(man_path): raise RuntimeError(f"no manifest at {man_path}")
    with open(man_path, "rb") as f: MAN_BYTES = f.read()
    MAN = json.loads(MAN_BYTES.decode("utf-8")); MAN_SHA = sha256_bytes(MAN_BYTES)
    SOURCE_DAYS = manifest_guards(MAN, LABEL_SOURCE, ALLOW_EXTRA_DAYS, ALLOW_BELOW_FLOOR, ALLOW_UNDENSIFIED_EXTRA_DAY)
    TRAIN_STRIDES, _stride_notes = extra_day_strides_check(MAN, ALLOW_UNDENSIFIED_EXTRA_DAY)   # already passed inside the guards; recorded
    for note in _stride_notes: print(f"⚠ {note}"); SILENT_FALLBACKS.append(note)
    print(f"training lattice strides (manifest constants['strides']): {TRAIN_STRIDES}")
    HG_REQUIRED = human_gate_required(MAN)
    HUMAN_GATE_REC = {"required": HG_REQUIRED, "status": MAN["human_gate"].get("status"), "present": MAN["human_gate"].get("present"),
                      "n_counted": MAN["human_gate"].get("n_counted"), "n_bad_counted": MAN["human_gate"].get("n_bad_counted"),
                      "unlisted_days": MAN["human_gate"].get("unlisted_days"), "sha256": MAN["human_gate"].get("sha256")}
    if HG_REQUIRED:
        for note in human_gate_size_notes(MAN["human_gate"], manifest_sheet_n(MAN)): print(f"⚠ {note}"); SILENT_FALLBACKS.append(note)
    else:
        note = (f"human gate not required by the export (decision 2026-09-23: the reviewed fault list is the human gate); "
                f"status found {HUMAN_GATE_REC['status']!r}")
        print(f"⚠ {note}"); SILENT_FALLBACKS.append(note)
    MAN_PROV = constants_provenance_check(MAN, ALLOW_EXTRA_DAYS)   # pooled constants from TRAIN_DAYS only; A_ref[d]/ACTIVE_THR[d] from their own source day (already passed inside the guards)
    for note in MAN_PROV["notes"]: print(f"⚠ {note}"); SILENT_FALLBACKS.append(note)
    GATES_REC = gates_check(MAN["gates"])                           # already passed inside the guards; the record goes to train_summary.json
    print(f"band: frag_lo {MAN_PROV['frag_lo']} ({MAN_PROV['frag_lo_provenance']}), over_hi {MAN_PROV['over_hi']}, valid_buffer_min {MAN_PROV['valid_buffer_min']} | "
          f"G9: {GATES_REC}")
    ALLOW_TEST_DAY_REASON = str(MAN.get("allow_test_day_reason", "") or "").strip()
    TEACHER_DETECT = {k: (MAN.get("teacher_detect") or {}).get(k) for k in ("ran", "n_frames", "n_dropped", "drop_frac", "drop_high", "intersection_n")}
    F9_NOTES = f9_control_check(LABEL_SOURCE, MAN.get("teacher_detect"))
    for note in F9_NOTES: print(f"⚠ {note}"); SILENT_FALLBACKS.append(note)
    print(f"teacher_detect: {TEACHER_DETECT}")
    print(f"manifest {MAN['tag']} sha256 {MAN_SHA[:16]} sealed={MAN['sealed']} human_gate={MAN['human_gate'].get('status')} "
          f"n_bad_by_kind={MAN['human_gate'].get('n_bad_by_kind')} n_counted={MAN['human_gate'].get('n_counted')} "
          f"veto.ran={MAN['sam3_veto'].get('ran')} label_sources={MAN['label_sources_available']}")
    print(f"band_json sha256 {MAN_PROV['band_json_sha256'][:16]} derived_from {MAN_PROV['band_json_derived_from']} | "
          f"faults_json sha256 {MAN_PROV['faults_json_sha256'][:16]} | {len(MAN['constants'])} constants: pooled ones from TRAIN_DAYS, "
          f"per-day ones from their own source day (checked)")
    print(f"SOURCE DAYS: {SOURCE_DAYS}   forbidden_eval_days: {MAN['forbidden_eval_days']}")
    if set(SOURCE_DAYS) - TRAIN_DAYS:
        print(f"⚠ EXTRA TRAINING DAYS {sorted(set(SOURCE_DAYS) - TRAIN_DAYS)} — 04-22 leaves the detector test set "
              f"(allow_test_day_reason: {ALLOW_TEST_DAY_REASON!r})")
        SILENT_FALLBACKS.append(f"extra training days {sorted(set(SOURCE_DAYS) - TRAIN_DAYS)}: {ALLOW_TEST_DAY_REASON!r}")
    SEAL_WAIVER = []                     # train_summary.json seal_waiver: Tol_44 evaluates an unsealed manifest only when this names below_floor
    if not MAN["sealed"]:
        # the only waivable term is below_floor (the guards raised otherwise) — name it
        waived = seal_terms_unmet(MAN, allow_below_floor=False)
        if len(waived) != 1 or not waived[0].startswith("counts.below_floor"):
            raise RuntimeError(f"unsealed manifest passed the guards with unmet terms {waived} — expected exactly below_floor")
        SEAL_WAIVER = ["below_floor"]
        print(f"⚠ manifest UNSEALED — the only unmet seal term, waived by allow_below_floor=true: {waived}")
        SILENT_FALLBACKS.append(f"trained on an unsealed manifest; waived seal terms: {waived}")
    if not MAN["sam3_veto"].get("ran", False):
        # sealed with Tol_42 require_veto=false: a deliberate, recorded skip upstream
        if MAN.get("require_veto") is not False:
            raise RuntimeError(f"sam3_veto.ran=false but manifest.require_veto={MAN.get('require_veto')!r}: G11 was skipped without Tol_42's "
                               "deliberate require_veto=false — refused")
        print("⚠ sam3_veto.ran=false — exported with require_veto=false: G11 (the unlabelled-animal veto) never ran on these labels")
        SILENT_FALLBACKS.append("sam3_veto not run (Tol_42 require_veto=false; G11 inactive on these labels)")

    # ---- images.tar -> /local_disk0 (timed separately from compute) ----
    for d in ("train", "valid"): os.makedirs(osp.join(DS_ROOT, d), exist_ok=True)
    tar_local = osp.join(LOCAL, "images.tar")
    T_STAGE["images_tar_s"], T_STAGE["images_tar_mb"] = stage(osp.join(DS_VOL, "images.tar"), tar_local, "images.tar")
    tar_sha = sha256_file(tar_local)
    if tar_sha != MAN["sha256"]["images_tar"]:
        raise RuntimeError(f"images.tar sha256 {tar_sha[:12]} != manifest {MAN['sha256']['images_tar'][:12]}")
    t0 = time.time(); n_members = 0
    with tarfile.open(tar_local) as tf:
        for m in tf:
            if not m.isfile(): continue
            if not (m.name.startswith("train/") or m.name.startswith("valid/")) or ".." in m.name:
                raise RuntimeError(f"unexpected tar member {m.name!r}")
            with tf.extractfile(m) as src, open(osp.join(DS_ROOT, m.name), "wb") as dst: shutil.copyfileobj(src, dst)
            n_members += 1
    T_STAGE["untar_s"] = time.time() - t0
    os.remove(tar_local)
    print(f"untarred {n_members} JPEGs into {DS_ROOT} in {T_STAGE['untar_s']:.0f} s")

    # ---- annotations: label source -> train JSON; valid is ALWAYS the tracked valid ----
    def read_ann(rel, sha_key):
        p = osp.join(DS_VOL, "annotations", rel, "_annotations.coco.json")
        if not osp.exists(p): raise RuntimeError(f"annotation file missing: {p}")
        with open(p, "rb") as f: b = f.read()
        sha = sha256_bytes(b); want = MAN["sha256"]["annotations"].get(sha_key)
        if sha != want: raise RuntimeError(f"annotation sha256 mismatch for {sha_key}: have {sha[:12]} want {str(want)[:12]}")
        return json.loads(b.decode("utf-8")), sha

    VALID_COCO, VALID_SHA = read_ann("tracked/valid", "tracked/valid")
    if LABEL_SOURCE == "teacher_detect":
        TRAIN_COCO, TRAIN_SHA = read_ann("teacher_detect/train", "teacher_detect/train")
    else:
        TRAIN_COCO, TRAIN_SHA = read_ann("tracked/train", "tracked/train")
    if LABEL_SOURCE == "tracked_intersection":
        ip = osp.join(DS_VOL, "intersection_image_ids.json")
        if not osp.exists(ip): raise RuntimeError(f"missing {ip}")
        with open(ip) as f: IDS = json.load(f)
        IDS = IDS["image_ids"] if isinstance(IDS, dict) else IDS
        n_before = len(TRAIN_COCO["images"])
        TRAIN_COCO = filter_coco_by_ids(TRAIN_COCO, IDS)
        print(f"tracked_intersection: {n_before} -> {len(TRAIN_COCO['images'])} train images "
              f"({len(IDS)} intersection ids; crops kept via parent_file_name)")
        if not TRAIN_COCO["images"]: raise RuntimeError("intersection filter left 0 train images")
        if MAN.get("teacher_detect", {}).get("intersection_n") is not None:
            n_full = sum(1 for im in TRAIN_COCO["images"] if not im.get("crop_variant", False))
            if n_full != int(MAN["teacher_detect"]["intersection_n"]):
                raise RuntimeError(f"intersection full-frame count {n_full} != manifest.teacher_detect.intersection_n {MAN['teacher_detect']['intersection_n']}")

    for split, coco in (("train", TRAIN_COCO), ("valid", VALID_COCO)):
        if coco["categories"] != CATEGORIES_EXPECTED:
            raise RuntimeError(f"{split} categories {coco['categories']} != {CATEGORIES_EXPECTED}")
        missing = [im["file_name"] for im in coco["images"] if not osp.exists(osp.join(DS_ROOT, split, im["file_name"]))]
        if missing: raise RuntimeError(f"{len(missing)} {split} images referenced but absent, e.g. {missing[:3]}")
        if len({im["file_name"] for im in coco["images"]}) != len(coco["images"]):
            raise RuntimeError(f"duplicate file_name in {split}")
        if any("segmentation" not in a for a in coco["annotations"]):
            raise RuntimeError(f"{split}: an annotation lacks 'segmentation' (rfdetr checks only anno[0], coco.py:733)")
        badph = [(im["file_name"], im.get("phase4")) for im in coco["images"] if im.get("phase4") not in PHASES4]
        if badph: raise RuntimeError(f"{split}: {len(badph)} images with phase4 outside {PHASES4}, e.g. {badph[:3]}")
        with open(osp.join(DS_ROOT, split, "_annotations.coco.json"), "w") as f: json.dump(coco, f)
    TRAIN_CLIPS, VALID_CLIPS, VALID_CLIPS_EMPTY = clip_split_check(TRAIN_COCO["images"], VALID_COCO["images"], MAN["split"])   # whole-clip disjoint, ⊆ manifest.split
    print(f"clip split OK: {len(TRAIN_CLIPS)} train clips, {len(VALID_CLIPS)} valid clips {VALID_CLIPS}, disjoint and within manifest.split")
    if VALID_CLIPS_EMPTY:
        note = (f"manifest.split.valid_clips {VALID_CLIPS_EMPTY} contribute ZERO valid images (all their lattice frames were gated/dropped); "
                f"checkpoint selection and op_threshold see only {VALID_CLIPS} (valid phase4 shares {phase_shares(VALID_COCO['images'])})")
        print(f"⚠ {note}"); SILENT_FALLBACKS.append(note)
    VALID_PHASE4, _vnotes = valid_phase_check(VALID_COCO["images"], VALID_NIGHT_MIN, ALLOW_THIN_NIGHT_VALID)   # night must be in valid
    for note in _vnotes: print(f"⚠ {note}"); SILENT_FALLBACKS.append(note)
    print(f"valid images per phase4 (checkpoint selection + op_threshold see these): {VALID_PHASE4}")
    VALID_BUFFER_MIN_GAP = valid_buffer_check(TRAIN_COCO["images"], VALID_COCO["images"], MAN_PROV["valid_buffer_min"])   # G12 on the images
    print(f"valid buffer OK: nearest same-day train/valid pair {VALID_BUFFER_MIN_GAP if VALID_BUFFER_MIN_GAP is None else round(VALID_BUFFER_MIN_GAP, 2)} min "
          f">= valid_buffer_min {MAN_PROV['valid_buffer_min']}")
    IMG_DAYS = {sp: image_provenance_check(co["images"], SOURCE_DAYS, MAN["sources"], sp) for sp, co in (("train", TRAIN_COCO), ("valid", VALID_COCO))}
    print(f"image provenance OK (test-day exclusion on the images themselves): {IMG_DAYS}")
    N_TRAIN, N_VALID = len(TRAIN_COCO["images"]), len(VALID_COCO["images"])
    n_full = sum(1 for im in TRAIN_COCO["images"] if not im.get("crop_variant", False)); n_crop = N_TRAIN - n_full
    C = MAN["counts"]
    if N_VALID != int(C["valid"]): raise RuntimeError(f"valid image count {N_VALID} != manifest {C['valid']}")
    if LABEL_SOURCE == "tracked" and (n_full != int(C["train_full"]) or n_crop != int(C["train_crops"])):
        raise RuntimeError(f"train counts full/crop {n_full}/{n_crop} != manifest {C['train_full']}/{C['train_crops']}")
    if LABEL_SOURCE == "teacher_detect":
        td = MAN.get("teacher_detect", {})
        # manifest.teacher_detect.n_frames counts EVERY full frame in admitted.json (train + valid + crop parents),
        # so the train-only full-frame count is bounded above by n_frames - n_dropped — an equality would be wrong by design.
        if td.get("n_frames") is not None and n_full > int(td["n_frames"]) - int(td.get("n_dropped", 0)):
            raise RuntimeError(f"teacher_detect train full-frame count {n_full} > manifest n_frames-n_dropped "
                               f"{int(td['n_frames']) - int(td.get('n_dropped', 0))} — the export's accounting is inconsistent")
        if n_full > int(C["train_full"]): raise RuntimeError(f"teacher_detect has more full frames ({n_full}) than tracked ({C['train_full']})")
    EXPORT_BALANCE, _bal_notes = export_balance_check(MAN)          # hour cap / 10-min blocks / quotas of the FINAL export (review 2026-09-23)
    for note in _bal_notes: print(f"⚠ {note}"); SILENT_FALLBACKS.append(note)
    print(f"export balance: max hour share {EXPORT_BALANCE['hour_share_max']} (cap {EXPORT_BALANCE['hour_cap']}), "
          f"{EXPORT_BALANCE['distinct_10min_blocks']} distinct 10-min blocks, quotas achieved {EXPORT_BALANCE['quota_achieved']}, unmet {EXPORT_BALANCE['quota_unmet']}")
    TRAIN_FRAMESET = frameset_identity(TRAIN_COCO["images"])     # Tol_44 asserts DET-R and DET-C trained on the same full-frame set
    print(f"train {N_TRAIN} images ({n_full} full + {n_crop} crops), valid {N_VALID} | train phase4 shares {phase_shares(TRAIN_COCO['images'])}")
    print(f"train frame-set identity: {TRAIN_FRAMESET['n_full']} full frames, file_names sha256 {TRAIN_FRAMESET['file_names_sha256'][:16]}, "
          f"image_ids sha256 {TRAIN_FRAMESET['image_ids_sha256'][:16]}")
    print(f"staging: {T_STAGE}")

# COMMAND ----------

# MAGIC %md ## Pretrained weights (pre-staged or downloaded) + coexistence cell (every mode)

# COMMAND ----------

if not SELFTEST_ONLY:
    import torch, importlib.metadata as im_meta
    MP_BEFORE_RFDETR = torch.get_float32_matmul_precision()      # captured BEFORE `import rfdetr`, which sets "high" at import (detr.py:52-55)
    from PIL import Image
    from huggingface_hub import login
    login(token=dbutils.secrets.get(scope="hf", key="token").strip().strip('"').strip("'").strip())

    pre_vol = osp.join(PRE_VOL, PRETRAIN_FILE); pre_local = osp.join(RF_HOME, PRETRAIN_FILE)
    PRETRAINED_SOURCE = None
    if osp.exists(pre_vol):
        T_STAGE["pretrained_s"], T_STAGE["pretrained_mb"] = stage(pre_vol, pre_local, PRETRAIN_FILE)
        PRETRAINED_SOURCE = "volume:" + pre_vol
    else:
        print(f"⚠ no pre-staged weights at {pre_vol}; rfdetr will download {PRETRAIN_FILE} from storage.googleapis.com")
        PRETRAINED_SOURCE = "download"; SILENT_FALLBACKS.append("pretrained weights downloaded (not pre-staged)")

    import transformers
    if transformers.__version__ != "5.2.0": raise RuntimeError(f"transformers {transformers.__version__} != 5.2.0")
    from transformers import Sam3Model, Sam3Processor
    import rfdetr                                                    # RF_HOME already set above
    from rfdetr import RFDETRSegSmall, RFDETRSegXLarge, from_checkpoint
    import rfdetr.training
    from rfdetr.datasets.coco import compute_multi_scale_scales    # coco.py:277
    from rfdetr.assets.model_weights import ModelWeights            # model_weights.py:236-271 (md5 registry)
    import cv2
    try:
        im_meta.distribution("opencv-python-headless")
    except im_meta.PackageNotFoundError:
        raise RuntimeError("opencv-python-headless is not installed")
    for gui in ("opencv-python", "opencv-contrib-python"):
        try:
            im_meta.distribution(gui); raise RuntimeError(f"{gui} is installed and would shadow the headless build")
        except im_meta.PackageNotFoundError:
            pass
    import pycocotools.mask, faster_coco_eval, torchmetrics, pytorch_lightning
    from packaging.version import Version
    if Version(torch.__version__.split("+")[0]) < Version("2.5"):
        raise RuntimeError(f"torch {torch.__version__} < 2.5 (rfdetr needs torch.accelerator)")
    if not torch.cuda.is_available(): raise RuntimeError("no CUDA device")
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("GPU has no bf16 support (bf16 only, never fp16)")
    device = torch.device("cuda"); DTYPE = torch.bfloat16
    VERSIONS = {"torch": torch.__version__, "transformers": transformers.__version__, "rfdetr": rfdetr.__version__
                if hasattr(rfdetr, "__version__") else im_meta.version("rfdetr"),
                "pytorch_lightning": pytorch_lightning.__version__, "torchmetrics": torchmetrics.__version__,
                "faster_coco_eval": getattr(faster_coco_eval, "__version__", "?"), "cv2": cv2.__version__,
                "numpy": np.__version__, "python": sys.version.split()[0]}
    print("versions:", json.dumps(VERSIONS))
    VARIANT_CLS = {"seg-small": RFDETRSegSmall, "seg-xlarge": RFDETRSegXLarge}[VARIANT]   # variants.py:186-194 / 219-227

    def build_model():
        """Constructor-only kwargs (config.py:433-527): resolution (PE re-derived, config.py:529-563) and
        num_classes=1 so a category-count surprise is caught rather than silently absorbed (detr.py:2054-2064)."""
        return VARIANT_CLS(resolution=RES, num_classes=1)

    def rf_predict_rows(model, frame_rgb_u8, thr):
        """predict() (detr.py:2333-2808) -> [(bool mask [H,W], score)] with the '__background__' rows dropped
        (class_id >= num_classes, detr.py:2757-2772). No NMS inside predict() (postprocess.py:121-144)."""
        d = model.predict(frame_rgb_u8, threshold=thr, include_source_image=False)
        ncls = int(model.model_config.num_classes)
        if d.mask is None or len(d) == 0: return []
        keep = d.class_id < ncls
        return [(np.asarray(m, dtype=bool), float(s)) for m, s in zip(d.mask[keep], d.confidence[keep])]

    # VERBATIM from Tol_03_track_v4.py:453-472
    def detect(pil, topk=None):
        """Text-prompted instance segmentation -> deduped (mask, score) list, largest first.
        IoS dedupe because night IR over-segments one calf into overlapping fragments (up to 9
        for 4 calves) and area/score thresholds provably do NOT separate them."""
        with torch.no_grad():
            di = dproc(images=pil, text=DET_TEXT, return_tensors="pt").to(device)
            do = det(**di)
            res = dproc.post_process_instance_segmentation(
                do, threshold=DET_SCORE_MIN, mask_threshold=0.5,
                target_sizes=di["original_sizes"].tolist())[0]
        cand = []
        for m, s in zip(res["masks"], res["scores"]):
            mm = m.cpu().numpy().astype(bool)
            if mm.sum() >= MIN_MASK_PX: cand.append((mm, float(s)))
        cand.sort(key=lambda z: -z[0].sum())
        keep = []
        for m, s in cand:
            if all(m_ios(m, k) < DEDUPE_IOS and m_iou(m, k) < 0.5 for k, _ in keep):
                keep.append((m, s))
        return keep[:topk] if topk else keep

    # ---- coexistence: both models on one GPU, one forward each; any exception raises with versions ----
    t0 = time.time()
    try:
        torch.cuda.reset_peak_memory_stats()
        mp_before = MP_BEFORE_RFDETR                                  # read after `import torch`, before `import rfdetr` (see above)
        rf = build_model()                                            # triggers download_pretrain_weights (detr.py:509-531)
        rf_path = str(rf.model_config.pretrain_weights)
        if not osp.exists(rf_path):
            raise RuntimeError(f"pretrained weights absent after construction: {rf_path}; pre-stage {PRETRAIN_FILE} under "
                               f"{PRE_VOL} or allow egress to storage.googleapis.com")
        PRETRAINED_MD5 = hashlib.md5(open(rf_path, "rb").read()).hexdigest()
        asset = ModelWeights.from_filename(PRETRAIN_FILE)
        if asset is not None and asset.md5_hash != PRETRAINED_MD5:
            raise RuntimeError(f"{PRETRAIN_FILE} md5 {PRETRAINED_MD5} != registry {asset.md5_hash} (model_weights.py:236-271)")
        dproc = Sam3Processor.from_pretrained(SAM3_ID)                # Tol_03:422-423
        det = Sam3Model.from_pretrained(SAM3_ID, torch_dtype=DTYPE).to(device).eval()
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, size=(FRAME_H, FRAME_W, 3), dtype=np.uint8)
        rows_rf = rf_predict_rows(rf, noise, DET_SCORE_MIN)           # random frame: any count is legal here
        rows_t = detect(Image.fromarray(noise), topk=TOPK_DET)
        torch.cuda.synchronize()
        COEXIST = {"peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3), "versions": VERSIONS,
                   "rf_rows_on_noise": len(rows_rf), "sam3_rows_on_noise": len(rows_t),
                   "matmul_precision_before": mp_before, "matmul_precision_after": torch.get_float32_matmul_precision(),
                   "pretrained_md5": PRETRAINED_MD5, "pretrained_source": PRETRAINED_SOURCE, "seconds": round(time.time() - t0, 1)}
        print("coexistence OK:", json.dumps(COEXIST))
        if COEXIST["matmul_precision_after"] != mp_before:
            print(f"  note: rfdetr import set float32 matmul precision {mp_before} -> {COEXIST['matmul_precision_after']} (detr.py:52-55); noted, not failed")
        del det, dproc, rf; gc.collect(); torch.cuda.empty_cache()   # free SAM3; training owns the GPU
    except Exception as e:
        raise RuntimeError(f"COEXISTENCE FAILED ({type(e).__name__}: {e}) with versions {json.dumps(VERSIONS)} — "
                           "a two-process split is a design change, never adopted silently") from e

    # cross-check the pure replica against rfdetr's own scale computation
    _sc = compute_multi_scale_scales(RES, expanded_scales=False, patch_size=12, num_windows=2)
    if list(_sc) != multi_scale_scales(RES, False, 12, 2):
        raise RuntimeError(f"scale replica {multi_scale_scales(RES, False, 12, 2)} != rfdetr {_sc}")
    SCALES_RESOLVED = list(_sc)
    print(f"multi-scale set at {RES} (expanded_scales=False): {SCALES_RESOLVED} — the dataset delivers {max(_sc)}², per-batch scale on GPU")

# COMMAND ----------

# MAGIC %md ## Training call builder, timing probe, rendering

# COMMAND ----------

if not SELFTEST_ONLY:
    # mirrors Tol_42:2123-2129: a notebook has no __file__, so fall back to the deployed Workspace source (review 2026-09-23)
    NOTEBOOK_SHA = None; NOTEBOOK_SHA_SRC = None
    for _cand in ([__file__] if "__file__" in globals() else []) + ["/Workspace/Users/<DATABRICKS_USER>/Model Distillation/Testing on Tol2/Tol_43_train_rfdetr_seg.py",
                                                                     "/Workspace/Users/<DATABRICKS_USER>/Model Distillation/Testing on Tol2/Tol_43_train_rfdetr_seg"]:
        try:
            if osp.isfile(_cand): NOTEBOOK_SHA = sha256_file(_cand); NOTEBOOK_SHA_SRC = _cand; break
        except Exception: pass
    if NOTEBOOK_SHA is None:
        NOTEBOOK_SHA = "n/a (neither __file__ nor the /Workspace notebook source was readable as a file)"
        print(f"⚠ notebook_sha256 unavailable: {NOTEBOOK_SHA}"); SILENT_FALLBACKS.append("notebook_sha256 unavailable (source not readable as a file)")
    else:
        print(f"notebook sha256 {NOTEBOOK_SHA[:16]} from {NOTEBOOK_SHA_SRC}")
    CONSTANTS_SHA = sha256_bytes(json.dumps(MAN["constants"], sort_keys=True).encode("utf-8"))   # canonical JSON of the manifest's constants block
    PROVENANCE = {"notebook": "Tol_43_train_rfdetr_seg", "notebook_sha256": NOTEBOOK_SHA, "notebook_sha256_source": NOTEBOOK_SHA_SRC, "dataset_tag": DATASET_TAG,
                  "dataset_manifest_sha256": MAN_SHA, "source_days": SOURCE_DAYS, "label_source": LABEL_SOURCE,
                  "variant": VARIANT, "resolution": RES, "seed": SEED, "run_tag": RUN_TAG,
                  "allow_test_day_reason": ALLOW_TEST_DAY_REASON,
                  "band_json_sha256": MAN_PROV["band_json_sha256"], "band_json_derived_from": MAN_PROV["band_json_derived_from"],
                  "faults_json_sha256": MAN_PROV["faults_json_sha256"], "constants_sha256": CONSTANTS_SHA,
                  "train_clips": TRAIN_CLIPS, "valid_clips": VALID_CLIPS,      # Tol_44 asserts against these
                  "train_frameset_file_names_sha256": TRAIN_FRAMESET["file_names_sha256"], "train_frameset_n_full": TRAIN_FRAMESET["n_full"],
                  "teacher_detect_drop_high": TEACHER_DETECT.get("drop_high")}

    def train_kwargs(dataset_dir, output_dir, epochs, bs, ga, resume=None):
        """Every key is a verified TrainConfig/SegmentationTrainConfig field; one-line reasons inline.
        Each literal names whether it is an rfdetr default (config.py line) or a project choice."""
        KW = dict(dataset_dir=dataset_dir, output_dir=output_dir, dataset_file="roboflow",   # coco.py:1455-1459 layout
            epochs=epochs, batch_size=bs, grad_accum_steps=ga,
            eval_batch_size=1,                        # project choice: validation upsamples num_select masks to 2304x1296 (postprocess.py:234-269)
            lr=1e-4,                                  # rfdetr default (config.py:1011)
            lr_encoder=1.5e-5,                        # project choice: 10x below the 1.5e-4 default (config.py:1012) — one pen, one camera, ~5k images
            weight_decay=1e-4,                        # rfdetr default (config.py:1095)
            clip_max_norm=0.1,                        # rfdetr default (config.py:1230)
            amp_dtype="bf16",                         # project choice (bf16 only, never fp16): trainer.py:594-596 -> bf16-mixed
            multi_scale=True, expanded_scales=False,  # rfdetr defaults; scales 552..720 at 624 (coco.py:277-291)
            scale_jitter=SCALE_JITTER,                # widget, default false: coco.py:874-886 crop branch; the export's 16:9 crops are the crop aug
            augmentation_backend="torchvision",       # project choice: config.py:137-144 "cpu" silently picks Albumentations if installed
            aug_config=None,                          # rfdetr default: horizontal flip only (coco.py:1076-1096)
            use_ema=True,                             # rfdetr default (config.py:1069)
            ema_decay=0.993,                          # rfdetr default (config.py:1040)
            early_stopping=True,                      # project choice
            early_stopping_patience=8,                # project choice; rfdetr default is 10 (config.py:1117)
            early_stopping_min_delta=0.001,           # rfdetr default (config.py:1118)
            best_model_metric="map",                  # rfdetr default -> val/segm_mAP_50_95 (trainer.py:277-280, 351-357)
            checkpoint_interval=5,                    # project choice: != 1 so last.ckpt exists (trainer.py:320-333); default 10 (config.py:1043)
            eval_interval=1,                          # rfdetr default (config.py:1130): validation every epoch so early stopping and best-EMA see every epoch
            eval_masks_head_resolution=EVAL_HEAD_RES, num_workers=NUM_WORKERS, seed=SEED,   # widgets; seed=None would mean no seeding (module_model.py:471-472)
            tensorboard=False, wandb=False, mlflow=False, progress_bar=None, run_test=False,
            class_names=["calf"], notes=json.dumps(PROVENANCE), device="cuda")
        if resume: KW["resume"] = resume               # forwarded verbatim to trainer.fit(ckpt_path=) (config.py:1601-1613)
        check_train_kwargs(KW)
        return KW

    class FirstStepProbe:
        """T_setup = time to the first training step, observed as the first CUDA allocation jump of >= 2 GB above
        the pre-train() baseline (activations of a 720² batch dwarf the weights); metrics.csv first-appearance is
        recorded too (it flushes at step 50, so it is NOT the setup time). Fallback 60 s if nothing was observed."""
        def __init__(self, out_dir, jump_gb=2.0):
            self.out_dir, self.jump = out_dir, jump_gb * 1e9
            self.t_first_step = None; self.t_csv = None; self._stop = threading.Event()
        def start(self):
            self.t0 = time.perf_counter(); self.base = torch.cuda.memory_allocated()
            self.th = threading.Thread(target=self._run, daemon=True); self.th.start(); return self
        def _run(self):
            csvp = osp.join(self.out_dir, "metrics.csv")
            while not self._stop.is_set():
                try:
                    if self.t_first_step is None and torch.cuda.memory_allocated() - self.base >= self.jump:
                        self.t_first_step = time.perf_counter() - self.t0
                    if self.t_csv is None and osp.exists(csvp):
                        self.t_csv = time.perf_counter() - self.t0
                    if self.t_first_step is not None and self.t_csv is not None: break
                except Exception:
                    pass
                time.sleep(0.25)
        def stop(self):
            self._stop.set(); self.th.join(timeout=2)
            return self.t_first_step, self.t_csv

    def replacement_sampler_engaged(n_train, bs, ga):
        return n_train < bs * ga * MIN_TRAIN_BATCHES          # module_data.py:374-396

    def read_training_config(out_dir):
        p = osp.join(out_dir, "training_config.json")        # detr.py:1117-1132
        if not osp.exists(p): raise RuntimeError(f"training_config.json not written to {out_dir}")
        with open(p) as f: return json.load(f)

    def read_ckpt_meta(path):
        """epoch / best_total_source from a .pth payload (best_model.py:196-233, strip_checkpoint extra_metadata)."""
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"⚠ could not read checkpoint metadata from {path}: {e}"); return {}
        out = {k: ck.get(k) for k in ("epoch", "best_total_source", "global_step", "model_name", "rfdetr_version")}
        del ck; return out

    def decode_train_image(split, file_name):
        """Raw JPEG bytes as written by the export (never stretched — deployment input is raw IR)."""
        return np.asarray(Image.open(osp.join(DS_ROOT, split, file_name)).convert("RGB"))

    def valid_label_masks(coco):
        by = defaultdict(list)
        for a in coco["annotations"]: by[a["image_id"]].append(a)
        return by

    def render_panel(rgb, kept, title, stretch):
        """Tol_17 conventions: 960x540 INTER_AREA, INTER_NEAREST masks, 0.55 img + 0.45 colour fill, 2-px contour,
        black bar with the title. One neutral colour per instance rank; display-only stretch for dark phases."""
        small = cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
        if stretch: small = luminance_stretch(small)
        small = np.ascontiguousarray(small)
        cols = [(255, 60, 60), (60, 220, 60), (40, 160, 255), (255, 220, 40), (255, 60, 255), (60, 220, 220)]
        for i, (m, s) in enumerate(kept):
            ms = cv2.resize(m.astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not ms.any(): continue
            col = cols[i % len(cols)]
            small[ms] = (0.55 * small[ms] + 0.45 * np.array(col, np.float32)).astype(np.uint8)
            cnts, _ = cv2.findContours(ms.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(small, cnts, -1, col, 2)
            ys, xs = np.where(ms)
            cv2.putText(small, f"{s:.2f}", (int(xs.mean()) - 14, int(ys.mean()) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(small, f"{s:.2f}", (int(xs.mean()) - 14, int(ys.mean()) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.rectangle(small, (0, 0), (960, 22), (0, 0, 0), -1)
        cv2.putText(small, title + ("  [display stretch]" if stretch else ""), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return small

    def render_valid_preds(model, coco, out_png, thr=DET_SCORE_MIN, n=8, seed=0):
        """8 valid frames (seeded, stratified), predictions at `thr` after the post-chain: a broken pipeline looks broken here."""
        full = [im for im in coco["images"] if not im.get("crop_variant", False)]
        pick = stratified_subset(full, n, seed)
        panels = []
        for im in pick:
            rgb = decode_train_image("valid", im["file_name"])
            kept = apply_threshold_and_chain(rf_predict_rows(model, rgb, SWEEP_PROBE_THR), thr)
            ph = im.get("phase4", "?")
            panels.append(render_panel(rgb, kept, f"{im.get('day', '')} {str(im.get('wall', ''))[11:19]} {ph} thr={thr} kept={len(kept)}",
                                       ph in ("night", "dusk", "dawn")))
        while len(panels) % 4: panels.append(np.zeros((540, 960, 3), np.uint8))
        rows = [np.concatenate(panels[i:i + 4], 1) for i in range(0, len(panels), 4)]
        grid = np.concatenate(rows, 0)
        loc = osp.join(LOCAL, osp.basename(out_png))
        Image.fromarray(grid).save(loc, quality=90); put(loc, out_png)
        print(f"rendered {out_png}")

    def reload_best(out_dir):
        """from_checkpoint (detr.py:537-872): class from model_name, num_classes from class_embed.weight. Resolution is
        PINNED: strip_checkpoint (state_dict.py:184-205, called on checkpoint_best_total.pth by best_model.py:633-634)
        keeps model/args/model_name/rfdetr_version/_PTL_COMPAT_KEYS only and drops 'model_config' — the one place
        from_checkpoint reads resolution from — so an unpinned reload builds the class default (SegSmall 384,
        config.py:903) and bicubic-shrinks the trained PE. resolution=RES is forwarded to the constructor and the PE
        validator re-derives RES//12 (config.py:529-563). The in-memory model after train() holds the LIVE, non-EMA
        weights (detr.py:1097) — always reload. Tol_44's DET-R loader must pin the same resolution (train_summary.json)."""
        p = osp.join(out_dir, BEST_CKPT_FILE)
        if not osp.exists(p): raise RuntimeError(f"{p} missing after train()")
        m = from_checkpoint(p, resolution=RES)
        if int(m.model_config.num_classes) != 1: raise RuntimeError(f"reloaded num_classes {m.model_config.num_classes} != 1")
        if list(m.class_names) != ["calf"]: raise RuntimeError(f"reloaded class_names {m.class_names} != ['calf']")
        if int(m.model_config.resolution) != RES: raise RuntimeError(f"reloaded resolution {m.model_config.resolution} != {RES}")
        pe_want = RES // int(m.model_config.patch_size)
        if int(m.model_config.positional_encoding_size) != pe_want:
            raise RuntimeError(f"reloaded positional_encoding_size {m.model_config.positional_encoding_size} != {RES}//{m.model_config.patch_size} = {pe_want}: "
                               "the trained PE would be interpolated at load (weights.py:622)")
        black = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        kb = apply_threshold_and_chain(rf_predict_rows(m, black, SWEEP_PROBE_THR), DET_SCORE_MIN)
        if kb: raise RuntimeError(f"reloaded model keeps {len(kb)} masks on a black frame at {DET_SCORE_MIN}")
        return m, p

# COMMAND ----------

# MAGIC %md ## `mode=smoke` — measure, project, gate (mandatory before `full`)

# COMMAND ----------

if not SELFTEST_ONLY and MODE == "smoke":
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    for d in ("train", "valid"): os.makedirs(osp.join(SMOKE_DIR, d), exist_ok=True)
    SM_TRAIN = stratified_subset(TRAIN_COCO["images"], SMOKE_TRAIN_N, SEED)
    SM_VALID = stratified_subset(VALID_COCO["images"], SMOKE_VALID_N, SEED + 1)
    if len(SM_TRAIN) < SMOKE_TRAIN_N: print(f"⚠ smoke train subset {len(SM_TRAIN)} < requested {SMOKE_TRAIN_N} (dataset smaller)")
    if len(SM_VALID) < SMOKE_VALID_N: print(f"⚠ smoke valid subset {len(SM_VALID)} < requested {SMOKE_VALID_N}")
    for split, sub, coco in (("train", SM_TRAIN, TRAIN_COCO), ("valid", SM_VALID, VALID_COCO)):
        ids = {im["id"] for im in sub}
        for im in sub: os.link(osp.join(DS_ROOT, split, im["file_name"]), osp.join(SMOKE_DIR, split, im["file_name"]))
        with open(osp.join(SMOKE_DIR, split, "_annotations.coco.json"), "w") as f:
            json.dump({"images": sub, "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
                       "categories": coco["categories"]}, f)
    print(f"smoke subset: train {len(SM_TRAIN)} {phase_shares(SM_TRAIN)} (full set {phase_shares(TRAIN_COCO['images'])}) | "
          f"valid {len(SM_VALID)} {phase_shares(SM_VALID)}")
    SM_OUT = osp.join("/local_disk0/rfdetr_out", RUN_TAG + "_smoke")

    bs, ga = BS, GA; OOM_STEPS = []
    while True:
        shutil.rmtree(SM_OUT, ignore_errors=True); os.makedirs(SM_OUT, exist_ok=True)
        KW = train_kwargs(SMOKE_DIR, SM_OUT, SMOKE_EPOCHS, bs, ga)
        if replacement_sampler_engaged(len(SM_TRAIN), bs, ga):
            print(f"⚠ replacement sampler will engage: {len(SM_TRAIN)} < 5 x {bs * ga} (module_data.py:374-396)")
        model = build_model()
        torch.cuda.reset_peak_memory_stats(); gc.collect(); torch.cuda.empty_cache()
        probe = FirstStepProbe(SM_OUT).start()
        t0 = time.perf_counter()
        try:
            model.train(**KW)
            T_total = time.perf_counter() - t0; break
        except torch.cuda.OutOfMemoryError as e:
            probe.stop()
            OOM_STEPS.append({"batch_size": bs, "grad_accum": ga, "error": str(e)[:200]})
            print(f"⚠ OOM at batch {bs} x accum {ga}: halving / doubling (smoke-only policy)")
            del model; gc.collect(); torch.cuda.empty_cache()
            if bs // 2 < 2: raise RuntimeError(f"OOM persists down to batch_size 2: {OOM_STEPS}")
            bs, ga = bs // 2, ga * 2
    t_first, t_csv = probe.stop()
    PEAK_VRAM = torch.cuda.max_memory_allocated() / 1e9
    if OOM_STEPS: SILENT_FALLBACKS.append(f"smoke OOM fallback to batch {bs} x accum {ga}")
    if t_first is None:
        T_setup = T_SETUP_FALLBACK_S; print(f"⚠ first step not observed; T_setup assumed {T_setup:.0f} s"); SILENT_FALLBACKS.append("T_setup assumed 60 s")
        T_SETUP_SRC = "assumed"
    else:
        T_setup = t_first; T_SETUP_SRC = "cuda allocation jump"
    print(f"smoke train: {T_total:.0f} s total, T_setup {T_setup:.0f} s ({T_SETUP_SRC}; metrics.csv appeared at {t_csv}), peak VRAM {PEAK_VRAM:.1f} GB")

    # ---- T_val: one validation pass via evaluate(split="val") — "val", not "valid" (detr.py:1195 raises) ----
    # evaluate() (detr.py:1228-1290) builds a NEW RFDETRModelModule, moves the live model CPU<->GPU, a datamodule with
    # persistent workers and a PTL Trainer before the pass; that fixed setup is measured separately on a 1-image split
    # and subtracted, otherwise it would be subtracted TWICE from the smoke train time (denominator = T_total - 2*T_val - T_setup).
    KWV = {k: v for k, v in KW.items() if k not in ("resume",)}
    KWV.update(dataset_dir=SMOKE_DIR, output_dir=osp.join(SM_OUT, "eval"), eval_batch_size=1)
    T_val_setup, T_VAL_SETUP_SRC = 0.0, "not measured"
    t0 = time.perf_counter()
    try:
        ev = model.evaluate(split="val", **KWV)
        T_val_raw = time.perf_counter() - t0
        print(f"evaluate(val) on {len(SM_VALID)} images: {T_val_raw:.0f} s incl. setup; keys {sorted(ev)[:6]}...")
        # 1-image probe split (hard links; both split dirs so the roboflow layout detection passes): its wall time ≈ setup
        PROBE_DIR = osp.join("/local_disk0/rfdetr_ds_probe", RUN_TAG); shutil.rmtree(PROBE_DIR, ignore_errors=True)
        for split, sub, coco in (("train", SM_TRAIN[:1], TRAIN_COCO), ("valid", SM_VALID[:1], VALID_COCO)):
            os.makedirs(osp.join(PROBE_DIR, split), exist_ok=True)
            for im in sub: os.link(osp.join(DS_ROOT, split, im["file_name"]), osp.join(PROBE_DIR, split, im["file_name"]))
            ids = {im["id"] for im in sub}
            with open(osp.join(PROBE_DIR, split, "_annotations.coco.json"), "w") as f:
                json.dump({"images": sub, "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
                           "categories": coco["categories"]}, f)
        try:
            KWP = dict(KWV); KWP.update(dataset_dir=PROBE_DIR, output_dir=osp.join(SM_OUT, "eval_probe"))
            t1 = time.perf_counter(); model.evaluate(split="val", **KWP); T_val_setup = time.perf_counter() - t1
            T_VAL_SETUP_SRC = "measured evaluate(split='val') on a 1-image split (setup + one image)"
            print(f"evaluate(val) on 1 image: {T_val_setup:.0f} s = the fixed setup cost")
        except Exception as e:
            T_VAL_SETUP_SRC = f"1-image probe failed ({type(e).__name__}: {str(e)[:120]}); setup NOT subtracted"
            print(f"⚠ {T_VAL_SETUP_SRC}"); SILENT_FALLBACKS.append("T_val setup probe failed: T_val is an upper bound incl. setup")
        if T_val_setup >= T_val_raw:
            raise RuntimeError(f"1-image evaluate ({T_val_setup:.0f} s) took longer than the {len(SM_VALID)}-image one ({T_val_raw:.0f} s): timing inconsistent")
        T_val = T_val_raw - T_val_setup
        T_VAL_SRC = ("measured evaluate(split='val') minus the 1-image setup probe" if T_val_setup > 0
                     else "measured evaluate(split='val') INCLUDING module/datamodule/trainer setup (upper bound)")
        print(f"T_val = {T_val:.0f} s ({T_VAL_SRC})")
    except (TypeError, ValueError, AttributeError) as e:
        T_val_raw = None
        T_val = T_VAL_FALLBACK_FRAC * T_total; T_VAL_SRC = f"assumed 0.25*T_total (evaluate failed: {type(e).__name__}: {str(e)[:120]})"
        print(f"⚠ {T_VAL_SRC}"); SILENT_FALLBACKS.append("T_val assumed 0.25*T_total")

    PROJ = project_hours(T_total, T_val, T_setup, len(SM_TRAIN), len(SM_VALID), N_TRAIN, N_VALID, EPOCHS)
    print(f"ips {PROJ['ips']:.2f} img/s | t_val_img {PROJ['t_val_img']:.3f} s | projected FULL run "
          f"{EPOCHS} epochs x ({N_TRAIN} train + {N_VALID} valid) x1.15 = {PROJ['hours_full']:.2f} h (cap {MAX_HOURS} h)")

    # ---- sanity: labels are being read, checkpoint reloads, predictions look like calves ----
    rows_csv = read_metrics_csv(osp.join(SM_OUT, "metrics.csv"))
    SM = summarize_metrics_csv(rows_csv)
    if not SM["best_val_segm_mAP_50"] or SM["best_val_segm_mAP_50"] <= 0:
        raise RuntimeError(f"val/segm_mAP_50 = {SM['best_val_segm_mAP_50']} after {SMOKE_EPOCHS} epochs: labels or masks are not being read")
    tc = read_training_config(SM_OUT)
    PREC = resolve_precision_replica(tc["model_config"].get("amp", True), tc["train_config"].get("amp_dtype"), True, torch.cuda.is_bf16_supported())
    if PREC != "bf16-mixed" or tc["train_config"].get("amp_dtype") != "bf16":
        raise RuntimeError(f"resolved precision {PREC} / amp_dtype {tc['train_config'].get('amp_dtype')} — not bf16-mixed")
    AUG_BACKEND = tc["train_config"].get("augmentation_backend")
    best_sm, best_sm_path = reload_best(SM_OUT)
    lm = valid_label_masks(VALID_COCO)
    ious = []
    for im in stratified_subset([i for i in SM_VALID if not i.get("crop_variant", False)], 20, SEED + 2):
        rgb = decode_train_image("valid", im["file_name"])
        kept = apply_threshold_and_chain(rf_predict_rows(best_sm, rgb, SWEEP_PROBE_THR), DET_SCORE_MIN)
        _, _, _, miou = hungarian_counts([m for m, _ in kept], [coco_seg_to_mask(a["segmentation"]) for a in lm[im["id"]]])
        ious.append(miou)
    print(f"descriptive only (mimicry): mean Hungarian mask IoU vs valid labels on 20 frames = {np.mean(ious):.3f}")
    render_valid_preds(best_sm, {"images": SM_VALID}, osp.join(FIG_VOL, "valid_preds_smoke.png"), seed=SEED)

    OUT = {"run_tag": RUN_TAG, "dataset_tag": DATASET_TAG, "label_source": LABEL_SOURCE, "dataset_manifest_sha256": MAN_SHA,
           "source_days": SOURCE_DAYS, "variant": VARIANT, "resolution": RES, "smoke": True,
           "smoke_train_n": len(SM_TRAIN), "smoke_valid_n": len(SM_VALID), "smoke_epochs": SMOKE_EPOCHS,
           "batch_size": bs, "grad_accum": ga, "effective_batch": bs * ga, "oom_steps": OOM_STEPS,
           "T_total_s": round(T_total, 1), "T_setup_s": round(T_setup, 1), "T_setup_source": T_SETUP_SRC, "t_metrics_csv_s": t_csv,
           "T_val": round(T_val, 1), "T_val_source": T_VAL_SRC, "T_val_raw_s": (round(T_val_raw, 1) if T_val_raw is not None else None),
           "T_val_setup_s": round(T_val_setup, 1), "T_val_setup_source": T_VAL_SETUP_SRC,
           "ips": round(PROJ["ips"], 3), "t_val_img": round(PROJ["t_val_img"], 4),
           "N_train": N_TRAIN, "N_valid": N_VALID, "epochs": EPOCHS, "hours_full": round(PROJ["hours_full"], 3), "max_hours": MAX_HOURS,
           "peak_vram_gb": round(PEAK_VRAM, 2), "gpu_name": torch.cuda.get_device_name(0), "precision_resolved": PREC,
           "scales_resolved": SCALES_RESOLVED, "augmentation_backend_resolved": AUG_BACKEND, "scale_jitter": SCALE_JITTER,
           "eval_masks_head_resolution": EVAL_HEAD_RES, "train_clips": TRAIN_CLIPS, "valid_clips": VALID_CLIPS,
           "valid_phase4_counts": VALID_PHASE4, "allow_thin_night_valid": ALLOW_THIN_NIGHT_VALID, "valid_buffer_min_gap_min": VALID_BUFFER_MIN_GAP,
           "gates": GATES_REC, "band": {k: MAN_PROV[k] for k in ("frag_lo", "frag_lo_provenance", "over_hi", "valid_buffer_min")},
           "provenance": PROVENANCE, "human_gate": HUMAN_GATE_REC, "f9_notes": F9_NOTES, "export_balance": EXPORT_BALANCE, "train_strides": TRAIN_STRIDES,
           "replacement_sampler": replacement_sampler_engaged(len(SM_TRAIN), bs, ga), "pretrained_md5": PRETRAINED_MD5,
           "pretrained_source": PRETRAINED_SOURCE, "coexist": COEXIST, "smoke_val": SM, "mean_hungarian_iou_20_descriptive_F3": round(float(np.mean(ious)), 4),
           "staging": T_STAGE, "seed": SEED, "versions": VERSIONS, "silent_fallbacks": SILENT_FALLBACKS,
           "note": "valid mAP is agreement with teacher-derived labels on training days; it is not detector quality (DATA_SPLITS F3).",
           "created": datetime.utcnow().isoformat() + "Z"}
    with open(MET_SMOKE, "w") as f: json.dump(OUT, f, indent=1)
    print(f"written {MET_SMOKE}")
    stop_gate(PROJ["hours_full"], MAX_HOURS, EPOCHS, RES)     # raises AFTER the projection is on the Volume
    print(f"SMOKE PASSED: projected {PROJ['hours_full']:.2f} h <= {MAX_HOURS} h — mode=full may run")

# COMMAND ----------

# MAGIC %md ## `mode=full` / `mode=resume` — train, sync, reload, operating threshold, copy out

# COMMAND ----------

if not SELFTEST_ONLY and MODE in ("full", "resume"):
    if not osp.exists(MET_SMOKE):
        raise RuntimeError(f"mode={MODE} refused: no smoke projection at {MET_SMOKE} (a full run without a measured projection is the refuted budget again)")
    with open(MET_SMOKE) as f: SMK = json.load(f)
    if not SMK.get("smoke"): raise RuntimeError(f"{MET_SMOKE} is not a smoke record")
    if SMK["dataset_manifest_sha256"] != MAN_SHA or SMK["label_source"] != LABEL_SOURCE:
        raise RuntimeError("smoke projection belongs to a different dataset/label_source than this run")
    # The smoke's ips / t_val_img were measured under ITS variant, resolution, counts, batch/accum (after the OOM policy),
    # scale_jitter and eval_masks_head_resolution; every one must equal the current widgets or the projection is not this run's.
    CURRENT = {"variant": VARIANT, "resolution": RES, "N_train": N_TRAIN, "N_valid": N_VALID, "batch_size": BS, "grad_accum": GA,
               "scale_jitter": SCALE_JITTER, "eval_masks_head_resolution": EVAL_HEAD_RES}
    mism = smoke_mismatches(SMK, CURRENT)
    if mism:
        raise RuntimeError(f"mode={MODE} refused: the smoke record does not describe this run — " +
                           "; ".join(f"{k}: smoke {a!r} vs now {b!r}" for k, a, b in mism) +
                           ". Set the widgets to the smoke values (batch_size/grad_accum are the values that SURVIVED the smoke OOM policy) or re-run mode=smoke.")
    # Re-project with the CURRENT epochs (epochs is not part of run_tag, so the smoke's hours_full may belong to another epochs value)
    HOURS_REPROJ = reproject_hours(SMK, N_TRAIN, N_VALID, EPOCHS)
    print(f"smoke projection: {SMK['ips']:.2f} img/s, t_val_img {SMK['t_val_img']:.3f} s; smoke said {SMK['hours_full']:.2f} h at epochs={SMK.get('epochs')}; "
          f"re-projected {HOURS_REPROJ:.2f} h at epochs={EPOCHS} (cap {MAX_HOURS} h)")
    stop_gate(HOURS_REPROJ, MAX_HOURS, EPOCHS, RES)             # the gate binds the re-projection, never the stale number

    RESUME_PATH = None; RESUMED = None
    RS_VOL = osp.join(CK_VOL, RESUME_DIRNAME)                 # resume_state/g<NNNNNN>/ generations (header, review 2026-09-23)
    def vol_path(fn): return osp.join(CK_VOL, fn)
    KW_PLAN = train_kwargs(DS_ROOT, OUT_DIR, EPOCHS, BS, GA)  # the hyper-parameters a resume state is bound to (RESUME_BOUND_HPARAMS)
    HP_BOUND = {k: KW_PLAN[k] for k in RESUME_BOUND_HPARAMS}

    def resume_consistency(paths):
        """THE rule a resume state must satisfy — applied by the sync daemon BEFORE it publishes a generation and by
        mode=resume after staging one: zip containers intact (torch.save / PTL checkpoints are zips), then
        resume_highwater_action on (last.ckpt BestModelCallback._best_ema, metrics.csv best, checkpoint_best_ema.pth epoch).
        Returns the facts + the action ('ok' | 'patch'); raises RuntimeError when the three files disagree."""
        for fn, p in paths.items():
            if p is not None and is_zip_container(fn): validate_zip_file(p)
        ck = torch.load(paths["last.ckpt"], map_location="cpu", weights_only=False)
        marks = [float(s["_best_ema"]) for s in (ck.get("callbacks") or {}).values() if isinstance(s, dict) and "_best_ema" in s]
        info = {"ckpt_epoch": ck.get("epoch"), "global_step": ck.get("global_step"), "best_ema_in_ckpt": (max(marks) if marks else None)}
        del ck
        ema_p = paths.get("checkpoint_best_ema.pth")
        ema = read_ckpt_meta(ema_p) if ema_p else {}
        if ema_p and ema.get("epoch") is None:
            raise RuntimeError(f"{ema_p}: no readable 'epoch' in the EMA checkpoint payload (best_model.py:146-233)")
        ms = summarize_metrics_csv(read_metrics_csv(paths["metrics.csv"]))
        info.update({"csv_best": ms["best_val_segm_mAP_50_95"], "csv_best_epoch": ms["best_epoch"], "csv_epochs_run": ms["epochs_run"],
                     "ema_epoch": ema.get("epoch")})
        info["action"] = resume_highwater_action(info["best_ema_in_ckpt"], info["csv_best"], info["csv_best_epoch"], info["ema_epoch"], ema_p is not None,
                                                 ckpt_epoch=info["ckpt_epoch"])
        return info

    # ---- provenance immutability: a COMPLETED run under this run_tag is never overwritten silently ----
    existing_run = [p for p in (vol_path(SUMMARY_FILE), MET_FULL) if osp.exists(p)]
    for p in refuse_overwrite(existing_run, FORCE, RUN_TAG, MODE):
        try:
            with open(p) as f: prev = json.load(f)
            prev_desc = f"created {prev.get('created')} epochs_run {prev.get('epochs_run')} best_epoch {prev.get('best_epoch')} op_threshold {prev.get('op_threshold')}"
        except Exception as e:
            prev_desc = f"unreadable ({e})"
        print(f"⚠ force=true: overwriting completed run {p} ({prev_desc}, sha256 {sha256_file(p)[:12]})")
        SILENT_FALLBACKS.append(f"force=true overwrote {osp.basename(p)} of a completed run: {prev_desc}; sha256 {sha256_file(p)[:12]}")
    if MODE == "full":
        if osp.exists(OUT_DIR): print(f"⚠ removing stale local output_dir {OUT_DIR}")
        shutil.rmtree(OUT_DIR, ignore_errors=True); os.makedirs(OUT_DIR, exist_ok=True)
        # a fresh run must not leave an older run's resume state behind it: every generation, plus the pre-generation flat layout
        gens_old = [gen_name(g[0]) for g in list_generations(RS_VOL)]
        flat_old = [fn for fn in [f for r in RESUMABLE for f in (r, r + ".sha256")] + ["last.ckpt.partial"] if osp.exists(vol_path(fn))]
        for fn in flat_old: os.remove(vol_path(fn))
        if osp.isdir(RS_VOL): shutil.rmtree(RS_VOL)
        if gens_old or flat_old:
            note = f"mode=full removed an earlier resume state under {CK_VOL}: generations {gens_old}, flat files {flat_old}"
            print(f"⚠ {note}"); SILENT_FALLBACKS.append(note)
    else:
        # The resume state is a Volume generation ONLY: a warm cluster's OUT_DIR could hold an older run's checkpoint_best_ema.pth
        # under the same run_tag, which rfdetr would then promote as "best" (best_model.py:606-617).
        if osp.exists(OUT_DIR): print(f"⚠ removing local output_dir {OUT_DIR}: resume state is staged from the Volume only")
        shutil.rmtree(OUT_DIR, ignore_errors=True); os.makedirs(OUT_DIR, exist_ok=True)
        gens = list_generations(RS_VOL); complete = [g for g in gens if g[2]]
        if not complete:
            raise RuntimeError(f"mode=resume: no complete generation under {RS_VOL} ({len(gens)} incomplete: {[gen_name(g[0]) for g in gens]}) — "
                               "nothing verified to resume from; use mode=full")
        tried = []
        for n, gdir, _ in complete:                                # newest first; an older complete generation is the fallback
            with open(osp.join(gdir, GEN_COMPLETE)) as f: REC = json.load(f)
            mism = resume_mismatches(REC.get("provenance"), PROVENANCE, REC.get("train_hparams"), HP_BOUND)
            if mism:                                               # a different run's state: never a fallback case
                raise RuntimeError(f"mode=resume refused: {gen_name(n)} was written by a different run — " +
                                   "; ".join(f"{k}: recorded {a!r} vs now {b!r}" for k, a, b in mism) + " — restore the widgets or use mode=full")
            paths = {fn: None for fn in RESUMABLE}
            try:
                for fn in RESUMABLE:
                    if fn not in (REC.get("files") or {}): continue
                    lp = osp.join(OUT_DIR, fn); stage(osp.join(gdir, fn), lp, f"{gen_name(n)}/{fn}")
                    have = sha256_file(lp)
                    if have != REC["files"][fn]:
                        raise RuntimeError(f"{gen_name(n)}/{fn}: sha256 {have[:12]} != {GEN_COMPLETE} {REC['files'][fn][:12]}")
                    verify_sidecar(lp, osp.join(gdir, fn + ".sha256"))
                    paths[fn] = lp
                if paths["last.ckpt"] is None or paths["metrics.csv"] is None:
                    raise RuntimeError(f"{gen_name(n)} lacks last.ckpt or metrics.csv")
                RESUMED = resume_consistency(paths); RESUMED["generation"] = gen_name(n)
                break
            except RuntimeError as e:
                tried.append(f"{gen_name(n)}: {e}"); print(f"⚠ resume generation {gen_name(n)} unusable: {e}")
                for fn in RESUMABLE:
                    if osp.exists(osp.join(OUT_DIR, fn)): os.remove(osp.join(OUT_DIR, fn))
        if RESUMED is None:
            raise RuntimeError(f"mode=resume: no complete generation verified ({tried}); use mode=full")
        if tried: SILENT_FALLBACKS.append(f"resume fell back past unusable generation(s): {tried}")
        RESUME_PATH = osp.join(OUT_DIR, "last.ckpt")
        # PTL 2.x saves rfdetr's `last` checkpoint in on_train_epoch_end, AFTER the epoch-end validation in which
        # BestModelCallback raised _best_ema and wrote checkpoint_best_ema.pth (trainer.py:320-333, best_model.py:500-574), so
        # last.ckpt normally carries its own epoch's mark. A generation snapshotted in the in-flight window of epoch e+1 (EMA file
        # + metrics.csv row written, last.ckpt still epoch e) carries the previous mark; resume_highwater_action admits that ONLY
        # when last.ckpt's epoch == csv best epoch - 1 and raises otherwise. The metrics.csv history (preserved across resume,
        # trainer.py:233-260) and the EMA checkpoint's own epoch are the truth; the mark is raised to the csv best before fit
        # so a later, worse epoch cannot overwrite the true best EMA checkpoint (best_model.py:570).
        if RESUMED["ema_epoch"] is None:
            print("⚠ the generation holds no checkpoint_best_ema.pth (EMA metric never improved before the snapshot) — legal resume without it")
        if RESUMED["action"] == "patch":
            ck = torch.load(RESUME_PATH, map_location="cpu", weights_only=False); n_patched = 0
            for s in (ck.get("callbacks") or {}).values():
                if isinstance(s, dict) and "_best_ema" in s: s["_best_ema"] = float(RESUMED["csv_best"]); n_patched += 1
            torch.save(ck, RESUME_PATH); del ck; gc.collect()
            print(f"⚠ last.ckpt EMA high-water mark {RESUMED['best_ema_in_ckpt']} was STALE relative to checkpoint_best_ema.pth (epoch "
                  f"{RESUMED['ema_epoch']}, metrics.csv best {RESUMED['csv_best']} @ {RESUMED['csv_best_epoch']}): patched {n_patched} "
                  f"callback state(s) to the csv best before fit (in-flight snapshot: last.ckpt is epoch {RESUMED['ckpt_epoch']}, saved in "
                  "on_train_epoch_end before the csv best epoch's own last.ckpt was written)")
            SILENT_FALLBACKS.append(f"resume: last.ckpt (epoch {RESUMED['ckpt_epoch']}) _best_ema {RESUMED['best_ema_in_ckpt']} -> {RESUMED['csv_best']} "
                                    f"(in-flight snapshot one epoch behind the csv best epoch {RESUMED['csv_best_epoch']}, {n_patched} callback states patched)")
        print(f"resuming from {RESUMED['generation']} (sha256 vs {GEN_COMPLETE} + sidecars + zip verified; ckpt epoch {RESUMED['ckpt_epoch']}, "
              f"best EMA mark {RESUMED['best_ema_in_ckpt']} [{RESUMED['action']}]) with metrics.csv history of {RESUMED['csv_epochs_run']} epochs "
              f"(best {RESUMED['csv_best']} @ {RESUMED['csv_best_epoch']}; staged EMA epoch {RESUMED['ema_epoch']}) in the SAME output_dir "
              f"{OUT_DIR} — never from a .pth (no optimizer state)")

    class CkptSync:
        """Daemon: every ckpt_sync_min publish ONE resume generation to the Volume (RS_VOL/g<NNNNNN>/). A tick (1) takes
        local snapshots of last.ckpt + checkpoint_best_ema.pth + metrics.csv back to back, each only when its size+mtime are
        stable (PTL rewrites last.ckpt in place every epoch) and retakes the set until it passes resume_consistency — the
        SAME rule mode=resume applies (snapshot_set); (2) uploads into a NEW directory, file then sidecar, COMPLETE.json
        last (publish_generation): a cut upload has no COMPLETE.json and is invisible to mode=resume, and the previous
        generation is never touched; (3) keeps the newest RESUME_KEEP_GENERATIONS complete generations (prune_generations).
        An unchanged set (same signatures as the last publish) is skipped. A lock serialises the daemon tick and the
        main-thread final call; stop() joins WITHOUT a timeout. Uploads use put() = dbutils.fs.cp from the daemon too (review
        2026-09-23): a FUSE copy is ~30x slower and a ~0.7 GB generation would lag training by ~23 min and
        block stop(); the repo rule forbids dbutils in SUBPROCESSES, not in driver threads
        — this supersedes the earlier rule "dbutils is not used from the thread". Only a failure of the
        COPY itself (CopierError from publish_generation) switches later daemon ticks to shutil (recorded); an inconsistent or
        unstable snapshot set skips the tick with the copier unchanged, and a prune failure after a published generation is
        recorded without republishing it. A tick slower than ckpt_sync_min is flagged ⚠ and recorded.
        Exceptions are logged and retried next tick, never fatal."""
        def __init__(self, out_dir, vol_root, every_min, record, copier=None):
            self.out_dir, self.vol_root, self.every, self.record = out_dir, vol_root, every_min * 60.0, record
            self.copier = copier; self.fallbacks = []; self.slow_ticks = []; self.prune_errors = []
            self.stg_dir = osp.join("/local_disk0/rfdetr_sync", osp.basename(out_dir))   # outside output_dir: PTL/rfdetr never see the snapshots
            shutil.rmtree(self.stg_dir, ignore_errors=True); os.makedirs(self.stg_dir, exist_ok=True)
            self._stop = threading.Event(); self._lock = threading.Lock(); self.log = []; self.last_sigs = None; self.last_gen = None
        def start(self):
            if self.every <= 0: print("ckpt sync OFF (ckpt_sync_min=0): only the final generation is published"); return self
            self.th = threading.Thread(target=self._run, daemon=True); self.th.start(); return self
        def sync_once(self, copier=None):
            with self._lock:
                res = snapshot_set(self.out_dir, self.stg_dir, resume_consistency)
                if res is None: return None                         # last.ckpt / metrics.csv not written yet
                sigs, info = res
                if sigs == self.last_sigs: return self.last_gen           # unchanged since the last publish: that generation IS the state
                stg = {fn: (osp.join(self.stg_dir, fn) if sigs[fn] is not None else None) for fn in RESUMABLE}
                shas = {fn: sha256_file(p) for fn, p in stg.items() if p is not None}
                os.makedirs(self.vol_root, exist_ok=True)
                gens = list_generations(self.vol_root); n = (gens[0][0] + 1) if gens else 1   # never reuse a number, incomplete ones included
                rec = dict(self.record); rec.update({"tick_utc": datetime.utcnow().isoformat() + "Z", "state": info})
                t0 = time.time()
                publish_generation(stg, shas, self.vol_root, n, rec, copier=copier)
                self.last_sigs, self.last_gen = sigs, gen_name(n)          # BEFORE pruning: a prune error must not republish this set
                try: removed = prune_generations(self.vol_root)
                except Exception as e:                                     # the new generation IS published; pruning is housekeeping
                    removed = []; msg = f"ckpt sync {gen_name(n)}: pruning older generations failed ({type(e).__name__}: {str(e)[:120]}); they stay on the Volume"
                    print(f"  ⚠ {msg}"); self.prune_errors.append(msg)
                mb = sum(osp.getsize(p) for p in stg.values() if p is not None) / 1e6; dt = time.time() - t0
                self.log.append({"t": rec["tick_utc"], "generation": gen_name(n), "mb": round(mb, 1), "s": round(dt), "mb_s": round(mb / max(dt, 1e-6), 2),
                                 "ckpt_epoch": info.get("ckpt_epoch"), "action": info.get("action"), "pruned": removed,
                                 "via": "dbutils.fs.cp" if copier else "shutil"})
                print(f"  ckpt sync {gen_name(n)}: {mb:.0f} MB in {dt:.0f} s ({mb / max(dt, 1e-6):.2f} MB/s), ckpt epoch {info.get('ckpt_epoch')}, "
                      f"high-water {info.get('action')}, pruned {removed}")
                if self.every > 0 and dt > self.every:
                    msg = (f"ckpt sync {gen_name(n)} took {dt / 60:.1f} min > ckpt_sync_min {self.every / 60:.0f} "
                           f"({mb / max(dt, 1e-6):.2f} MB/s via {'dbutils.fs.cp' if copier else 'shutil'}): the resume state lags training by more than a tick")
                    print(f"  ⚠ {msg}"); self.slow_ticks.append(msg)
                return gen_name(n)
        def _run(self):
            while not self._stop.wait(self.every):
                try: self.sync_once(copier=self.copier)
                except CopierError as e:
                    print(f"  ⚠ ckpt sync failed (retry next tick): {e}"); self.log.append({"error": str(e)[:200], "kind": "copier"})
                    if self.copier is not None:            # the dbutils COPY failed in this thread: later ticks use shutil (slower, correct)
                        self.copier = None
                        self.fallbacks.append(f"ckpt sync daemon: dbutils.fs.cp failed in the thread ({str(e)[:120]}); later ticks used shutil over FUSE")
                        print(f"  ⚠ {self.fallbacks[-1]}")
                except Exception as e:                     # inconsistent/unstable snapshot set etc.: skip this tick, copier unchanged
                    print(f"  ⚠ ckpt sync tick skipped (retry next tick, copier unchanged): {e}"); self.log.append({"error": str(e)[:200], "kind": "tick"})
        def stop(self):
            self._stop.set()
            if hasattr(self, "th"):
                t0 = time.time(); self.th.join()                  # NO timeout: a tick in flight must finish before anyone else writes
                print(f"  ckpt sync thread joined after {time.time() - t0:.0f} s")

    KW = train_kwargs(DS_ROOT, OUT_DIR, EPOCHS, BS, GA, resume=RESUME_PATH)
    if {k: KW[k] for k in RESUME_BOUND_HPARAMS} != HP_BOUND: raise RuntimeError("train kwargs drifted from the resume-bound hyper-parameters")
    if replacement_sampler_engaged(N_TRAIN, BS, GA):
        print(f"⚠ replacement sampler engaged: {N_TRAIN} < 5 x {BS * GA}"); SILENT_FALLBACKS.append("replacement sampler")
    model = build_model()
    torch.cuda.reset_peak_memory_stats(); gc.collect(); torch.cuda.empty_cache()
    syncer = CkptSync(OUT_DIR, RS_VOL, CKPT_SYNC_MIN, {"provenance": PROVENANCE, "train_hparams": HP_BOUND, "mode": MODE,
                                                       "resumed_from_generation": (RESUMED or {}).get("generation")}, copier=put).start()
    t0 = time.perf_counter()
    try:
        model.train(**KW)
    except Exception as _train_err:
        # train() raised (worker OOM kill, CUDA error, NaN...): /local_disk0 dies with the job cluster, so publish the newest local
        # set NOW. snapshot_set/resume_consistency still refuse an inconsistent crash-time set; the original exception propagates.
        syncer.stop()
        try:
            _g = syncer.sync_once(copier=put)
            print(f"⚠ train() raised {type(_train_err).__name__}; crash-time resume generation published: {_g}")
        except Exception as _e2:
            print(f"⚠ train() raised {type(_train_err).__name__} and the crash-time resume generation was NOT published: {_e2}")
            SILENT_FALLBACKS.append(f"crash-time resume generation not published: {str(_e2)[:120]}")
        raise
    finally:
        syncer.stop()                                            # joins the daemon fully (idempotent); the lock makes the call below the only writer
    HOURS_ACTUAL = (time.perf_counter() - t0) / 3600
    PEAK_VRAM = torch.cuda.max_memory_allocated() / 1e9
    for _fb in syncer.fallbacks + syncer.slow_ticks + syncer.prune_errors: SILENT_FALLBACKS.append(_fb)
    FINAL_GEN = None
    try: FINAL_GEN = syncer.sync_once(copier=put)                # final generation via dbutils.fs.cp
    except Exception as e: print(f"⚠ final ckpt sync failed: {e}"); SILENT_FALLBACKS.append(f"final resume generation not published: {str(e)[:120]}")
    print(f"train() returned after {HOURS_ACTUAL:.2f} h (re-projected {HOURS_REPROJ:.2f}); peak VRAM {PEAK_VRAM:.1f} GB; final generation {FINAL_GEN}")

    tc = read_training_config(OUT_DIR)
    PREC = resolve_precision_replica(tc["model_config"].get("amp", True), tc["train_config"].get("amp_dtype"), True, torch.cuda.is_bf16_supported())
    if PREC != "bf16-mixed" or tc["train_config"].get("amp_dtype") != "bf16":
        raise RuntimeError(f"resolved precision {PREC} / amp_dtype {tc['train_config'].get('amp_dtype')} — not bf16-mixed")
    AUG_BACKEND = tc["train_config"].get("augmentation_backend")
    MS = summarize_metrics_csv(read_metrics_csv(osp.join(OUT_DIR, "metrics.csv")))
    best, BEST_PATH = reload_best(OUT_DIR)
    CKM = read_ckpt_meta(BEST_PATH)                              # strip_checkpoint drops 'epoch' (state_dict.py:21-35): epoch comes from the EMA file
    EMA_META = read_ckpt_meta(osp.join(OUT_DIR, "checkpoint_best_ema.pth")) if osp.exists(osp.join(OUT_DIR, "checkpoint_best_ema.pth")) else {}
    BEST_EPOCH = EMA_META.get("epoch") if EMA_META.get("epoch") is not None else MS["best_epoch"]
    # The selected checkpoint must be the metrics.csv best epoch (SegmentationTrainConfig: skip_best_epochs 0, smooth_alpha 0,
    # eval_base_model False -> the primary key mirrors the EMA track, config.py:1076-1090). A mismatch is the on_fit_end
    # backfill ("EMA metric never improved; saved final EMA weights", best_model.py:606-617) replacing the true best.
    if CKM.get("best_total_source") == "ema" and EMA_META.get("epoch") is not None and MS["best_epoch"] is not None \
            and int(EMA_META["epoch"]) != int(MS["best_epoch"]):
        why = (f"on a resumed run ({RESUMED.get('generation')}: staged last.ckpt epoch {RESUMED.get('ckpt_epoch')}, EMA mark {RESUMED.get('best_ema_in_ckpt')}, "
               f"high-water action {RESUMED.get('action')}) this means the staged triple was inconsistent or a later epoch overwrote the best EMA checkpoint"
               if RESUMED is not None else
               "on a fresh run this is rfdetr's on_fit_end backfill (\"EMA metric never improved; saved final EMA weights\", best_model.py:606-617) "
               "replacing the true best")
        raise RuntimeError(f"checkpoint_best_total.pth (EMA, epoch {EMA_META['epoch']}) is not the metrics.csv best epoch {MS['best_epoch']} "
                           f"(val {MS['best_val_segm_mAP_50_95']}): {why} — do not evaluate it; re-run mode=full")
    print(f"selected checkpoint: source {CKM.get('best_total_source')}, epoch {BEST_EPOCH}, metrics.csv best {MS['best_val_segm_mAP_50_95']} @ {MS['best_epoch']}")
    # one random full-frame train image => >= 3 kept at 0.5; all outputs finite
    im0 = random.Random(SEED).choice([i for i in TRAIN_COCO["images"] if not i.get("crop_variant", False)])
    rows0 = rf_predict_rows(best, decode_train_image("train", im0["file_name"]), SWEEP_PROBE_THR)
    if not all(np.isfinite(s) for _, s in rows0): raise RuntimeError("non-finite scores from the reloaded model")
    k0 = apply_threshold_and_chain(rows0, 0.5)
    if len(k0) < 3: raise RuntimeError(f"reloaded model keeps only {len(k0)} masks at 0.5 on train image {im0['file_name']} (need >= 3)")

    # ---- operating threshold on valid (training days only; a tuning constant, never a claim) ----
    lm = valid_label_masks(VALID_COCO); ACC, ACC_N = {}, {}; t0 = time.time()
    for j, im in enumerate(VALID_COCO["images"]):
        rgb = decode_train_image("valid", im["file_name"])
        rows = rf_predict_rows(best, rgb, SWEEP_PROBE_THR)
        labels = [coco_seg_to_mask(a["segmentation"]) for a in lm[im["id"]]]
        sweep_accumulate(ACC, rows, labels)
        if im.get("phase4") == "night": sweep_accumulate(ACC_N, rows, labels)
        if (j + 1) % 100 == 0: print(f"  sweep {j + 1}/{N_VALID} ({(time.time() - t0) / 60:.1f} min)")
    OP_TAU, F1 = op_threshold_from_sweep({t: tuple(v) for t, v in ACC.items()})
    OP_TAU_N, F1_N = (op_threshold_from_sweep({t: tuple(v) for t, v in ACC_N.items()}) if ACC_N else (None, {}))
    print(f"op_threshold = {OP_TAU} (F1 {F1[OP_TAU]:.4f}; ties -> lower) | night-only argmax {OP_TAU_N} (descriptive)")
    print("f1_by_threshold:", {t: round(v, 4) for t, v in F1.items()})
    render_valid_preds(best, VALID_COCO, osp.join(FIG_VOL, "valid_preds_best.png"), thr=OP_TAU, seed=SEED)

    # ---- copy out (main thread, dbutils.fs.cp): the final files, flat under CK_VOL; last.ckpt lives only in resume_state/g*/ ----
    COPIED = {}
    for fn in (BEST_CKPT_FILE, "checkpoint_best_ema.pth", "last_ema.pth", "training_config.json", "metrics.csv"):
        p = osp.join(OUT_DIR, fn)
        if osp.exists(p):
            put(p, osp.join(CK_VOL, fn)); COPIED[fn] = sha256_file(p); print(f"  copied {fn} (sha256 {COPIED[fn][:12]})")
        elif fn == BEST_CKPT_FILE:
            raise RuntimeError(f"{p} vanished after reload — Tol_44 has nothing to load")
        else: print(f"  ⚠ {fn} not produced (best_model.py:565-644)"); SILENT_FALLBACKS.append(f"{fn} absent")
    with open(osp.join(CK_VOL, "manifest_copy.json"), "wb") as f: f.write(MAN_BYTES)
    BEST_TOTAL_SHA = COPIED[BEST_CKPT_FILE]                      # Tol_44 asserts the file it loads is the one this summary describes
    if sha256_file(osp.join(CK_VOL, BEST_CKPT_FILE)) != BEST_TOTAL_SHA:
        raise RuntimeError("checkpoint_best_total.pth on the Volume does not hash to the local file — copy cut; re-run the copy-out")
    print(f"checkpoint_best_total.pth sha256 {BEST_TOTAL_SHA[:16]} (verified on the Volume)")
    EVAL_V2 = eval_v2_contract(OP_TAU, RES, VARIANT, SOURCE_DAYS, MAN["forbidden_eval_days"])   # what Tol_44's rfdetr arm reads
    OUT = {"run_tag": RUN_TAG, "dataset_tag": DATASET_TAG, "label_source": LABEL_SOURCE, "dataset_manifest_sha256": MAN_SHA,
           "source_days": SOURCE_DAYS, "allow_test_day_reason": ALLOW_TEST_DAY_REASON, "provenance": PROVENANCE, "constants": MAN["constants"],
           "manifest_sealed": bool(MAN["sealed"]), "seal_waiver": SEAL_WAIVER,   # Tol_44 seal_admissible() reads both (unsealed => ['below_floor'])
           "train_clips": TRAIN_CLIPS, "valid_clips": VALID_CLIPS,
           "valid_phase4_counts": VALID_PHASE4, "allow_thin_night_valid": ALLOW_THIN_NIGHT_VALID, "valid_buffer_min_gap_min": VALID_BUFFER_MIN_GAP,
           "gates": GATES_REC,                                   # collapse_overrides + G9 counts, copied from the manifest
           "band": {k: MAN_PROV[k] for k in ("frag_lo", "frag_lo_provenance", "over_hi", "valid_buffer_min")},
           "teacher_detect": TEACHER_DETECT, "f9_notes": F9_NOTES, "human_gate": HUMAN_GATE_REC,
           "export_balance": EXPORT_BALANCE, "train_strides": TRAIN_STRIDES,  # hour cap / quotas outcome; the lattice strides (densify-first check)
           "train_frameset": TRAIN_FRAMESET,                     # a later DET-C must share this (file_names_sha256 / image_ids)
           "checkpoint_best_total_sha256": BEST_TOTAL_SHA, "copied_sha256": COPIED, "force": FORCE,
           "resume_state_dir": RS_VOL, "final_generation": FINAL_GEN,
           **EVAL_V2,                                            # op_threshold, resolution, variant, post_chain, reload, source/forbidden days
           "variant": VARIANT, "resolution": RES, "effective_batch": BS * GA, "batch_size": BS, "grad_accum": GA,
           "epochs": EPOCHS, "epochs_run": MS["epochs_run"], "early_stopped": bool(MS["epochs_run"] and MS["epochs_run"] < EPOCHS),
           "best_epoch": BEST_EPOCH, "best_total_source": CKM.get("best_total_source"), "best_ema_ckpt_meta": EMA_META,
           "best_val_segm_mAP_50_95": MS["best_val_segm_mAP_50_95"], "best_val_segm_mAP_50": MS["best_val_segm_mAP_50"],
           "valid_map_label": "model selection only — tracker-derived labels, not evidence (DATA_SPLITS F3)",
           "ips": SMK["ips"], "t_val_img": SMK["t_val_img"], "hours_projected_smoke_record": SMK["hours_full"], "smoke_epochs_recorded": SMK.get("epochs"),
           "hours_projected_recomputed": round(HOURS_REPROJ, 3), "hours_actual": round(HOURS_ACTUAL, 3), "peak_vram_gb": round(PEAK_VRAM, 2),
           "gpu_name": torch.cuda.get_device_name(0), "precision_resolved": PREC, "scales_resolved": SCALES_RESOLVED, "scale_jitter": SCALE_JITTER,
           "eval_masks_head_resolution": EVAL_HEAD_RES,
           "augmentation_backend_resolved": AUG_BACKEND, "op_threshold": OP_TAU, "op_threshold_night": OP_TAU_N,
           "f1_by_threshold": {str(t): round(v, 5) for t, v in F1.items()}, "f1_by_threshold_night": {str(t): round(v, 5) for t, v in F1_N.items()},
           "sweep_counts": {str(t): v for t, v in ACC.items()}, "n_valid_swept": N_VALID, "mode": MODE, "resumed_from": RESUME_PATH, "resumed_state": RESUMED,
           "ckpt_sync_log": syncer.log, "pretrained_md5": PRETRAINED_MD5, "pretrained_source": PRETRAINED_SOURCE, "coexist": COEXIST,
           "seed": SEED, "versions": VERSIONS, "staging": T_STAGE, "silent_fallbacks": SILENT_FALLBACKS, "train_kwargs": {k: v for k, v in KW.items() if k != "notes"},
           "note": "valid mAP is agreement with teacher-derived labels on training days; it is not detector quality (DATA_SPLITS F3).",
           "created": datetime.utcnow().isoformat() + "Z"}
    train_summary_contract_check(OUT)                            # every key Tol_44 read_train_summary() + the rfdetr arm read
    for p in (MET_FULL, osp.join(CK_VOL, SUMMARY_FILE)):        # sequential "w" JSON: no append/seek on the Volume
        with open(p, "w") as f: json.dump(OUT, f, indent=1)
    with open(osp.join(CK_VOL, SUMMARY_FILE)) as f: _ts_back = json.load(f)
    train_summary_contract_check(_ts_back)                       # the file Tol_44 will read, re-read from the Volume
    if _ts_back["checkpoint_best_total_sha256"] != sha256_file(osp.join(CK_VOL, BEST_CKPT_FILE)):
        raise RuntimeError(f"{SUMMARY_FILE} checkpoint_best_total_sha256 does not hash {BEST_CKPT_FILE} on the Volume — Tol_44 would refuse it")
    print(f"written {MET_FULL} and {CK_VOL}/{SUMMARY_FILE} (re-read and contract-checked)")
    print(f"for Tol_44: checkpoint {osp.join(CK_VOL, BEST_CKPT_FILE)} | op_threshold {OP_TAU} | "
          f"reload {EVAL_V2['reload']['call']} | eval days must avoid {MAN['forbidden_eval_days']}")
    print("valid mAP is agreement with teacher-derived labels on training days; it is not detector quality.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading it
# MAGIC
# MAGIC **The smoke JSON decides whether the full run may start.** `hours_full` is the projection
# MAGIC `epochs × (N_train / ips + N_valid × t_val_img) × 1.15`; if it exceeds `max_hours` the notebook has
# MAGIC already raised with the two cheapest fallbacks (`epochs 40→25`, `resolution 624→552`). Neither is applied
# MAGIC for you. `ips` is images per second of *training* (setup and validation subtracted — check
# MAGIC `T_setup_source` and `T_val_source`; "assumed" means the timing rests on a stated guess, not a measurement).
# MAGIC `valid_preds_smoke.png` is the eyeball check: after two epochs the outlines should already sit on calves.
# MAGIC A blank or garbage sheet with a positive mAP means the labels are read but the pipeline is wrong.
# MAGIC
# MAGIC **`op_threshold` is a tuning constant, not a result.** It is the score threshold that maximised instance
# MAGIC F1 against the tracked `valid` labels (Hungarian at mask IoU ≥ 0.5, after the post-chain), with ties
# MAGIC broken towards the lower value. `Tol_44` reads it from `train_summary.json` and never re-tunes it; the
# MAGIC `post_chain` and `reload` blocks next to it say exactly how the detector must be run so the replacement check sees
# MAGIC the same detector this sweep chose (score > τ, background rows dropped, ≥ 8,000 px, largest first, IoS/IoU dedupe
# MAGIC < 0.50, top 6; `from_checkpoint(..., resolution=<resolution>)` — the stripped best checkpoint has lost its
# MAGIC resolution, so an unpinned reload would silently run at 384). `op_threshold_night` is the same argmax on night
# MAGIC frames only — descriptive, to see whether night wants a different operating point.
# MAGIC
# MAGIC **`best_val_segm_mAP_*` is model selection, nothing more.** The valid labels are the teacher tracker's own
# MAGIC masks on the training days, so the number measures agreement with the teacher (mimicry). Whether this
# MAGIC detector may replace the SAM3 teacher `detect()` without re-tracking is decided only by `Tol_44`: on every frame
# MAGIC where the tracker actually called the teacher on 04-22/04-23, both detectors are compared with the reviewer-verified
# MAGIC masks, and the fine-tuned one must be at least as good (misses, splits, extras, mask agreement, every seed call).
# MAGIC
# MAGIC **`silent_fallbacks` is the honesty list.** Every assumed timing, OOM step-down, absent output file,
# MAGIC replacement sampler, downloaded (rather than pre-staged) pretrained file, extra training day, optional human gate,
# MAGIC export hour-cap breach or unmet quota (`export_balance`), 04-22 trained without the densify-first order
# MAGIC (`allow_undensified_extra_day`) or resume fallback lands there. An empty list means the run did exactly what the widgets said.
# MAGIC
# MAGIC **`source_days` / `forbidden_eval_days` are the leakage guard.** `Tol_44` refuses to evaluate on any day listed
# MAGIC there. If 04-22 was added to training (`allow_extra_days=true`), the replacement check runs on 04-23 only.
# MAGIC
# MAGIC **`human_gate`** records whether the export required the per-cell review (`required`) and the status it found.
# MAGIC With `required: false` the reviewed fault list was the human gate (decision 2026-09-23); a `failed` status would
# MAGIC have stopped this notebook either way.
# MAGIC
# MAGIC **Checkpoints.** `checkpoint_best_total.pth` is the selected (EMA, `best_total_source`) model and its sha256 is in
# MAGIC the summary. The resume state lives in `resume_state/g<NNNNNN>/`: each background tick writes a NEW generation
# MAGIC (`last.ckpt`, `checkpoint_best_ema.pth`, `metrics.csv`, sidecars, then `COMPLETE.json`), only after the three files
# MAGIC were checked to agree with each other; the newest two complete generations are kept. `mode=resume` takes the newest
# MAGIC generation that verifies (sha256 against `COMPLETE.json` and the sidecars, zip integrity, EMA high-water mark vs
# MAGIC `metrics.csv` vs the EMA file's epoch), falls back to the older one if it does not, and refuses a generation written
# MAGIC under different provenance or hyper-parameters. A missing `checkpoint_best_ema.pth` while `last.ckpt` carries an
# MAGIC EMA high-water mark means the best model was lost (rfdetr would otherwise backfill the final EMA weights as "best")
# MAGIC and the generation is refused. After every run the notebook asserts that the selected EMA checkpoint's epoch
# MAGIC equals the `metrics.csv` best epoch. `mode=full` / `mode=resume` refuse a smoke record whose variant / resolution /
# MAGIC counts / batch / accum / jitter / eval-head settings differ from the current widgets, and re-project the hours
# MAGIC with the current `epochs` (`hours_projected_recomputed`).

# COMMAND ----------

# MAGIC %md ## SELFTEST — pure python on synthetic data; runs unconditionally, raises on any failure

# COMMAND ----------

def _selftest():
    import tempfile
    rng = np.random.default_rng(0)
    # --- RLE round trip on synthetic 64x48 masks ---
    for _ in range(20):
        m = rng.random((48, 64)) < 0.3
        assert np.array_equal(rle_decode(rle_encode(m)), m)
        assert coco_seg_to_mask(rle_encode(m)).shape == (48, 64)
    try:
        from pycocotools import mask as maskUtils
        for _ in range(20):
            m = rng.random((48, 64)) < 0.3
            enc = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
            enc["counts"] = enc["counts"].decode("ascii")
            assert np.array_equal(coco_seg_to_mask(enc), m), "compressed RLE decode mismatch"
            fr = maskUtils.frPyObjects(rle_encode(m), 48, 64)
            assert np.array_equal(maskUtils.decode(fr).astype(bool), m), "project RLE is not COCO uncompressed RLE"
        print("  pycocotools round-trip: 20/20 compressed + 20/20 frPyObjects exact")
    except ImportError:
        print("SELFTEST: pycocotools unavailable — skipping the COCO round-trip only")

    # --- train kwargs whitelist: a planted unknown key fails locally ---
    good = dict(dataset_dir="/x", output_dir="/y", epochs=2, batch_size=8, grad_accum_steps=2, amp_dtype="bf16",
                lr=1e-4, lr_encoder=1.5e-5, scale_jitter=False, augmentation_backend="torchvision", device="cuda",
                resolution=624, seed=0, mask_ce_loss_coef=5.0, class_names=["calf"], notes="{}")
    assert check_train_kwargs(good)
    for bad, msg in (({"patience": 8}, "patience"), ({"num_classes": 1}, "constructor-only"), ({"precision": "bf16"}, "precision")):
        try: check_train_kwargs({**good, **bad}); raise AssertionError(f"planted {bad} passed")
        except RuntimeError as e: assert msg in str(e), str(e)
    try: check_train_kwargs({**good, "batch_size": "auto"}); raise AssertionError("auto batch passed")
    except RuntimeError: pass
    try: check_train_kwargs({**good, "amp_dtype": "fp16"}); raise AssertionError("fp16 passed")
    except RuntimeError: pass
    assert multi_scale_scales(624, False, 12, 2) == [552, 576, 600, 624, 648, 672, 696, 720]
    assert multi_scale_scales(624, True, 12, 2) == [504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744]
    assert resolve_precision_replica(True, "bf16", True, True) == "bf16-mixed"
    assert resolve_precision_replica(True, "bf16", True, False) == "16-mixed"
    assert resolve_precision_replica(False, "bf16", True, True) == "32-true"

    # --- projection arithmetic incl. the gate (planted numbers) ---
    pr = project_hours(T_total=400.0, T_val=50.0, T_setup=60.0, smoke_train_n=1000, smoke_valid_n=200,
                       N_train=5500, N_valid=600, epochs=40)
    assert abs(pr["ips"] - 2000 / 240) < 1e-9, pr
    exp_h = 40 * (5500 / (2000 / 240) + 600 * 0.25) * 1.15 / 3600
    assert abs(pr["hours_full"] - exp_h) < 1e-9, (pr["hours_full"], exp_h)
    assert stop_gate(pr["hours_full"], 20.0, 40, 624)
    try: stop_gate(pr["hours_full"], 5.0, 40, 624); raise AssertionError("gate did not fire")
    except RuntimeError as e: assert "epochs 40 -> 25" in str(e) and "624 -> 552" in str(e), str(e)
    try: project_hours(100.0, 60.0, 60.0, 1000, 200, 5500, 600, 40); raise AssertionError("bad denominator passed")
    except RuntimeError: pass

    # --- sha256 sidecar resume logic on a temp file ---
    with tempfile.TemporaryDirectory() as td:
        ck = osp.join(td, "last.ckpt"); sc = ck + ".sha256"
        with open(ck, "wb") as f: f.write(os.urandom(4096))
        sha = write_sidecar(ck, sc); assert verify_sidecar(ck, sc) == sha
        with open(ck, "ab") as f: f.write(b"partial")
        try: verify_sidecar(ck, sc); raise AssertionError("corrupt ckpt passed")
        except RuntimeError as e: assert "partial checkpoint" in str(e)
        try: verify_sidecar(ck, osp.join(td, "none.sha256")); raise AssertionError("missing sidecar passed")
        except RuntimeError: pass

    # --- manifest guards (required keys; human-gate rule; test-day and pooled-constant provenance; 04-22 admission) ---
    NOT_FROM = [f"202404{d}" for d in range(22, 29)]
    def const(v, df=("20240418", "20240419"), nf=NOT_FROM): return {"value": v, "provenance": "test", "derived_from": list(df), "not_from": list(nf)}
    # constants exactly as Tol_42's manifest cell names them (Tol_42:2138-2162): A_ref[d]/ACTIVE_THR[d] derived_from=[d]; band ones pooled;
    # widget literals (valid_buffer_min, DISJ_IOU, ...) derived_from=[]
    def tol42_consts(days, pooled_df=("20240418", "20240419"), prov="derived_P1_of_U", fl=(0.45, 0.45), strides=None):
        c = {f"A_ref[{d}]": const(80000, df=(d,)) for d in days}
        # widget literal as Tol_42 writes it; a 04-22 export is dense by default (densify first)
        sd = strides or ((8.0, 4.0) if "20240422" in days else (10.0, 5.0))
        c["strides"] = const({"day": sd[0], "other": sd[1], "valid_day": 20.0, "valid_other": 10.0}, df=())
        c.update({f"ACTIVE_THR[{d}]": const(3.1, df=(d,)) for d in days})
        c.update({"frag_lo[dayl]": const(fl[0], df=pooled_df), "frag_lo[dark]": const(fl[1], df=pooled_df), "frag_lo_provenance": const(prov, df=pooled_df),
                  "S_lo[dayl]": const(0.6, df=pooled_df), "S_lo[dark]": const(0.4, df=pooled_df), "S_hi[dayl]": const(1.2, df=pooled_df),
                  "S_hi[dark]": const(1.25, df=pooled_df), "STAB_LO": const(0.9, df=pooled_df), "STAB_HI": const(1.1, df=pooled_df),
                  "body_len_px": const(310, df=pooled_df), "over_hi": const(1.32), "DISJ_IOU": const(0.02, df=()), "valid_buffer_min": const(10.0, df=())})
        return c
    D2 = ["20240418", "20240419"]
    GATES_OK = {"funnel": {"20240418": {"night": {"G6": [10, 1], "G9": [9, 0]}, "day": {"G6": [5, 0]}}},
                "curled_or_clipped_blocks": [{"block": "c:night"}], "blocks_failing_per_slot_0.75": 1, "collapse_overrides": [],
                "furniture_flagged": 0, "upper_edge_clips_admitted_mass": False, "bad_overlap_high": False}
    def man(**o):
        base = {"tag": "segv1", "sources": [{"root": "Tracking_train", "day": "20240418"}], "source_days": ["20240418", "20240419"],
                "forbidden_eval_days": ["20240418", "20240419"], "sealed": True, "smoke": False, "allow_test_day_reason": "",
                "counts": {"train_full": 5200, "train_crops": 1300, "valid": 600, "below_floor": False},
                "human_gate": {"status": "passed", "n_counted": 300, "n_bad_by_kind": {"partial": 2, "merged": 1}, "reject_rate": 0.01},
                "label_sources_available": ["tracked", "teacher_detect", "tracked_intersection"],
                "sha256": {"images_tar": "x", "annotations": {}}, "sam3_veto": {"ran": True}, "teacher_detect": {"ran": True},
                "constants": tol42_consts(D2),
                "band_json": {"path": "p", "sha256": "b" * 64, "derived_from": ["20240418", "20240419"], "over_hi": 1.32},
                "faults_json": {"path": "f", "sha256": "f" * 64, "n_faults": 26},
                "split": {"valid_clips": ["c_v1", "c_v2"], "train_clips": ["c_t1", "c_t2", "c_t3"], "valid_max": 600},
                "gates": json.loads(json.dumps(GATES_OK))}
        base.update(o); return base
    assert manifest_guards(man(), "tracked", False, False) == ["20240418", "20240419"]
    def must_raise(m, ls="tracked", extra=False, floor=False, needle=""):
        try: manifest_guards(m, ls, extra, floor); raise AssertionError(f"guard passed: {needle}")
        except RuntimeError as e: assert needle in str(e), (needle, str(e))
    below = {"train_full": 4000, "train_crops": 0, "valid": 600, "below_floor": True}
    must_raise(man(sealed=False, counts=below), needle="below_floor")
    assert manifest_guards(man(sealed=False, counts=below), "tracked", False, True)
    must_raise(man(sealed=False), needle="cannot see")                              # unsealed although every term is met: still refused
    # allow_below_floor must NOT also waive the veto term: only Tol_42's own require_veto=false makes a skipped veto legal
    must_raise(man(sealed=False, counts=below, sam3_veto={"ran": False}), floor=True, needle="sam3_veto.ran=false")
    assert manifest_guards(man(sealed=False, counts=below, sam3_veto={"ran": False}, require_veto=False), "tracked", False, True)
    assert manifest_guards(man(sam3_veto={"ran": False}, require_veto=False), "tracked", False, False)   # sealed deliberate skip: accepted, noted by the cell
    unmet = seal_terms_unmet(man(sealed=False, counts=below, sam3_veto={"ran": False}, smoke=True), False)
    assert len(unmet) == 3 and any("below_floor" in u for u in unmet) and any("smoke" in u for u in unmet), unmet
    assert seal_terms_unmet(man(), False) == []
    assert len(seal_terms_unmet(man(sealed=False, counts=below), True)) == 0                  # below_floor is the only waivable term

    # --- human gate NOT required (decision 2026-09-23): the reviewed fault list is the gate; only 'failed' refuses ---
    def hg_opt(status, **kw):
        d = {"required": False, "status": status, "n_counted": 0}; d.update(kw); return d
    for st in ("pending", "passed", None):
        assert manifest_guards(man(human_gate=hg_opt(st)), "tracked", False, False) == ["20240418", "20240419"], st
    assert manifest_guards(man(human_gate=hg_opt("pending", present=True, n_bad_by_kind={"merged": 3}, n_bad_counted=3, n_wrong_object=0, n_counted=40)),
                           "tracked", False, False) == D2                                                     # a present, small, clean review: accepted
    # review 2026-09-23: with required=false the failed condition is still RE-DERIVED from the counts of a present gate ...
    must_raise(man(human_gate=hg_opt("passed", present=True, n_bad_by_kind={"merged": 9}, reject_rate=0.03, n_counted=300)), needle="gate FAILED")
    must_raise(man(human_gate=hg_opt("passed", present=True, n_bad_by_kind={"merged": 1}, n_bad_counted=7, n_wrong_object=1, n_counted=300)), needle="gate FAILED")
    must_raise(man(human_gate=hg_opt("pending", present=True, n_bad_by_kind={"wrong_object": 1}, n_bad_counted=0, n_counted=300)), needle="wrong_object=1")
    must_raise(man(human_gate=hg_opt("passed", present=True, n_bad_by_kind={}, n_counted=300)), needle="lacks both n_bad_counted and reject_rate")
    must_raise(man(human_gate=hg_opt("pending", present="yes")), needle="not a bool")
    # ... and the fault-list waiver never covers a day without a fault list (04-22 from Tracking_teacher)
    must_raise(man(human_gate=hg_opt("pending", unlisted_days=["20240422"])), needle="unlisted_days")
    must_raise(man(source_days=["20240418", "20240419", "20240422"], forbidden_eval_days=["20240418", "20240419", "20240422"],
                   allow_test_day_reason="r", constants=tol42_consts(["20240418", "20240419", "20240422"]),
                   human_gate=hg_opt("pending", unlisted_days=["20240422"])), extra=True, needle="unlisted_days")
    # --- densify (8 s / 4 s) BEFORE 04-22 may train; skipping the order needs an explicit widget and is recorded ---
    D22s = ["20240418", "20240419", "20240422"]
    m22u = man(source_days=D22s, forbidden_eval_days=D22s, allow_test_day_reason="r", constants=tol42_consts(D22s, strides=(10.0, 5.0)))
    must_raise(m22u, extra=True, needle="DENSIFY FIRST")
    try: manifest_guards(m22u, "tracked", True, False, allow_undensified_extra_day=False); raise AssertionError("undensified 04-22 passed")
    except RuntimeError as e: assert "allow_undensified_extra_day=true" in str(e), str(e)
    assert manifest_guards(m22u, "tracked", True, False, allow_undensified_extra_day=True) == D22s
    st_, nts_ = extra_day_strides_check(m22u, True); assert st_["day"] == 10.0 and len(nts_) == 1 and "skipped explicitly" in nts_[0], nts_
    m22d = man(source_days=D22s, forbidden_eval_days=D22s, allow_test_day_reason="r", constants=tol42_consts(D22s))
    assert manifest_guards(m22d, "tracked", True, False) == D22s and extra_day_strides_check(m22d) == (m22d["constants"]["strides"]["value"], [])
    m22n = man(source_days=D22s, forbidden_eval_days=D22s, allow_test_day_reason="r", constants={k: v for k, v in tol42_consts(D22s).items() if k != "strides"})
    must_raise(m22n, extra=True, needle="does not record the training strides")
    assert extra_day_strides_check(man()) == ({"day": 10.0, "other": 5.0, "valid_day": 20.0, "valid_other": 10.0}, [])   # no 04-22: nothing to enforce
    # --- export balance (hour cap / quotas): recorded, violations become notes, never a refusal ---
    mb = man(counts={"train_full": 5200, "train_crops": 1300, "valid": 600, "below_floor": False, "distinct_10min_blocks": 180,
                     "hour_share_max": 0.079, "hour_share_max_at_gate": 0.08, "hour_cap_exceeded_after_drops": False},
             quotas={"targets": {"night": 0.35}, "achieved": {"night": 0.37}, "shortfall": {}},
             constants={**tol42_consts(D2), "hour_cap": const(0.08, df=())})
    rec_, nts_ = export_balance_check(mb); assert nts_ == [] and rec_["distinct_10min_blocks"] == 180 and rec_["hour_cap"] == 0.08, (rec_, nts_)
    mb["counts"].update({"hour_share_max": 0.093, "hour_cap_exceeded_after_drops": True})
    mb["quotas"]["shortfall"] = {"dusk": {"have": 0.02, "need": 0.05}, "dawn": 0}
    rec_, nts_ = export_balance_check(mb)
    assert len(nts_) == 2 and "HOUR CAP exceeded" in nts_[0] and "QUOTA UNMET" in nts_[1] and list(rec_["quota_unmet"]) == ["dusk"], (rec_, nts_)
    mb["counts"].pop("hour_cap_exceeded_after_drops")                                          # flag absent: re-derived from share vs cap
    assert "HOUR CAP exceeded" in export_balance_check(mb)[1][0]
    mb["counts"].pop("hour_share_max"); assert "cannot be re-checked" in export_balance_check(mb)[1][0]
    for bad_m, needle in ((man(counts={**man()["counts"], "distinct_10min_blocks": 5}, quotas={"targets": {}}), "lacks achieved/shortfall"),(man(quotas={"achieved": {}, "shortfall": {}}), "distinct_10min_blocks")):
        try: export_balance_check(bad_m); raise AssertionError(f"balance check passed: {needle}")
        except RuntimeError as e: assert needle in str(e), (needle, str(e))
    assert manifest_guards(man(human_gate=hg_opt("pending", present=False, n_bad_counted=0, n_wrong_object=0, unlisted_days=[])), "tracked", False, False)
    must_raise(man(human_gate=hg_opt("failed")), needle="failed")
    must_raise(man(sealed=False, counts=below, human_gate=hg_opt("failed")), floor=True, needle="failed")
    assert seal_terms_unmet(man(human_gate=hg_opt("pending")), False) == []
    assert any("failed" in u for u in seal_terms_unmet(man(human_gate=hg_opt("failed")), False))
    assert manifest_guards(man(sealed=False, counts=below, human_gate=hg_opt("pending")), "tracked", False, True)   # below floor waived, gate optional
    must_raise(man(human_gate={"required": "false", "status": "pending"}), needle="not a bool")    # a string must not read as truthy
    assert human_gate_required(man()) is True and human_gate_required(man(human_gate=hg_opt("pending"))) is False
    # required=true keeps the full seal-rule re-check (status alone is not enough)
    must_raise(man(human_gate={"required": True, "status": "pending", "n_counted": 300, "n_bad_by_kind": {}}), needle="need 'passed'")
    assert manifest_guards(man(human_gate={"required": True, "status": "passed", "n_counted": 300, "n_bad_by_kind": {}, "reject_rate": 0.0}), "tracked", False, False)
    must_raise(man(source_days=["20240418", "20240419", "20240423"], forbidden_eval_days=["20240418", "20240419", "20240423"]), needle="detector TEST days")
    must_raise(man(source_days=["20240418", "20240419", "20240423"], forbidden_eval_days=["20240418", "20240419", "20240423"]), extra=True, needle="detector TEST days")
    must_raise(man(source_days=["20240418", "20240610"], forbidden_eval_days=["20240418", "20240610"]), extra=True, needle="outside TRAIN_DAYS")
    m22 = man(source_days=["20240418", "20240419", "20240422"], forbidden_eval_days=["20240418", "20240419", "20240422"])
    must_raise(m22, needle="allow_extra_days")
    must_raise(m22, extra=True, needle="allow_test_day_reason")                     # 04-22 needs a NAMED reason in the manifest
    m22r = man(source_days=["20240418", "20240419", "20240422"], forbidden_eval_days=["20240418", "20240419", "20240422"],
               allow_test_day_reason="data short after gating; user decision 2026-09-22",
               constants=tol42_consts(["20240418", "20240419", "20240422"]))
    assert manifest_guards(m22r, "tracked", True, False) == ["20240418", "20240419", "20240422"]
    must_raise(man(label_sources_available=["tracked"]), ls="teacher_detect", needle="label_sources_available")
    must_raise(man(human_gate={"status": "pending", "n_counted": 300, "n_bad_by_kind": {}}), needle="human_gate")
    must_raise(man(human_gate={"status": "passed"}), needle="lacks")                # status alone is not enough
    must_raise(man(human_gate={"status": "passed", "n_counted": 300, "n_bad_by_kind": {"merged": 6, "wrong_object": 1}, "reject_rate": 0.0}), needle="wrong_object")
    must_raise(man(human_gate={"status": "passed", "n_counted": 300, "n_bad_by_kind": {"merged": 4}, "reject_rate": 0.0133}), needle="count(BAD among counted cells)=4")
    must_raise(man(human_gate={"status": "passed", "n_counted": 300, "n_bad_by_kind": {"merged": 4}}), needle="lacks both n_bad_counted and reject_rate")
    must_raise(man(human_gate={"status": "passed", "n_counted": 0, "n_bad_by_kind": {}, "reject_rate": 0.0}), needle="no counted cell")
    assert human_gate_check({"status": "passed", "n_counted": 300, "n_bad_by_kind": {"partial_minor": 40, "split": 3}, "reject_rate": 0.01}) == 3   # partial_minor is ok
    # the seal rule is COUNTED-only: 6 'merged' verdicts on the uncounted rejected_G*/vetoed extras with 3 counted BAD must PASS
    # (Tol_42.human_gate_eval fills n_bad_by_kind from every cell; only reject_rate = n_bad_counted/n_counted is counted-only)
    assert human_gate_check({"status": "passed", "n_counted": 300, "n_bad_by_kind": {"merged": 6}, "reject_rate": 0.01}) == 3
    assert manifest_guards(man(human_gate={"status": "passed", "n_counted": 300, "n_bad_by_kind": {"merged": 6, "split": 2}, "reject_rate": 0.01}), "tracked", False, False)
    assert counted_bad_from_gate({"n_counted": 300, "reject_rate": 0.0133}) == 4 and counted_bad_from_gate({"n_counted": 260, "reject_rate": 0.0115}) == 3
    assert counted_bad_from_gate({"n_counted": 300, "reject_rate": 0.9, "n_bad_counted": 2}) == 2      # an explicit n_bad_counted wins
    must_raise(man(human_gate={"status": "passed", "n_counted": 300, "n_bad_by_kind": {"merged": 1}, "reject_rate": 0.0, "n_bad_counted": 4}), needle="count(BAD among counted cells)=4")
    # sheet_n (required gate only): a per-stratum shortfall is legal (reported by human_gate_size_notes); a recorded plan < 300 raises;
    # counted cells may exceed the sheet_n WIDGET by the tail enlargement (50 per 300) but never a recorded sheet_n_planned
    def hgp(n): return {"status": "passed", "n_counted": n, "n_bad_by_kind": {}, "reject_rate": 0.0}
    assert human_gate_check(hgp(260)) == 0 and human_gate_check(hgp(260), sheet_n=(300, "sheet_n")) == 0
    assert human_gate_check(hgp(350), sheet_n=(300, "sheet_n")) == 0
    for n_c, sn, needle in ((351, (300, "sheet_n"), "inconsistent"), (320, (300, "sheet_n_planned"), "inconsistent"),
                            (200, (200, "sheet_n"), "PLANNED fewer")):
        try: human_gate_check(hgp(n_c), sheet_n=sn); raise AssertionError(f"sheet_n check passed: {n_c} {sn}")
        except RuntimeError as e: assert needle in str(e), (needle, str(e))
    notes = human_gate_size_notes(hgp(260), (300, "sheet_n")); assert len(notes) == 2 and "shortfall" in notes[0], notes
    assert len(human_gate_size_notes(hgp(300), None)) == 1 and human_gate_size_notes(hgp(300), (300, "sheet_n_planned")) == []
    assert manifest_sheet_n(man()) is None and manifest_sheet_n(man(widgets={"sheet_n": "200"})) == (200, "sheet_n")
    assert manifest_sheet_n(man(human_gate={**hgp(300), "sheet_n": 300, "sheet_n_planned": 310})) == (310, "sheet_n_planned")
    assert manifest_guards(man(human_gate={**hgp(260), "sheet_n": 300}), "tracked", False, False)
    must_raise(man(widgets={"sheet_n": 200}), needle="PLANNED fewer")
    must_raise(man(human_gate={**hgp(300), "sheet_n_planned": 280}), needle="PLANNED fewer")
    assert manifest_guards(man(human_gate={**hgp(0), "required": False, "sheet_n_planned": 10}), "tracked", False, False)   # not required: size not a term
    must_raise(man(smoke=True), needle="smoke")
    must_raise(man(forbidden_eval_days=["20240418"]), needle="forbidden_eval_days")
    must_raise(man(constants={}), needle="constants block empty")
    for k in ("constants", "band_json", "faults_json", "split", "gates"):
        m = man(); del m[k]; must_raise(m, needle="required keys")
    # a test-day-derived constant or band never enters training, allow_extra_days or not
    must_raise(man(constants={**tol42_consts(D2), "frag_lo[dark]": const(0.45, df=("20240418", "20240419", "20240422"))}), extra=True, needle="pooled constants")
    must_raise(man(band_json={"path": "p", "sha256": "b" * 64, "derived_from": ["20240422", "20240423"]}), needle="band_json.derived_from")
    must_raise(man(constants={**tol42_consts(D2), "A_ref": const(43000, nf=["20240423"])}), needle="not_from lacks")
    must_raise(man(constants={**tol42_consts(D2), "A_ref": {"value": 1}}), needle="lacks derived_from")
    prov = constants_provenance_check(man()); assert prov["band_json_sha256"] == "b" * 64 and prov["band_json_derived_from"] == ["20240418", "20240419"]
    assert prov["frag_lo"] == {"dayl": 0.45, "dark": 0.45} and prov["frag_lo_provenance"] == "derived_P1_of_U" and prov["notes"] == []
    # --- presence + values of the band constants Tol_44 reads (review 2026-09-23): the guard no longer passes a manifest that lacks them ---
    def drop(*ks):
        c = tol42_consts(D2)
        for k in ks: del c[k]
        return c
    must_raise(man(constants={"DISJ_IOU": const(0.02, df=())}), needle="lack required entries")          # the reviewer's "DISJ_IOU only" manifest
    for k in ("frag_lo[dark]", "frag_lo[dayl]", "frag_lo_provenance", "over_hi", "S_hi[dark]", "STAB_HI", "valid_buffer_min", "A_ref[20240419]"):
        must_raise(man(constants=drop(k)), needle=k)
    must_raise(man(constants={**tol42_consts(D2), "frag_lo[dark]": {**const(0.45), "value": None}}), needle="lack required entries")
    must_raise(man(constants={**tol42_consts(D2), "frag_lo[dayl]": const(0.45, df=())}), needle="derived_from is empty")
    must_raise(man(constants=tol42_consts(D2, prov="tol41_band", fl=(0.59, 0.59))), needle="frag_lo_provenance")   # Tol_41's band: ruled out (its 0.59 lower edge rejects the curled night stretch)
    must_raise(man(constants=tol42_consts(D2, fl=(0.45, 0.62))), needle="sanity band")
    must_raise(man(constants=tol42_consts(D2, fl=(0.25, 0.45))), needle="sanity band")
    must_raise(man(constants={**tol42_consts(D2), "frag_lo[dark]": const("0.4x")}), needle="not a number")
    pov = constants_provenance_check(man(constants=tol42_consts(D2, prov="user_override", fl=(0.45, 0.28))))
    assert len(pov["notes"]) == 2 and "user_override" in pov["notes"][1] and pov["frag_lo"]["dark"] == 0.28, pov
    must_raise(man(constants=tol42_consts(D2, prov="user_override", fl=(0.45, 0.61))), needle="nothing was relaxed")
    must_raise(man(constants={**tol42_consts(D2), "over_hi": const(0.40)}), needle="band is empty")
    must_raise(man(constants={**tol42_consts(D2), "valid_buffer_min": const(5.0, df=())}), needle="valid_buffer_min=5.0")
    # --- per-day constants exactly as Tol_42's manifest cell writes them (A_ref[d]/ACTIVE_THR[d]: derived_from=[d], not_from=all 7 test days) ---
    assert constants_provenance_check(man(constants=tol42_consts(["20240418", "20240419"])))
    D22 = ["20240418", "20240419", "20240422"]
    m22c = man(source_days=D22, forbidden_eval_days=D22, allow_test_day_reason="data short after gating; user decision 2026-09-22",
               constants=tol42_consts(D22))
    assert constants_provenance_check(m22c, allow_extra_days=True)                            # the allow_extra_days path is reachable
    assert manifest_guards(m22c, "tracked", True, False) == D22
    must_raise(m22c, extra=False, needle="allow_extra_days")                                   # source-day guard fires first
    try: constants_provenance_check(m22c, allow_extra_days=False); raise AssertionError("per-day 04-22 constant passed without allow_extra_days")
    except RuntimeError as e: assert "A_ref[20240422]" in str(e) and "allow_extra_days=true is required" in str(e), str(e)
    # pooled constants stay TRAIN_DAYS-only even with allow_extra_days (Tol_42 excludes non-training days from U; its const()
    # default of derived_from=SOURCE_DAYS for pooled constants is a Tol_42 labelling bug this guard is meant to catch)
    must_raise(man(source_days=D22, forbidden_eval_days=D22, allow_test_day_reason="r", constants=tol42_consts(D22, pooled_df=D22)), extra=True, needle="pooled constants")
    must_raise(man(constants=tol42_consts(["20240418", "20240419", "20240423"])), extra=True, needle="not a source day")   # A_ref[20240423]: a test day
    must_raise(man(constants={**tol42_consts(D2), "A_ref[20240418]": const(80000, df=("20240418", "20240419"))}), needle="!= [20240418]")
    must_raise(man(source_days=D22, forbidden_eval_days=D22, allow_test_day_reason="r",
                   constants={**tol42_consts(D22), "A_ref[20240422]": const(80000, df=("20240422",), nf=["20240422"])}), extra=True, needle="not_from lacks test days ['20240423'")

    # --- the seal is re-derived even when sealed=true (review 2026-09-23): an inconsistent sealed manifest is refused ---
    must_raise(man(sam3_veto={"ran": False}, require_veto=True), needle="inconsistent")
    must_raise(man(sam3_veto={"ran": False}), needle="inconsistent")                                  # require_veto absent => not a deliberate skip
    must_raise(man(counts={"train_full": 4000, "train_crops": 0, "valid": 600, "below_floor": True}), floor=True, needle="inconsistent")
    must_raise(man(human_gate={"required": True, "status": "pending", "n_counted": 300, "n_bad_by_kind": {}}), needle="need 'passed'")
    must_raise(man(sealed="true"), needle="not a bool")
    # --- G9 proof in the gates block ---
    assert gates_check(GATES_OK) == {"collapse_overrides": [], "n_curled_or_clipped_blocks": 1, "blocks_failing_per_slot_0.75": 1}
    must_raise(man(gates={"funnel": {}}), needle="gates lacks")                                        # the reviewer's accepted-before case
    must_raise(man(gates={**GATES_OK, "funnel": {"20240418": {"day": {"G6": [5, 0]}}}}), needle="no G9 entry")
    must_raise(man(gates={**GATES_OK, "blocks_failing_per_slot_0.75": "3"}), needle="not a count")
    must_raise(man(gates={**GATES_OK, "collapse_overrides": "c:night"}), needle="not a list")
    assert manifest_guards(man(gates={**GATES_OK, "collapse_overrides": ["c1:night"]}), "tracked", False, False) == D2

    # --- valid must contain night and train/valid respect the G12 buffer on the images themselves ---
    vimgs = [{"phase4": "day"}] * 300 + [{"phase4": "dusk"}] * 60 + [{"phase4": "night"}] * 240
    cnt, nts = valid_phase_check(vimgs); assert cnt == {"night": 240, "dawn": 0, "dusk": 60, "day": 300, "dark": 300} and nts == []
    try: valid_phase_check([{"phase4": "day"}] * 300 + [{"phase4": "dusk"}] * 60); raise AssertionError("night-free valid passed")
    except RuntimeError as e: assert "allow_thin_night_valid" in str(e), str(e)
    cnt, nts = valid_phase_check([{"phase4": "day"}] * 300 + [{"phase4": "night"}] * 49, allow_thin=True); assert cnt["night"] == 49 and len(nts) == 1
    vb = [{"day": "20240418", "wall": "2024-04-18T20:00:00", "file_name": "v0"}, {"day": "20240418", "wall": "2024-04-18T20:40:45", "file_name": "v1"}]
    tb_ok = [{"day": "20240418", "wall": "2024-04-18T20:51:00", "file_name": "t0"}, {"day": "20240419", "wall": "2024-04-19T20:41:00", "file_name": "t1"},
             {"day": "20240418", "wall": "2024-04-18T19:49:30", "file_name": "t2", "crop_variant": True}]
    assert abs(valid_buffer_check(tb_ok, vb, 10.0) - 10.25) < 1e-9
    try: valid_buffer_check(tb_ok + [{"day": "20240418", "wall": "2024-04-18T20:40:52", "file_name": "t_dup"}], vb, 10.0); raise AssertionError("near-duplicate passed")
    except RuntimeError as e: assert "t_dup" in str(e) and "G12" in str(e), str(e)
    assert valid_buffer_check([{"day": "20240419", "wall": "2024-04-19T08:00:00"}], vb, 10.0) is None     # no same-day pair
    try: valid_buffer_check([{"day": "20240418", "file_name": "nowall"}], vb, 10.0); raise AssertionError("missing wall passed")
    except RuntimeError as e: assert "lacks 'wall'" in str(e)

    # --- mechanism control (DET-C not planned): a tracked run never depends on the teacher-detect by-product ---
    td_hi = {"ran": True, "n_frames": 6000, "n_dropped": 1320, "drop_frac": 0.22, "drop_high": True, "intersection_n": 4680}
    assert f9_control_check("tracked", {"ran": False}) == [] and f9_control_check("tracked", None) == []
    assert f9_control_check("tracked", {"ran": True, "drop_high": False, "drop_frac": 0.05}) == []
    assert f9_control_check("tracked_intersection", td_hi) == []
    assert "tracked_intersection" in f9_control_check("tracked", td_hi)[0]         # a note, not a refusal
    assert "tracked_intersection" in f9_control_check("teacher_detect", td_hi)[0]
    assert manifest_guards(man(teacher_detect=td_hi), "tracked", False, False) == ["20240418", "20240419"]
    assert manifest_guards(man(teacher_detect=td_hi), "tracked_intersection", False, False) == ["20240418", "20240419"]
    # tracked needs neither teacher_detect nor its label source: a manifest from a veto-only teacher stage still trains
    m_noteach = man(label_sources_available=["tracked"]); del m_noteach["teacher_detect"]
    assert manifest_guards(m_noteach, "tracked", False, False) == ["20240418", "20240419"]
    must_raise(m_noteach, ls="tracked_intersection", needle="label_sources_available")
    must_raise(man(teacher_detect={"ran": False}), ls="teacher_detect", needle="teacher_detect.ran")
    assert manifest_guards(man(teacher_detect={"ran": False}, label_sources_available=["tracked"]), "tracked", False, False) == D2

    # --- label_source defaults to tracked; the mechanism-control arms stay reachable only when asked for, with a ⚠ note ---
    for raw in ("", None, "tracked", " Tracked "):
        assert resolve_label_source(raw) == ("tracked", []), raw
    for raw in ("teacher_detect", "tracked_intersection"):
        ls_, nts_ = resolve_label_source(raw); assert ls_ == raw and len(nts_) == 1 and "NOT in the current plan" in nts_[0], (raw, nts_)
    try: resolve_label_source("teacher"); raise AssertionError("unknown label_source passed")
    except RuntimeError as e: assert "tracked|teacher_detect|tracked_intersection" in str(e), str(e)
    assert LABEL_SOURCE_DEFAULT == "tracked" and LABEL_SOURCE_DEFAULT in LABEL_SOURCES

    # --- human gate not required relaxes ONLY the gate term; an unsealed manifest is still refused ---
    must_raise(man(sealed=False, human_gate=hg_opt("pending")), needle="cannot see")                         # every term met but unsealed
    must_raise(man(sealed=False, human_gate=hg_opt("pending"), sam3_veto={"ran": False}), needle="sam3_veto.ran=false")
    must_raise(man(sealed=False, human_gate=hg_opt("pending"), counts=below), needle="below_floor")            # no allow_below_floor
    must_raise(man(sealed=False, human_gate=hg_opt("passed"), smoke=True), needle="smoke")
    assert manifest_guards(man(human_gate=hg_opt("pending"), label_sources_available=["tracked"], teacher_detect={"ran": False}),
                           "tracked", False, False) == D2                                                     # the planned run's manifest

    # --- provenance immutability: a completed run_tag is never overwritten without force ---
    assert refuse_overwrite([], False, "r1", "full") == []
    try: refuse_overwrite(["/vol/r1/train_summary.json"], False, "r1", "full"); raise AssertionError("overwrite passed")
    except RuntimeError as e: assert "force=true" in str(e) and "r1" in str(e)
    assert refuse_overwrite(["/vol/r1/train_summary.json"], True, "r1", "full") == ["/vol/r1/train_summary.json"]

    # --- resume: PTL 2.x saves last.ckpt in on_train_epoch_end, AFTER BestModelCallback raised the mark (review 2026-09-23);
    #     only the in-flight window (last.ckpt still epoch best-1, EMA file + csv row of the best epoch written) is patched ---
    assert resume_highwater_action(0.41, 0.41, 7, 7, True, ckpt_epoch=7) == "ok"             # last.ckpt of the best epoch carries its mark
    assert resume_highwater_action(0.41, 0.41, 7, 7, True, ckpt_epoch=9) == "ok"             # later epochs, best unchanged
    assert resume_highwater_action(0.38, 0.41, 7, 7, True, ckpt_epoch=6) == "patch"          # in-flight window of epoch 7
    assert resume_highwater_action(0.0, 0.41, 1, 1, True, ckpt_epoch=0) == "patch"           # epoch 0 scored 0.0 (mark stays 0.0); epoch 1 in flight
    assert resume_highwater_action(0.0, None, None, None, False) == "ok"                     # nothing improved yet, no EMA file: legal
    assert resume_highwater_action(None, None, None, None, False) == "ok"
    for args, kw, needle in (((0.38, 0.41, 7, 6, True), {"ckpt_epoch": 6}, "epoch 6"),       # staged EMA file is not the csv best epoch
                             ((0.41, 0.38, 7, 7, True), {"ckpt_epoch": 7}, "exceeds"),       # mark above the csv history
                             ((0.38, 0.41, 7, 7, True), {"ckpt_epoch": 7}, "in-flight"),     # last.ckpt already covers the best epoch: a state PTL cannot produce
                             ((0.38, 0.41, 7, 7, True), {"ckpt_epoch": 9}, "in-flight"),     # mark below although last.ckpt is past the best epoch
                             ((0.38, 0.41, 7, 7, True), {"ckpt_epoch": 5}, "in-flight"),     # last.ckpt lags by more than one epoch
                             ((0.38, 0.41, 7, 7, True), {}, "in-flight"),                    # ckpt epoch unknown: cannot prove the window
                             ((0.0, 0.41, 0, 0, True), {"ckpt_epoch": None}, "in-flight"),   # no epoch -1 last.ckpt exists
                             ((0.0, 0.41, 7, None, False), {}, "lost with the cluster"),     # EMA file missing although csv improved
                             ((0.2, None, None, None, False), {}, "lost with the cluster"),
                             ((0.2, None, None, 3, True), {}, "no validation row"),
                             ((None, 0.41, 7, 7, True), {"ckpt_epoch": 6}, "no BestModelCallback state")):
        try: resume_highwater_action(*args, **kw); raise AssertionError(f"resume action passed: {args} {kw}")
        except RuntimeError as e: assert needle in str(e), (needle, str(e))

    # --- the VERBATIM Tol_40 / Tol_03 helpers (runs_of / inter / m_bbox) against dense numpy ---
    for _ in range(50):
        a = rng.random((48, 64)) < 0.3; b = rng.random((48, 64)) < 0.3
        ra, rb = runs_of(rle_encode(a)), runs_of(rle_encode(b))
        assert ra[2] == int(a.sum()) and inter(ra, rb) == int(np.logical_and(a, b).sum())
        assert abs(inter(ra, rb) / max(1, ra[2] + rb[2] - inter(ra, rb)) - m_iou(a, b)) < 1e-12 or not np.logical_or(a, b).any()
    bm = np.zeros((48, 64), bool); bm[5:20, 10:33] = True
    assert m_bbox(bm) == [10, 5, 32, 19] and m_bbox(np.zeros((4, 4), bool)) is None
    fs = frameset_identity([{"id": 3, "file_name": "c.jpg"}, {"id": 1, "file_name": "a.jpg"}, {"id": 9, "file_name": "a_crop.jpg", "crop_variant": True}])
    assert fs["n_full"] == 2 and fs["image_ids"] == [1, 3] and fs == frameset_identity([{"id": 1, "file_name": "a.jpg"}, {"id": 3, "file_name": "c.jpg"}])

    # --- whole-clip split check: leak, manifest mismatch, missing clip_stem ---
    tr_im = [{"file_name": f"t{i}.jpg", "clip_stem": f"c_t{1 + i % 3}"} for i in range(9)]
    va_im = [{"file_name": f"v{i}.jpg", "clip_stem": f"c_v{1 + i % 2}"} for i in range(4)]
    sp = {"valid_clips": ["c_v1", "c_v2"], "train_clips": ["c_t1", "c_t2", "c_t3"]}
    assert clip_split_check(tr_im, va_im, sp) == (["c_t1", "c_t2", "c_t3"], ["c_v1", "c_v2"], [])
    assert clip_split_check(tr_im, va_im[:1], sp) == (["c_t1", "c_t2", "c_t3"], ["c_v1"], ["c_v2"])   # an emptied valid clip is reported, not fatal
    for bad_tr, bad_va, bad_sp, needle in ((tr_im + [{"file_name": "x.jpg", "clip_stem": "c_v1"}], va_im, sp, "SPLIT LEAK"),
                                           (tr_im, [], sp, "no image at all"),
                                           (tr_im, va_im + [{"file_name": "y.jpg", "clip_stem": "c_v7"}], sp, "not in manifest.split.valid_clips"),
                                           (tr_im + [{"file_name": "x.jpg", "clip_stem": "c_t9"}], va_im, sp, "not in manifest.split.train_clips"),
                                           (tr_im + [{"file_name": "x.jpg"}], va_im, sp, "lack clip_stem"),
                                           (tr_im, va_im, {"valid_clips": ["c_v1", "c_v2"], "train_clips": ["c_t1", "c_t2", "c_t3", "c_v1"]}, "both")):
        try: clip_split_check(bad_tr, bad_va, bad_sp); raise AssertionError(f"split check passed: {needle}")
        except RuntimeError as e: assert needle in str(e), (needle, str(e))

    # --- full/resume: the smoke record must describe this run; hours re-projected with the CURRENT epochs ---
    smk = {"variant": "seg-small", "resolution": 624, "N_train": 5500, "N_valid": 600, "batch_size": 4, "grad_accum": 4,
           "scale_jitter": False, "eval_masks_head_resolution": False, "ips": 2000 / 240, "t_val_img": 0.25, "hours_full": pr["hours_full"], "epochs": 40}
    cur = {k: smk[k] for k in SMOKE_BOUND_KEYS}
    assert smoke_mismatches(smk, cur) == []
    assert [m[0] for m in smoke_mismatches(smk, {**cur, "batch_size": 8, "grad_accum": 2})] == ["batch_size", "grad_accum"]   # widget 8x2 vs the OOM-surviving 4x4
    assert smoke_mismatches({k: v for k, v in smk.items() if k != "N_train"}, cur)[0][1] == "<absent>"
    assert abs(reproject_hours(smk, 5500, 600, 40) - pr["hours_full"]) < 1e-9
    assert abs(reproject_hours(smk, 5500, 600, 100) - 2.5 * pr["hours_full"]) < 1e-9        # epochs 40 -> 100 scales the projection 2.5x
    try: stop_gate(reproject_hours(smk, 5500, 600, 100), pr["hours_full"] * 2, 100, 624); raise AssertionError("re-projected gate did not fire")
    except RuntimeError as e: assert "STOP GATE" in str(e)

    # --- ckpt sync: stable snapshot vs an in-place rewrite; zip validation catches a cut copy ---
    import zipfile
    with tempfile.TemporaryDirectory() as td:
        src, dst = osp.join(td, "last.ckpt"), osp.join(td, "snap.ckpt")
        with zipfile.ZipFile(src, "w") as z: z.writestr("data.pkl", os.urandom(20000))
        assert stable_snapshot(src, dst, settle_s=0, sleep=lambda s: None) == file_sig(src) and validate_zip_file(dst)
        assert stable_snapshot(osp.join(td, "none"), dst, settle_s=0, sleep=lambda s: None) is None
        n = [0]
        def rewrite(s):                                             # a "sleep" during which PTL rewrites the file in place
            n[0] += 1
            with open(src, "ab") as f: f.write(b"x")
            os.utime(src, ns=(1, 1000 + n[0]))
        try: stable_snapshot(src, dst, settle_s=0, tries=3, sleep=rewrite); raise AssertionError("unstable file was snapshotted")
        except RuntimeError as e: assert "no stable snapshot" in str(e) and n[0] == 3
        cut = osp.join(td, "cut.ckpt")
        with open(src, "rb") as f, open(cut, "wb") as g: g.write(f.read()[:12000])
        try: validate_zip_file(cut); raise AssertionError("truncated zip passed")
        except RuntimeError as e: assert "zip" in str(e) or "CRC" in str(e), str(e)
        assert is_zip_container("last.ckpt") and is_zip_container("checkpoint_best_ema.pth") and not is_zip_container("metrics.csv")

    # --- resume generations: consistent snapshot set, publish into a NEW dir (COMPLETE.json last), prune, provenance binding ---
    with tempfile.TemporaryDirectory() as td:
        src, stg, vol = osp.join(td, "out"), osp.join(td, "stg"), osp.join(td, "vol", RESUME_DIRNAME)
        for d in (src, stg): os.makedirs(d)
        def wr(fn, data):
            with open(osp.join(src, fn), "wb") as f: f.write(data)
        nosleep = lambda s: None
        snap0 = lambda a, b: stable_snapshot(a, b, settle_s=0, sleep=nosleep)
        assert snapshot_set(src, stg, lambda p: "ok", sleep=nosleep, snap=snap0) is None          # nothing written yet
        wr("last.ckpt", b"L1"); wr("metrics.csv", b"epoch\n0\n")
        sigs, info = snapshot_set(src, stg, lambda p: {"action": "ok", "ema": p["checkpoint_best_ema.pth"]}, sleep=nosleep, snap=snap0)
        assert info == {"action": "ok", "ema": None} and sigs["checkpoint_best_ema.pth"] is None and sigs["last.ckpt"] is not None
        # an inconsistent set is retaken; the CSV flush arriving between attempts makes it consistent
        calls = [0]
        def cons(p):
            calls[0] += 1
            if calls[0] == 1:
                wr("metrics.csv", b"epoch\n0\n1\n"); raise RuntimeError("csv behind the EMA file")
            return "patch"
        assert snapshot_set(src, stg, cons, sleep=nosleep, snap=snap0)[1] == "patch" and calls[0] == 2
        try: snapshot_set(src, stg, lambda p: (_ for _ in ()).throw(RuntimeError("never agrees")), tries=3, sleep=nosleep, snap=snap0); raise AssertionError("inconsistent set published")
        except RuntimeError as e: assert "no consistent snapshot set after 3" in str(e) and "never agrees" in str(e) and not isinstance(e, CopierError), str(e)
        stg_paths = {fn: (osp.join(stg, fn) if osp.exists(osp.join(stg, fn)) else None) for fn in RESUMABLE}
        shas = {fn: sha256_file(p) for fn, p in stg_paths.items() if p}
        os.makedirs(vol)
        g1 = publish_generation(stg_paths, shas, vol, 1, {"provenance": {"run_tag": "r"}})
        assert list_generations(vol) == [(1, g1, True)] and sorted(os.listdir(g1)) == sorted(
            ["last.ckpt", "last.ckpt.sha256", "metrics.csv", "metrics.csv.sha256", GEN_COMPLETE])
        with open(osp.join(g1, GEN_COMPLETE)) as f: rec = json.load(f)
        assert rec["generation"] == 1 and rec["files"] == shas and rec["provenance"] == {"run_tag": "r"}
        assert verify_sidecar(osp.join(g1, "last.ckpt"), osp.join(g1, "last.ckpt.sha256")) == shas["last.ckpt"]
        try: publish_generation(stg_paths, shas, vol, 1, {}); raise AssertionError("generation rewritten")
        except RuntimeError as e: assert "never rewritten" in str(e)
        # a cut upload: the copier dies on the 2nd file -> no COMPLETE.json -> invisible; g1 untouched
        def dying(a, b, k=[0]):
            k[0] += 1
            if k[0] == 2: raise OSError("cluster died mid-upload")
            shutil.copyfile(a, b)
        try: publish_generation(stg_paths, shas, vol, 2, {}, copier=dying); raise AssertionError("cut upload passed")
        except CopierError as e: assert isinstance(e.__cause__, OSError) and "cluster died" in str(e), str(e)   # only a COPY failure is a CopierError
        assert [(g[0], g[2]) for g in list_generations(vol)] == [(2, False), (1, True)]
        assert [g for g in list_generations(vol) if g[2]][0][0] == 1                                   # resume picks g1, not the cut g2
        for n in (3, 4, 5): publish_generation(stg_paths, shas, vol, n, {})
        removed = prune_generations(vol)
        assert sorted(removed) == ["g000001", "g000002", "g000003"] and [g[0] for g in list_generations(vol)] == [5, 4], removed
        os.makedirs(osp.join(vol, gen_name(6)))                                                         # an in-flight (newer, incomplete) upload
        assert prune_generations(vol) == [] and [g[0] for g in list_generations(vol)] == [6, 5, 4]      # never touched
        assert list_generations(osp.join(td, "nope")) == [] and prune_generations(osp.join(td, "nope")) == []
    prov = {k: f"v_{k}" for k in RESUME_BOUND_PROV}; hp = {k: 1 for k in RESUME_BOUND_HPARAMS}
    assert resume_mismatches(prov, dict(prov), hp, dict(hp)) == []
    mm = resume_mismatches(prov, {**prov, "dataset_manifest_sha256": "other"}, hp, {**hp, "epochs": 60})
    assert [m[0] for m in mm] == ["dataset_manifest_sha256", "train.epochs"], mm
    assert resume_mismatches({k: v for k, v in prov.items() if k != "seed"}, prov, hp, hp) == [("seed", "<absent>", "v_seed")]
    assert len(resume_mismatches(None, prov, None, hp)) == len(RESUME_BOUND_PROV) + len(RESUME_BOUND_HPARAMS)

    # --- the contract Tol_44 reads from train_summary.json ---
    c = eval_v2_contract(0.35, 624, "seg-small", ["20240418", "20240419"], ["20240419", "20240418"])
    assert c["op_threshold"] == 0.35 and c["resolution"] == 624 and "resolution=624" in c["reload"]["call"]
    assert c["post_chain"]["min_mask_px"] == MIN_MASK_PX == 8000 and c["post_chain"]["topk"] == TOPK_DET == 6
    assert set(EVAL_V2_KEYS) - {"run_tag", "checkpoint_best_total_sha256"} <= set(c), set(EVAL_V2_KEYS) - set(c)
    for args, needle in (((0.33, 624, "seg-small", ["20240418"], ["20240418"]), "swept thresholds"),
                         ((0.35, 630, "seg-small", ["20240418"], ["20240418"]), "multiple of 24"),
                         ((0.35, 624, "seg-small", ["20240418", "20240422"], ["20240418"]), "forbidden_eval_days")):
        try: eval_v2_contract(*args); raise AssertionError(f"contract passed: {needle}")
        except RuntimeError as e: assert needle in str(e), (needle, str(e))
    # the written summary carries every key Tol_44 read_train_summary() + the rfdetr arm read, under the exact file names
    assert BEST_CKPT_FILE == "checkpoint_best_total.pth" and SUMMARY_FILE == "train_summary.json"
    ts = {"run_tag": "segv1_tracked_seg-small_r624", "dataset_tag": "segv1", "dataset_manifest_sha256": "m" * 64, "mode": "full",
          "checkpoint_best_total_sha256": "c" * 64, **c}
    assert "op_threshold" in train_summary_contract_check(ts) and train_summary_contract_check({**ts, "mode": "resume"})
    for bad, needle in (({"mode": None}, "lack the keys"), ({"dataset_tag": None}, "dataset_tag"), ({"mode": "smoke"}, "accepts only"),
                        ({"smoke": True}, "smoke run"), ({"op_threshold": 0.33}, "swept grid"), ({"op_threshold": "0.35"}, "swept grid"),
                        ({"resolution": 630}, "multiple of 24"), ({"forbidden_eval_days": ["20240418"]}, "forbidden_eval_days")):
        try: train_summary_contract_check({**ts, **bad}); raise AssertionError(f"summary contract passed: {bad}")
        except RuntimeError as e: assert needle in str(e), (needle, str(e))
    ts_rt = json.loads(json.dumps(ts)); assert train_summary_contract_check(ts_rt)                        # survives the JSON round trip
    # opportunistic cross-check against the consumer's source (skipped when Tol_44 is not next to this file)
    _t44 = osp.join(osp.dirname(osp.abspath(__file__)), "Tol_44_detector_eval.py") if "__file__" in globals() else None
    if _t44 and osp.exists(_t44):
        with open(_t44, encoding="utf-8") as f: _src = f.read()
        _m = re.search(r"def read_train_summary\(\):.*?for k in \(([^)]*)\):", _src, re.S)
        assert _m, "Tol_44 read_train_summary() key loop not found"
        _keys = set(re.findall(r'"([^"]+)"', _m.group(1)))
        assert _keys == set(TOL44_TS_KEYS), (sorted(_keys ^ set(TOL44_TS_KEYS)))
        assert f'"{BEST_CKPT_FILE}"' in _src and f'"{SUMMARY_FILE}"' in _src and 'TS["op_threshold"]' in _src
        assert 'ts.get("manifest_sealed")' in _src and 'ts.get("seal_waiver")' in _src, "Tol_44 seal_admissible() no longer reads manifest_sealed/seal_waiver"
        # Tol_44's post_chain (its own signature, returns n_floor too) must keep exactly what this notebook's sweep kept
        # (the op_threshold is chosen here and applied there): exec its source with ITS constants, compare on random masks
        _ns = {"np": np, "m_ios": m_ios, "m_iou": m_iou}
        for _ln in re.findall(r"^(?:N_CALVES|MIN_MASK_PX|DEDUPE_IOS|DET_TOPK)\s*=.*$", _src, re.M): exec(_ln.split("#")[0], _ns)
        assert (_ns["MIN_MASK_PX"], _ns["DEDUPE_IOS"], _ns["DET_TOPK"]) == (MIN_MASK_PX, DEDUPE_IOS, TOPK_DET), "Tol_44 post-chain constants differ"
        _pc = re.search(r"^def post_chain\(cands.*?(?=^\S)", _src, re.S | re.M)
        assert _pc, "Tol_44 post_chain(cands, ...) not found"
        exec(_pc.group(0), _ns)
        _rng = np.random.default_rng(0); _n_eq = 0
        def _rect(y, x, h, w):
            _m = np.zeros((300, 400), bool); _m[y:y + h, x:x + w] = True; return _m
        _edges = [[_rect(0, 0, 80, 100), _rect(100, 0, 79, 101)],   # exactly 8000 px kept; 7979 px dropped
                  [_rect(0, 0, 100, 100), _rect(0, 50, 100, 100)],                                             # IoS exactly 0.50 -> dropped
                  [_rect(0, 0, 100, 100), _rect(0, 51, 100, 100)],                                             # IoS 0.49 -> kept
                  [_rect(100 * (i // 4), 100 * (i % 4), 90, 90) for i in range(8)]]                            # 8 disjoint -> top-6
        for _trial in range(150 + len(_edges)):
            _ms, _ss = [], []
            if _trial < len(_edges):
                _ms = _edges[_trial]; _ss = [float(_rng.random()) for _ in _ms]
            for _ in (range(int(_rng.integers(0, 10))) if _trial >= len(_edges) else ()):
                _h, _w = int(_rng.integers(40, 180)), int(_rng.integers(40, 180)); _y, _x = int(_rng.integers(0, 120)), int(_rng.integers(0, 220))
                _m = np.zeros((300, 400), bool); _m[_y:_y + _h, _x:_x + _w] = True; _ms.append(_m); _ss.append(float(_rng.random()))
            _a = post_chain(_ms, _ss)
            _b, _nf = _ns["post_chain"](list(zip(_ms, _ss)))
            assert len(_a) == len(_b) and all(np.array_equal(ma, mb) and sa == sb for (ma, sa), (mb, sb) in zip(_a, _b)), f"post_chain mismatch, trial {_trial}"
            assert _nf == sum(1 for _m in _ms if _m.sum() < MIN_MASK_PX); _n_eq += 1
        print(f"  Tol_44 cross-check: read_train_summary() keys == TOL44_TS_KEYS; it opens {BEST_CKPT_FILE} and {SUMMARY_FILE}; "
              f"reads seal_waiver; its post_chain == this post_chain on {_n_eq} mask sets ({len(_edges)} edge cases: 8000-px floor, IoS 0.50, top-6)")
    else:
        print("SELFTEST: Tol_44_detector_eval.py not beside this file — skipping the source cross-check only")

    # --- intersection filter on a synthetic COCO dict (ids stay consistent) ---
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "phase4": "day"}, {"id": 2, "file_name": "b.jpg", "phase4": "night"},
                       {"id": 3, "file_name": "c.jpg", "phase4": "day"},
                       {"id": 10, "file_name": "a_crop.jpg", "crop_variant": True, "parent_file_name": "a.jpg", "phase4": "day"},
                       {"id": 11, "file_name": "b_crop.jpg", "crop_variant": True, "parent_file_name": "b.jpg", "phase4": "night"}],
            "annotations": [{"id": i, "image_id": im} for i, im in enumerate([1, 1, 2, 3, 10, 11])],
            "categories": CATEGORIES_EXPECTED}
    f = filter_coco_by_ids(coco, [1, 3])
    assert [im["id"] for im in f["images"]] == [1, 3, 10], f["images"]
    assert sorted(a["image_id"] for a in f["annotations"]) == [1, 1, 3, 10]
    assert coco["images"][0]["id"] == 1 and f["categories"] == CATEGORIES_EXPECTED

    # --- smoke stratifier keeps phase shares ---
    imgs = [{"id": i, "phase4": ph} for i, ph in enumerate(["day"] * 600 + ["night"] * 200 + ["dusk"] * 100 + ["dawn"] * 100)]
    sub = stratified_subset(imgs, 200, 0)
    assert len(sub) == 200 and len({im["id"] for im in sub}) == 200
    sh = phase_shares(sub)
    assert abs(sh["day"] - 0.6) < 0.03 and abs(sh["night"] - 0.2) < 0.03 and abs(sh["dusk"] - 0.1) < 0.03, sh
    assert stratified_subset(imgs, 200, 0) == sub, "stratifier is not deterministic"
    assert len(stratified_subset(imgs[:50], 200, 0)) == 50

    # --- post-chain dedupe reproduces Tol_03.detect() ordering ---
    H, W = 600, 900
    def block(y, x, h, w):
        m = np.zeros((H, W), bool); m[y:y + h, x:x + w] = True; return m
    A = block(0, 0, 200, 200)            # 40k px
    B = block(50, 50, 100, 100)          # fragment inside A: IoS 1.0 -> dropped
    Cm = block(300, 0, 120, 120)         # 14.4k px, disjoint -> kept after A
    D = block(0, 300, 80, 80)            # 6.4k px < MIN_MASK_PX -> dropped
    E = block(0, 260, 150, 150)          # overlaps A? no (x 260..410 vs 0..200) -> kept; 22.5k
    kept = post_chain([B, D, Cm, A, E], [0.9, 0.9, 0.5, 0.7, 0.6])
    assert [int(m.sum()) for m, _ in kept] == [40000, 22500, 14400], [int(m.sum()) for m, _ in kept]
    assert [s for _, s in kept] == [0.7, 0.6, 0.5]
    many = [block(0, 110 * i, 100, 100) for i in range(8)]     # 8 disjoint 10k masks -> top-6
    assert len(post_chain(many, [0.5] * 8)) == 6 and len(post_chain(many, [0.5] * 8, topk=None)) == 8
    G = block(0, 0, 200, 200); Gp = block(0, 100, 200, 200)     # IoU 1/3, IoS 0.5 -> Gp dropped (IoS >= 0.50)
    assert len(post_chain([G, Gp], [0.5, 0.5])) == 1

    # --- F1 sweep with a planted best threshold and the tie rule ---
    L1, L2, L3 = block(0, 0, 150, 150), block(200, 200, 150, 150), block(400, 500, 150, 150)
    FP = block(0, 700, 150, 150)
    rows = [(L1, 0.90), (L2, 0.70), (L3, 0.45), (FP, 0.30)]
    acc = sweep_accumulate({}, rows, [L1, L2, L3])
    tau, f1 = op_threshold_from_sweep({t: tuple(v) for t, v in acc.items()})
    assert abs(f1[0.20] - 6 / 7) < 1e-9 and abs(f1[0.30] - 1.0) < 1e-9 and abs(f1[0.40] - 1.0) < 1e-9 and abs(f1[0.45] - 0.8) < 1e-9, f1
    assert tau == 0.30, tau                                     # 0.30/0.35/0.40 tie at 1.0 -> lower wins
    tp, fp, fn, miou = hungarian_counts([L1, FP], [L1, L2]); assert (tp, fp, fn) == (1, 1, 1) and 0 < miou < 1
    assert hungarian_counts([], [L1]) == (0, 0, 1, 0.0) and hungarian_counts([L1], []) == (0, 1, 0, 0.0)

    # --- metrics.csv summary + phase/time helpers ---
    rows_csv = [{"epoch": "0", "val/segm_mAP_50_95": "0.10", "val/segm_mAP_50": "0.3"}, {"epoch": "0", "train/loss": "1.0"},
                {"epoch": "1", "val/segm_mAP_50_95": "0.25", "val/segm_mAP_50": "0.6"}, {"epoch": "2", "val/segm_mAP_50_95": "0.20", "val/segm_mAP_50": "0.5"}]
    s = summarize_metrics_csv(rows_csv)
    assert s == {"epochs_run": 3, "best_epoch": 1, "best_val_segm_mAP_50_95": 0.25, "best_val_segm_mAP_50": 0.6}, s
    assert summarize_metrics_csv([])["epochs_run"] == 0
    assert phase4(datetime(2024, 4, 19, 6, 30)) == "dawn" and phase4(datetime(2024, 4, 19, 20, 5)) == "dusk"
    assert phase4(datetime(2024, 4, 19, 21, 0)) == "night" and phase4(datetime(2024, 4, 19, 12, 0)) == "day"
    assert clip_start("Tol_ch01_20240419-204145-212402") == datetime(2024, 4, 19, 20, 41, 45)
    assert luminance_stretch(np.full((4, 4, 3), 7, np.uint8)).dtype == np.uint8
    print("SELFTEST PASS — kwargs whitelist, scale/precision replicas, projection + stop gate, sha256 sidecar, manifest guards "
          "(seal terms, human gate required vs optional per the 2026-09-23 decision, counted-only seal rule + sheet_n, per-day/pooled "
          "provenance, band-constant presence/values, sealed=true re-derived, G9 gates proof, 04-22 reason, tracked independent of "
          "teacher_detect), valid night floor + G12 buffer on the images, run_tag immutability, resume high-water mark, resume "
          "generations (consistent set, cut upload invisible, prune, provenance/hparam binding), clip split, smoke binding + re-projection, "
          "stable snapshot + zip validation, intersection filter, stratifier, post-chain ordering, F1 sweep + tie rule, Tol_44 "
          "contract + train_summary.json key/name contract, label_source default tracked, unsealed "
          "manifest still refused with the gate optional, metrics.csv summary, verbatim RLE helpers, frame-set identity")

_selftest()
