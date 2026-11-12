# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N42 — export `Tracking_train` masks (04-18/04-19) to a COCO instance-segmentation dataset
# MAGIC
# MAGIC **Purpose.** Turn the stored RLE masks of the training days into an INSTANCE-indexed COCO
# MAGIC segmentation dataset that contains **no frame with an unlabelled visible calf and no merged mask**,
# MAGIC honouring the reviewed fault list (`train_label_faults.json` — a FAULT list, not a whitelist) plus
# MAGIC automatic gates whose constants are derived from the training days only;
# MAGIC decorrelated in time; split by WHOLE clip; with a 16:9 count-prior crop variant; human-review
# MAGIC contact sheets; an optional-but-default SAM3 stage that produces the unlabelled-animal veto (G11) and, as a
# MAGIC free by-product, `teacher_detect` labels; verified by read-back; described by a manifest that Tol_43 / Tol_44
# MAGIC enforce.
# MAGIC
# MAGIC **Design decisions 2026-09-23.**
# MAGIC 1. Below the 5,000-frame floor the next step is re-running with strides 8 s / 4 s; adding 04-22 is only the second option.
# MAGIC 2. No mechanism-control (DET-C) training is planned: `teacher_detect` / `tracked_intersection` are kept as a free
# MAGIC    by-product of the veto pass and nothing downstream may require them.
# MAGIC 3. **The reviewed fault list IS the human gate** (both training-day videos were reviewed end to end). Widget
# MAGIC    `require_human_gate` (default `false`): the seal no longer needs a passed 300-cell review; the sheets and the template
# MAGIC    are still rendered, and a `human_gate.json` that IS present is still validated, its BAD cells still remove ±90 s, and
# MAGIC    a FAILED one always unseals. `true` restores the original rule (only a passed review seals).
# MAGIC 4. The evaluation is Tol_44 (teacher vs fine-tuned detector on the tracker's own detector calls on 04-22/04-23,
# MAGIC    against the verified GT). From this export it reads only `manifest.forbidden_eval_days` and
# MAGIC    `manifest.constants["frag_lo[dayl]" | "frag_lo[dark]" | "over_hi"]`. `position_prior.json`, `plates/`, `body_len_px` and
# MAGIC    `sheet_frames/` were inputs of an earlier, superseded evaluation; they are still produced (they are
# MAGIC    cheap, and the plates feed the furniture diagnostic) but nothing currently consumes them.
# MAGIC
# MAGIC **Stages** (widget `stage`): `gate` (CPU `Standard_D16ds_v5`) → `teacher` (A100) → `finalize` (CPU). The SAM3 veto is
# MAGIC isolated in the one `teacher` cell: on a cluster without CUDA it is SKIPPED with a loud banner and writes nothing
# MAGIC (no `.done_teacher`), so `finalize` with `require_veto=true` then refuses to seal.
# MAGIC
# MAGIC **Inputs.** `TOL/<root>/<window>/{masks_*.json.gz, segments.json, events.jsonl}` for every
# MAGIC `(root, day)` in `sources`; `train_label_faults.json`; `TOL/Metrics_healthy_band_train.json`
# MAGIC (`over_hi`, `ref_pass1_largest_daylight` — READ, never typed); `TOL/Frames_5fps/<day>/<clip_stem>.tar`
# MAGIC + `<clip_stem>_index.json` (JPEG q90 2304×1296, `frame_idx` identical to the masks rows).
# MAGIC
# MAGIC **Outputs.** `TOL/Detector_COCO/<tag>/` (manifest.json, gate_table.json.gz, admitted.json, veto.json,
# MAGIC teacher_dets.json.gz, human_gate_template.json, images.tar, annotations/{tracked,teacher_detect}/…,
# MAGIC intersection_teacher_image_ids.json (stage teacher, never rewritten), intersection_image_ids.json (finalize, post-drop),
# MAGIC position_prior.json, plates/, sheet_frames/, .done_<stage>),
# MAGIC `TOL/Figures/seg_export/<tag>/` (sheets, band_derivation.png, distribution.png, _verify/),
# MAGIC `TOL/Metrics_seg_export_<tag>.json`, provenance copies of the fault file (`Detector_COCO/<tag>/train_label_faults.json`,
# MAGIC `Detector_Labels/train_label_faults_<sha8>.json`, and the untagged `Detector_Labels/train_label_faults.json`).
# MAGIC
# MAGIC **Staleness / provenance rules (review 2026-09-22, 2026-09-23).** `admitted.json` records the sources, the fault/band file
# MAGIC sha256, the sha256 of every `masks_*.json.gz` / `segments.json` / `events.jsonl` it read, a `gate_run_id`, and every
# MAGIC gate-time widget (`gate_widgets`, incl. `band_json`, `allow_test_day_reason`, `unlisted_day_policy`) it was gated with;
# MAGIC `teacher` and `finalize` resolve those widgets from `admitted.json` BEFORE any source/fault/band guard runs, raise if the
# MAGIC files, the tracking root or the sources differ, RUN WITH the gate's widget values, and raise if a gate-time widget is
# MAGIC explicitly set to another value. `teacher`/`finalize` require `.done_gate` (SMOKE aside); a `gate` run first deletes the
# MAGIC previous `admitted.json`, `.done_gate` and `Figures/seg_export/<tag>/`, and `finalize` checks that
# MAGIC `human_gate_template.json` / `position_prior.json` carry the same `gate_run_id` as `admitted.json`.
# MAGIC A source day absent from the fault file (e.g. a 04-22 source) has no fault list, so `gate` and `finalize` require
# MAGIC `require_human_gate=true` for it (the fault-list waiver covers only the two reviewed training-day videos).
# MAGIC Every `teacher` output is stamped with `sha256(admitted.json)` and `finalize` refuses a stale one; a forced `gate`
# MAGIC re-run deletes them (plus an unsealed manifest, images.tar, annotations/, sheet_frames/ and the later stages' metrics
# MAGIC keys). A tag with a SEALED manifest refuses `gate` and `teacher` even with `force=true` (bump the tag). The fault-file provenance
# MAGIC copies are written only by `gate`, and a per-tag copy with other content is never overwritten. The filled
# MAGIC `human_gate.json` must carry exactly the template's cells. `finalize` is re-runnable (new verdicts) while the
# MAGIC manifest is unsealed; a sealed manifest needs `force=true` and the supersession is recorded in `finalize_runs`.
# MAGIC `finalize` writes nothing contract-bearing to the Volume until every check has passed (publish step).
# MAGIC
# MAGIC **Assumptions (each asserted).** masks are `[1296, 2304]`; `track_fps == 5.0`; `backbone == "teacher"`
# MAGIC in every masks file; every source day appears in the fault file's `windows` (else
# MAGIC `unlisted_day_policy=gates_only`); every window dir of a source day is listed there; the fault file's
# MAGIC `tracking_root == root`; `band_json.derived_from ⊆ TRAIN_DAYS`; per clip the set of `frame_idx` in the
# MAGIC masks file equals the set in `<clip_stem>_index.json`; `Tracking_train` ran `partial_repair=false`
# MAGIC (no `partial_state` / `grown_state` / `*_repair*` events exist).
# MAGIC
# MAGIC Runnable locally as `TOL_SELFTEST=1 python Tol_42_export_seg_coco.py` (pure-python SELFTEST only).

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "opencv-python-headless==4.11.0.86" "Pillow==11.1.0" "pycocotools" "scipy"

# COMMAND ----------

# stage=teacher additionally needs the SAM3 stack; a %pip
# line cannot be conditional, so the extras are installed here BEFORE the restart, only for that stage.
try:
    dbutils
    dbutils.widgets.text("stage", "")
    if (dbutils.widgets.get("stage") or "").strip() == "teacher":
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "transformers==5.2.0", "timm", "huggingface_hub", "accelerate"])
    dbutils.library.restartPython()
except NameError:
    pass   # local / worker mode: no dbutils, no restart

# COMMAND ----------

import os, os.path as osp, sys, re, io, json, gzip, glob, csv, math, time, shutil, hashlib, tarfile, random
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from bisect import bisect_right
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # a cp1252 console (Windows 11 local SELFTEST) cannot print the ⚠ glyph
except Exception: pass

SELFTEST_ONLY = os.environ.get("TOL_SELFTEST") == "1"
IS_NOTEBOOK = True
try:
    dbutils
except NameError:
    IS_NOTEBOOK = False

W_DEFAULTS = {}                      # widget name -> default string (gate_widgets_resolve tells "not passed" from "passed differently")
def P(name, default):
    """Widget helper, structure from Tol_03_track_v4.py:71-76; env prefix TOL_."""
    W_DEFAULTS[name] = default
    if IS_NOTEBOOK:
        try: dbutils.widgets.text(name, default)
        except Exception: pass
        return (dbutils.widgets.get(name) or default)
    return os.environ.get("TOL_" + name.upper(), default)

# ---- widgets. REQUIRED ones default "" and raise when empty (no silent defaults) ----------
W = {}
W["stage"]                 = P("stage", "").strip()                      # REQUIRED gate|teacher|finalize
W["sources"]               = P("sources", "").strip()                    # REQUIRED "Tracking_train:20240418,20240419[;Root:day]"
W["tag"]                   = P("tag", "").strip()                        # REQUIRED dataset version, e.g. segv1
W["faults_json"]           = P("faults_json", "").strip()                # REQUIRED path of train_label_faults.json
W["allow_test_day_reason"] = P("allow_test_day_reason", "").strip()
W["unlisted_day_policy"]   = P("unlisted_day_policy", "").strip()
W["band_json"]             = P("band_json", "Metrics_healthy_band_train.json").strip()
W["stride_day_s"]          = float(P("stride_day_s", "10") or 10)        # protocol 2026-09-22
W["stride_other_s"]        = float(P("stride_other_s", "5") or 5)        # protocol 2026-09-22
W["valid_clips"]           = P("valid_clips", "Tol_ch01_20240419-121632-125839,Tol_ch01_20240418-195830-204050,"
                                              "Tol_ch01_20240418-220618-224904,Tol_ch01_20240419-220647-224924").strip()
W["valid_stride_day_s"]    = float(P("valid_stride_day_s", "20") or 20)  # validation cost; not derived from any day
W["valid_stride_other_s"]  = float(P("valid_stride_other_s", "10") or 10) # not derived from any day
W["valid_max"]             = int(P("valid_max", "600") or 600)           # valid cap; not derived from any day
W["min_train"]             = int(P("min_train", "5000") or 5000)         # floor on train full frames
W["n_max"]                 = int(P("n_max", "8000") or 8000)             # diagnostic cap; not derived from any day
W["night_quota"]           = float(P("night_quota", "0.35") or 0.35)     # target; not derived from any day
W["dusk_quota"]            = float(P("dusk_quota", "0.08") or 0.08)      # target; not derived from any day
W["dawn_quota"]            = float(P("dawn_quota", "0.08") or 0.08)      # target; not derived from any day
W["active_quota"]          = float(P("active_quota", "0.15") or 0.15)    # target; not derived from any day
W["hour_cap_frac"]         = float(P("hour_cap_frac", "0.08") or 0.08)   # sampling step 4; not derived from any day
W["crop_frac"]             = float(P("crop_frac", "0.25") or 0.25)       # not derived from any day
W["crop_wh"]               = P("crop_wh", "1536x864").strip()            # 16:9 = deployment squash geometry
W["valid_buffer_min"]      = float(P("valid_buffer_min", "10") or 10)    # G12; not derived from any day
W["frag_lo_override"]      = P("frag_lo_override", "").strip()           # "dayl=0.xx,dark=0.yy", only after a raise
W["collapse_override_blocks"] = P("collapse_override_blocks", "").strip()
W["curl_band_ack"]         = P("curl_band_ack", "").strip()              # free-text reason; ONLY after the relative curled-stretch check (G6-G9) raised and the curl_rejected sheet was inspected (review 2026-09-23)
W["require_veto"]          = (P("require_veto", "true") or "true").strip().lower() == "true"
W["require_human_gate"]    = (P("require_human_gate", "false") or "false").strip().lower() == "true"   # decision 2026-09-23: the fault list IS the human gate
W["sheet_n"]               = int(P("sheet_n", "300") or 300)             # >= SHEET_N_REF outside SMOKE when require_human_gate (~300 reviewed overlays)
W["seed"]                  = int(P("seed", "0") or 0)
W["max_frames"]            = int(P("max_frames", "0") or 0)
W["force"]                 = (P("force", "false") or "false").strip().lower() == "true"

# gate-time widgets: recorded in admitted.json by stage gate; teacher/finalize take them FROM admitted.json and raise
# when this run explicitly passes a different (non-default) value (review 2026-09-22, gate_widgets_resolve)
# (band_json IS listed since review 2026-09-23: the band file is read from the gate's band_json, its sha256 is then asserted)
GATE_KEYS = ("band_json", "unlisted_day_policy", "allow_test_day_reason", "stride_day_s", "stride_other_s", "valid_clips",
             "valid_stride_day_s", "valid_stride_other_s", "valid_max", "min_train", "n_max", "night_quota", "dusk_quota",
             "dawn_quota", "active_quota", "hour_cap_frac", "crop_frac", "crop_wh", "valid_buffer_min", "frag_lo_override",
             "collapse_override_blocks", "curl_band_ack", "sheet_n", "seed")

STAGE, TAG_IN, SEED = W["stage"], W["tag"], W["seed"]
SMOKE = W["max_frames"] > 0
TAG = TAG_IN + ("_smoke" if SMOKE and TAG_IN else "")

# ---- paths --------------------------------------------------------------------
VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
DS_DIR   = osp.join(TOL, "Detector_COCO", TAG) if TAG else None
FIG_DIR  = osp.join(TOL, "Figures", "seg_export", TAG) if TAG else None
MET_PATH = osp.join(TOL, f"Metrics_seg_export_{TAG}.json") if TAG else None
LAB_DIR  = osp.join(TOL, "Detector_Labels")
FR_ROOT  = osp.join(TOL, "Frames_5fps")
LOCAL    = "/local_disk0/tmp/n42"

# ---- constants: every literal carries its provenance -----------------------------------------------
N_CALVES = 4                                   # Tol_03_track_v4.py:136
H_FRAME, W_FRAME = 1296, 2304                  # frame size, STORAGE_LAYOUT.md
TRACK_FPS_EXPECTED = 5.0                       # Tol_03 INV-6 reference cadence
TRAIN_DAYS = {"20240418", "20240419"}
DET_TEST_DAYS = {f"202404{d}" for d in range(22, 29)}   # 04-22..04-28
POOL = {"dayl": {"day", "dawn"}, "dark": {"dusk", "night"}}
DISJ_IOU, DISJ_IOS = 0.02, 0.05                # G4
LINK_IOU_MIN = 0.50                            # G5, protocol literal; SUPPORTED on 04-18/04-19 only: pass A asserts P1 of their G1-G4-survivor link IoU > 0.50 (link_iou_report); Tol_12's test-day figures are not used
CC_FRAC, CC_OPEN_PX = 0.95, 3                  # G10 (B3 / B14 fragment signature)
BLOCK_LARGEST_LO = 0.75                        # G9(a), at block-largest level
EVENT_BUF_S = 60.0                             # G2, half DETECT_DAY_SEC (Tol_03:86)
CONTRA_PRE_S = 120.0                           # G2: detector_contradiction closes
CHAOS_TYPES = ("mask_area_collapse", "merge_resolved", "occluded", "segment_open", "segment_close", "strike")  # Tol_30:77 verbatim
EXTRA_TYPES = ("pass_no_detections", "all_closed", "clip_unreadable")
CONTRA_REASONS = ("detector_contradiction", "displaced_by_detection")    # Tol_03:757, :787 close reasons
CHAOS_EVENTS, CHAOS_MIN = 3, 5                 # Tol_30:63-64
VETO_AREA_FRAC, VETO_IOS = 0.30, 0.30          # G11 / ignored_detection (25,000 px) as a training-day ratio
FRAG_LO_SANITY = (0.30, 0.59)                  # raise band
FRAG_LO_DAY_AGREE = 0.10                       # per-day P1 agreement
CURL_ADMIT_MIN = 0.90                          # human-verdict check (G6 only, in derive_constants)
CURL_REL_MIN = CURL_ADMIT_MIN                  # review 2026-09-23: the 0.90 above reused as a RELATIVE factor after G6-G9 (stretch admission >= 0.90 x the dark-pool U admission outside it); not derived from any day
BAD_OVERLAP_HI = 0.10                          # known-partial overlap flag
CURL_START, CURL_END = "2024-04-19T20:41:45", "2024-04-19T21:17:30"   # reviewer-confirmed curled stretch (fault file note; B8 exclude_start)
STAB_WIN_S = 10.0                              # G8 window ±10 s
ACT_HALF_S = 5.0                               # activity score ±5 s
CROP_KEEP_FRAC, CROP_DROP_FRAC, CROP_MAX_DRAWS, CROP_ZERO_MAX = 0.25, 0.02, 30, 0.15
HUMAN_MARGIN_S = 90.0                          # mirrors exclusion_margin_s
LATTICE_TOL_S = 2.0                            # sampling step 1
PROBE_WALLS = ("2024-04-18T23:15:08", "2024-04-19T23:33:29", "2024-04-18T22:48:32")   # fault-file notes: "unlisted but visible"
NOTE_EXCL_WALLS = ("2024-04-18T23:15:08", "2024-04-19T23:33:29")   # fault-file notes (04-18/04-19): "unlisted but visible" OVER-EXTENDED masks; excluded notebook-side (review 2026-09-23) because an over-extension onto straw with r <= over_hi passes G4/G6/G7/G8, so the notes' "gates cover these" is not checkable. (22:48:32 lies inside A7.)
NOTE_EXCL_MAX_S = 600.0                        # design choice: a note exclusion spans the fault-free gap around its wall, capped at ±10 min (on the shipped file the gaps are 9.5 min A8→A9 and 2.0 min B15→B16, so the cap is inactive); not derived from any day
FURN_CONSIST, FURN_MOVE_PX, FURN_LOOKBACK_S = 0.80, 20.0, 600.0   # furniture diagnostic
PLATE_EVERY_S, PLATE_MIN_FRAMES = 120.0, 15    # review sheets (the current Tol_44 evaluation no longer uses plates)
PLATE_W, PLATE_H = 576, 324                    # 4x downscale of 2304x1296
BAD_VERDICTS = ("partial", "split", "merged", "unlabelled_calf", "wrong_object")   # human-gate verdict vocabulary
HUMAN_GATE_MAX_BAD = 3                         # seal rule: <= 3 BAD among counted cells; protocol value, not derived from any day
HUMAN_STATUS = ("pending", "passed", "failed") # the only legal human_gate.json 'status' values
GATE_VOCAB = ["ok", "partial_minor", "partial", "split", "merged", "unlabelled_calf", "wrong_object"]
CROP_ID_OFFSET = 5_000_000                     # crop image ids = parent id + offset (unique in the train JSON)
SHEET_MIX = {"tail": 100, "tail_enlarged": 150, "night": 80, "dusk": 40, "dawn": 40, "day": 40}   # counted-cell mix, stated for sheet_n = SHEET_N_REF
SHEET_N_REF = 300                              # sheet_n=300; another sheet_n scales SHEET_MIX proportionally
PROBE_TOL_S = 120.0                            # design choice: a probe wall is "admitted" if a frame lies within ±120 s = one 10 s-lattice dozen
FURN_SHEET_MAX = 60                            # design choice: flagged_furniture extras capped at 5 sheets × 12 (≈ 0 flagged expected)
PLATE_RES_SAMPLE = 40                          # design choice: SEEDED random residual sample per plate for the 3·MAD threshold (the earlier Tol_44 plate rule used 40 frames/plate); capped samples are printed
PRIOR_SAMPLE_N, PRIOR_DS = 500, 8              # design choice: DET-0 medoid over ≤ 500 seeded frames at 1/8 scale (O(n²) Hungarian)
LOW_HOUR_MIN_ROWS = 60                         # design choice: an hour with < 60 1-Hz rows (1 min) is too short to flag admitted/full < 0.10 (sampling step 6)
VERIFY_POOL_N, VERIFY_N, OVERLAY_N = 400, 60, 50   # 60 seeded images decoded exactly, 50 overlays; 400 = stored-RLE pool kept in memory for that draw
CURL_MIN_ROWS = 100                            # review 2026-09-22: the curled-stretch check needs ≥ 100 of its ≈ 2,145 1-Hz rows in U, else it is vacuous (raise)
SMOKE_PLATE_MAX = 40                           # SMOKE: plate frames per (day, phase4) capped at 40 (printed)
DOWNSTREAM_OF_GATE = (".done_teacher", ".done_finalize", "veto.json", "teacher_dets.json.gz", "intersection_teacher_image_ids.json",
                      "intersection_image_ids.json", "manifest.json", "images.tar")   # keyed by image_id => stale after a gate re-run (an UNSEALED manifest only; a sealed tag refuses gate)
DOWNSTREAM_DIRS_OF_GATE = ("annotations", "sheet_frames")                            # removed with the files above (review 2026-09-22)
GATE_OWN_OUTPUTS = (".done_gate", "admitted.json")   # review 2026-09-23: removed at the START of a gate run, so a gate that dies mid-way leaves no admitted.json for teacher/finalize to pair with its new sheets/template/prior
LINK_VALLEY_PCT = 1.0                          # G5 support check: P1 of training-day link IoU must exceed LINK_IOU_MIN — the same 1 % convention as frag_lo; protocol value, not derived from any day
MET_KEYS_OF_LATER_STAGES = ("teacher_detect", "veto", "finalize")                    # metrics keys a gate re-run invalidates (review 2026-09-22)
TEACHER_MATCH_MIN = 0.50                       # Hungarian max(IoU,IoS) floor for a teacher_detect label; protocol value, not derived from any day
TEACHER_DROP_HIGH = 0.15                       # descriptive 'drop_high' flag only; a mechanism-control comparison must use tracked_intersection at ANY drop rate
LOW_HOUR_FRAC = 0.10                           # sampling step 6: flag a (day, hour) whose admitted/full < 0.10; protocol value, not derived from any day
NV_WARN_MIN = 20                               # design choice: warn when a crop n_visible value (0..4) occurs < 20 times; not derived from any day
SHEET_CELLS = 12                               # 3 x 4 cells per sheet; also the per-gate rejected_G<k> sample (one sheet)
SHEET_CROP_CELLS = 36                          # crops_<k>.png = 3 sheets x 12 crop cells
# incumbent detector contract (Tol_03_track_v4.py:225-227), used verbatim in stage teacher
MIN_MASK_PX      = 8000
DET_SCORE_MIN, DET_TEXT = 0.40, "calf"
DEDUPE_IOS       = 0.50
SAM3_ID = "facebook/sam3"
# Tol_17:104-106 BGR colours: slot 0 red / 1 green / 2 blue / 3 yellow (the fault list's mapping)
COLORS = [(60, 60, 255), (60, 220, 60), (255, 160, 40), (40, 220, 255)]

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

if not SELFTEST_ONLY:
    for k in ("stage", "sources", "tag", "faults_json"):
        if not W[k]: raise RuntimeError(f"widget '{k}' is REQUIRED and empty (no silent defaults)")
    if STAGE not in ("gate", "teacher", "finalize"): raise RuntimeError(f"stage must be gate|teacher|finalize, got {STAGE!r}")
    print("widgets:"); [print(f"  {k:24s} = {v!r}") for k, v in W.items()]
    if SMOKE: print(f"⚠ SMOKE: max_frames={W['max_frames']} -> tag {TAG}; nothing is sealed, no .done_ written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pure functions — RLE arithmetic, time, phases (VERBATIM copies where the brief says so)

# COMMAND ----------

# VERBATIM from Tol_40_train_label_survey.py:63-78 (runs_of, inter)
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

# VERBATIM from Tol_03_track_v4.py:280-308 (rle_encode, rle_decode, m_iou, m_ios, m_bbox)
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

def rle_area(rle): return int(sum(rle["counts"][1::2]))     # Tol_03:992 — area without decoding

def runs_iou(A, B):
    it = inter(A, B); u = A[2] + B[2] - it
    return it / u if u else 0.0

def runs_ios(A, B):
    it = inter(A, B); m = min(A[2], B[2])
    return it / m if m else 0.0

def centroid_from_runs(A, H):
    """Column-major centroid (x=col, y=row) in closed form from foreground runs. A run may
    cross a column boundary (bottom of col c -> top of col c+1); those rare runs are split explicitly."""
    s, e, area = A
    if area == 0: return None
    s = np.asarray(s, np.int64); e = np.asarray(e, np.int64); L = e - s
    c0 = s // H; c1 = (e - 1) // H
    same = c0 == c1
    r0 = s[same] % H; Ls = L[same]
    sx = float((Ls * c0[same]).sum()); sy = float((Ls * (2 * r0 + Ls - 1) / 2.0).sum())
    for si, ei in zip(s[~same], e[~same]):
        si = int(si); ei = int(ei)
        while si < ei:
            c = si // H; ce = min(ei, (c + 1) * H); l = ce - si; r = si % H
            sx += l * c; sy += l * (2 * r + l - 1) / 2.0; si = ce
    return (sx / area, sy / area)

# VERBATIM from Tol_40_train_label_survey.py:80-83
RXT = re.compile(r"-(\d{8})-(\d{6})-")
def clip_start(v):
    m = RXT.search(v.replace("_", "-"))
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

RX_EV = re.compile(r'[-_](\d{8})[-_](\d{6})[-_]')        # Tol_30_periodic_identity.py:98

def phase4(w):                       # night/dawn/dusk/day
    if w.hour >= 21 or w.hour < 6: return "night"
    if w.hour < 8:                 return "dawn"      # 06:00-07:59
    if w.hour == 20:               return "dusk"
    return "day"

def phase3(w): return "day" if phase4(w) == "dawn" else phase4(w)   # = Tol_40:85 phase_of

def pool_of(ph): return "dayl" if ph in POOL["dayl"] else "dark"

def clip_stem_of(vname): return vname[:-4].replace(" ", "_")     # Tol_03:1102

def pct(v, q):
    v = np.asarray(v, float); return float(np.percentile(v, q)) if len(v) else float("nan")
def floor2(x): return math.floor(x * 100 + 1e-9) / 100.0
def ceil2(x):  return math.ceil(x * 100 - 1e-9) / 100.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pure functions — sources guard (G0), fault list (G1), event/segment buffers (G2), segments

# COMMAND ----------

def parse_sources(s):
    """'Root:day1,day2;Root2:day3' -> [(root, day)], validated."""
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part: continue
        if ":" not in part: raise RuntimeError(f"sources entry {part!r} must be '<root>:<day>[,<day>]'")
        root, days = part.split(":", 1)
        root = root.strip()
        for d in days.split(","):
            d = d.strip()
            if not re.fullmatch(r"\d{8}", d): raise RuntimeError(f"bad day {d!r} in sources")
            out.append((root, d))
    if not out: raise RuntimeError("sources parsed to an empty list")
    if len(set(out)) != len(out): raise RuntimeError(f"duplicate (root, day) in sources: {out}")
    return out

def guard_source_days(pairs, allow_test_day_reason):
    """G0 (detector test days never train): 20240423+ always raises; 20240422 needs a non-empty reason."""
    for root, d in pairs:
        if d in DET_TEST_DAYS and d != "20240422":
            raise RuntimeError(f"source day {d} is a detector TEST day (test days never train) — refused, no widget overrides this")
        if d == "20240422" and not allow_test_day_reason:
            raise RuntimeError("20240422 in sources requires a non-empty allow_test_day_reason")
        if d not in TRAIN_DAYS and d != "20240422":
            raise RuntimeError(f"source day {d} is neither a training day nor 20240422")
    return sorted({d for _, d in pairs})

def fault_intervals(faults, day):
    """[(exclude_start, exclude_end, id)] for one day (the JSON bounds already carry the 90 s margin)."""
    out = []
    for f in faults["faults"]:
        if f["day"] != day: continue
        out.append((datetime.fromisoformat(f["exclude_start"]), datetime.fromisoformat(f["exclude_end"]), f["id"]))
    return sorted(out)

def fault_hit(wall, ivs):
    """First fault id whose [exclude_start, exclude_end) contains wall, else None."""
    for a, b, fid in ivs:
        if a <= wall < b: return fid
    return None

def note_exclusions(faults, walls, max_s):
    """Notebook-side exclusions for 'unlisted but visible' mask defects named in the fault file's NOTES (review 2026-09-23).
    Each wall excludes the fault-free gap that contains it — from the previous fault's exclude_end to the next fault's
    exclude_start on that day — capped at ±max_s around the wall, so it always meets the neighbouring exclusions and
    covers at least their 90 s margin. A wall already inside a listed exclusion adds nothing. A wall that the notes no
    longer mention raises (NOTE_EXCL_WALLS is then stale). Returns ({day: [(a, b, id)]}, report list)."""
    notes = " ".join(str(n) for n in faults.get("notes", []))
    by_day, rep = defaultdict(list), []
    for i, w in enumerate(walls):
        wd = datetime.fromisoformat(w); day = wd.strftime("%Y%m%d")
        if f"{wd:%m-%d %H:%M:%S}" not in notes:
            raise RuntimeError(f"note exclusion {w}: '{wd:%m-%d %H:%M:%S}' is not in the fault file's notes — NOTE_EXCL_WALLS is stale for this fault file")
        ivs = fault_intervals(faults, day); hit = fault_hit(wd, ivs); nid = f"N{i + 1}"
        if hit is not None:
            rep.append({"id": nid, "wall": w, "day": day, "already_excluded_by": hit}); continue
        prev_end = max((b for a, b, _ in ivs if b <= wd), default=None); next_start = min((a for a, b, _ in ivs if a > wd), default=None)
        lo, hi = wd - timedelta(seconds=max_s), wd + timedelta(seconds=max_s)
        a = max(prev_end, lo) if prev_end is not None else lo; b = min(next_start, hi) if next_start is not None else hi
        by_day[day].append((a, b, nid))
        rep.append({"id": nid, "wall": w, "day": day, "exclude_start": a.isoformat(), "exclude_end": b.isoformat(), "span_s": (b - a).total_seconds(),
                    "capped": bool((prev_end is None or prev_end < lo) or (next_start is None or next_start > hi)),
                    "source": "fault-file notes: unlisted but visible over-extended mask"})
    return dict(by_day), rep

def known_partial(faults, day):
    """BAD[day]: (wall_start, wall_end, slots) of faults whose kind contains 'partial' (A1-A5)."""
    return [(datetime.fromisoformat(f["wall_start"]), datetime.fromisoformat(f["wall_end"]), list(f["slots"]))
            for f in faults["faults"] if f["day"] == day and "partial" in f["kind"]]

def check_faults_file(faults, root, source_days, unlisted_policy):
    if not str(faults.get("semantics", "")).startswith("FAULT list"):
        raise RuntimeError(f"fault file semantics must start with 'FAULT list', got {faults.get('semantics')!r}")
    if int(faults.get("exclusion_margin_s", -1)) != 90:
        raise RuntimeError(f"exclusion_margin_s must be 90, got {faults.get('exclusion_margin_s')}")
    unlisted = [d for d in source_days if d not in faults.get("windows", {})]
    if len(unlisted) < len(source_days) and faults.get("tracking_root") != root:
        raise RuntimeError(f"fault file tracking_root {faults.get('tracking_root')!r} != root {root!r} (a listed day must come from the root the faults were marked on)")
    if unlisted and unlisted_policy != "gates_only":
        raise RuntimeError(f"source days {unlisted} absent from the fault file's windows; set unlisted_day_policy=gates_only to admit them on gates alone")
    return unlisted

def parse_event_line(line):
    try: e = json.loads(line)
    except Exception: return None
    return e if isinstance(e, dict) and "type" in e else None

def event_wall(e):
    """wall if present else clip_start(clip)+t (Tol_30:96-106 tolerance)."""
    if e.get("wall"):
        try: return datetime.fromisoformat(e["wall"])
        except Exception: pass
    if e.get("clip"):
        m = RX_EV.search(e["clip"])
        if m: return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") + timedelta(seconds=float(e.get("t", 0) or 0))
    return None

REPAIR_REASONS = ("partial_repair", "growth_repair")   # open/close reasons that exist only when Tol_03 ran partial_repair=true

def event_intervals(events, a_ref):
    """G2 buffers -> ([(start, end, label)], stats). Raises on partial_repair-era events (types AND
    open/close reasons), and on a buffer-bearing event whose wall clock cannot be resolved or a merge_resolved whose
    onset is present but unparsable (review 2026-09-23: such an event would silently lose its buffer)."""
    iv, density, n_nowall, n_by = [], Counter(), 0, Counter()
    for e in events:
        t = e["type"]
        if t in ("partial_state", "grown_state") or t.startswith("partial_repair") or t.startswith("growth_repair"):
            raise RuntimeError(f"event type {t!r} cannot exist in a partial_repair=false root — provenance error")
        if t in ("segment_open", "segment_close") and e.get("reason") in REPAIR_REASONS:
            raise RuntimeError(f"{t} with reason {e.get('reason')!r} cannot exist in a partial_repair=false root — provenance error")
        w = event_wall(e)
        if w is None:
            if t in CHAOS_TYPES or t in EXTRA_TYPES or t == "ignored_detection":
                raise RuntimeError(f"event {t!r} has no resolvable wall clock (wall={e.get('wall')!r}, clip={e.get('clip')!r}) — its G2 buffer would be lost silently")
            n_nowall += 1; continue
        buf = timedelta(seconds=EVENT_BUF_S)
        if t in CHAOS_TYPES or t in EXTRA_TYPES:
            iv.append((w - buf, w + buf, t)); n_by[t] += 1
        if t == "merge_resolved" and e.get("onset"):
            try: on = datetime.fromisoformat(e["onset"])
            except Exception: raise RuntimeError(f"merge_resolved at {w} has an unparsable onset {e.get('onset')!r} — the onset exclusion would be lost silently")
            iv.append((on - buf, w + buf, "merge_onset")); n_by["merge_onset"] += 1
        if t == "segment_close" and e.get("reason") in CONTRA_REASONS:
            iv.append((w - timedelta(seconds=CONTRA_PRE_S), w + buf, "close_" + e["reason"])); n_by["close_" + e["reason"]] += 1
        if t == "ignored_detection" and float(e.get("area", 0) or 0) >= VETO_AREA_FRAC * a_ref:
            iv.append((w - buf, w + buf, "ignored_detection_whole")); n_by["ignored_detection_whole"] += 1
        if t in CHAOS_TYPES:
            density[w.replace(second=0, microsecond=0)] += 1
    n_chaos = 0
    for m in sorted(density):
        if sum(density.get(m - timedelta(minutes=k), 0) for k in range(CHAOS_MIN)) >= CHAOS_EVENTS:
            iv.append((m - timedelta(minutes=CHAOS_MIN - 1), m + timedelta(minutes=1), "chaos")); n_chaos += 1
    return iv, {"n_events": len(events), "n_no_wall": n_nowall, "by_label": dict(n_by), "chaos_windows": n_chaos}

def merge_intervals(iv):
    iv = sorted((a, b) for a, b, _ in iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out

def union_seconds(iv): return sum((b - a).total_seconds() for a, b in merge_intervals(iv))

def in_intervals(wall, merged):
    """merged = sorted disjoint [(a,b)]; bisect on starts."""
    if not merged: return False
    i = bisect_right([a for a, _ in merged], wall) - 1
    return i >= 0 and merged[i][0] <= wall < merged[i][1]

class SegLookup:
    """seg_id at (slot, wall) from segments.json records (open records have no t_end_wall)."""
    def __init__(self, segments):
        self.by = {s: [] for s in range(N_CALVES)}
        for r in segments:
            a = datetime.fromisoformat(r["t_start_wall"])
            b = datetime.fromisoformat(r["t_end_wall"]) if r.get("t_end_wall") else datetime.max
            self.by[int(r["slot"])].append((a, b, r["seg_id"]))
        for s in self.by: self.by[s].sort()
        self.starts = {s: [a for a, _, _ in self.by[s]] for s in self.by}
    def at(self, slot, wall):
        i = bisect_right(self.starts[slot], wall) - 1
        if i < 0: return None
        a, b, sid = self.by[slot][i]
        return sid if a <= wall <= b else None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pure functions — per-row measurements (pass A), gates G3–G10, derived constants

# COMMAND ----------

def measure_clip(frames, clip_name, window, seg, stride=5):
    """Pass A on one clip: 1-Hz rows with area / centroid / link IoU / pairwise overlap /
    seg_id / stability ratio / activity score, all from RLE runs (no decode)."""
    day = window[:8]; w0 = clip_start(clip_name); stem = clip_stem_of(clip_name)
    rows = []
    cache = {}
    def runs_at(i):
        if i in cache: return cache[i]
        r = frames[i]; out = []
        for s in range(N_CALVES):
            rl = (r.get("masks") or {}).get(str(s))
            if rl:
                if list(rl["size"]) != [H_FRAME, W_FRAME]:
                    raise RuntimeError(f"{stem} frame {r['frame_idx']} slot {s}: mask size {rl['size']} != [{H_FRAME},{W_FRAME}]")
                out.append(runs_of(rl))
            else: out.append(None)
        if len(cache) > 8: cache.clear()
        cache[i] = out; return out
    for i in range(0, len(frames), stride):
        r = frames[i]; wall = w0 + timedelta(seconds=r["t_sec"])
        cur = runs_at(i)
        prv = runs_at(i - 1) if i > 0 else [None] * N_CALVES
        nxt = runs_at(i + 1) if i + 1 < len(frames) else [None] * N_CALVES
        area, link, cent, segid = [None] * 4, [None] * 4, [None] * 4, [None] * 4
        for s in range(N_CALVES):
            if cur[s] is None: continue
            area[s] = cur[s][2]
            c = centroid_from_runs(cur[s], H_FRAME); cent[s] = [round(c[0], 1), round(c[1], 1)]
            if prv[s] is not None and nxt[s] is not None:
                link[s] = round(min(runs_iou(cur[s], prv[s]), runs_iou(cur[s], nxt[s])), 4)
            segid[s] = seg.at(s, wall) if seg is not None else None
        pres = [s for s in range(N_CALVES) if cur[s] is not None]
        piou = pios = 0.0
        for a in range(len(pres)):
            for b in range(a + 1, len(pres)):
                it = inter(cur[pres[a]], cur[pres[b]])
                A, B = cur[pres[a]][2], cur[pres[b]][2]
                if it:
                    piou = max(piou, it / (A + B - it)); pios = max(pios, it / min(A, B))
        rows.append({"window": window, "day": day, "clip_stem": stem, "row_i": i, "frame_idx": int(r["frame_idx"]),
                     "t_sec": float(r["t_sec"]), "wall": wall, "phase4": phase4(wall), "n_present": len(pres),
                     "slots": pres, "area": area, "link": link, "cent": cent, "seg": segid,
                     "pair_iou": round(piou, 4), "pair_ios": round(pios, 4)})
    add_temporal(rows)
    return rows

def add_temporal(rows):
    """G8 stability ratio (area / median over ±STAB_WIN_S, same slot, same seg) and the activity score
    (mean centroid displacement between t-5 s and t+5 s over slots present in both) on one clip's 1-Hz rows."""
    ts = np.array([r["t_sec"] for r in rows], float)
    for k, r in enumerate(rows):
        lo = int(np.searchsorted(ts, ts[k] - STAB_WIN_S)); hi = int(np.searchsorted(ts, ts[k] + STAB_WIN_S, side="right"))
        stab = [None] * N_CALVES
        for s in range(N_CALVES):
            if r["area"][s] is None: continue
            vals = [rows[j]["area"][s] for j in range(lo, hi)
                    if rows[j]["area"][s] is not None and rows[j]["seg"][s] == r["seg"][s]]
            if len(vals) >= 3:
                med = float(np.median(vals)); stab[s] = round(r["area"][s] / med, 4) if med > 0 else None
        r["stab"] = stab
        jm = int(np.searchsorted(ts, ts[k] - ACT_HALF_S)); jp = int(np.searchsorted(ts, ts[k] + ACT_HALF_S))
        act = None
        if jm < len(rows) and jp < len(rows) and abs(ts[jm] - (ts[k] - ACT_HALF_S)) <= 1.0 and abs(ts[jp] - (ts[k] + ACT_HALF_S)) <= 1.0:
            d = []
            for s in range(N_CALVES):
                a, b = rows[jm]["cent"][s], rows[jp]["cent"][s]
                if a is not None and b is not None: d.append(math.hypot(a[0] - b[0], a[1] - b[1]))
            if d: act = round(float(np.mean(d)), 2)
        r["act"] = act

def a_ref_of(rows, day):
    """DERIVED(day): median of the largest mask over the day's phase3=='day' 1-Hz rows — Tol_40:141-142."""
    dayl = [max(a for a in r["area"] if a is not None) for r in rows
            if r["day"] == day and r["n_present"] > 0 and phase3(r["wall"]) == "day"]
    if not dayl: raise RuntimeError(f"{day}: no daylight rows — cannot anchor A_ref")
    return float(np.median(dayl))

def gate_pre(r, fault_iv, ev_merged):
    """G1..G5 in order; returns the first failing gate name or None."""
    if fault_hit(r["wall"], fault_iv) is not None: return "G1"
    if in_intervals(r["wall"], ev_merged): return "G2"
    if r["n_present"] != N_CALVES: return "G3"
    if r["pair_iou"] > DISJ_IOU or r["pair_ios"] > DISJ_IOS: return "G4"
    if any(l is None or l < LINK_IOU_MIN for l in r["link"]): return "G5"
    return None

def ratios(r, a_ref): return [a / a_ref for a in r["area"]]

def gate_post(r, a_ref, C, dropped_blocks):
    """G6..G9 in order on a row that passed G0-G5. C = derived constants dict."""
    pool = pool_of(r["phase4"]); rr = ratios(r, a_ref)
    if any(x < C["frag_lo"][pool] or x > C["over_hi"] for x in rr): return "G6"
    S = sum(rr) / N_CALVES
    if S < C["S_lo"][pool] or S > C["S_hi"][pool]: return "G7"
    if any(v is None or v < C["STAB_LO"] or v > C["STAB_HI"] for v in r["stab"]): return "G8"
    if r["phase4"] in ("dusk", "night") and (r["clip_stem"], r["phase4"]) in dropped_blocks: return "G9"
    return None

def largest_cc_frac(mask):
    """G10: after a CC_OPEN_PX binary opening, area of the largest 8-connected component / mask area.
    Computed inside the bbox (+margin) for speed."""
    bb = m_bbox(mask)
    if bb is None: return 0.0
    x1, y1, x2, y2 = bb; pad = CC_OPEN_PX + 1
    sub = mask[max(0, y1 - pad):y2 + pad + 1, max(0, x1 - pad):x2 + pad + 1]
    op = ndimage.binary_opening(sub, structure=np.ones((CC_OPEN_PX, CC_OPEN_PX), bool))
    lab, n = ndimage.label(op, structure=np.ones((3, 3), int))
    if n == 0: return 0.0
    sizes = ndimage.sum(op, lab, index=np.arange(1, n + 1))
    return float(np.max(sizes)) / float(mask.sum())

def body_len_of(mask):
    """4*sqrt(lambda_max) of the pixel-coordinate covariance = full major-axis extent."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3: return None
    cov = np.cov(np.stack([xs, ys]).astype(np.float64))
    return float(4.0 * math.sqrt(max(np.linalg.eigvalsh(cov))))

def _block_stats(rs, a_ref):
    big = float(np.median([max(a for a in r["area"] if a is not None) / a_ref for r in rs]))
    slot_med = {}
    for s in range(N_CALVES):
        v = [r["area"][s] / a_ref for r in rs if r["area"][s] is not None]
        if v: slot_med[s] = float(np.median(v))
    return big, slot_med

def block_collapse(rows_adm, a_ref_by_day, frag_lo_dark, overrides=(), block_has_fault=None, rows_all=None):
    """G9: blocks = (clip_stem, phase4 in {dusk,night}) over rows with >=1 mask. rows_adm must be
    the rows that passed G1 (fault list) and G2 (event buffers): slivers excluded by the fault list must not decide
    whether the clean remainder of a block is dropped (review 2026-09-22). rows_all (optional) gives the
    all-row medians for the report only, so the difference is visible.
    (a) block median of the LARGEST ratio < BLOCK_LARGEST_LO -> drop; (b) any slot's block median
    ratio < frag_lo[dark] -> drop; (c) slots with median in [frag_lo[dark], 0.75) are curled_or_clipped.
    Returns (dropped set, report dict)."""
    blocks, blocks_all = defaultdict(list), defaultdict(list)
    for r in rows_adm:
        if r["phase4"] in ("dusk", "night") and r["n_present"] > 0: blocks[(r["clip_stem"], r["phase4"])].append(r)
    for r in (rows_all or []):
        if r["phase4"] in ("dusk", "night") and r["n_present"] > 0: blocks_all[(r["clip_stem"], r["phase4"])].append(r)
    dropped, curled, fail_slot075, needs_human, detail = set(), [], 0, [], {}
    ov = set(overrides); bhf = block_has_fault or {}
    for key, rs in sorted(blocks.items()):
        a_ref = a_ref_by_day[rs[0]["day"]]
        big, slot_med = _block_stats(rs, a_ref)
        why = None
        if big < BLOCK_LARGEST_LO: why = f"largest-median {big:.3f} < {BLOCK_LARGEST_LO}"
        elif any(m < frag_lo_dark for m in slot_med.values()):
            why = f"slot median below frag_lo[dark]={frag_lo_dark}: " + str({s: round(m, 3) for s, m in slot_med.items() if m < frag_lo_dark})
        if any(m < BLOCK_LARGEST_LO for m in slot_med.values()): fail_slot075 += 1
        cur = [s for s, m in slot_med.items() if frag_lo_dark <= m < BLOCK_LARGEST_LO]
        name = f"{key[0]}:{key[1]}"
        detail[name] = {"n_rows": len(rs), "largest_median": round(big, 3), "slot_median": {str(s): round(m, 3) for s, m in slot_med.items()},
                        "has_user_fault": bool(bhf.get(key, False)), "dropped": False, "why": why, "override": name in ov}
        if key in blocks_all:
            big_a, slot_a = _block_stats(blocks_all[key], a_ref)
            detail[name]["all_rows"] = {"n_rows": len(blocks_all[key]), "largest_median": round(big_a, 3), "slot_median": {str(s): round(m, 3) for s, m in slot_a.items()}}
        if why and name in ov:
            detail[name]["why"] = why + " (RE-ADMITTED by collapse_override_blocks)"
        elif why:
            dropped.add(key); detail[name]["dropped"] = True
            if not detail[name]["has_user_fault"]: needs_human.append(name)
        if cur and key not in dropped:
            curled.append({"block": name, "slots": cur, "slot_median": {str(s): round(slot_med[s], 3) for s in cur}})
    return dropped, {"n_blocks": len(blocks), "rows_basis": "rows passing G1 (fault list) and G2 (event buffers)", "dropped": sorted(f"{k[0]}:{k[1]}" for k in dropped),
                     "curled_or_clipped_blocks": curled, "blocks_failing_per_slot_0.75": fail_slot075, "needs_human_verdict": needs_human, "detail": detail}

def derive_constants(rows_u, a_ref_by_day, over_hi, curl_start, curl_end, override=None, curl_required=True, curl_context=None):
    """Band constants from the admitted set U (G0-G5 survivors, training days only).
    Raises on the sanity / per-day / curled checks unless override={'dayl':x,'dark':y} is given.
    curl_required: the curled stretch's day is a source day, so fewer than CURL_MIN_ROWS U rows inside it
    means the check it exists for cannot run -> raise (never overridable; curl_context names the gates that
    removed the stretch)."""
    R = {"dayl": [], "dark": []}; Rday = defaultdict(lambda: {"dayl": [], "dark": []})
    Sv = {"dayl": [], "dark": []}; stabv = []; act_by_day = defaultdict(list)
    c0, c1 = datetime.fromisoformat(curl_start), datetime.fromisoformat(curl_end)
    curl_rows = []
    for r in rows_u:
        if r["day"] not in TRAIN_DAYS: continue          # a 04-22 source contributes no constant
        a_ref = a_ref_by_day[r["day"]]; pool = pool_of(r["phase4"]); rr = ratios(r, a_ref)
        R[pool].extend(rr); Rday[r["day"]][pool].extend(rr); Sv[pool].append(sum(rr) / N_CALVES)
        stabv.extend(v for v in r["stab"] if v is not None)
        if r["act"] is not None: act_by_day[r["day"]].append(r["act"])
        if c0 <= r["wall"] < c1: curl_rows.append(rr)
    for pool in R:
        if len(R[pool]) < 40: raise RuntimeError(f"pool {pool}: only {len(R[pool])} admitted ratios — cannot derive a lower edge")
    if curl_required and len(curl_rows) < CURL_MIN_ROWS:
        raise RuntimeError(f"curled-stretch check would be vacuous: only {len(curl_rows)} U rows inside {curl_start}..{curl_end} (need >= {CURL_MIN_ROWS}); "
                           f"the reviewer-confirmed stretch (which the band must keep admitting) was removed upstream — first-failing gate per 1-Hz row inside it: {curl_context}")
    C = {"over_hi": float(over_hi), "frag_lo": {p: floor2(pct(R[p], 1)) for p in R},
         "frag_lo_per_day": {d: {p: (round(pct(v[p], 1), 4) if len(v[p]) >= 40 else None) for p in v} for d, v in Rday.items()},
         "S_lo": {p: floor2(pct(Sv[p], 1)) for p in Sv}, "S_hi": {p: ceil2(pct(Sv[p], 99)) for p in Sv},
         "STAB_LO": round(pct(stabv, 0.5), 4), "STAB_HI": round(pct(stabv, 99.5), 4),
         "ACTIVE_THR": {d: round(pct(v, 70), 2) for d, v in act_by_day.items()},
         "frag_lo_provenance": "derived_P1_of_U"}
    problems = []
    for p, v in C["frag_lo"].items():
        if not (FRAG_LO_SANITY[0] <= v <= FRAG_LO_SANITY[1]):
            problems.append(f"frag_lo[{p}]={v} outside sanity {FRAG_LO_SANITY}")
    for p in R:
        vals = [v[p] for v in C["frag_lo_per_day"].values() if v[p] is not None]
        if len(vals) >= 2 and (max(vals) - min(vals)) > FRAG_LO_DAY_AGREE:
            problems.append(f"per-day P1 for pool {p} disagree by {max(vals)-min(vals):.3f} > {FRAG_LO_DAY_AGREE}: {vals}")
    def curl_check(fl_dark):
        if not curl_rows: return None
        ok = sum(1 for rr in curl_rows if all(fl_dark <= x <= over_hi for x in rr))
        return ok / len(curl_rows)
    cc = curl_check(C["frag_lo"]["dark"])
    if cc is not None and cc < CURL_ADMIT_MIN:
        problems.append(f"curled-stretch admission {cc:.3f} < {CURL_ADMIT_MIN} with frag_lo[dark]={C['frag_lo']['dark']}")
    C["curled_check"] = {"n_rows": len(curl_rows), "admitted_frac": None if cc is None else round(cc, 4)}
    dist = {}
    for p in R:
        v = np.asarray(R[p]); hist, edges = np.histogram(v, bins=np.arange(0.0, 2.55, 0.05))
        dist[p] = {"n": int(len(v)), "pctiles": {str(q): round(pct(v, q), 4) for q in (0.1, 0.5, 1, 2.5, 5, 25, 50, 75, 95, 99, 99.5, 99.9)},
                   "hist": [int(x) for x in hist], "hist_edges": [round(float(e), 2) for e in edges]}
    for d, v in Rday.items():
        for p in v:
            if v[p]: dist[f"{d}/{p}"] = {"n": len(v[p]), "pctiles": {str(q): round(pct(v[p], q), 4) for q in (0.1, 0.5, 1, 5, 50, 95, 99, 99.5)}}
    allR = np.asarray(R["dayl"] + R["dark"])
    C["upper_diag"] = {"P99": round(pct(allR, 99), 4), "P99_5": round(pct(allR, 99.5), 4),
                       "upper_edge_clips_admitted_mass": bool(pct(allR, 99.5) > over_hi)}
    C["distribution"] = dist; C["problems"] = problems
    if problems:
        if override is None:
            msg = "\n".join(["band derivation FAILED its checks:"] + [f"  - {p}" for p in problems] +
                            ["distributions (pool: P0.1/P0.5/P1/P2.5/P5/P50):"] +
                            [f"  {p}: " + " ".join(f"{dist[p]['pctiles'][q]}" for q in ('0.1','0.5','1','2.5','5','50')) for p in R] +
                            ["set frag_lo_override='dayl=0.xx,dark=0.yy' with these in hand"])
            raise RuntimeError(msg)
        # an override is also G9(b)'s block-collapse threshold, so it must stay inside the sanity band (a typo such
        # as 0.045 would silently disable sliver rejection) and must still admit the curled stretch (review 2026-09-23)
        for p in ("dayl", "dark"):
            v = float(override[p])
            if not (FRAG_LO_SANITY[0] <= v <= FRAG_LO_SANITY[1]):
                raise RuntimeError(f"frag_lo_override {p}={v} outside the sanity band {FRAG_LO_SANITY} — an override may move the edge only within it")
        C["frag_lo_derived_rejected"] = dict(C["frag_lo"])
        C["frag_lo"] = {"dayl": float(override["dayl"]), "dark": float(override["dark"])}
        C["frag_lo_provenance"] = "user_override"
        cc2 = curl_check(C["frag_lo"]["dark"])
        C["curled_check"]["admitted_frac_override"] = None if cc2 is None else round(cc2, 4)
        if cc2 is not None and cc2 < CURL_ADMIT_MIN:
            raise RuntimeError(f"frag_lo_override dark={C['frag_lo']['dark']} admits only {cc2:.3f} < {CURL_ADMIT_MIN} of the reviewer-confirmed curled stretch "
                               "— an override must not reject the case the band was relaxed for")
    elif override is not None:
        raise RuntimeError("frag_lo_override given but the derivation did not raise — remove the override (provenance must stay derived)")
    return C

def bad_overlap(rows_all, bad_by_day, a_ref_by_day, C):
    """P(BAD > frag_lo[pool]) over the partial masks listed in the fault file (diagnostic)."""
    vals = []
    for r in rows_all:
        for a, b, slots in bad_by_day.get(r["day"], []):
            if a <= r["wall"] < b:
                for s in slots:
                    if r["area"][s] is not None: vals.append((r["area"][s] / a_ref_by_day[r["day"]], pool_of(r["phase4"])))
    if not vals: return {"n": 0, "frac_admitted": None, "high": False}
    adm = sum(1 for v, p in vals if v > C["frag_lo"][p])
    return {"n": len(vals), "frac_admitted": round(adm / len(vals), 4), "high": adm / len(vals) > BAD_OVERLAP_HI}

def link_iou_report(rows_g1_g4):
    """G5 support on the TRAINING days (review 2026-09-23; training-only constants): the per-slot 5 fps link IoU over rows that
    passed G1-G4 (the population G5 acts on), by pool. Raises unless P1 of every pool (and overall) > LINK_IOU_MIN,
    i.e. the 0.50 edge sits below the same-calf mass, so G5 removes flicker and not a posture. None links (a missing
    neighbour) are counted separately."""
    vals, n_none = {"dayl": [], "dark": []}, 0
    for r in rows_g1_g4:
        if r["day"] not in TRAIN_DAYS: continue
        for l in r["link"]:
            if l is None: n_none += 1
            else: vals[pool_of(r["phase4"])].append(l)
    vals["all"] = vals["dayl"] + vals["dark"]
    rep = {"basis": "per-slot link IoU (min over both 5 fps neighbours) of training-day 1-Hz rows that passed G1-G4", "n_none": n_none,
           "edge": LINK_IOU_MIN, "check": f"P{LINK_VALLEY_PCT:g} > {LINK_IOU_MIN}", "by_pool": {}}
    bad = []
    for k, v in vals.items():
        if not v: continue
        v = np.asarray(v, float); p = pct(v, LINK_VALLEY_PCT)
        rep["by_pool"][k] = {"n": int(len(v)), "frac_below_edge": round(float((v < LINK_IOU_MIN).mean()), 5),
                             "pctiles": {str(q): round(pct(v, q), 4) for q in (0.1, 0.5, 1, 5, 25, 50)}}
        if not p > LINK_IOU_MIN: bad.append(f"{k}: P{LINK_VALLEY_PCT:g}={p:.4f}")
    if not vals["all"]: raise RuntimeError("link_iou_report: no training-day link IoU values — G5's edge cannot be supported")
    if bad:
        raise RuntimeError(f"G5 edge LINK_IOU_MIN={LINK_IOU_MIN} is NOT below the training-day same-calf link mass ({'; '.join(bad)}); "
                           f"G5 would remove > {LINK_VALLEY_PCT:g} % of G1-G4 survivors' links. Distribution: {rep['by_pool']}. A protocol decision is needed before this gate is used.")
    return rep

def curl_admission(first_fail_in_stretch):
    """Fraction of the certified curled stretch's U rows that survive G6-G9 (None = admitted), with the first-failing
    gate of the rest (review 2026-09-23: the curled stretch must stay admitted after the WHOLE band, not only after G6)."""
    c = Counter((g or "pass") for g in first_fail_in_stretch)
    n = sum(c.values())
    return {"n_rows": n, "admitted_frac": round(c["pass"] / n, 4) if n else None, "first_fail": dict(c)}

def curl_band_check(stretch_ff, ref_ff, ack=""):
    """Curled-stretch admission after the WHOLE band (G6-G9), RELATIVE (review 2026-09-23). G7/G8 are percentile trims of U by
    construction (P1/P99 of S; P0.5/P99.5 of per-mask stability over 4 slots), so even an average U stretch keeps only
    ~0.86-0.93 after G6-G9 and the absolute 0.90 (specified for G6 alone, still enforced in derive_constants)
    cannot be reused as-is. The question here is whether G6-G9 reject the certified curled stretch DISPROPORTIONATELY:
    fails iff admitted(stretch) < CURL_REL_MIN x admitted(ref), ref = training-day dark-pool U rows OUTSIDE the stretch.
    Returns the report with 'blocking' = fails and no ack. An ack given while the check passes raises (the override
    must stay tied to an observed failure, like frag_lo_override). The caller raises on 'blocking' only AFTER the
    sheets (incl. curl_rejected) are written, so the decision is made with the rejected stretch rows in hand."""
    st, ref = curl_admission(stretch_ff), curl_admission(ref_ff)
    rep = {**st, "ref_dark_outside_stretch": ref, "rel_min": CURL_REL_MIN, "abs_min_reported_only": CURL_ADMIT_MIN,
           "below_abs_0.90": bool(st["admitted_frac"] is not None and st["admitted_frac"] < CURL_ADMIT_MIN),
           "relative": None, "fails": False, "ack": ack or None, "blocking": False}
    if st["admitted_frac"] is not None and ref["admitted_frac"] is not None:
        rep["relative"] = round(st["admitted_frac"] / ref["admitted_frac"], 4) if ref["admitted_frac"] > 0 else None
        rep["fails"] = bool(st["admitted_frac"] < CURL_REL_MIN * ref["admitted_frac"] or ref["admitted_frac"] == 0)
    if ack and not rep["fails"]:
        raise RuntimeError(f"curl_band_ack given but the relative curled-stretch check did not fail ({rep['admitted_frac']} vs ref {ref['admitted_frac']}) — remove the ack")
    rep["blocking"] = bool(rep["fails"] and not ack)
    return rep

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pure functions — sampling, split, crops, COCO, seal, plates, position prior

# COMMAND ----------

def lattice(rows, stride_day, stride_other):
    """Sampling step 1 on ONE clip's admitted rows (sorted by wall): ticks every stride s of wall time
    (stride by phase4 of the tick) from the first admitted row; first admitted row within LATTICE_TOL_S
    at/after each tick, else the tick is skipped."""
    if not rows: return []
    rows = sorted(rows, key=lambda r: r["wall"])
    walls = [r["wall"] for r in rows]
    picks, t, last = [], walls[0], walls[-1]
    tol = timedelta(seconds=LATTICE_TOL_S)
    while t <= last:
        i = bisect_right(walls, t - timedelta(microseconds=1))
        if i < len(rows) and walls[i] < t + tol: picks.append(rows[i])
        t = t + timedelta(seconds=stride_day if phase4(t) == "day" else stride_other)
    return picks

def cross_clip_dedupe(picks, min_gap_s):
    """Consecutive Tol clip files overlap in wall time by 0.5-3.4 s (measured on the 23 training index files, 04-18/04-19),
    and lattice() runs per clip, so one instant can be picked from BOTH files. Sorted by (day, wall), a pick that lies
    < min_gap_s after an already-kept pick from a DIFFERENT clip of the same day is dropped (the later one). Picks of the
    same clip are never compared (their spacing is the lattice's). Returns (kept in the input order, n_dropped)."""
    order = sorted(range(len(picks)), key=lambda i: (picks[i]["day"], picks[i]["wall"]))
    kept_idx, drop = [], set()
    for i in order:
        r = picks[i]; j = len(kept_idx) - 1
        while j >= 0:
            k = picks[kept_idx[j]]
            if k["day"] != r["day"] or (r["wall"] - k["wall"]).total_seconds() >= min_gap_s: break
            if k["clip_stem"] != r["clip_stem"]: drop.add(i); break
            j -= 1
        if i not in drop: kept_idx.append(i)
    return [r for i, r in enumerate(picks) if i not in drop], len(drop)

def split_clips(all_stems, valid_stems):
    valid = [s for s in valid_stems if s]
    missing = sorted(set(valid) - set(all_stems))
    if missing: raise RuntimeError(f"valid_clips not among the source clips: {missing}")
    return sorted(set(all_stems) - set(valid)), sorted(valid)

def valid_buffer_drop(train_rows, valid_bounds, buffer_min):
    """G12: drop train rows within buffer_min of any valid clip's [start, end] wall bounds (same day)."""
    b = timedelta(minutes=buffer_min); keep, dropped = [], 0
    for r in train_rows:
        hit = any(r["day"] == d and (a - b) <= r["wall"] <= (e + b) for d, a, e in valid_bounds)
        if hit: dropped += 1
        else: keep.append(r)
    return keep, dropped

def allocate(rows, n_max, quotas, active_thr, seed):
    """Sampling step 3: supply <= n_max -> everything; else greedy night -> dusk -> dawn -> active ->
    uniform fill, seeded. Returns (selected rows, report)."""
    rng = random.Random(seed)
    for r in rows: r["active"] = bool(r.get("act") is not None and active_thr.get(r["day"]) is not None and r["act"] >= active_thr[r["day"]])
    rep = {"supply": len(rows), "n_max": n_max, "binding": len(rows) > n_max}
    if len(rows) <= n_max:
        sel = list(rows)
    else:
        pool = list(rows); rng.shuffle(pool); sel, used = [], set()
        def take(pred, n):
            got = 0
            for r in pool:
                if got >= n: break
                if id(r) in used or not pred(r): continue
                sel.append(r); used.add(id(r)); got += 1
            return got
        for ph in ("night", "dusk", "dawn"):
            take(lambda r, ph=ph: r["phase4"] == ph, int(round(quotas[ph] * n_max)))
        take(lambda r: r["active"], int(round(quotas["active"] * n_max)))
        take(lambda r: True, n_max - len(sel))
    ach, short = quota_achieved(sel, quotas)
    rep.update({"achieved": ach, "shortfall": short, "selected": len(sel)})
    return sel, rep

def quota_achieved(sel, quotas):
    """Achieved phase4/active shares and the unmet quotas of ANY frame set (rows or admitted-frame dicts);
    recomputed after hour_cap and again after finalize's human/G11 drops (review 2026-09-22)."""
    n = max(len(sel), 1)
    ach = {ph: sum(1 for r in sel if r["phase4"] == ph) / n for ph in ("night", "dusk", "dawn", "day")}
    ach["active"] = sum(1 for r in sel if r.get("active")) / n
    short = {}
    for k in ("night", "dusk", "dawn", "active"):
        need = quotas[k] * len(sel); have = ach[k] * len(sel)
        if have + 0.5 < need: short[k] = {"have": int(round(have)), "need": int(round(need))}
    return {k: round(v, 4) for k, v in ach.items()}, short

def hour_cap(rows, frac, seed, strict=True):
    """Sampling step 4: no (day, hour) bin above frac of the FINAL set; excess trimmed at random (seeded).
    Iterated to a fixed point: trimming shrinks the set, which lowers the cap, so a single pass computed from
    the pre-trim size leaves capped bins above frac (review 2026-09-22). Raises if the cap is infeasible
    (fewer than ceil(1/frac) non-empty bins drive it to zero); strict=False (SMOKE) stops there with a ⚠ instead."""
    rng = random.Random(seed + 1)
    bins = defaultdict(list)
    for r in rows: bins[(r["day"], r["wall"].hour)].append(r)
    for k in bins: rng.shuffle(bins[k])          # one seeded order per bin; each pass keeps a prefix of it
    keep = {k: len(v) for k, v in bins.items()}
    while True:
        n = sum(keep.values()); cap = int(math.floor(frac * n))
        if n and cap <= 0:
            msg = f"hour cap {frac} infeasible: {n} rows in {sum(1 for v in keep.values() if v)} (day, hour) bins"
            if strict: raise RuntimeError(msg)
            print(f"⚠ {msg} — cap not enforced (SMOKE)"); break
        over = [k for k, v in keep.items() if v > cap]
        if not over: break
        for k in over: keep[k] = cap
    out, trimmed = [], {}
    for k in sorted(bins):
        if keep[k] < len(bins[k]): trimmed[f"{k[0]}_h{k[1]:02d}"] = len(bins[k]) - keep[k]
        out.extend(bins[k][:keep[k]])
    return out, trimmed

def max_hour_share(rows):
    """Largest (day, hour) bin share of a frame set (rows with datetime 'wall' or dicts with ISO 'wall')."""
    if not rows: return 0.0
    c = Counter((r["day"], (r["wall"] if isinstance(r["wall"], datetime) else datetime.fromisoformat(r["wall"])).hour) for r in rows)
    return max(c.values()) / len(rows)

def distinct_10min_blocks(rows):
    return len({(r["day"], int(r["wall"].timestamp() // 600)) for r in rows})

def draw_crop_window(masks, rng, cw, ch, allow_zero=True):
    """Crop origin uniform in [0, W-cw] x [0, H-ch]; accept only if every mask the window touches
    is either >= CROP_KEEP_FRAC inside (kept, clipped) or <= CROP_DROP_FRAC inside (dropped); up to
    CROP_MAX_DRAWS draws. masks = list of bool [H,W] (None allowed). Returns (x, y, kept_slots) or None."""
    H, W = None, None
    for m in masks:
        if m is not None: H, W = m.shape; break
    if H is None: return None
    areas = [int(m.sum()) if m is not None else 0 for m in masks]
    for _ in range(CROP_MAX_DRAWS):
        x = rng.randint(0, W - cw); y = rng.randint(0, H - ch)
        kept, ok = [], True
        for s, m in enumerate(masks):
            if m is None or areas[s] == 0: continue
            f = int(m[y:y + ch, x:x + cw].sum()) / areas[s]
            if f >= CROP_KEEP_FRAC: kept.append(s)
            elif f > CROP_DROP_FRAC: ok = False; break
        if not ok: continue
        if not kept and not allow_zero: continue
        return x, y, kept
    return None

def crop_mask_rle(mask, x, y, cw, ch):
    """rle_decode -> slice -> rle_encode; the caller asserts exact decode equality."""
    sub = np.ascontiguousarray(mask[y:y + ch, x:x + cw])
    r = rle_encode(sub)
    if not np.array_equal(rle_decode(r), sub): raise RuntimeError("crop RLE re-encode is not exact")
    return r

def coco_annotation(ann_id, image_id, stored_rle, h, w, slot, seg_id, maskUtils):
    """One INSTANCE annotation: compressed RLE via frPyObjects, area/bbox from pycocotools,
    both asserted against the stored counts / decoded mask."""
    if list(stored_rle["size"]) != [h, w]: raise RuntimeError(f"segmentation.size {stored_rle['size']} != [{h},{w}]")
    enc = maskUtils.frPyObjects({"size": [h, w], "counts": list(stored_rle["counts"])}, h, w)
    area = int(maskUtils.area(enc)); bb = [float(v) for v in maskUtils.toBbox(enc)]
    if area != rle_area(stored_rle): raise RuntimeError(f"pycocotools area {area} != stored {rle_area(stored_rle)}")
    dec = rle_decode(stored_rle); mb = m_bbox(dec)
    if mb is None: raise RuntimeError("empty mask in an annotation")
    exp = [mb[0], mb[1], mb[2] - mb[0] + 1, mb[3] - mb[1] + 1]
    if [int(round(v)) for v in bb] != exp: raise RuntimeError(f"toBbox {bb} != decoded bbox {exp}")
    if bb[2] <= 0 or bb[3] <= 0: raise RuntimeError("bbox with non-positive size")
    counts = enc["counts"].decode("ascii") if isinstance(enc["counts"], bytes) else str(enc["counts"])
    return {"id": int(ann_id), "image_id": int(image_id), "category_id": 1, "iscrowd": 0,
            "segmentation": {"size": [h, w], "counts": counts}, "area": area, "bbox": bb,
            "slot": int(slot), "seg_id": seg_id}

def coco_check(coco, expect_full_n=4):
    """Structural asserts on an assembled COCO dict."""
    if coco["categories"] != [{"id": 1, "name": "calf", "supercategory": "none"}]: raise RuntimeError("categories must be exactly one 'calf'")
    names = [im["file_name"] for im in coco["images"]]
    if len(set(names)) != len(names): raise RuntimeError("duplicate file_name in a split")
    ids = {im["id"]: im for im in coco["images"]}
    if len(ids) != len(coco["images"]): raise RuntimeError("duplicate image id")
    cnt = Counter(a["image_id"] for a in coco["annotations"])
    for a in coco["annotations"]:
        if "segmentation" not in a or not isinstance(a["segmentation"]["counts"], str): raise RuntimeError(f"annotation {a['id']} lacks str-counts segmentation")
        im = ids.get(a["image_id"])
        if im is None: raise RuntimeError(f"annotation {a['id']} references unknown image {a['image_id']}")
        if a["segmentation"]["size"] != [im["height"], im["width"]]: raise RuntimeError(f"annotation {a['id']} size {a['segmentation']['size']} != image {[im['height'], im['width']]}")
        if a["bbox"][2] <= 0 or a["bbox"][3] <= 0: raise RuntimeError(f"annotation {a['id']} bbox w/h <= 0")
    for im in coco["images"]:
        n = cnt.get(im["id"], 0)
        if im.get("crop_variant"):
            if not (0 <= n <= N_CALVES) or n != im.get("n_visible", n): raise RuntimeError(f"crop image {im['id']} has {n} annotations, n_visible {im.get('n_visible')}")
        elif n != expect_full_n: raise RuntimeError(f"full image {im['id']} has {n} annotations, expected {expect_full_n}")
    aid = [a["id"] for a in coco["annotations"]]
    if len(set(aid)) != len(aid): raise RuntimeError("duplicate annotation id")
    return True

DENSE_STRIDES = (8.0, 4.0)                     # sampling step 5 option (a), chosen FIRST (decision 2026-09-23); not derived from any day

def below_floor_message(n_train, min_train, when, stride_day, stride_other):
    """The printed below-floor options, in the chosen order (sampling step 5; decision 2026-09-23)."""
    dense = (float(stride_day), float(stride_other)) == DENSE_STRIDES
    a = (f"   (a) [FIRST CHOICE, 2026-09-23] re-run with stride_day_s={DENSE_STRIDES[0]:g}, stride_other_s={DENSE_STRIDES[1]:g} "
         f"under a NEW tag (keeps 04-22 in the detector test set)")
    if dense: a += f" — ALREADY APPLIED in this run (strides {stride_day:g}/{stride_other:g}), so the next step is (b)"
    return (f"⚠ BELOW FLOOR {when}: {n_train} train full frames < min_train {min_train}. Everything is still written; the manifest is UNSEALED "
            f"(below_floor=true). The protocol is to DENSIFY FIRST (option (a) before (b); decision 2026-09-23). "
            f"Options, in the chosen order:\n{a}\n"
            f"   (b) only if (a) is still short: add Tracking_teacher:20240422 to sources with allow_test_day_reason "
            f"(04-22 then LEAVES the detector test set; the Tol_44 evaluation keeps only 04-23)")

TOL43_VALID_NIGHT_MIN = 50                     # = Tol_43 VALID_NIGHT_MIN (project choice, review 2026-09-23, valid clips of 04-18/04-19); NOT a seal term here

def night_valid_message(valid_rows, when, night_min=TOL43_VALID_NIGHT_MIN):
    """None when valid holds >= night_min night frames; else the ⚠ line saying Tol_43 will refuse this export at the
    start of its A100 job unless allow_thin_night_valid=true (the seal is unaffected: Tol_43 re-derives it)."""
    n = sum(1 for r in valid_rows if r.get("phase4") == "night")
    if n >= night_min: return None
    return (f"⚠ THIN NIGHT VALID {when}: {n} night valid frames < {night_min} (Tol_43 VALID_NIGHT_MIN). The seal is unaffected, but Tol_43 "
            f"REFUSES this export unless allow_thin_night_valid=true — checkpoint selection and op_threshold would see (almost) no night")

HG_VERDICTS = ("absent", "pending", "passed", "failed")

def human_gate_verdict(present, status, n_bad_counted, n_wrong_object, n_counted, sheet_n_planned):
    """The human gate recomputed from the verdicts (the reviewer's status is necessary, not sufficient).
      absent  - no human_gate.json at finalize
      failed  - status 'failed', OR any 'wrong_object' anywhere, OR > HUMAN_GATE_MAX_BAD BAD counted cells, whatever
                the status says (the verdicts are evidence even in a review marked 'pending')
      passed  - status 'passed', not failed, >= 1 counted cell reviewed and the gate PLANNED >= SHEET_N_REF counted
                cells (review 2026-09-22; a per-stratum pool shortfall below the plan stays legal)
      pending - anything else (status 'pending', or a 'passed' review that is undersized)."""
    if not present: return "absent"
    if status not in HUMAN_STATUS: raise RuntimeError(f"human_gate.json status {status!r} not in {HUMAN_STATUS}")
    if status == "failed" or n_wrong_object > 0 or n_bad_counted > HUMAN_GATE_MAX_BAD: return "failed"
    if status == "passed" and n_counted > 0 and sheet_n_planned >= SHEET_N_REF: return "passed"
    return "pending"

def effective_hg_status(hg_verdict):
    """manifest.human_gate.status (vocabulary pending|passed|failed) = the RECOMPUTED outcome, so a reader that
    checks only `status` (Tol_43) sees a failed review as 'failed' even when the reviewer's file said 'pending' or 'passed'."""
    if hg_verdict not in HG_VERDICTS: raise RuntimeError(f"human-gate verdict {hg_verdict!r} not in {HG_VERDICTS}")
    return "pending" if hg_verdict == "absent" else hg_verdict

def seal_rule(hg_verdict, require_human_gate, below_floor, veto_ran, require_veto, smoke):
    """Seal rule (as amended 2026-09-23):
    sealed = (not require_human_gate or human gate passed) and a present gate has not FAILED
             and not below_floor and (veto_ran or not require_veto) and not smoke.
    require_human_gate=true is exactly the original rule (only 'passed' seals)."""
    if hg_verdict not in HG_VERDICTS: raise RuntimeError(f"human-gate verdict {hg_verdict!r} not in {HG_VERDICTS}")
    return bool((not require_human_gate or hg_verdict == "passed") and hg_verdict != "failed"
                and not below_floor and (veto_ran or not require_veto) and not smoke)

def gate_sheet_n_check(sheet_n, require_human_gate, smoke):
    """Stage gate (decision 2026-09-23): sheet_n < SHEET_N_REF raises ONLY when require_human_gate=true (outside
    SMOKE); with the default false it is allowed and a warning string is returned. None when nothing to say."""
    if sheet_n >= SHEET_N_REF or smoke: return None
    if require_human_gate:
        raise RuntimeError(f"sheet_n={sheet_n} < {SHEET_N_REF} with require_human_gate=true: the protocol requires ~300 reviewed "
                           "overlays before training (only SMOKE may use fewer)")
    return (f"⚠ sheet_n={sheet_n} < {SHEET_N_REF}: allowed because require_human_gate=false (decision 2026-09-23 — the fault list "
            "is the human gate); a later finalize with require_human_gate=true will refuse this template")

EVAL_V2_CONSTANTS = ("frag_lo[dayl]", "frag_lo[dark]", "over_hi")   # the Tol_44 GT trust filter reads exactly these (training-derived)

def eval_contract_check(man):
    """Raise unless the manifest carries everything the Tol_44 evaluation reads: forbidden_eval_days (a list of 8-digit days,
    == source_days, never a day from 04-23 on) and constants frag_lo[dayl], frag_lo[dark], over_hi as finite
    {"value": float} entries derived from training days only, with frag_lo <= 1 <= over_hi."""
    fb = man.get("forbidden_eval_days")
    if not isinstance(fb, list) or not fb or not all(isinstance(d, str) and re.fullmatch(r"\d{8}", d) for d in fb):
        raise RuntimeError(f"manifest.forbidden_eval_days must be a non-empty list of YYYYMMDD strings, got {fb!r}")
    if sorted(fb) != sorted(man.get("source_days", [])): raise RuntimeError(f"manifest.forbidden_eval_days {fb} != source_days {man.get('source_days')}")
    if any(d in DET_TEST_DAYS and d != "20240422" for d in fb): raise RuntimeError(f"manifest.forbidden_eval_days {fb} contains a detector test day")
    cs = man.get("constants", {})
    for k in EVAL_V2_CONSTANTS:
        c = cs.get(k)
        if not isinstance(c, dict) or not isinstance(c.get("value"), (int, float)) or not math.isfinite(float(c["value"])):
            raise RuntimeError(f"manifest.constants[{k!r}] missing or not a finite value: {c!r}")
        if not set(c.get("derived_from") or []) <= TRAIN_DAYS or not c.get("derived_from"):
            raise RuntimeError(f"manifest.constants[{k!r}].derived_from {c.get('derived_from')} must be a non-empty subset of the training days")
    if not (cs["frag_lo[dayl]"]["value"] < 1.0 and cs["frag_lo[dark]"]["value"] < 1.0 < cs["over_hi"]["value"]):
        raise RuntimeError("manifest constants violate frag_lo < 1 < over_hi")
    return True

def not_from_days(derived_from):
    """A manifest constant's not_from = every detector test day (04-22..04-28) it was NOT derived from. Per constant, so a
    pooled constant lists 04-22 even when 04-22 is a label source (U excludes non-training days), while
    A_ref[20240422] — the only constant a 04-22 source derives — omits it (review 2026-09-22; this is the
    rule Tol_43.constants_provenance_check re-checks: pooled not_from ⊇ all test days, per-day not_from ⊇ test days - {d})."""
    return sorted(DET_TEST_DAYS - {str(d) for d in (derived_from or [])})

def _same_widget(a, b):
    try: return float(a) == float(b)
    except (TypeError, ValueError): return str(a).strip() == str(b).strip()

def gate_widgets_resolve(adm_gw, w_now, defaults):
    """teacher/finalize: the gate-time widgets come from admitted.json. A widget of this run that differs from the
    gate's value raises unless it is at its default (i.e. was not passed); returns the gate's values (review 2026-09-22)."""
    if not isinstance(adm_gw, dict): raise RuntimeError("admitted.json lacks gate_widgets — it predates the widget-drift guard; re-run stage=gate")
    out, ignored = {}, []
    for k in GATE_KEYS:
        if k not in adm_gw: raise RuntimeError(f"admitted.json gate_widgets lacks {k!r} — re-run stage=gate")
        g, now = adm_gw[k], w_now[k]
        if not _same_widget(now, g):
            if not _same_widget(now, defaults[k]):
                raise RuntimeError(f"widget {k}={now!r} differs from the value stage gate ran with ({g!r}); gate-time widgets cannot change "
                                   "after gate — leave it at its default (the gate's value is used) or bump the tag and re-run gate")
            ignored.append(k)
        out[k] = g
    return out, ignored

def human_gate_eval(gate):
    """Fold a filled human_gate.json into counts + the rejected cells."""
    status = gate.get("status", "pending")
    bad_cells, n_bad_counted, n_wrong, by_kind = [], 0, 0, Counter()
    for c in gate.get("cells", []):
        v = c.get("verdict")
        if v is None or v in ("ok", "partial_minor"): continue
        if v not in GATE_VOCAB: raise RuntimeError(f"cell {c.get('cell_id')}: verdict {v!r} not in vocabulary {GATE_VOCAB}")
        by_kind[v] += 1
        if v == "wrong_object": n_wrong += 1
        if v in BAD_VERDICTS:
            bad_cells.append(c)
            if c.get("counted"): n_bad_counted += 1
    n_counted = sum(1 for c in gate.get("cells", []) if c.get("counted"))
    return {"status": status, "n_counted": n_counted, "n_bad_counted": n_bad_counted, "n_wrong_object": n_wrong,
            "n_bad_by_kind": dict(by_kind), "reject_rate": round(n_bad_counted / max(n_counted, 1), 4), "bad_cells": bad_cells}

def check_human_gate_vs_template(gate, template_cells):
    """The filled human_gate.json must carry exactly the template's cells (same cell_id, image_id, day, wall,
    clip_stem, frame_idx, counted, kind) — otherwise the seal could pass on cells nobody looked at, or on the
    sheets of a previous gate run (review 2026-09-22; the reviewer's status is necessary, not sufficient).
    'day' is checked because human_reject_set keys the ±90 s rejection on it: a missing or wrong day would make a
    BAD cell remove nothing (review 2026-09-23).
    Returns the number of counted template cells."""
    keys = ("image_id", "day", "wall", "clip_stem", "frame_idx", "counted", "kind")
    tpl = {c["cell_id"]: c for c in template_cells}
    got = {}
    for c in gate.get("cells", []):
        cid = c.get("cell_id")
        if cid in got: raise RuntimeError(f"human_gate.json: duplicate cell_id {cid!r}")
        got[cid] = c
    if set(got) != set(tpl):
        raise RuntimeError(f"human_gate.json cells != human_gate_template.json cells: {len(set(tpl) - set(got))} template cells missing "
                           f"(e.g. {sorted(set(tpl) - set(got))[:5]}), {len(set(got) - set(tpl))} unknown (e.g. {sorted(set(got) - set(tpl))[:5]}) — "
                           "the file was edited structurally or belongs to another gate run")
    for cid, t in tpl.items():
        g = got[cid]
        for k in keys:
            if g.get(k) != t.get(k): raise RuntimeError(f"human_gate.json cell {cid}: {k}={g.get(k)!r} != template {t.get(k)!r}")
    return sum(1 for t in tpl.values() if t.get("counted"))

def human_reject_set(bad_cells, frames):
    """Every admitted frame within ±HUMAN_MARGIN_S of a BAD cell's wall (same day) is excluded."""
    out = set()
    walls = [(c["day"], datetime.fromisoformat(c["wall"])) for c in bad_cells if c.get("wall")]
    for f in frames:
        w = datetime.fromisoformat(f["wall"]) if isinstance(f["wall"], str) else f["wall"]
        if any(f["day"] == d and abs((w - cw).total_seconds()) <= HUMAN_MARGIN_S for d, cw in walls): out.add(f["image_id"])
    return out

# ---- background plate — review material only; the current Tol_44 no longer copies build_plate / gray4 / plate_thr ----
def gray4(jpeg_bytes):
    """Grayscale 576x324 via PIL draft (DCT-domain 1/4 downscale)."""
    im = Image.open(io.BytesIO(jpeg_bytes)); im.draft("L", (PLATE_W, PLATE_H))
    im = im.convert("L")
    if im.size != (PLATE_W, PLATE_H): im = im.resize((PLATE_W, PLATE_H), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)

def build_plate(frames_gray):
    """Per-pixel temporal median (uint8) over a stack of 576x324 grayscale frames."""
    st = np.stack(frames_gray).astype(np.uint8)
    return np.median(st, axis=0).astype(np.uint8)

def plate_thr(res_values, k_mad):
    """median + k*MAD of residuals, floored at 12/255 (a noise-model rule, model-free)."""
    v = np.asarray(res_values, np.float64)
    med = float(np.median(v)); mad = float(np.median(np.abs(v - med)))
    return max(med + k_mad * mad, 12.0), med, mad          # 12.0 = 12/255 grey-level floor: plate rule (pre-declared, model-free noise floor); not derived from any day

def furniture_consistency(mask4, gray4_frame, plate, thr3):
    """Fraction of the (4x-downscaled) mask's pixels whose |frame - plate| residual is <= thr3."""
    if not mask4.any(): return 0.0
    res = np.abs(gray4_frame.astype(np.int16) - plate.astype(np.int16))
    return float((res[mask4] <= thr3).mean())

def position_prior_medoid(mask_sets, ids):
    """DET-0 input: among frames (each = 4 downsampled bool masks), the one maximising the mean
    over the sample of the best Hungarian-matched IoU. mask_sets: list of [4, h, w] bool arrays."""
    n = len(mask_sets)
    if n == 0: return None
    flat = np.stack([np.stack([m.reshape(-1) for m in ms]) for ms in mask_sets]).astype(np.float32)   # [n,4,P]
    F = flat.reshape(n * 4, -1); area = F.sum(1)
    inter_m = F @ F.T; union = area[:, None] + area[None, :] - inter_m
    iou = np.where(union > 0, inter_m / np.maximum(union, 1e-9), 0.0)
    score = np.zeros(n)
    for i in range(n):
        tot = 0.0
        for j in range(n):
            if i == j: continue
            sub = iou[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            ri, cj = linear_sum_assignment(-sub); tot += sub[ri, cj].mean()
        score[i] = tot / max(n - 1, 1)
    k = int(np.argmax(score))
    return {"image_id": int(ids[k]), "index": k, "mean_matched_iou": round(float(score[k]), 4)}

def manifest_guard_days(source_days):
    for d in source_days:
        if d in DET_TEST_DAYS and d != "20240422": raise RuntimeError(f"manifest source day {d} is a test day (F1)")
    return sorted(source_days)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster-only helpers — staging, tar access, metrics read-modify-write, contact-sheet drawing (Tol_17 conventions)

# COMMAND ----------

if not SELFTEST_ONLY:
    import cv2
    from pycocotools import mask as maskUtils
    TIMING = defaultdict(float); STAGED_MB = defaultdict(float)

    for d in (DS_DIR, FIG_DIR, LAB_DIR): os.makedirs(d, exist_ok=True)
    if osp.exists(LOCAL): shutil.rmtree(LOCAL)           # warm clusters keep stale files
    os.makedirs(LOCAL, exist_ok=True)

    def done_path(stage): return osp.join(DS_DIR, f".done_{stage}")
    def check_done(stage):
        if osp.exists(done_path(stage)) and not W["force"]:
            raise RuntimeError(f"stage {stage} already done for tag {TAG} ({done_path(stage)}); set force=true to re-run")
    def mark_done(stage):
        if SMOKE: print(f"⚠ SMOKE: not writing .done_{stage}")
        else: open(done_path(stage), "w").close()

    def load_metrics():
        return json.load(open(MET_PATH)) if osp.exists(MET_PATH) else {"tag": TAG, "notebook": "Tol_42_export_seg_coco", "smoke": SMOKE}
    def save_metrics(m):
        with open(MET_PATH, "w") as f: json.dump(m, f, indent=1, default=str)

    def stage_file(vol_path, local_path, label="stage"):
        """dbutils.fs.cp with 3 retries / 5 s sleeps (Tol_17:155-161); timed and reported in MB/s."""
        if osp.exists(local_path): os.remove(local_path)
        t0 = time.time(); ok = False; err = None
        for attempt in range(3):
            try:
                dbutils.fs.cp("dbfs:" + vol_path, "file:" + local_path); ok = True; break
            except Exception as e:
                err = e; print(f"  stage retry {attempt+1} for {osp.basename(vol_path)}: {e}"); time.sleep(5)
        if not ok: raise RuntimeError(f"could not stage {vol_path}: {err}")
        el = time.time() - t0; mb = osp.getsize(local_path) / 1e6
        TIMING[label] += el; STAGED_MB[label] += mb
        print(f"    staged {osp.basename(vol_path)} {mb:.0f} MB in {el:.0f} s ({mb/max(el,1e-9):.1f} MB/s)", flush=True)
        return local_path

    class Tars:
        """Staged Frames_5fps tars, one tarfile handle each; members '<frame_idx:07d>.jpg'."""
        def __init__(self): self.h = {}; self.local = {}
        def stage(self, day, stem):
            if stem in self.local: return self.local[stem]
            src = osp.join(FR_ROOT, day, f"{stem}.tar")
            if not osp.exists(src): raise RuntimeError(f"missing frame tar {src}")
            dst = osp.join(LOCAL, f"{stem}.tar"); stage_file(src, dst, "stage_tars"); self.local[stem] = dst; return dst
        def read(self, day, stem, fidx):
            if stem not in self.h: self.h[stem] = tarfile.open(self.stage(day, stem), "r")
            try: m = self.h[stem].extractfile(f"{fidx:07d}.jpg")          # TarFile.extractfile raises KeyError for an unknown name
            except KeyError: m = None
            if m is None: raise RuntimeError(f"{day}/{stem}: member {fidx:07d}.jpg missing from the frame tar (a frame the decoder skipped, Tol_00e:99?)")
            return m.read()
        def drop(self, stem):
            if stem in self.h: self.h[stem].close(); del self.h[stem]
            if stem in self.local and osp.exists(self.local[stem]): os.remove(self.local[stem])
            self.local.pop(stem, None)
        def close_all(self):
            for s in list(self.local): self.drop(s)

    def decode_bgr(jpeg_bytes, size=None):
        im = Image.open(io.BytesIO(jpeg_bytes))
        if size is not None: im.draft("RGB", size)
        arr = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def stretch_display(bgr):
        """Display-only global 1-99 percentile luminance stretch. Never applied to training JPEGs."""
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = np.percentile(g, 1), np.percentile(g, 99)
        if hi <= lo + 1: return bgr
        return np.clip((bgr.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

    CELL_W, CELL_H = 960, 540                          # sheet cell 960×540 (Tol_17 conventions); display-only, not derived from any day
    def render_cell(jpeg_bytes, rles, day, wall, stem, fidx, strip, ph, cell_id, crop=None, border=None):
        """One 960x540 panel in Tol_17 conventions: INTER_NEAREST masks, 0.55/0.45 fill, 2-px contour,
        timestamp bar; night/dusk/dawn get the display-only stretch, stated in the panel."""
        full = decode_bgr(jpeg_bytes, None if crop else (1152, 648))   # (1152, 648) = PIL draft hint, half of 2304×1296 (design choice, display-only; ≥ the 960×540 cell)
        if crop:
            x, y, cw, ch = crop; full = full[y:y + ch, x:x + cw]
        img = cv2.resize(full, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
        stretched = ph in ("night", "dusk", "dawn")
        if stretched: img = stretch_display(img)
        for i, rl in enumerate(rles):
            if not rl: continue
            m = rle_decode(rl) if isinstance(rl, dict) else rl
            if crop:
                x, y, cw, ch = crop; m = m[y:y + ch, x:x + cw]
            ms = cv2.resize(m.astype(np.uint8), (CELL_W, CELL_H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not ms.any(): continue
            col = COLORS[i % 4]
            img[ms] = (0.55 * img[ms] + 0.45 * np.array(col, np.float32)).astype(np.uint8)
            cnts, _ = cv2.findContours(ms.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, col, 2)
        cv2.rectangle(img, (0, 0), (CELL_W, 22), (0, 0, 0), -1)
        cv2.putText(img, f"{day}  {wall:%H:%M:%S}   {stem} f{fidx:07d}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, CELL_H - 22), (CELL_W, CELL_H), (0, 0, 0), -1)
        cv2.putText(img, f"{cell_id}  {strip}  {ph}" + ("  [display stretch 1-99%]" if stretched else ""), (6, CELL_H - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if border is not None: cv2.rectangle(img, (0, 0), (CELL_W - 1, CELL_H - 1), border, 6)
        return img

    def write_sheet(cells, name):
        """cells: list of up to 12 panels -> 3x4 PNG under FIG_DIR (built locally, then copied)."""
        sheet = np.zeros((3 * CELL_H, 4 * CELL_W, 3), np.uint8)
        for k, im in enumerate(cells[:12]):
            r, c = divmod(k, 4); sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = im
        loc = osp.join(LOCAL, name); cv2.imwrite(loc, sheet); shutil.copy(loc, osp.join(FIG_DIR, name)); os.remove(loc)

    def read_json(p):
        with open(p) as f: return json.load(f)

    def read_masks_gz_sha(p):
        """One FUSE read: the raw gzip bytes are hashed, then decompressed (review 2026-09-23 — admitted.json records the hash)."""
        with open(p, "rb") as f: raw = f.read()
        return json.loads(gzip.decompress(raw)), hashlib.sha256(raw).hexdigest()

    def read_masks_checked(p, expect_sha):
        """teacher/finalize: re-read a masks file only if it is byte-identical to the one gate admitted frames from."""
        if not expect_sha: raise RuntimeError(f"admitted.json has no masks_sha256 for {p} — it predates the tracking-root guard; bump the tag and re-run gate")
        d, got = read_masks_gz_sha(p)
        if got != expect_sha:
            raise RuntimeError(f"{p}: sha256 {got[:12]} != {expect_sha[:12]} recorded at gate — the tracking root changed since gate "
                               "(its masks never went through G3-G10 / the fault list / G11); bump the tag and re-run gate")
        return d

    def assert_window_files(adm):
        """teacher/finalize: segments.json / events.jsonl of every gated window are unchanged since gate."""
        wf = adm.get("window_files")
        if not isinstance(wf, dict) or not wf: raise RuntimeError("admitted.json lacks window_files — it predates the tracking-root guard; bump the tag and re-run gate")
        for key, rec in wf.items():
            root, win = key.split("/", 1)
            for name in ("segments.json", "events.jsonl"):
                p = osp.join(TOL, root, win, name); exp = rec.get(name)
                got = sha256_file(p) if osp.exists(p) else None
                if got != exp: raise RuntimeError(f"{p}: sha256 {str(got)[:12]} != {str(exp)[:12]} recorded at gate — the tracking root changed since gate; bump the tag and re-run gate")

    # ---- teacher/finalize: gate-time widgets resolved from admitted.json BEFORE the guards below read them ----------
    # (review 2026-09-23: allow_test_day_reason / unlisted_day_policy / band_json are read by guard_source_days,
    # check_faults_file and the band load, so a payload with only the required widgets must already see the gate's values)
    if STAGE in ("teacher", "finalize"):
        _ap = osp.join(DS_DIR, "admitted.json")
        if not osp.exists(_ap): raise RuntimeError(f"stage {STAGE}: {_ap} missing — run stage=gate for tag {TAG} first (or the last gate run died)")
        if not SMOKE and not osp.exists(done_path("gate")):
            raise RuntimeError(f"stage {STAGE}: {done_path('gate')} missing — the last gate run of tag {TAG} did not complete; re-run stage=gate")
        _gw, _ign = gate_widgets_resolve(read_json(_ap).get("gate_widgets"), W, W_DEFAULTS)
        if _ign: print(f"gate-time widgets taken from admitted.json (this run left them at their defaults): {_ign}")
        W.update(_gw); SEED = W["seed"]

    # ---- sources, fault file, band file: parsed and asserted once for every stage -------------------
    PAIRS = parse_sources(W["sources"])
    SOURCE_DAYS = guard_source_days(PAIRS, W["allow_test_day_reason"])
    ROOTS = sorted({r for r, _ in PAIRS})
    print(f"sources: {PAIRS}\nsource days: {SOURCE_DAYS}  (refuses 20240423+; 20240422 reason={W['allow_test_day_reason']!r})")
    if not osp.isfile(W["faults_json"]):   # the fault list is authored locally; it must be uploaded to the Volume first
        raise RuntimeError(f"faults_json {W['faults_json']!r} is not a file on this cluster — upload train_label_faults.json to the Volume "
                           "(Tol/Detector_Labels/user/train_label_faults_2026-09-22.json) and pass that /Volumes/... path")
    FAULTS = read_json(W["faults_json"]); FAULTS_SHA = sha256_file(W["faults_json"])
    BAND_PATH = osp.join(TOL, W["band_json"]); BAND = read_json(BAND_PATH); BAND_SHA = sha256_file(BAND_PATH)
    if not set(BAND.get("derived_from", [])) <= TRAIN_DAYS: raise RuntimeError(f"band_json derived_from {BAND.get('derived_from')} is not ⊆ TRAIN_DAYS")
    OVER_HI = float(BAND["band"]["over_hi"]); TOL41_FRAG_LO = float(BAND["band"]["frag_lo"])
    print(f"fault JSON sha256 {FAULTS_SHA}\nband JSON  sha256 {BAND_SHA}  over_hi={OVER_HI} (READ)  Tol_41 frag_lo={TOL41_FRAG_LO} (relaxed-tail reference only)")
    UNLISTED = []
    for root in ROOTS:
        UNLISTED += check_faults_file(FAULTS, root, [d for r, d in PAIRS if r == root], W["unlisted_day_policy"])
    if UNLISTED: print(f"⚠ days {UNLISTED} are not in the fault file; admitted on automatic gates only (unlisted_day_policy=gates_only)")
    # the second review is waived (decision 2026-09-23) BECAUSE both training-day videos were reviewed end to end;
    # a day with no fault list (e.g. Tracking_teacher 04-22: identity certified, mask extent not)
    # has had no geometry review at all, so it needs the human gate (review 2026-09-23). SMOKE never seals.
    if UNLISTED and STAGE in ("gate", "finalize") and not W["require_human_gate"] and not SMOKE:
        raise RuntimeError(f"source days {UNLISTED} have no fault list, so the fault-list-as-human-gate waiver does not cover them: set "
                           "require_human_gate=true (gate AND finalize) so their masks get a reviewed sheet sample, or list their faults first")
    # provenance copies: the untagged Detector_Labels copy is overwritten by every tag, so a sha-named copy and a
    # per-tag copy are written too (the manifest names the per-tag one) — review 2026-09-22. They are WRITTEN only by
    # stage gate, after check_done and the sealed-tag refusal, and a per-tag copy with other content is never
    # overwritten (raise); teacher/finalize only verify the per-tag copy against admitted.faults_sha256.
    FAULTS_COPIES = {"untagged": osp.join(LAB_DIR, "train_label_faults.json"), "sha_named": osp.join(LAB_DIR, f"train_label_faults_{FAULTS_SHA[:8]}.json"),
                     "tagged": osp.join(DS_DIR, "train_label_faults.json")}
    def check_tagged_fault_copy(expect_sha):
        """Raise if the per-tag provenance copy exists with a sha other than expect_sha (it belongs to another gating)."""
        p = FAULTS_COPIES["tagged"]
        if osp.exists(p):
            got = sha256_file(p)
            if got != expect_sha: raise RuntimeError(f"{p} has sha256 {got[:12]} != {expect_sha[:12]}: tag {TAG} was gated with another fault file — bump the tag")
            return True
        return False
    def write_fault_copies():
        if not SMOKE: check_tagged_fault_copy(FAULTS_SHA)       # a SMOKE tag never seals, so its per-tag copy may follow the file being iterated
        for p in FAULTS_COPIES.values():
            if not (osp.exists(p) and sha256_file(p) == FAULTS_SHA): shutil.copy(W["faults_json"], p)

    def assert_admitted_matches(adm):
        """teacher/finalize: the widget-time sources and the fault/band files read NOW must be the ones gate ran with;
        the manifest is then derived from admitted.json, never from the widgets (review 2026-09-22)."""
        if adm.get("sources_str") != W["sources"] or [list(p) for p in adm["sources"]] != [list(p) for p in PAIRS] or list(adm["source_days"]) != list(SOURCE_DAYS):
            raise RuntimeError(f"admitted.json was gated with sources {adm.get('sources_str')!r} / days {adm.get('source_days')} but this stage got "
                               f"{W['sources']!r} / {SOURCE_DAYS} — run every stage of tag {TAG} with the gate's sources")
        if adm.get("faults_sha256") != FAULTS_SHA:
            raise RuntimeError(f"fault file changed since gate: gate sha256 {adm.get('faults_sha256')} != now {FAULTS_SHA} — bump the tag and re-run gate")
        if adm.get("band_sha256") != BAND_SHA:
            raise RuntimeError(f"band file changed since gate: gate sha256 {adm.get('band_sha256')} != now {BAND_SHA} — bump the tag and re-run gate")
        if not check_tagged_fault_copy(adm["faults_sha256"]): raise RuntimeError(f"{FAULTS_COPIES['tagged']} missing — stage gate writes it; re-run gate")
        if not adm.get("gate_run_id"): raise RuntimeError("admitted.json lacks gate_run_id — it predates the gate-run stamp; bump the tag and re-run gate")
        assert_window_files(adm)                   # the masks files themselves are re-hashed where they are re-read (review 2026-09-23)
        # gate-time widgets: this stage runs with the GATE's values (strides, quotas, caps, overrides, seed, ...), so the
        # manifest and the floor check describe the dataset that was actually built (review 2026-09-22)
        global SEED, QUOTAS, VALID_STEMS
        gw, ignored = gate_widgets_resolve(adm.get("gate_widgets"), W, W_DEFAULTS)
        if ignored: print(f"gate-time widgets taken from admitted.json (this run left them at their defaults): {ignored}")
        W.update(gw); SEED = W["seed"]
        QUOTAS = {"night": W["night_quota"], "dusk": W["dusk_quota"], "dawn": W["dawn_quota"], "active": W["active_quota"]}
        VALID_STEMS = [s.strip() for s in W["valid_clips"].split(",") if s.strip()]

    def assert_fresh(obj, name, adm_sha):
        """A teacher-stage output is keyed by image_id; it is only valid for the admitted.json it was computed on."""
        if obj.get("admitted_sha256") != adm_sha:
            raise RuntimeError(f"{name} is STALE: computed on admitted.json {str(obj.get('admitted_sha256'))[:12]} but the current admitted.json hashes to "
                               f"{adm_sha[:12]} (gate was re-run after teacher) — re-run stage=teacher")

    # window dirs per (root, day); every one must be listed in the fault file
    WINDOWS = []          # (root, day, window)
    for root, day in PAIRS:
        base = osp.join(TOL, root)
        ws = sorted(d for d in os.listdir(base) if re.match(rf"^{day}(_w\d{{4}}-\d{{4}})?$", d))
        if not ws: raise RuntimeError(f"no window dir for {day} under {base}")
        if day in FAULTS.get("windows", {}):
            listed = set(FAULTS["windows"][day])
            if set(ws) != listed: raise RuntimeError(f"{root}/{day}: window dirs {ws} != fault file windows {sorted(listed)}")
        WINDOWS += [(root, day, w) for w in ws]
    print(f"{len(WINDOWS)} window dirs: {[w for _, _, w in WINDOWS]}")

    # existing tag with different sources => raise (versions are immutable); admitted.json is checked too,
    # since it exists before any manifest does
    for _name in ("manifest.json", "admitted.json"):
        _mp = osp.join(DS_DIR, _name)
        if osp.exists(_mp):
            _old = read_json(_mp)
            if _old.get("sources_str") != W["sources"]:
                raise RuntimeError(f"tag {TAG} exists ({_name}) with sources {_old.get('sources_str')!r} != {W['sources']!r}; bump the tag")

    CW, CH = [int(v) for v in W["crop_wh"].lower().split("x")]
    if abs(CW / CH - 16 / 9) > 1e-3: raise RuntimeError(f"crop_wh {W['crop_wh']} is not 16:9")
    VALID_STEMS = [s.strip() for s in W["valid_clips"].split(",") if s.strip()]
    QUOTAS = {"night": W["night_quota"], "dusk": W["dusk_quota"], "dawn": W["dawn_quota"], "active": W["active_quota"]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage `gate` (CPU) — pass A gate table, fault/event buffers, G3–G10, derived constants, sampling, split, crops, sheets, plates

# COMMAND ----------

if not SELFTEST_ONLY and STAGE == "gate":
    check_done("gate")
    # a SEALED tag is immutable: Tol_43/Tol_44 hash its manifest, and a gate re-run would rewrite admitted.json, the plates,
    # position_prior.json and the sheets under it — force does not override this; bump the tag (review 2026-09-22)
    _mp = osp.join(DS_DIR, "manifest.json")
    if osp.exists(_mp) and read_json(_mp).get("sealed"):
        raise RuntimeError(f"tag {TAG} has a SEALED manifest.json — a sealed tag is immutable (force=true does not override); bump the tag")
    _msg = gate_sheet_n_check(W["sheet_n"], W["require_human_gate"], SMOKE)
    if _msg: print(_msg)
    if not SMOKE: check_tagged_fault_copy(FAULTS_SHA)   # early: a per-tag fault copy from another gating means this must be a new tag
    # the previous gate's own outputs go FIRST: a gate that dies after rewriting the sheets / template / prior must not
    # leave the old admitted.json + .done_gate for teacher/finalize to pair with them (review 2026-09-23)
    for _n in GATE_OWN_OUTPUTS:
        _p = osp.join(DS_DIR, _n)
        if osp.exists(_p): os.remove(_p); print(f"⚠ gate re-run: removed the previous gate's {_n}")
    # figures of a previous run (rejected_G<k>, crops, probes, S<nnn> beyond the new count, vetoed, _verify) would sit
    # beside this run's sheets and contradict them (review 2026-09-23)
    if osp.isdir(FIG_DIR): shutil.rmtree(FIG_DIR); print(f"⚠ gate re-run: cleared {FIG_DIR}")
    os.makedirs(FIG_DIR, exist_ok=True)
    GATE_RUN_ID = f"{datetime.now():%Y%m%dT%H%M%S}_{hashlib.sha256(os.urandom(16)).hexdigest()[:8]}"   # stamped into admitted.json, the template and position_prior.json
    print(f"gate_run_id {GATE_RUN_ID}")
    # a gate re-run re-assigns image ids, so every downstream output keyed by image_id is stale: remove it loudly
    # (review 2026-09-22). human_gate.json is the reviewer's file and is left in place; finalize rejects it against the new template.
    for _n in DOWNSTREAM_OF_GATE:
        _p = osp.join(DS_DIR, _n)
        if osp.exists(_p): os.remove(_p); print(f"⚠ gate re-run: removed stale {_n} (keyed by image_id; re-run stage=teacher / finalize)")
    for _n in DOWNSTREAM_DIRS_OF_GATE:
        _p = osp.join(DS_DIR, _n)
        if osp.isdir(_p): shutil.rmtree(_p); print(f"⚠ gate re-run: removed stale {_n}/ (written by finalize from the previous admitted set)")
    if osp.exists(osp.join(DS_DIR, "human_gate.json")): print("⚠ gate re-run: an existing human_gate.json belongs to the previous sheets and will be rejected by finalize")
    t_gate0 = time.time()
    MET = load_metrics()
    for _k in MET_KEYS_OF_LATER_STAGES:            # their outputs were just deleted; a stale key would contradict them (review 2026-09-22)
        if MET.pop(_k, None) is not None: print(f"⚠ gate re-run: dropped stale metrics key {_k!r}")
    # a previous gate's curl_band_check abort marker must not survive a gate that completes (review 2026-09-23)
    if MET.pop("gate_aborted", None) is not None: print("⚠ gate re-run: dropped the previous gate's 'gate_aborted' marker")
    MET.update({"tag": TAG, "smoke": SMOKE, "seed": SEED, "widgets": W, "sources": PAIRS, "source_days": SOURCE_DAYS,
                "faults_sha256": FAULTS_SHA, "band_sha256": BAND_SHA})
    # ---------------- pass A: measurements from RLE arithmetic (no decode) ------------------------
    ROWS, CLIPS, SEGS_BY_WIN, EVENTS_BY_DAY = [], [], {}, defaultdict(list)
    WINDOW_FILES = {}     # "<root>/<window>" -> {segments.json: sha256, events.jsonl: sha256|None} (review 2026-09-23)
    t0 = time.time()
    for root, day, win in WINDOWS:
        wdir = osp.join(TOL, root, win)
        segp = osp.join(wdir, "segments.json")
        if not osp.exists(segp): raise RuntimeError(f"{wdir}: segments.json missing")
        SEGS_BY_WIN[win] = SegLookup(read_json(segp)["segments"])
        evp = osp.join(wdir, "events.jsonl")
        WINDOW_FILES[f"{root}/{win}"] = {"segments.json": sha256_file(segp), "events.jsonl": sha256_file(evp) if osp.exists(evp) else None}
        if osp.exists(evp):
            with open(evp, encoding="utf-8", errors="ignore") as f:
                EVENTS_BY_DAY[day] += [e for e in (parse_event_line(l) for l in f) if e]
        for gz in sorted(glob.glob(osp.join(wdir, "masks_*.json.gz"))):
            d, gz_sha = read_masks_gz_sha(gz)
            if abs(float(d.get("track_fps", 0)) - TRACK_FPS_EXPECTED) > 1e-6: raise RuntimeError(f"{gz}: track_fps {d.get('track_fps')} != 5.0")
            if d.get("backbone") != "teacher": raise RuntimeError(f"{gz}: backbone {d.get('backbone')!r} != 'teacher'")
            if d.get("smoke"): raise RuntimeError(f"{gz}: a SMOKE masks file cannot be a label source")
            stem = clip_stem_of(d["clip"])
            ip = osp.join(FR_ROOT, day, f"{stem}_index.json")
            if not osp.exists(ip): raise RuntimeError(f"{stem}: frame index {ip} missing")
            idx = read_json(ip)
            s_masks = {int(r["frame_idx"]) for r in d["frames"]}; s_idx = {int(r["frame_idx"]) for r in idx["frames"]}
            if s_masks != s_idx: raise RuntimeError(f"{stem}: masks frame_idx set != index set ({len(s_masks)} vs {len(s_idx)}) — different decodes")
            rows = measure_clip(d["frames"], d["clip"], win, SEGS_BY_WIN[win])
            # keep int32 counts for rows with all four masks (G10 / crops / sheets need them; ~340 MB for 23 clips)
            fr = d["frames"]
            for r in rows:
                if r["n_present"] == N_CALVES:
                    r["rle_np"] = [np.asarray(fr[r["row_i"]]["masks"][str(s)]["counts"], np.int32) for s in range(N_CALVES)]
            ROWS += rows
            CLIPS.append({"root": root, "day": day, "window": win, "clip": d["clip"], "clip_stem": stem, "n_rows_5fps": len(fr), "n_index": len(s_idx),
                          "n_rows_1hz": len(rows), "wall_start": rows[0]["wall"], "wall_end": rows[-1]["wall"],
                          "masks_file": osp.basename(gz), "masks_sha256": gz_sha})   # teacher/finalize re-read ONLY this exact file (review 2026-09-23)
            del d, fr
            print(f"  {win}/{stem}: {len(rows)} 1-Hz rows  ({(time.time()-t0)/60:.1f} min)", flush=True)
    TIMING["gating_pass_a_s"] = time.time() - t0
    print(f"pass A: {len(ROWS)} 1-Hz rows from {len(CLIPS)} clips in {TIMING['gating_pass_a_s']/60:.1f} min")
    if SMOKE: print("⚠ SMOKE: pass A still covers every clip (the A_ref cross-check needs whole days, ≈ 25 min); caps apply from the lattice on, and only the tars those frames need are staged")

    with gzip.open(osp.join(DS_DIR, "gate_table.json.gz"), "wt") as f:
        json.dump([{k: (v.isoformat() if k == "wall" else v) for k, v in r.items() if k != "rle_np"} for r in ROWS], f)

    # ---------------- A_ref per day (DERIVED) + cross-check against the band JSON ------------------
    A_REF = {d: a_ref_of(ROWS, d) for d in SOURCE_DAYS}
    BAND_XCHK = {}
    for d, v in A_REF.items():
        ref = BAND.get("per_day", {}).get(d, {}).get("ref_pass1_largest_daylight")
        if ref is None: BAND_XCHK[d] = "n/a"; print(f"  A_ref[{d}] = {v:,.0f} px (no band_json entry — contributes no other constant)")
        else:
            dev = abs(v / ref - 1)
            if dev >= 0.01: raise RuntimeError(f"A_ref[{d}]={v:.0f} deviates {dev:.3%} from band_json {ref} — the root changed since the band was derived")
            BAND_XCHK[d] = round(dev, 5); print(f"  A_ref[{d}] = {v:,.0f} px  (band_json {ref}, dev {dev:.4%} ✓)")

    # ---------------- G1 fault intervals, G2 event buffers ------------------------------------------
    FAULT_IV = {d: fault_intervals(FAULTS, d) for d in SOURCE_DAYS}
    # notebook-side exclusions for the notes' "unlisted but visible" over-extended masks (review 2026-09-23): G1 treats them
    # like faults (LOST_BY_FAULT counts them as N1/N2; a G9 block containing one counts as having a fault inside)
    NOTE_IV, NOTE_REP = note_exclusions(FAULTS, NOTE_EXCL_WALLS, NOTE_EXCL_MAX_S)
    NOTE_REP = [n for n in NOTE_REP if n["day"] in SOURCE_DAYS]
    for d, ivs in NOTE_IV.items():
        if d in FAULT_IV: FAULT_IV[d] = sorted(FAULT_IV[d] + ivs)
    for n in NOTE_REP: print(f"  note exclusion {n['id']} {n['wall']}: " + (f"already inside {n['already_excluded_by']}" if n.get("already_excluded_by") else f"{n['exclude_start'][11:]} -> {n['exclude_end'][11:]} ({n['span_s']:.0f} s{', capped' if n['capped'] else ''})"))
    BAD_BY_DAY = {d: known_partial(FAULTS, d) for d in SOURCE_DAYS}
    EV_IV, EV_STATS = {}, {}
    for d in SOURCE_DAYS:
        iv, st = event_intervals(EVENTS_BY_DAY.get(d, []), A_REF[d]); EV_IV[d] = merge_intervals(iv); EV_STATS[d] = st
        st["union_s_by_phase4"] = {}
        for ph in ("night", "dawn", "dusk", "day"):
            st["union_s_by_phase4"][ph] = sum((b - a).total_seconds() for a, b in EV_IV[d] if phase4(a) == ph)
        print(f"  {d}: {st['n_events']} events -> {len(EV_IV[d])} merged buffer intervals, {union_seconds([(a,b,'') for a,b in EV_IV[d]])/60:.1f} min excluded; {st['by_label']}; chaos windows {st['chaos_windows']}; "
              f"{st['n_no_wall']} non-buffer events without a wall clock (ignored)")
        if st["by_label"].get("close_detector_contradiction") or st["by_label"].get("close_displaced_by_detection"):
            print(f"  ⚠ FINDING: contradiction/displacement closes fired on {d} (telemetry said 0 strikes)")

    # ---------------- G1-G5 funnel -> U ----------------------------------------------------------------
    FUNNEL = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))   # day -> phase4 -> gate -> [in, out]
    LOST_BY_FAULT = Counter(); FIRST_FAIL = {}
    U = []
    for r in ROWS:
        fid = fault_hit(r["wall"], FAULT_IV[r["day"]])
        if fid: LOST_BY_FAULT[fid] += 1
        g = gate_pre(r, FAULT_IV[r["day"]], EV_IV[r["day"]])
        f = FUNNEL[r["day"]][r["phase4"]]
        for name in ("G1", "G2", "G3", "G4", "G5"):
            f[name][0] += 1
            if g == name: f[name][1] += 1; break
        FIRST_FAIL[id(r)] = g
        if g is None: U.append(r)
    print(f"U (G0-G5 survivors) = {len(U)} rows; lost per fault id: {dict(LOST_BY_FAULT)}")
    # G5's edge is supported on the TRAINING days, not by Tol_12's test-day relink (review 2026-09-23)
    LINK_REP = link_iou_report([r for r in ROWS if FIRST_FAIL.get(id(r)) in (None, "G5")])
    print(f"G5 link IoU on the training days (G1-G4 survivors): {LINK_REP['by_pool']}; {LINK_REP['n_none']} links without both neighbours")

    # ---------------- derived constants from U ------------------------------------------------
    OVR = None
    if W["frag_lo_override"]:
        OVR = {k.strip(): float(v) for k, v in (kv.split("=") for kv in W["frag_lo_override"].split(","))}
        if set(OVR) != {"dayl", "dark"}: raise RuntimeError(f"frag_lo_override must name dayl and dark, got {OVR}")
    _c0, _c1 = datetime.fromisoformat(CURL_START), datetime.fromisoformat(CURL_END); CURL_DAY = CURL_START[:10].replace("-", "")
    CURL_CTX = dict(Counter((FIRST_FAIL.get(id(r)) or "pass") for r in ROWS if _c0 <= r["wall"] < _c1))   # which gate removed the certified stretch, if any
    C = derive_constants(U, A_REF, OVER_HI, CURL_START, CURL_END, override=OVR, curl_required=CURL_DAY in SOURCE_DAYS, curl_context=CURL_CTX)
    C["curled_check"]["first_fail_in_stretch"] = CURL_CTX
    C["link_iou"] = LINK_REP
    C["A_ref"] = {d: int(round(v)) for d, v in A_REF.items()}
    C["bad_overlap"] = bad_overlap(ROWS, BAD_BY_DAY, A_REF, C)
    print(f"frag_lo {C['frag_lo']} ({C['frag_lo_provenance']}), per-day P1 {C['frag_lo_per_day']}; over_hi {OVER_HI} (READ)")
    print(f"S band {C['S_lo']}..{C['S_hi']}; STAB [{C['STAB_LO']}, {C['STAB_HI']}]; ACTIVE_THR {C['ACTIVE_THR']}")
    print(f"curled check {C['curled_check']}; upper diag {C['upper_diag']}; BAD overlap {C['bad_overlap']}")
    if C["upper_diag"]["upper_edge_clips_admitted_mass"]: print("⚠ upper edge clips admitted mass (P99.5 > over_hi) — never auto-relaxed; see rejected_G6 sheet")
    if C["bad_overlap"]["high"]: print("⚠ bad_overlap_high: the band would admit >10 % of the partial masks listed in the fault file; tail sheets enlarged")

    # ---------------- G9 blocks, then G6-G9 funnel ---------------------------------------------------
    BLOCK_FAULT = {}
    for r in ROWS:
        if r["phase4"] in ("dusk", "night") and fault_hit(r["wall"], FAULT_IV[r["day"]]): BLOCK_FAULT[(r["clip_stem"], r["phase4"])] = True
    OVR_BLOCKS = [b.strip() for b in W["collapse_override_blocks"].split(",") if b.strip()]
    ROWS_G12 = [r for r in ROWS if FIRST_FAIL.get(id(r)) not in ("G1", "G2")]      # block statistics over admissible rows only (review 2026-09-22)
    DROPPED_BLOCKS, BLOCK_REP = block_collapse(ROWS_G12, A_REF, C["frag_lo"]["dark"], OVR_BLOCKS, BLOCK_FAULT, rows_all=ROWS)
    print(f"G9: {BLOCK_REP['n_blocks']} dusk/night blocks, dropped {BLOCK_REP['dropped']}, curled/clipped {len(BLOCK_REP['curled_or_clipped_blocks'])}, "
          f"blocks failing a per-slot 0.75 rule {BLOCK_REP['blocks_failing_per_slot_0.75']}")
    for b in BLOCK_REP["needs_human_verdict"]: print(f"  ⚠ block {b} dropped with NO listed fault inside — needs a human verdict (collapse_override_blocks re-admits)")
    ADM = []
    for r in U:
        g = gate_post(r, A_REF[r["day"]], C, DROPPED_BLOCKS)
        f = FUNNEL[r["day"]][r["phase4"]]
        for name in ("G6", "G7", "G8", "G9"):
            f[name][0] += 1
            if g == name: f[name][1] += 1; break
        FIRST_FAIL[id(r)] = g
        if g is None: ADM.append(r)
    print(f"admitted after G9: {len(ADM)} rows")
    # curled-stretch admission after the WHOLE band (G6-G9), RELATIVE to the rest of the dark pool (review 2026-09-23, curl_band_check):
    # the absolute 0.90 stays G6-only (derive_constants). A failure is raised only at the END of stage gate, after the
    # curl_rejected / rejected_G<k> sheets are written; escapes: collapse_override_blocks (G9) or curl_band_ack (recorded)
    CURL_U = [r for r in U if _c0 <= r["wall"] < _c1]
    CURL_REF = [r for r in U if r["day"] in TRAIN_DAYS and pool_of(r["phase4"]) == "dark" and not (_c0 <= r["wall"] < _c1)]
    if CURL_DAY in SOURCE_DAYS and CURL_U:
        C["curled_check"]["after_G6_G9"] = curl_band_check([FIRST_FAIL.get(id(r)) for r in CURL_U], [FIRST_FAIL.get(id(r)) for r in CURL_REF], W["curl_band_ack"])
    else:
        if W["curl_band_ack"]: raise RuntimeError("curl_band_ack given but the curled-stretch check did not run (its day is not a source day) — remove the ack")
        C["curled_check"]["after_G6_G9"] = {**curl_admission([]), "fails": False, "blocking": False, "ack": None, "note": "curled stretch not in the sources"}
    _cb = C["curled_check"]["after_G6_G9"]
    print(f"curled stretch after G6-G9: {_cb}")
    if _cb.get("below_abs_0.90"): print(f"  (reported only) stretch admission {_cb['admitted_frac']} < {CURL_ADMIT_MIN} absolute; the dark-pool reference keeps {_cb['ref_dark_outside_stretch']['admitted_frac']}")
    if _cb["fails"]:
        print(f"⚠ the curled stretch is rejected DISPROPORTIONATELY by G6-G9 (relative {_cb['relative']} < {CURL_REL_MIN}); "
              + (f"ACKNOWLEDGED via curl_band_ack={W['curl_band_ack']!r} (recorded)" if _cb["ack"] else "stage gate will RAISE after writing the curl_rejected sheet"))

    # ---------------- split, lattice, G12, G10 (decode), quotas, hour cap, floor -------------------
    ALL_STEMS = [c["clip_stem"] for c in CLIPS]
    TRAIN_STEMS, VALID_STEMS = split_clips(ALL_STEMS, VALID_STEMS)
    VALID_BOUNDS = [(c["day"], c["wall_start"], c["wall_end"]) for c in CLIPS if c["clip_stem"] in VALID_STEMS]
    by_clip = defaultdict(list)
    for r in ADM: by_clip[r["clip_stem"]].append(r)
    train_c, valid_c = [], []
    for stem, rs in by_clip.items():
        if stem in VALID_STEMS: valid_c += lattice(rs, W["valid_stride_day_s"], W["valid_stride_other_s"])
        else: train_c += lattice(rs, W["stride_day_s"], W["stride_other_s"])
    # the lattice is per clip; overlapping clip files (0.5-3.4 s, 04-18/04-19 index files) could export one instant twice (review 2026-09-23)
    train_c, n_xclip_train = cross_clip_dedupe(train_c, min(W["stride_day_s"], W["stride_other_s"]))
    valid_c, n_xclip_valid = cross_clip_dedupe(valid_c, min(W["valid_stride_day_s"], W["valid_stride_other_s"]))
    N_XCLIP = {"train": n_xclip_train, "valid": n_xclip_valid}
    print(f"cross-clip overlap: dropped {N_XCLIP} near-duplicate lattice picks (< min stride apart, different clip files)")
    train_c, n_g12 = valid_buffer_drop(train_c, VALID_BOUNDS, W["valid_buffer_min"])
    print(f"lattice: train candidates {len(train_c)}, valid candidates {len(valid_c)}; G12 dropped {n_g12} train rows near valid clips")
    if SMOKE:
        rng0 = random.Random(SEED); rng0.shuffle(train_c); rng0.shuffle(valid_c)
        train_c, valid_c = train_c[:W["max_frames"]], valid_c[:max(4, W["max_frames"] // 4)]
        print(f"⚠ SMOKE cap: train {len(train_c)}, valid {len(valid_c)}")
    rngv = random.Random(SEED + 2)
    if len(valid_c) > W["valid_max"]:
        print(f"⚠ valid capped {len(valid_c)} -> {W['valid_max']} (seeded uniform)"); valid_c = rngv.sample(valid_c, W["valid_max"])
    # G10 on lattice candidates only (decode). body_len_px: EVERY decoded lattice mask of a TRAINING day (both
    # splits, all phases; no cap — one np.cov per mask is negligible next to the decode), per-phase4 medians reported
    # (a 04-22 source contributes no constant) — review 2026-09-22
    t0 = time.time(); BODY = defaultdict(list); DEC_CACHE = {}
    def decoded(r):
        k = (r["clip_stem"], r["frame_idx"])
        if k not in DEC_CACHE:
            if len(DEC_CACHE) > 64: DEC_CACHE.clear()
            DEC_CACHE[k] = [rle_decode({"size": [H_FRAME, W_FRAME], "counts": c}) for c in r["rle_np"]]
        return DEC_CACHE[k]
    def g10_pass(r):
        ms = decoded(r); r["cc_frac"] = [round(largest_cc_frac(m), 4) for m in ms]
        if r["day"] in TRAIN_DAYS: BODY[r["phase4"]].extend(v for v in (body_len_of(m) for m in ms) if v)
        return all(v >= CC_FRAC for v in r["cc_frac"])
    def apply_g10(cands):
        keep = []
        for r in cands:
            f = FUNNEL[r["day"]][r["phase4"]]; f["G10"][0] += 1
            if g10_pass(r): keep.append(r)
            else: f["G10"][1] += 1; FIRST_FAIL[id(r)] = "G10"
        return keep
    _curl_cand = [r for r in train_c + valid_c if _c0 <= r["wall"] < _c1]
    train_c = apply_g10(train_c); valid_c = apply_g10(valid_c)
    TIMING["decode_g10_s"] = time.time() - t0
    # reported, not raised: G10 runs on lattice candidates only, a different basis from the U-row checks above (review 2026-09-23)
    C["curled_check"]["G10_on_lattice_candidates"] = curl_admission([FIRST_FAIL.get(id(r)) if FIRST_FAIL.get(id(r)) == "G10" else None for r in _curl_cand])
    _g10c = C["curled_check"]["G10_on_lattice_candidates"]
    print(f"curled stretch at G10 (lattice candidates): {_g10c}" + (f"  ⚠ below {CURL_ADMIT_MIN}" if _g10c["admitted_frac"] is not None and _g10c["admitted_frac"] < CURL_ADMIT_MIN else ""))
    BODY_ALL = [v for vs in BODY.values() for v in vs]
    C["body_len_px"] = round(float(np.median(BODY_ALL)), 1) if BODY_ALL else None
    C["body_len_detail"] = {"n_masks": len(BODY_ALL), "derived_from": sorted(d for d in SOURCE_DAYS if d in TRAIN_DAYS), "basis": "every decoded lattice-candidate mask (train+valid) of the training days, uncapped",
                            "per_phase4": {ph: {"n": len(vs), "median": round(float(np.median(vs)), 1)} for ph, vs in sorted(BODY.items()) if vs}}
    print(f"G10: train {len(train_c)}, valid {len(valid_c)} survive ({TIMING['decode_g10_s']/60:.1f} min); body_len_px {C['body_len_px']} over {len(BODY_ALL)} masks, per phase4 {C['body_len_detail']['per_phase4']}")
    for r in valid_c: r["active"] = bool(r.get("act") is not None and r["act"] >= C["ACTIVE_THR"].get(r["day"], 1e9))
    TRAIN, ALLOC = allocate(train_c, W["n_max"], QUOTAS, C["ACTIVE_THR"], SEED)
    TRAIN, TRIMMED = hour_cap(TRAIN, W["hour_cap_frac"], SEED, strict=not SMOKE)
    if TRIMMED: print(f"⚠ hour cap trimmed {sum(TRIMMED.values())} rows: {TRIMMED}")
    HOUR_SHARE_MAX = max_hour_share(TRAIN)
    if HOUR_SHARE_MAX > W["hour_cap_frac"] + 1e-12 and not SMOKE:
        raise RuntimeError(f"hour cap violated after trimming: max (day, hour) share {HOUR_SHARE_MAX:.4f} > {W['hour_cap_frac']}")
    ALLOC["achieved_after_cap"], ALLOC["shortfall_after_cap"] = quota_achieved(TRAIN, QUOTAS)   # the cap trims night bins most; report the FINAL shares (review 2026-09-22)
    for k, v in ALLOC["shortfall_after_cap"].items(): print(f"⚠ QUOTA UNMET {k} (after hour cap): have {v['have']}, need {v['need']}")
    print(f"quota shares after cap: {ALLOC['achieved_after_cap']} (before cap: {ALLOC['achieved']})")
    BELOW_FLOOR = len(TRAIN) < W["min_train"]
    print(f"TRAIN full frames = {len(TRAIN)} (floor {W['min_train']}, cap {W['n_max']}); VALID = {len(valid_c)}; distinct 10-min blocks {distinct_10min_blocks(TRAIN)}")
    if BELOW_FLOOR: print(below_floor_message(len(TRAIN), W["min_train"], "after gating", W["stride_day_s"], W["stride_other_s"]))
    _nvm = night_valid_message(valid_c, "after gating")
    if _nvm: print(_nvm)

    # ---------------- image ids, crop windows ------------------------------------------------------------
    TRAIN.sort(key=lambda r: (r["day"], r["clip_stem"], r["frame_idx"])); valid_c.sort(key=lambda r: (r["day"], r["clip_stem"], r["frame_idx"]))
    ROOT_OF = {c["clip_stem"]: c["root"] for c in CLIPS}
    FRAMES = []
    for split, rs in (("train", TRAIN), ("valid", valid_c)):
        for r in rs:
            r["image_id"] = len(FRAMES) + 1; r["split"] = split
            rr = ratios(r, A_REF[r["day"]])
            FRAMES.append({"image_id": r["image_id"], "split": split, "day": r["day"], "root": ROOT_OF[r["clip_stem"]], "window": r["window"],
                           "clip_stem": r["clip_stem"], "frame_idx": r["frame_idx"], "row_i": r["row_i"], "t_sec": r["t_sec"], "wall": r["wall"].isoformat(),
                           "phase4": r["phase4"], "active": bool(r.get("active", False)), "r": [round(x, 4) for x in rr],
                           "S": round(sum(rr) / N_CALVES, 4), "seg": r["seg"], "link": r["link"], "cc_frac": r["cc_frac"],
                           "file_name": f"{r['clip_stem']}_f{r['frame_idx']:07d}.jpg", "crop": None})
    BYID = {f["image_id"]: f for f in FRAMES}; ROW_BYID = {r["image_id"]: r for r in TRAIN + valid_c}
    rngc = random.Random(SEED + 3)
    n_crop_target = int(round(W["crop_frac"] * len(TRAIN)))
    parents = rngc.sample(TRAIN, min(n_crop_target, len(TRAIN)))
    n_zero, no_window, NV = 0, 0, Counter()
    for r in parents:
        allow_zero = n_zero < CROP_ZERO_MAX * max(n_crop_target, 1)
        res = draw_crop_window(decoded(r), rngc, CW, CH, allow_zero=allow_zero)
        if res is None: no_window += 1; continue
        x, y, kept = res
        if not kept: n_zero += 1
        NV[len(kept)] += 1
        BYID[r["image_id"]]["crop"] = {"x": x, "y": y, "w": CW, "h": CH, "kept": kept, "n_visible": len(kept)}
    print(f"crops: {sum(NV.values())} windows from {len(parents)} parents ({no_window} parents got no window); n_visible histogram {dict(sorted(NV.items()))}; zero-calf {n_zero}")
    for k in range(N_CALVES + 1):
        if NV.get(k, 0) < NV_WARN_MIN: print(f"⚠ n_visible={k} occurs only {NV.get(k,0)} times (< {NV_WARN_MIN})")

    # ---------------- stage every tar (plates need whole days), plates, furniture diagnostic ------------
    # SMOKE stages only the tars the capped frames need plus one clip per (day, phase4) for the plates (review 2026-09-22)
    TARS = Tars(); t0 = time.time()
    STAGE_CLIPS = CLIPS
    if SMOKE:
        need = {r["clip_stem"] for r in train_c + valid_c}; seen = {(c["day"], phase4(c["wall_start"])) for c in CLIPS if c["clip_stem"] in need}
        for c in CLIPS:
            k = (c["day"], phase4(c["wall_start"]))
            if k not in seen: need.add(c["clip_stem"]); seen.add(k)
        STAGE_CLIPS = [c for c in CLIPS if c["clip_stem"] in need]
        print(f"⚠ SMOKE: staging {len(STAGE_CLIPS)} of {len(CLIPS)} tars; plates, furniture, sheets and rejected-gate samples are restricted to those clips")
    STAGED_STEMS = {c["clip_stem"] for c in STAGE_CLIPS}
    for c in STAGE_CLIPS: TARS.stage(c["day"], c["clip_stem"])
    print(f"staging: {STAGED_MB['stage_tars']:.0f} MB in {TIMING['stage_tars']:.0f} s ({STAGED_MB['stage_tars']/max(TIMING['stage_tars'],1e-9):.1f} MB/s)")
    t0 = time.time(); PLATE_FR = defaultdict(list)
    for c in STAGE_CLIPS:
        idx = read_json(osp.join(FR_ROOT, c["day"], f"{c['clip_stem']}_index.json")); w0 = clip_start(c["clip"]); last = -1e9
        for fr in idx["frames"]:
            if fr["t_sec"] - last < PLATE_EVERY_S: continue
            last = fr["t_sec"]; wl = w0 + timedelta(seconds=fr["t_sec"])
            PLATE_FR[(c["day"], phase4(wl))].append(gray4(TARS.read(c["day"], c["clip_stem"], int(fr["frame_idx"]))))
            if SMOKE and len(PLATE_FR[(c["day"], phase4(wl))]) >= SMOKE_PLATE_MAX: print(f"⚠ SMOKE: plate {c['day']}/{phase4(wl)} capped at {SMOKE_PLATE_MAX} frames"); break
    PLATES, PLATE_META = {}, {}; rngp = random.Random(SEED + 6)
    os.makedirs(osp.join(DS_DIR, "plates"), exist_ok=True)
    for d in SOURCE_DAYS:
        allf = [g for (dd, _), gs in PLATE_FR.items() if dd == d for g in gs]
        day_plate = build_plate(allf) if allf else None
        for ph in ("night", "dawn", "dusk", "day"):
            gs = PLATE_FR.get((d, ph), [])
            if len(gs) >= PLATE_MIN_FRAMES: PLATES[(d, ph)] = build_plate(gs); PLATE_META[f"{d}_{ph}"] = {"n_frames": len(gs), "borrowed": False}
            elif day_plate is not None:
                PLATES[(d, ph)] = day_plate; PLATE_META[f"{d}_{ph}"] = {"n_frames": len(gs), "borrowed": True}; print(f"  plate {d}/{ph}: only {len(gs)} frames — borrows the day plate")
            else: continue
            # seeded random sample over the WHOLE (day, phase4), not the first 40 in clip order (review 2026-09-22)
            res_fr = rngp.sample(gs, min(PLATE_RES_SAMPLE, len(gs)))
            if len(gs) > PLATE_RES_SAMPLE: print(f"⚠ plate residual sample {d}/{ph}: {PLATE_RES_SAMPLE} of {len(gs)} frames (seeded)")
            res = [np.abs(g.astype(np.int16) - PLATES[(d, ph)].astype(np.int16)).ravel() for g in res_fr]
            thr3, med, mad = plate_thr(np.concatenate(res) if res else np.zeros(1), 3.0)   # 3.0 = the 3·MAD furniture threshold; not derived from any day
            PLATE_META[f"{d}_{ph}"].update({"thr_3mad": thr3, "res_median": med, "res_mad": mad, "res_n_frames": len(res_fr)})
            Image.fromarray(PLATES[(d, ph)]).save(osp.join(LOCAL, f"{d}_{ph}.png")); shutil.copy(osp.join(LOCAL, f"{d}_{ph}.png"), osp.join(DS_DIR, "plates", f"{d}_{ph}.png"))
    TIMING["plates_s"] = time.time() - t0
    # furniture diagnostic (reported, not gated)
    t0 = time.time(); FLAGGED = []
    cent_index = defaultdict(list)
    for r in ROWS:
        for s in range(N_CALVES):
            if r["cent"][s] is not None: cent_index[(r["window"], s)].append((r["wall"], r["cent"][s]))
    for k in cent_index: cent_index[k].sort()
    cent_walls = {k: [w for w, _ in v] for k, v in cent_index.items()}
    def moved_px(window, s, wall, cent):
        lst = cent_index[(window, s)]; i = bisect_right(cent_walls[(window, s)], wall - timedelta(seconds=FURN_LOOKBACK_S)) - 1
        if i < 0: return None
        w, c = lst[i]
        if (wall - w).total_seconds() > FURN_LOOKBACK_S + 60: return None
        return math.hypot(c[0] - cent[0], c[1] - cent[1])
    for f in FRAMES:
        key = (f["day"], f["phase4"])
        if key not in PLATES: continue
        r = ROW_BYID[f["image_id"]]; g = gray4(TARS.read(f["day"], f["clip_stem"], f["frame_idx"])); thr3 = PLATE_META[f"{key[0]}_{key[1]}"]["thr_3mad"]
        f["furniture_frac"] = []
        for s, m in enumerate(decoded(r)):
            fc = furniture_consistency(m[::4, ::4], g, PLATES[key], thr3); f["furniture_frac"].append(round(fc, 3))
            mv = moved_px(f["window"], s, r["wall"], r["cent"][s])
            if fc >= FURN_CONSIST and mv is not None and mv < FURN_MOVE_PX: FLAGGED.append({"image_id": f["image_id"], "slot": s, "frac": round(fc, 3), "moved_px": round(mv, 1)})
    TIMING["furniture_s"] = time.time() - t0
    print(f"furniture diagnostic: {len(FLAGGED)} flagged masks (expected ≈ 0)")

    # ---------------- human-gate sheets ---------------------------------------------------
    t0 = time.time(); rngs = random.Random(SEED + 4)
    trainF = [f for f in FRAMES if f["split"] == "train"]
    tail = [f for f in trainF if any(x < TOL41_FRAG_LO for x in f["r"])]
    scale = W["sheet_n"] / float(SHEET_N_REF)                                   # the mix is stated for 300 counted cells
    n_tail = int(round(SHEET_MIX["tail_enlarged" if C["bad_overlap"]["high"] else "tail"] * scale))
    plan = [("tail", n_tail)] + [(k, int(round(SHEET_MIX[k] * scale))) for k in ("night", "dusk", "dawn", "day")]
    chosen, used = [], set()
    for kind, n in plan:
        pool = tail if kind == "tail" else [f for f in trainF if f["phase4"] == kind]
        pool = [f for f in pool if f["image_id"] not in used]
        pick = rngs.sample(pool, min(n, len(pool)))
        if len(pick) < n: print(f"⚠ sheet shortfall {kind}: {len(pick)}/{n}")
        for f in pick: used.add(f["image_id"]); chosen.append((kind, f, True))
    extras = []
    for pw in PROBE_WALLS:
        pwd = datetime.fromisoformat(pw); best = min(FRAMES, key=lambda f: abs((datetime.fromisoformat(f["wall"]) - pwd).total_seconds()), default=None)
        if best and abs((datetime.fromisoformat(best["wall"]) - pwd).total_seconds()) <= PROBE_TOL_S: extras.append(("probe", best))
        else: print(f"  ⚠ probe {pw}: no admitted frame within {PROBE_TOL_S:.0f} s (not admitted)")
    for fl in FLAGGED[:FURN_SHEET_MAX]: extras.append(("flagged_furniture", BYID[fl["image_id"]]))
    if len(FLAGGED) > FURN_SHEET_MAX: print(f"⚠ flagged_furniture sheet capped at {FURN_SHEET_MAX} of {len(FLAGGED)}")
    rej_by_gate = defaultdict(list)
    for r in ROWS:
        g = FIRST_FAIL.get(id(r))
        if g in ("G4", "G5", "G6", "G7", "G8", "G9", "G10") and "rle_np" in r and r["clip_stem"] in STAGED_STEMS: rej_by_gate[g].append(r)
    for g in ("G4", "G5", "G6", "G7", "G8", "G9", "G10"):
        for r in rngs.sample(rej_by_gate[g], min(SHEET_CELLS, len(rej_by_gate[g]))):
            rr = ratios(r, A_REF[r["day"]])
            extras.append((f"rejected_{g}", {"image_id": None, "day": r["day"], "clip_stem": r["clip_stem"], "frame_idx": r["frame_idx"], "wall": r["wall"].isoformat(),
                                             "phase4": r["phase4"], "r": [round(x, 3) for x in rr], "S": round(sum(rr) / 4, 3), "_row": r, "crop": None}))
    # the certified curled stretch's own G6-G9 rejects (review 2026-09-23): the evidence for a curl_band_ack decision.
    # Own rng so the counted/crop draws above and below are unchanged
    curl_rej = [r for r in CURL_U if FIRST_FAIL.get(id(r)) in ("G6", "G7", "G8", "G9") and "rle_np" in r and r["clip_stem"] in STAGED_STEMS]
    for r in random.Random(SEED + 7).sample(curl_rej, min(SHEET_CELLS, len(curl_rej))):
        rr = ratios(r, A_REF[r["day"]])
        extras.append(("curl_rejected", {"image_id": None, "day": r["day"], "clip_stem": r["clip_stem"], "frame_idx": r["frame_idx"], "wall": r["wall"].isoformat(),
                                         "phase4": r["phase4"], "r": [round(x, 3) for x in rr], "S": round(sum(rr) / 4, 3), "_row": r, "crop": None}))
    if curl_rej: print(f"curl_rejected sheet: {min(SHEET_CELLS, len(curl_rej))} of {len(curl_rej)} G6-G9-rejected rows of the curled stretch (first-fail {C['curled_check']['after_G6_G9'].get('first_fail')})")
    crop_par =[f for f in trainF if f["crop"]]
    for f in rngs.sample(crop_par, min(SHEET_CROP_CELLS, len(crop_par))): extras.append(("crop", f))
    cells_meta, sheet_rows = [], []
    def masks_for(f):
        r = f.get("_row") or ROW_BYID.get(f["image_id"])
        return [{"size": [H_FRAME, W_FRAME], "counts": c.tolist()} for c in r["rle_np"]]
    def emit_sheets(items, prefix, counted):
        sheet_no = 0; batch = []
        for kind, f in items:
            wall = datetime.fromisoformat(f["wall"]); cell_id = f"{prefix}{sheet_no:03d}c{len(batch):02d}"
            strip = "r=" + "/".join(f"{x:.2f}" for x in f["r"]) + f" S={f['S']:.2f}"
            crop = (f["crop"]["x"], f["crop"]["y"], f["crop"]["w"], f["crop"]["h"]) if (kind == "crop" and f.get("crop")) else None
            im = render_cell(TARS.read(f["day"], f["clip_stem"], f["frame_idx"]), masks_for(f), f["day"], wall, f["clip_stem"], f["frame_idx"], strip, f["phase4"], cell_id, crop=crop)
            batch.append(im)
            cells_meta.append({"cell_id": cell_id, "sheet": f"{prefix}{sheet_no:03d}.png", "cell": len(batch) - 1, "image_id": f["image_id"], "day": f["day"],
                               "clip_stem": f["clip_stem"], "frame_idx": f["frame_idx"], "wall": f["wall"], "phase4": f["phase4"], "counted": counted, "kind": kind, "verdict": None})
            if len(batch) == SHEET_CELLS: write_sheet(batch, f"{prefix}{sheet_no:03d}.png"); sheet_no += 1; batch = []
        if batch: write_sheet(batch, f"{prefix}{sheet_no:03d}.png")
    emit_sheets([(k, f) for k, f, _ in chosen], "S", True)
    by_kind = defaultdict(list)
    for k, f in extras: by_kind[k].append((k, f))
    for k, items in by_kind.items(): emit_sheets(items, f"{k}_", False)
    SHEET_N_PLANNED = sum(n for _, n in plan)          # may exceed sheet_n by 50 when the tail stratum is enlarged (bad_overlap_high)
    # per-day review basis (review 2026-09-23): a fault-listed day was reviewed end to end; an unlisted day is
    # reviewed ONLY through its counted cells here, so it must have some
    COUNTED_BY_DAY = dict(Counter(c["day"] for c in cells_meta if c["counted"]))
    REVIEW_BASIS = {d: ("user fault list (whole video reviewed)" if d not in UNLISTED else f"human_gate counted cells only (n={COUNTED_BY_DAY.get(d, 0)})") for d in SOURCE_DAYS}
    print(f"counted cells by day: {COUNTED_BY_DAY}; review basis {REVIEW_BASIS}")
    for d in UNLISTED:
        if any(f["day"] == d for f in trainF) and not COUNTED_BY_DAY.get(d) and not SMOKE:
            raise RuntimeError(f"unlisted source day {d} contributes train frames but got 0 counted sheet cells — nobody would ever look at its masks")
    TEMPLATE = {"tag": TAG, "gate_run_id": GATE_RUN_ID, "counted_by_day": COUNTED_BY_DAY, "review_basis_by_day": REVIEW_BASIS, "status": "pending", "vocabulary": GATE_VOCAB, "sheet_n": W["sheet_n"], "sheet_n_planned": SHEET_N_PLANNED,
                "sheet_plan": dict(plan), "sheet_n_required_min": SHEET_N_REF, "require_human_gate_at_gate": W["require_human_gate"],
                "rule": "PASSED iff status=='passed' AND count(verdict in BAD) <= 3 among counted cells AND zero 'wrong_object' anywhere; "
                        "FAILED iff status=='failed' OR any 'wrong_object' OR > 3 BAD counted cells (whatever the status says); "
                        "BAD = partial|split|merged|unlabelled_calf|wrong_object; partial_minor (clipped head/ear/tail per the user's criterion) counts as ok. "
                        "require_human_gate=false (default, SPEC addendum 2026-09-23): filling this file is OPTIONAL — without it the dataset can seal; "
                        "if you fill it, every BAD cell removes +-90 s of admitted frames and a FAILED gate unseals. require_human_gate=true: only PASSED seals.",
                "cells": [{k: v for k, v in c.items() if k not in ("_row",)} for c in cells_meta]}
    with open(osp.join(DS_DIR, "human_gate_template.json"), "w") as f: json.dump(TEMPLATE, f, indent=1)
    with open(osp.join(FIG_DIR, "sheets_index.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["cell_id", "sheet", "cell", "image_id", "day", "clip_stem", "frame_idx", "wall", "phase4", "counted", "kind"])
        for c in cells_meta: wr.writerow([c[k] for k in ("cell_id", "sheet", "cell", "image_id", "day", "clip_stem", "frame_idx", "wall", "phase4", "counted", "kind")])
    for c in cells_meta:
        if c["counted"]: BYID[c["image_id"]]["counted_cell"] = c["cell_id"]
    TIMING["sheets_s"] = time.time() - t0
    print(f"sheets: {sum(1 for c in cells_meta if c['counted'])} counted cells + {sum(1 for c in cells_meta if not c['counted'])} extras in {TIMING['sheets_s']/60:.1f} min")

    # ---------------- position prior (frame-blind medoid per phase4) -------------------------------------
    PP = {}
    for ph in ("night", "dawn", "dusk", "day"):
        pool = [f for f in trainF if f["phase4"] == ph]
        if not pool: continue
        samp = rngs.sample(pool, min(PRIOR_SAMPLE_N, len(pool)))
        sets = [np.stack([m[::PRIOR_DS, ::PRIOR_DS] for m in decoded(ROW_BYID[f["image_id"]])]) for f in samp]
        med = position_prior_medoid(sets, [f["image_id"] for f in samp])
        f = BYID[med["image_id"]]
        PP[ph] = {"image_id": med["image_id"], "clip_stem": f["clip_stem"], "frame_idx": f["frame_idx"], "wall": f["wall"], "n_sample": len(samp),
                  "mean_matched_iou": med["mean_matched_iou"], "masks": masks_for(f)}
    with open(osp.join(DS_DIR, "position_prior.json"), "w") as f: json.dump({"tag": TAG, "gate_run_id": GATE_RUN_ID, "phase_function": "phase4", "source_days": SOURCE_DAYS, "per_phase4": PP}, f)

    # ---------------- figures: band derivation + distribution --------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
        for i, p in enumerate(("dayl", "dark")):
            dst = C["distribution"][p]; e = np.asarray(dst["hist_edges"]); ax[i].bar(e[:-1], dst["hist"], width=0.05, align="edge", color="#888")
            ax[i].axvline(C["frag_lo"][p], color="r", label=f"frag_lo[{p}]={C['frag_lo'][p]}"); ax[i].axvline(OVER_HI, color="b", label=f"over_hi={OVER_HI}")
            ax[i].axvline(TOL41_FRAG_LO, color="orange", ls="--", label=f"Tol_41 {TOL41_FRAG_LO}"); ax[i].set_title(f"r = area/A_ref over U, pool {p} (n={dst['n']})"); ax[i].legend()
        fig.tight_layout(); fig.savefig(osp.join(LOCAL, "band.png"), dpi=110); plt.close(fig); shutil.copy(osp.join(LOCAL, "band.png"), osp.join(FIG_DIR, "band_derivation.png"))
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
        hrs = sorted({(f["day"], datetime.fromisoformat(f["wall"]).hour) for f in FRAMES}); full_h = Counter((r["day"], r["wall"].hour) for r in ROWS)
        adm_h = Counter((f["day"], datetime.fromisoformat(f["wall"]).hour) for f in trainF)
        ax[0].bar(range(len(hrs)), [adm_h[h] for h in hrs]); ax[0].set_xticks(range(len(hrs))); ax[0].set_xticklabels([f"{d[4:]}h{h}" for d, h in hrs], rotation=90, fontsize=6); ax[0].set_title("train frames by (day, hour)")
        phc = Counter(f["phase4"] for f in trainF); ax[1].bar(list(phc), list(phc.values())); ax[1].set_title("train by phase4")
        acts = [ROW_BYID[f["image_id"]].get("act") or 0 for f in trainF]; ax[2].hist(acts, bins=40); ax[2].set_title("activity score (train)")
        fig.tight_layout(); fig.savefig(osp.join(LOCAL, "dist.png"), dpi=110); plt.close(fig); shutil.copy(osp.join(LOCAL, "dist.png"), osp.join(FIG_DIR, "distribution.png"))
    except Exception as e:
        print(f"⚠ figures not rendered: {e}")

    # ---------------- admitted.json + metrics + done ------------------------------------------------------
    full_by_hour = Counter((r["day"], r["wall"].hour) for r in ROWS); adm_by_hour = Counter((f["day"], datetime.fromisoformat(f["wall"]).hour) for f in trainF)
    low_hours = [f"{d}_h{h:02d}" for (d, h), n in full_by_hour.items() if adm_by_hour[(d, h)] / max(n, 1) < LOW_HOUR_FRAC and n >= LOW_HOUR_MIN_ROWS]
    if low_hours: print(f"⚠ hours with admitted/full < {LOW_HOUR_FRAC}: {low_hours}")
    for f in FRAMES: f.pop("_row", None)
    ADMITTED = {"tag": TAG, "gate_run_id": GATE_RUN_ID, "window_files": WINDOW_FILES, "unlisted_days": UNLISTED, "review_basis_by_day": REVIEW_BASIS,
                "smoke": SMOKE, "seed": SEED, "sources": PAIRS, "sources_str": W["sources"], "source_days": SOURCE_DAYS,
                "faults_json": W["faults_json"], "faults_sha256": FAULTS_SHA, "band_json": BAND_PATH, "band_sha256": BAND_SHA,   # provenance as GATED (review 2026-09-22)
                "note_exclusions": NOTE_REP,                   # notebook-side exclusions from the fault file's notes (review 2026-09-23)
                "constants": C, "band_json_crosscheck": BAND_XCHK,
                "split": {"train_clips": TRAIN_STEMS, "valid_clips": VALID_STEMS, "valid_max": W["valid_max"]}, "clips": [{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in c.items()} for c in CLIPS],
                "frames": FRAMES, "blocks": BLOCK_REP, "below_floor": BELOW_FLOOR, "crop_wh": [CW, CH], "counts": {"train_full": len(TRAIN), "valid": len(valid_c), "train_crops": sum(1 for f in trainF if f["crop"])},
                "plates": PLATE_META, "furniture_flagged": FLAGGED, "sheet_cells": TEMPLATE["cells"],
                "gate_widgets": {k: W[k] for k in GATE_KEYS}, "hour_share_max": round(HOUR_SHARE_MAX, 4)}   # teacher/finalize run with these (review 2026-09-22)
    if C["curled_check"]["after_G6_G9"].get("blocking"):
        # deferred raise (review 2026-09-23): sheets + metrics exist for the decision, admitted.json / .done_gate do not
        MET.update({"band_derivation": C, "funnel": {d: {p: {g: v for g, v in gs.items()} for p, gs in ps.items()} for d, ps in FUNNEL.items()}, "gate_aborted": "curl_band_check"})
        save_metrics(MET); TARS.close_all()
        _cb = C["curled_check"]["after_G6_G9"]
        raise RuntimeError(f"the reviewer-confirmed curled stretch {CURL_START}..{CURL_END} is rejected DISPROPORTIONATELY by G6-G9: admitted {_cb['admitted_frac']} "
                           f"(first-fail {_cb['first_fail']}) < {CURL_REL_MIN} x the dark-pool reference {_cb['ref_dark_outside_stretch']['admitted_frac']} "
                           f"(first-fail {_cb['ref_dark_outside_stretch']['first_fail']}) — the band must keep admitting this stretch. Inspect {FIG_DIR}/curl_rejected_*.png and rejected_G6..G9_*.png; "
                           "then re-run gate with collapse_override_blocks (dropped G9 blocks) and/or curl_band_ack='<reason>' (recorded in admitted.json and the manifest). "
                           "No admitted.json was written.")
    write_fault_copies()                          # provenance copies only now, by the gate that actually used the file (review 2026-09-22)
    with open(osp.join(DS_DIR, "admitted.json"), "w") as f: json.dump(ADMITTED, f, default=str)
    MET.update({"funnel": {d: {p: {g: v for g, v in gs.items()} for p, gs in ps.items()} for d, ps in FUNNEL.items()}, "lost_rows_by_fault_id": dict(LOST_BY_FAULT),
                "event_stats": EV_STATS, "band_derivation": C, "band_json_crosscheck": BAND_XCHK, "blocks": BLOCK_REP,
                "admitted_vs_full": {"by_hour": {f"{d}_h{h:02d}": [adm_by_hour[(d, h)], n] for (d, h), n in sorted(full_by_hour.items())},
                                     "by_phase4": {ph: [sum(1 for f in trainF if f["phase4"] == ph), sum(1 for r in ROWS if r["phase4"] == ph)] for ph in ("night", "dawn", "dusk", "day")},
                                     "low_hours": low_hours, "distinct_10min_blocks": distinct_10min_blocks(TRAIN)},
                "sampling": {"alloc": ALLOC, "hour_cap_trimmed": TRIMMED, "hour_share_max": round(HOUR_SHARE_MAX, 4), "g12_dropped": n_g12, "cross_clip_overlap_dropped": N_XCLIP, "below_floor": BELOW_FLOOR},
                "split": ADMITTED["split"], "counts": ADMITTED["counts"], "crop_variant": {"n_visible_hist": {str(k): v for k, v in sorted(NV.items())}, "no_window_parents": no_window, "zero_calf": n_zero},
                "plates": PLATE_META, "furniture": {"flagged": len(FLAGGED)}, "timing": {**{k: round(v, 1) for k, v in TIMING.items()}, "staged_mb": {k: round(v) for k, v in STAGED_MB.items()}, "gate_total_s": round(time.time() - t_gate0, 1)}})
    save_metrics(MET); TARS.close_all(); mark_done("gate")
    print(f"\nstage gate complete for {TAG} in {(time.time()-t_gate0)/60:.1f} min. Next: stage=teacher (A100). Sheets are in {FIG_DIR}; "
          + ("filling human_gate.json is REQUIRED (require_human_gate=true)." if W["require_human_gate"] else
             "filling human_gate.json is OPTIONAL (require_human_gate=false: the fault list is the human gate)."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage `teacher` (A100, ≈ 1 A100-h) — SAM3 `detect()` verbatim on every admitted full frame: G11 veto (+ `teacher_detect` by-product)
# MAGIC
# MAGIC **The only GPU cell.** Runs on a GPU job cluster (`17.3.x-gpu-ml-scala2.13` on `Standard_NC24ads_A100_v4`, the
# MAGIC `sub_trk0418n.json` template). On a cluster without CUDA (or without torch) it is **SKIPPED with a loud banner**: nothing
# MAGIC is written and `.done_teacher` is not marked, so `finalize` with `require_veto=true` refuses to seal. The veto is waived
# MAGIC only by choosing `require_veto=false` at `finalize`, recorded as `sam3_veto.ran=false`.
# MAGIC Every output is stamped with `sha256(admitted.json)`; `finalize` refuses outputs computed on another gate run.
# MAGIC The `teacher_detect` labels are a free by-product (decision 2026-09-23: no DET-C training planned); nothing
# MAGIC downstream requires them.

# COMMAND ----------

RUN_TEACHER = False
if not SELFTEST_ONLY and STAGE == "teacher":
    check_done("teacher")
    # a SEALED tag is immutable (mirrors stage gate): a re-run would rewrite veto.json / teacher_dets.json.gz /
    # intersection_teacher_image_ids.json and MET veto/teacher_detect under a manifest Tol_43/Tol_44 already trust,
    # and the manifest's sha256 block does not cover those files — force does not override this (review 2026-09-23)
    _mp = osp.join(DS_DIR, "manifest.json")
    if osp.exists(_mp) and read_json(_mp).get("sealed"):
        raise RuntimeError(f"tag {TAG} has a SEALED manifest.json — a sealed tag is immutable (force=true does not override); bump the tag")
    try:
        import torch
        _cuda = torch.cuda.is_available()
    except ImportError:
        _cuda = False
    if _cuda:
        RUN_TEACHER = True
    else:
        print("\n" + "⚠" * 60 + "\n⚠ STAGE teacher SKIPPED — no CUDA device on this cluster. The SAM3 unlabelled-animal veto (G11) did NOT run.\n"
              "⚠ Nothing was written and .done_teacher was NOT marked. finalize with require_veto=true will REFUSE to seal.\n"
              "⚠ Run stage=teacher on 17.3.x-gpu-ml-scala2.13 / Standard_NC24ads_A100_v4 (sub_trk0418n.json template),\n"
              "⚠ or waive the veto deliberately with require_veto=false at finalize (recorded as sam3_veto.ran=false).\n" + "⚠" * 60 + "\n")

if RUN_TEACHER:
    print("stage=teacher on GPU:", torch.cuda.get_device_name(0))
    from huggingface_hub import login
    login(token=dbutils.secrets.get(scope="hf", key="token").strip().strip('"').strip("'").strip())
    from transformers import Sam3Model, Sam3Processor
    device = torch.device("cuda"); DTYPE = torch.bfloat16                       # Tol_03:120 on Ampere+, Tol_35:69
    dproc = Sam3Processor.from_pretrained(SAM3_ID)                               # Tol_03:422
    det = Sam3Model.from_pretrained(SAM3_ID, torch_dtype=DTYPE).to(device).eval()  # Tol_03:423

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

    ADM_P = osp.join(DS_DIR, "admitted.json"); ADM_SHA = sha256_file(ADM_P)
    ADMITTED = read_json(ADM_P); assert_admitted_matches(ADMITTED); C = ADMITTED["constants"]; A_REF = C["A_ref"]
    FRAMES = ADMITTED["frames"]; MET = load_metrics(); CW, CH = ADMITTED["crop_wh"]
    by_clip = defaultdict(list)
    for f in FRAMES: by_clip[(f["root"], f["day"], f["window"], f["clip_stem"])].append(f)
    TARS = Tars(); OUT, VETO, INTER = {}, {}, []
    n_drop_gate, n_drop_match, n_crop_fail = Counter(), 0, 0; t_det = 0.0; n_done = 0; t0 = time.time()
    def teacher_crop_ok(tm, crop):
        """The parent's crop window applied to the teacher masks must satisfy the same keep/drop rule as the tracked
        crop; a parent whose teacher crop fails leaves BOTH label sources' intersection so that
        DET-R(intersection) and DET-C train on identical image sets (review 2026-09-22)."""
        x, y = crop["x"], crop["y"]; kept = []
        for s, m in enumerate(tm):
            fr_in = int(m[y:y + CH, x:x + CW].sum()) / max(int(m.sum()), 1)
            if fr_in >= CROP_KEEP_FRAC: kept.append(s)
            elif fr_in > CROP_DROP_FRAC: return None
        return kept
    CLIP_SHA = {c["clip_stem"]: c.get("masks_sha256") for c in ADMITTED["clips"]}
    for (root, day, win, stem), fs in sorted(by_clip.items()):
        d = read_masks_checked(osp.join(TOL, root, win, f"masks_{stem}.json.gz"), CLIP_SHA.get(stem))   # byte-identical to the gated file (review 2026-09-23)
        want = {f["frame_idx"] for f in fs}
        rle_by_idx = {int(r["frame_idx"]): r["masks"] for r in d["frames"] if int(r["frame_idx"]) in want}
        del d
        TARS.stage(day, stem)
        for f in sorted(fs, key=lambda z: z["frame_idx"]):
            pil = Image.open(io.BytesIO(TARS.read(day, stem, f["frame_idx"]))).convert("RGB")
            stored = [rle_decode(rle_by_idx[f["frame_idx"]][str(s)]) for s in range(N_CALVES)]
            union = np.logical_or.reduce(stored)
            tt = time.time(); dets = detect(pil, topk=N_CALVES + 2); t_det += time.time() - tt
            a_ref = A_REF[day]; rec = {"image_id": f["image_id"], "n_det": len(dets), "areas": [], "scores": [], "max_ios_vs_union": [], "best_iou_per_det": []}
            vet, vet_area, vet_ios = False, None, None
            for m, sc in dets:
                a = int(m.sum()); ios_u = m_ios(m, union); biou = max(m_iou(m, sm) for sm in stored)
                rec["areas"].append(a); rec["scores"].append(round(sc, 4)); rec["max_ios_vs_union"].append(round(ios_u, 4)); rec["best_iou_per_det"].append(round(biou, 4))
                if a >= VETO_AREA_FRAC * a_ref and ios_u < VETO_IOS and not vet: vet, vet_area, vet_ios = True, a, round(ios_u, 4)
            VETO[str(f["image_id"])] = {"vetoed": vet, "det_area": vet_area, "ios": vet_ios}
            # mechanism-control labels: Hungarian on max(IoU, IoS), keep the four matched at >= TEACHER_MATCH_MIN, gates G4/G6/G7/G10 as a set
            tm, hung = None, None
            if len(dets) >= N_CALVES:
                S = np.zeros((N_CALVES, len(dets)), np.float32)
                for a_, sm in enumerate(stored):
                    for b_, (dm, _) in enumerate(dets): S[a_, b_] = max(m_iou(sm, dm), m_ios(sm, dm))
                ri, cj = linear_sum_assignment(-S)
                if len(ri) == N_CALVES and all(S[a_, b_] >= TEACHER_MATCH_MIN for a_, b_ in zip(ri, cj)):
                    tm = [None] * N_CALVES
                    for a_, b_ in zip(ri, cj): tm[a_] = dets[b_][0]
                    hung = round(float(np.mean([m_iou(stored[a_], tm[a_]) for a_ in range(N_CALVES)])), 4)
            ok, why = False, "lt4_matched"
            if tm is not None:
                rr = [int(m.sum()) / a_ref for m in tm]; pool = pool_of(f["phase4"]); why = None
                if any(m_iou(tm[i], tm[j]) > DISJ_IOU or m_ios(tm[i], tm[j]) > DISJ_IOS for i in range(4) for j in range(i + 1, 4)): why = "G4"
                elif any(x < C["frag_lo"][pool] or x > C["over_hi"] for x in rr): why = "G6"
                elif not (C["S_lo"][pool] <= sum(rr) / 4 <= C["S_hi"][pool]): why = "G7"
                elif any(largest_cc_frac(m) < CC_FRAC for m in tm): why = "G10"
                ok = why is None
            else: n_drop_match += 1
            if why: n_drop_gate[why] += 1
            crop_kept = None
            if ok and f.get("crop"):
                crop_kept = teacher_crop_ok(tm, f["crop"])
                if crop_kept is None: ok, why = False, "crop_window"; n_crop_fail += 1; n_drop_gate[why] += 1
            rec.update({"teacher_ok": ok, "teacher_fail": why, "mimicry_descriptive_only": {"hungarian_iou_tracked_vs_teacher": hung},   # descriptive only: measures mimicry, not detector quality
                        "teacher_masks": [rle_encode(m) for m in tm] if ok else None, "teacher_crop_kept": crop_kept if ok else None})
            OUT[str(f["image_id"])] = rec
            # intersection = TRAIN full frames both label sources emit (teacher_ok incl. the crop rule, not vetoed);
            # valid stays unfiltered for both arms and Tol_43 keeps crops via parent_file_name
            if ok and not vet and f["split"] == "train": INTER.append(f["image_id"])
            n_done += 1
            if n_done % 200 == 0: print(f"  {n_done}/{len(FRAMES)} frames, {t_det/max(n_done,1)*1000:.0f} ms/detect, {(time.time()-t0)/60:.1f} min", flush=True)
        TARS.drop(stem)
    n_vet = sum(1 for v in VETO.values() if v["vetoed"]); n_drop = sum(1 for r in OUT.values() if not r["teacher_ok"])
    drop_frac = n_drop / max(len(OUT), 1)
    # every output is keyed by image_id and therefore valid only for THIS admitted.json (review 2026-09-22)
    with gzip.open(osp.join(DS_DIR, "teacher_dets.json.gz"), "wt") as f: json.dump({"tag": TAG, "admitted_sha256": ADM_SHA, "detector": "Sam3Model facebook/sam3 bf16, detect() verbatim Tol_03:453-472", "frames": OUT}, f)
    with open(osp.join(DS_DIR, "veto.json"), "w") as f: json.dump({"tag": TAG, "admitted_sha256": ADM_SHA, "ran": True, "n_checked": len(VETO), "n_vetoed": n_vet, "rule": "det area >= 0.30*A_ref[day] and IoS < 0.30 vs union of the 4 stored masks", "by_image": VETO}, f)
    # stage-teacher intersection, never rewritten: finalize writes the post-drop set to intersection_image_ids.json (the file
    # Tol_43 reads) and can therefore be re-run with corrected verdicts (review 2026-09-22)
    with open(osp.join(DS_DIR, "intersection_teacher_image_ids.json"), "w") as f: json.dump({"tag": TAG, "admitted_sha256": ADM_SHA, "basis": "train full frames: teacher_ok (incl. crop rule) and not vetoed (before human drops)", "image_ids": sorted(INTER)}, f)
    MET["veto"] = {"ran": True, "n_checked": len(VETO), "n_vetoed": n_vet, "admitted_sha256": ADM_SHA}
    MET["teacher_detect"] = {"ran": True, "n_frames": len(OUT), "n_dropped": n_drop, "drop_frac": round(drop_frac, 4), "drop_high": drop_frac > TEACHER_DROP_HIGH,
                             "drop_high_threshold": TEACHER_DROP_HIGH, "f9_comparator_label_source": "tracked_intersection",
                             "drop_by_reason": dict(n_drop_gate), "lt4_matched": n_drop_match, "crop_window_fail": n_crop_fail, "intersection_n": len(INTER), "admitted_sha256": ADM_SHA,
                             "mimicry_descriptive_only": {"hungarian_iou_tracked_vs_teacher_mean": round(float(np.mean([r["mimicry_descriptive_only"]["hungarian_iou_tracked_vs_teacher"] for r in OUT.values() if r["mimicry_descriptive_only"]["hungarian_iou_tracked_vs_teacher"] is not None] or [0])), 4)},
                             "ms_per_detect": round(1000 * t_det / max(n_done, 1), 1)}
    MET.setdefault("timing", {}).update({"teacher_stage_s": round(time.time() - t0, 1), "teacher_detect_s": round(t_det, 1),
                                         "teacher_staging_s": round(TIMING["stage_tars"], 1), "teacher_staged_mb": round(STAGED_MB["stage_tars"])})
    save_metrics(MET); TARS.close_all(); mark_done("teacher")
    print(f"\nstage teacher: {len(OUT)} frames, vetoed {n_vet}, teacher labels dropped {n_drop} ({drop_frac:.1%}"
          f"{f' — drop_high (> {TEACHER_DROP_HIGH:.0%})' if drop_frac > TEACHER_DROP_HIGH else ''}), intersection {len(INTER)}")
    print(f"teacher_detect is a by-product (decision 2026-09-23: no DET-C training planned; nothing downstream requires it). "
          f"It covers {len(OUT) - n_drop}/{len(OUT)} frames. Should DET-C ever be trained, the comparison requires the DET-R arm on "
          "label_source=tracked_intersection (same admitted frames) at ANY drop rate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage `finalize` (CPU) — human verdicts, G11, COCO writing, verification, packaging, manifest

# COMMAND ----------

if not SELFTEST_ONLY and STAGE == "finalize":
    # finalize is re-runnable with a new human_gate.json — but never silently over a SEALED manifest,
    # which Tol_43/Tol_44 hash (review 2026-09-22); every re-run is recorded in manifest.finalize_runs
    t_fin0 = time.time(); _mp = osp.join(DS_DIR, "manifest.json"); PREV_MAN = read_json(_mp) if osp.exists(_mp) else None
    if PREV_MAN is not None:
        if PREV_MAN.get("sealed") and not W["force"]:
            raise RuntimeError(f"tag {TAG} already has a SEALED manifest (created {PREV_MAN.get('created')}); re-sealing changes the manifest sha Tol_43/Tol_44 rely on — set force=true only if nothing was trained on it")
        print(f"⚠ finalize re-run: previous manifest created {PREV_MAN.get('created')} sealed={PREV_MAN.get('sealed')} is superseded (recorded in finalize_runs)")
    ADM_P = osp.join(DS_DIR, "admitted.json"); ADM_SHA = sha256_file(ADM_P)
    ADMITTED = read_json(ADM_P); assert_admitted_matches(ADMITTED); C = ADMITTED["constants"]; FRAMES = ADMITTED["frames"]; MET = load_metrics()
    BYID = {f["image_id"]: f for f in FRAMES}
    veto_p = osp.join(DS_DIR, "veto.json"); VETO_RAN = osp.exists(veto_p) and osp.exists(done_path("teacher")) or (osp.exists(veto_p) and SMOKE)
    if W["require_veto"] and not VETO_RAN: raise RuntimeError("finalize: stage teacher has not run (no veto.json/.done_teacher) and require_veto=true")
    if not VETO_RAN: print("⚠ SAM3 veto NOT run (require_veto=false) — recorded as sam3_veto.ran=false; the manifest may still seal")
    VETO = {}
    if VETO_RAN:
        _vj = read_json(veto_p); assert_fresh(_vj, "veto.json", ADM_SHA); VETO = _vj["by_image"]
    TEACHER = {}; INTER_TEACHER = set()
    if osp.exists(osp.join(DS_DIR, "teacher_dets.json.gz")):
        with gzip.open(osp.join(DS_DIR, "teacher_dets.json.gz"), "rt") as f: _tj = json.load(f)
        assert_fresh(_tj, "teacher_dets.json.gz", ADM_SHA); TEACHER = _tj["frames"]
        # the stage-teacher intersection is REBUILT from its sources (teacher_ok, not vetoed, train) and cross-checked against
        # the teacher stage's own file, which finalize never rewrites — so finalize stays re-runnable (review 2026-09-22)
        if not osp.exists(veto_p): raise RuntimeError("teacher_dets.json.gz present but veto.json missing — stage teacher incomplete; re-run it")
        _vj0 = read_json(veto_p); assert_fresh(_vj0, "veto.json", ADM_SHA)
        INTER_TEACHER = {int(k) for k, r in TEACHER.items() if r.get("teacher_ok") and not _vj0["by_image"].get(k, {}).get("vetoed") and BYID[int(k)]["split"] == "train"}
        _itp = osp.join(DS_DIR, "intersection_teacher_image_ids.json")
        if not osp.exists(_itp): raise RuntimeError(f"{_itp} missing — stage teacher incomplete; re-run it")
        _ij = read_json(_itp); assert_fresh(_ij, "intersection_teacher_image_ids.json", ADM_SHA)
        if set(_ij["image_ids"]) != INTER_TEACHER: raise RuntimeError(f"intersection_teacher_image_ids.json ({len(_ij['image_ids'])}) != rebuilt teacher_ok∧¬vetoed∧train ({len(INTER_TEACHER)}) — stage teacher inconsistent")
    # metrics vs teacher outputs: checked BEFORE any staging or Volume write (review 2026-09-22)
    TDM = MET.get("teacher_detect", {"ran": False})
    if bool(TDM.get("ran")) != bool(TEACHER): raise RuntimeError(f"metrics teacher_detect.ran={TDM.get('ran')} but teacher_dets.json.gz present={bool(TEACHER)} — re-run stage=teacher")
    TPL_P = osp.join(DS_DIR, "human_gate_template.json"); TPL = read_json(TPL_P); TPL_SHA = sha256_file(TPL_P)
    if TPL.get("tag") != TAG: raise RuntimeError(f"human_gate_template.json tag {TPL.get('tag')!r} != {TAG!r}")
    # the template and the position prior must come from the SAME gate run as admitted.json (review 2026-09-23)
    _pp = read_json(osp.join(DS_DIR, "position_prior.json"))
    for _nm, _obj in (("human_gate_template.json", TPL), ("position_prior.json", _pp)):
        if _obj.get("gate_run_id") != ADMITTED["gate_run_id"]:
            raise RuntimeError(f"{_nm} gate_run_id {_obj.get('gate_run_id')!r} != admitted.json {ADMITTED['gate_run_id']!r} — written by another gate run; re-run stage=gate")
    # an unlisted source day is reviewed only through the human gate (review 2026-09-23); the module-level guard checked the
    # widget, this checks what was actually gated
    if ADMITTED.get("unlisted_days") and not W["require_human_gate"] and not SMOKE:
        raise RuntimeError(f"admitted.json has unlisted source days {ADMITTED['unlisted_days']} — finalize needs require_human_gate=true")
    SHEET_N_PLANNED = int(TPL.get("sheet_n_planned", 0))
    if SHEET_N_PLANNED < SHEET_N_REF and not SMOKE and W["require_human_gate"]:
        raise RuntimeError(f"human_gate_template.json plans {SHEET_N_PLANNED} counted cells < {SHEET_N_REF} with require_human_gate=true "
                           f"— re-run gate (new tag) with sheet_n >= {SHEET_N_REF}")
    print(f"human gate REQUIRED: {W['require_human_gate']} " + ("(only a PASSED review seals)" if W["require_human_gate"] else
          "(decision 2026-09-23: the fault list is the human gate; a present human_gate.json is still applied and a FAILED one unseals)"))
    n_tpl_counted_all = sum(1 for c in TPL["cells"] if c.get("counted"))
    hg_p = osp.join(DS_DIR, "human_gate.json")
    if osp.exists(hg_p):
        HG = read_json(hg_p); HG_SHA = sha256_file(hg_p)
        if HG.get("tag") != TAG: raise RuntimeError(f"human_gate.json tag {HG.get('tag')!r} != {TAG!r}")
        n_tpl_counted = check_human_gate_vs_template(HG, TPL["cells"])     # the gate cannot pass on cells nobody looked at (review 2026-09-22)
        HG_EVAL = human_gate_eval(HG)
        if HG_EVAL["n_counted"] != n_tpl_counted: raise RuntimeError(f"human_gate.json counted cells {HG_EVAL['n_counted']} != template {n_tpl_counted}")
        print(f"human_gate.json matches the template: {len(TPL['cells'])} cells, {n_tpl_counted} counted (sheet_n {W['sheet_n']})")
    else:
        HG_EVAL = {"status": "pending", "n_counted": 0, "n_bad_counted": 0, "n_wrong_object": 0, "n_bad_by_kind": {}, "reject_rate": 0.0, "bad_cells": []}; HG_SHA = None
        if W["require_human_gate"]: print("⚠ no human_gate.json and require_human_gate=true — manifest will be written UNSEALED (human_gate.status=pending)")
        else: print("no human_gate.json — not required (require_human_gate=false); the seal does not wait for it")
    HG_PRESENT = HG_SHA is not None
    HG_VERDICT = human_gate_verdict(HG_PRESENT, HG_EVAL["status"], HG_EVAL["n_bad_counted"], HG_EVAL["n_wrong_object"], HG_EVAL["n_counted"], SHEET_N_PLANNED)
    if HG_VERDICT == "failed":
        print(f"⚠ HUMAN GATE FAILED (file status {HG_EVAL['status']!r}; BAD counted cells {HG_EVAL['n_bad_counted']}, max {HUMAN_GATE_MAX_BAD}; "
              f"wrong_object {HG_EVAL['n_wrong_object']}, max 0) — "
              "the manifest is UNSEALED whatever require_human_gate says; extend train_label_faults.json (new ids), bump the tag and re-run gate")
    elif HG_PRESENT and HG_EVAL["status"] == "passed" and HG_VERDICT != "passed":
        print(f"⚠ human_gate.json says 'passed' but the review is undersized (counted {HG_EVAL['n_counted']}, planned {SHEET_N_PLANNED} < {SHEET_N_REF}) — recorded as {HG_VERDICT!r}")
    REJ = human_reject_set(HG_EVAL["bad_cells"], FRAMES)
    VET_IDS = {int(k) for k, v in VETO.items() if v["vetoed"]}
    print(f"human gate: file status {HG_EVAL['status']}, verdict {HG_VERDICT}, bad counted {HG_EVAL['n_bad_counted']}, wrong_object {HG_EVAL['n_wrong_object']} -> "
          f"{len(REJ)} frames rejected (±{HUMAN_MARGIN_S:.0f} s); G11 vetoed {len(VET_IDS)}")
    for f in FRAMES:
        f["dropped"] = "human" if f["image_id"] in REJ else ("G11" if f["image_id"] in VET_IDS else None)
    for d in set(f["day"] for f in FRAMES):
        for ph in ("night", "dawn", "dusk", "day"):
            n_in = sum(1 for f in FRAMES if f["day"] == d and f["phase4"] == ph); n_out = sum(1 for f in FRAMES if f["day"] == d and f["phase4"] == ph and f["dropped"] == "G11")
            MET.setdefault("funnel", {}).setdefault(d, {}).setdefault(ph, {})["G11"] = [n_in, n_out]
    KEEP = [f for f in FRAMES if not f["dropped"]]
    TRAIN = [f for f in KEEP if f["split"] == "train"]; VALID = [f for f in KEEP if f["split"] == "valid"]
    BELOW_FLOOR = len(TRAIN) < W["min_train"]
    if BELOW_FLOOR: print(below_floor_message(len(TRAIN), W["min_train"], "after human/G11 drops", W["stride_day_s"], W["stride_other_s"]))
    _nvm = night_valid_message(VALID, "after human/G11 drops")
    if _nvm: print(_nvm)
    CW, CH = ADMITTED["crop_wh"]
    ACH_FINAL, SHORT_FINAL = quota_achieved(TRAIN, QUOTAS)                     # shares of the FINAL train set (review 2026-09-22)
    for k, v in SHORT_FINAL.items(): print(f"⚠ QUOTA UNMET {k} (final train set): have {v['have']}, need {v['need']}")
    HOUR_SHARE_FINAL = max_hour_share(TRAIN)                                    # gate enforced the cap; human/G11 drops can raise a bin's share
    if HOUR_SHARE_FINAL > W["hour_cap_frac"] + 1e-12: print(f"⚠ HOUR CAP exceeded after human/G11 drops: max (day, hour) share {HOUR_SHARE_FINAL:.4f} > {W['hour_cap_frac']} (reported, not re-trimmed)")
    # source-day guard is derived from admitted.json AND the frames actually written, never from the widget
    ADM_DAYS = list(ADMITTED["source_days"]); KEEP_DAYS = sorted({f["day"] for f in KEEP})
    if not set(KEEP_DAYS) <= set(ADM_DAYS): raise RuntimeError(f"frames from {sorted(set(KEEP_DAYS) - set(ADM_DAYS))} are admitted but not in admitted.json source_days {ADM_DAYS}")
    FORBIDDEN_DAYS = manifest_guard_days(sorted(set(ADM_DAYS) | set(KEEP_DAYS)))

    # figures of a previous finalize run (vetoed sheets with frames no longer dropped, old overlays) would contradict this
    # run's; they are re-rendered below (review 2026-09-23)
    for _p in glob.glob(osp.join(FIG_DIR, "vetoed_*.png")): os.remove(_p)
    if osp.isdir(osp.join(FIG_DIR, "_verify")): shutil.rmtree(osp.join(FIG_DIR, "_verify"))
    # ---------------- JPEG extraction (verbatim bytes), crops, COCO assembly ----------------------------
    DSL = osp.join(LOCAL, "ds"); [os.makedirs(osp.join(DSL, s), exist_ok=True) for s in ("train", "valid")]
    # sheet_frames/ is built locally and REPLACES the Volume directory as a whole at publish time, so it holds exactly this
    # run's non-dropped counted frames (listed with sha256 in the manifest) — review 2026-09-22
    SFL = osp.join(LOCAL, "sheet_frames"); os.makedirs(SFL, exist_ok=True); SHEET_FRAMES = {}
    TARS = Tars(); by_clip = defaultdict(list)
    for f in FRAMES:
        if not f["dropped"] or f["image_id"] in VET_IDS: by_clip[(f["root"], f["day"], f["window"], f["clip_stem"])].append(f)   # vetoed frames: sheet only, rendered while the tar is local
    rngv = random.Random(SEED + 5)
    VER_IDS = set(rngv.sample([f["image_id"] for f in KEEP], min(VERIFY_POOL_N, len(KEEP)))); STORED_RLE = {}   # verification sample: RLEs kept in memory
    coco = {k: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "calf", "supercategory": "none"}]} for k in ("tracked/train", "tracked/valid", "teacher_detect/train")}
    ann_id = {k: 0 for k in coco}; n_teacher_crop_drop = 0; n_teacher_not_ok = 0; t0 = time.time(); t_dec = 0.0; n_img = 0
    vbatch, vk = [], 0
    def add_ann(key, image_id, rle, h, w, slot, seg):
        ann_id[key] += 1; coco[key]["annotations"].append(coco_annotation(ann_id[key], image_id, rle, h, w, slot, seg, maskUtils))
    CLIP_SHA = {c["clip_stem"]: c.get("masks_sha256") for c in ADMITTED["clips"]}
    for (root, day, win, stem), fs in sorted(by_clip.items()):
        d = read_masks_checked(osp.join(TOL, root, win, f"masks_{stem}.json.gz"), CLIP_SHA.get(stem))   # byte-identical to the gated file (review 2026-09-23)
        want = {f["frame_idx"] for f in fs}
        rle_by_idx = {int(r["frame_idx"]): r["masks"] for r in d["frames"] if int(r["frame_idx"]) in want}
        del d; TARS.stage(day, stem)
        for f in sorted(fs, key=lambda z: z["frame_idx"]):
            jb = TARS.read(day, stem, f["frame_idx"])
            if f["image_id"] in VET_IDS:            # vetoed sheet cell (red border) — no second staging pass
                v = VETO[str(f["image_id"])]
                vbatch.append(render_cell(jb, [rle_by_idx[f["frame_idx"]][str(s)] for s in range(N_CALVES)], day, datetime.fromisoformat(f["wall"]), stem, f["frame_idx"],
                                          f"VETO det {v['det_area']} px ios {v['ios']}", f["phase4"], f"vetoed{vk:03d}c{len(vbatch):02d}", border=(0, 0, 255)))
                if len(vbatch) == 12: write_sheet(vbatch, f"vetoed_{vk:03d}.png"); vk += 1; vbatch = []
            if f["dropped"]: continue
            with open(osp.join(DSL, f["split"], f["file_name"]), "wb") as fh: fh.write(jb)     # verbatim bytes, never re-encoded
            if f.get("counted_cell"):
                with open(osp.join(SFL, f["file_name"]), "wb") as fh: fh.write(jb)
                SHEET_FRAMES[f["file_name"]] = hashlib.sha256(jb).hexdigest()
            im_rec = {"id": f["image_id"], "file_name": f["file_name"], "width": W_FRAME, "height": H_FRAME, "wall": f["wall"], "phase4": f["phase4"], "day": day, "root": root,
                      "clip_stem": stem, "frame_idx": f["frame_idx"], "active": f["active"], "crop_variant": False}
            key = f"tracked/{f['split']}"; coco[key]["images"].append(dict(im_rec))
            stored = {s: rle_by_idx[f["frame_idx"]][str(s)] for s in range(N_CALVES)}
            if f["image_id"] in VER_IDS: STORED_RLE[f["image_id"]] = stored
            for s in range(N_CALVES): add_ann(key, f["image_id"], stored[s], H_FRAME, W_FRAME, s, f["seg"][s])
            trec = TEACHER.get(str(f["image_id"]))
            t_ok = bool(f["split"] == "train" and trec and trec.get("teacher_ok"))   # teacher_ok already includes the crop-window rule (stage teacher)
            if f["split"] == "train" and trec and not t_ok:
                n_teacher_not_ok += 1
                if trec.get("teacher_fail") == "crop_window": n_teacher_crop_drop += 1
            if t_ok:
                if f["image_id"] not in INTER_TEACHER: raise RuntimeError(f"image {f['image_id']}: teacher_ok, not dropped, but absent from the stage-teacher intersection (stage teacher inconsistent)")
                coco["teacher_detect/train"]["images"].append(dict(im_rec))
                for s in range(N_CALVES): add_ann("teacher_detect/train", f["image_id"], trec["teacher_masks"][s], H_FRAME, W_FRAME, s, f["seg"][s])
            if f["split"] == "train" and f.get("crop"):
                cr = f["crop"]; x, y = cr["x"], cr["y"]; cid = f["image_id"] + CROP_ID_OFFSET
                cname = f"{stem}_f{f['frame_idx']:07d}_crop{x}_{y}.jpg"
                tt = time.time(); pil = Image.open(io.BytesIO(jb)).convert("RGB"); pil.crop((x, y, x + CW, y + CH)).save(osp.join(DSL, "train", cname), quality=90); t_dec += time.time() - tt
                crec = {"id": cid, "file_name": cname, "width": CW, "height": CH, "wall": f["wall"], "phase4": f["phase4"], "day": day, "root": root, "clip_stem": stem,
                        "frame_idx": f["frame_idx"], "active": f["active"], "crop_variant": True, "parent_file_name": f["file_name"], "crop_xywh": [x, y, CW, CH], "n_visible": len(cr["kept"])}
                coco["tracked/train"]["images"].append(crec)
                for s in cr["kept"]: add_ann("tracked/train", cid, crop_mask_rle(rle_decode(stored[s]), x, y, CW, CH), CH, CW, s, f["seg"][s])
                if t_ok:      # same window on the teacher masks; the kept set was decided in stage teacher, so both label sources emit this crop
                    tms = [rle_decode(m) for m in trec["teacher_masks"]]; keep_t = list(trec["teacher_crop_kept"])
                    coco["teacher_detect/train"]["images"].append({**crec, "n_visible": len(keep_t)})
                    for s in keep_t: add_ann("teacher_detect/train", cid, crop_mask_rle(tms[s], x, y, CW, CH), CH, CW, s, f["seg"][s])
            n_img += 1
            if n_img % 500 == 0: print(f"  {n_img}/{len(KEEP)} frames written ({(time.time()-t0)/60:.1f} min)", flush=True)
        TARS.drop(stem)
    if vbatch: write_sheet(vbatch, f"vetoed_{vk:03d}.png")
    if VET_IDS: print(f"vetoed sheets: {len(VET_IDS)} frames on {vk + (1 if vbatch else 0)} sheets")
    TIMING["finalize_extract_s"] = time.time() - t0; TIMING["crop_decode_s"] = t_dec
    # the intersection after human/G11 drops = exactly the teacher_detect/train full frames; written (at publish) to
    # intersection_image_ids.json so Tol_43's `n_full == manifest.intersection_n` check holds and DET-R(intersection) ==
    # DET-C image set. The stage-teacher list lives in intersection_teacher_image_ids.json and is never rewritten.
    INTER_FINAL = sorted(im["id"] for im in coco["teacher_detect/train"]["images"] if not im.get("crop_variant")) if TEACHER else []
    if TEACHER and set(INTER_FINAL) != (INTER_TEACHER & {f["image_id"] for f in TRAIN}): raise RuntimeError("intersection after drops != teacher_detect/train full frames")
    for key in list(coco):
        if key == "teacher_detect/train" and not TEACHER: del coco[key]; continue
        coco_check(coco[key]); print(f"  {key}: {len(coco[key]['images'])} images, {len(coco[key]['annotations'])} annotations")
    # annotation JSONs are written and verified on /local_disk0 and copied to the Volume only at publish time, after the
    # manifest is assembled — a failure before that leaves the previous Volume state intact (review 2026-09-22)
    ANN_PATHS, ANN_SHA = {}, {}
    for key, cc in coco.items():
        p = osp.join(LOCAL, "annotations", key, "_annotations.coco.json"); os.makedirs(osp.dirname(p), exist_ok=True)
        with open(p, "w") as fh: json.dump(cc, fh)
        ANN_PATHS[key] = p; ANN_SHA[key] = sha256_file(p)

    # ---------------- verify(): read the WRITTEN JSON back with pycocotools ----------------------------
    from pycocotools.coco import COCO
    os.makedirs(osp.join(FIG_DIR, "_verify"), exist_ok=True); n_overlay = 0
    for key, p in ANN_PATHS.items():
        cc = COCO(p); split = key.split("/")[1]
        if len(cc.getImgIds()) != len(coco[key]["images"]): raise RuntimeError(f"{key}: pycocotools sees {len(cc.getImgIds())} images, wrote {len(coco[key]['images'])}")
        for im in cc.loadImgs(cc.getImgIds()):
            fp = osp.join(DSL, split, im["file_name"])
            if not osp.exists(fp): raise RuntimeError(f"{key}: {fp} missing")
        pool_v = [im for im in cc.loadImgs(cc.getImgIds()) if (im["id"] - CROP_ID_OFFSET if im.get("crop_variant") else im["id"]) in VER_IDS]
        for im in rngv.sample(pool_v, min(VERIFY_N, len(pool_v))):
            fp = osp.join(DSL, split, im["file_name"]); pil = Image.open(fp)
            if pil.size != (im["width"], im["height"]): raise RuntimeError(f"{fp}: PIL size {pil.size} != {(im['width'], im['height'])}")
            anns = cc.loadAnns(cc.getAnnIds(imgIds=[im["id"]]))
            src = BYID[im["id"] - CROP_ID_OFFSET if im.get("crop_variant") else im["id"]]
            for a in anns:
                dec = maskUtils.decode({"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"].encode("ascii")}).astype(bool)
                if key.startswith("tracked"):
                    ref = rle_decode(STORED_RLE[src["image_id"]][a["slot"]])
                else:
                    ref = rle_decode(TEACHER[str(src["image_id"])]["teacher_masks"][a["slot"]])
                if im.get("crop_variant"):
                    x, y, w_, h_ = im["crop_xywh"]; ref = ref[y:y + h_, x:x + w_]
                if not np.array_equal(dec, ref): raise RuntimeError(f"{key} image {im['id']} slot {a['slot']}: decoded mask != stored mask")
            if n_overlay < OVERLAY_N:
                bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
                for a in anns:
                    m = maskUtils.decode({"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"].encode("ascii")}).astype(bool)
                    col = COLORS[a["slot"] % 4]; bgr[m] = (0.55 * bgr[m] + 0.45 * np.array(col, np.float32)).astype(np.uint8)
                small = cv2.resize(bgr, (960, int(960 * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
                loc = osp.join(LOCAL, "v.jpg"); cv2.imwrite(loc, small); shutil.copy(loc, osp.join(FIG_DIR, "_verify", f"{key.replace('/', '_')}_{im['id']}.jpg")); n_overlay += 1
        print(f"  verify {key}: OK ({min(VERIFY_N, len(pool_v))} images decoded exactly, {len(cc.getImgIds())} files present)")

    # ---------------- packaging: images.tar (local, then dbutils.fs.cp) --------------------------------
    t0 = time.time(); tar_loc = osp.join(LOCAL, "images.tar")
    with tarfile.open(tar_loc, "w") as tf:
        for split in ("train", "valid"):
            for name in sorted(os.listdir(osp.join(DSL, split))): tf.add(osp.join(DSL, split, name), arcname=f"{split}/{name}")
    TAR_SHA = sha256_file(tar_loc); tar_mb = osp.getsize(tar_loc) / 1e6
    TIMING["package_s"] = time.time() - t0; print(f"images.tar {tar_mb:.0f} MB built locally in {TIMING['package_s']:.0f} s (copied to the Volume at publish)")

    TARS.close_all()   # vetoed sheets were rendered inside the per-clip loop (no second staging pass)

    # ---------------- manifest ------------------------------------------------------------------
    nb_sha = None
    for cand in ([__file__] if "__file__" in globals() else []) + ["/Workspace/Users/<DATABRICKS_USER>/Model Distillation/Testing on Tol2/Tol_42_export_seg_coco.py",
                                                                       "/Workspace/Users/<DATABRICKS_USER>/Model Distillation/Testing on Tol2/Tol_42_export_seg_coco"]:
        try:
            if osp.exists(cand): nb_sha = sha256_file(cand); break
        except Exception: pass
    if nb_sha is None: print("⚠ notebook_sha256 unavailable (notebook source not readable as a file)")
    counts_by_ph = {ph: {"train": sum(1 for f in TRAIN if f["phase4"] == ph), "valid": sum(1 for f in VALID if f["phase4"] == ph)} for ph in ("night", "dawn", "dusk", "day")}
    n_crops = sum(1 for f in TRAIN if f.get("crop"))
    lsa = ["tracked"] + (["teacher_detect", "tracked_intersection"] if TDM.get("ran") else [])
    sealed = seal_rule(HG_VERDICT, W["require_human_gate"], BELOW_FLOOR, VETO_RAN, W["require_veto"], SMOKE)
    TRAIN_SRC = [d for d in ADM_DAYS if d in TRAIN_DAYS]
    def const(name, value, prov, derived_from=None):
        df = derived_from if derived_from is not None else TRAIN_SRC
        return {"value": value, "provenance": prov, "derived_from": df, "not_from": not_from_days(df)}   # per constant: pooled ones list 04-22 even when it is a source
    consts = {f"A_ref[{d}]": const(f"A_ref[{d}]", v, "DERIVED(day): median largest daylight mask at 1 Hz (Tol_40:141-142)", [d]) for d, v in C["A_ref"].items()}
    for p in ("dayl", "dark"): consts[f"frag_lo[{p}]"] = const("", C["frag_lo"][p], f"{C['frag_lo_provenance']}: P1 of user-admitted ratios, pool {p} (SPEC §2.7)")
    consts["frag_lo_provenance"] = const("", C["frag_lo_provenance"], "SPEC §2.7")
    consts["over_hi"] = const("", C["over_hi"], f"READ band_json.band.over_hi ({osp.basename(ADMITTED['band_json'])}, Tol_41)", BAND.get("derived_from"))
    for p in ("dayl", "dark"):
        consts[f"S_lo[{p}]"] = const("", C["S_lo"][p], "DERIVED floor_2dp(P1(S|U,pool))"); consts[f"S_hi[{p}]"] = const("", C["S_hi"][p], "DERIVED ceil_2dp(P99(S|U,pool))")
    consts["STAB_LO"] = const("", C["STAB_LO"], "DERIVED P0.5 of area/median(±10 s) over U"); consts["STAB_HI"] = const("", C["STAB_HI"], "DERIVED P99.5 of area/median(±10 s) over U")
    consts["body_len_px"] = const("", C.get("body_len_px"), f"DERIVED median 4*sqrt(lambda_max) over EVERY decoded lattice-candidate mask of the training days (n={C.get('body_len_detail', {}).get('n_masks')}, per-phase4 medians in band_derivation.body_len_detail)",
                                  C.get("body_len_detail", {}).get("derived_from", TRAIN_SRC))
    for d, v in C["ACTIVE_THR"].items(): consts[f"ACTIVE_THR[{d}]"] = const("", v, "DERIVED(day) P70 of the activity score over U", [d])
    for name, val, prov in (("DISJ_IOU", DISJ_IOU, "DATA_SPLITS §6.5"), ("DISJ_IOS", DISJ_IOS, "DATA_SPLITS §6.5"),
                            ("CC_FRAC", CC_FRAC, "SPEC §2.6 G10"), ("BLOCK_LARGEST_LO", BLOCK_LARGEST_LO, "DATA_SPLITS §6.4"), ("EVENT_BUF_S", EVENT_BUF_S, "half DETECT_DAY_SEC Tol_03:86"),
                            ("CHAOS", f"{CHAOS_EVENTS}/{CHAOS_MIN}min", "Tol_30:63-64"), ("VETO_AREA_FRAC", VETO_AREA_FRAC, "DATA_SPLITS §6.6 as ratio"), ("VETO_IOS", VETO_IOS, "DATA_SPLITS §6.6"),
                            ("strides", {"day": W["stride_day_s"], "other": W["stride_other_s"], "valid_day": W["valid_stride_day_s"], "valid_other": W["valid_stride_other_s"]}, "user protocol 2026-09-22"),
                            ("quotas", {"night": W["night_quota"], "dusk": W["dusk_quota"], "dawn": W["dawn_quota"], "active": W["active_quota"]}, "SPEC §2.2"),
                            ("hour_cap", W["hour_cap_frac"], "SPEC §2.2"), ("crop_wh", [CW, CH], "SPEC §2.10 16:9"), ("crop_frac", W["crop_frac"], "SPEC §2.10"),
                            ("crop_keep_frac", CROP_KEEP_FRAC, "SPEC §2.10"), ("crop_drop_frac", CROP_DROP_FRAC, "SPEC §2.10"), ("valid_buffer_min", W["valid_buffer_min"], "SPEC §2.6 G12"),
                            ("min_train", W["min_train"], "user's floor"), ("n_max", W["n_max"], "SPEC §2.2"), ("valid_max", W["valid_max"], "SPEC §2.2"),
                            ("TEACHER_MATCH_MIN", TEACHER_MATCH_MIN, "SPEC §2.11 Hungarian max(IoU,IoS) floor for teacher_detect labels"),
                            ("TEACHER_DROP_HIGH", TEACHER_DROP_HIGH, "SPEC §2.11 descriptive drop_high flag (F9 requires tracked_intersection at any drop)"),
                            ("LOW_HOUR_FRAC", LOW_HOUR_FRAC, "SPEC §2.8 step 6 low-hour flag"), ("HUMAN_MARGIN_S", HUMAN_MARGIN_S, "SPEC §2.13"),
                            ("sheet_n", W["sheet_n"], "SPEC §2.12 (>= 300 outside SMOKE only when require_human_gate=true, DATA_SPLITS §6 item 8; addendum 2026-09-23 §4)"),
                            ("HUMAN_GATE_MAX_BAD", HUMAN_GATE_MAX_BAD, "SPEC §2.12 seal rule"), ("require_human_gate", W["require_human_gate"], "SPEC addendum 2026-09-23 §4 (default false)"),
                            ("require_veto", W["require_veto"], "SPEC §2.2 (default true)")):
        consts[name] = const(name, val, prov, [])
    # G5's edge: a protocol literal whose SUPPORT was measured on the training days only (review 2026-09-23)
    consts["LINK_IOU_MIN"] = const("LINK_IOU_MIN", LINK_IOU_MIN, f"SPEC §2.6 G5 protocol literal; supported by the training-day link-IoU distribution "
                                   f"(band_derivation.link_iou: {C['link_iou']['check']} over G1-G4 survivors, asserted at gate); Tol_12's test-day relink figures are not used",
                                   TRAIN_SRC)
    # sources / source_days / forbidden_eval_days come from admitted.json (what was GATED) plus the days actually
    # written — never from this stage's widget (review 2026-09-22; assert_admitted_matches already proved they agree)
    ADM_PAIRS = [tuple(p) for p in ADMITTED["sources"]]
    # gate outputs hashed for provenance only: the current Tol_44 evaluation superseded the earlier one, so no consumer reads position_prior.json / plates/ any more
    # (kept because stage gate always writes them and finalize's gate_run_id check covers them; review 2026-09-22/23)
    PP_SHA = sha256_file(osp.join(DS_DIR, "position_prior.json"))
    PLATES_SHA = {n: sha256_file(osp.join(DS_DIR, "plates", n)) for n in sorted(os.listdir(osp.join(DS_DIR, "plates")))}
    INTER_SHA, INTER_LOC = None, osp.join(LOCAL, "intersection_image_ids.json")
    if TEACHER:
        with open(INTER_LOC, "w") as f:
            json.dump({"tag": TAG, "admitted_sha256": ADM_SHA, "human_gate_sha256": HG_SHA, "basis": "teacher_detect/train full frames after human/G11 drops (== the frames both label sources emit; crops follow their parent)", "image_ids": INTER_FINAL}, f)
        INTER_SHA = sha256_file(INTER_LOC)
    FIN_RUNS = []
    if PREV_MAN:   # audit trail of superseded finalize runs (finalize is re-runnable)
        FIN_RUNS = list(PREV_MAN.get("finalize_runs", [])) + [{"created": PREV_MAN.get("created"), "sealed": PREV_MAN.get("sealed"), "manifest_sha256": sha256_file(_mp),
                                                               "human_gate_sha256": (PREV_MAN.get("human_gate") or {}).get("sha256"), "superseded": True}]
    MANIFEST = {"tag": TAG, "created": datetime.now().isoformat(), "notebook": "Tol_42_export_seg_coco", "notebook_sha256": nb_sha, "seed": SEED, "smoke": SMOKE,
                "admitted_sha256": ADM_SHA, "human_gate_template_sha256": TPL_SHA, "finalize_runs": FIN_RUNS,
                "previous_manifest_sha256": FIN_RUNS[-1]["manifest_sha256"] if FIN_RUNS else None,
                "sources": [{"root": r, "day": d, "windows": sorted({c["window"] for c in ADMITTED["clips"] if c["root"] == r and c["day"] == d}),
                             "clips": [c["clip_stem"] for c in ADMITTED["clips"] if c["root"] == r and c["day"] == d]} for r, d in ADM_PAIRS],
                "sources_str": ADMITTED["sources_str"], "source_days": manifest_guard_days(ADM_DAYS), "days_written": KEEP_DAYS, "forbidden_eval_days": FORBIDDEN_DAYS,
                "allow_test_day_reason": W["allow_test_day_reason"], "unlisted_day_policy": W["unlisted_day_policy"],
                "faults_json": {"path": ADMITTED["faults_json"], "sha256": ADMITTED["faults_sha256"], "n_faults": len(FAULTS["faults"]), "exclusion_margin_s": FAULTS["exclusion_margin_s"],
                                "excluded_s_by_id": MET.get("lost_rows_by_fault_id", {}), "copies": FAULTS_COPIES,
                                "note_exclusions": ADMITTED.get("note_exclusions")},    # notebook-side, from the notes' "unlisted but visible" walls (review 2026-09-23)
                "band_json": {"path": ADMITTED["band_json"], "sha256": ADMITTED["band_sha256"], "derived_from": BAND.get("derived_from"), "over_hi": OVER_HI, "ref_pass1_largest_daylight": {d: BAND["per_day"][d]["ref_pass1_largest_daylight"] for d in BAND.get("per_day", {})}},
                "phase_function": "phase4 (night h>=21|h<6; dawn 6-7; dusk 20; day)", "pools": {"dayl": ["day", "dawn"], "dark": ["dusk", "night"]}, "constants": consts,
                "gates": {"funnel": MET.get("funnel"), "curled_or_clipped_blocks": ADMITTED["blocks"]["curled_or_clipped_blocks"], "blocks_failing_per_slot_0.75": ADMITTED["blocks"]["blocks_failing_per_slot_0.75"],
                          "collapse_overrides": [b for b in W["collapse_override_blocks"].split(",") if b.strip()], "curl_band_ack": W["curl_band_ack"] or None, "furniture_flagged": len(ADMITTED.get("furniture_flagged", [])),
                          "upper_edge_clips_admitted_mass": C["upper_diag"]["upper_edge_clips_admitted_mass"], "bad_overlap_high": C["bad_overlap"]["high"],
                          "curled_check": C["curled_check"], "link_iou": C["link_iou"]},
                "split": {"valid_clips": ADMITTED["split"]["valid_clips"], "train_clips": ADMITTED["split"]["train_clips"], "valid_max": W["valid_max"]},
                "counts": {"train_full": len(TRAIN), "train_crops": n_crops, "valid": len(VALID), "by_phase4": counts_by_ph, "distinct_10min_blocks": len({(f["day"], int(datetime.fromisoformat(f["wall"]).timestamp() // 600)) for f in TRAIN}), "below_floor": BELOW_FLOOR,
                           "below_floor_next_step": (None if not BELOW_FLOOR else
                                                     ("add Tracking_teacher:20240422 (option b; densify already applied)" if (float(W["stride_day_s"]), float(W["stride_other_s"])) == DENSE_STRIDES
                                                      else f"densify: new tag with stride_day_s={DENSE_STRIDES[0]:g}, stride_other_s={DENSE_STRIDES[1]:g} (option a, the user's first choice, SPEC addendum 2026-09-23 §1)")),
                           "hour_share_max": round(HOUR_SHARE_FINAL, 4), "hour_share_max_at_gate": ADMITTED.get("hour_share_max"), "hour_cap_exceeded_after_drops": HOUR_SHARE_FINAL > W["hour_cap_frac"] + 1e-12},
                "gate_widgets": ADMITTED["gate_widgets"],
                "quotas": {"targets": {"night": W["night_quota"], "dusk": W["dusk_quota"], "dawn": W["dawn_quota"], "active": W["active_quota"]},
                           "achieved_before_cap": MET.get("sampling", {}).get("alloc", {}).get("achieved"), "achieved_after_cap": MET.get("sampling", {}).get("alloc", {}).get("achieved_after_cap"),
                           "achieved": ACH_FINAL, "shortfall": SHORT_FINAL, "achieved_note": "achieved/shortfall describe the FINAL train set (after hour cap, human and G11 drops)"},
                "sam3_veto": {"ran": VETO_RAN, "n_checked": len(VETO), "n_vetoed": len(VET_IDS), "image_ids": sorted(VET_IDS)},
                "teacher_detect": {"role": "free by-product of the G11 veto pass; NOT required by anything downstream (SPEC addendum 2026-09-23 §3: no DET-C training planned)",
                                   "ran": bool(TDM.get("ran")), "n_frames": TDM.get("n_frames"), "n_dropped": TDM.get("n_dropped"), "drop_frac": TDM.get("drop_frac"), "drop_high": TDM.get("drop_high"),
                                   "drop_high_threshold": TEACHER_DROP_HIGH, "drop_high_note": "descriptive only (SPEC §2.11); DATA_SPLITS F9 overrides it",
                                   "f9_comparator_label_source": "tracked_intersection" if TDM.get("ran") else None,
                                   "f9_note": "DATA_SPLITS F9 (binding): the mechanism control DET-C must be compared with DET-R trained on label_source=tracked_intersection "
                                              "(the same admitted frames) at ANY drop rate; a DET-R(tracked) vs DET-C pairing is not an F9 comparison",
                                   "intersection_n": len(INTER_FINAL) if TEACHER else None, "intersection_n_before_drops": TDM.get("intersection_n"),
                                   "frame_set_difference_vs_tracked": {"train_parents_without_teacher_labels": n_teacher_not_ok, "of_which_crop_window_fail": n_teacher_crop_drop,
                                                                        "note": "a parent dropped from teacher_detect takes its crop with it; DET-R(tracked_intersection) and DET-C share the image set exactly"},
                                   "valid_source": "tracked valid annotations (same checkpoint-selection criterion for both arms)"},
                "require_veto": W["require_veto"], "require_human_gate": W["require_human_gate"],
                "label_sources_available": lsa, "human_gate": {"required": W["require_human_gate"], "present": HG_PRESENT, "verdict": HG_VERDICT,
                                                               "required_note": "SPEC addendum 2026-09-23 §4: with required=false the user's fault list is the human gate. "
                                                                                "status = the gate outcome RECOMPUTED from the cells (absent/pending -> 'pending', 'passed', "
                                                                                "'failed'); status_in_file = the user's own field (null when absent)",
                                                               "status": effective_hg_status(HG_VERDICT), "status_in_file": HG_EVAL["status"] if HG_PRESENT else None, "sha256": HG_SHA, "template_sha256": TPL_SHA, "n_counted": HG_EVAL["n_counted"], "n_template_cells": len(TPL["cells"]),
                                                               "n_bad_counted": HG_EVAL["n_bad_counted"], "n_wrong_object": HG_EVAL["n_wrong_object"],
                                                               # Tol_43.manifest_sheet_n prefers sheet_n_planned (the counted cells actually planned, incl. the
                                                               # bad_overlap_high tail enlargement) and reads sheet_n as the WIDGET value (review 2026-09-23)
                                                               "sheet_n_planned": SHEET_N_PLANNED, "sheet_n": TPL.get("sheet_n"), "sheet_n_widget": TPL.get("sheet_n"), "sheet_n_required_min": SHEET_N_REF,
                                                               "sheet_shortfall": max(SHEET_N_PLANNED - n_tpl_counted_all, 0),
                                                               "n_bad_by_kind": HG_EVAL["n_bad_by_kind"], "reject_rate": HG_EVAL["reject_rate"], "n_frames_rejected": len(REJ),
                                                               "unlisted_days": ADMITTED["unlisted_days"], "review_basis_by_day": ADMITTED["review_basis_by_day"], "counted_by_day": TPL.get("counted_by_day")},
                "sheet_frames": {"n": len(SHEET_FRAMES), "sha256_by_file": SHEET_FRAMES,
                                 "note": "exactly the non-dropped counted sheet frames of THIS finalize run; the Volume directory is replaced as a whole"},
                "crop_augmentation_note": "this 16:9 crop variant is the only crop augmentation (Tol_43 runs scale_jitter=False)",
                "gate_run_id": ADMITTED["gate_run_id"], "tracking_files": {"masks_sha256_by_clip": {c["clip_stem"]: c["masks_sha256"] for c in ADMITTED["clips"]}, "window_files": ADMITTED["window_files"]},
                "sealed": sealed, "sha256": {"images_tar": TAR_SHA, "annotations": ANN_SHA, "position_prior": PP_SHA, "plates": PLATES_SHA,
                                             "intersection_image_ids": INTER_SHA}}
    eval_contract_check(MANIFEST)       # Tol_44 reads these keys; a manifest without them must never be published
    # ---------------- publish: every Volume write of finalize happens here, after all checks passed -------------------
    t0 = time.time()
    dbutils.fs.cp("file:" + tar_loc, "dbfs:" + osp.join(DS_DIR, "images.tar"))
    if osp.getsize(osp.join(DS_DIR, "images.tar")) != osp.getsize(tar_loc): raise RuntimeError("images.tar on the Volume has another size than the local build")   # size only: re-hashing GBs over FUSE is slow
    for key, lp in ANN_PATHS.items():
        vp = osp.join(DS_DIR, "annotations", key, "_annotations.coco.json"); os.makedirs(osp.dirname(vp), exist_ok=True); shutil.copy(lp, vp)
        if sha256_file(vp) != ANN_SHA[key]: raise RuntimeError(f"{vp} on the Volume != local build")
    if not TEACHER and osp.isdir(osp.join(DS_DIR, "annotations", "teacher_detect")): shutil.rmtree(osp.join(DS_DIR, "annotations", "teacher_detect"))
    _sfv = osp.join(DS_DIR, "sheet_frames")
    if osp.isdir(_sfv): shutil.rmtree(_sfv)
    os.makedirs(_sfv, exist_ok=True)
    for name in sorted(SHEET_FRAMES): shutil.copy(osp.join(SFL, name), osp.join(_sfv, name))
    if TEACHER: shutil.copy(INTER_LOC, osp.join(DS_DIR, "intersection_image_ids.json"))
    elif osp.exists(osp.join(DS_DIR, "intersection_image_ids.json")): os.remove(osp.join(DS_DIR, "intersection_image_ids.json"))
    TIMING["publish_s"] = time.time() - t0; print(f"published images.tar ({tar_mb:.0f} MB), annotations, {len(SHEET_FRAMES)} sheet_frames in {TIMING['publish_s']:.0f} s")
    with open(osp.join(DS_DIR, "manifest.json"), "w") as f: json.dump(MANIFEST, f, indent=1)
    MET.update({"finalize": {"human_gate": {**{k: v for k, v in HG_EVAL.items() if k != "bad_cells"}, "required": W["require_human_gate"], "present": HG_PRESENT, "verdict": HG_VERDICT},
                             "require_veto": W["require_veto"], "n_rejected_human": len(REJ), "n_vetoed": len(VET_IDS), "sealed": sealed, "below_floor": BELOW_FLOOR,
                             "counts": MANIFEST["counts"], "quotas_final": MANIFEST["quotas"], "teacher_detect": MANIFEST["teacher_detect"], "admitted_sha256": ADM_SHA}})
    MET.setdefault("timing", {}).update({k: round(v, 1) for k, v in TIMING.items()}); MET["timing"]["finalize_total_s"] = round(time.time() - t_fin0, 1); MET["timing"]["finalize_staged_mb"] = round(STAGED_MB["stage_tars"])
    save_metrics(MET); shutil.rmtree(DSL, ignore_errors=True); shutil.rmtree(SFL, ignore_errors=True); os.remove(tar_loc); mark_done("finalize")
    print(f"\nmanifest written: sealed={sealed} (human gate required {W['require_human_gate']}, verdict {HG_VERDICT}, bad {HG_EVAL['n_bad_counted']}, "
          f"wrong_object {HG_EVAL['n_wrong_object']}, below_floor {BELOW_FLOOR}, veto ran {VETO_RAN} (required {W['require_veto']}), smoke {SMOKE})")
    print(f"train {len(TRAIN)} full + {n_crops} crops, valid {len(VALID)}; label sources {lsa}; {(time.time()-t_fin0)/60:.1f} min")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SELFTEST — pure python on synthetic data (runs unconditionally, locally and on the cluster; raises on any failure)

# COMMAND ----------

def _selftest():
    rng = np.random.default_rng(42); H, Wd = 48, 64
    def blob(cx, cy, rx, ry):
        yy, xx = np.mgrid[0:H, 0:Wd]; return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    # 1. rle round trip (incl. first pixel foreground)
    for _ in range(50):
        m = rng.random((H, Wd)) < 0.3
        if _ % 2: m[0, 0] = True
        r = rle_encode(m); assert r["counts"][0] == 0 if m[0, 0] else r["counts"][0] > 0
        assert np.array_equal(rle_decode(r), m)
    # 2. runs IoU/IoS vs dense on 200 pairs
    for _ in range(200):
        a = blob(rng.uniform(10, 54), rng.uniform(10, 38), rng.uniform(3, 12), rng.uniform(3, 12)); b = blob(rng.uniform(10, 54), rng.uniform(10, 38), rng.uniform(3, 12), rng.uniform(3, 12))
        A, B = runs_of(rle_encode(a)), runs_of(rle_encode(b))
        assert abs(runs_iou(A, B) - m_iou(a, b)) < 1e-9 and abs(runs_ios(A, B) - m_ios(a, b)) < 1e-9 and A[2] == a.sum()
        assert rle_area(rle_encode(a)) == a.sum()
    # 3. centroid from runs vs numpy, incl. a mask spanning full columns (runs crossing column boundaries)
    m2 = np.zeros((H, Wd), bool); m2[:, 10:20] = True                  # full columns 10-19: every run crosses a column boundary
    for m in (blob(30, 20, 9, 6), m2):
        c = centroid_from_runs(runs_of(rle_encode(m)), H); ys, xs = np.nonzero(m)
        assert abs(c[0] - xs.mean()) < 1e-6 and abs(c[1] - ys.mean()) < 1e-6, (c, xs.mean(), ys.mean())
    # 4. fault intervals: 3 intervals, 40 timestamps
    faults = {"faults": [{"day": "20240418", "id": f"F{i}", "exclude_start": f"2024-04-18T1{i}:00:00", "exclude_end": f"2024-04-18T1{i}:30:00", "kind": "partial" if i == 0 else "missing",
                          "wall_start": f"2024-04-18T1{i}:05:00", "wall_end": f"2024-04-18T1{i}:25:00", "slots": [2]} for i in range(3)]}
    iv = fault_intervals(faults, "20240418"); assert len(iv) == 3
    t0 = datetime(2024, 4, 18, 9, 45); hits = [fault_hit(t0 + timedelta(minutes=10 * k), iv) for k in range(40)]
    assert hits[2] == "F0" and hits[0] is None and hits[3] == "F0" and hits[5] is None and hits[8] == "F1" and sum(h is not None for h in hits) == 9, hits
    assert len(known_partial(faults, "20240418")) == 1
    # 5. event buffers with every event type/reason, wall- and clip+t-bearing
    clip = "Tol_ch01_20240418-100000-104000.mp4"; base = datetime(2024, 4, 18, 10, 0, 0)
    ev = [{"type": "segment_open", "seg_id": "0_001", "slot": 0, "wall": (base + timedelta(minutes=1)).isoformat(), "reason": "seed_day"},
          {"type": "segment_close", "seg_id": "0_001", "slot": 0, "wall": (base + timedelta(minutes=5)).isoformat(), "reason": "detector_contradiction", "n_frames": 3},
          {"type": "segment_close", "seg_id": "1_001", "slot": 1, "wall": (base + timedelta(minutes=30)).isoformat(), "reason": "day_end", "n_frames": 3},
          {"type": "merge_resolved", "pair": [1, 2], "jumpers": [1], "selfiou": [0.1, 0.9], "onset": (base + timedelta(minutes=9)).isoformat(), "wall": (base + timedelta(minutes=12)).isoformat()},
          {"type": "pass_no_detections", "clip": clip, "t": 1200.0, "forced": None}, {"type": "strike", "slot": 2, "seg_id": "2_001", "clip": clip, "t": 1500.0, "best_overlap": 0.0, "n": 1},
          {"type": "ignored_detection", "clip": clip, "t": 1800.0, "area": 50000, "whole_animal": False, "note": "x"}, {"type": "ignored_detection", "clip": clip, "t": 2500.0, "area": 9000, "whole_animal": False, "note": "x"},
          {"type": "pass", "clip": clip, "t": 100.0, "n_det": 4}, {"type": "clip_unreadable", "clip": clip}, {"type": "session_carry", "clip": clip, "t": 0.0, "slots": [0]},
          {"type": "seeded", "clip": clip, "t": 3.0, "slots": [0, 1, 2, 3], "day": True}, {"type": "mask_area_collapse", "slot": 3, "seg_id": "3_001", "clip": clip, "t": 2280.0, "area": 5000, "median": 60000},
          {"type": "occluded", "slot": 3, "clip": clip, "t": 2290.0, "seg_id": "3_001"}, {"type": "all_closed", "clip": clip, "t": 2300.0},
          {"type": "segment_open", "seg_id": "3_002", "slot": 3, "wall": (base + timedelta(seconds=2295)).isoformat(), "reason": "reopen"}]
    lines = [json.dumps(e) for e in ev] + ["not json", ""]
    parsed = [e for e in (parse_event_line(l) for l in lines) if e]; assert len(parsed) == len(ev)
    iv, st = event_intervals(parsed, a_ref=80000.0); merged = merge_intervals(iv)
    assert in_intervals(base + timedelta(minutes=3, seconds=30), merged)          # contradiction close: -120 s
    assert in_intervals(base + timedelta(minutes=8, seconds=30), merged)          # merge onset - 60 s
    assert not in_intervals(base + timedelta(minutes=7, seconds=30), merged)
    assert in_intervals(base + timedelta(seconds=1800 + 30), merged) and not in_intervals(base + timedelta(seconds=2500 + 30), merged)   # ignored_detection area rule (9000 px < 0.30*A_ref)
    assert st["chaos_windows"] >= 1 and in_intervals(base + timedelta(seconds=2285), merged)                  # 3 chaos events within 5 min at ~38 min
    assert not in_intervals(base + timedelta(seconds=150), merged)                                            # between the seed-open (+60) and the contradiction close (-120): clean
    assert st["by_label"]["clip_unreadable"] == 1                                                              # clip-only event resolved via clip_start + t(=0)
    try: event_intervals([{"type": "partial_state", "clip": clip, "t": 1.0}], 80000.0); raise AssertionError("partial_state must raise")
    except RuntimeError: pass
    # 5b. every remaining type and open/close reason (review 2026-09-23), placed from 55 min on so
    # the assertions above are unaffected; displaced_by_detection gets the same [wall-120 s, wall+60 s] buffer as a contradiction
    b2 = base + timedelta(minutes=55)
    ev2 = [{"type": "segment_close", "seg_id": "2_001", "slot": 2, "wall": b2.isoformat(), "reason": "displaced_by_detection", "n_frames": 9}]
    for k, rs in enumerate(("seed_day", "seed_night", "reopen", "displaced", "self_recovery")):
        ev2.append({"type": "segment_open", "seg_id": f"{k}_009", "slot": k % 4, "wall": (base + timedelta(minutes=75 + k)).isoformat(), "reason": rs})
    for k, rs in enumerate(("merge_jumper", "merge_ambiguous", "not_seedable_at_rebuild", "clip_boundary_stale", "absence_break", "low_iou_reappear", "occluded_timeout", "day_end")):
        ev2.append({"type": "segment_close", "seg_id": f"{k % 4}_008", "slot": k % 4, "wall": (base + timedelta(minutes=80 + k)).isoformat(), "reason": rs, "n_frames": 5,
                    **({"merge_onset": (base + timedelta(minutes=79)).isoformat(), "merge_pair": [0, 1]} if rs.startswith("merge_") else {})})
    ev2.append({"type": "pass", "clip": clip, "t": 5400.0, "forced": None, "n_det": 4, "tracked": [0, 1, 2, 3], "matched": {}, "struck": [], "opened": [], "refreshed": [], "rebuilt": False})
    iv2, st2 = event_intervals(ev + ev2, a_ref=80000.0); m2 = merge_intervals(iv2)
    assert {e["type"] for e in ev + ev2} >= {"segment_open", "segment_close", "merge_resolved", "pass_no_detections", "strike", "ignored_detection", "pass",
                                            "clip_unreadable", "session_carry", "seeded", "mask_area_collapse", "occluded", "all_closed"}
    assert in_intervals(b2 - timedelta(seconds=110), m2) and not in_intervals(b2 - timedelta(seconds=130), m2) and in_intervals(b2 + timedelta(seconds=50), m2)
    assert st2["by_label"]["close_displaced_by_detection"] == 1 and st2["by_label"]["close_detector_contradiction"] == 1 and st2["n_no_wall"] == 0
    for rt in ("partial_state", "grown_state", "partial_repair", "growth_repair", "partial_repair_rejected", "growth_repair_rejected",
               "partial_repair_skipped", "partial_repair_suspended"):
        try: event_intervals([{"type": rt, "slot": 0, "clip": clip, "t": 1.0}], 80000.0); raise AssertionError(f"{rt} must raise")
        except RuntimeError as e: assert "provenance" in str(e), (rt, e)
    for st_, rs in (("segment_open", "partial_repair"), ("segment_open", "growth_repair"), ("segment_close", "partial_repair"), ("segment_close", "growth_repair")):
        try: event_intervals([{"type": st_, "slot": 0, "wall": base.isoformat(), "reason": rs}], 80000.0); raise AssertionError(f"{st_}/{rs} must raise")
        except RuntimeError as e: assert "provenance" in str(e)
    # a buffer-bearing event without a resolvable wall, or a merge with an unparsable onset, raises instead of losing its buffer
    for bad_ev in ({"type": "segment_close", "slot": 0, "wall": "2024-04-18 10:00:00Zjunk", "reason": "day_end"}, {"type": "merge_resolved", "pair": [0, 1]},
                   {"type": "ignored_detection", "area": 90000}, {"type": "merge_resolved", "pair": [0, 1], "wall": base.isoformat(), "onset": "yesterday"}):
        try: event_intervals([bad_ev], 80000.0); raise AssertionError(f"must raise: {bad_ev}")
        except RuntimeError: pass
    assert event_intervals([{"type": "pass", "n_det": 4}], 80000.0)[1]["n_no_wall"] == 1                    # a non-buffer event without a clock is only counted
    # 6. G3-G10 truth table
    seg = SegLookup([{"seg_id": f"{s}_001", "slot": s, "t_start_wall": "2024-04-18T10:00:00"} for s in range(4)])
    C = {"frag_lo": {"dayl": 0.45, "dark": 0.45}, "over_hi": 1.32, "S_lo": {"dayl": 0.55, "dark": 0.55}, "S_hi": {"dayl": 1.25, "dark": 1.25}, "STAB_LO": 0.85, "STAB_HI": 1.15}
    def row(**kw):
        r = {"window": "20240418_w1000-1040", "day": "20240418", "clip_stem": "c", "row_i": 0, "frame_idx": 0, "t_sec": 0.0, "wall": base, "phase4": "day", "n_present": 4, "slots": [0, 1, 2, 3],
             "area": [80000] * 4, "link": [0.9] * 4, "cent": [[1, 1]] * 4, "seg": ["0_001", "1_001", "2_001", "3_001"], "pair_iou": 0.0, "pair_ios": 0.0, "stab": [1.0] * 4, "act": 3.0}
        r.update(kw); return r
    assert gate_pre(row(), [], []) is None
    assert gate_pre(row(n_present=3, area=[80000, 80000, 80000, None]), [], []) == "G3"
    assert gate_pre(row(pair_ios=0.06), [], []) == "G4" and gate_pre(row(pair_iou=0.03), [], []) == "G4"
    assert gate_pre(row(link=[0.9, 0.4, 0.9, 0.9]), [], []) == "G5" and gate_pre(row(link=[0.9, None, 0.9, 0.9]), [], []) == "G5"
    assert gate_pre(row(wall=base + timedelta(minutes=3, seconds=30)), [], merged) == "G2"
    assert gate_post(row(), 80000.0, C, set()) is None
    assert gate_post(row(area=[80000, 80000, 80000, 30000]), 80000.0, C, set()) == "G6"
    assert gate_post(row(area=[80000, 80000, 80000, 36000], phase4="night"), 80000.0, C, set()) is None      # curled 0.45x passes with the derived edge
    assert gate_post(row(area=[104000, 104000, 104000, 104000]), 80000.0, C, set()) == "G7"   # each r=1.30 passes G6, S=1.30 > S_hi
    assert gate_post(row(stab=[1.0, 1.0, 0.7, 1.0]), 80000.0, C, set()) == "G8"
    assert gate_post(row(phase4="night"), 80000.0, C, {("c", "night")}) == "G9"
    one = blob(30, 24, 12, 9); two = blob(15, 24, 6, 6) | blob(48, 24, 6, 6)
    assert largest_cc_frac(one) >= CC_FRAC and largest_cc_frac(two) < CC_FRAC and body_len_of(one) > body_len_of(blob(30, 24, 5, 5))
    # 7. band derivation on a synthetic U with a planted 0.5 % sliver tail
    def synth_u(n, tail=0.005, curl_ok=True, day2_shift=0.0):
        rows = []
        for k in range(n):
            day = "20240418" if k % 2 == 0 else "20240419"; ph = ["day", "dawn", "dusk", "night"][k % 4]
            wall = datetime(2024, 4, 19, 20, 50) + timedelta(seconds=k) if (ph in ("dusk", "night") and curl_ok and k % 8 == 3) else datetime(2024, 4, 18, 12, 0) + timedelta(seconds=k)
            base_r = 0.47 if ph in ("dusk", "night") else 0.55          # body: uniform above a posture floor (curled dark / standing dayl)
            areas = [int(80000 * (base_r + (0.0 if day == "20240418" else day2_shift) + rng.uniform(0, 0.7))) for _ in range(4)]
            if rng.random() < tail: areas[0] = int(80000 * rng.uniform(0.08, 0.20))
            rows.append(row(day=day, phase4=ph, wall=wall, area=areas, stab=[float(rng.uniform(0.9, 1.1)) for _ in range(4)], act=float(rng.uniform(0, 10))))
        return rows
    Cd = derive_constants(synth_u(4000), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END)
    assert 0.30 <= Cd["frag_lo"]["dark"] <= 0.59 and Cd["frag_lo"]["dark"] > 0.20 and Cd["frag_lo"]["dark"] <= 0.62, Cd["frag_lo"]     # edge between tail and body
    assert Cd["curled_check"]["admitted_frac"] >= CURL_ADMIT_MIN and Cd["frag_lo_provenance"] == "derived_P1_of_U" and Cd["STAB_LO"] < 1 < Cd["STAB_HI"]
    try: derive_constants(synth_u(4000, tail=0.30), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END); raise AssertionError("sanity check must raise")
    except RuntimeError as e: assert "sanity" in str(e)
    try: derive_constants(synth_u(4000, day2_shift=0.5), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END); raise AssertionError("per-day check must raise")
    except RuntimeError as e: assert "per-day" in str(e)
    # an override may not reject the curled stretch (planted 30 % sliver tail -> curl admission ~0.7) nor leave the sanity band
    try: derive_constants(synth_u(4000, tail=0.30), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END, override={"dayl": 0.45, "dark": 0.45}); raise AssertionError("override failing the curl check must raise")
    except RuntimeError as e: assert "curled stretch" in str(e), e
    try: derive_constants(synth_u(4000, day2_shift=0.12), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END, override={"dayl": 0.45, "dark": 0.045}); raise AssertionError("override 0.045 must raise")
    except RuntimeError as e: assert "sanity band" in str(e), e
    Co = derive_constants(synth_u(4000, day2_shift=0.12), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END, override={"dayl": 0.45, "dark": 0.45})
    assert Co["frag_lo_provenance"] == "user_override" and Co["frag_lo"]["dark"] == 0.45 and Co["curled_check"]["admitted_frac_override"] >= CURL_ADMIT_MIN
    # G5 support on the training days (link_iou_report) and the curled stretch after G6-G9 (curl_admission)
    lk_rows = [row(link=[0.98, 0.97, 0.99, 0.96]) for _ in range(300)] + [row(link=[0.98, 0.3, 0.97, None])] + [row(day="20240422", link=[0.1] * 4) for _ in range(300)]
    lr = link_iou_report(lk_rows); assert lr["by_pool"]["all"]["n"] == 1203 and lr["n_none"] == 1 and lr["by_pool"]["all"]["frac_below_edge"] < 0.01   # 04-22 rows ignored (training-only constants)
    try: link_iou_report([row(link=[0.98, 0.4, 0.97, 0.99]) for _ in range(50)]); raise AssertionError("an edge inside the link mass must raise")
    except RuntimeError as e: assert "NOT below" in str(e)
    ca = curl_admission([None] * 9 + ["G7"]); assert ca == {"n_rows": 10, "admitted_frac": 0.9, "first_fail": {"pass": 9, "G7": 1}} and curl_admission([])["admitted_frac"] is None
    # relative after-G6-G9 check (review 2026-09-23): 0.86 of the stretch vs 0.88 of the dark pool is proportional (passes,
    # though below the absolute 0.90, which is reported only); 0.70 vs 0.90 fails -> blocking unless acked; an ack without a failure raises
    rb = curl_band_check([None] * 86 + ["G7"] * 8 + ["G8"] * 6, [None] * 88 + ["G8"] * 12)
    assert not rb["fails"] and not rb["blocking"] and rb["below_abs_0.90"] and rb["relative"] == round(0.86 / 0.88, 4)
    rf = curl_band_check([None] * 70 + ["G7"] * 30, [None] * 90 + ["G6"] * 10)
    assert rf["fails"] and rf["blocking"] and rf["ack"] is None
    ra = curl_band_check([None] * 70 + ["G7"] * 30, [None] * 90 + ["G6"] * 10, ack="sheet curl_rejected_000 inspected: complete curled calves")
    assert ra["fails"] and not ra["blocking"] and ra["ack"].startswith("sheet")
    try: curl_band_check([None] * 86 + ["G7"] * 14, [None] * 88 + ["G8"] * 12, ack="x"); raise AssertionError("ack without a failure must raise")
    except RuntimeError as e: assert "did not fail" in str(e), e
    # note exclusions (review 2026-09-23): each 'unlisted but visible' wall excludes the fault-free gap around it, capped at ±max_s
    nf = {"notes": ["Unlisted but visible on sheets: 04-18 23:15:08 blue over-extended; 04-19 23:33:29 yellow over-extended; 04-18 22:48:32 clustered."],
          "faults": [{"day": "20240418", "id": "A7", "exclude_start": "2024-04-18T22:47:02", "exclude_end": "2024-04-18T23:00:48"},
                     {"day": "20240418", "id": "A8", "exclude_start": "2024-04-18T23:11:08", "exclude_end": "2024-04-18T23:14:58"},
                     {"day": "20240418", "id": "A9", "exclude_start": "2024-04-18T23:24:28", "exclude_end": "2024-04-18T23:34:04"},
                     {"day": "20240419", "id": "B15", "exclude_start": "2024-04-19T23:29:29", "exclude_end": "2024-04-19T23:33:20"}]}
    niv, nrep = note_exclusions(nf, ("2024-04-18T23:15:08", "2024-04-19T23:33:29", "2024-04-18T22:48:32"), 600.0)
    assert niv["20240418"] == [(datetime(2024, 4, 18, 23, 14, 58), datetime(2024, 4, 18, 23, 24, 28), "N1")]            # whole A8 -> A9 gap
    assert niv["20240419"] == [(datetime(2024, 4, 19, 23, 33, 20), datetime(2024, 4, 19, 23, 43, 29), "N2")]            # no next fault -> +600 s cap
    assert nrep[0]["capped"] is False and nrep[1]["capped"] is True and nrep[2] == {"id": "N3", "wall": "2024-04-18T22:48:32", "day": "20240418", "already_excluded_by": "A7"}
    ivn = sorted(fault_intervals(nf, "20240418") + niv["20240418"])
    assert fault_hit(datetime(2024, 4, 18, 23, 20), ivn) == "N1" and fault_hit(datetime(2024, 4, 18, 23, 14, 57), ivn) == "A8" and fault_hit(datetime(2024, 4, 18, 23, 10), ivn) is None
    try: note_exclusions(nf, ("2024-04-18T12:00:00",), 600.0); raise AssertionError("a wall absent from the notes must raise")
    except RuntimeError as e: assert "stale" in str(e), e
    try: derive_constants(synth_u(4000), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END, override={"dayl": 0.4, "dark": 0.4}); raise AssertionError("override without a raise must raise")
    except RuntimeError: pass
    # an emptied curled stretch makes the human-verdict check vacuous -> raise (not overridable); only a run without 04-19 may skip it
    try: derive_constants(synth_u(4000, curl_ok=False), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END, curl_context={"G2": 2000}); raise AssertionError("empty curled stretch must raise")
    except RuntimeError as e: assert "vacuous" in str(e) and "G2" in str(e)
    try: derive_constants(synth_u(4000, curl_ok=False), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END, override={"dayl": 0.45, "dark": 0.45}); raise AssertionError("override must not bypass it")
    except RuntimeError as e: assert "vacuous" in str(e)
    Cn = derive_constants(synth_u(4000, curl_ok=False), {"20240418": 80000.0, "20240419": 80000.0}, 1.32, CURL_START, CURL_END, curl_required=False)
    assert Cn["curled_check"] == {"n_rows": 0, "admitted_frac": None}
    # 8. block collapse (a), (b), (c) + override
    def brows(stem, areas, ph="night"): return [row(clip_stem=stem, phase4=ph, area=list(areas), wall=datetime(2024, 4, 18, 22, 0) + timedelta(seconds=k)) for k in range(20)]
    U = brows("ok", [80000] * 4) + brows("a", [50000] * 4) + brows("b", [80000, 80000, 80000, 20000]) + brows("c", [80000, 80000, 80000, 45000])
    dropped, rep = block_collapse(U, {"20240418": 80000.0}, 0.45)
    assert ("a", "night") in dropped and ("b", "night") in dropped and ("c", "night") not in dropped and ("ok", "night") not in dropped
    assert rep["blocks_failing_per_slot_0.75"] == 3 and rep["curled_or_clipped_blocks"][0]["block"] == "c:night" and "a:night" in rep["needs_human_verdict"]
    dropped2, _ = block_collapse(U, {"20240418": 80000.0}, 0.45, overrides=["b:night"]); assert ("b", "night") not in dropped2 and ("a", "night") in dropped2
    # G1/G2-excluded slivers must not drop a block: block "d" is clean on its admissible rows, sliver-dominated over all rows
    d_adm = brows("d", [80000] * 4); d_all = d_adm + brows("d", [80000, 80000, 80000, 15000]) * 2
    dropped3, rep3 = block_collapse(U + d_adm, {"20240418": 80000.0}, 0.45, rows_all=U + d_all)
    assert ("d", "night") not in dropped3 and rep3["detail"]["d:night"]["all_rows"]["slot_median"]["3"] < 0.45 < rep3["detail"]["d:night"]["slot_median"]["3"]
    assert ("d", "night") in block_collapse(U + d_all, {"20240418": 80000.0}, 0.45)[0]        # the same block over ALL rows would have been dropped
    # 9. lattice + quota + hour cap on a supply above n_max; clip split; G12
    sup = []
    for k in range(3000):
        wall = datetime(2024, 4, 18, 9, 0) + timedelta(seconds=k * 3)
        sup.append(row(clip_stem="L", wall=wall, phase4=phase4(wall), frame_idx=k, t_sec=k * 3.0, act=float(k % 10)))
    for k in range(3000):
        wall = datetime(2024, 4, 18, 21, 30) + timedelta(seconds=k * 3)
        sup.append(row(clip_stem="N", wall=wall, phase4=phase4(wall), frame_idx=10000 + k, t_sec=k * 3.0, act=float(k % 10)))
    picks = lattice([r for r in sup if r["clip_stem"] == "L"], 10, 5); gaps = np.diff([p["wall"].timestamp() for p in picks])
    # every pick sits within LATTICE_TOL_S of a 10 s tick (a 3 s supply cannot hit every tick, so gaps are multiples of 10 ± 2)
    assert len(picks) > 100 and gaps.min() >= 8 - 1e-6 and all(min(g % 10, 10 - g % 10) <= LATTICE_TOL_S + 1e-6 for g in gaps), (len(picks), gaps.min(), gaps.max())
    picksN = lattice([r for r in sup if r["clip_stem"] == "N"], 10, 5); gN = np.diff([p["wall"].timestamp() for p in picksN])
    assert gN.min() >= 3 - 1e-6 and all(min(g % 5, 5 - g % 5) <= LATTICE_TOL_S + 1e-6 for g in gN) and len(picksN) > len(picks) * 1.3
    # cross-clip overlap (review 2026-09-23): clip A 09:00:00-09:01:00 and clip B from 09:00:58 overlap by 2 s at 5 fps; the
    # per-clip lattices pick A@09:01:00 and B@09:00:58 — exactly one pick may survive inside the overlap
    t_a = datetime(2024, 4, 18, 9, 0); t_b = t_a + timedelta(seconds=58)
    ov_rows = [row(clip_stem="A", wall=t_a + timedelta(seconds=k * 0.2), frame_idx=k) for k in range(301)] + \
              [row(clip_stem="B", wall=t_b + timedelta(seconds=k * 0.2), frame_idx=k) for k in range(301)]
    ov_picks = lattice([r for r in ov_rows if r["clip_stem"] == "A"], 10, 5) + lattice([r for r in ov_rows if r["clip_stem"] == "B"], 10, 5)
    in_ov = lambda ps: [p for p in ps if t_b <= p["wall"] <= t_a + timedelta(seconds=60)]
    assert len(in_ov(ov_picks)) == 2, [(p["clip_stem"], p["wall"]) for p in in_ov(ov_picks)]           # the defect being guarded
    ov_kept, ov_n = cross_clip_dedupe(ov_picks, 5)
    assert ov_n == 1 and len(in_ov(ov_kept)) == 1 and in_ov(ov_kept)[0]["clip_stem"] == "B" and len(ov_kept) == len(ov_picks) - 1   # the LATER (A@09:01:00) goes
    assert [id(p) for p in ov_kept] == [id(p) for p in ov_picks if p in ov_kept]                        # input order kept (seeded allocation downstream)
    assert cross_clip_dedupe(picks, 5) == (picks, 0)                                                    # one clip: never deduped, whatever the spacing
    same = [row(clip_stem="S", wall=t_a), row(clip_stem="S", wall=t_a + timedelta(seconds=1))]
    assert cross_clip_dedupe(same, 5)[1] == 0 and cross_clip_dedupe([row(clip_stem="X", wall=t_a), row(clip_stem="Y", wall=t_a, day="20240419")], 5)[1] == 0   # same clip / other day untouched
    sel, rep = allocate(picks + picksN, 500, {"night": 0.35, "dusk": 0.08, "dawn": 0.08, "active": 0.15}, {"20240418": 7.0}, 0)
    # quotas are FLOORS (night >= 0.35, active >= 0.15); dusk/dawn have no supply and must be reported as shortfalls
    assert len(sel) == 500 and rep["binding"] and rep["achieved"]["night"] >= 0.35 - 1e-9 and "dusk" in rep["shortfall"] and "dawn" in rep["shortfall"] and rep["achieved"]["active"] >= 0.15 - 1e-9, rep
    sel_small, rep_small = allocate(picks[:50], 500, {"night": 0.35, "dusk": 0.08, "dawn": 0.08, "active": 0.15}, {"20240418": 7.0}, 0)
    assert len(sel_small) == 50 and not rep_small["binding"] and "night" in rep_small["shortfall"]         # below n_max nothing is dropped, shortfalls still printed
    capped, trimmed = hour_cap(sel, 0.20, 0); assert max_hour_share(capped) <= 0.20 and sum(trimmed.values()) > 0, trimmed
    # the cap holds on the FINAL set (fixed point), not only on the pre-trim size: 3 bins x 600 + 10 x 330 left 9.0 % bins in one pass
    hc_rows = [row(wall=datetime(2024, 4, 18, h, 0)) for h in range(3) for _ in range(600)] + [row(wall=datetime(2024, 4, 18, 8 + h, 0)) for h in range(10) for _ in range(330)]
    hc_out, hc_tr = hour_cap(hc_rows, 0.08, 0); assert max_hour_share(hc_out) <= 0.08 and len(hc_out) < 4524 and sum(hc_tr.values()) == 5100 - len(hc_out), (len(hc_out), max_hour_share(hc_out))
    assert hour_cap(hc_out, 0.08, 0)[1] == {}                                                       # idempotent at the fixed point
    try: hour_cap([row(wall=datetime(2024, 4, 18, h, 0)) for h in range(12) for _ in range(50)], 0.08, 0); raise AssertionError("12 equal bins cannot satisfy an 8 % cap")
    except RuntimeError as e: assert "infeasible" in str(e)
    ach_c, short_c = quota_achieved(capped, {"night": 0.35, "dusk": 0.08, "dawn": 0.08, "active": 0.15})   # the cap can change the shares: they must be recomputed on the final set
    assert quota_achieved(sel, {"night": 0.35, "dusk": 0.08, "dawn": 0.08, "active": 0.15})[0] == rep["achieved"] and set(ach_c) == {"night", "dusk", "dawn", "day", "active"} and "dusk" in short_c
    tr, va = split_clips(["a", "b", "c"], ["b"]); assert tr == ["a", "c"] and va == ["b"]
    try: split_clips(["a"], ["zz"]); raise AssertionError("missing valid clip must raise")
    except RuntimeError: pass
    keep, nd = valid_buffer_drop(sup[:100], [("20240418", datetime(2024, 4, 18, 9, 2), datetime(2024, 4, 18, 9, 3))], 1.0); assert nd > 0 and len(keep) + nd == 100
    assert distinct_10min_blocks(sup) == len({(r["day"], int(r["wall"].timestamp() // 600)) for r in sup})
    # 10. crop-window acceptance and exact re-encode
    m_edge = blob(20, 24, 6, 6); m_far = blob(52, 24, 5, 5)
    rr = random.Random(0); win = draw_crop_window([m_edge, m_far, None, None], rr, 32, 18)
    if win is not None:
        x, y, kept = win
        for s, m in ((0, m_edge), (1, m_far)):
            f = m[y:y + 18, x:x + 32].sum() / m.sum(); assert f >= CROP_KEEP_FRAC or f <= CROP_DROP_FRAC
            if s in kept: assert f >= CROP_KEEP_FRAC
        assert np.array_equal(rle_decode(crop_mask_rle(m_edge, x, y, 32, 18)), m_edge[y:y + 18, x:x + 32])
    class _R:   # forced draws: a fixed sequence of (x, y) origins, the last one repeated (randint is called x then y per draw)
        def __init__(self, xy): self.xy = list(xy); self.n = 0
        def randint(self, a, b):
            v = self.xy[min(self.n // 2, len(self.xy) - 1)][self.n % 2]; self.n += 1
            assert a <= v <= b; return v
    f_sl = m_edge[2:20, 0:32].sum() / m_edge.sum(); f_ok = m_edge[10:28, 19:51].sum() / m_edge.sum(); f_0 = m_edge[0:18, 26:58].sum() / m_edge.sum()
    assert CROP_DROP_FRAC < f_sl < CROP_KEEP_FRAC and f_ok >= CROP_KEEP_FRAC and f_0 == 0.0, (f_sl, f_ok, f_0)   # the three regimes really are exercised
    assert draw_crop_window([m_edge, None, None, None], _R([(0, 2)]), 32, 18, allow_zero=True) is None            # every draw slices a sliver -> no window
    assert draw_crop_window([m_edge, None, None, None], _R([(0, 2), (19, 10)]), 32, 18) == (19, 10, [0])          # sliver draw rejected, >= 25 % draw kept
    assert draw_crop_window([m_edge, None, None, None], _R([(26, 0)]), 32, 18, allow_zero=True) == (26, 0, [])    # 0 % inside -> zero-calf window
    assert draw_crop_window([m_edge, None, None, None], _R([(26, 0)]), 32, 18, allow_zero=False) is None          # ... refused when the zero-calf cap is hit
    assert draw_crop_window([m_edge, m_far, None, None], _R([(0, 2), (19, 10)]), 32, 18) == (19, 10, [0, 1]) or m_far[10:28, 19:51].sum() / m_far.sum() < CROP_KEEP_FRAC
    # 11. COCO assembly + pycocotools round trip
    try:
        from pycocotools import mask as mU
        masks = [blob(rng.uniform(12, 52), rng.uniform(10, 38), rng.uniform(4, 10), rng.uniform(4, 10)) for _ in range(20)]
        anns, imgs = [], []
        for i, m in enumerate(masks):
            iid = i // 4 + 1
            if i % 4 == 0: imgs.append({"id": iid, "file_name": f"f{iid}.jpg", "width": Wd, "height": H, "crop_variant": False})
            a = coco_annotation(i + 1, iid, rle_encode(m), H, Wd, i % 4, f"{i%4}_001", mU); anns.append(a)
            dec = mU.decode({"size": [H, Wd], "counts": a["segmentation"]["counts"].encode("ascii")}).astype(bool)
            assert np.array_equal(dec, m) and a["area"] == m.sum() and isinstance(a["segmentation"]["counts"], str)
        coco = {"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "calf", "supercategory": "none"}]}
        assert coco_check(coco)
        bad = json.loads(json.dumps(coco)); bad["annotations"].pop()
        try: coco_check(bad); raise AssertionError("3-annotation full image must fail")
        except RuntimeError: pass
        print("SELFTEST: pycocotools round trip 20/20 exact")
    except ImportError:
        print("SELFTEST: pycocotools unavailable — skipping the COCO round-trip only")
    # 12. human-gate verdict + seal rule truth table, BOTH require_human_gate settings
    V = human_gate_verdict
    assert V(False, "pending", 0, 0, 0, 0) == "absent"
    assert V(True, "passed", 3, 0, 300, 300) == "passed" and V(True, "passed", 4, 0, 300, 300) == "failed"       # <= 3 BAD counted
    assert V(True, "passed", 0, 1, 300, 300) == "failed" and V(True, "failed", 0, 0, 300, 300) == "failed"       # wrong_object anywhere / the reviewer's own 'failed'
    assert V(True, "pending", 5, 0, 300, 300) == "failed" and V(True, "pending", 1, 0, 300, 300) == "pending"    # verdicts are evidence even in a pending review
    assert V(True, "passed", 0, 0, 0, 0) == "pending" and V(True, "passed", 0, 0, 0, 300) == "pending"           # sheet_n=1 / no counted cell reviewed: not passed
    assert V(True, "passed", 0, 0, 100, 100) == "pending"                                                        # planned below SHEET_N_REF
    assert V(True, "passed", 0, 0, 280, 300) == "passed" and V(True, "passed", 0, 0, 350, 350) == "passed"       # pool shortfall legal; enlarged tail stratum
    try: V(True, "approved", 0, 0, 300, 300); raise AssertionError("an unknown status must raise")
    except RuntimeError: pass
    assert int(round(SHEET_MIX["tail"] * (1 / SHEET_N_REF))) == 0                                            # the sheet_n=1 case really plans nothing
    # seal_rule(hg_verdict, require_human_gate, below_floor, veto_ran, require_veto, smoke)
    for req in (True, False):
        assert seal_rule("passed", req, False, True, True, False)                                            # a passed review seals under both settings
        assert not seal_rule("failed", req, False, True, True, False)                                        # a present-but-failed gate ALWAYS unseals
        assert not seal_rule("passed", req, True, True, True, False)                                         # below floor
        assert not seal_rule("passed", req, False, False, True, False) and seal_rule("passed", req, False, False, False, False)   # veto missing / waived
        assert not seal_rule("passed", req, False, True, True, True)                                         # smoke never seals
    assert not seal_rule("absent", True, False, True, True, False) and not seal_rule("pending", True, False, True, True, False)    # required: only passed seals
    assert seal_rule("absent", False, False, True, True, False) and seal_rule("pending", False, False, True, True, False)          # not required: the fault list is the gate
    assert not seal_rule("absent", False, True, True, True, False) and not seal_rule("absent", False, False, False, True, False)   # ...but the other terms still bind
    try: seal_rule("ok", False, False, True, True, False); raise AssertionError("an unknown verdict must raise")
    except RuntimeError: pass
    # stage-gate sheet_n rule: the >= 300 raise applies only when require_human_gate=true (and never in SMOKE)
    assert gate_sheet_n_check(300, True, False) is None and gate_sheet_n_check(300, False, False) is None
    assert gate_sheet_n_check(12, True, True) is None and gate_sheet_n_check(12, False, True) is None
    assert "allowed because require_human_gate=false" in gate_sheet_n_check(12, False, False)
    try: gate_sheet_n_check(12, True, False); raise AssertionError("sheet_n < 300 with require_human_gate=true must raise")
    except RuntimeError: pass
    # end-to-end human-gate outcome under both settings: a present file is still evaluated, its BAD cells still reject
    # frames, and a failed one unseals even when not required
    def _hg_outcome(present, status, n_bad, n_wrong, n_counted, planned, req):
        v = human_gate_verdict(present, status, n_bad, n_wrong, n_counted, planned); return v, seal_rule(v, req, False, True, True, False)
    assert _hg_outcome(False, "pending", 0, 0, 0, 300, False) == ("absent", True) and _hg_outcome(False, "pending", 0, 0, 0, 300, True) == ("absent", False)
    assert _hg_outcome(True, "pending", 2, 0, 300, 300, False) == ("pending", True) and _hg_outcome(True, "pending", 2, 0, 300, 300, True) == ("pending", False)
    assert _hg_outcome(True, "passed", 0, 0, 120, 120, False) == ("pending", True)     # undersized review: fine when not required
    for req in (True, False):
        assert _hg_outcome(True, "passed", 4, 0, 300, 300, req) == ("failed", False)   # > 3 BAD counted
        assert _hg_outcome(True, "pending", 0, 1, 300, 300, req) == ("failed", False)  # any wrong_object
        assert _hg_outcome(True, "failed", 0, 0, 12, 12, req) == ("failed", False)     # the reviewer's own 'failed', even on a tiny sheet
    # manifest.human_gate.status is the recomputed outcome: a reader that checks only `status` (Tol_43's seal_terms_unmet /
    # human_gate_optional_check) must reach the same human-gate term as seal_rule for every verdict x setting
    assert [effective_hg_status(v) for v in HG_VERDICTS] == ["pending", "pending", "passed", "failed"]
    for v in HG_VERDICTS:
        for req in (True, False):
            st = effective_hg_status(v); reader_ok = (st == "passed") if req else (st != "failed")
            assert reader_ok == seal_rule(v, req, False, True, True, False), (v, req)
    msg_a = below_floor_message(4000, 5000, "x", 10, 5); msg_b = below_floor_message(4000, 5000, "x", 8, 4)
    assert msg_a.index("(a)") < msg_a.index("(b)") and "FIRST CHOICE" in msg_a and "ALREADY APPLIED" not in msg_a and "ALREADY APPLIED" in msg_b
    assert night_valid_message([{"phase4": "night"}] * 50 + [{"phase4": "day"}] * 9, "x") is None           # Tol_43 VALID_NIGHT_MIN edge: 50 passes
    _nv = night_valid_message([{"phase4": "night"}] * 49 + [{"phase4": "dusk"}] * 30, "x")
    assert _nv and "49 night valid frames < 50" in _nv and "allow_thin_night_valid" in _nv                    # dusk does not count as night
    assert "protocol is to DENSIFY FIRST" in msg_a and "stride_day_s=8, stride_other_s=4" in msg_a and msg_a.index("stride_day_s=8") < msg_a.index("Tracking_teacher:20240422")
    hg = {"status": "passed", "cells": [{"cell_id": "S000c00", "counted": True, "verdict": "partial", "day": "20240418", "wall": "2024-04-18T10:00:00"},
                                        {"cell_id": "S000c01", "counted": True, "verdict": "partial_minor"}, {"cell_id": "X0", "counted": False, "verdict": "wrong_object", "day": "20240418", "wall": "2024-04-18T11:00:00"}]}
    ev_ = human_gate_eval(hg); assert ev_["n_bad_counted"] == 1 and ev_["n_wrong_object"] == 1 and ev_["n_counted"] == 2 and len(ev_["bad_cells"]) == 2
    fr = [{"image_id": 1, "day": "20240418", "wall": "2024-04-18T10:01:00"}, {"image_id": 2, "day": "20240418", "wall": "2024-04-18T10:05:00"}, {"image_id": 3, "day": "20240419", "wall": "2024-04-19T10:01:00"}]
    assert human_reject_set(ev_["bad_cells"], fr) == {1}
    # the filled gate must carry exactly the template's cells: an emptied or foreign file cannot pass the seal
    tpl_cells = [{"cell_id": "S000c00", "image_id": 1, "day": "20240418", "wall": "2024-04-18T10:00:00", "clip_stem": "c", "frame_idx": 0, "counted": True, "kind": "tail", "verdict": None},
                 {"cell_id": "S000c01", "image_id": 2, "day": "20240418", "wall": "2024-04-18T10:05:00", "clip_stem": "c", "frame_idx": 5, "counted": True, "kind": "night", "verdict": None},
                 {"cell_id": "X0", "image_id": 3, "day": "20240418", "wall": "2024-04-18T11:00:00", "clip_stem": "c", "frame_idx": 9, "counted": False, "kind": "probe", "verdict": None}]
    filled = {"status": "passed", "cells": [dict(c, verdict="ok") for c in tpl_cells]}
    assert check_human_gate_vs_template(filled, tpl_cells) == 2
    for bad_gate in ({"status": "passed", "cells": []}, {"status": "passed", "cells": filled["cells"][:2]},
                     {"status": "passed", "cells": filled["cells"] + [{"cell_id": "S009c00", "counted": True}]},
                     {"status": "passed", "cells": [dict(c, image_id=99) if c["cell_id"] == "S000c00" else c for c in filled["cells"]]},
                     {"status": "passed", "cells": [dict(c, day="20240419") if c["cell_id"] == "S000c00" else c for c in filled["cells"]]},     # wrong day: its BAD verdict would reject nothing
                     {"status": "passed", "cells": [{k: v for k, v in c.items() if k != "day"} if c["cell_id"] == "S000c00" else c for c in filled["cells"]]},   # missing day
                     {"status": "passed", "cells": [dict(c, counted=False) if c["cell_id"] == "S000c00" else c for c in filled["cells"]]}):
        try: check_human_gate_vs_template(bad_gate, tpl_cells); raise AssertionError(f"must raise: {bad_gate}")
        except RuntimeError: pass
    # 13. manifest / source guards
    for bad_src in ("Tracking_train:20240423", "Tracking_teacher:20240425"):
        try: guard_source_days(parse_sources(bad_src), "reason"); raise AssertionError(f"{bad_src} must raise")
        except RuntimeError: pass
    try: guard_source_days(parse_sources("Tracking_teacher:20240422"), ""); raise AssertionError("04-22 without a reason must raise")
    except RuntimeError: pass
    assert guard_source_days(parse_sources("Tracking_train:20240418,20240419;Tracking_teacher:20240422"), "data short") == ["20240418", "20240419", "20240422"]
    try: manifest_guard_days(["20240418", "20240424"]); raise AssertionError("manifest guard must raise")
    except RuntimeError: pass
    # the manifest carries what Tol_44 reads (forbidden_eval_days + frag_lo[dayl|dark] + over_hi, training-derived)
    _c = lambda v, df=("20240418", "20240419"): {"value": v, "derived_from": list(df), "not_from": not_from_days(df)}
    man_ok = {"source_days": ["20240418", "20240419", "20240422"], "forbidden_eval_days": ["20240418", "20240419", "20240422"],
              "constants": {"frag_lo[dayl]": _c(0.52), "frag_lo[dark]": _c(0.45), "over_hi": _c(1.32)}}
    assert eval_contract_check(man_ok)
    for mut in (lambda m: m.pop("forbidden_eval_days"), lambda m: m.__setitem__("forbidden_eval_days", ["20240418"]),
                lambda m: m["constants"].pop("frag_lo[dark]"), lambda m: m["constants"].__setitem__("over_hi", _c(float("nan"))),
                lambda m: m["constants"].__setitem__("frag_lo[dayl]", _c(0.5, ("20240422",))),
                lambda m: m["constants"].__setitem__("over_hi", _c(0.9))):
        m2 = json.loads(json.dumps(man_ok)); mut(m2)
        try: eval_contract_check(m2); raise AssertionError("eval contract check must raise")
        except RuntimeError: pass
    # not_from never contains a source day: with 04-22 as a source, A_ref[20240422] is derived_from it and not_from excludes it
    # not_from is per constant: a pooled constant (derived from the training days) lists EVERY test day, 04-22 included, even when
    # 04-22 is a label source; A_ref[20240422] (derived from 04-22) omits exactly 04-22; a literal (derived_from []) lists all
    assert not_from_days(["20240418", "20240419"]) == sorted(DET_TEST_DAYS) and not_from_days([]) == sorted(DET_TEST_DAYS)
    nf = not_from_days(["20240422"]); assert "20240422" not in nf and nf == [f"202404{d}" for d in range(23, 29)]
    assert not (set(nf) & {"20240422"}) and all(not (set(not_from_days(df)) & set(df)) for df in (["20240418"], ["20240422"], ["20240418", "20240422"]))
    # gate-time widgets: the gate's values win; an explicit different value raises; a widget left at its default is ignored
    dflt = {k: "0" for k in GATE_KEYS}; dflt.update({"stride_day_s": "10", "stride_other_s": "5", "collapse_override_blocks": "", "min_train": "5000", "sheet_n": "300"})
    gw_gate = {k: 0 for k in GATE_KEYS}; gw_gate.update({"stride_day_s": 8.0, "stride_other_s": 4.0, "collapse_override_blocks": "Tol_ch01_x:night", "min_train": 5000, "sheet_n": 300})
    now = dict(gw_gate); now.update({"stride_day_s": 10.0, "stride_other_s": 5.0, "collapse_override_blocks": ""})     # a finalize payload that passed only the required widgets
    res, ign = gate_widgets_resolve(gw_gate, now, dflt); assert res["stride_day_s"] == 8.0 and res["collapse_override_blocks"] == "Tol_ch01_x:night" and set(ign) == {"stride_day_s", "stride_other_s", "collapse_override_blocks"}
    try: gate_widgets_resolve(gw_gate, dict(gw_gate, min_train=4000), dflt); raise AssertionError("min_train changed after gate must raise")
    except RuntimeError as e: assert "min_train" in str(e)
    try: gate_widgets_resolve({k: v for k, v in gw_gate.items() if k != "seed"}, now, dflt); raise AssertionError("missing gate widget must raise")
    except RuntimeError: pass
    try: gate_widgets_resolve(None, now, dflt); raise AssertionError("admitted.json without gate_widgets must raise")
    except RuntimeError: pass
    try: check_faults_file({"semantics": "whitelist", "exclusion_margin_s": 90, "tracking_root": "Tracking_train", "windows": {}}, "Tracking_train", ["20240418"], ""); raise AssertionError()
    except RuntimeError: pass
    assert check_faults_file({"semantics": "FAULT list", "exclusion_margin_s": 90, "tracking_root": "Tracking_train", "windows": {"20240418": []}}, "Tracking_train", ["20240418", "20240422"], "gates_only") == ["20240422"]
    # 14. plate helpers + position prior on synthetic data
    plate = build_plate([np.full((PLATE_H, PLATE_W), 100, np.uint8) + (k % 3) for k in range(9)]); assert plate.shape == (PLATE_H, PLATE_W) and int(plate[0, 0]) == 101
    thr, med, mad = plate_thr(np.array([0, 1, 1, 2, 30]), 3.0); assert thr >= 12.0
    fr4 = plate.copy(); fr4[:10, :10] = 200; m4 = np.zeros((PLATE_H, PLATE_W), bool); m4[:10, :10] = True; m4[20:30, 20:30] = True
    assert abs(furniture_consistency(m4, fr4, plate, thr) - 0.5) < 1e-6
    sets = [np.stack([blob(20, 24, 6, 6), blob(44, 24, 6, 6), blob(32, 10, 5, 5), blob(32, 38, 5, 5)]) for _ in range(5)]
    sets.append(np.stack([blob(10, 10, 4, 4), blob(54, 38, 4, 4), blob(10, 38, 4, 4), blob(54, 10, 4, 4)]))
    pp = position_prior_medoid(sets, list(range(6))); assert pp["image_id"] in range(5) and pp["mean_matched_iou"] > 0.5
    # 15. phases / pools / measure_clip on a synthetic clip
    assert phase4(datetime(2024, 4, 18, 6, 30)) == "dawn" and phase3(datetime(2024, 4, 18, 6, 30)) == "day" and phase4(datetime(2024, 4, 18, 20, 5)) == "dusk" and pool_of("dusk") == "dark"
    frames = []
    for i in range(40):
        ms = {str(s): rle_encode(np.pad(blob(12 + 14 * s + (i // 5) * 0.2, 24, 5, 5), ((0, H_FRAME - H), (0, W_FRAME - Wd)))) for s in range(4)}
        frames.append({"frame_idx": i * 5, "t_sec": round(i * 0.2, 2), "masks": ms})
    rows = measure_clip(frames, "Tol_ch01_20240418-100000-104000.mp4", "20240418_w1000-1040", seg)
    assert len(rows) == 8 and all(r["n_present"] == 4 for r in rows) and all(l is not None and l > 0.5 for r in rows[1:-1] for l in r["link"]) and rows[0]["link"][0] is None
    assert rows[3]["stab"][0] is not None and rows[0]["wall"] == datetime(2024, 4, 18, 10, 0, 0) and rows[0]["clip_stem"] == "Tol_ch01_20240418-100000-104000"
    print("SELFTEST: PASS (rle, runs, centroid, faults, events, gates G3-G10, band derivation incl. vacuous-curl raise, blocks incl. all-row report, "
          "lattice incl. cross-clip overlap dedupe/quota/cap incl. post-cap shares and fixed-point hour cap, split, crops incl. sliver rejection, COCO, human-gate verdict + seal under "
          "require_human_gate true AND false incl. the gate sheet_n rule and a present/failed file, below-floor option order + densify-first wording, "
          "Tol_44 manifest contract, human-gate template check, guards incl. not_from and gate-widget drift, plates, prior, measure_clip, "
          "every event type/reason incl. displaced_by_detection and the repair-era raises, override range + curl raise, G5 training-day support, curl after G6-G9 (relative + ack), note exclusions, template day check)")

_selftest()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading it
# MAGIC
# MAGIC **The funnel is the first thing to read** (`Metrics_seg_export_<tag>.json → funnel`, per day × phase4 × gate,
# MAGIC `[rows_in, rows_out]`). G1 is the reviewed fault list, G2 the tracker's own event buffers, G3 exactly-4, G4
# MAGIC disjointness, G5 the 5 fps link, G6–G9 the derived area/stability/block rules, G10 single-component, G11 the
# MAGIC SAM3 veto (finalize), G12 the valid-clip buffer. A gate that removes a large share of one phase is either doing
# MAGIC real work or is mis-set; the `rejected_G<k>.png` sheets show which — look at what each rule throws away
# MAGIC before trusting its count.
# MAGIC
# MAGIC **`band_derivation`** ensures the reviewer-confirmed curled stretch stays admitted. `frag_lo[dayl]`/`frag_lo[dark]` are the P1 of the
# MAGIC per-mask ratios inside the admitted set `U` (G0–G5 survivors — no area rule in their definition), so
# MAGIC the band can only trim the tails of frames the reviewers certified. Expected ≈ 0.45 for `dark`. The three checks
# MAGIC (sanity 0.30–0.59, per-day agreement 0.10, ≥ 90 % of the curled 04-19 20:41–21:17 stretch admitted) raise
# MAGIC rather than warn; after a raise an operator sets `frag_lo_override` with the printed distributions in hand and the
# MAGIC manifest records `frag_lo_provenance: user_override`. `over_hi` is READ from `Metrics_healthy_band_train.json`.
# MAGIC `band_derivation.png` overlays the derived edge, `over_hi` and Tol_41's 0.59 on the two pooled histograms.
# MAGIC `curled_check.after_G6_G9` repeats the question after the WHOLE band, but RELATIVELY: G7/G8 are percentile trims
# MAGIC of `U` by construction, so the stretch only has to keep ≥ 0.90× what the rest of the dark pool keeps (the absolute
# MAGIC 0.90 is printed for information). If it does not, stage gate still writes every sheet (incl. `curl_rejected_*.png`,
# MAGIC the stretch's own rejects), then raises without writing `admitted.json`; re-run gate with `collapse_override_blocks`
# MAGIC and/or `curl_band_ack='<reason>'` (recorded in the manifest).
# MAGIC
# MAGIC **`note_exclusions`** — the fault file's notes name two "unlisted but visible" OVER-EXTENDED masks
# MAGIC (04-18 23:15:08, 04-19 23:33:29). An over-extension onto straw inside the band passes every automatic gate, so the
# MAGIC notebook excludes the fault-free gap around each (A8→A9, 9.5 min; B15→B16, 2 min) as ids N1/N2 of G1.
# MAGIC
# MAGIC **`blocks`** — G9 drops a whole dusk/night block only when its LARGEST-mask median falls under 0.75× the day
# MAGIC anchor (camera-level collapse) or a slot's median falls under `frag_lo[dark]` (a persistent sliver would become
# MAGIC the boundary convention). Slots between the two are listed as `curled_or_clipped_blocks` and admitted; a
# MAGIC dropped block with no listed fault inside is printed as "needs a human verdict" and `collapse_override_blocks`
# MAGIC re-admits it, recorded in the manifest.
# MAGIC
# MAGIC **`counts` / `sampling`** — `train_full` is what matters against the 5,000 floor; `distinct_10min_blocks` says
# MAGIC how many independent-ish stretches those frames come from (do not read 5 k frames as n = 5 k). Quota
# MAGIC shortfalls print as `⚠ QUOTA UNMET`; nothing is dropped for a quota unless supply exceeds `n_max`.
# MAGIC Below the floor everything is still written but the manifest is `sealed: false, below_floor: true`. The order
# MAGIC (decision 2026-09-23): **first** re-run under a new tag with `stride_day_s=8, stride_other_s=4` (04-22 stays a
# MAGIC detector test day); only if that is still short, add `Tracking_teacher:20240422` with `allow_test_day_reason` (04-22 then
# MAGIC leaves the test set and the Tol_44 evaluation can use only 04-23).
# MAGIC
# MAGIC **The human gate.** By default (`require_human_gate=false`) the reviewed fault list — both training-day videos reviewed
# MAGIC end to end — IS the human gate, and the dataset can seal without a second review. The contact sheets are still
# MAGIC rendered: 300 counted cells (100 from the relaxed tail below Tol_41's 0.59, 80 night, 40 dusk, 40 dawn, 40 day) plus
# MAGIC extras (probes, flagged furniture, per-gate rejects, crops, vetoed). Looking at them is optional; the most useful ones
# MAGIC are `rejected_G6_*.png` (is the derived lower edge cutting fragments or curled calves?) and the tail sheets. If you do
# MAGIC review, copy `human_gate_template.json` → `human_gate.json`, set `status` and the `verdict` of every faulty cell
# MAGIC (unset = ok; `partial_minor` = clipped head/ear/tail, counts as ok) and re-run `finalize`: every BAD cell removes ±90 s
# MAGIC of admitted frames, and a FAILED review (your `failed`, any `wrong_object`, or > 3 BAD counted cells) unseals the
# MAGIC dataset whatever the widget says. `manifest.human_gate` records `required`, `present`, the file's `status` and the
# MAGIC recomputed `verdict`. With `require_human_gate=true` only a PASSED review of ≥ 300 planned cells seals.
# MAGIC
# MAGIC **`sam3_veto`** — the only GPU step: frames where SAM3's own `detect()` finds an animal-sized mask (≥ 0.30 × A_ref)
# MAGIC lying outside the four tracked masks (IoS < 0.30) are dropped and shown on `vetoed_*.png` with a red border. It is the
# MAGIC one automatic catch for "a calf with no mask" that lasted under ~2 s (invisible in the 50× review video). If the teacher
# MAGIC stage printed the SKIPPED banner, it ran on a CPU cluster and did nothing.
# MAGIC
# MAGIC **`teacher_detect`** — a free by-product of the veto pass (no DET-C training is planned; nothing downstream requires
# MAGIC it): SAM3 `detect()` masks Hungarian-matched to the four tracked masks and passed through the same set gates.
# MAGIC Should the mechanism control ever be trained, its DET-R arm must use `tracked_intersection` at ANY drop rate.
# MAGIC `mimicry_descriptive_only.hungarian_iou_tracked_vs_teacher` measures mimicry, never detector quality.
# MAGIC
# MAGIC **The manifest** (`Detector_COCO/<tag>/manifest.json`) is the contract: `source_days ==
# MAGIC forbidden_eval_days`, every constant with `derived_from` / `not_from`, `label_sources_available`, and
# MAGIC `sealed`. Tol_43 refuses an unsealed manifest; the Tol_44 evaluation refuses any eval day in
# MAGIC `forbidden_eval_days` and reads `constants["frag_lo[dayl]"]`, `constants["frag_lo[dark]"]` and `constants["over_hi"]`
# MAGIC for its GT trust filter.
