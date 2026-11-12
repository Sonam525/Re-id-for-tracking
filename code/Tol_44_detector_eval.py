# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N44 — detector replacement check on the tracker's own detector calls (04-22 / 04-23)
# MAGIC
# MAGIC **Purpose (design decision 2026-09-23).** One engineering question, so the
# MAGIC detector can be swapped **without re-tracking**: *on the frames where `Tol_03_track_v4` actually called
# MAGIC the SAM3 teacher detector, is the fine-tuned RF-DETR-Seg detector at least as good as that teacher,
# MAGIC when both are compared against the ground truth the reviewers already verified?* The same check is run for
# MAGIC the zero-training fallback `student_sam3` (the same `Sam3Model` with the distilled student backbone).
# MAGIC Output per challenger: **PASS / FAIL** plus every failed condition with its numbers. This is an
# MAGIC acceptance check, not a paper protocol (no pre-registration hashes, blind cards, plates, null arms or
# MAGIC mechanism control — all deliberately dropped). Its numbers are GT-referenced agreement on the reviewer-
# MAGIC verified days; they are **not** quotable as detector quality for the paper, and the
# MAGIC count columns are descriptive only (`n_det` counts instances, not animals).
# MAGIC
# MAGIC **Stages.** `extract` (CPU): parse the tracker's `events.jsonl` calls (`pass`, `pass_no_detections`,
# MAGIC `seeded`), join each to its stored mask row, stage only the frame-cache tars that hold a call, write
# MAGIC `frames.tar` + `calls.json`. `detect` (A100): run each arm on the identical cached JPEG, one model at a
# MAGIC time, write `dets_<arm>.json.gz`. `score` (CPU or local): GT construction, trust filter, metrics,
# MAGIC verdict, decision replay, `per_call.csv`, Metrics JSON, optional review sheet. `all` = detect + score.
# MAGIC
# MAGIC **Inputs.** `Tol/<eval_root>/<day>/{events.jsonl, masks_<clip_stem>.json.gz}` (default root
# MAGIC `Tracking_v4v2`, the production root the teacher detector served and whose identities the reviewers
# MAGIC verified); `Tol/Frames_5fps/<day>/<clip_stem>.tar` (members `<frame_idx:07d>.jpg`);
# MAGIC `Tol/Detector_COCO/<dataset_tag>/manifest.json` (leakage guard `forbidden_eval_days`, READ constants
# MAGIC `frag_lo[dayl]`, `frag_lo[dark]`, `over_hi`); `Tol/Detector_Checkpoints/<run_tag>/{checkpoint_best_total.pth,
# MAGIC train_summary.json}` (`op_threshold`, `resolution`, `checkpoint_best_total_sha256`,
# MAGIC `dataset_manifest_sha256`); `Tol/Checkpoints/best_backbone_student_v2.pt`; `facebook/sam3`.
# MAGIC
# MAGIC **Outputs.** `Tol/Detector_Eval/<eval_tag>/{frames.tar, calls.json, dets_<arm>.json.gz, per_call.csv,
# MAGIC timing_<stage>.json}`, `Tol/Metrics_detector_replacement_<eval_tag>.json` (next to every other
# MAGIC `Metrics_*.json` in `Tol/`), optional `Tol/Figures/detector_eval/<eval_tag>/disagreements_<k>.png`.
# MAGIC With `max_calls > 0` (SMOKE) every path uses `<eval_tag>_smoke`.
# MAGIC
# MAGIC **Assumptions (each asserted).** `eval_days ⊆ {20240422, 20240423}` and none of them is in the export
# MAGIC manifest's `forbidden_eval_days` / `source_days` (if 04-22 is added to training, only 04-23 remains);
# MAGIC the RF-DETR run was trained on exactly this manifest (sha256) and the checkpoint is the one its summary
# MAGIC describes (sha256); the manifest is sealed, or unsealed only through the one Tol_43 waiver (`below_floor` with
# MAGIC `allow_below_floor=true`, recorded as `seal_waiver` in that run's `train_summary.json`); each call joins to exactly one stored row by `round(t_sec, 2)`; that row's
# MAGIC `frame_idx` exists in the frame cache; masks are `[1296, 2304]`; the teacher re-run reproduces the logged
# MAGIC `n_det` on ≥ 85 % of passes in every phase (else the replay does not represent what the tracker saw);
# MAGIC on `Tracking_v4v2` the logs hold exactly the measured 290+1 (04-22) / 562+1 (04-23) calls and no accepted
# MAGIC repair; cached `dets_<arm>.json.gz` are reused (detect) and scored (score) only if their meta names the model
# MAGIC loaded now (rfdetr: run_tag, checkpoint sha256, op_threshold, resolution; student: ckpt name + sha256;
# MAGIC teacher/student: fp32 matmul precision `highest`, restored after `import rfdetr` sets TF32). SMOKE never resumes.
# MAGIC
# MAGIC **Tracker claims — VERIFIED in `Tol_03_track_v4.py` before relying on them.**
# MAGIC 1. *"The row is recorded from the tracker's own propagation before the pass acts"* — **true for mask
# MAGIC    content, false as worded.** The row is written at step (G) (`Tol_03:1356-1366`) **after** the pass (F)
# MAGIC    (`Tol_03:1342-1347`). Its masks are still the pre-pass propagation: `masks` comes from `track_frame`
# MAGIC    (`Tol_03:1204`) and the pass never writes into it (the only writer, `LAST_REPAIRS`, needs
# MAGIC    `partial_repair`; the 04-22/04-23 logs carry no `repaired` key and no partial/growth event — **asserted**
# MAGIC    at extract by `check_no_repairs`, which refuses any root with an accepted repair). But *which*
# MAGIC    slots are recorded follows the **post-pass** state: a slot the pass struck closed records `None`.
# MAGIC 2. *"A pass's newly opened slot first carries a mask on the next row"* — **false in one case.** A slot the
# MAGIC    pass reopens records `masks[s]` at the call row if that slot's old object was still in the pre-pass
# MAGIC    session (closed earlier by a strike, a merge resolution on this very frame, or an occluded timeout, but
# MAGIC    not yet rebuilt out): a **stale propagated mask of the closed object**, typically lying on the
# MAGIC    neighbour it merged into. Otherwise the call row holds `None` and the next row is the first carrying the
# MAGIC    new session's mask (session rebuilt inside the pass, `Tol_03:904-909`). **Implemented:** every slot in
# MAGIC    `opened` takes its GT from the next stored row whatever the call row holds; a call-row mask for such a
# MAGIC    slot is ignored and counted as `opened_stale_at_call`.
# MAGIC 3. *"The seeding branch runs `track_frame` on the seed frame itself"* — **true** (branch (B) opens the
# MAGIC    session at `Tol_03:1195`; `Tol_03:1204` runs `track_frame` on the same frame).
# MAGIC
# MAGIC Runnable locally as `TOL_SELFTEST=1 python Tol_44_detector_eval.py` (pure-python SELFTEST only).

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm" "scipy" "rfdetr==1.10.1" "pycocotools" "peft"
# MAGIC %pip uninstall -y opencv-python opencv-contrib-python
# MAGIC %pip install --no-deps --force-reinstall "opencv-python-headless==4.11.0.86"

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

import os, os.path as osp, sys, io, re, json, gzip, glob, time, math, csv, hashlib, tarfile, shutil, gc
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image

SELFTEST_ONLY = os.environ.get("TOL_SELFTEST") == "1"     # selftest mode: every Volume/model/GPU cell is skipped

IS_NOTEBOOK = True
try:
    dbutils
except NameError:
    IS_NOTEBOOK = False
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # local run: the Windows cp1252 console cannot print '⚠'

def P(name, default):                    # Tol_03:71-76 with the TOL_ env prefix
    if IS_NOTEBOOK:
        try: dbutils.widgets.text(name, default)
        except Exception: pass
        return (dbutils.widgets.get(name) or default)
    return os.environ.get("TOL_" + name.upper(), default)

# ---- widgets. REQUIRED widgets default "" and raise when empty (no default day/tag lists). ----
WIDGETS = [  # (name, default, required)
    ("stage",        "", True),  ("eval_tag", "", True), ("eval_days", "", True), ("dataset_tag", "", True),
    ("run_tag",      "", False),                               # REQUIRED for detect/all when rfdetr is an arm (checked below)
    ("eval_root",    "Tracking_v4v2", False),                  # the production root the teacher detector served (reviewer-verified identities)
    ("arms",         "teacher,rfdetr,student_sam3", False),
    ("student_ckpt", "best_backbone_student_v2.pt", False),    # the production student is v2
    ("force",        "false", False),
    ("max_calls",    "0", False),                              # >0 = SMOKE (suffix _smoke on every output path)
    ("review_sheet", "true", False),                           # optional disagreement sheet at stage=score (nothing depends on it)
]
W = {}
for _n, _d, _req in WIDGETS:
    W[_n] = (P(_n, _d) or "").strip()
    if _req and not W[_n] and not SELFTEST_ONLY:
        raise RuntimeError(f"widget '{_n}' is REQUIRED and empty (no default day/tag lists)")

STAGE       = W["stage"].lower()
EVAL_DAYS   = [d for d in re.split(r"[,\s;]+", W["eval_days"]) if d]
DATASET_TAG, RUN_TAG, EVAL_ROOT = W["dataset_tag"], W["run_tag"], W["eval_root"]
ARMS        = [a.strip().lower() for a in W["arms"].split(",") if a.strip()]
STUDENT_CKPT = W["student_ckpt"]
FORCE       = W["force"].lower() == "true"
MAX_CALLS   = int(W["max_calls"] or 0)
SMOKE       = MAX_CALLS > 0
REVIEW_SHEET = W["review_sheet"].lower() == "true"
EVAL_TAG    = W["eval_tag"] + ("_smoke" if SMOKE else "")
KNOWN_ARMS  = ("teacher", "rfdetr", "student_sam3")
CHALLENGERS = [a for a in ARMS if a != "teacher"]
if not SELFTEST_ONLY:
    if STAGE not in ("extract", "detect", "score", "all"):
        raise RuntimeError(f"stage must be extract|detect|score|all, got {STAGE!r}")
    if "teacher" not in ARMS or any(a not in KNOWN_ARMS for a in ARMS) or not CHALLENGERS:
        raise RuntimeError(f"arms must include 'teacher' and at least one of rfdetr/student_sam3, all from {KNOWN_ARMS}; got {ARMS}")
    if STAGE in ("detect", "score", "all") and "rfdetr" in ARMS and not RUN_TAG:
        raise RuntimeError("widget 'run_tag' is REQUIRED for stage detect/score/all when rfdetr is an arm "
                           "(score checks the cached rfdetr detections against that run's checkpoint sha256 / op_threshold)")
    if STUDENT_CKPT != "best_backbone_student_v2.pt":
        print(f"⚠ student_ckpt={STUDENT_CKPT!r} is not the production v2 student")
    if EVAL_ROOT != "Tracking_v4v2":
        print(f"⚠ eval_root={EVAL_ROOT!r}: the verified production root is Tracking_v4v2 — the call counts and the reviewers' GT verdict refer to it")
    if SMOKE:
        print(f"⚠⚠ SMOKE: max_calls={MAX_CALLS} — every output goes to eval_tag {EVAL_TAG!r}; the verdict is NOT a decision")
    for k, v in W.items(): print(f"  widget {k:13s} = {v!r}")

# ---- paths — always os.path.join, never string concatenation ----
VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
EVAL_DIR     = osp.join(TOL, "Detector_Eval", EVAL_TAG)
FIG_DIR      = osp.join(TOL, "Figures", "detector_eval", EVAL_TAG)
COCO_DIR     = osp.join(TOL, "Detector_COCO", DATASET_TAG)
CKPT_DIR     = osp.join(TOL, "Detector_Checkpoints", RUN_TAG)
FRAMES_DIR   = osp.join(TOL, "Frames_5fps")
METRICS_PATH = osp.join(TOL, f"Metrics_detector_replacement_{EVAL_TAG}.json")
LOCAL        = "/local_disk0/tmp/n44"
SAM3_ID      = "facebook/sam3"

# ---- constants ledger (each names its provenance; none is fitted on 04-22/04-23 except A_ref_eval, below) ----
H_FULL, W_FULL = 1296, 2304            # frame/mask size: Tol ch01 camera, all nine days (Frames_5fps index files, 04-18..04-28)
N_CALVES      = 4                      # Tol_03:127; pen population verified 2026-07-27 (00d_tol_detect_probe, 9 days)
MIN_MASK_PX   = 8000                   # Tol_03:225 incumbent post-chain (set from the 2026-07-27 Tol ch01 probe, not a test-day fit)
DET_SCORE_MIN, DET_TEXT = 0.40, "calf" # Tol_03:226 incumbent contract (same provenance)
DEDUPE_IOS    = 0.50                   # Tol_03:227 incumbent dedupe (same provenance)
DET_TOPK      = N_CALVES + 2           # Tol_03:727 detect(pil, topk=N_CALVES + 2) at every pass and seeding (Tol_03:1167)
ASSOC_MIN     = 0.30                   # Tol_03:139 pass association floor (code constant; not fitted on any day)
STRIKE_MAX    = 0.10                   # Tol_03:140 strike ceiling (code constant)
FRESH_ROWS    = 10                     # Tol_03:128 FRESH_SEC 2.0 s × 5 fps: look-back for a struck slot's pre-pass mask (replay only)
FOUND_IOU     = 0.50                   # a GT calf is found at matched IoU ≥ 0.50 (rule, not derived from a day)
SPLIT_IOS     = 0.50                   # det lies mostly inside a GT calf (rule)
EXTRA_IOS     = 0.30                   # = the Tol_42 G11 veto IoS (rule, training-day protocol)
EXTRA_AREA_FRAC = 0.30                 # = the Tol_42 G11 veto area as a ratio of the reference (rule)
WHOLE_FRAC    = 0.50                   # whole-animal unmatched det (mirrors Tol_03:162 REPAIR_AREA_LO; code constant)
GT_IOS_MAX    = 0.05                   # trust filter: a trusted GT mask overlaps no other GT mask (rule)
A_REF_STRIDE_ROWS = 5                  # Tol_40:111 STRIDE=5 rows at 5 fps = 1 Hz (the A_ref definition the manifest uses)
TOL_FRAC      = 0.01                   # tolerance = max(1, 1 % of calves / of F4 calls) (protocol rule)
IOU_TOL       = 0.02                   # verdict condition 4 (protocol rule)
FAITH_WARN, FAITH_RAISE = 0.95, 0.85   # replay faithfulness (protocol rule)
N_BOOT, SEED  = 2000, 0                # clip-cluster bootstrap resamples; fixed seed (not a fitted value)
SHEET_MAX, SHEET_PER_PAGE, SHEET_W = 24, 6, 768   # review sheet (≤ 24 calls); panel width (display only)
STRETCH_LO, STRETCH_HI = 1, 99         # display-only luminance stretch for dusk/night panels
COLORS_RGB = [(255, 60, 60), (60, 220, 60), (40, 160, 255), (255, 220, 40)]   # Tol_17:105 BGR COLORS in RGB (slot 0 red/1 green/2 blue/3 yellow)
ALLOWED_EVAL_DAYS = ("20240422", "20240423")   # the two days with reviewer-verified identities
EXPECTED_CALLS = {"20240422": {"pass": 290, "seeded": 1}, "20240423": {"pass": 562, "seeded": 1}}   # measured on Tracking_v4v2 logs 2026-09-23
CALL_TYPES    = ("pass", "pass_no_detections", "seeded")      # Tol_03:730, 910, 1198 — every detect() call the tracker logs
PASS_KEYS     = ("clip", "t", "forced", "n_det", "tracked", "matched", "struck", "opened", "refreshed", "rebuilt")   # Tol_03:910-913
POOL = {"dayl": {"day", "dawn"}, "dark": {"dusk", "night"}}   # daylight vs dark phase pools
STRATA = ("all", "dark", "night", "dusk", "dawn", "day")       # verdict on all + dark; the rest descriptive

# COMMAND ----------

# MAGIC %md ## 1 · Pure logic (no dbutils, no GPU, no network) — everything the SELFTEST exercises

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

def rle_area(r):                        # area without decoding
    return int(sum(r["counts"][1::2]))

# ---- run arithmetic on python lists (same runs as runs_of; lists make inter() ~20× faster than numpy scalars) ----
def runs_l(rle):
    s, e, a = runs_of(rle)
    return s.tolist(), e.tolist(), a

def inter_fast(A, B):
    """inter() with an O(1) reject when the flat [start, end) spans do not overlap."""
    if not A[0] or not B[0] or A[0][0] >= B[1][-1] or B[0][0] >= A[1][-1]: return 0
    return inter(A, B)

def iou_r(A, B):
    i = inter_fast(A, B); u = A[2] + B[2] - i
    return i / u if u else 0.0

def ios_r(A, B):
    m = min(A[2], B[2]); return inter_fast(A, B) / m if m else 0.0

def frac_in(A, B):
    """Fraction of A that lies inside B (IoS det → GT for the 'split' condition)."""
    return inter_fast(A, B) / A[2] if A[2] else 0.0

def runs_union(lst):
    """Union of several run triples -> one merged run triple."""
    iv = sorted((s, e) for R in lst for s, e in zip(R[0], R[1]))
    us, ue = [], []
    for s, e in iv:
        if ue and s <= ue[-1]: ue[-1] = max(ue[-1], e)
        else: us.append(s); ue.append(e)
    return us, ue, int(sum(e - s for s, e in zip(us, ue)))

# ---- time, phases, keys ----
RXT = re.compile(r"-(\d{8})-(\d{6})-")          # Tol_40:80-83
def clip_start(v):
    m = RXT.search(v.replace("_", "-"))
    if m is None: raise RuntimeError(f"cannot parse clip start from {v!r}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

def phase4(w):                       # the one phase function (night/dawn/dusk/day)
    if w.hour >= 21 or w.hour < 6: return "night"
    if w.hour < 8:                 return "dawn"
    if w.hour == 20:               return "dusk"
    return "day"
def pool_of(ph): return "dayl" if ph in POOL["dayl"] else "dark"
def in_stratum(ph, st): return st == "all" or (st == "dark" and ph in POOL["dark"]) or st == ph

def clip_stem_of(vname):             # Tol_03:1102
    if not vname.endswith(".mp4"): raise RuntimeError(f"event clip {vname!r} is not an .mp4 name")
    return vname[:-4].replace(" ", "_")

def tkey(x): return round(float(x), 2)          # events log t = round(t_sec, 2) (Tol_03:730/910/1198); rows t_sec = round(t_sec, 2) (Tol_03:1364)
def fkey(clip_stem, frame_idx): return f"{clip_stem}:{int(frame_idx):07d}"
def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

# ---- leakage guard + manifest constants ----
def check_eval_days(days, manifest):
    """Refuse any eval day outside the verified pair and any day the export used or forbids."""
    if not days: raise RuntimeError("eval_days is empty")
    if len(set(days)) != len(days): raise RuntimeError(f"eval_days has duplicates: {days}")
    if "forbidden_eval_days" not in manifest: raise RuntimeError("manifest has no forbidden_eval_days — not a Tol_42 manifest")
    forb = set(manifest["forbidden_eval_days"]) | set(manifest.get("source_days", []))
    for d in days:
        if d not in ALLOWED_EVAL_DAYS:
            raise RuntimeError(f"eval day {d} is not one of {ALLOWED_EVAL_DAYS} (the days with reviewer-verified GT)")
        if d in forb:
            raise RuntimeError(f"eval day {d} is a training source of dataset {manifest.get('tag')} "
                               f"(forbidden_eval_days={sorted(forb)}) — train/test leakage")
    return sorted(days)

def seal_admissible(manifest, manifest_sha, ts):
    """A sealed manifest is admitted (returns None). An UNSEALED one is admitted only when the Tol_43 run under
    evaluation trained on it through the single waiver Tol_43 allows (manifest_guards: below_floor the sole
    unmet term and allow_below_floor=true), which that run records in train_summary.json as manifest_sealed=false,
    seal_waiver=['below_floor'] for THIS manifest's sha256. Returns a ⚠ note in that case; raises otherwise."""
    if manifest.get("sealed") is True: return None
    if not ts:
        raise RuntimeError(f"dataset {manifest.get('tag')} manifest is not sealed and no Tol_43 train_summary.json is given — "
                           "Tol_43 trains on an unsealed manifest only under allow_below_floor=true and records it; set run_tag to that run")
    if ts.get("dataset_manifest_sha256") != manifest_sha:
        raise RuntimeError(f"manifest is not sealed and train_summary.json was trained on manifest "
                           f"{str(ts.get('dataset_manifest_sha256'))[:12]}, not {manifest_sha[:12]}")
    if ts.get("manifest_sealed") is not False or list(ts.get("seal_waiver") or []) != ["below_floor"]:
        raise RuntimeError(f"manifest is not sealed and run {ts.get('run_tag')} records no below_floor waiver "
                           f"(manifest_sealed={ts.get('manifest_sealed')!r}, seal_waiver={ts.get('seal_waiver')!r}) — refused")
    if not bool((manifest.get("counts") or {}).get("below_floor")):
        raise RuntimeError("manifest is not sealed but counts.below_floor is false — the recorded waiver does not explain the missing seal")
    return (f"manifest {manifest.get('tag')} is UNSEALED: run {ts.get('run_tag')} trained on it under Tol_43 allow_below_floor=true "
            f"(train_full={(manifest.get('counts') or {}).get('train_full')} < min_train); evaluated with that waiver recorded")

def manifest_const(manifest, key):
    """READ a training-derived constant as Tol_42 writes it: constants['frag_lo[dark]'] = {'value': ..., 'provenance': ...}."""
    C = manifest.get("constants") or {}
    if key not in C: raise RuntimeError(f"manifest.constants lacks {key!r} (have {sorted(C)[:30]})")
    v = C[key]; v = v["value"] if isinstance(v, dict) and "value" in v else v
    if v is None: raise RuntimeError(f"manifest.constants[{key!r}] is None")
    return float(v)

# ---- events -> calls; call -> stored row ----
def parse_calls(events, day):
    """events: parsed events.jsonl lines of ONE day dir. Returns the detector calls, sorted by wall, each with
    the logged strike slots at the same (clip, t) (strike events are separate records, Tol_03:754)."""
    strikes = defaultdict(set)
    for e in events:
        if e.get("type") == "strike": strikes[(e["clip"], tkey(e["t"]))].add(int(e["slot"]))
    calls = []
    for e in events:
        typ = e.get("type")
        if typ not in CALL_TYPES: continue
        need = PASS_KEYS if typ == "pass" else (("clip", "t", "slots") if typ == "seeded" else ("clip", "t"))
        miss = [k for k in need if k not in e]
        if miss: raise RuntimeError(f"{day}: {typ} event lacks {miss}: {e}")
        stem = clip_stem_of(e["clip"]); t = tkey(e["t"])
        w = clip_start(stem) + timedelta(seconds=t)
        calls.append(dict(call_id=f"{day}|{stem}|{t:.2f}|{typ}", day=day, clip=e["clip"], clip_stem=stem, t=t,
                          wall=w.isoformat(), phase4=phase4(w), type=typ, event=e,
                          strike_slots=sorted(strikes.get((e["clip"], t), ()))))
    ids = Counter(c["call_id"] for c in calls)
    dup = [k for k, n in ids.items() if n > 1]
    if dup: raise RuntimeError(f"{day}: duplicate detector calls {dup[:5]}")
    calls.sort(key=lambda c: (c["wall"], CALL_TYPES.index(c["type"])))
    return calls

REPAIR_TYPES = ("partial_repair", "growth_repair")   # Tol_03:862 accepted-repair event types (the *_rejected/_skipped ones write nothing)
def check_no_repairs(events, day):
    """GT independence (header claim 1): an accepted repair copies the teacher's own detection mask into the stored
    row (Tol_03:887, 1345-1347), so that row's mask would be circular GT. Refuse any root whose logs show one."""
    rep = [e for e in events if e.get("type") in REPAIR_TYPES
           or (e.get("type") == "pass" and e.get("repaired"))]
    if rep:
        raise RuntimeError(f"{day}: {len(rep)} repair event(s) (first: {rep[0]}) — the stored rows at those passes carry "
                           f"the teacher's detection mask, so they cannot be GT for a teacher-vs-challenger check. "
                           f"Use a root tracked with partial_repair=false (Tracking_v4v2).")

def meta_mismatch(meta, ident):
    """Keys of `ident` whose value differs from the cached dets meta (the cached detections are from another model)."""
    return {k: (meta.get(k), v) for k, v in ident.items() if meta.get(k) != v}

def index_rows(rows):
    by = defaultdict(list)
    for i, r in enumerate(rows): by[tkey(r["t_sec"])].append(i)
    return by

def join_event_row(rows_by_t, t):
    """Exact round(t_sec, 2) join. Returns (row position, None) or (None, 'missing'|'duplicate')."""
    idx = rows_by_t.get(tkey(t), [])
    if len(idx) == 1: return idx[0], None
    return None, ("missing" if not idx else "duplicate")

def row_masks(row):
    return [(row.get("masks") or {}).get(str(s)) for s in range(N_CALVES)]

def prev_masks_for(rows, pos, slots, lookback=FRESH_ROWS):
    """For each slot: the most recent non-None mask in the `lookback` rows before `pos` (decision replay of a struck slot)."""
    out = {}
    for s in slots:
        for q in range(pos - 1, max(-1, pos - 1 - lookback), -1):
            m = row_masks(rows[q])[s]
            if m is not None: out[str(s)] = m; break
    return out

# ---- GT at each call (with the verified tracker behaviour in the header) ----
def build_gt(call):
    """-> (gt, unscorable, stale). gt: [{slot, rle, kind, seeded_by_teacher, src}]; kind 'tracked' (the call row's
    own propagation) or 'reopened' (a teacher-seeded mask: next row after a pass opened the slot, or the seed
    frame's row). unscorable: [{slot, reason}]. stale: opened slots whose call row held a (stale) mask."""
    ev, typ = call["event"], call["type"]
    row, nxt = call["row_masks"], call.get("next_masks")
    gt, uns, stale = [], [], []
    if typ == "seeded":
        seeded = {int(s) for s in ev["slots"]}
        for s in range(N_CALVES):
            if s in seeded:
                if row[s] is not None:
                    gt.append(dict(slot=s, rle=row[s], kind="reopened", seeded_by_teacher=True, src="call"))
                else: uns.append(dict(slot=s, reason="seeded_no_mask_at_call"))
            elif row[s] is not None:
                gt.append(dict(slot=s, rle=row[s], kind="tracked", seeded_by_teacher=False, src="call"))
            else: uns.append(dict(slot=s, reason="absent"))
        return gt, uns, stale
    opened = {int(s) for s in ev.get("opened", [])}
    for s in range(N_CALVES):
        if s in opened:
            if row[s] is not None: stale.append(s)          # stale propagation of the closed object: never GT
            if nxt is None: uns.append(dict(slot=s, reason="reopened_no_next_row"))
            elif nxt[s] is None: uns.append(dict(slot=s, reason="reopened_no_mask_next_row"))
            else: gt.append(dict(slot=s, rle=nxt[s], kind="reopened", seeded_by_teacher=True, src="next"))
        elif row[s] is not None:
            gt.append(dict(slot=s, rle=row[s], kind="tracked", seeded_by_teacher=False, src="call"))
        else:
            uns.append(dict(slot=s, reason="absent"))
    return gt, uns, stale

def stored_union_runs(call):
    """Union of every stored mask at the call (call row, including stale ones, + next-row masks of opened slots)."""
    ms = [m for m in call["row_masks"] if m is not None]
    if call["type"] != "seeded" and call.get("next_masks") is not None:
        ms += [call["next_masks"][int(s)] for s in call["event"].get("opened", []) if call["next_masks"][int(s)] is not None]
    return runs_union([runs_l(m) for m in ms]) if ms else ([], [], 0)

def trust_filter(gt, a_ref, frag_lo, over_hi, ios_max=GT_IOS_MAX):
    """Marks each GT dict trusted iff area/a_ref ∈ [frag_lo, over_hi] and IoS with every other GT mask < ios_max.
    Returns the run triples (same order)."""
    R = [runs_l(g["rle"]) for g in gt]
    for i, g in enumerate(gt):
        ratio = R[i][2] / a_ref
        ios = max([ios_r(R[i], R[j]) for j in range(len(gt)) if j != i], default=0.0)
        why = []
        if ratio < frag_lo: why.append("area_below_frag_lo")
        if ratio > over_hi: why.append("area_above_over_hi")
        if ios >= ios_max: why.append("overlaps_other_gt")
        g.update(area=R[i][2], ratio=round(ratio, 4), max_ios=round(ios, 4), trusted=not why, untrusted_reasons=why)
    return R

# ---- scoring one arm on one call ----
def score_call(det_runs, gt_runs, union_runs, a_ref, f4):
    nD, nG = len(det_runs), len(gt_runs)
    M = np.zeros((nD, nG))
    for i in range(nD):
        for j in range(nG): M[i, j] = iou_r(det_runs[i], gt_runs[j])
    found, iou, det_of = [False] * nG, [0.0] * nG, [None] * nG
    if nD and nG:
        ri, cj = linear_sum_assignment(-M)
        for i, j in zip(ri, cj):
            if M[i, j] >= FOUND_IOU: found[j], iou[j], det_of[j] = True, float(M[i, j]), int(i)
    split = [sum(1 for i in range(nD) if frac_in(det_runs[i], gt_runs[j]) >= SPLIT_IOS) >= 2 for j in range(nG)]
    extra_idx = None
    if f4:
        extra_idx = [i for i in range(nD) if ios_r(det_runs[i], union_runs) < EXTRA_IOS
                     and det_runs[i][2] >= EXTRA_AREA_FRAC * a_ref]
    return dict(found=found, iou=iou, det_of=det_of, split=split,
                n_extra=(len(extra_idx) if extra_idx is not None else None), extra_idx=extra_idx, n_kept=nD)

# ---- decision replay (Tol_03:723-759 association + strikes) ----
def replay_tracks(call):
    """Tracked masks the pass associated against, reconstructed from stored rows: slots with a call-row mask that
    the pass did not open (pre-pass propagation), plus struck slots (the call row's mask if the slot was also
    reopened — that mask IS its pre-pass propagation — else its last mask within FRESH_ROWS earlier rows)."""
    ev = call["event"]; row = call["row_masks"]; prev = call.get("prev_masks") or {}
    opened = {int(s) for s in ev.get("opened", [])}; struck = {int(s) for s in ev.get("struck", [])}
    trk = {}
    for s in range(N_CALVES):
        if s in struck:
            m = row[s] if row[s] is not None else prev.get(str(s))
            if m is not None: trk[s] = m
        elif s in opened: continue
        elif row[s] is not None: trk[s] = row[s]
    return trk

def replay_pass(trk_runs, det_runs, a_ref):
    """trk_runs {slot: runs}; det_runs [runs]. Tol_03: no detections -> early return (no strikes, Tol_03:729-731)."""
    if not det_runs: return dict(matched=[], strikes=[], n_whole_unmatched=0, no_detections=True)
    slots = sorted(trk_runs)
    S = np.zeros((len(slots), len(det_runs)), np.float32)
    for a, s in enumerate(slots):
        for b, d in enumerate(det_runs): S[a, b] = max(iou_r(trk_runs[s], d), ios_r(trk_runs[s], d))
    matched, mdet = [], set()
    if slots:
        ri, cj = linear_sum_assignment(1.0 - S)
        for a, b in zip(ri, cj):
            if S[a, b] >= ASSOC_MIN: matched.append(slots[a]); mdet.add(int(b))
    strikes = [s for a, s in enumerate(slots) if s not in matched and float(S[a].max()) < STRIKE_MAX]
    n_whole = sum(1 for b, d in enumerate(det_runs) if b not in mdet and d[2] >= WHOLE_FRAC * a_ref)
    return dict(matched=sorted(matched), strikes=sorted(strikes), n_whole_unmatched=int(n_whole), no_detections=False)

def replay_key(r): return (tuple(r["matched"]), tuple(r["strikes"]), r["n_whole_unmatched"])

# ---- records, aggregation, verdict ----
def call_record(call, dets_by_arm, a_ref_eval, frag_lo, over_hi, arms):
    """dets_by_arm[arm][fkey] = {'kept': [{'rle', 'score', 'area'}], ...}. Pure; raises when a call has no detection row."""
    k = fkey(call["clip_stem"], call["frame_idx"]); a_ref = a_ref_eval[call["day"]]
    gt, uns, stale = build_gt(call)
    R = trust_filter(gt, a_ref, frag_lo[pool_of(call["phase4"])], over_hi)
    tg = [(g, r) for g, r in zip(gt, R) if g["trusted"]]
    f4 = len({g["slot"] for g, _ in tg}) == N_CALVES
    union = stored_union_runs(call)
    rec = dict(call_id=call["call_id"], day=call["day"], clip_stem=call["clip_stem"], frame_idx=call["frame_idx"],
               t=call["t"], wall=call["wall"], phase4=call["phase4"], type=call["type"], fkey=k, f4=f4,
               gt=[dict(slot=g["slot"], kind=g["kind"], seeded_by_teacher=g["seeded_by_teacher"], ratio=g["ratio"]) for g, _ in tg],
               untrusted=[dict(slot=g["slot"], kind=g["kind"], reasons=g["untrusted_reasons"]) for g in gt if not g["trusted"]],
               unscorable=uns, stale=stale, arms={}, replay={})
    gtr = [r for _, r in tg]
    trk = {s: runs_l(m) for s, m in replay_tracks(call).items()} if call["type"] != "seeded" else None
    for arm in arms:
        rows = dets_by_arm[arm]
        if k not in rows: raise RuntimeError(f"dets_{arm} has no row for {k} ({call['call_id']}) — re-run stage detect")
        dr = [runs_l(d["rle"]) for d in rows[k]["kept"]]
        rec["arms"][arm] = score_call(dr, gtr, union, a_ref, f4)
        if call["type"] == "seeded":
            # the tracker seeds from the N_CALVES LARGEST kept detections only (Tol_03 seeding branch: dets[:N_CALVES] /
            # stable[:N_CALVES]); kept lists are area-descending (post_chain / detect()), so the seed check scores dr[:4]
            rec["arms"][arm]["found_seed4"] = score_call(dr[:N_CALVES], gtr, union, a_ref, f4)["found"]
        if trk is not None: rec["replay"][arm] = replay_pass(trk, dr, a_ref)
    if trk is not None:
        ev = call["event"]
        rec["logged"] = dict(matched=sorted(int(s) for s in ev.get("matched", {})), strikes=list(call["strike_slots"]),
                             tracked=ev.get("tracked"), n_det=(int(ev["n_det"]) if call["type"] == "pass" else 0))
        rec["trk_slots"] = sorted(trk)
    return rec

def compare(recs, arm, st, ref="teacher"):
    """Pooled counts for one challenger vs the reference arm on one stratum."""
    sel = [r for r in recs if in_stratum(r["phase4"], st)]
    c = dict(stratum=st, n_calls=len(sel), N=0, F4=0, miss_T=0, miss_A=0, split_T=0, split_A=0, extra_T=0, extra_A=0,
             miss_kind_T={"tracked": 0, "reopened": 0}, miss_kind_A={"tracked": 0, "reopened": 0},
             N_kind={"tracked": 0, "reopened": 0}, disc_T_only=0, disc_A_only=0, n_both=0,
             med_T=None, med_A=None, n_seed=0, seed_fail_T=[], seed_fail_A=[], kept_T=0, kept_A=0)
    iT, iA = [], []
    for r in sel:
        T, A = r["arms"][ref], r["arms"][arm]
        c["kept_T"] += T["n_kept"]; c["kept_A"] += A["n_kept"]
        for j, g in enumerate(r["gt"]):
            c["N"] += 1; c["N_kind"][g["kind"]] += 1
            fT, fA = T["found"][j], A["found"][j]
            if not fT: c["miss_T"] += 1; c["miss_kind_T"][g["kind"]] += 1
            if not fA: c["miss_A"] += 1; c["miss_kind_A"][g["kind"]] += 1
            c["split_T"] += int(T["split"][j]); c["split_A"] += int(A["split"][j])
            if fT and not fA: c["disc_T_only"] += 1
            if fA and not fT: c["disc_A_only"] += 1
            if fT and fA: iT.append(T["iou"][j]); iA.append(A["iou"][j])
        if r["f4"]:
            c["F4"] += 1; c["extra_T"] += T["n_extra"]; c["extra_A"] += A["n_extra"]
        if r["type"] == "seeded":
            c["n_seed"] += 1
            if not all(T.get("found_seed4", T["found"])): c["seed_fail_T"].append(r["call_id"])
            if not all(A.get("found_seed4", A["found"])): c["seed_fail_A"].append(r["call_id"])
    c["n_both"] = len(iT)
    if iT: c["med_T"], c["med_A"] = float(np.median(iT)), float(np.median(iA))
    return c

def verdict_rule(cmp_by_stratum, strata=("all", "dark")):
    """PASS iff conditions 1-5 hold in BOTH strata. An empty stratum cannot certify -> FAIL."""
    fails = []
    for st in strata:
        c = cmp_by_stratum[st]
        if c["N"] == 0:
            fails.append(f"[{st}] no trusted GT calves — the stratum cannot certify the replacement"); continue
        tn, tf = max(1.0, TOL_FRAC * c["N"]), max(1.0, TOL_FRAC * c["F4"])
        if c["miss_A"] - c["miss_T"] > tn:
            fails.append(f"[{st}] (1) misses {c['miss_A']} vs teacher {c['miss_T']}: +{c['miss_A'] - c['miss_T']} > tolerance {tn:g} (N={c['N']})")
        if c["split_A"] - c["split_T"] > tn:
            fails.append(f"[{st}] (2) splits {c['split_A']} vs teacher {c['split_T']}: +{c['split_A'] - c['split_T']} > tolerance {tn:g} (N={c['N']})")
        if c["extra_A"] - c["extra_T"] > tf:
            fails.append(f"[{st}] (3) extras {c['extra_A']} vs teacher {c['extra_T']}: +{c['extra_A'] - c['extra_T']} > tolerance {tf:g} (F4={c['F4']})")
        if c["n_both"] == 0:
            fails.append(f"[{st}] (4) no calf found by both arms — mask agreement cannot be compared")
        elif c["med_A"] < c["med_T"] - IOU_TOL:
            fails.append(f"[{st}] (4) median matched IoU {c['med_A']:.4f} < teacher {c['med_T']:.4f} − {IOU_TOL} (n={c['n_both']})")
        if c["seed_fail_A"]:
            fails.append(f"[{st}] (5) misses a trusted calf at seeded call(s) {c['seed_fail_A']}")
    return (not fails), fails

def clip_bootstrap(per_clip, stat, n_boot=N_BOOT, seed=SEED):
    """per_clip {clip: payload}; stat(list of payloads) -> float. Clip-cluster percentile CI -> (point, lo, hi)."""
    clips = sorted(per_clip)
    if not clips: return (float("nan"), float("nan"), float("nan"))
    pt = stat([per_clip[c] for c in clips])
    rng = np.random.default_rng(seed); k = len(clips); d = []
    for _ in range(n_boot):
        d.append(stat([per_clip[clips[i]] for i in rng.integers(0, k, k)]))
    d = np.asarray(d, float)
    return (float(pt), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)))

def diff_cis(recs, arm, st, ref="teacher", n_boot=N_BOOT, seed=SEED):
    """Clip-cluster bootstrap CIs of arm − teacher for misses, splits, extras and median matched IoU."""
    per = defaultdict(lambda: dict(miss=0, split=0, extra=0, pairs=[]))
    for r in recs:
        if not in_stratum(r["phase4"], st): continue
        T, A, p = r["arms"][ref], r["arms"][arm], per[r["clip_stem"]]
        for j in range(len(r["gt"])):
            p["miss"] += int(not A["found"][j]) - int(not T["found"][j])
            p["split"] += int(A["split"][j]) - int(T["split"][j])
            if T["found"][j] and A["found"][j]: p["pairs"].append((T["iou"][j], A["iou"][j]))
        if r["f4"]: p["extra"] += A["n_extra"] - T["n_extra"]
    per = dict(per)
    def med_diff(ps):
        pr = [x for q in ps for x in q["pairs"]]
        return float(np.median([a for _, a in pr]) - np.median([t for t, _ in pr])) if pr else float("nan")
    out = {m: clip_bootstrap(per, (lambda ps, m=m: float(sum(q[m] for q in ps))), n_boot, seed) for m in ("miss", "split", "extra")}
    out["median_iou"] = clip_bootstrap(per, med_diff, n_boot, seed)
    out["n_clips"] = len(per)
    return out

def ndet_faithfulness(recs):
    """Teacher re-run count vs the logged n_det of every pass (replay faithfulness), by phase4 and overall."""
    by = defaultdict(lambda: [0, 0])
    for r in recs:
        if r["type"] not in ("pass", "pass_no_detections"): continue
        ok = int(r["arms"]["teacher"]["n_kept"] == r["logged"]["n_det"])
        for k in ("all", r["phase4"]): by[k][0] += ok; by[k][1] += 1
    return {k: dict(agree=a, n=n, rate=(a / n if n else float("nan"))) for k, (a, n) in by.items()}

def faithfulness_gate(rates, smoke):
    """Warn below FAITH_WARN in any phase; RAISE only when the POOLED ('all') rate is below FAITH_RAISE (SMOKE: warn only).
    Revised 2026-09-24: per-phase strata hold only ~60-80 passes (dusk/night/dawn) and night instance counts flip
    near the 0.40 score gate / 8,000 px floor between the tracker's mp4 decode and the cache JPEG, so a per-phase
    raise would void the whole GPU run on a stratum-level count wobble while the replay is faithful overall. Per-phase
    shortfalls stay in the returned status and in the Metrics JSON."""
    warn = sorted(k for k, v in rates.items() if v["n"] and v["rate"] < FAITH_WARN)
    bad = sorted(k for k, v in rates.items() if v["n"] and v["rate"] < FAITH_RAISE)
    for k in warn: print(f"⚠ replay faithfulness {k}: {rates[k]['agree']}/{rates[k]['n']} = {rates[k]['rate']:.3f} < {FAITH_WARN}")
    if bad and not smoke and "all" in bad:
        raise RuntimeError(f"replay faithfulness below {FAITH_RAISE} pooled: the cache-JPEG replay does not represent "
                           f"what the tracker saw ({ {k: round(rates[k]['rate'], 3) for k in bad} })")
    if bad and "all" not in bad: print(f"⚠⚠ per-phase replay faithfulness below {FAITH_RAISE} in {bad} (pooled OK) — reported, not raised")
    if bad and smoke: print(f"⚠⚠ SMOKE: faithfulness below {FAITH_RAISE} in {bad}")
    return dict(warn=warn, below_raise=bad)

def replay_faithfulness(recs):
    """Teacher decision replay vs the logged pass outcome (reported; not a gate)."""
    n = m_ok = s_ok = t_ok = t_n = 0
    for r in recs:
        if "logged" not in r: continue
        rp, lg = r["replay"]["teacher"], r["logged"]; n += 1
        m_ok += int(rp["matched"] == lg["matched"]); s_ok += int(rp["strikes"] == lg["strikes"])
        if r["type"] == "pass" and lg["tracked"] is not None:
            t_n += 1; t_ok += int(len(r["trk_slots"]) == int(lg["tracked"]))
    f = lambda a, b: (a / b if b else float("nan"))
    return dict(n=n, matched_same=f(m_ok, n), strikes_same=f(s_ok, n), tracked_count_same=f(t_ok, t_n), n_tracked_checked=t_n)

def replay_agreement(recs, arm, ref="teacher"):
    n = same = 0; diffs = []
    for r in recs:
        if arm not in r["replay"]: continue
        n += 1; a, t = r["replay"][arm], r["replay"][ref]
        if replay_key(a) == replay_key(t): same += 1
        else: diffs.append(dict(call_id=r["call_id"], wall=r["wall"], phase4=r["phase4"], teacher=t, arm=a))
    return dict(n=n, same=same, share_same=(same / n if n else float("nan")), differing=diffs)

def smoke_pick(calls, n):
    """Deterministic SMOKE subset that still exercises every branch: seeded calls, a pass that opened a slot, one call
    per phase4, then the rest in wall order."""
    order, seen = [], set()
    def add(i):
        if i not in seen and len(order) < n: seen.add(i); order.append(i)
    for i, c in enumerate(calls):
        if c["type"] == "seeded": add(i)
    for i, c in enumerate(calls):
        if c["type"] == "pass" and c["event"].get("opened"): add(i); break
    for ph in ("night", "dusk", "dawn", "day"):
        for i, c in enumerate(calls):
            if c["phase4"] == ph: add(i); break
    for i in range(len(calls)): add(i)
    return [calls[i] for i in sorted(order)]

# ---- the incumbent post-chain, pure (reproduces Tol_03.detect() ordering) — used by the RF-DETR arm ----
def post_chain(cands, topk=DET_TOPK, dedupe=DEDUPE_IOS, min_px=MIN_MASK_PX):
    """cands: list of (bool mask, score) already past the score gate. Returns (kept list, n_floor_removed)."""
    c2 = [(m, float(s)) for m, s in cands if m.sum() >= min_px]
    n_floor = len(cands) - len(c2)
    c2.sort(key=lambda z: -z[0].sum())
    keep = []
    for m, s in c2:
        if all(m_ios(m, k) < dedupe and m_iou(m, k) < dedupe for k, _ in keep):
            keep.append((m, s))
    return (keep[:topk] if topk else keep), n_floor

def to_rle_kept(kept):
    return [dict(rle=rle_encode(m), score=round(float(s), 4), area=int(m.sum())) for m, s in kept]

# COMMAND ----------

# MAGIC %md ## 2 · Volume helpers and guards (every stage)

# COMMAND ----------

def read_json(p):
    with open(p) as f: return json.load(f)
def read_json_gz(p):
    with gzip.open(p, "rt") as f: return json.load(f)
def write_json(p, obj, indent=1):
    os.makedirs(osp.dirname(p), exist_ok=True)
    with open(p, "w") as f: json.dump(obj, f, indent=indent, default=str)     # sequential "w" is Volume-safe

def put(local, vol):
    """Local file -> Volume with dbutils.fs.cp (never a seeking writer on the Volume), 3 retries (Tol_17:155-161)."""
    os.makedirs(osp.dirname(vol), exist_ok=True)
    for attempt in range(3):
        try:
            dbutils.fs.cp("file:" + local, "dbfs:" + vol); break
        except Exception as e:
            print(f"  put retry {attempt + 1} for {osp.basename(vol)}: {e}"); time.sleep(5)
    else: raise RuntimeError(f"could not copy {local} -> {vol}")
    if osp.getsize(vol) != osp.getsize(local): raise RuntimeError(f"{vol} size != local build")

def stage_file(src_vol, dst_local, label=""):
    """dbutils.fs.cp('dbfs:'+src, 'file:'+dst), 3 retries with 5 s sleeps (Tol_17:155-161); returns (seconds, MB)."""
    if osp.exists(dst_local): os.remove(dst_local)
    t0 = time.time()
    for attempt in range(3):
        try:
            dbutils.fs.cp("dbfs:" + src_vol, "file:" + dst_local); break
        except Exception as e:
            print(f"  stage retry {attempt + 1} for {label or src_vol}: {e}"); time.sleep(5)
    else: raise RuntimeError(f"could not stage {src_vol}")
    return time.time() - t0, osp.getsize(dst_local) / 1e6

def write_json_gz_vol(p, obj):
    tmp = osp.join(LOCAL, osp.basename(p) + ".tmp")
    with gzip.open(tmp, "wt") as f: json.dump(obj, f, default=str)
    put(tmp, p); os.remove(tmp)

def read_json_gz_vol(p):
    tmp = osp.join(LOCAL, osp.basename(p) + ".in")
    stage_file(p, tmp); d = read_json_gz(tmp); os.remove(tmp); return d

# ---- arm identity: cached dets_<arm>.json.gz are reused / scored only if their meta names the model loaded now ----
F32_PREC = "highest"                    # torch default; Tol_03 never sets matmul precision, so the production tracker ran fp32 at 'highest'
_STUDENT_SHA = {}
def student_ckpt_local():
    """Stage Checkpoints/<student_ckpt> to local disk once; -> (local path, sha256). The student arm loads THIS file."""
    if STUDENT_CKPT not in _STUDENT_SHA:
        loc = osp.join(LOCAL, "student_" + STUDENT_CKPT)
        dt, mb = stage_file(osp.join(TOL, "Checkpoints", STUDENT_CKPT), loc, STUDENT_CKPT)
        _STUDENT_SHA[STUDENT_CKPT] = (loc, sha256_file(loc))
        print(f"staged student {STUDENT_CKPT} {mb:.0f} MB in {dt:.0f} s, sha256 {_STUDENT_SHA[STUDENT_CKPT][1][:16]}")
    return _STUDENT_SHA[STUDENT_CKPT]

def ident_teacher():
    return dict(model=SAM3_ID, backbone="teacher (own)", float32_matmul_precision=F32_PREC)
def ident_rfdetr(TS):
    return dict(model="RF-DETR-Seg", run_tag=RUN_TAG, checkpoint_sha256=TS["checkpoint_best_total_sha256"],
                op_threshold=float(TS["op_threshold"]), resolution=int(TS["resolution"]))
def ident_student(sha):
    return dict(model=SAM3_ID, student_ckpt=STUDENT_CKPT, student_ckpt_sha256=sha, float32_matmul_precision=F32_PREC)

if not SELFTEST_ONLY:
    os.makedirs(EVAL_DIR, exist_ok=True)
    if osp.exists(LOCAL): shutil.rmtree(LOCAL)          # warm clusters keep stale files
    os.makedirs(LOCAL, exist_ok=True)
    MANIFEST_PATH = osp.join(COCO_DIR, "manifest.json")
    if not osp.exists(MANIFEST_PATH): raise RuntimeError(f"no export manifest at {MANIFEST_PATH}")
    MANIFEST = read_json(MANIFEST_PATH); MANIFEST_SHA = sha256_file(MANIFEST_PATH)
    if MANIFEST.get("smoke"): raise RuntimeError(f"dataset {DATASET_TAG} is a SMOKE export — its constants are not the dataset's")
    _TS_SEAL = None                      # unsealed: only a Tol_43 run that recorded the below_floor waiver makes it evaluable
    if not MANIFEST.get("sealed") and RUN_TAG and osp.exists(osp.join(CKPT_DIR, "train_summary.json")):
        _TS_SEAL = read_json(osp.join(CKPT_DIR, "train_summary.json"))
    SEAL_NOTE = seal_admissible(MANIFEST, MANIFEST_SHA, _TS_SEAL)
    if SEAL_NOTE: print(f"⚠ {SEAL_NOTE}")
    EVAL_DAYS = check_eval_days(EVAL_DAYS, MANIFEST)
    FRAG_LO = {p: manifest_const(MANIFEST, f"frag_lo[{p}]") for p in ("dayl", "dark")}
    OVER_HI = manifest_const(MANIFEST, "over_hi")
    print(f"dataset {DATASET_TAG}: source_days={MANIFEST.get('source_days')} forbidden_eval_days={MANIFEST['forbidden_eval_days']}")
    print(f"eval_days={EVAL_DAYS} — none is a training source (leakage guard passed)")
    print(f"READ from the manifest (training-derived): frag_lo={FRAG_LO} over_hi={OVER_HI}")
    CALLS_PATH = osp.join(EVAL_DIR, "calls.json"); FRAMES_TAR = osp.join(EVAL_DIR, "frames.tar")

# COMMAND ----------

# MAGIC %md ## 3 · `stage=extract` (CPU) — calls, GT rows, the frames the tracker's detector saw

# COMMAND ----------

def run_extract():
    if osp.exists(CALLS_PATH) and osp.exists(FRAMES_TAR) and not FORCE and not SMOKE:   # SMOKE never resumes (repo convention)
        print(f"extract: {CALLS_PATH} and frames.tar exist — skipped (force=true rebuilds)"); return
    t_all = time.time(); timing = dict(stage_s=0.0, stage_mb=0.0, n_tars=0)
    all_calls, skipped, a_ref_eval, counts = [], Counter(), {}, {}
    for day in EVAL_DAYS:
        droot = osp.join(TOL, EVAL_ROOT, day)
        evp = osp.join(droot, "events.jsonl")
        if not osp.exists(evp): raise RuntimeError(f"{evp} missing")
        with open(evp) as f: events = [json.loads(l) for l in f if l.strip()]
        calls = parse_calls(events, day)
        check_no_repairs(events, day)
        counts[day] = dict(Counter(c["type"] for c in calls))
        print(f"{day}: {len(calls)} detector calls {counts[day]} (no repair events: stored rows are pre-pass propagation)")
        exp = EXPECTED_CALLS.get(day)
        if EVAL_ROOT == "Tracking_v4v2" and exp and any(counts[day].get(k, 0) != v for k, v in exp.items()):
            raise RuntimeError(f"{day}: expected {exp} detector calls on Tracking_v4v2 (the reviewer-verified run, measured "
                               f"2026-09-23), got {counts[day]} — the logs on the Volume are not that run")
        by_stem = defaultdict(list)
        for c in calls: by_stem[c["clip_stem"]].append(c)
        samples = []                                   # A_ref_eval: largest stored mask, daylight rows at 1 Hz (Tol_40:111-141)
        mfiles = sorted(glob.glob(osp.join(droot, "masks_*.json.gz")))
        if not mfiles: raise RuntimeError(f"{droot}: no masks_*.json.gz")
        seen_stems = set()
        for p in mfiles:
            stem = osp.basename(p)[len("masks_"):-len(".json.gz")]
            d = read_json_gz(p)
            if clip_stem_of(d["clip"]) != stem: raise RuntimeError(f"{p}: clip {d['clip']} != file stem {stem}")
            seen_stems.add(stem); rows = d["frames"]; cs = clip_start(stem)
            for r in rows[::A_REF_STRIDE_ROWS]:
                if pool_of(phase4(cs + timedelta(seconds=float(r["t_sec"])))) != "dayl": continue
                areas = [rle_area(m) for m in row_masks(r) if m]
                if areas: samples.append(max(areas))
            by_t = index_rows(rows)
            for c in by_stem.get(stem, []):
                pos, err = join_event_row(by_t, c["t"])
                if err:
                    skipped[f"{day}:{err}"] += 1; print(f"  ⚠ {c['call_id']}: {err} stored row — skipped"); continue
                row = rows[pos]; nxt = rows[pos + 1] if pos + 1 < len(rows) else None
                c["frame_idx"] = int(row["frame_idx"]); c["row_t_sec"] = row["t_sec"]
                c["row_masks"] = row_masks(row)
                c["next_frame_idx"] = int(nxt["frame_idx"]) if nxt else None
                c["next_masks"] = row_masks(nxt) if nxt else None
                c["prev_masks"] = prev_masks_for(rows, pos, [int(s) for s in c["event"].get("struck", [])]) if c["type"] != "seeded" else {}
                for m in c["row_masks"] + (c["next_masks"] or []) + list(c["prev_masks"].values()):
                    if m and list(m["size"]) != [H_FULL, W_FULL]: raise RuntimeError(f"{c['call_id']}: mask size {m['size']} != [1296, 2304]")
            del d, rows
        for stem, cs_ in by_stem.items():
            if stem not in seen_stems:
                skipped[f"{day}:no_masks_file"] += len(cs_); print(f"  ⚠ {stem}: {len(cs_)} calls but no masks file — skipped")
        if not samples: raise RuntimeError(f"{day}: no daylight rows — cannot compute A_ref_eval")
        a_ref_eval[day] = dict(value=float(np.median(samples)), n_samples=len(samples),
                               rule="median largest stored mask over daylight (phase4 day|dawn) rows at 1 Hz of this eval day's own GT "
                                    "(normalisation of the GT for the trust filter only; not a training constant)")
        print(f"  A_ref_eval[{day}] = {a_ref_eval[day]['value']:,.0f} px over {len(samples)} daylight samples")
        all_calls += [c for c in calls if "frame_idx" in c]
    if SMOKE:
        n0 = len(all_calls); all_calls = smoke_pick(all_calls, MAX_CALLS)
        print(f"⚠ SMOKE cap: {len(all_calls)} of {n0} calls kept")
    if skipped: print(f"⚠ join skips: {dict(skipped)}")
    # ---- frames: stage only the tars holding a call, copy the member bytes verbatim into frames.tar ----
    need = defaultdict(set)
    for c in all_calls: need[(c["day"], c["clip_stem"])].add(c["frame_idx"])
    out_loc = osp.join(LOCAL, "frames.tar"); fsha = {}
    with tarfile.open(out_loc, "w") as out:
        for (day, stem), fids in sorted(need.items()):
            src = osp.join(FRAMES_DIR, day, stem + ".tar")
            if not osp.exists(src): raise RuntimeError(f"{src} missing — the frame cache must hold every call frame")
            loc = osp.join(LOCAL, "cur.tar"); dt, mb = stage_file(src, loc, stem)
            timing["stage_s"] += dt; timing["stage_mb"] += mb; timing["n_tars"] += 1
            with tarfile.open(loc) as tf:
                for fi in sorted(fids):
                    nm = f"{fi:07d}.jpg"
                    try: b = tf.extractfile(tf.getmember(nm)).read()
                    except KeyError: raise RuntimeError(f"{stem}.tar has no member {nm} — frame-identity contract broken")
                    ti = tarfile.TarInfo(f"{stem}/{nm}"); ti.size = len(b)
                    out.addfile(ti, io.BytesIO(b)); fsha[fkey(stem, fi)] = sha256_bytes(b)
            os.remove(loc)
            print(f"  staged {stem}.tar {mb:.0f} MB in {dt:.0f} s ({mb / max(dt, 1e-6):.1f} MB/s), {len(fids)} frames")
    for c in all_calls: c["frame_sha256"] = fsha[fkey(c["clip_stem"], c["frame_idx"])]
    put(out_loc, FRAMES_TAR); tar_mb = osp.getsize(out_loc) / 1e6; os.remove(out_loc)
    CJ = dict(eval_tag=EVAL_TAG, eval_days=EVAL_DAYS, eval_root=EVAL_ROOT, created=datetime.now().isoformat(), smoke=SMOKE,
              max_calls=MAX_CALLS, dataset_tag=DATASET_TAG, manifest_sha256=MANIFEST_SHA,
              forbidden_eval_days=MANIFEST["forbidden_eval_days"], a_ref_eval=a_ref_eval, counts_by_day=counts,
              join_skipped=dict(skipped), n_calls=len(all_calls), n_frames=len(fsha), calls=all_calls,
              tracker_claims_verified="see notebook header: row written after the pass; opened slots take the NEXT row; seeding runs track_frame on the seed frame")
    write_json(CALLS_PATH, CJ, indent=None)
    timing.update(total_s=time.time() - t_all, frames_tar_mb=tar_mb)
    write_json(osp.join(EVAL_DIR, "timing_extract.json"), timing)
    print(f"extract: {len(all_calls)} calls on {len(fsha)} frames; frames.tar {tar_mb:.0f} MB | staging {timing['stage_s']:.0f} s "
          f"({timing['stage_mb'] / max(timing['stage_s'], 1e-6):.1f} MB/s over {timing['n_tars']} tars) | total {timing['total_s'] / 60:.1f} min")

if not SELFTEST_ONLY and STAGE == "extract":
    run_extract()

# COMMAND ----------

# MAGIC %md ## 4 · `stage=detect` (A100) — every arm on the identical cached frame, one model at a time

# COMMAND ----------

def load_calls_checked():
    if not osp.exists(CALLS_PATH): raise RuntimeError(f"{CALLS_PATH} missing — run stage=extract first (CPU)")
    CJ = read_json(CALLS_PATH)
    if CJ["manifest_sha256"] != MANIFEST_SHA:
        raise RuntimeError(f"calls.json was extracted under manifest {CJ['manifest_sha256'][:12]}, now {MANIFEST_SHA[:12]} — re-extract (force=true)")
    if (sorted(CJ["eval_days"]) != sorted(EVAL_DAYS) or CJ["eval_root"] != EVAL_ROOT or bool(CJ["smoke"]) != SMOKE
            or int(CJ.get("max_calls", -1)) != MAX_CALLS):
        raise RuntimeError(f"calls.json is for days {CJ['eval_days']} root {CJ['eval_root']} smoke {CJ['smoke']} max_calls {CJ.get('max_calls')}; "
                           f"widgets say {EVAL_DAYS} {EVAL_ROOT} {SMOKE} {MAX_CALLS} — re-run stage=extract")
    check_eval_days(CJ["eval_days"], MANIFEST)
    return CJ

def read_train_summary():
    p = osp.join(CKPT_DIR, "train_summary.json")
    if not osp.exists(p): raise RuntimeError(f"{p} missing — Tol_43 mode=full writes it")
    TS = read_json(p)
    for k in ("run_tag", "dataset_tag", "dataset_manifest_sha256", "source_days", "op_threshold", "resolution", "checkpoint_best_total_sha256", "mode"):
        if k not in TS: raise RuntimeError(f"train_summary.json lacks {k!r}")
    if TS["run_tag"] != RUN_TAG: raise RuntimeError(f"train_summary run_tag {TS['run_tag']} != widget {RUN_TAG}")
    if TS["mode"] not in ("full", "resume") or TS.get("smoke"): raise RuntimeError(f"run {RUN_TAG} mode={TS['mode']} is not a full training run")
    if TS["dataset_tag"] != DATASET_TAG or TS["dataset_manifest_sha256"] != MANIFEST_SHA:
        raise RuntimeError(f"run {RUN_TAG} was trained on {TS['dataset_tag']} / manifest {TS['dataset_manifest_sha256'][:12]}, "
                           f"not {DATASET_TAG} / {MANIFEST_SHA[:12]} — the leakage guard would be checking the wrong manifest")
    if set(TS["source_days"]) & set(EVAL_DAYS): raise RuntimeError(f"run {RUN_TAG} trained on {TS['source_days']} ∩ eval_days {EVAL_DAYS}")
    return TS

if not SELFTEST_ONLY and STAGE in ("detect", "all"):
    import torch, torch.nn as nn, torch.nn.functional as F
    from huggingface_hub import login
    import timm
    from timm.data import resolve_data_config
    from torchvision import transforms
    import transformers
    from transformers import Sam3Model, Sam3Processor
    from transformers.modeling_outputs import BaseModelOutput
    if transformers.__version__ != "5.2.0": raise RuntimeError(f"transformers {transformers.__version__} != 5.2.0")
    login(token=dbutils.secrets.get(scope="hf", key="token").strip().strip('"').strip("'").strip())
    device = torch.device("cuda")
    DTYPE = torch.bfloat16                       # bf16 only, never fp16
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("bf16 unsupported on this GPU")
    torch.set_float32_matmul_precision(F32_PREC)  # explicit: `import rfdetr` sets 'high' (TF32) process-wide (detr.py:52-55); restored after its cell

    CJ = load_calls_checked(); CALLS = CJ["calls"]
    TS = read_train_summary() if "rfdetr" in ARMS else None
    OP_THRESHOLD = float(TS["op_threshold"]) if TS else None
    if TS: print(f"run {RUN_TAG}: op_threshold={OP_THRESHOLD} (READ from train_summary.json, never tuned here), resolution={TS['resolution']}")
    STUDENT_LOC, STUDENT_SHA = student_ckpt_local() if "student_sam3" in ARMS else (None, None)
    ARM_IDENT = dict(teacher=ident_teacher(), rfdetr=(ident_rfdetr(TS) if TS else None),
                     student_sam3=(ident_student(STUDENT_SHA) if STUDENT_SHA else None))
    FR_LOC = osp.join(LOCAL, "frames.tar"); dt, mb = stage_file(FRAMES_TAR, FR_LOC, "frames.tar")
    print(f"staged frames.tar {mb:.0f} MB in {dt:.0f} s ({mb / max(dt, 1e-6):.1f} MB/s)")
    T_STAGE = {"frames_tar_s": dt}
    FRAMES = {}
    with tarfile.open(FR_LOC) as tf:
        for m in tf.getmembers():
            stem, nm = m.name.split("/"); FRAMES[fkey(stem, int(nm[:7]))] = tf.extractfile(m).read()
    SHA_OF = {fkey(c["clip_stem"], c["frame_idx"]): c["frame_sha256"] for c in CALLS}
    KEYS = sorted(SHA_OF)
    for k in KEYS:
        if k not in FRAMES: raise RuntimeError(f"frames.tar lacks {k}")
        if sha256_bytes(FRAMES[k]) != SHA_OF[k]: raise RuntimeError(f"{k}: JPEG bytes differ from extract (sha256)")
    print(f"{len(CALLS)} calls on {len(KEYS)} unique frames; arms {ARMS}")

    def frame_of(k):
        pil = Image.open(io.BytesIO(FRAMES[k])).convert("RGB"); arr = np.asarray(pil)
        if arr.shape != (H_FULL, W_FULL, 3): raise RuntimeError(f"{k}: frame {arr.shape} != (1296, 2304, 3)")
        return pil, arr

    def dets_path(arm): return osp.join(EVAL_DIR, f"dets_{arm}.json.gz")

    def run_arm(arm, fn, meta):
        """fn(pil, arr) -> (kept [(mask, score)], n_raw or None, latency_ms detector-only). Writes dets_<arm>.json.gz."""
        want = ARM_IDENT[arm].get("float32_matmul_precision")
        if want and torch.get_float32_matmul_precision() != want:
            raise RuntimeError(f"{arm}: float32 matmul precision is {torch.get_float32_matmul_precision()!r}, the arm's contract is {want!r}")
        rows, lat, t0 = {}, [], time.time()
        for i, k in enumerate(KEYS):
            pil, arr = frame_of(k)
            kept, n_raw, ms = fn(pil, arr)
            rows[k] = dict(kept=to_rle_kept(kept), n_raw=n_raw, ms=round(ms, 2)); lat.append(ms)
            if (i + 1) % 100 == 0: print(f"  {arm}: {i + 1}/{len(KEYS)} | {(time.time() - t0) / 60:.1f} min")
        meta = dict(meta, **ARM_IDENT[arm])       # identity keys arm_done / run_score compare against the model loaded then
        meta = dict(meta, arm=arm, n_frames=len(rows), calls_sha256=sha256_file(CALLS_PATH), manifest_sha256=MANIFEST_SHA,
                    float32_matmul_precision_at_run=torch.get_float32_matmul_precision(),
                    latency_ms_median=(float(np.median(lat[3:])) if len(lat) > 3 else None), latency_n=len(lat),
                    latency_note="detector only (forward + post-chain), JPEG decode excluded, first 3 frames dropped as warm-up",
                    peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), wall_s=round(time.time() - t0, 1),
                    created=datetime.now().isoformat())
        write_json_gz_vol(dets_path(arm), dict(meta=meta, rows=rows))
        print(f"{arm}: {len(rows)} frames in {meta['wall_s'] / 60:.1f} min, median {meta['latency_ms_median']} ms/frame -> {dets_path(arm)}")
        return rows

    def arm_done(arm):
        """Reuse cached dets only for this calls.json AND the model loaded now (ARM_IDENT); SMOKE never resumes."""
        if SMOKE:
            print(f"{arm}: SMOKE — cached dets are never reused"); return None
        if osp.exists(dets_path(arm)) and not FORCE:
            d = read_json_gz_vol(dets_path(arm))
            if d["meta"].get("calls_sha256") == sha256_file(CALLS_PATH) and set(d["rows"]) >= set(KEYS):
                mm = meta_mismatch(d["meta"], ARM_IDENT[arm])
                if not mm:
                    print(f"{arm}: dets exist for this calls.json and model — skipped (force=true re-runs)"); return d["rows"]
                print(f"⚠ {arm}: cached dets are from another model {mm} (cached, current) — re-running")
            else:
                print(f"⚠ {arm}: existing dets are for another calls.json — re-running")
        return None

    def free():
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# COMMAND ----------

if not SELFTEST_ONLY and STAGE in ("detect", "all"):
    # ---------------- SAM3 (teacher arm; the student arm swaps its backbone) ----------------
    def load_sam3():
        global dproc, det, TEACHER_BB
        dproc = Sam3Processor.from_pretrained(SAM3_ID)
        det = Sam3Model.from_pretrained(SAM3_ID, torch_dtype=DTYPE).to(device).eval()
        TEACHER_BB = det.vision_encoder.backbone

    # VERBATIM from Tol_03_track_v4.py:453-472 (detect) — the incumbent
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

    def teacher_fn(pil, arr):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        kept = detect(pil, topk=DET_TOPK)
        torch.cuda.synchronize(); return kept, None, (time.perf_counter() - t0) * 1000

    T_ROWS = arm_done("teacher")
    if T_ROWS is None:
        load_sam3()
        T_ROWS = run_arm("teacher", teacher_fn, dict(model=SAM3_ID, backbone="teacher (own)", contract="detect() verbatim Tol_03:453-472, topk 6"))
        del det, dproc, TEACHER_BB; free()
    # early faithfulness gate: a replay that does not reproduce the tracker's counts voids every arm — stop before spending more GPU
    _recs = []
    for c in CALLS:
        if c["type"] in ("pass", "pass_no_detections"):
            _recs.append(dict(type=c["type"], phase4=c["phase4"], arms={"teacher": dict(n_kept=len(T_ROWS[fkey(c['clip_stem'], c['frame_idx'])]["kept"]))},
                              logged=dict(n_det=(int(c["event"]["n_det"]) if c["type"] == "pass" else 0))))
    _fr = ndet_faithfulness(_recs)
    print("teacher replay vs logged n_det:", {k: f"{v['agree']}/{v['n']} = {v['rate']:.3f}" for k, v in _fr.items()})
    faithfulness_gate(_fr, SMOKE)

# COMMAND ----------

if not SELFTEST_ONLY and STAGE in ("detect", "all") and "rfdetr" in ARMS:
    # ---------------- RF-DETR-Seg (as in Tol_43 reload_best) ----------------
    R_ROWS = arm_done("rfdetr")
    if R_ROWS is None:
        os.environ["RF_HOME"] = "/local_disk0/rf_home"          # before import rfdetr
        from rfdetr import from_checkpoint                       # side effect: set_float32_matmul_precision('high') (detr.py:52-55) — rfdetr's own
                                                                 # numerics (as in Tol_43's op sweep); reset to F32_PREC after this cell
        src = osp.join(CKPT_DIR, "checkpoint_best_total.pth")
        if not osp.exists(src): raise RuntimeError(f"{src} missing")
        loc = osp.join(LOCAL, "rf_best.pth"); dt, mb = stage_file(src, loc, "checkpoint_best_total.pth"); T_STAGE["rf_ckpt_s"] = dt
        sha = sha256_file(loc)
        if sha != TS["checkpoint_best_total_sha256"]:
            raise RuntimeError(f"checkpoint sha {sha[:16]} != train_summary {TS['checkpoint_best_total_sha256'][:16]} — not the file the summary describes")
        RES = int(TS["resolution"])
        RF = from_checkpoint(loc, resolution=RES)               # resolution PINNED: strip_checkpoint drops model_config (Tol_43 reload_best)
        RF_NC = int(RF.model_config.num_classes)
        if RF_NC != 1: raise RuntimeError(f"num_classes {RF_NC} != 1")
        if list(RF.class_names) != ["calf"]: raise RuntimeError(f"class_names {RF.class_names} != ['calf']")
        if int(RF.model_config.resolution) != RES: raise RuntimeError(f"resolution {RF.model_config.resolution} != {RES}")
        if int(RF.model_config.positional_encoding_size) != RES // int(RF.model_config.patch_size):
            raise RuntimeError("positional_encoding_size != resolution // patch_size — the trained PE would be interpolated (weights.py:622)")

        def rf_fn(pil, arr):
            """predict(threshold=op) keeps score > op (strict, postprocess.py:219-225) = Tol_43's sweep rule; drop
            '__background__' rows (class_id >= num_classes, detr.py:2757-2772); then the incumbent post-chain."""
            torch.cuda.synchronize(); t0 = time.perf_counter()
            d = RF.predict(arr, threshold=OP_THRESHOLD, include_source_image=False)
            rows = []
            if d.mask is not None and len(d):
                keep = d.class_id < RF_NC
                rows = [(np.asarray(m, dtype=bool), float(s)) for m, s in zip(d.mask[keep], d.confidence[keep])]
            for m, s in rows:
                if m.shape != (H_FULL, W_FULL): raise RuntimeError(f"RF-DETR mask {m.shape} != (1296, 2304)")
                if not np.isfinite(s): raise RuntimeError("non-finite RF-DETR score")
            kept, _ = post_chain(rows)
            torch.cuda.synchronize(); return kept, len(rows), (time.perf_counter() - t0) * 1000

        kb, _, _ = rf_fn(None, np.zeros((H_FULL, W_FULL, 3), np.uint8))
        if kb: raise RuntimeError(f"RF-DETR keeps {len(kb)} masks on a black frame at op_threshold {OP_THRESHOLD}")
        R_ROWS = run_arm("rfdetr", rf_fn, dict(model="RF-DETR-Seg", run_tag=RUN_TAG, checkpoint_sha256=sha, op_threshold=OP_THRESHOLD,
                                               resolution=RES, op_threshold_source="train_summary.json (Tol_43), never tuned here",
                                               contract="predict(threshold=op) -> class_id<1 -> MIN_MASK_PX -> area-desc -> dedupe max(IoS,IoU)<0.50 -> top-6"))
        del RF; free()
    torch.set_float32_matmul_precision(F32_PREC)            # undo rfdetr's import-time TF32 before any later fp32 arm (student TinyViT)

# COMMAND ----------

if not SELFTEST_ONLY and STAGE in ("detect", "all") and "student_sam3" in ARMS:
    # ---------------- student_sam3: VERBATIM from Tol_35_detector_ab.py:149-226 (+ the shim-identity control) ----------------
    S_ROWS = arm_done("student_sam3")
    if S_ROWS is None:
        load_sam3()

        def replace_bn_with_gn(module):
            """Verbatim from Tol_03_track_v4.py:350. The checkpoint has NO BatchNorm buffers; omitting this
            leaves running_mean=0/running_var=1 at fresh init while .weight/.bias silently absorb the GN values,
            so the load succeeds and the feature map is garbage. That is exactly what voided the first A/B."""
            for name, child in module.named_children():
                if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nc = child.num_features; ng = min(32, nc)
                    while nc % ng != 0: ng -= 1
                    gn = nn.GroupNorm(ng, nc, affine=child.affine)
                    if child.affine and child.weight is not None:
                        gn.weight.data.copy_(child.weight.data); gn.bias.data.copy_(child.bias.data)
                    setattr(module, name, gn)
                else: replace_bn_with_gn(child)

        class StudentEncoderTinyViT(nn.Module):
            """Verbatim from Tol_03_track_v4 / Tol_02_distill_student — must match the checkpoint exactly."""
            def __init__(self, out_channels=1024):
                super().__init__()
                self.backbone = timm.create_model("tiny_vit_21m_512", pretrained=False, features_only=True,
                                                  out_indices=(1, 2, 3), drop_path_rate=0.0)
                replace_bn_with_gn(self.backbone)
                def lat(c):
                    return nn.Sequential(nn.Conv2d(c, 256, 1, bias=False), nn.GroupNorm(32, 256), nn.GELU())
                self.lateral_s1, self.lateral_s2, self.lateral_s3 = lat(192), lat(384), lat(576)
                self.channel_proj = nn.Sequential(nn.Conv2d(768, out_channels, 1, bias=False),
                                                  nn.GroupNorm(32, out_channels), nn.GELU())
                self.spatial_refine = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, out_channels), nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, out_channels))
                self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
                self.output_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

            def forward(self, x):
                s = self.backbone(x)
                f1 = F.interpolate(self.lateral_s1(s[0]), size=(72, 72), mode="area")
                f2 = F.interpolate(self.lateral_s2(s[1]), size=(72, 72), mode="bilinear", align_corners=False)
                f3 = F.interpolate(self.lateral_s3(s[2]), size=(72, 72), mode="bilinear", align_corners=False)
                o = self.channel_proj(torch.cat([f1, f2, f3], 1))
                o = o + self.spatial_refine(o)
                return o * self.output_scale + self.output_bias

        CKPT = STUDENT_CKPT
        ck = torch.load(STUDENT_LOC, map_location="cpu")        # the staged copy of Checkpoints/<CKPT> whose sha256 is in ARM_IDENT
        student = StudentEncoderTinyViT()
        sd = student.state_dict()
        sd.update(ck["model_state_dict"])
        for k, v in ck["ema_shadow"].items(): sd[k] = v
        student.load_state_dict(sd)
        student = student.to(device).eval()
        # --- loud failure instead of a silent wrong load (the Tol_35 v1 fault) ---
        _nbn = sum(1 for _m in student.modules() if isinstance(_m, (nn.BatchNorm1d, nn.BatchNorm2d)))
        if _nbn: raise RuntimeError(f"{_nbn} BatchNorm layers survive: architecture does not match the checkpoint")
        _have = set(ck["model_state_dict"]) | set(ck["ema_shadow"])
        _uninit = [k for k in student.state_dict() if k not in _have]
        if _uninit:
            raise RuntimeError(f"{len(_uninit)} tensors absent from the checkpoint would stay at fresh init: {_uninit[:8]}")
        print(f"student: 0 BatchNorm, every one of {len(student.state_dict())} tensors loaded from the checkpoint")

        _t = timm.create_model("tiny_vit_21m_512", pretrained=False)
        _cfg = resolve_data_config(_t.pretrained_cfg); del _t
        STUDENT_TF = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                         transforms.Normalize(mean=_cfg["mean"], std=_cfg["std"])])

        class DummyBackbone(nn.Module):
            """Same shim the tracker uses. Counts calls: if the detector invokes the backbone more than once
            per forward, a single injected feature map is NOT a faithful substitution and the A/B is invalid."""
            def __init__(self):
                super().__init__(); self.student_features = None; self.calls = 0
            def forward(self, *a, **k):
                self.calls += 1
                f = self.student_features
                B, C, H, W = f.shape
                return BaseModelOutput(last_hidden_state=f.permute(0, 2, 3, 1).reshape(B, H * W, C))

        dummy = DummyBackbone().to(device)

        def same_kept(A, B):
            return len(A) == len(B) and all(abs(sa - sb) < 1e-6 and m_iou(ma, mb) == 1.0 for (ma, sa), (mb, sb) in zip(A, B))

        def shim_identity_control(pil):
            """Teacher features captured by a hook and replayed through the shim must reproduce detect() at IoU 1.000
            A failed control means the student arm would measure the harness, not the student."""
            cap = {}
            def hook(mod, inp, out): cap["f"] = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            hd = TEACHER_BB.register_forward_hook(hook)
            try: ref = detect(pil, topk=DET_TOPK)
            finally: hd.remove()
            f = cap["f"]; B, N, C = f.shape; s = int(round(math.sqrt(N)))
            if s * s != N: raise RuntimeError(f"teacher backbone tokens {N} not square")
            dummy.student_features = f.reshape(B, s, s, C).permute(0, 3, 1, 2).contiguous(); dummy.calls = 0
            det.vision_encoder.backbone = dummy
            try: rep = detect(pil, topk=DET_TOPK)
            finally: det.vision_encoder.backbone = TEACHER_BB; dummy.student_features = None
            return same_kept(ref, rep), len(ref), len(rep)

        for k in KEYS[:3]:
            ok, nr, np_ = shim_identity_control(frame_of(k)[0])
            print(f"  shim-identity control {k}: {'IoU 1.000' if ok else 'MISMATCH'} ({nr} vs {np_} kept)")
            if not ok: raise RuntimeError(f"shim-identity control failed on {k}: the DummyBackbone swap does not reproduce the teacher")

        def student_fn(pil, arr):
            """Tol_35:274-281 swap-in; dummy.calls == 1 asserted per forward."""
            with torch.no_grad():
                torch.cuda.synchronize(); t0 = time.perf_counter()
                feat = student(STUDENT_TF(pil).unsqueeze(0).to(device))
                dummy.student_features = feat.to(DTYPE); dummy.calls = 0
                det.vision_encoder.backbone = dummy
                try: kept = detect(pil, topk=DET_TOPK)
                finally: det.vision_encoder.backbone = TEACHER_BB
                torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
            if dummy.calls != 1: raise RuntimeError(f"DummyBackbone called {dummy.calls}× per forward — substitution not faithful")
            dummy.student_features = None
            return kept, None, ms

        S_ROWS = run_arm("student_sam3", student_fn, dict(model=SAM3_ID, backbone=f"student {CKPT} via DummyBackbone",
                                                          contract="detect() verbatim, backbone swapped (Tol_35:274-281)",
                                                          shim_identity="passed on 3 frames"))
        del det, dproc, TEACHER_BB, student, dummy; free()

if not SELFTEST_ONLY and STAGE in ("detect", "all"):
    write_json(osp.join(EVAL_DIR, "timing_detect.json"), dict(staging=T_STAGE, arms=ARMS, created=datetime.now().isoformat()))
    print("detect: done — stage=score needs no model")

# COMMAND ----------

# MAGIC %md ## 5 · `stage=score` (CPU or local) — GT, trust filter, metrics, verdict, decision replay

# COMMAND ----------

def fmt_ci(t): return f"{t[0]:+.3g} [{t[1]:+.3g}, {t[2]:+.3g}]"

def run_score():
    t0 = time.time()
    CJ = load_calls_checked(); CALLS = CJ["calls"]
    A_REF_EVAL = {d: float(v["value"]) for d, v in CJ["a_ref_eval"].items()}
    DETS, META = {}, {}
    for arm in ARMS:
        p = osp.join(EVAL_DIR, f"dets_{arm}.json.gz")
        if not osp.exists(p): raise RuntimeError(f"{p} missing — run stage=detect for arm {arm}")
        d = read_json_gz_vol(p)
        if d["meta"].get("calls_sha256") != sha256_file(CALLS_PATH): raise RuntimeError(f"dets_{arm} was computed on another calls.json")
        if d["meta"].get("manifest_sha256") != MANIFEST_SHA: raise RuntimeError(f"dets_{arm} was computed under another manifest")
        DETS[arm], META[arm] = d["rows"], d["meta"]
    # the verdict must be about the model the widgets name: compare every cached arm's identity with the current one
    CUR_IDENT = dict(teacher=ident_teacher())
    if "rfdetr" in META: CUR_IDENT["rfdetr"] = ident_rfdetr(read_train_summary())
    if "student_sam3" in META: CUR_IDENT["student_sam3"] = ident_student(student_ckpt_local()[1])
    for arm, ident in CUR_IDENT.items():
        mm = meta_mismatch(META[arm], ident)
        if mm: raise RuntimeError(f"dets_{arm} was computed by another model than the widgets name now: {mm} (cached, current) — "
                                  f"re-run stage=detect")
    recs = [call_record(c, DETS, A_REF_EVAL, FRAG_LO, OVER_HI, ARMS) for c in CALLS]
    print(f"score: {len(recs)} calls in {time.time() - t0:.0f} s")

    # ---- replay faithfulness (gate) ----
    FAITH = ndet_faithfulness(recs)
    print("teacher replay vs logged n_det:", {k: f"{v['agree']}/{v['n']} = {v['rate']:.3f}" for k, v in FAITH.items()})
    FAITH_STATUS = faithfulness_gate(FAITH, SMOKE)
    RFAITH = replay_faithfulness(recs)
    print(f"teacher decision replay vs log (reported): matched {RFAITH['matched_same']:.3f}, strikes {RFAITH['strikes_same']:.3f}, "
          f"tracked-count {RFAITH['tracked_count_same']:.3f} over {RFAITH['n']} passes")

    # ---- GT accounting ----
    gt_acc = defaultdict(Counter)
    for r in recs:
        g = gt_acc[r["phase4"]]
        g["trusted"] += len(r["gt"]); g["untrusted"] += len(r["untrusted"]); g["F4_calls"] += int(r["f4"])
        for x in r["gt"]: g[f"trusted_{x['kind']}"] += 1
        for u in r["untrusted"]:
            for why in u["reasons"]: g[f"untrusted_{why}"] += 1
        for u in r["unscorable"]: g[f"unscorable_{u['reason']}"] += 1
        g["opened_stale_at_call"] += len(r["stale"])
    gt_acc = {k: dict(v) for k, v in gt_acc.items()}
    print("GT per phase4:")
    for k, v in sorted(gt_acc.items()): print(f"  {k:5s} {v}")

    # ---- per-arm comparison, verdict ----
    CMP, VERD, CIS, REPLAY = {}, {}, {}, {}
    for arm in CHALLENGERS:
        CMP[arm] = {st: compare(recs, arm, st) for st in STRATA}
        ok, fails = verdict_rule(CMP[arm])
        VERD[arm] = dict(result="PASS" if ok else "FAIL", failures=fails,
                         strictly_better_misses={st: CMP[arm][st]["miss_A"] < CMP[arm][st]["miss_T"] for st in ("all", "dark")})
        CIS[arm] = {st: diff_cis(recs, arm, st) for st in ("all", "dark")}
        REPLAY[arm] = replay_agreement(recs, arm)
        print(f"\n===== {arm} vs teacher =====")
        for st in STRATA:
            c = CMP[arm][st]
            if not c["n_calls"]: continue
            print(f"  [{st:5s}] calls {c['n_calls']:4d} | N {c['N']:4d} (tracked {c['N_kind']['tracked']}, reopened {c['N_kind']['reopened']}) | F4 {c['F4']:4d} | "
                  f"miss T {c['miss_T']} A {c['miss_A']} (reopened T {c['miss_kind_T']['reopened']} A {c['miss_kind_A']['reopened']}) | "
                  f"split T {c['split_T']} A {c['split_A']} | extra T {c['extra_T']} A {c['extra_A']} | "
                  f"median IoU T {c['med_T'] if c['med_T'] is None else round(c['med_T'], 4)} A {c['med_A'] if c['med_A'] is None else round(c['med_A'], 4)} (n {c['n_both']}) | "
                  f"discordant T-only {c['disc_T_only']} / A-only {c['disc_A_only']}")
        for st in ("all", "dark"):
            ci = CIS[arm][st]
            print(f"  [{st}] arm − teacher, clip bootstrap 95 % CI over {ci['n_clips']} clips: misses {fmt_ci(ci['miss'])}, splits {fmt_ci(ci['split'])}, "
                  f"extras {fmt_ci(ci['extra'])}, median IoU {fmt_ci(ci['median_iou'])}")
        print(f"  decision replay (reported): same matched/strikes/whole-animal as teacher on {REPLAY[arm]['same']}/{REPLAY[arm]['n']} passes")
        print(f"  strictly better on misses (information only): {VERD[arm]['strictly_better_misses']}")
        print(f"  VERDICT {arm}: {VERD[arm]['result']}" + ("" if not VERD[arm]["failures"] else "\n    " + "\n    ".join(VERD[arm]["failures"])))
    line = f"Replacement check on {','.join(EVAL_DAYS)}: " + ", ".join(f"{a} {VERD[a]['result']}" for a in CHALLENGERS)
    if SMOKE: line = "SMOKE (not a decision) — " + line
    print("\n" + line)

    # ---- per_call.csv ----
    loc = osp.join(LOCAL, "per_call.csv")
    with open(loc, "w", newline="") as f:
        wr = csv.writer(f)
        hdr = ["call_id", "day", "clip_stem", "frame_idx", "t", "wall", "phase4", "type", "n_gt_trusted", "n_gt_untrusted",
               "n_unscorable", "f4", "opened_stale_at_call"]
        for arm in ARMS: hdr += [f"{arm}_kept", f"{arm}_found", f"{arm}_miss", f"{arm}_split", f"{arm}_extra", f"{arm}_median_iou"]
        for arm in ARMS: hdr += [f"{arm}_replay_matched", f"{arm}_replay_strikes", f"{arm}_replay_whole_unmatched"]
        hdr += ["logged_n_det", "logged_matched", "logged_strikes"]
        wr.writerow(hdr)
        for r in recs:
            row = [r["call_id"], r["day"], r["clip_stem"], r["frame_idx"], r["t"], r["wall"], r["phase4"], r["type"], len(r["gt"]),
                   len(r["untrusted"]), len(r["unscorable"]), int(r["f4"]), len(r["stale"])]
            for arm in ARMS:
                s = r["arms"][arm]; fi = [s["iou"][j] for j in range(len(r["gt"])) if s["found"][j]]
                row += [s["n_kept"], sum(s["found"]), len(s["found"]) - sum(s["found"]), sum(s["split"]),
                        "" if s["n_extra"] is None else s["n_extra"], round(float(np.median(fi)), 4) if fi else ""]
            for arm in ARMS:
                rp = r["replay"].get(arm)
                row += ["" if rp is None else " ".join(map(str, rp["matched"])), "" if rp is None else " ".join(map(str, rp["strikes"])),
                        "" if rp is None else rp["n_whole_unmatched"]]
            lg = r.get("logged")
            row += ["" if lg is None else lg["n_det"], "" if lg is None else " ".join(map(str, lg["matched"])),
                    "" if lg is None else " ".join(map(str, lg["strikes"]))]
            wr.writerow(row)
    put(loc, osp.join(EVAL_DIR, "per_call.csv"))

    # ---- Metrics JSON ----
    MET = dict(eval_tag=EVAL_TAG, smoke=SMOKE, created=datetime.now().isoformat(), eval_days=EVAL_DAYS, eval_root=EVAL_ROOT,
               dataset_tag=DATASET_TAG, manifest_sha256=MANIFEST_SHA, forbidden_eval_days=MANIFEST["forbidden_eval_days"],
               arms=ARMS, arm_meta=META, verdict_line=line, verdict=VERD,
               question="Is each challenger at least as good as the SAM3 teacher detector on the frames where the tracker called it, vs the user-verified GT?",
               not_a_paper_claim="GT-referenced agreement on the user's verified days (engineering acceptance). Not quotable as detector quality (DATA_SPLITS F3); kept counts are instances, descriptive only (F4).",
               calls=dict(n=len(recs), by_day=CJ["counts_by_day"], join_skipped=CJ["join_skipped"],
                          by_phase4=dict(Counter(r["phase4"] for r in recs)), by_type=dict(Counter(r["type"] for r in recs))),
               gt=gt_acc,
               constants=dict(frag_lo=FRAG_LO, over_hi=OVER_HI, a_ref_eval=CJ["a_ref_eval"], FOUND_IOU=FOUND_IOU, SPLIT_IOS=SPLIT_IOS,
                              EXTRA_IOS=EXTRA_IOS, EXTRA_AREA_FRAC=EXTRA_AREA_FRAC, WHOLE_FRAC=WHOLE_FRAC, GT_IOS_MAX=GT_IOS_MAX,
                              ASSOC_MIN=ASSOC_MIN, STRIKE_MAX=STRIKE_MAX, TOL_FRAC=TOL_FRAC, IOU_TOL=IOU_TOL,
                              FAITH_WARN=FAITH_WARN, FAITH_RAISE=FAITH_RAISE, N_BOOT=N_BOOT,
                              provenance=dict(frag_lo="manifest constants (training-derived, Tol_42 §2.7)", over_hi="manifest <- Tol_41 (training days)",
                                              a_ref_eval="this eval day's own stored GT, daylight 1 Hz (trust-filter normalisation only)",
                                              rules="SPEC_eval_v2 §3-§5 (user decision 2026-09-23)")),
               faithfulness=dict(n_det=FAITH, status=FAITH_STATUS, decision_replay=RFAITH),
               comparison={a: CMP[a] for a in CHALLENGERS}, bootstrap_ci=CIS,
               decision_replay={a: dict(n=REPLAY[a]["n"], same=REPLAY[a]["same"], share_same=REPLAY[a]["share_same"],
                                        differing=REPLAY[a]["differing"]) for a in CHALLENGERS},
               tracker_claims_verified=CJ.get("tracker_claims_verified"), manifest_seal_note=SEAL_NOTE)
    loc = osp.join(LOCAL, "metrics.json")
    with open(loc, "w") as f: json.dump(MET, f, indent=1, default=str)
    put(loc, METRICS_PATH)
    print(f"written {METRICS_PATH} and per_call.csv ({time.time() - t0:.0f} s)")
    return recs, CALLS

if not SELFTEST_ONLY and STAGE in ("score", "all"):
    RECS, SCORED_CALLS = run_score()

# COMMAND ----------

# MAGIC %md ## 6 · Optional review sheet — teacher and rfdetr side by side on disagreement calls (nothing depends on it)

# COMMAND ----------

def pick_disagreements(recs, a="rfdetr", ref="teacher", n=SHEET_MAX):
    """Calls where the arms disagree on a trusted calf (found by one only) or on the extra count; dark and seeded first."""
    out = []
    for r in recs:
        if a not in r["arms"]: continue
        T, A = r["arms"][ref], r["arms"][a]
        if any(ft != fa for ft, fa in zip(T["found"], A["found"])) or (r["f4"] and T["n_extra"] != A["n_extra"]):
            out.append(r)
    out.sort(key=lambda r: (r["type"] != "seeded", r["phase4"] not in POOL["dark"], r["wall"]))
    return out[:n]

def render_panel(jpeg, kept, score, gt_rles, label, dark):
    """One panel at SHEET_W px: detections filled in the Tol_17 colour of the GT slot they found (magenta if
    they found none), trusted GT outlines in white; display-only luminance stretch on dusk/night."""
    from PIL import ImageDraw
    from scipy import ndimage
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    sc = SHEET_W / W_FULL; hh = int(round(H_FULL * sc))
    arr = np.asarray(img.resize((SHEET_W, hh), Image.BILINEAR)).astype(np.float32)
    if dark:
        lum = arr.mean(2); lo, hi = np.percentile(lum, STRETCH_LO), np.percentile(lum, STRETCH_HI)
        arr = np.clip((arr - lo) * 255.0 / max(hi - lo, 1.0), 0, 255)
    step = int(round(1 / sc))
    slot_of_det = {di: g["slot"] for j, (g, di) in enumerate(zip(gt_rles, score["det_of"])) if di is not None}
    for i, k in enumerate(kept):
        m = rle_decode(k["rle"])[::step, ::step][:hh, :SHEET_W]
        col = np.array(COLORS_RGB[slot_of_det[i]] if i in slot_of_det else (255, 0, 255), np.float32)
        arr[m] = 0.55 * arr[m] + 0.45 * col
    for g in gt_rles:
        m = rle_decode(g["rle"])[::step, ::step][:hh, :SHEET_W]
        arr[m & ~ndimage.binary_erosion(m, iterations=2)] = 255
    im = Image.fromarray(arr.astype(np.uint8)); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, SHEET_W, 16], fill=(0, 0, 0)); dr.text((4, 2), label + ("  [display stretch]" if dark else ""), fill=(255, 255, 255))
    return im

def run_review_sheet(recs, calls):
    if "rfdetr" not in ARMS: print("review sheet: rfdetr not an arm — skipped"); return
    picks = pick_disagreements(recs)
    if not picks: print("review sheet: no disagreement calls"); return
    os.makedirs(FIG_DIR, exist_ok=True)
    fl = osp.join(LOCAL, "frames_sheet.tar")
    if not osp.exists(fl): stage_file(FRAMES_TAR, fl, "frames.tar")
    dets = {a: read_json_gz_vol(osp.join(EVAL_DIR, f"dets_{a}.json.gz"))["rows"] for a in ("teacher", "rfdetr")}
    by_id = {c["call_id"]: c for c in calls}
    with tarfile.open(fl) as tf:
        for page in range(0, len(picks), SHEET_PER_PAGE):
            rows_im = []
            for r in picks[page:page + SHEET_PER_PAGE]:
                c = by_id[r["call_id"]]; gt, _, _ = build_gt(c)
                tg_keys = {(x["slot"], x["kind"]) for x in r["gt"]}
                tgs = [g for g in gt if (g["slot"], g["kind"]) in tg_keys]      # same order as r["gt"] / score["det_of"] (slot order)
                if len(tgs) != len(r["gt"]): raise RuntimeError(f"{r['call_id']}: trusted GT could not be rebuilt for the sheet")
                jpeg = tf.extractfile(f"{c['clip_stem']}/{c['frame_idx']:07d}.jpg").read()
                dark = r["phase4"] in POOL["dark"]; w = datetime.fromisoformat(r["wall"])
                panels = []
                for a in ("teacher", "rfdetr"):
                    s = r["arms"][a]
                    lab = (f"{a} | {r['day']} {w:%H:%M:%S} {r['phase4']} {r['type']} | found {sum(s['found'])}/{len(s['found'])} "
                           f"split {sum(s['split'])} extra {'-' if s['n_extra'] is None else s['n_extra']}")
                    panels.append(render_panel(jpeg, dets[a][r["fkey"]]["kept"], s, tgs, lab, dark))
                row = Image.new("RGB", (2 * SHEET_W + 8, panels[0].height), (255, 255, 255))
                row.paste(panels[0], (0, 0)); row.paste(panels[1], (SHEET_W + 8, 0)); rows_im.append(row)
            sheet = Image.new("RGB", (rows_im[0].width, sum(x.height + 8 for x in rows_im)), (255, 255, 255)); y = 0
            for x in rows_im: sheet.paste(x, (0, y)); y += x.height + 8
            loc = osp.join(LOCAL, f"disagreements_{page // SHEET_PER_PAGE}.png"); sheet.save(loc)
            put(loc, osp.join(FIG_DIR, osp.basename(loc)))
    print(f"review sheet: {len(picks)} calls on {math.ceil(len(picks) / SHEET_PER_PAGE)} page(s) in {FIG_DIR} "
          "(left = teacher, right = rfdetr; fill colour = the GT slot the detection found, magenta = found none; white outline = trusted GT)")

if not SELFTEST_ONLY and STAGE in ("score", "all") and REVIEW_SHEET:
    run_review_sheet(RECS, SCORED_CALLS)

# COMMAND ----------

# MAGIC %md ## SELFTEST — pure python on synthetic data (runs unconditionally, raises on any failure)

# COMMAND ----------

def _selftest():
    H, Wd = 40, 60
    def rect(y0, y1, x0, x1):
        m = np.zeros((H, Wd), bool); m[y0:y1, x0:x1] = True; return m
    enc = rle_encode
    # --- RLE helpers: run arithmetic == decoded arithmetic ---
    a, b = rect(2, 12, 3, 20), rect(8, 20, 10, 30)
    A, B = runs_l(enc(a)), runs_l(enc(b))
    assert A[2] == a.sum() == rle_area(enc(a)) and (rle_decode(enc(a)) == a).all()
    assert abs(iou_r(A, B) - m_iou(a, b)) < 1e-12 and abs(ios_r(A, B) - m_ios(a, b)) < 1e-12
    assert abs(frac_in(A, B) - (a & b).sum() / a.sum()) < 1e-12
    U = runs_union([A, B]); assert U[2] == (a | b).sum()
    assert inter_fast(runs_l(enc(rect(0, 5, 0, 5))), runs_l(enc(rect(0, 5, 50, 55)))) == 0
    full = np.ones((H, Wd), bool); assert runs_l(enc(full))[2] == H * Wd       # mask starting with a foreground pixel
    # --- phases, stems, times ---
    assert phase4(datetime(2024, 4, 22, 20, 30)) == "dusk" and phase4(datetime(2024, 4, 22, 5, 59)) == "night"
    assert phase4(datetime(2024, 4, 22, 7, 0)) == "dawn" and phase4(datetime(2024, 4, 22, 12, 0)) == "day"
    assert pool_of("dusk") == "dark" and pool_of("dawn") == "dayl" and in_stratum("night", "dark") and not in_stratum("day", "dark")
    assert clip_stem_of("Tol_ch01-20240422-124516-131900.mp4") == "Tol_ch01-20240422-124516-131900"
    assert clip_start("Tol_ch01-20240422-124516-131900") == datetime(2024, 4, 22, 12, 45, 16)
    try: clip_stem_of("x.avi"); raise AssertionError("non-mp4 accepted")
    except RuntimeError: pass
    # --- event -> row join: exact round(t_sec, 2), missing and duplicate ---
    native = 25.01241465028761
    rows = [dict(frame_idx=f, t_sec=round(f / native, 2), masks={}) for f in range(0, 200, 5)]
    by = index_rows(rows)
    t_ev = round(15 / native, 2)                          # the tracker logs t = round(t_sec, 2) from the same fidx
    assert join_event_row(by, t_ev) == (3, None)
    assert join_event_row(by, 0.3) == (None, "missing")
    rows_d = rows + [dict(frame_idx=999, t_sec=rows[3]["t_sec"], masks={})]
    assert join_event_row(index_rows(rows_d), t_ev) == (None, "duplicate")
    # --- parse_calls on the real vocabulary ---
    clip = "Tol_ch01-20240422-124516-131900.mp4"
    evs = [dict(type="seeded", clip=clip, t=0.0, slots=[0, 1, 2, 3], day=True),
           dict(type="pass", clip=clip, t=120.14, forced=None, n_det=4, tracked=4, matched={"0": 0.99, "1": 0.9, "2": 0.9, "3": 0.9},
                struck=[], opened=[], refreshed=[], rebuilt=False),
           dict(type="strike", slot=3, seg_id="3_001", clip=clip, t=315.24, best_overlap=0.0, n=1),
           dict(type="pass", clip=clip, t=315.24, forced="merge_1_3", n_det=3, tracked=3, matched={"0": 0.9, "2": 0.9},
                struck=[], opened=[1], refreshed=[], rebuilt=True),
           dict(type="pass_no_detections", clip=clip, t=615.0, forced=None),
           dict(type="segment_open", seg_id="0_001", slot=0, wall="2024-04-22T12:45:16", reason="seed_day")]
    calls = parse_calls(evs, "20240422")
    assert [c["type"] for c in calls] == ["seeded", "pass", "pass", "pass_no_detections"]
    assert calls[2]["strike_slots"] == [3] and calls[1]["strike_slots"] == [] and calls[0]["phase4"] == "day"
    try: parse_calls([dict(type="pass", clip=clip, t=1.0)], "20240422"); raise AssertionError("incomplete pass accepted")
    except RuntimeError: pass
    try: parse_calls([evs[1], dict(evs[1])], "20240422"); raise AssertionError("duplicate call accepted")
    except RuntimeError: pass
    # --- GT independence: any accepted repair (event or non-empty pass 'repaired') refuses the root ---
    check_no_repairs(evs + [dict(evs[1], repaired=[]), dict(type="partial_repair_rejected", slot=0, clip=clip, t=1.0),
                            dict(type="partial_repair_skipped", slot=0, clip=clip, t=1.0)], "20240422")
    for bad_ev in (dict(evs[1], repaired=[2]), dict(type="partial_repair", slot=2, clip=clip, t=120.14),
                   dict(type="growth_repair", slot=1, clip=clip, t=120.14)):
        try: check_no_repairs(evs + [bad_ev], "20240422"); raise AssertionError(f"repair accepted: {bad_ev['type']}")
        except RuntimeError as e: assert "repair" in str(e)
    # --- cached-dets identity: a different checkpoint / threshold / student is a mismatch ---
    cur = dict(model="RF-DETR-Seg", run_tag="r1", checkpoint_sha256="aa", op_threshold=0.35, resolution=576)
    assert meta_mismatch(dict(cur, arm="rfdetr", calls_sha256="x"), cur) == {}
    assert meta_mismatch(dict(cur, checkpoint_sha256="bb"), cur) == {"checkpoint_sha256": ("bb", "aa")}
    assert set(meta_mismatch(dict(cur, op_threshold=0.45, run_tag="r0"), cur)) == {"op_threshold", "run_tag"}
    assert meta_mismatch(dict(backbone="student best_backbone_student.pt via DummyBackbone"),
                         dict(student_ckpt="best_backbone_student_v2.pt")) == {"student_ckpt": (None, "best_backbone_student_v2.pt")}
    # --- GT construction: tracked / reopened (next row, stale call-row mask ignored) / absent; seeded uses the call row ---
    g0, g1, g2, g3 = rect(0, 10, 0, 12), rect(0, 10, 20, 32), rect(20, 30, 0, 12), rect(20, 30, 20, 32)
    stale = rect(0, 10, 0, 12)                            # slot 3's stale propagation sitting on slot 0's calf
    c_pass = dict(type="pass", event=dict(opened=[1, 3], struck=[]), row_masks=[enc(g0), None, enc(g2), enc(stale)],
                  next_masks=[enc(g0), enc(g1), enc(g2), enc(g3)])
    gt, uns, st_ = build_gt(c_pass)
    assert [(g["slot"], g["kind"], g["src"]) for g in gt] == [(0, "tracked", "call"), (1, "reopened", "next"), (2, "tracked", "call"), (3, "reopened", "next")]
    assert rle_decode(gt[3]["rle"]).sum() == g3.sum() and st_ == [3] and not uns
    gt, uns, _ = build_gt(dict(c_pass, next_masks=None))
    assert [g["slot"] for g in gt] == [0, 2] and sorted(u["reason"] for u in uns) == ["reopened_no_next_row"] * 2
    gt, uns, _ = build_gt(dict(type="pass", event=dict(opened=[], struck=[2]), row_masks=[enc(g0), enc(g1), None, None], next_masks=None))
    assert [g["slot"] for g in gt] == [0, 1] and [u["reason"] for u in uns] == ["absent", "absent"]
    gt, uns, _ = build_gt(dict(type="seeded", event=dict(slots=[0, 1, 2]), row_masks=[enc(g0), enc(g1), None, None], next_masks=[enc(g0)] * 4))
    assert [(g["slot"], g["kind"], g["seeded_by_teacher"]) for g in gt] == [(0, "reopened", True), (1, "reopened", True)]
    assert sorted(u["reason"] for u in uns) == ["absent", "seeded_no_mask_at_call"]
    # --- trust filter: area band and mutual overlap ---
    a_ref = 120.0                                         # 10×12 calf
    tg = [dict(slot=0, rle=enc(g0)), dict(slot=1, rle=enc(rect(0, 3, 20, 25))), dict(slot=2, rle=enc(rect(20, 36, 0, 16))),
          dict(slot=3, rle=enc(g3)), dict(slot=9, rle=enc(rect(22, 30, 22, 32)))]
    trust_filter(tg, a_ref, frag_lo=0.5, over_hi=1.32)
    assert tg[0]["trusted"] and tg[1]["untrusted_reasons"] == ["area_below_frag_lo"]
    assert tg[2]["untrusted_reasons"] == ["area_above_over_hi"]
    assert "overlaps_other_gt" in tg[3]["untrusted_reasons"] and "overlaps_other_gt" in tg[4]["untrusted_reasons"]
    # --- scoring: found / miss / split / extra (F4 only) ---
    G = [runs_l(enc(m)) for m in (g0, g1, g2, g3)]
    left, right = rect(0, 10, 20, 25), rect(0, 10, 27, 32)            # calf 1 split in two pieces, each IoU 0.417 < 0.50
    ext = rect(12, 19, 40, 58)                                        # animal-sized blob on no stored mask (126 px ≥ 0.30×120)
    D = [runs_l(enc(m)) for m in (g0, left, right, g2, ext)]          # calf 3 missed
    union = runs_union(G)
    s = score_call(D, G, union, a_ref, f4=True)
    assert s["found"] == [True, False, True, False] and s["split"] == [False, True, False, False]
    assert s["n_extra"] == 1 and s["extra_idx"] == [4] and s["det_of"][0] == 0 and abs(s["iou"][0] - 1.0) < 1e-12
    s3 = score_call(D, G[:3], runs_union(G[:3]), a_ref, f4=False)     # 3-GT frame: the extra must be ignored
    assert s3["n_extra"] is None and s3["extra_idx"] is None
    tiny = runs_l(enc(rect(12, 14, 40, 44)))                          # small blob: below 0.30×A_ref -> not an extra
    assert score_call([tiny], G, union, a_ref, True)["n_extra"] == 0
    assert score_call([], G, union, a_ref, True)["found"] == [False] * 4
    # --- call_record end to end (F4 via the next-row GT; stale mask in the union) ---
    call = dict(call_id="d|c|1.00|pass", day="20240422", clip_stem="c", frame_idx=5, t=1.0, wall="2024-04-22T12:00:00", phase4="day",
                type="pass", event=dict(opened=[1, 3], struck=[], matched={"0": 0.9, "2": 0.9}, tracked=2, n_det=4),
                strike_slots=[], row_masks=c_pass["row_masks"], next_masks=c_pass["next_masks"], prev_masks={})
    kept = lambda ms: dict(kept=[dict(rle=enc(m), score=0.9, area=int(m.sum())) for m in ms])
    dets = {"teacher": {"c:0000005": kept([g0, g1, g2, g3])}, "rfdetr": {"c:0000005": kept([g0, g1, g2])}}
    rec = call_record(call, dets, {"20240422": a_ref}, {"dayl": 0.5, "dark": 0.5}, 1.32, ["teacher", "rfdetr"])
    assert rec["f4"] and rec["stale"] == [3] and [g["kind"] for g in rec["gt"]] == ["tracked", "reopened", "tracked", "reopened"]
    assert rec["arms"]["teacher"]["found"] == [True] * 4 and rec["arms"]["rfdetr"]["found"] == [True, True, True, False]
    assert rec["trk_slots"] == [0, 2] and rec["replay"]["teacher"]["matched"] == [0, 2] and rec["logged"]["matched"] == [0, 2]
    assert rec["replay"]["teacher"]["n_whole_unmatched"] == 2 and rec["replay"]["rfdetr"]["n_whole_unmatched"] == 1
    try: call_record(call, {"teacher": {}, "rfdetr": {}}, {"20240422": a_ref}, {"dayl": .5, "dark": .5}, 1.32, ["teacher"]); raise AssertionError
    except RuntimeError: pass
    # --- decision replay on a synthetic pass matching the logged matched / struck ---
    trk = {0: runs_l(enc(g0)), 1: runs_l(enc(g1)), 2: runs_l(enc(rect(30, 40, 45, 60)))}   # slot 2 contradicted by every det
    rp = replay_pass(trk, [runs_l(enc(g0)), runs_l(enc(rect(0, 10, 21, 33))), runs_l(enc(g3))], a_ref)
    assert rp["matched"] == [0, 1] and rp["strikes"] == [2] and rp["n_whole_unmatched"] == 1
    logged = dict(matched=sorted(int(k) for k in {"0": 0.99, "1": 0.95}), strikes=[2])
    assert rp["matched"] == logged["matched"] and rp["strikes"] == logged["strikes"]
    assert replay_pass(trk, [], a_ref) == dict(matched=[], strikes=[], n_whole_unmatched=0, no_detections=True)
    amb = replay_pass({0: runs_l(enc(g0))}, [runs_l(enc(rect(0, 10, 10, 22)))], a_ref)   # overlap 0.10..0.30: neither
    assert amb["matched"] == [] and amb["strikes"] == []
    ev_s = dict(type="pass", event=dict(opened=[2], struck=[1, 2]), row_masks=[enc(g0), None, enc(stale), None],
                prev_masks={"1": enc(g1)})
    assert sorted(replay_tracks(ev_s)) == [0, 1, 2]                   # struck-not-reopened uses the prior row; struck+reopened the call row
    assert prev_masks_for([dict(masks={"1": enc(g1)}), dict(masks={}), dict(masks={})], 2, [1]) == {"1": enc(g1)}
    assert prev_masks_for([dict(masks={"1": enc(g1)})] + [dict(masks={})] * 11, 12, [1]) == {}   # beyond FRESH_ROWS
    # --- verdict rule: PASS, FAIL on dark only, the max(1, ·) floor ---
    def cmpd(N, F4, mT, mA, sT=0, sA=0, eT=0, eA=0, iT=0.9, iA=0.9, nb=10, sf=()):
        return dict(N=N, F4=F4, miss_T=mT, miss_A=mA, split_T=sT, split_A=sA, extra_T=eT, extra_A=eA,
                    med_T=iT, med_A=iA, n_both=nb, seed_fail_A=list(sf))
    ok, f = verdict_rule({"all": cmpd(400, 90, 5, 8), "dark": cmpd(50, 10, 1, 2)}); assert ok and not f
    ok, f = verdict_rule({"all": cmpd(400, 90, 5, 8), "dark": cmpd(50, 10, 1, 3)})
    assert not ok and len(f) == 1 and f[0].startswith("[dark] (1)")
    assert verdict_rule({"all": cmpd(400, 90, 5, 9), "dark": cmpd(50, 10, 1, 1)})[0]           # +4 ≤ 0.01×400 = 4
    assert not verdict_rule({"all": cmpd(400, 90, 5, 10), "dark": cmpd(50, 10, 1, 1)})[0]      # +5 > 4
    assert verdict_rule({"all": cmpd(40, 9, 0, 1, eT=0, eA=1), "dark": cmpd(40, 9, 0, 1)})[0]   # floor of 1
    assert not verdict_rule({"all": cmpd(40, 9, 0, 0, sT=0, sA=2), "dark": cmpd(40, 9, 0, 0)})[0]
    assert not verdict_rule({"all": cmpd(40, 9, 0, 0, eT=0, eA=2), "dark": cmpd(40, 9, 0, 0)})[0]
    assert verdict_rule({"all": cmpd(40, 9, 0, 0, iT=0.90, iA=0.881), "dark": cmpd(40, 9, 0, 0)})[0]
    assert not verdict_rule({"all": cmpd(40, 9, 0, 0, iT=0.90, iA=0.879), "dark": cmpd(40, 9, 0, 0)})[0]
    assert not verdict_rule({"all": cmpd(40, 9, 0, 0, sf=["seed"]), "dark": cmpd(40, 9, 0, 0)})[0]
    ok, f = verdict_rule({"all": cmpd(40, 9, 0, 0), "dark": cmpd(0, 0, 0, 0, nb=0)}); assert not ok and "cannot certify" in f[0]
    assert not verdict_rule({"all": cmpd(40, 9, 0, 0, nb=0), "dark": cmpd(40, 9, 0, 0)})[0]
    # --- compare() on records, discordance, seeds ---
    def R(phase, typ, fT, fA, clip="c1", f4=False, eT=0, eA=0):
        return dict(call_id=f"{phase}{typ}{clip}{len(fT)}{fA}", phase4=phase, type=typ, clip_stem=clip, f4=f4,
                    gt=[dict(slot=i, kind="tracked", seeded_by_teacher=False) for i in range(len(fT))],
                    arms={"teacher": dict(found=fT, iou=[0.9] * len(fT), split=[False] * len(fT), n_extra=eT if f4 else None, n_kept=len(fT)),
                          "rfdetr": dict(found=fA, iou=[0.95] * len(fA), split=[False] * len(fA), n_extra=eA if f4 else None, n_kept=len(fA))},
                    replay={})
    recs = [R("day", "pass", [True] * 4, [True] * 4, f4=True, eT=0, eA=1), R("night", "pass", [True, True], [True, False], clip="c2"),
            R("dusk", "seeded", [True, True], [True, True], clip="c3"), R("night", "pass", [False], [True], clip="c2")]
    c_all, c_dark = compare(recs, "rfdetr", "all"), compare(recs, "rfdetr", "dark")
    assert (c_all["N"], c_all["miss_T"], c_all["miss_A"], c_all["F4"], c_all["extra_A"]) == (9, 1, 1, 1, 1)
    assert (c_dark["N"], c_dark["disc_T_only"], c_dark["disc_A_only"], c_dark["n_seed"]) == (5, 1, 1, 1)
    assert c_all["n_both"] == 7 and abs(c_all["med_A"] - 0.95) < 1e-12
    ci = diff_cis(recs, "rfdetr", "all", n_boot=200)
    assert ci["n_clips"] == 3 and ci["miss"][0] == 0 and ci["extra"][0] == 1 and ci["miss"][1] <= ci["miss"][2]
    # --- faithfulness: rates by phase, warn / raise thresholds ---
    fr = [dict(type="pass", phase4="day", arms={"teacher": dict(n_kept=4)}, logged=dict(n_det=4)) for _ in range(19)]
    fr += [dict(type="pass", phase4="day", arms={"teacher": dict(n_kept=5)}, logged=dict(n_det=4))]
    rates = ndet_faithfulness(fr); assert rates["all"]["rate"] == 0.95 and rates["day"]["n"] == 20
    assert faithfulness_gate(rates, False) == dict(warn=[], below_raise=[])
    bad = ndet_faithfulness(fr[:16] + [fr[-1]] * 4); assert abs(bad["all"]["rate"] - 0.8) < 1e-12
    try: faithfulness_gate(bad, False); raise AssertionError("faithfulness 0.80 must raise")
    except RuntimeError: pass
    assert faithfulness_gate(bad, True)["below_raise"] == ["all", "day"]
    # per-phase shortfall with a faithful pooled rate: reported, NOT raised (revised gate 2026-09-24)
    mix = [dict(type="pass", phase4="day", arms={"teacher": dict(n_kept=4)}, logged=dict(n_det=4)) for _ in range(90)]
    mix += [dict(type="pass", phase4="night", arms={"teacher": dict(n_kept=4 if i < 7 else 5)}, logged=dict(n_det=4)) for i in range(10)]
    mrates = ndet_faithfulness(mix); assert abs(mrates["night"]["rate"] - 0.7) < 1e-12 and mrates["all"]["rate"] >= 0.85
    assert faithfulness_gate(mrates, False)["below_raise"] == ["night"]
    # --- seed check uses the N_CALVES largest kept detections only (the tracker seeds dets[:N_CALVES]) ---
    def _sq(x0, y0, s=120):
        m = np.zeros((1296, 2304), bool); m[y0:y0 + s, x0:x0 + s] = True; return runs_l(rle_encode(m))
    _g = [_sq(100 + 300 * i, 100) for i in range(4)]
    _d_all = [_sq(100 + 300 * i, 100) for i in range(3)] + [_sq(1500, 800, 200), _sq(100 + 900, 100)]   # 4th calf found only by the 5th (smaller) detection
    assert all(score_call(_d_all, _g, _g[0], 1e9, False)["found"])
    assert not all(score_call(_d_all[:N_CALVES], _g, _g[0], 1e9, False)["found"])
    # --- leakage guard: a manifest day is refused; only the verified pair is allowed ---
    man = dict(tag="t", source_days=["20240418", "20240419", "20240422"], forbidden_eval_days=["20240418", "20240419", "20240422"])
    try: check_eval_days(["20240422", "20240423"], man); raise AssertionError("manifest day accepted")
    except RuntimeError as e: assert "leakage" in str(e)
    assert check_eval_days(["20240423"], man) == ["20240423"]
    for badd in (["20240424"], [], ["20240423", "20240423"]):
        try: check_eval_days(badd, man); raise AssertionError(f"{badd} accepted")
        except RuntimeError: pass
    try: check_eval_days(["20240423"], dict(source_days=[])); raise AssertionError("manifest without forbidden_eval_days accepted")
    except RuntimeError: pass
    # --- seal: sealed passes; unsealed only with the Tol_43 below_floor waiver recorded for THIS manifest ---
    assert seal_admissible(dict(sealed=True), "s" * 64, None) is None
    mu = dict(tag="t", sealed=False, counts=dict(below_floor=True, train_full=4000))
    tsw = dict(run_tag="r", dataset_manifest_sha256="s" * 64, manifest_sealed=False, seal_waiver=["below_floor"])
    assert "UNSEALED" in seal_admissible(mu, "s" * 64, tsw)
    for m_, ts_, needle in ((mu, None, "no Tol_43"), (mu, {**tsw, "dataset_manifest_sha256": "x" * 64}, "trained on manifest"),
                            (mu, {**tsw, "seal_waiver": []}, "no below_floor waiver"), (mu, {**tsw, "manifest_sealed": True}, "no below_floor waiver"),
                            (dict(mu, counts=dict(below_floor=False)), tsw, "below_floor is false")):
        try: seal_admissible(m_, "s" * 64, ts_); raise AssertionError(f"unsealed accepted: {needle}")
        except RuntimeError as e: assert needle in str(e), (needle, str(e))
    # --- manifest constants as Tol_42 writes them ---
    mc = dict(constants={"frag_lo[dayl]": dict(value=0.55, provenance="x"), "frag_lo[dark]": dict(value=0.45), "over_hi": dict(value=1.32)})
    assert manifest_const(mc, "frag_lo[dark]") == 0.45 and manifest_const(mc, "over_hi") == 1.32
    try: manifest_const(mc, "A_ref"); raise AssertionError
    except RuntimeError: pass
    # --- smoke pick covers every branch ---
    sp = smoke_pick(calls, 3); assert calls[0] in sp and calls[2] in sp and len(sp) == 3
    # --- post-chain (the RF-DETR arm) ---
    big, frag = rect(0, 30, 0, 40), rect(0, 10, 0, 10)
    kc, nf = post_chain([(frag, 0.9), (big, 0.5), (rect(35, 40, 50, 60), 0.99)], topk=6, min_px=60)
    assert nf == 1 and len(kc) == 1 and kc[0][0] is big
    # --- optional pycocotools round-trip of the project RLE (not needed by this notebook) ---
    try:
        from pycocotools import mask as mask_util
        r = enc(b); coco = mask_util.frPyObjects({"size": r["size"], "counts": r["counts"]}, H, Wd)
        assert (mask_util.decode(coco).astype(bool) == b).all()
        print("SELFTEST: pycocotools round-trip OK")
    except ImportError:
        print("SELFTEST: pycocotools unavailable — skipping the COCO round-trip only")
    print("SELFTEST PASSED (join, repair refusal, cached-dets identity, GT construction, trust filter, found/miss/split/extra, "
          "verdict, decision replay, faithfulness, leakage guard, seal waiver, manifest constants, smoke pick, post-chain)")

if SELFTEST_ONLY:
    _selftest()
else:
    # production run: the synthetic cases print deliberate warnings — keep them out of the job log after the real verdict
    import contextlib
    _buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_buf): _selftest()
    except Exception:
        print("SELFTEST (synthetic) FAILED — its output:\n" + _buf.getvalue()); raise
    print("SELFTEST (synthetic data, not this run's results): passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading it
# MAGIC
# MAGIC **The one line to look for** is printed at the end of `stage=score`:
# MAGIC `Replacement check on 20240422,20240423: rfdetr PASS|FAIL, student_sam3 PASS|FAIL`.
# MAGIC
# MAGIC **What was compared.** Every time the tracker asked the SAM3 teacher detector for detections on 04-22 and
# MAGIC 04-23 (854 calls: the periodic passes every 2 min by day / 5 min by night, the merge-forced passes, and the
# MAGIC two seedings), this notebook takes the exact frame, runs the teacher again and runs each challenger on it,
# MAGIC and checks both against the calves the tracker was holding at that moment — the masks whose identities you
# MAGIC verified. A calf the tracker had lost at that moment has no reference and is simply not scored, which can
# MAGIC only favour the teacher, so a PASS is a conservative answer.
# MAGIC
# MAGIC **PASS means**, in both all calls and the dark ones (dusk + night): the challenger misses at most about one
# MAGIC more calf per hundred than the teacher (never fewer than one allowed), splits a calf into two pieces at most
# MAGIC about one more time per hundred, puts at most about one more detection on nothing per hundred fully-covered
# MAGIC calls, draws calf outlines no more than 0.02 IoU worse (median), and finds every calf at the two seedings.
# MAGIC A FAIL lists exactly which of these failed, by how much.
# MAGIC
# MAGIC **Two numbers to sanity-check first.** (1) *Replay faithfulness*: the teacher, re-run here on the cached
# MAGIC JPEG, should reproduce the number of detections the tracker logged on ≥ 95 % of passes; below 85 % the
# MAGIC notebook stops, because the replay would not be showing what the tracker saw. (2) *Trusted GT per phase*:
# MAGIC a stored mask is only used as reference if it is calf-sized (the training-day band) and does not overlap
# MAGIC another calf's mask — your identity check does not certify every mask's outline, so fragments and merged
# MAGIC masks are set aside and counted, not scored.
# MAGIC
# MAGIC **How to read the rest.** "reopened" rows are calves the tracker had just re-seeded from the teacher's own
# MAGIC detection, so they lean towards the teacher — they are kept (they are the most consequential calls) and
# MAGIC reported separately. The discordant counts say how many calves only one arm found; the bootstrap intervals
# MAGIC (resampling whole clips) show how much the differences could move. The decision-replay line says how often
# MAGIC the challenger would have led the tracker to the same matches, strikes and "unclaimed whole animal"
# MAGIC detections as the teacher did — informative, not part of PASS/FAIL. The optional sheet
# MAGIC `Figures/detector_eval/<eval_tag>/disagreements_<k>.png` shows up to 24 calls where the two arms disagree
# MAGIC (left teacher, right RF-DETR; each detection coloured by the calf it found, magenta if none; white outlines
# MAGIC are the reference calves) if you want to glance at them; nothing depends on it.
# MAGIC
# MAGIC **What it does not say.** Nothing about tracking quality, re-ID or other days, and it is not a detector-quality
# MAGIC number for the paper (the reference masks came from a tracker seeded by the teacher). Detection counts are
# MAGIC printed only as description: a count of 4 can be four calves or one calf in two pieces plus one missed.
# MAGIC One tracker detail changed how the reference is built (header, point 2): when a pass re-opens a slot, the
# MAGIC tracker's stored row at that instant can still hold the old, closed object's mask, so the reference for a
# MAGIC re-opened calf is always taken from the next row (0.2 s later).
