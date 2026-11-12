# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N3 **v4** — anonymous segment tracking (no online identity)
# MAGIC
# MAGIC Implements `DESIGN_v4_7day_tracking.md` §3. The two prior generations failed symmetrically:
# MAGIC gen-1 decided identity online from ~0.02-margin appearance (43.8 % of frames mislabelled);
# MAGIC gen-2 carried stale masks across continuity breaks and re-fed them to SAM3 (masks migrated
# MAGIC onto the ground). v4 separates **slot** (session capacity) / **segment** (unbroken spatial
# MAGIC continuity) / **identity** (offline only):
# MAGIC
# MAGIC - **INV-1 seed whitelist** — SAM3 is only ever prompted with a mask it emitted within
# MAGIC   `FRESH_SEC` (2 s; ≤ `CLIP_CARRY_MAX` 10 s at a clip boundary, validated by the next
# MAGIC   detector pass) or a detector mask from the current frame. There is NO fallback path.
# MAGIC - **INV-2 no online identity** — no pool, no PERM, no confirmation. Identity is decided
# MAGIC   offline (Tol_15/16) from per-segment aggregated embeddings.
# MAGIC - **INV-3 absence is honest** — a slot without a genuine mask emits None.
# MAGIC - Loss is measured from the real per-frame mask stream; drift-onto-ground is caught by
# MAGIC   **detector-contradiction strikes** (SAM3 confidently masks the ground while the detector
# MAGIC   keeps saying no calf is there); merges close only the **jumper** (self-IoU discontinuity).
# MAGIC - **INV-7** — the run aborts itself on the v3 failure signature (association acceptance,
# MAGIC   accepted-overlap median, genuine-mask coverage), instead of completing plausibly.
# MAGIC - **Partial-band repair (2026-09-02)** — a mask sitting at a sustained fraction of its
# MAGIC   DAY-anchored reference area is replaced from the next deduped detection that spatially
# MAGIC   contains it (hard gates; day-anchored, never rolling-median-anchored). Fixes the 04-23
# MAGIC   dusk plateau (2-hour 0.3x fragments invisible to the rolling-median guard, untouchable
# MAGIC   by the day-only refresh). `partial_repair=false` reproduces legacy tracking behavior
# MAGIC   exactly (masks/segments/state); events and telemetry then carry only the flag itself.
# MAGIC   **Revision 2 (2026-09-06):** repairs only in a settled night scene (stationary mask,
# MAGIC   no pen-wide chaos event for 5 min) and every repair is a SEGMENT BOUNDARY — the first
# MAGIC   patch laundered dusk fragment-hopping into unbroken segments (a silent identity switch,
# MAGIC   user-verified). Growth guard added (over-extension is the mirror failure).
# MAGIC
# MAGIC Output (`Tracking_v4/<day>/`): slot-keyed RLE masks per frame, `segments.json` (the break
# MAGIC record), `observations.npz` (+meta, per-obs seg_id + gate fields), `crops/`, `events.jsonl`,
# MAGIC and a telemetry digest. Nothing here is identity-keyed.
# MAGIC
# MAGIC ---
# MAGIC ## Tol_47 · tracker **v5** = the v4 file above + changes C1–C15 <!-- [v5-C1] -->
# MAGIC Constraints fixed 2026-09-24: 5 fps, GPU-memory-fit claim, no pixel-changing speed-ups this round.
# MAGIC Every changed hunk carries a `# [v5-Cn]` marker. All v5 *decisions* live in `tol_v5_core.py` (imported from
# MAGIC this notebook's workspace directory, sha256 echoed): RF-DETR-Seg detector (C3), the dark step — the detector
# MAGIC also fires while a calf is dark and a detection at its last healthy place becomes a NEW `dark_link` segment,
# MAGIC by day a moved calf can be fast-opened (C11–C12) — the `v5_fixes` guards (C13), `processed_frames` pop +
# MAGIC memory tripwires (C5–C6), exact fast masks and a decode thread (C7–C8). Identity stays offline (INV-2).
# MAGIC **Regression mode** (`detector=sam3 dark_fire=off detect_night_sec=300 v5_fixes=off`, pop/exact_fast on,
# MAGIC `decode=seek_thread`) must reproduce Tol_03 bit-for-bit (E6). `mode=equiv` runs E1–E6 and never writes a
# MAGIC tracking root. Local check: `TOL_SELFTEST=1 python Tol_47_track_v5.py` (no torch/cv2/GPU needed).
# MAGIC <!-- [v5-C2] the install cell below adds rfdetr and forces headless OpenCV (verbatim Tol_44:70-72) -->

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm" "scipy" "rfdetr==1.10.1" "pycocotools" "peft"
# MAGIC %pip uninstall -y opencv-python opencv-contrib-python
# MAGIC %pip install --no-deps --force-reinstall "opencv-python-headless==4.11.0.86"

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass   # subprocess-worker mode (multiplex launcher): plain python, deps inherited via sys.executable

# COMMAND ----------

import os, os.path as osp, re, json, gzip, time, gc, glob, shutil
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys, hashlib, subprocess, importlib.metadata                              # [v5-C1]
# [v5-C2] allocator config BEFORE this process touches CUDA (Tol_03:120 initialises it); a job/launcher value wins
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # [v5-C2]
# [v5-C15] local SELFTEST path -- TOL_SELFTEST=1 python Tol_47_track_v5.py -- needs no torch / cv2 / GPU / Volume:
# tol_v5_core.selftest() plus a static audit of this file (review checklist 1, 3, 4, 5, 6, 8), then exits here.
if os.environ.get("TOL_SELFTEST") == "1":
    _here = osp.dirname(osp.abspath(__file__))
    if _here not in sys.path: sys.path.insert(0, _here)
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import tol_v5_core as _core_st
    _core_st.selftest(data=os.environ.get("TOL_SELFTEST_DATA") or None,
                      quick=os.environ.get("TOL_SELFTEST_DATA") is None)
    _core_st.audit_tol47(osp.abspath(__file__))
    print("Tol_47 SELFTEST PASSED")
    raise SystemExit(0)
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
import timm
from timm.data import resolve_data_config
from torchvision import transforms
from huggingface_hub import login

# Params come from widgets in notebook mode, or TOLV4_* env vars in subprocess-worker mode
# (the multiplex launcher runs several day-trackers as subprocesses on one A100 — a SHARED job
# cluster exposes no GPU to its task REPLs on this workspace, measured 2026-08-17, so process-
# level sharing inside one task is the only consolidation mechanism available).
IS_NOTEBOOK = True
try:
    dbutils
except NameError:
    IS_NOTEBOOK = False

# [v5-C1] tol_v5_core -- the single source of truth for every v5 decision -- is imported from EXACTLY ONE
# directory, so the sha256 the launcher pins is the sha256 of the module that runs (the old search
# list put the hard-coded Workspace directory AHEAD of the staged, pinned copy in subprocess mode):
#   subprocess mode (Tol_47b): the directory of this file = the staged CODE dir whose copy the launcher hashed;
#   notebook mode (direct task): this notebook's own workspace directory (the hard-coded path only if the notebook
#   context is unreadable). Nothing else on sys.path can shadow it, and the resolved file is asserted.
def _core_dir():
    if not IS_NOTEBOOK:
        return osp.dirname(osp.abspath(__file__))
    try:
        return "/Workspace" + osp.dirname(
            dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
    except Exception as e:
        print(f"⚠ notebook path unreadable ({e!r}): tol_v5_core from the default Workspace directory")
        return "/Workspace/Users/<DATABRICKS_USER>/Model Distillation/Testing on Tol2"
CORE_DIR = _core_dir()
if not osp.isfile(osp.join(CORE_DIR, "tol_v5_core.py")):
    raise RuntimeError(f"tol_v5_core.py is not in {CORE_DIR} (deploy it beside Tol_47: workspace FILE / staged copy)")
while CORE_DIR in sys.path: sys.path.remove(CORE_DIR)
sys.path.insert(0, CORE_DIR); sys.modules.pop("tol_v5_core", None)
import tol_v5_core as core                                                       # [v5-C1]
if osp.normcase(osp.realpath(osp.dirname(osp.abspath(core.__file__)))) != osp.normcase(osp.realpath(CORE_DIR)):
    raise RuntimeError(f"tol_v5_core resolved to {core.__file__}, not {CORE_DIR}: the sha pin would not cover it")
CORE_SHA = core.sha256_file(core.__file__)
print(f"tol_v5_core {core.CORE_VERSION} from {core.__file__} sha256 {CORE_SHA}")

def P(name, default):
    if IS_NOTEBOOK:
        try: dbutils.widgets.text(name, default)
        except Exception: pass
        return (dbutils.widgets.get(name) or default)
    return os.environ.get(core.ENV_PREFIX + name.upper(), default)             # [v5-C1] TOLV5_* (Tol_03: TOLV4_*)

# [v5-C1] every widget goes through tol_v5_core.config_from_flags (typed; unknown names / values RAISE). The
# unchanged widgets keep Tol_03's names and meaning; changed v5 defaults: detect_night_sec 120, student_ckpt v2,
# track_root Tracking_v5a10. Regression values: core.REGRESSION_FLAGS.
_RAW = {_n: P(_n, _d) for _n, (_d, _k, _c) in core.WIDGETS.items()}
CFG = core.config_from_flags(_RAW); _V = CFG["values"]
MODE       = _V["mode"]
STAGE      = _V["stage"]
MAX_CLIPS  = _V["max_clips"]
MAX_FRAMES = _V["max_frames"]
FROM_HHMM  = _V["from_hhmm"]
TO_HHMM    = _V["to_hhmm"]
TRACK_ROOT = _V["track_root"]
TRACK_FPS  = _V["track_fps"]
ALLOW_NONSTD_FPS = _V["allow_nonstandard_fps"]
DETECT_DAY_SEC   = _V["detect_day_sec"]
DETECT_NIGHT_SEC = _V["detect_night_sec"]
REFRESH_MIN_SEC  = _V["refresh_min_sec"]
EMBED_EVERY_SEC  = _V["embed_every_sec"]
CROP_EVERY_SEC   = _V["crop_every_sec"]
STAGE_SVC  = os.environ.get("TOLV5_STAGE_SVC", "")     # set by the launcher: staging-service dir
# staging-service request key: defaults to the stage (Tol_03 protocol); Tol_47b sets TOLV5_SVC_KEY when two arms of
# the SAME day share one node (E6 ref || reg), otherwise their req__/ready__ files would collide
SVC_KEY    = os.environ.get("TOLV5_SVC_KEY", "") or STAGE
# backbone=teacher runs SAM3's ORIGINAL vision backbone (no student swap) — used to produce the
# visually-verified ground-truth track of 04-22 against which every distilled student is scored.
BACKBONE   = _V["backbone"]
STUDENT_CKPT = _V["student_ckpt"]
SAM3_ID    = _V["sam3_model"]
# partial_repair=false runs the pre-2026-09-02 tracking behavior identically (masks, segments,
# state transitions; events/telemetry carry only the flag itself). Teacher GT runs MUST use
# false so their provenance matches the frozen 04-22 GT; student production runs use true.
PARTIAL_REPAIR = _V["partial_repair"]
DETECTOR, DET_RUN_TAG = _V["detector"], _V["det_run_tag"]
DARK_FIRE, DARK_LINK, DARK_OPEN = CFG["dark_fire"], CFG["dark_link"], CFG["dark_open"]   # link/open off unless dark_fire
DARK_REPEAT_SEC, DARK_FIRST_SEC = CFG["dark_repeat_sec"], CFG["dark_first_sec"]        # D3: one value, both roles
V5_FIXES, POP_FRAMES, EXACT_FAST = _V["v5_fixes"], _V["pop_frames"], _V["exact_fast"]
DECODE, GPU_TF, NUMERIC_CHANGE = _V["decode"], _V["gpu_tf"], CFG["numeric_change"]
MEM_CAP_GIB = _V["mem_cap_gib"]
MEM_ASSERT_ALLOC_GIB, MEM_ASSERT_RESERVED_GIB = _V["mem_assert_alloc_gib"], _V["mem_assert_reserved_gib"]
MEM_ASSERT_SESSION_FRAMES = _V["mem_assert_session_frames"]
CUDA_PROBE_RETRIES, CUDA_PROBE_SLEEP_S = _V["cuda_probe_retries"], _V["cuda_probe_sleep_s"]
ALLOW_MIXED_GPU = _V["allow_mixed_gpu"]
DCFG = core.dark_cfg(CFG)
print(f"v5 MODE={MODE} detector={DETECTOR} dark_fire={DARK_FIRE} (link {DARK_LINK}, open {DARK_OPEN}, repeat/first "
      f"{DARK_REPEAT_SEC:g}s) v5_fixes={V5_FIXES} pop={POP_FRAMES} exact_fast={EXACT_FAST} decode={DECODE} "
      f"gpu_tf={GPU_TF} regression={CFG['is_regression']}")
if NUMERIC_CHANGE:
    print("⚠⚠⚠ allow_numeric_change=true: decode/transform CHANGE PIXELS OR FEATURES vs Tol_03 (D7 deferred this) ⚠⚠⚠")
else:
    print("allow_numeric_change=false: every enabled speed-up is bit-identical to Tol_03 (E1/E2/E4)")
assert BACKBONE in ("student", "teacher"), f"backbone must be student|teacher, got {BACKBONE}"
if BACKBONE == "teacher" and PARTIAL_REPAIR:
    raise RuntimeError("backbone=teacher is a GT run: set partial_repair=false explicitly so its "
                       "provenance matches the frozen 04-22 teacher GT (review 2026-09-02).")
SMOKE = bool(MAX_CLIPS or MAX_FRAMES)

# ---- INV-6: the paper's reference cadence is 5 Hz. The silent 5 -> 2 fps drop cost a full
# re-track; any deviation must be explicit and shouted, never buried in a run name.
if abs(TRACK_FPS - 5.0) > 1e-6 and not ALLOW_NONSTD_FPS:
    raise RuntimeError(f"track_fps={TRACK_FPS} deviates from the 5.0 reference. "
                       f"Set allow_nonstandard_fps=true ONLY with explicit user sign-off.")
if ALLOW_NONSTD_FPS:
    print(f"⚠⚠⚠ NON-STANDARD CADENCE {TRACK_FPS} fps — explicitly authorised, logged in outputs")

if IS_NOTEBOOK:
    login(token=dbutils.secrets.get(scope="hf", key="token").strip().strip('"').strip("'").strip())
else:
    login(token=os.environ["HF_TOKEN"].strip())
# ---- [v5-C2] allocator config asserted, then a SUBPROCESS CUDA probe before this process touches CUDA (3 of 18 A10
# boots failed at CUDA start-up); a task retry must land on a NEW cluster (recorded + checked) ----------
if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    raise RuntimeError(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r} lacks expandable_segments:True")
_CUDA_FAIL = osp.join("/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id", "Tol",
                      "Metrics_v5_cuda_probe_failures.json")
def _cluster_id():
    cid = os.environ.get("DB_CLUSTER_ID", "")
    if not cid and IS_NOTEBOOK:
        try: cid = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")   # noqa: F821 (notebook global)
        except Exception: cid = ""
    return cid or "unknown"
def _cuda_probe():
    rec = {"cluster_id": CLUSTER_ID, "retries": CUDA_PROBE_RETRIES}
    if osp.exists(_CUDA_FAIL):
        try: _fails = json.load(open(_CUDA_FAIL))
        except Exception: _fails = []
        if CLUSTER_ID != "unknown" and any(f.get("cluster_id") == CLUSTER_ID for f in _fails):
            raise RuntimeError(f"cluster {CLUSTER_ID} already failed a CUDA probe -- a retry must land on a NEW cluster")
    if CUDA_PROBE_RETRIES <= 0:
        print("⚠ cuda_probe_retries=0: the subprocess CUDA probe is SKIPPED"); rec["skipped"] = True; return rec
    code = "import torch; torch.zeros(1, device='cuda'); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0))"
    last = None
    for k in range(CUDA_PROBE_RETRIES):
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                rec.update(ok=True, attempt=k + 1, gpu=(r.stdout.strip().splitlines() or ["?"])[-1]); break
            last = (r.stderr or r.stdout or "")[-600:]
        except Exception as e:
            last = repr(e)
        print(f"⚠ CUDA probe attempt {k + 1}/{CUDA_PROBE_RETRIES} failed: {last}", flush=True)
        if k + 1 < CUDA_PROBE_RETRIES: time.sleep(CUDA_PROBE_SLEEP_S)
    else:
        try:
            _fails = json.load(open(_CUDA_FAIL)) if osp.exists(_CUDA_FAIL) else []
            _fails.append({"cluster_id": CLUSTER_ID, "time": datetime.now().isoformat(), "stage": STAGE,
                           "track_root": TRACK_ROOT, "error": last})
            with open(_CUDA_FAIL, "w") as f: json.dump(_fails, f, indent=1)
        except Exception as e:
            print(f"⚠ could not record the probe failure: {e}")
        raise RuntimeError(f"CUDA start-up failed {CUDA_PROBE_RETRIES}x on cluster {CLUSTER_ID}: {last}")
    try:                                    # driver + vGPU licence lines only
        q = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=120).stdout
        rec["nvidia_smi"] = [l.strip() for l in q.splitlines()
                             if any(t in l for t in ("Driver Version", "CUDA Version", "License Status",
                                                     "vGPU Software Licensed Product", "Product Name"))][:12]
    except Exception as e:
        rec["nvidia_smi"] = [f"unavailable: {e}"]
    print(f"CUDA probe: {rec}")
    return rec
CLUSTER_ID = _cluster_id()
PROBE = _cuda_probe()
device = torch.device("cuda")
DTYPE = torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16
print(f"GPU {torch.cuda.get_device_name(0)} -> tracking dtype {DTYPE}")
# ---- [v5-C6] A10 emulation cap + [v5-C3] explicit fp32 precision + [v5-C2] version asserts ----
GPU_NAME = torch.cuda.get_device_name(0); GPU_CLASS = core.gpu_class(GPU_NAME)
_props = torch.cuda.get_device_properties(0)
if MEM_CAP_GIB > 0:
    torch.cuda.set_per_process_memory_fraction(MEM_CAP_GIB * 2**30 / _props.total_memory, 0)
    print(f"⚠ memory capped at {MEM_CAP_GIB} GiB of {_props.total_memory / 2**30:.2f} GiB (A10 emulation)")
torch.set_float32_matmul_precision("highest")     # Tol_03 never changed it: production ran fp32 at 'highest'
import transformers, timm as _timm_v, scipy as _scipy_v, PIL as _pil_v
if transformers.__version__ != "5.2.0": raise RuntimeError(f"transformers {transformers.__version__} != 5.2.0")
if not cv2.__version__.startswith("4.11.0"): raise RuntimeError(f"cv2 {cv2.__version__}: headless 4.11.0 expected")
if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 5):
    raise RuntimeError(f"torch {torch.__version__} < 2.5 (rfdetr needs torch.accelerator)")

VIDEO_DIR = "/dbfs/mnt/playbehavior/Tol/Videos"
DBFS_VIDEO = "dbfs:/mnt/playbehavior/Tol/Videos"
VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
WSUF = f"_w{FROM_HHMM or '0000'}-{TO_HHMM or '2400'}" if (FROM_HHMM or TO_HHMM) else ""
TRACK_DIR = osp.join(TOL, TRACK_ROOT, STAGE + WSUF)
CROP_DIR  = osp.join(TRACK_DIR, "crops")
# Per-stage local scratch: multiple day-tasks share one node under the consolidated plan, and a
# shared dir would have them overwriting each other's staged clip and obs.npz mid-read.
LOCAL = f"/local_disk0/tmp/tol_v5_{STAGE}{WSUF}"                               # [v5-C1]
for d in ((TRACK_DIR, CROP_DIR, LOCAL) if MODE == "track" else (LOCAL,)):       # [v5-C15] equiv writes no root
    os.makedirs(d, exist_ok=True)
DET_CKPT_DIR = osp.join(TOL, "Detector_Checkpoints", DET_RUN_TAG)                # [v5-C3]

# ---- parameters (single source of truth: DESIGN §7) ------------------------------------------
N_CALVES         = 4
PRUNE_EVERY      = 25
FRESH_SEC        = 2.0     # a mask this old may still associate / seed (matches Tol_15 min_absence)
CLIP_CARRY_MAX   = 10.0    # boundary exception to FRESH_SEC; the pass at +15 s validates it
SEG_BREAK_ABSENCE = 60.0   # reappearance after >= this is a NEW segment (= Tol_15 gap_sec)
LOW_IOU_REAPPEAR = 0.20    # low-IoU reappearance after >= MIN_REAPPEAR_ABSENCE is a break too
MIN_REAPPEAR_ABSENCE = 2.0
OCCLUDED_TIMEOUT = 600.0
ASSOC_MIN        = 0.30    # max(IoU, IoS) to keep a track-detection match
STRIKE_MAX       = 0.10    # below this vs ALL detections = a strike
STRIKES_TO_CLOSE_DAY, STRIKES_TO_CLOSE_NIGHT = 2, 3
REFRESH_IOU      = 0.60    # refresh a healthy track from a detection only above this
# Mask-area collapse guard (added after Tol_25: the student held animal 0 all night with a
# SLIVER mask — IoU 0.36 vs teacher — which passed every health check because a sliver inside a
# detection satisfies the IoS>=0.3 association. A sliver is not a genuine observation: treat a
# sustained collapse vs the slot's own rolling median as ABSENCE (INV-3), so the slot goes
# OCCLUDED and the next detector pass re-seeds it with a full detection mask.)
AREA_COLLAPSE_FRAC = 0.35    # mask < 35 % of the slot's rolling median area
AREA_COLLAPSE_N    = max(2, round(TRACK_FPS))   # sustained ~1 s
AREA_HIST_LEN, AREA_HIST_MIN = 100, 20
# ---- Partial-band repair (2026-09-02, from the 04-23 night diagnosis) -------------------------
# At the dusk/IR transition the student's mask can step down to a STABLE 0.23-0.41x of the
# animal's daytime area and sit there for HOURS (04-23: calf plateaus of 121 and 131 min). The
# collapse guard is blind to it — its ROLLING median adapts down within minutes (by 23:00 the
# median itself sat at 0.31x day reference) — and the only repair mechanism (detection refresh)
# was gated day-only, so the night detector matched the fragments every 300 s (IoS is size-blind)
# and did nothing. Fix: learn each slot's DAY reference area (slow median over daylight frames),
# flag a sustained sub-0.6x-of-day-ref mask as PARTIAL, and let the next detector pass replace it
# with the current-frame detection — through hard spatial gates only (fragment CONTAINED in the
# detection, detection day-plausible in area, no conflict with any other live mask). All gates
# are DAY-anchored, never rolling-median-anchored: a rolling-anchored gate would reject the
# correct full-size repair (3.2x the collapsed median) and accept fragment detections — the
# documented backfire pattern of weak-signal correction layers. Existence proof the repair works:
# the 04-23 23:14:04 detector reopen restored the mask to 1.06x day size instantly.
PARTIAL_FRAC       = 0.60   # below this x day-ref = partial (healthy night calves: 0.94-1.04x;
                            # observed plateaus: 0.23-0.41x — wide separation on both sides)
PARTIAL_N          = max(2, round(30 * TRACK_FPS))   # sustained 30 s before repair arms
REPAIR_CONTAIN_IOS = 0.60   # detection must CONTAIN the fragment (same-animal spatial proof)
REPAIR_AREA_LO, REPAIR_AREA_HI = 0.50, 1.25  # det area vs day-ref: rejects fragment dets (~8k)
                                             # AND two-calf merged dets -- review: at 1.6 the worst
                                             # full+full pairing (90k+63k at the 0.94x night factor
                                             # = 143.8k vs 144k) slipped through; 1.25 blocks the
                                             # smallest merge (2x63k x0.94 = 118k) by >5 %
REPAIR_GROW_MIN    = 1.40   # det must materially GROW the mask (genuine repairs are >=2.5x;
                            # an equal-size det on a day feeder-occlusion buys nothing but a rebuild)
REPAIR_CONFLICT_IOS = 0.10  # det must not overlap any OTHER slot's live mask (0.20 admitted a
                            # full-body + ~12k neighbour-sliver merged detection)
REPAIR_MIN_SEC     = 120.0  # per-slot repair rate limit
REPAIR_MAX_PER_CLIP = 6     # per slot; beyond it the slot is suspended for the clip (oscillation
                            # tripwire -- a correction loop must not run open-loop all night)
DAYREF_DISTRUST    = 0.80   # a slot's OWN day-ref below this x the other slots' median is treated
                            # as fragment-contaminated and the cross-slot anchor is used instead
# ---- Revision 2 (2026-09-06) — after the user found a SILENT dusk identity switch in the L2
# render of the first patched run. The repair is the right medicine for the NIGHT plateau
# (settled cows, fragments drift ~20 px/h) and the wrong medicine during DUSK activity: fragments
# hop between walking animals within seconds (measured: 17k fragment moved ~1000 px in 15 s),
# and completing a hopping fragment legitimises whatever animal it landed on while making it look
# healthy enough to escape the guard/strike/merge breaks that quarantine dusk chaos in the
# unpatched tracker (its slot 1 broke 3x in 11 min; the patched slot 1 ran 18:02->23:59 unbroken).
# So: (1) repair only in a SETTLED scene — night, all four animals claimed, no chaos event and no
# UNCLAIMED whole-animal detection pen-wide for REPAIR_QUIET_SEC, the fragment stationary for a
# full pass interval; (2) a repair is a re-seed from a detection and therefore a SEGMENT
# BOUNDARY. NB (review 2026-09-06): the boundary alone is NOT protection — the offline layer's
# identity exclusivity + pigeonhole hand a post-repair segment the incumbent whenever the true
# identity is held by the wandering slot, so a bijection break is uncorrectable offline. The
# online gates are the real defence; the offline layer additionally FLAGS a repair segment it
# resolved without appearance evidence while an unclaimed animal was seen (Tol_15_v2).
REPAIR_NIGHT_ONLY  = True    # tod_of_wall == "night" (>= 21:00): the plateau regime
REPAIR_STABLE_SEC  = 300.0   # the fragment must have stayed put for a FULL night pass interval
REPAIR_STABLE_PX   = 150.0   #   (centroid excursion < REPAIR_STABLE_PX): a hopper must sit through
                             #   the pass at which its abandoned animal shows up unclaimed (F2)
REPAIR_QUIET_SEC   = 300.0   # no guard/merge/segment event anywhere in the pen for this long
CENT_HIST_LEN      = int(REPAIR_STABLE_SEC * TRACK_FPS) + 5
# Growth guard — the mirror image of the partial band (the student sat at 1.6x day-ref for
# minutes after dusk; the teacher reaches 2.5-3.5x and bleeds onto the ground / a neighbour).
# Same machinery and gates; the detection must be CONTAINED in the mask and materially smaller.
GROW_FRAC          = 1.60
GROW_N             = PARTIAL_N
REPAIR_SHRINK_MAX  = 0.70    # det area <= 0.70 x the over-extended mask
GUARD_FLOOR_FRAC   = 0.45   # collapse guard baseline = max(rolling median, 0.45 x day-ref): a
                            # step-then-stable shrink can no longer drag the baseline down with it
DAYREF_EVERY_SEC   = 5.0    # day-reference learning cadence (slow-moving, spans hours)
DAYREF_STEP        = max(1, round(DAYREF_EVERY_SEC * TRACK_FPS))
DAYREF_LEN         = 2880   # ~4 h of samples at 5 s
DAYREF_MIN         = 60     # >= 5 min of daylight evidence before a slot's day-ref is trusted
MERGE_IOU, MERGE_IOS = 0.50, 0.80
MERGE_N          = max(2, round(TRACK_FPS))     # ~1 s of sustained overlap
JUMPER_LOOKBACK  = 3.0
JUMPER_LOW, JUMPER_HIGH = 0.30, 0.50
MIN_MASK_PX      = 8000
DET_SCORE_MIN, DET_TEXT = 0.40, "calf"
DEDUPE_IOS       = 0.50
SEED_RETRY_SEC   = 30.0
NIGHT_STABLE_IOU = 0.50    # night detections must repeat across passes before opening a segment
FILL = (124, 116, 104)
ST_TRACK, ST_OCC, ST_CLOSED = "TRACKING", "OCCLUDED", "CLOSED"

print(f"STAGE={STAGE} root={TRACK_ROOT} fps={TRACK_FPS} smoke={SMOKE}")
print(f"detect {DETECT_DAY_SEC:.0f}s day / {DETECT_NIGHT_SEC:.0f}s night | refresh "
      f"{'OFF' if REFRESH_MIN_SEC<=0 else f'{REFRESH_MIN_SEC:.0f}s day-only, IoU>={REFRESH_IOU}'} "
      f"| embed {EMBED_EVERY_SEC:.0f}s | crops {CROP_EVERY_SEC:.0f}s | "
      f"partial-band repair {'ON' if PARTIAL_REPAIR else 'OFF (legacy behavior)'}")
# [v5-C10] strikes scale with the cadence to hold v4's time from first strike to close (core.strikes_to_close):
print(f"v5 strikes to close: day {CFG['strikes_day']} @ {DETECT_DAY_SEC:g}s, night {CFG['strikes_night']} @ "
      f"{DETECT_NIGHT_SEC:g}s (v4: 2 @ 120 s / 3 @ 300 s); pending window 2.5 x interval (unchanged)")

# COMMAND ----------

RX = re.compile(r'^(Tol_ch\d+)[-_](\d{8})[-_](\d{6})[-_](\d{6})(?: \((\d+)\))?\.mp4$')
def parse(n):
    m = RX.match(n)
    if not m: return None
    ch, d, s, e, dup = m.groups()
    ds = datetime.strptime(d+s, '%Y%m%d%H%M%S'); de = datetime.strptime(d+e, '%Y%m%d%H%M%S')
    if de < ds: de += timedelta(days=1)
    return {"ch": ch, "date": d, "start": ds, "end": de, "dup": dup is not None}

CLIPS = []
for n in sorted(os.listdir(VIDEO_DIR)):
    if not n.endswith(".mp4"): continue
    p = parse(n)
    if p and p["ch"] == "Tol_ch01" and not p["dup"] and p["date"] == STAGE:
        CLIPS.append((n, p))
CLIPS.sort(key=lambda z: z[1]["start"])
N_ALL = len(CLIPS)
if FROM_HHMM:
    cut = int(FROM_HHMM[:2])*60 + int(FROM_HHMM[2:])
    CLIPS = [(n, p) for n, p in CLIPS if p["start"].hour*60 + p["start"].minute >= cut]
    print(f"⚠ WINDOWED: {len(CLIPS)}/{N_ALL} clips from {FROM_HHMM}")
if TO_HHMM:
    cut2 = int(TO_HHMM[:2])*60 + int(TO_HHMM[2:])
    before = len(CLIPS)
    CLIPS = [(n, p) for n, p in CLIPS if p["start"].hour*60 + p["start"].minute < cut2]
    print(f"⚠ WINDOWED: {len(CLIPS)}/{before} clips before {TO_HHMM}")
if MAX_CLIPS: CLIPS = CLIPS[:MAX_CLIPS]
assert CLIPS, f"no clips for {STAGE}"
vid_h = sum((p["end"]-p["start"]).total_seconds() for _, p in CLIPS)/3600
print(f"{len(CLIPS)} clips  {CLIPS[0][1]['start']:%H:%M} -> {CLIPS[-1][1]['end']:%H:%M}  "
      f"({vid_h:.1f} h video ≈ {vid_h*3600*TRACK_FPS/1000:.0f}k frames @ {TRACK_FPS} fps)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mask utilities (unchanged from v2/v3 — measured and correct)

# COMMAND ----------

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

# [v5-C7] exact fast twins (tol_v5_core; S1/S2 locally, E4 on the GPU): same integer counts and the same true
# division, so every result is bit-identical. The verbatim definitions above stay as the E4 references.
_rle_encode_v4, _m_iou_v4, _m_ios_v4 = rle_encode, m_iou, m_ios
if EXACT_FAST:
    rle_encode, m_iou, m_ios = core.rle_fast, core.m_iou_fast, core.m_ios_fast

# COMMAND ----------

# MAGIC %md
# MAGIC ## Models — student-in-SAM3 tracker, Sam3Model detector, DINOv3 embedder

# COMMAND ----------

def replace_bn_with_gn(module):
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
    def __init__(self, out_channels=1024):
        super().__init__()
        self.backbone = timm.create_model('tiny_vit_21m_512', pretrained=False,
                                          features_only=True, out_indices=(1, 2, 3),
                                          drop_path_rate=0.0)
        replace_bn_with_gn(self.backbone)
        def lat(c): return nn.Sequential(nn.Conv2d(c, 256, 1, bias=False),
                                         nn.GroupNorm(32, 256), nn.GELU())
        self.lateral_s1, self.lateral_s2, self.lateral_s3 = lat(192), lat(384), lat(576)
        self.channel_proj = nn.Sequential(nn.Conv2d(768, out_channels, 1, bias=False),
                                          nn.GroupNorm(32, out_channels), nn.GELU())
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels))
        self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.output_bias  = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
    def forward(self, x):
        s = self.backbone(x)
        f1 = F.interpolate(self.lateral_s1(s[0]), size=(72, 72), mode='area')
        f2 = F.interpolate(self.lateral_s2(s[1]), size=(72, 72), mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.lateral_s3(s[2]), size=(72, 72), mode='bilinear', align_corners=False)
        o = self.channel_proj(torch.cat([f1, f2, f3], 1))
        o = o + self.spatial_refine(o)
        return o * self.output_scale + self.output_bias

# ---- [v5-C3] the training-day calf band (read + provenance-asserted, never a default) and RF-DETR-Seg,
# loaded BEFORE the student and SAM3. In mode=equiv the RF-DETR load is deferred into E3 (student features first). ----
TS = json.load(open(osp.join(DET_CKPT_DIR, "train_summary.json")))
BAND = core.band_from_train_summary(TS, DET_RUN_TAG)
OP_THRESHOLD = BAND["op_threshold"]
print(f"band (train days {sorted(core.TRAIN_DAYS)}): A_REF_TRAIN {BAND['A_REF_TRAIN']:.0f}, dayl "
      f"[{core.band('dayl', BAND)[0]:,.0f}, {core.band('dayl', BAND)[1]:,.0f}], dark [{core.band('dark', BAND)[0]:,.0f}, "
      f"{core.band('dark', BAND)[1]:,.0f}] px; op_threshold {OP_THRESHOLD} (READ, never tuned)")
RF, RF_NC, RF_PEAK_GIB = None, None, None
DET_SHA = f"hf:{SAM3_ID}" if DETECTOR == "sam3" else None

def _stage_small(src, dst):
    """Volume file -> local disk (notebook: dbutils.fs.cp; subprocess: plain copy over the Volume FUSE)."""
    if osp.exists(dst): os.remove(dst)
    if IS_NOTEBOOK:
        dbutils.fs.cp("dbfs:" + src, "file:" + dst)
    else:
        shutil.copy(src, dst)
    return dst

def _rf_raw(arr):
    """RF.predict(threshold=op) keeps score > op (strict); drop class_id >= num_classes (Tol_44:1096-1108). TF32
    only inside predict; 'highest' restored on every path and asserted (core.run_with_precision, S9)."""
    def _pred():
        with torch.inference_mode():
            return RF.predict(arr, threshold=OP_THRESHOLD, include_source_image=False)
    d = core.run_with_precision(torch.set_float32_matmul_precision, torch.get_float32_matmul_precision, _pred)
    rows = []
    if d.mask is not None and len(d):
        k = d.class_id < RF_NC
        rows = [(np.asarray(m, dtype=bool), float(s)) for m, s in zip(d.mask[k], d.confidence[k])]
    for m, s in rows:
        if m.shape != (1296, 2304) or not np.isfinite(s): raise RuntimeError("RF-DETR output contract")
    return rows

def load_rfdetr():
    """Load RF-DETR-Seg. The ONLY place rfdetr is imported (detector=rfdetr or mode=equiv)."""
    global RF, RF_NC, DET_SHA, RF_PEAK_GIB
    os.environ["RF_HOME"] = "/local_disk0/rf_home"
    from rfdetr import from_checkpoint          # side effect: set_float32_matmul_precision('high') (detr.py:52-55)
    torch.set_float32_matmul_precision("highest")                                # immediately restored
    loc = _stage_small(osp.join(DET_CKPT_DIR, "checkpoint_best_total.pth"), osp.join(LOCAL, "rf_best.pth"))
    if osp.getsize(loc) != core.RF_CKPT_BYTES: raise RuntimeError(f"RF checkpoint is {osp.getsize(loc)} B, not {core.RF_CKPT_BYTES}")
    sha = core.sha256_file(loc)
    if sha != TS["checkpoint_best_total_sha256"] or not sha.startswith(core.RF_CKPT_SHA_PREFIX):
        raise RuntimeError(f"RF checkpoint sha {sha[:16]} is not the train_summary checkpoint")
    RF = from_checkpoint(loc, resolution=core.RF_RESOLUTION)   # pinned: strip_checkpoint drops model_config
    RF_NC = int(RF.model_config.num_classes)
    if RF_NC != 1: raise RuntimeError(f"num_classes {RF_NC} != 1")
    if list(RF.class_names) != ["calf"]: raise RuntimeError(f"class_names {RF.class_names} != ['calf']")
    if int(RF.model_config.resolution) != core.RF_RESOLUTION: raise RuntimeError("resolution != 624")
    if int(RF.model_config.positional_encoding_size) != core.RF_RESOLUTION // int(RF.model_config.patch_size):
        raise RuntimeError("positional_encoding_size != resolution // patch_size")
    kb, _nf = core.post_chain_idx(_rf_raw(np.zeros((1296, 2304, 3), np.uint8)), topk=0)
    if kb: raise RuntimeError(f"RF-DETR keeps {len(kb)} masks on a black frame")
    if torch.get_float32_matmul_precision() != "highest": raise RuntimeError("precision not 'highest' after the RF load")
    DET_SHA = sha
    RF_PEAK_GIB = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    print(f"RF-DETR-Seg ready: sha256 {sha[:16]}, resolution {core.RF_RESOLUTION}, op {OP_THRESHOLD}, "
          f"max_memory_allocated {RF_PEAK_GIB} GiB")

if DETECTOR == "rfdetr" and MODE == "track":
    load_rfdetr()
if MODE == "equiv" and DETECTOR != "rfdetr":
    raise RuntimeError("mode=equiv tests the v5 detector: set detector=rfdetr")

if BACKBONE == "student":
    STUDENT_SHA = core.sha256_file(osp.join(TOL, "Checkpoints", STUDENT_CKPT))   # [v5-C4] logged + resume-guarded
    ck = torch.load(osp.join(TOL, "Checkpoints", STUDENT_CKPT), map_location='cpu')
    student = StudentEncoderTinyViT()
    sd = student.state_dict(); sd.update(ck['model_state_dict'])
    for k, v in ck['ema_shadow'].items(): sd[k] = v
    student.load_state_dict(sd); student = student.to(device).eval()
    # [v5-C4] INV-9b: a silent load is not a validated load -- zero surviving BatchNorm, every tensor from the ckpt
    _nbn = sum(1 for _m in student.modules() if isinstance(_m, (nn.BatchNorm1d, nn.BatchNorm2d)))
    if _nbn: raise RuntimeError(f"{_nbn} BatchNorm layers survive: architecture does not match the checkpoint")
    _have = set(ck['model_state_dict']) | set(ck['ema_shadow'])
    _uninit = [k for k in student.state_dict() if k not in _have]
    if _uninit: raise RuntimeError(f"{len(_uninit)} student tensors absent from the checkpoint: {_uninit[:8]}")
    _t = timm.create_model('tiny_vit_21m_512', pretrained=False)
    _cfg = resolve_data_config(_t.pretrained_cfg); del _t
    STUDENT_TF = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                     transforms.Normalize(mean=_cfg['mean'], std=_cfg['std'])])
    if GPU_TF:      # [v5-C9] GPU transform: CHANGES FEATURES, needs allow_numeric_change (config raises otherwise; D7)
        _GMEAN = torch.tensor(_cfg['mean'], device=device).view(1, 3, 1, 1)
        _GSTD = torch.tensor(_cfg['std'], device=device).view(1, 3, 1, 1)
        def STUDENT_INPUT(pil):
            t = torch.from_numpy(np.asarray(pil)).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            t = F.interpolate(t, size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True)
            return (t - _GMEAN) / _GSTD
    else:           # [v5-C9] default: exactly Tol_03:627's input
        def STUDENT_INPUT(pil):
            return STUDENT_TF(pil).unsqueeze(0).to(device)
    print(f"Tol student loaded: {STUDENT_CKPT} (epoch {ck['epoch']+1}, cosine {ck.get('cosine', float('nan')):.4f}) "
          f"sha256 {STUDENT_SHA[:16]}; 0 BatchNorm, all {len(student.state_dict())} tensors from the checkpoint")
else:
    student, STUDENT_TF, STUDENT_INPUT, STUDENT_SHA = None, None, None, "teacher (own backbone)"   # [v5-C4]
    print(f"BACKBONE=teacher — {SAM3_ID}'s own vision backbone will run (ground-truth mode)")

from transformers import (Sam3TrackerVideoModel, Sam3TrackerVideoProcessor,
                          Sam3Model, Sam3Processor, AutoImageProcessor, AutoModel)
from transformers.modeling_outputs import BaseModelOutput

tproc = Sam3TrackerVideoProcessor.from_pretrained(SAM3_ID)
sam3  = Sam3TrackerVideoModel.from_pretrained(SAM3_ID, torch_dtype=DTYPE).to(device).eval()

class DummyBackbone(nn.Module):
    def __init__(self): super().__init__(); self.student_features = None
    def forward(self, *a, **k):
        f = self.student_features; B, C, H, W = f.shape
        return BaseModelOutput(last_hidden_state=f.permute(0, 2, 3, 1).reshape(B, H*W, C))

dummy = DummyBackbone().to(device)
if BACKBONE == "student":
    _teacher_bb = sam3.vision_encoder.backbone; sam3.vision_encoder.backbone = dummy   # [v5-C4] teacher backbone
    del _teacher_bb                                                                   # DELETED (Tol_03: .cpu())
    gc.collect(); torch.cuda.empty_cache()

if DETECTOR == "sam3":                                    # [v5-C3] the Sam3Model detector: regression arm only
    dproc = Sam3Processor.from_pretrained(SAM3_ID)
    det = Sam3Model.from_pretrained(SAM3_ID, torch_dtype=DTYPE).to(device).eval()
    DET_SHA = f"hf:{SAM3_ID}@{getattr(det.config, '_commit_hash', None)}"
# bf16 for DINOv3: fp16 already produced silent NaN in this project (ViT-L -> plausible 25 %).
# ViT-S never NaN'd in fp16, but Ampere bf16 costs nothing and the assert makes it moot.
dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
dino = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m",
                                 torch_dtype=DTYPE).to(device).eval()
print("tracker + detector + DINOv3 ready")

def _augment(crop):
    outs = [crop, crop.rotate(180, resample=Image.BILINEAR, fillcolor=tuple(FILL))]
    outs.append(crop.rotate(12, resample=Image.BILINEAR, fillcolor=tuple(FILL)))
    outs.append(crop.rotate(-12, resample=Image.BILINEAR, fillcolor=tuple(FILL)))
    outs.append(crop.transpose(Image.FLIP_LEFT_RIGHT))
    for f in (0.85, 1.18):
        a = np.clip(np.array(crop).astype(np.float32)*f, 0, 255).astype(np.uint8)
        outs.append(Image.fromarray(a))
    return outs

@torch.no_grad()
def embed_robust(pil, mask):
    """Augmentation-averaged embedding of the canonical crop (None if unusable / non-finite)."""
    c = canonical_crop(pil, mask)
    if c is None: return None
    variants = _augment(c)
    px = dino_proc(images=variants, return_tensors="pt")["pixel_values"].to(device, DTYPE)
    e = F.normalize(dino(pixel_values=px).pooler_output.float(), dim=-1)
    out = F.normalize(e.mean(0), dim=-1).cpu()
    if not torch.isfinite(out).all(): return None
    return out

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

# ---- [v5-C3/C14] v5 detect(): RF-DETR-Seg or the VERBATIM Tol_03 Sam3Model arm above (detector=sam3).
# Same return type (deduped (mask, score) list, largest first, top-k). DET_CALLED (reset every frame) lets v5 paths
# add at most one detect() per frame; every call is logged for detections_<clip>.json.gz. ----------------------------
_detect_sam3_v4 = detect
DET_CALLED = False; DET_N_FRAME = 0
DET_LOG = []                                    # one record per detect() call of the current clip
CUR = {"clip": None, "fidx": None, "t_sec": None, "wall": None}

def _area_px(m):
    return core.area_fast(m) if EXACT_FAST else int(m.sum())

def detect(pil, topk=None, kind="pass"):
    global DET_CALLED, DET_N_FRAME
    DET_CALLED = True; DET_N_FRAME += 1
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if DETECTOR == "rfdetr":
        rows = _rf_raw(np.asarray(pil))                                   # RGB uint8 (BGR->RGB at the decode)
        kidx, _nf = core.post_chain_idx(rows, topk=(topk or 0), iou=m_iou, ios=m_ios, area=_area_px)   # = Tol_44:702-711
        kept = [rows[i] for i in kidx]; n_raw = len(rows)
    else:
        kept = _detect_sam3_v4(pil, topk=topk)                            # Tol_03:453-472 verbatim
        rows, kidx, n_raw = kept, list(range(len(kept))), None            # raw = kept (post-chain is inside it)
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
    w = CUR["wall"]; pool = core.pool_of_wall(w) if w is not None else None
    DET_LOG.append({"clip": CUR["clip"], "frame_idx": CUR["fidx"],
                    "t_sec": None if CUR["t_sec"] is None else round(CUR["t_sec"], 2), "kind": kind,
                    "n_raw": n_raw, "det_ms": round(ms, 1),
                    "n_inband": None if pool is None else sum(1 for m, _s in kept if core.in_band(_area_px(m), pool, BAND)),
                    "raw": [{"rle": rle_encode(m), "score": round(float(s), 4), "area": int(_area_px(m))} for m, s in rows],
                    "kept_idx": list(kidx)})
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("float32 matmul precision is not 'highest' after detect()")
    return kept

def prune(session, k=8):
    for oi in range(session.get_obj_num()):
        nc = session.output_dict_per_obj[oi].get("non_cond_frame_outputs", {})
        if len(nc) > k:
            for key in sorted(nc.keys())[:-k]: del nc[key]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segment state machine + events

# COMMAND ----------

EV_VOL = osp.join(TRACK_DIR, "events.jsonl"); EV_LOC = osp.join(LOCAL, f"ev_{STAGE}{WSUF}.jsonl")
if osp.exists(EV_VOL):
    if not osp.exists(EV_LOC): shutil.copy(EV_VOL, EV_LOC)
elif osp.exists(EV_LOC):
    os.remove(EV_LOC)     # /local_disk0 survives on a warm cluster; drop stale logs
def log_event(**kw):
    with open(EV_LOC, "a") as f: f.write(json.dumps(kw, default=str) + "\n")
def sync_events():
    if osp.exists(EV_LOC): shutil.copy(EV_LOC, EV_VOL)

state         = [ST_CLOSED]*N_CALVES
seg           = [None]*N_CALVES        # open segment record per slot
seg_counter   = [0]*N_CALVES
last_gen_wall = [None]*N_CALVES        # wall datetime of last GENUINE SAM3 mask
last_gen_mask = [None]*N_CALVES        # the mask itself (numpy bool)
strikes       = [0]*N_CALVES
occ_logged    = [False]*N_CALVES
segments_done = []
hist = [deque(maxlen=32) for _ in range(N_CALVES)]   # (wall, mask[::4,::4]) for jumper verdicts
area_hist = [deque(maxlen=AREA_HIST_LEN) for _ in range(N_CALVES)]
collapse_run = [0]*N_CALVES
merge_run = defaultdict(int); merge_onset = {}
pending_open = []                      # night stability: [{"mask","wall","seen"}]
last_refresh_wall = None
session = None; session_slots = []
day_area_hist = [deque(maxlen=DAYREF_LEN) for _ in range(N_CALVES)]  # daylight areas, ~5 s cadence
partial_run   = [0]*N_CALVES           # consecutive frames below PARTIAL_FRAC x day-ref
partial_armed = [False]*N_CALVES       # one-shot latch: crossing PARTIAL_N earns ONE early pass
last_repair_wall = [None]*N_CALVES
repairs_clip  = [0]*N_CALVES           # per-slot repairs in the current clip (suspension tripwire)
LAST_REPAIRS  = {}                     # slot -> det mask repaired on the current frame (row record)
DAYREF_VER    = [0]                    # bumps on every day_area_hist append (median cache key)
_dayref_cache = {"ver": -1, "meds": {}}
cent_hist     = [deque(maxlen=CENT_HIST_LEN) for _ in range(N_CALVES)]  # (wall, cx, cy) per frame
grown_run     = [0]*N_CALVES           # consecutive frames above GROW_FRAC x day-ref
grown_armed   = [False]*N_CALVES
last_scene_event = [None]              # wall of the last guard/merge/segment event (any slot)
# ---- [v5-C11] per-slot dark and healthy state. Written only when dark_fire=on, so regression mode reaches
# no new branch; a resumed day starts it EMPTY (logged as type=resume). healthy_mask is a GATE REFERENCE only:
# it is never passed to open_session_v4 (INV-1).
healthy_mask  = [None]*N_CALVES; healthy_wall = [None]*N_CALVES; healthy_seg = [None]*N_CALVES
dark_onset    = [None]*N_CALVES; dark_cause = [None]*N_CALVES; dark_seg = [None]*N_CALVES
dark_covered  = [False]*N_CALVES; dark_merge_near = [False]*N_CALVES
dark_close_wall = [None]*N_CALVES; dark_onset_pending = [False]*N_CALVES
last_merge_touch = [None]*N_CALVES
dark_actions_clip = [0]*N_CALVES; last_dark_action_wall = [None]*N_CALVES; dark_suspended_logged = [False]*N_CALVES
next_dark = 0.0                        # clip-local t_sec of the next allowed dark-only pass
ROW_OVERRIDE = {}                      # [v5-C12] generalises LAST_REPAIRS: slot -> det mask this frame's row carries

def note_scene_event(wall):
    last_scene_event[0] = wall

def scene_quiet(wall):
    return last_scene_event[0] is None or \
           (wall - last_scene_event[0]).total_seconds() >= REPAIR_QUIET_SEC

def slot_stable(s, wall):
    """Has this slot's mask stayed put? (max centroid excursion over the last REPAIR_STABLE_SEC,
    and the window must actually be that long — a fresh segment is not 'stable', it is unknown)"""
    pts = [p for p in cent_hist[s] if (wall - p[0]).total_seconds() <= REPAIR_STABLE_SEC]
    if not pts or (wall - pts[0][0]).total_seconds() < 0.8 * REPAIR_STABLE_SEC:
        return False, None
    cx, cy = pts[-1][1], pts[-1][2]
    dmax = max(((p[1] - cx) ** 2 + (p[2] - cy) ** 2) ** 0.5 for p in pts)
    return dmax < REPAIR_STABLE_PX, dmax

def _slot_day_medians():
    if _dayref_cache["ver"] != DAYREF_VER[0]:
        _dayref_cache["meds"] = {s: float(np.median(day_area_hist[s])) for s in range(N_CALVES)
                                 if len(day_area_hist[s]) >= DAYREF_MIN}
        _dayref_cache["ver"] = DAYREF_VER[0]
    return _dayref_cache["meds"]

def day_ref_of(s):
    """Slot's daytime reference area (median of its slow daylight history), with two defences:
    a slot whose own median sits far below the OTHER slots' median-of-medians is treated as
    fragment-contaminated and borrows the cross-slot anchor (all four animals sit in 63-90k px),
    and a slot with no daylight history borrows it too. None (nothing learned yet) disables the
    repair/floor for that slot -- legacy behavior -- rather than guessing."""
    meds = _slot_day_medians()
    if not meds: return None
    own = meds.get(s)
    others = [v for k, v in meds.items() if k != s]
    if own is not None:
        if not others: return own
        med_o = float(np.median(others))
        if DAYREF_DISTRUST * med_o <= own <= med_o / DAYREF_DISTRUST: return own   # symmetric
    return float(np.median(others)) if others else own

SEG_PATH = osp.join(TRACK_DIR, "segments.json")

def write_segments():
    recs = list(segments_done)
    for s in range(N_CALVES):
        if seg[s] is not None:
            r = dict(seg[s]); r["open"] = True
            recs.append(r)
    with open(SEG_PATH, "w") as f:
        json.dump({"day": STAGE, "track_fps": TRACK_FPS, "segments": recs}, f, indent=1)

def open_segment(slot, wall, reason, extra=None):         # [v5-C14] extra: dark_link evidence / via=dark_pass
    seg_counter[slot] += 1
    sid = f"{slot}_{seg_counter[slot]:03d}"
    seg[slot] = {"seg_id": sid, "slot": slot, "t_start_wall": wall.isoformat(),
                 "open_reason": reason, "n_frames": 0}
    if extra: seg[slot].update(extra)                                   # [v5-C14]
    if DARK_FIRE:                                                       # [v5-C11] a new segment is not dark; its
        dark_onset[slot] = None; dark_cause[slot] = None; dark_seg[slot] = None   # healthy mask restarts (the seed
        dark_onset_pending[slot] = False; dark_covered[slot] = False     # is noted by the caller if in band)
        dark_merge_near[slot] = False; dark_close_wall[slot] = None
        healthy_mask[slot] = None; healthy_wall[slot] = None; healthy_seg[slot] = sid
    state[slot] = ST_TRACK
    strikes[slot] = 0; occ_logged[slot] = False
    area_hist[slot].clear(); collapse_run[slot] = 0
    partial_run[slot] = 0; partial_armed[slot] = False
    grown_run[slot] = 0; grown_armed[slot] = False; cent_hist[slot].clear()
    if not reason.endswith("_repair"): note_scene_event(wall)   # a repair boundary is not chaos
    log_event(type="segment_open", seg_id=sid, slot=slot, wall=wall.isoformat(), reason=reason,
              **(extra or {}))                                          # [v5-C14]
    return sid

def close_segment(slot, wall, reason, extra=None, drop_mask=True):
    """End the slot's segment. drop_mask=True kills the last mask too (INV-1: a closed slot's
    mask must never associate or seed again). Self-recovery breaks pass drop_mask=False because
    the slot continues immediately with a fresh genuine mask."""
    if seg[slot] is not None:
        r = seg[slot]; r["t_end_wall"] = wall.isoformat(); r["close_reason"] = reason
        if extra: r.update(extra)
        segments_done.append(r)
        log_event(type="segment_close", seg_id=r["seg_id"], slot=slot, wall=wall.isoformat(),
                  reason=reason, n_frames=r["n_frames"])
        seg[slot] = None
    if DARK_FIRE:                                  # [v5-C11] a close of a NOT-dark slot starts a dark episode with
        if state[slot] == ST_TRACK:                # the close reason as its cause; a close of an already-dark slot
            dark_onset[slot] = wall; dark_cause[slot] = reason; dark_onset_pending[slot] = True   # keeps them
            dark_seg[slot] = segments_done[-1]["seg_id"] if segments_done and segments_done[-1]["slot"] == slot else None
        dark_close_wall[slot] = wall
    state[slot] = ST_CLOSED
    if drop_mask:
        last_gen_wall[slot] = None; last_gen_mask[slot] = None
        hist[slot].clear()
    strikes[slot] = 0; occ_logged[slot] = False
    partial_run[slot] = 0; partial_armed[slot] = False   # a frozen counter must not outlive the segment
    grown_run[slot] = 0; grown_armed[slot] = False
    if not reason.endswith("_repair"): note_scene_event(wall)

def open_session_v4(pil, slot_mask_pairs):
    """INV-1: every mask in slot_mask_pairs is either a genuine SAM3 output <= FRESH_SEC old
    (<= CLIP_CARRY_MAX at a clip boundary) or a detector mask from THIS frame. No other source
    exists in this file. No box fallback either — mask prompts are load-bearing (mask-first)."""
    inputs = tproc(images=pil, device=device, return_tensors="pt")
    slots = [s for s, _ in slot_mask_pairs]
    sess = tproc.init_video_session(inference_device=device, dtype=DTYPE)
    mask_list = [torch.from_numpy(m).to(device) for _, m in slot_mask_pairs]  # python LIST of 2-D
    tproc.add_inputs_to_inference_session(
        inference_session=sess, frame_idx=0, obj_ids=list(range(len(slots))),
        input_masks=mask_list, original_size=inputs.original_sizes[0])
    sess._tol_fidx = 0                  # [v5-C5] explicit frame index 0, 1, 2, ... (= v4's implicit len(processed_frames))
    V5C["sessions"] += 1                # [v5-C14]
    return sess, slots

def track_frame(sess, slots, pil, pop=None, raw=False):   # [v5-C5] pop: E1 arms override POP_FRAMES; raw: E1 outputs
    inputs = tproc(images=pil, device=device, return_tensors="pt")
    _lap("tproc")                                                        # [v5-C14]
    with torch.no_grad():
        if BACKBONE == "student":
            feat = student(STUDENT_INPUT(pil))                           # [v5-C9] = STUDENT_TF(pil).unsqueeze(0).to(device)
            if DTYPE == torch.float16:
                mx = feat.abs().max().item()
                if not np.isfinite(mx) or mx > 60000:
                    raise RuntimeError(f"student feature magnitude {mx:.1f} unsafe for fp16")
            dummy.student_features = feat
            _lap("student")                                              # [v5-C14]
        # [v5-C5] pop_frames: explicit frame_idx + drop that frame's pixels after its own forward (core.sam3_forward;
        # pop off = Tol_03:633 verbatim). Memory no longer grows 5.8 MiB per tracked frame.
        out = core.sam3_forward(sam3, sess, inputs.pixel_values[0], POP_FRAMES if pop is None else pop)
        if BACKBONE == "student":
            dummy.student_features = None
        if not torch.isfinite(out.pred_masks).all():
            raise RuntimeError("non-finite SAM3 mask logits — dtype unsafe on this GPU")
    _lap("sam3")                                                         # [v5-C14]
    mm = tproc.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes,
                                  binarize=False)[0]
    res = {}
    for k, slot in enumerate(slots):
        ml = mm[k, -1] if mm.dim() == 4 else mm[k]
        b = (torch.sigmoid(ml) > 0.5).cpu().numpy().astype(bool)
        res[slot] = b if b.sum() >= 200 else None
    _lap("post")                                                         # [v5-C14]
    if raw: return res, out                                              # [v5-C15] E1 compares the raw outputs
    return res

def tod_of_wall(w):
    return "night" if (w.hour >= 21 or w.hour < 6) else "day"

def resolve_merge(i, j, wall):
    """Two slots on one animal. The jumper shows a self-IoU discontinuity (consecutive-mask IoU
    is ~0.986 when tracking is healthy); close ONLY it. Ambiguous -> close both (honest)."""
    def min_selfiou(s):
        pts = [(w, mm) for w, mm in hist[s] if (wall - w).total_seconds() <= JUMPER_LOOKBACK]
        if len(pts) < 3: return None
        return min(m_iou(pts[k][1], pts[k+1][1]) for k in range(len(pts)-1))
    a, b = min_selfiou(i), min_selfiou(j)
    onset = merge_onset.get((i, j), wall)
    if DARK_FIRE: last_merge_touch[i] = last_merge_touch[j] = wall      # [v5-C11]
    if a is not None and b is not None and a < JUMPER_LOW and b >= JUMPER_HIGH:
        jumpers = [i]
    elif a is not None and b is not None and b < JUMPER_LOW and a >= JUMPER_HIGH:
        jumpers = [j]
    else:
        jumpers = [i, j]
        # Night fragment-pair mitigation (review 2b): two IR fragments of ONE calf can pass the
        # detection dedupe, open two slots, and both grow smoothly into the same animal — the
        # ambiguous close-both then re-enters the same cycle all night. If exactly one of the
        # pair is freshly opened while the other is established, close only the newcomer.
        ages = {}
        for s in (i, j):
            if seg[s] is not None:
                try: ages[s] = (wall - datetime.fromisoformat(seg[s]["t_start_wall"])).total_seconds()
                except Exception: ages[s] = None
        if ages.get(i) is not None and ages.get(j) is not None:
            young, old = (i, j) if ages[i] < ages[j] else (j, i)
            if ages[young] < 300.0 <= ages[old]:
                jumpers = [young]
    for s in jumpers:
        close_segment(s, wall, "merge_jumper" if len(jumpers) == 1 else "merge_ambiguous",
                      extra={"merge_onset": onset.isoformat(), "merge_pair": [i, j]})
    note_scene_event(wall)
    log_event(type="merge_resolved", pair=[i, j], jumpers=jumpers,
              selfiou=[a, b], onset=onset.isoformat(), wall=wall.isoformat())
    for key in list(merge_run):
        if i in key or j in key: merge_run.pop(key, None); merge_onset.pop(key, None)
    return jumpers

# COMMAND ----------

# MAGIC %md
# MAGIC ## The detector pass — association, strikes, reopen, refresh, rebuild (DESIGN §3.2)

# COMMAND ----------

obs = []                        # DINOv3 observations, keyed slot+seg
CLIP_PASS = None                # telemetry accumulator for the current clip

def preload_obs():
    """On a RESUMED run, reload observations for already-.done_ clips. Segments and events have
    resume paths but obs lived only in memory — a resume would silently overwrite the day's
    observations.npz with post-resume entries only (review 2026-08-17, HIGH). Obs whose clip has
    no .done_ marker are skipped: that clip re-runs and regenerates them."""
    npz_p = osp.join(TRACK_DIR, "observations.npz")
    meta_p = osp.join(TRACK_DIR, "observations_meta.json")
    if not (osp.exists(npz_p) and osp.exists(meta_p)): return
    try:
        E = np.load(npz_p)["emb"]
        meta = json.load(open(meta_p))
        if len(meta.get("obs", [])) != len(E):
            print(f"⚠ preload_obs: meta/emb mismatch {len(meta.get('obs', []))} vs {len(E)} — skipping")
            return
        kept = 0
        for m, e in zip(meta["obs"], E):
            stem = m.get("clip", "")[:-4].replace(" ", "_")
            if not osp.exists(osp.join(TRACK_DIR, f".done_{stem}")): continue
            m2 = dict(m); m2["emb"] = np.asarray(e, np.float16)
            obs.append(m2); kept += 1
        if kept: print(f"resume: preloaded {kept} observations from completed clips")
    except Exception as ex:
        print(f"⚠ preload_obs failed ({ex}) — continuing with empty obs")
preload_obs()

# ---- [v5-C11/C12] dark-step glue: the DECISIONS are tol_v5_core's; these helpers only read and apply tracker state.
# INV-2: nothing here (or in detector_pass) reads an embedding, a crop, the observations or DINOv3. -----------------
def _live():
    return [s for s in range(N_CALVES) if state[s] == ST_TRACK and last_gen_mask[s] is not None]

def _note_healthy(s, m, wall, a=None):
    """The latest genuine mask with collapse_run == 0 and its area in band(pool) (seeds count when in band)."""
    if m is None or collapse_run[s] != 0: return
    if core.in_band(_area_px(m) if a is None else a, core.pool_of_wall(wall), BAND):
        healthy_mask[s] = m; healthy_wall[s] = wall
        healthy_seg[s] = seg[s]["seg_id"] if seg[s] is not None else None

def _slot_views(wall):
    out = []
    for s in range(N_CALVES):
        st = state[s]
        if st == ST_TRACK:
            out.append({"state": st, "live_mask": last_gen_mask[s]}); continue
        if st == ST_OCC:
            age = (wall - last_gen_wall[s]).total_seconds() if last_gen_wall[s] is not None else None
        else:
            age = (wall - dark_close_wall[s]).total_seconds() if dark_close_wall[s] is not None else None
        out.append({"state": st, "live_mask": None, "dark_age_s": age, "healthy_mask": healthy_mask[s],
                    "healthy_age_s": (wall - healthy_wall[s]).total_seconds() if healthy_wall[s] is not None else None,
                    "healthy_seg": healthy_seg[s], "cause": dark_cause[s], "covered": dark_covered[s],
                    "merge_near": dark_merge_near[s], "actions_clip": dark_actions_clip[s],
                    "since_action_s": ((wall - last_dark_action_wall[s]).total_seconds()
                                       if last_dark_action_wall[s] is not None else None)})
    return out

def _any_actionable(wall):
    pool, views = core.pool_of_wall(wall), _slot_views(wall)
    return any((a["link"] or a["open"]) for a in (core.slot_actionable(views[s], pool, DCFG)
                                                   for s in range(N_CALVES) if state[s] != ST_TRACK))

def _dark_slots_view(views, pool):
    out = {}
    for s in range(N_CALVES):
        if views[s]["state"] == ST_TRACK: continue
        v = views[s]; a = core.slot_actionable(v, pool, DCFG)
        out[str(s)] = {"state": v["state"], "dark_age_s": None if v["dark_age_s"] is None else round(v["dark_age_s"], 1),
                       "healthy_age_s": None if v["healthy_age_s"] is None else round(v["healthy_age_s"], 1),
                       "cause": v["cause"], "actionable": "link" if a["link"] else ("open" if a["open"] else a["why"])}
    return out

def _flush_dark_onsets(wall, clip, t_sec):
    """Log dark_onset once per episode (end of frame): covered = max IoS(healthy, another live slot's mask) >=
    COVER_IOS at onset; merge_near = a merge touched the slot within MERGE_GUARD_SEC before onset."""
    for s in range(N_CALVES):
        if not dark_onset_pending[s]: continue
        dark_onset_pending[s] = False
        if state[s] == ST_TRACK: continue
        on = dark_onset[s] or wall
        cov = (max([m_ios(healthy_mask[s], last_gen_mask[l]) for l in _live() if l != s], default=0.0)
               if healthy_mask[s] is not None else 0.0)
        dark_covered[s] = cov >= core.COVER_IOS
        dark_merge_near[s] = (last_merge_touch[s] is not None
                              and (on - last_merge_touch[s]).total_seconds() <= core.MERGE_GUARD_SEC)
        V5C["dark"]["onsets"] += 1
        log_event(**core.ev_dark_onset(s, dark_seg[s], clip, t_sec, on.isoformat(), dark_cause[s], dark_covered[s],
                                       dark_merge_near[s], (on - healthy_wall[s]).total_seconds()
                                       if healthy_wall[s] is not None else None))

def _sched_would_rebuild(wall, is_day, dets, matched_det, pool, dl):
    """Young-OCC deferral: would the scheduled part rebuild even if the dark step did nothing? With a young
    OCC slot present refresh is blocked (an OCC slot < 60 s dark, Tol_03:805-807) and repair needs every slot
    tracking (Tol_03:824), so only a step-4 open rebuilds: dry-run the v4 unmatched loop on copies."""
    un0 = [dets[b][0] for b in range(len(dets)) if b not in matched_det]
    allow = None                                     # review r1: the same per-detection veto as the real loop
    if V5_FIXES:
        allow, _why, _ref = core.reopen_considered(un0, [last_gen_mask[l] for l in _live()], pool, BAND, dl["n_inband"],
                                                   ios=m_ios, area=_area_px)
    age = [((wall - last_gen_wall[s]).total_seconds() if last_gen_wall[s] is not None else 0.0) for s in range(N_CALVES)]
    return bool(core.v4_step4_opens(un0, pending_open, list(state), [last_gen_wall[s] is not None for s in range(N_CALVES)],
                                    age, is_day, wall, 2.5 * (DETECT_DAY_SEC if is_day else DETECT_NIGHT_SEC),
                                    iou=m_iou, ios=m_ios, allow=allow))

def _dark_action(s, wall, clip, t_sec, evs):
    dark_actions_clip[s] += 1; last_dark_action_wall[s] = wall
    if dark_actions_clip[s] >= core.DARK_MAX_PER_CLIP and not dark_suspended_logged[s]:   # per-clip tripwire
        dark_suspended_logged[s] = True; V5C["dark"]["suspended"] += 1
        evs.append(core.ev_dark_suspended(s, clip, dark_actions_clip[s], t_sec))

def _dark_step(wall, clip, t_sec, dets, matched_det, scheduled, pool, is_day, dl, opens, open_extra):
    """Pass step 3: core.dark_step_decide on the detections step 2 did not match, then APPLY it. A link or
    a fast-open is an ORDINARY open -- a NEW segment through the one rebuild (an OCC slot is first closed 'dark_gap' /
    'displaced_by_detection'): no continue-links. Never touches strikes, pending_open, refresh, repair,
    the INV-7 counters or the pass event. Returns (decision or None when nothing was actionable, events)."""
    views = _slot_views(wall)
    acts = {s: core.slot_actionable(views[s], pool, DCFG) for s in range(N_CALVES) if state[s] != ST_TRACK}
    if not any((a["link"] or a["open"]) for a in acts.values()):
        return None, []
    young = any(state[s] == ST_OCC and views[s]["dark_age_s"] is not None and views[s]["dark_age_s"] < DARK_FIRST_SEC
                for s in range(N_CALVES))
    sched_rebuilds = bool(scheduled and young and _sched_would_rebuild(wall, is_day, dets, matched_det, pool, dl))
    res = core.dark_step_decide(dets, set(matched_det), views, pool, BAND, DCFG, scheduled=scheduled,
                                sched_rebuilds=sched_rebuilds, iou=m_iou, ios=m_ios, area=_area_px)
    res["dark_slots"] = _dark_slots_view(views, pool)
    D, evs = V5C["dark"], []
    D["passes"] += 1
    if res["refused"]: D["refused_by_reason"][res["refused"]] = D["refused_by_reason"].get(res["refused"], 0) + 1
    if res["deferred"]: D["deferred"] += 1
    for _s, _rj in res["rejected"].items():
        for _b, _why in _rj: D["rejected_by_reason"][_why] = D["rejected_by_reason"].get(_why, 0) + 1
    if res["refused"] == "pen_count":
        V5C["pen_count_refused"]["dark"] += 1
        evs.append(core.ev_pen_count_refused("dark", res["n_inband"], res["areas"], res["scores"], clip, t_sec))
    if res["open_refused"]:
        evs.append(core.ev_dark_open_refused(clip, t_sec, res["open_refused"]))
    for s in sorted(res["links"]):
        L = res["links"][s]; dm = dets[L["det"]][0]
        if state[s] == ST_OCC:
            close_segment(s, wall, "dark_gap", drop_mask=False); CLIP_PASS["closes"] += 1
        opens.append((s, dm, "dark_link"))
        open_extra[s] = {"link_from": L["prev_seg"], "link_gap_s": L["gap_s"] if L["gap_s"] is None else round(L["gap_s"], 2),
                         "link_ios": L["ios"], "link_iou": L["iou"], "det_area": L["det_area"], "n_dark": L["n_dark"],
                         "sole_dark": L["sole_dark"], "cause": L["cause"]}
        _dark_action(s, wall, clip, t_sec, evs)
        D["links"] += 1; D["links_sole"] += int(L["sole_dark"])
    for s in sorted(res["opens"]):
        O = res["opens"][s]; dm = dets[O["det"]][0]
        if O["reason"] == "displaced":
            close_segment(s, wall, "displaced_by_detection"); CLIP_PASS["closes"] += 1
        opens.append((s, dm, O["reason"]))
        open_extra[s] = {"via": "dark_pass"}
        _dark_action(s, wall, clip, t_sec, evs)
        D["opens"] += 1
    return res, evs

def detector_pass(pil, wall, clip, t_sec, forced_by=None, scheduled=True):    # [v5-C12]
    """One measurement + correction cycle. Spatial only (INV-2/5). Returns True if session rebuilt."""
    # [v5-C12] scheduled=False = a DARK-ONLY pass: no association / strikes / pending / refresh / repair,
    # no pass / pass_no_detections event, CLIP_PASS untouched. v5 returns True when the dark step ran or a dark-only
    # call spent its detect() (re-arms next_dark); Tol_03 returned 'rebuilt', which its caller ignored.
    global session, session_slots, pending_open, last_refresh_wall
    is_day = tod_of_wall(wall) == "day"
    pool = core.pool_of_wall(wall)                                        # [v5-C12] phase4 pools (dusk = dark)
    dets = detect(pil, topk=N_CALVES + 2, kind="pass" if scheduled else "dark")   # [v5-C3/C14]
    dl = DET_LOG[-1]                                                      # [v5-C14] n_raw / det_ms / n_inband
    if scheduled: CLIP_PASS["passes"] += 1                                # [v5-C12] scheduled calls only
    if not dets:
        if not scheduled:                                                 # [v5-C12] dark_pass only
            _v0 = _slot_views(wall); _r0 = core.dark_step_decide([], set(), _v0, pool, BAND, DCFG)
            V5C["dark"]["passes"] += 1
            log_event(**core.ev_dark_pass(clip, t_sec, wall.isoformat(), _r0, "dark_only", dl["n_raw"], dl["det_ms"],
                                          _dark_slots_view(_v0, pool), False))
            return True
        log_event(type="pass_no_detections", clip=clip, t=round(t_sec, 2), forced=forced_by)
        return False
    trk = [s for s in range(N_CALVES) if state[s] == ST_TRACK and last_gen_mask[s] is not None] \
        if scheduled else []                                              # [v5-C12] dark-only: no step 2
    S = np.zeros((len(trk), len(dets)), np.float32)
    for a, s in enumerate(trk):
        for b, (dm, _) in enumerate(dets):
            S[a, b] = max(m_iou(last_gen_mask[s], dm), m_ios(last_gen_mask[s], dm))
    matched, matched_det = {}, set()
    if trk:
        ri, cj = linear_sum_assignment(1.0 - S)
        for a, b in zip(ri, cj):
            if S[a, b] >= ASSOC_MIN:
                matched[trk[a]] = (dets[b][0], float(S[a, b])); matched_det.add(b)
        CLIP_PASS["assignable"] += len(trk)
        CLIP_PASS["matched"] += len(matched)
        CLIP_PASS["accepted"] += [v[1] for v in matched.values()]
    closed_now = []
    for a, s in enumerate(trk):
        if s in matched:
            strikes[s] = 0; continue
        best = float(S[a].max()) if len(dets) else 0.0
        if best < STRIKE_MAX:
            strikes[s] += 1
            CLIP_PASS["strikes"] += 1
            log_event(type="strike", slot=s, seg_id=seg[s]["seg_id"] if seg[s] else None,
                      clip=clip, t=round(t_sec, 2), best_overlap=round(best, 4), n=strikes[s])
            if strikes[s] >= (CFG["strikes_day"] if is_day else CFG["strikes_night"]):   # [v5-C10] = 2/3 at v4 cadences
                close_segment(s, wall, "detector_contradiction")
                closed_now.append(s); CLIP_PASS["closes"] += 1
        # 0.10..0.30: ambiguous — neither match nor strike; counter unchanged
    # ---- unmatched detections -> reopen / displacement (with night stability) ----------------
    un = [dets[b][0] for b in range(len(dets)) if b not in matched_det]
    opens, open_extra, dark_res, dark_evs, pen_refused = [], {}, None, [], False   # [v5-C12]
    if DARK_FIRE:                                    # [v5-C12] step 3: the dark step runs BEFORE the v4 loop
        dark_res, dark_evs = _dark_step(wall, clip, t_sec, dets, matched_det, scheduled, pool, is_day, dl,
                                        opens, open_extra)
        if dark_res is not None and dark_res["used"]:
            un = [dets[b][0] for b in range(len(dets)) if b not in matched_det and b not in dark_res["used"]]
    un_ok, un_why = [True] * len(un), [None] * len(un)   # [v5-C13] per-detection veto (v5_fixes off: never vetoed)
    if not scheduled:
        un, un_ok, un_why = [], [], []               # [v5-C12] no unmatched loop on a dark-only pass
    elif V5_FIXES:                                   # [v5-C13] band + conflict + pen-count guard = a VETO on opens
        # and displacements only. The loop below -- pending bookkeeping, two-sighting rule, ignored_detection and its
        # whole-animal scene event (which gates the night repairs) -- still sees EVERY unmatched detection, as in v4
        # (review r1: filtering the list hid that signal; the ignored_detection branch is unchanged)
        un_ok, un_why, pen_refused = core.reopen_considered(un, [last_gen_mask[l] for l in _live()], pool, BAND,
                                                            dl["n_inband"], ios=m_ios, area=_area_px)
        if pen_refused:
            V5C["pen_count_refused"]["scheduled"] += 1
            dark_evs.append(core.ev_pen_count_refused("scheduled", dl["n_inband"], [_area_px(m) for m, _s in dets],
                                                      [_s for _m, _s in dets], clip, t_sec))
    survivors = []
    for p in pending_open:
        if (wall - p["wall"]).total_seconds() <= 2.5 * (DETECT_DAY_SEC if is_day else DETECT_NIGHT_SEC):
            survivors.append(p)
    if scheduled: pending_open = survivors           # [v5-C12] dark-only passes never touch pending_open
    for dm, ok_, why_ in zip(un, un_ok, un_why):     # [v5-C13] ok_ False = vetoed (why_: pen_count | band | conflict)
        hit = None
        for p in pending_open:
            if m_iou(p["mask"], dm) >= NIGHT_STABLE_IOU or m_ios(p["mask"], dm) >= NIGHT_STABLE_IOU:
                hit = p; break
        persisted = hit is not None
        if hit: hit["mask"], hit["wall"], hit["seen"] = dm, wall, hit["seen"] + 1
        else: pending_open.append({"mask": dm, "wall": wall, "seen": 1})
        free = [s for s in range(N_CALVES) if state[s] == ST_CLOSED and s not in [o[0] for o in opens]]
        if is_day and free:
            if ok_: opens.append((free[0], dm, "reopen"))                  # [v5-C13] veto
            else: V5C["sched_vetoed"][why_] += 1
        elif (not is_day) and persisted and free:
            if ok_: opens.append((free[0], dm, "reopen"))                  # [v5-C13] veto
            else: V5C["sched_vetoed"][why_] += 1
        elif persisted and not free:
            occ = [(s, (wall - last_gen_wall[s]).total_seconds())
                   for s in range(N_CALVES) if state[s] == ST_OCC and last_gen_wall[s] is not None
                   and s not in [o[0] for o in opens]]
            if occ and not ok_:
                V5C["sched_vetoed"][why_] += 1                             # [v5-C13] veto: no displacement
            elif occ:
                s_disp = max(occ, key=lambda z: z[1])[0]
                close_segment(s_disp, wall, "displaced_by_detection"); CLIP_PASS["closes"] += 1
                opens.append((s_disp, dm, "displaced"))
            elif all(state[s] == ST_TRACK for s in range(N_CALVES)):
                _whole = False
                if PARTIAL_REPAIR:
                    _meds = _slot_day_medians()
                    _ref = float(np.median(list(_meds.values()))) if _meds else None
                    _whole = bool(_ref is not None and dm.sum() >= REPAIR_AREA_LO * _ref)
                    if _whole: note_scene_event(wall)   # the animal a hopper left behind
                log_event(type="ignored_detection", clip=clip, t=round(t_sec, 2),
                          area=int(dm.sum()), whole_animal=_whole,
                          note="pen holds exactly 4; all slots tracking",
                          **({"filtered": why_} if why_ else {}))            # [v5-C13] vetoed detections too
    if scheduled:                                    # [v5-C12]
        pending_open = [p for p in pending_open
                        if not any(m_iou(p["mask"], o[1]) >= NIGHT_STABLE_IOU for o in opens)]
    if opens and (dl["n_inband"] or 0) > N_CALVES:   # [v5-C14] worker-guard audit (G2): stays 0 under v5_fixes
        V5C["pen_guard_violations"] += 1
    # ---- optional conservative refresh (day only, overlap-gated, rate-limited) ---------------
    refreshed = {}
    if (REFRESH_MIN_SEC > 0 and is_day and matched
            and (last_refresh_wall is None or (wall - last_refresh_wall).total_seconds() >= REFRESH_MIN_SEC)
            and not any(state[s] == ST_OCC and last_gen_wall[s] is not None
                        and (wall - last_gen_wall[s]).total_seconds() < SEG_BREAK_ABSENCE
                        for s in range(N_CALVES))):
        for s, (dm, sc) in matched.items():
            if (core.refresh_ok(last_gen_mask[s], dm, [last_gen_mask[l] for l in _live() if l != s], pool, BAND,
                                iou=m_iou, ios=m_ios, area=_area_px)
                    if V5_FIXES else sc >= REFRESH_IOU): refreshed[s] = dm   # [v5-C13] refresh gate
        if refreshed: last_refresh_wall = wall
    # ---- partial-band / growth repair: replace a sustained excursion with the current detection
    # Day-anchored gates only (see param block). PRECONDITIONS (revision 2): night; every slot
    # TRACKING with a live mask (the pen holds exactly 4 — a dark/closed slot means an unclaimed
    # animal the matched detection may belong to); the scene QUIET (no guard/merge/segment event
    # pen-wide for REPAIR_QUIET_SEC) and the slot's mask STATIONARY for REPAIR_STABLE_SEC — the
    # measured signature of the plateau regime, and the negation of the dusk-hopping regime
    # where the first patch produced a silent switch. A repair is a re-seed from a detection and
    # therefore OPENS A NEW SEGMENT: the offline identity layer decides it. Every skip / reject /
    # accept is logged — a correction layer must never act or abstain silently.
    repaired = {}
    if PARTIAL_REPAIR and scheduled:                 # [v5-C12] the dark step never touches repair
        armed = [s for s in range(N_CALVES)
                 if (partial_run[s] >= PARTIAL_N or grown_run[s] >= GROW_N) and s not in refreshed]
        all_tracking = all(state[s] == ST_TRACK and last_gen_mask[s] is not None
                           for s in range(N_CALVES))
        quiet = scene_quiet(wall)
        for s in armed:
            mode = "partial" if partial_run[s] >= PARTIAL_N else "grown"
            stable, dmax = slot_stable(s, wall)
            skip = None
            if REPAIR_NIGHT_ONLY and is_day: skip = "not_night"
            elif not all_tracking: skip = "not_all_tracking"
            elif not quiet: skip = "scene_active"
            elif not stable: skip = "unstable"
            elif s not in matched: skip = "unmatched"
            elif repairs_clip[s] >= REPAIR_MAX_PER_CLIP: skip = "suspended_this_clip"
            elif last_repair_wall[s] is not None and \
                 (wall - last_repair_wall[s]).total_seconds() < REPAIR_MIN_SEC: skip = "rate_limited"
            drf = day_ref_of(s)
            if skip is None and drf is None: skip = "no_day_ref"
            if skip is not None:
                log_event(type="partial_repair_skipped", slot=s, mode=mode,
                          seg_id=seg[s]["seg_id"] if seg[s] else None, clip=clip,
                          t=round(t_sec, 2), reason=skip,
                          stable_px=None if dmax is None else round(dmax, 1))
                continue
            dm, sc = matched[s]
            da, ca = int(dm.sum()), int(last_gen_mask[s].sum())
            contain = m_ios(last_gen_mask[s], dm)      # intersection / min-area: the smaller one
            conflict = max([m_ios(dm, last_gen_mask[s2]) for s2 in range(N_CALVES) if s2 != s],
                           default=0.0)
            n_inside = 0
            if mode == "partial":
                size_ok = REPAIR_AREA_LO * drf <= da <= REPAIR_AREA_HI * drf and da >= REPAIR_GROW_MIN * ca
            else:
                # a mask containing TWO deduped detections is two animals: which one it "is"
                # would be decided by Hungarian tie-break order — refuse (review F3)
                n_inside = sum(1 for d2, _ in dets if m_ios(last_gen_mask[s], d2) >= REPAIR_CONTAIN_IOS)
                size_ok = (REPAIR_AREA_LO * drf <= da <= REPAIR_AREA_HI * drf
                           and da <= REPAIR_SHRINK_MAX * ca and n_inside < 2)
            ok = (contain >= REPAIR_CONTAIN_IOS and size_ok and conflict < REPAIR_CONFLICT_IOS)
            base = "partial_repair" if mode == "partial" else "growth_repair"
            log_event(type=base if ok else base + "_rejected", slot=s, mode=mode,
                      seg_id=seg[s]["seg_id"] if seg[s] else None, clip=clip, t=round(t_sec, 2),
                      det_area=da, cur_area=ca, day_ref=int(drf),
                      contain=round(contain, 3), conflict=round(conflict, 3),
                      stable_px=round(dmax, 1), n_dets_inside=n_inside)
            if ok:
                repaired[s] = dm
                # REVISION 2: the repair is a segment boundary — close the excursion segment and
                # open a new one seeded by this frame's detection (INV-1-legal). L2 decides it.
                if seg[s] is not None and seg[s]["n_frames"] > 0: seg[s]["n_frames"] -= 1
                close_segment(s, wall, base, drop_mask=False,
                              extra={"repair_from_area": ca, "repair_to_area": da})
                open_segment(s, wall, base)          # resets guards, partial/grown state, cent_hist
                seg[s]["n_frames"] = 1               # this frame's row carries the new seed
                last_gen_mask[s] = dm; last_gen_wall[s] = wall
                if DARK_FIRE: _note_healthy(s, dm, wall)                   # [v5-C11]
                hist[s].clear(); hist[s].append((wall, dm[::4, ::4].copy()))
                last_repair_wall[s] = wall
                repairs_clip[s] += 1; CLIP_PASS["repairs"] += 1
                if repairs_clip[s] >= REPAIR_MAX_PER_CLIP:
                    log_event(type="partial_repair_suspended", slot=s, clip=clip,
                              t=round(t_sec, 2), n=repairs_clip[s],
                              note="oscillation tripwire: no more repairs for this slot this clip")
        if repaired:
            refreshed = {**refreshed, **repaired}
            LAST_REPAIRS.update(repaired)
    # ---- rebuild the session once if membership or seeds changed -----------------------------
    rebuilt = False
    if opens or refreshed:
        seeds = []
        for s in range(N_CALVES):
            if s in [o[0] for o in opens]: continue
            if state[s] == ST_TRACK and last_gen_wall[s] is not None and \
               (wall - last_gen_wall[s]).total_seconds() <= FRESH_SEC:
                seeds.append((s, refreshed.get(s, last_gen_mask[s])))
            elif state[s] == ST_OCC:
                # not seedable (>FRESH_SEC dark) — an honest break, L2 stitches it
                close_segment(s, wall, "not_seedable_at_rebuild"); CLIP_PASS["closes"] += 1
        for slot, dm, reason in opens:
            open_segment(slot, wall, reason, extra=open_extra.get(slot))    # [v5-C14] link evidence / via
            last_gen_wall[slot] = wall; last_gen_mask[slot] = dm
            seeds.append((slot, dm)); CLIP_PASS["opens"] += 1
            if DARK_FIRE: _note_healthy(slot, dm, wall)                   # [v5-C11] reset to the seed if in band
        if seeds:
            if session is not None: del session
            gc.collect(); torch.cuda.empty_cache()
            session, session_slots = open_session_v4(pil, seeds)
            rebuilt = True
            merge_run.clear(); merge_onset.clear()
            V5C["rebuilds"] += 1                                          # [v5-C14]
    # [v5-C12/C13] rows of THIS frame carry the seed detection for every dark-step open, every scheduled open under
    # v5_fixes (the stale-row fix) and every repair (= Tol_03 LAST_REPAIRS); new segments start at n_frames 1
    _ovr = core.row_overrides(opens, open_extra, repaired, V5_FIXES)
    for _s in _ovr:
        if _s not in repaired and seg[_s] is not None: seg[_s]["n_frames"] = 1
    ROW_OVERRIDE.update(_ovr)
    for slot, dm, reason in opens:                                        # [v5-C14] now that the seg_ids exist
        if reason == "dark_link":
            dark_evs.append(core.ev_dark_link(slot, seg[slot]["seg_id"], dark_res["links"][slot], clip, t_sec))
        elif open_extra.get(slot, {}).get("via") == "dark_pass":
            dark_evs.append(core.ev_dark_open(slot, seg[slot]["seg_id"], dark_res["opens"][slot], clip, t_sec))
    _ran = dark_res is not None and dark_res["ran"]
    if scheduled:                                                         # [v5-C12] dark-only calls emit dark_pass only
        _pk = dict(type="pass", clip=clip, t=round(t_sec, 2), forced=forced_by, n_det=len(dets),
                   tracked=len(trk), matched={s: round(v[1], 3) for s, v in matched.items()},
                   struck=closed_now, opened=[o[0] for o in opens],
                   refreshed=list(refreshed.keys()), rebuilt=rebuilt)
        if PARTIAL_REPAIR: _pk["repaired"] = list(repaired.keys())
        _pk.update(dark=None if not _ran else {"links": sorted(dark_res["links"]), "opens": sorted(dark_res["opens"]),
                                               "refused": dark_res["refused"], "deferred": dark_res["deferred"]},
                   n_raw=dl["n_raw"], n_inband=dl["n_inband"], det_ms=dl["det_ms"], detector=DETECTOR,
                   pen_refused=bool(pen_refused))                         # [v5-C14]
        log_event(**_pk)
    if _ran or not scheduled:                                             # [v5-C14]
        log_event(**core.ev_dark_pass(clip, t_sec, wall.isoformat(), dark_res if dark_res is not None else
                                      core.dark_step_decide(dets, set(), _slot_views(wall), pool, BAND, DCFG),
                                      "combined" if scheduled else "dark_only", dl["n_raw"], dl["det_ms"],
                                      (dark_res or {}).get("dark_slots", {}), rebuilt))
    for _e in dark_evs: log_event(**_e)
    return _ran or not scheduled                                          # [v5-C12] re-arms next_dark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resume state, staging, telemetry

# COMMAND ----------

def stage_clip(vname, dst):
    if osp.exists(dst): os.remove(dst)
    t = time.time()
    if STAGE_SVC:
        # Subprocess-worker mode: no dbutils here, and shutil over /dbfs FUSE runs at ~0.5 MB/s
        # (43 min per clip — measured). The launcher owns dbutils and stages on our behalf:
        # we drop a request file, it dbutils.fs.cp's the clip and answers with a ready file
        # whose content is the local path.
        req = osp.join(STAGE_SVC, f"req__{SVC_KEY}__{vname}")          # [v5-C1] SVC_KEY defaults to STAGE
        rdy = osp.join(STAGE_SVC, f"ready__{SVC_KEY}__{vname}")
        err = osp.join(STAGE_SVC, f"error__{SVC_KEY}__{vname}")
        tmp = osp.join(STAGE_SVC, f"tmp__{SVC_KEY}__{vname}")   # outside the launcher's req__* glob
        with open(tmp, "w") as f: f.write(dst)
        os.replace(tmp, req)      # atomic: the launcher must never read a half-written request
        deadline = time.time() + 1800
        while time.time() < deadline:
            if osp.exists(rdy):
                os.remove(rdy)
                break
            if osp.exists(err):
                msg = open(err).read(); os.remove(err)
                raise RuntimeError(f"staging service failed for {vname}: {msg}")
            time.sleep(3)
        else:
            raise RuntimeError(f"staging service timed out for {vname}")
        how = "staging-service"
    else:
        how = "dbutils.fs.cp"
        try:
            dbutils.fs.cp(f"{DBFS_VIDEO}/{vname}", "file:" + dst)
        except Exception as e:
            how = f"shutil (dbutils failed: {e})"
            shutil.copy(osp.join(VIDEO_DIR, vname), dst)
    el = time.time()-t
    mb = osp.getsize(dst)/1e6 if osp.exists(dst) else 0
    print(f"    staged {mb:.0f} MB in {el/60:.1f} min ({mb/max(el,1e-9):.1f} MB/s) via {how}", flush=True)
    return el

def restore_after(stem, clip_meta):
    """Rebuild slot state from a finished clip so a resumed run continues seamlessly."""
    global segments_done
    if osp.exists(SEG_PATH):
        with open(SEG_PATH) as f: d = json.load(f)
        segments_done = [r for r in d["segments"] if not r.get("open")]
        for s in range(N_CALVES): seg[s] = None; state[s] = ST_CLOSED
        for r in d["segments"]:
            sc = int(r["seg_id"].split("_")[1])
            seg_counter[r["slot"]] = max(seg_counter[r["slot"]], sc)
            if r.get("open"):
                rr = {k: v for k, v in r.items() if k != "open"}
                seg[rr["slot"]] = rr; state[rr["slot"]] = ST_OCC   # OCC until a mask proves fresh
    gzp = osp.join(TRACK_DIR, f"masks_{stem}.json.gz")
    if not osp.exists(gzp): return
    with gzip.open(gzp, "rt") as f: dprev = json.load(f)
    frames = dprev.get("frames", [])
    if not frames: return
    end_t = frames[-1]["t_sec"]
    # Rebuild the slow day-reference history from the finished clip (RLE area = sum of the
    # foreground runs, no decode): the partial-band repair must survive a resume with its
    # day anchor intact, and area_hist/day_area_hist otherwise live only in memory.
    if PARTIAL_REPAIR:
        cand = []
        for r in frames[::DAYREF_STEP]:
            w = clip_meta["start"] + timedelta(seconds=r["t_sec"])
            if tod_of_wall(w) != "day": continue
            for s in range(N_CALVES):
                rl = r["masks"].get(str(s))
                if rl: cand.append((s, int(sum(rl["counts"][1::2]))))
        if cand:
            # two passes: the live learner skips partial-band frames, so skip what it would have
            # skipped (areas below PARTIAL_FRAC x the clip's pooled median) -- otherwise a resume
            # re-ingests exactly the plateau frames the gate exists to keep out
            pm = float(np.median([a for _, a in cand]))
            for s, a in cand:
                if a >= PARTIAL_FRAC * pm: day_area_hist[s].append(a)
            DAYREF_VER[0] += 1
    for s in range(N_CALVES):
        if seg[s] is None: continue
        for r in reversed(frames):
            if end_t - r["t_sec"] > CLIP_CARRY_MAX * 2: break
            rl = r["masks"].get(str(s))
            if rl:
                last_gen_wall[s] = clip_meta["start"] + timedelta(seconds=r["t_sec"])
                last_gen_mask[s] = rle_decode(rl)
                state[s] = ST_TRACK
                break

def save_obs():
    if not obs: return
    E = np.stack([o["emb"] for o in obs]).astype(np.float16)
    meta = [{k: v for k, v in o.items() if k != "emb"} for o in obs]
    tmp = osp.join(LOCAL, "obs.npz")
    np.savez_compressed(tmp, emb=E)            # /local_disk0 then copy: Volumes reject seek-on-write
    shutil.copy(tmp, osp.join(TRACK_DIR, "observations.npz")); os.remove(tmp)
    with open(osp.join(TRACK_DIR, "observations_meta.json"), "w") as f:
        json.dump({"n": len(obs), "dim": int(E.shape[1]), "embed_every_sec": EMBED_EVERY_SEC,
                   "keyed_by": "slot+seg_id (anonymous; identity is offline)", "obs": meta}, f)

# ---- [v5-C1] config echo, guarded params, ROOT_CONFIG.json (one config per root) --------
def _vers(pkg):
    try: return importlib.metadata.version(pkg)
    except Exception: return None
def _source_bytes():
    """The bytes of the code that is RUNNING. Subprocess mode: this file (the staged copy the
    launcher pinned). Notebook mode (direct tasks): this notebook, read through the workspace FUSE when that yields the
    SOURCE format, else exported (format SOURCE) with the Databricks SDK. Returns (bytes | None, basis / errors)."""
    if not IS_NOTEBOOK:
        _p = osp.abspath(__file__)
        with open(_p, "rb") as f: return f.read(), "file:" + _p
    try:
        nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    except Exception as e:
        return None, f"notebook path unreadable: {e!r}"
    errs = []
    for cand in ("/Workspace" + nbp, "/Workspace" + nbp + ".py"):
        try:
            if osp.isfile(cand):
                with open(cand, "rb") as f: b = f.read()
                if b.lstrip(b"\xef\xbb\xbf").startswith(b"# Databricks notebook source"): return b, "fuse:" + cand
                errs.append(f"{cand}: not the SOURCE format")
        except Exception as e:
            errs.append(f"{cand}: {e!r}")
    try:
        import base64
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.workspace import ExportFormat
        return (base64.b64decode(WorkspaceClient().workspace.export(nbp, format=ExportFormat.SOURCE).content),
                "sdk_export:" + nbp)
    except Exception as e:
        errs.append(f"sdk export: {e!r}")
    return None, "; ".join(errs)
_SRC_BYTES, SOURCE_SRC = _source_bytes()
# review r1: two digests, one name = one meaning (core.source_shas).
#   shas.source      = sha256 of the RAW running bytes: what the launcher and gate_v5 compare with the
#                      payload pin (subprocess mode = the staged, pinned copy; equiv always runs there);
#   shas.source_norm = core.source_norm_sha256 (EOL / trailing-space / blank-line insensitive): the GUARDED value
#                      (ROOT_CONFIG.json shas.source_norm, resume guard source_norm_sha256) -- a notebook read through
#                      the workspace FUSE or exported by the SDK and the local .py give one value.
if _SRC_BYTES is not None:
    SRC_SHAS = core.source_shas(_SRC_BYTES)
    print(f"Tol_47 source ({SOURCE_SRC}): sha256 {SRC_SHAS['source']}, normalised {SRC_SHAS['source_norm']}")
elif MODE == "track":
    raise RuntimeError("Tol_47 cannot fingerprint its own source, so ROOT_CONFIG.json and the resume guard would be "
                       f"blind to a code change across days: {SOURCE_SRC}")
else:
    SRC_SHAS = {"source": "n/a", "source_norm": "n/a"}
    print(f"⚠ Tol_47 source sha256 unavailable (mode=equiv writes no tracking root): {SOURCE_SRC}")
SHAS = {"detector": DET_SHA, "student": STUDENT_SHA, "core": CORE_SHA, **SRC_SHAS}
ECHO = {"notebook": "Tol_47_track_v5", "core_version": core.CORE_VERSION, "mode": MODE, "widgets": CFG["values"],
        "derived": {k: CFG[k] for k in ("numeric_change", "dark_link", "dark_open", "dark_first_sec", "strikes_day",
                                        "strikes_night", "pending_day", "pending_night", "is_regression")},
        "shas": SHAS, "core_file": core.__file__, "core_dir": CORE_DIR, "source_file": SOURCE_SRC,
        "source_raw_sha256": SRC_SHAS["source"], "source_sha_rule": "shas.source = sha256 of the running bytes (= the "
        "payload pin in subprocess mode); shas.source_norm = core.source_norm_sha256 (EOL / trailing whitespace / "
        "blank-line insensitive), the value ROOT_CONFIG.json and the resume guard hold -- reproduce it locally from the "
        ".py", "op_threshold": OP_THRESHOLD,
        "band": {k: BAND[k] for k in ("A_REF_TRAIN", "FRAG_LO", "OVER_HI", "provenance", "A_REF_TRAIN_rule")},
        "band_px": {p: list(core.band(p, BAND)) for p in ("dayl", "dark")},
        "gpu": {"name": GPU_NAME, "class": GPU_CLASS, "capability": list(torch.cuda.get_device_capability(0)),
                "total_gib": round(_props.total_memory / 2**30, 2), "driver": PROBE.get("nvidia_smi")},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "rfdetr": _vers("rfdetr"),
                     "supervision": _vers("supervision"), "cv2": cv2.__version__, "timm": _timm_v.__version__,
                     "numpy": np.__version__, "Pillow": _pil_v.__version__, "accelerate": _vers("accelerate"),
                     "scipy": _scipy_v.__version__, "python": sys.version.split()[0]},
        "matmul_precision": torch.get_float32_matmul_precision(), "decode": DECODE, "numeric_change": NUMERIC_CHANGE,
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "cluster_id": CLUSTER_ID,
        "cuda_probe": PROBE, "rf_max_memory_allocated_gib": RF_PEAK_GIB, "mem_cap_gib": MEM_CAP_GIB,
        "stage_svc_key": SVC_KEY}
GUARD = core.guard_params(CFG, SHAS, GPU_CLASS, OP_THRESHOLD)
print("v5 config echo:\n" + json.dumps(ECHO, indent=1, default=str))
if MODE == "track":
    # Critic finding 11: two tasks of one root start together (L3B || L3C, L4 04-22 || 04-23), so each file is published
    # by an ATOMIC write-if-absent (temp file + link / rename, core.write_json_if_absent) and read back with a retry on a
    # half-visible file; every task -- the publisher included -- then checks the stored record against its own, so a
    # concurrent task with a different config raises instead of racing.
    ROOT_REC = core.root_config_record(CFG, SHAS, GPU_CLASS, BAND)
    _rc_cmp = json.loads(json.dumps(ROOT_REC, default=str))       # compare in the stored (JSON) representation
    _rc_root, _rc_day = osp.join(TOL, TRACK_ROOT, "ROOT_CONFIG.json"), osp.join(TRACK_DIR, "ROOT_CONFIG.json")
    for _p in (_rc_root, _rc_day):
        _new = core.write_json_if_absent(_p, dict(ROOT_REC, written_by=f"{STAGE}{WSUF}", created=datetime.now().isoformat(),
                                                  echo=ECHO), tag=f"{STAGE}{WSUF}")
        _tol = core.root_config_check(core.read_json_retry(_p), _rc_cmp, ALLOW_MIXED_GPU)
        print(f"ROOT_CONFIG {_p}: {'written by this task' if _new else 'present'}, matches"
              + (f" (tolerated: {_tol})" if _tol else ""))

TELEMETRY = {"stage": STAGE, "track_fps": TRACK_FPS, "params": {
                 "fresh_sec": FRESH_SEC, "seg_break_absence": SEG_BREAK_ABSENCE,
                 "assoc_min": ASSOC_MIN, "strike_max": STRIKE_MAX,
                 "detect_day_sec": DETECT_DAY_SEC, "detect_night_sec": DETECT_NIGHT_SEC,
                 "refresh_min_sec": REFRESH_MIN_SEC,
                 "partial_repair": PARTIAL_REPAIR},
             "clips": []}
if PARTIAL_REPAIR:
    TELEMETRY["params"].update({"partial_frac": PARTIAL_FRAC,
                                "repair_area_band": [REPAIR_AREA_LO, REPAIR_AREA_HI],
                                "repair_grow_min": REPAIR_GROW_MIN,
                                "repair_contain_ios": REPAIR_CONTAIN_IOS,
                                "repair_conflict_ios": REPAIR_CONFLICT_IOS,
                                "repair_max_per_clip": REPAIR_MAX_PER_CLIP,
                                "guard_floor_frac": GUARD_FLOOR_FRAC,
                                "revision": 2, "repair_night_only": REPAIR_NIGHT_ONLY,
                                "repair_stable_sec_px": [REPAIR_STABLE_SEC, REPAIR_STABLE_PX],
                                "repair_quiet_sec": REPAIR_QUIET_SEC,
                                "repair_is_boundary": True, "grow_frac": GROW_FRAC,
                                "repair_shrink_max": REPAIR_SHRINK_MAX})
TELEMETRY["params"].update({"v5": ECHO, "v5_guard": GUARD})              # [v5-C1]
# track_root in the name: a teacher run and a student run of the SAME day must not share a
# telemetry file (the teacher GT run silently overwrote the student day's summary once)
TELEM_PATH = osp.join(TOL, f"Metrics_track_v5_{TRACK_ROOT}_{STAGE}{WSUF}.json")   # [v5-C1]
if MODE == "track":
    # resume: keep completed clips' telemetry entries — they are the day's acceptance evidence
    # [v5-C1] the guards now RAISE (Tol_03 caught its own RuntimeError in the except below and only printed it), and
    # so does a MISSING / UNREADABLE / INCOMPLETE file while .done_ clips exist (review r1: an empty clip list would let
    # the next write erase the done clips' evidence and would silently skip the resume guard) -- core.resume_telemetry
    _done = [osp.basename(p)[len(".done_"):] for p in glob.glob(osp.join(TRACK_DIR, ".done_*"))]
    _old, _kept = core.resume_telemetry(TELEM_PATH, _done)
    if _kept:
        _op = _old.get("params", {}).get("partial_repair")
        if _op is not None and bool(_op) != PARTIAL_REPAIR:
            raise RuntimeError(f"resume with partial_repair={PARTIAL_REPAIR} but {len(_kept)} done "
                               f"clips were tracked with {_op} -- a day must not mix behaviors")
        _tolr = core.resume_guard(_old.get("params", {}).get("v5_guard", {}), GUARD, ALLOW_MIXED_GPU)
        if _tolr: print(f"⚠ resume across GPU classes tolerated (allow_mixed_gpu): {_tolr}")
        TELEMETRY["clips"] = _kept
        print(f"resume: preloaded {len(_kept)} clip telemetry entries")

def finish_clip_telemetry(stem, rows, t_clip, t_stage):
    cp = CLIP_PASS
    med = float(np.median(cp["accepted"])) if cp["accepted"] else None
    acc = cp["matched"]/cp["assignable"] if cp["assignable"] else None
    n_open_rows = sum(1 for r in rows if r["_n_open"] >= 3)
    n_good_rows = sum(1 for r in rows if r["_n_open"] >= 3 and r["_n_gen"] >= 3)
    cov3 = n_good_rows/n_open_rows if n_open_rows else None
    entry = {"clip": stem, "frames": len(rows), "passes": cp["passes"],
             "assignable": cp["assignable"], "matched": cp["matched"],
             "acceptance": None if acc is None else round(acc, 3),
             "median_accepted_overlap": None if med is None else round(med, 3),
             "strikes": cp["strikes"], "opens": cp["opens"], "closes": cp["closes"],
             "merge_events": cp["merges"],
             "coverage3_frac": None if cov3 is None else round(cov3, 3),
             "track_min": round(t_clip/60, 1), "stage_min": round(t_stage/60, 1)}
    if PARTIAL_REPAIR: entry["repairs"] = cp.get("repairs", 0)
    entry["v5"] = _v5_clip_entry(len(rows), t_clip)                      # [v5-C14] t_clip = elapsed track s
    entry["gpu"], entry["gpu_class"] = GPU_NAME, GPU_CLASS               # [v5-C14] also top-level: Tol_48 reads clip["gpu"]
    TELEMETRY["clips"].append(entry)
    _tl = osp.join(LOCAL, osp.basename(TELEM_PATH))   # [v5-C14] review r1: built on local disk, then published (.part
    with open(_tl, "w") as f: json.dump(TELEMETRY, f, indent=1)   # + replace): a crash never leaves a truncated file
    core.publish_file(_tl, TELEM_PATH)
    print(f"    telemetry: acc={entry['acceptance']} med_ovl={entry['median_accepted_overlap']} "
          f"strikes={cp['strikes']} opens={cp['opens']} closes={cp['closes']} cov3={entry['coverage3_frac']}")
    # ---- INV-7: die loudly on the v3 signature instead of completing plausibly ---------------
    if cp["assignable"] >= 10 and acc is not None and acc < 0.5:
        raise RuntimeError(f"ABORT {stem}: association acceptance {acc:.2f} < 0.5 "
                           f"(v3 signature was 0.32). Masks are not where calves are.")
    if len(cp["accepted"]) >= 5 and med is not None and med < 0.20:
        raise RuntimeError(f"ABORT {stem}: median accepted overlap {med:.3f} < 0.20 "
                           f"(v3 signature was 0.000).")
    if n_open_rows >= 1000 and cov3 is not None and cov3 < 0.5:
        raise RuntimeError(f"ABORT {stem}: only {cov3:.0%} of frames carry >=3 genuine masks "
                           f"while >=3 slots are open — tracking has collapsed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## [v5-C6/C14] per-clip v5 telemetry, synchronised laps, memory tripwires, detections log

# COMMAND ----------

# [v5-C14] synchronised per-component laps (Tol_45 convention: a sync at every boundary puts GPU time in the component
# that queued it) and the per-clip counters of the v5 telemetry. Instrumentation only: no decision reads them.
LAP_SYNC = True
LAPS = {}; _LAP_T = [time.perf_counter()]

def _lap(name):
    if LAP_SYNC: torch.cuda.synchronize()
    now = time.perf_counter(); LAPS[name] = LAPS.get(name, 0.0) + (now - _LAP_T[0]); _LAP_T[0] = now

def _lap_reset():
    _LAP_T[0] = time.perf_counter()

RESUME_INFO = {"resumed": False, "restored_clips": []}
V5C = {"sessions": 0, "rebuilds": 0, "resume_logged": False}
GATE_PEAK_ALLOC_GIB, GATE_PEAK_RESERVED_GIB = 4.0, 10.0   # [runtime] the gate_v5 G2 / G3-must-hold-8 levels

def _rss_gib():
    try:
        with open("/proc/self/statm") as f: return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30, 3)
    except Exception:
        return None

def _tracked_entries(sess):
    """Host-side per-frame bookkeeping SAM3 never prunes (session.frames_tracked_per_obj, TV:174/:1833): the measured
    driver of the ~0.46 GB/run-hour host-RSS trend (full A100 profile 2026-09-24). Logged only; no behaviour change."""
    try:
        return int(sum(len(v) for v in (getattr(sess, "frames_tracked_per_obj", None) or {}).values()))
    except Exception:
        return None

def _v5c_reset(clip):
    V5C.update({"clip": clip, "sessions": 0, "rebuilds": 0, "max_det_calls_per_frame": 0, "max_pf": 0,
                "rss_peak_gib": 0.0, "rss_start_gib": _rss_gib(), "cpu_s": 0.0, "mem_checks": 0, "mem_rechecks": 0,
                "pen_guard_violations": 0,
                "dark": {"onsets": 0, "passes": 0, "links": 0, "links_sole": 0, "opens": 0, "refused_by_reason": {},
                         "rejected_by_reason": {}, "deferred": 0, "suspended": 0},
                "pen_count_refused": {"dark": 0, "scheduled": 0, "seed": 0},
                "sched_vetoed": {"pen_count": 0, "band": 0, "conflict": 0}})   # opens / displacements v5_fixes blocked
    LAPS.clear(); DET_LOG.clear()
    torch.cuda.reset_peak_memory_stats()
_v5c_reset(None)

def _mem_bad(a, r, npf):
    bad = []
    if POP_FRAMES and npf > MEM_ASSERT_SESSION_FRAMES: bad.append(f"processed_frames {npf} > {MEM_ASSERT_SESSION_FRAMES}")
    if a >= MEM_ASSERT_ALLOC_GIB: bad.append(f"allocated {a:.2f} GiB >= {MEM_ASSERT_ALLOC_GIB}")
    if r >= MEM_ASSERT_RESERVED_GIB: bad.append(f"reserved {r:.2f} GiB >= {MEM_ASSERT_RESERVED_GIB}")
    return bad

def _mem_check(si, clip, where="periodic"):
    """[v5-C6] in-run memory tripwires, every MEM_CHECK_EVERY frames and at clip end (mode=equiv is exempt):
    processed_frames <= mem_assert_session_frames (pop on), allocated < mem_assert_alloc_gib, reserved <
    mem_assert_reserved_gib. On failure: mem_assert_fail, sync the event log, raise. Not fooled by a DELAYED FREE: a
    torn-down session can sit in a reference cycle until the next gc.collect() (full A100 profile 2026-09-24, 4 of 16
    frees), so an ALLOCATED failure is re-measured once after gc.collect() -- cyclic garbage is not a leak, and
    anything still live stays counted; the recheck is logged. No empty_cache() there (review r1): it would hand every
    cached block back and so hide exactly the fragmentation the reserved tripwire exists for --
    reserved is judged on the FIRST measurement (session teardowns already gc + empty_cache). processed_frames counts
    the live session: never rechecked."""
    if MODE != "track": return
    a = torch.cuda.memory_allocated() / 2**30; r = torch.cuda.memory_reserved() / 2**30
    npf = core.n_pf(session)                            # None-safe: no session / processed_frames None
    V5C["max_pf"] = max(V5C["max_pf"], npf); V5C["mem_checks"] += 1
    rss = _rss_gib()
    if rss is not None: V5C["rss_peak_gib"] = max(V5C["rss_peak_gib"], rss)
    log_event(**core.ev_mem("mem_check", si, a, r, npf, clip=clip, where=where, host_rss_gib=rss,
                            tracked_entries=_tracked_entries(session)))
    bad = _mem_bad(a, r, npf)
    if a >= MEM_ASSERT_ALLOC_GIB and not (POP_FRAMES and npf > MEM_ASSERT_SESSION_FRAMES):
        gc.collect(); V5C["mem_rechecks"] += 1
        a2 = torch.cuda.memory_allocated() / 2**30
        log_event(**core.ev_mem("mem_check", si, a2, r, npf, clip=clip, where="gc_recheck",
                                before_gib=[round(a, 3), round(r, 3)], reserved_from="first_measurement"))
        a = a2
        bad = _mem_bad(a, r, npf)
    if bad:
        log_event(**core.ev_mem("mem_assert_fail", si, a, r, npf, clip=clip, where=where, why=bad))
        sync_events()
        raise RuntimeError(f"ABORT {clip}: memory tripwire at frame {si}: {'; '.join(bad)}")

def _write_detections(stem):
    """[v5-C14] detections_<clip>.json.gz -- one record per detect() call {clip, frame_idx, t_sec, kind, n_raw, det_ms,
    n_inband, raw: [{rle, score, area}], kept_idx}; raw = after the score / class filter, before the post-chain (the
    sam3 regression arm logs its already post-chained list, n_raw None). Built on local disk, copied to the Volume."""
    tmp = osp.join(LOCAL, f"detections_{stem}.json.gz")
    with gzip.open(tmp, "wt") as f:
        json.dump({"clip": stem, "detector": DETECTOR, "det_run_tag": DET_RUN_TAG, "op_threshold": OP_THRESHOLD,
                   "band_px": {p: list(core.band(p, BAND)) for p in ("dayl", "dark")}, "records": DET_LOG}, f)
    shutil.copy(tmp, osp.join(TRACK_DIR, f"detections_{stem}.json.gz")); os.remove(tmp)

def _v5_clip_entry(n_frames, track_s):
    """[v5-C14] per-clip v5 telemetry + `gate`, the per-clip inputs of the automated G2 / G3-8 checks.
    track_s = the clip's ELAPSED tracking seconds (finish_clip_telemetry passes time.time() - t_clip; review r1: this
    was once treated as a start time, which zeroed fps and the host-RSS trend)."""
    el = max(float(track_s), 1e-9); vid_h = n_frames / TRACK_FPS / 3600.0
    ms = sorted(d["det_ms"] for d in DET_LOG)
    kinds = {}
    for d in DET_LOG: kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    try: free = round(torch.cuda.mem_get_info()[0] / 2**30, 3)
    except Exception: free = None
    pk_a, pk_r = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
    rss1, rss0 = _rss_gib(), V5C.get("rss_start_gib")
    if rss1 is not None: V5C["rss_peak_gib"] = max(V5C["rss_peak_gib"], rss1)
    gate = {"peak_allocated_ok": pk_a <= GATE_PEAK_ALLOC_GIB, "peak_reserved_ok": pk_r <= GATE_PEAK_RESERVED_GIB,
            "processed_frames_ok": (V5C["max_pf"] <= MEM_ASSERT_SESSION_FRAMES) if POP_FRAMES else None,
            "det_calls_per_frame_ok": V5C["max_det_calls_per_frame"] <= 1,
            "pen_guard_ok": V5C["pen_guard_violations"] == 0, "timings_present": bool(LAPS),
            "precision_highest": torch.get_float32_matmul_precision() == "highest"}
    gate["all_ok"] = all(v for v in gate.values() if v is not None)
    gate["levels"] = {"peak_allocated_gib": GATE_PEAK_ALLOC_GIB, "peak_reserved_gib": GATE_PEAK_RESERVED_GIB,
                      "processed_frames": MEM_ASSERT_SESSION_FRAMES, "det_calls_per_frame": 1}
    return {"gpu": GPU_NAME, "gpu_class": GPU_CLASS, "fps": round(n_frames / el, 3),
            "peak_allocated_gib": round(pk_a, 3), "peak_reserved_gib": round(pk_r, 3), "mem_free_gib": free,
            "host_rss_gib_start": rss0, "host_rss_gib_end": rss1,
            "host_rss_trend_gib_per_run_h": (round((rss1 - rss0) / (el / 3600.0), 3)
                                             if rss1 is not None and rss0 is not None and el >= 60 else None),
            "session_tracked_entries": _tracked_entries(session), "mem_rechecks": V5C["mem_rechecks"],
            "pen_guard_violations": V5C["pen_guard_violations"], "gate": gate,
            "max_processed_frames": V5C["max_pf"], "mem_checks": V5C["mem_checks"],
            "sessions": V5C["sessions"], "rebuilds": V5C["rebuilds"],
            "rebuilds_per_hour": round(V5C["rebuilds"] / vid_h, 2) if vid_h > 0 else None,
            "det_calls": {k: kinds.get(k, 0) for k in ("pass", "dark", "seed")},
            "max_det_calls_per_frame": V5C["max_det_calls_per_frame"],
            "det_ms_median": ms[len(ms) // 2] if ms else None,
            "det_ms_p95": ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))] if ms else None,
            "dark": V5C["dark"], "pen_count_refused": V5C["pen_count_refused"], "sched_vetoed": V5C["sched_vetoed"],
            "host_rss_gib": V5C["rss_peak_gib"], "cpu_ms_per_frame": round(1000 * V5C["cpu_s"] / max(n_frames, 1), 1),
            "ms_per_frame": {k: round(1000 * v / max(n_frames, 1), 2) for k, v in sorted(LAPS.items())},
            "lap_sync": LAP_SYNC, "decode": DECODE, "resumed": RESUME_INFO["resumed"],
            "restored_clips": RESUME_INFO["restored_clips"]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## [v5-C15] `mode=equiv` — the automated equivalence tests E1–E6 (writes a Metrics JSON, never a tracking root)
# MAGIC **E1** `processed_frames` pop is bit-identical: arm A pop off, arm B pop on, arm A′ = A again (A/A determinism
# MAGIC floor); 900 frames of the 04-22 12:45 clip at stride 5, seeds from one detect() on frame 0, a rebuild forced at
# MAGIC frame 450 from each arm's own frame-449 masks; `torch.equal` on `pred_masks` and `object_score_logits`, binarised
# MAGIC RLE; memory slope per arm. **E2** SHA-1 of 2,000 decoded frames, `seek` vs `seek_thread` (+ `sequential`,
# MAGIC report). **E3** student features bit-equal before / after `import rfdetr` and after one detect(), precision
# MAGIC `highest` after detect() (A/A first). **E4** exact fast twins vs the verbatim Tol_03 functions on every E1 mask,
# MAGIC temporal pair and detection. **E5** (report only) coexistence latency / peak memory on 300 frames, and the
# MAGIC domain shift of raw decodes vs the stored JPEG-q90 `replace_v1` RF-DETR detections (46 calls, 3 clips, 1 night).
# MAGIC **E6** regression pair: the Tol_03 reference root vs the Tol_47 regression root (masks, `segments.json`, events
# MAGIC projected onto the v4 schema) + the branch-coverage table.

# COMMAND ----------

if MODE == "equiv":
    EQ_T0 = time.time()
    EQ_TESTS = [t.strip().upper() for t in _V["equiv_tests"].split(",") if t.strip()]
    if [t for t in EQ_TESTS if t not in ("E1", "E2", "E3", "E4", "E5", "E6")]:
        raise RuntimeError(f"equiv_tests {EQ_TESTS}: only E1..E6")
    EQ = {"tests": EQ_TESTS, "job": SVC_KEY, "gpu": GPU_NAME, "gpu_class": GPU_CLASS, "created": datetime.now().isoformat(),
          "echo": ECHO, "results": {}}
    EQ_PATH = osp.join(TOL, f"Metrics_v5_equiv_{GPU_CLASS}_{datetime.now():%Y%m%d_%H%M%S}.json")
    def _eq_save():
        with open(EQ_PATH, "w") as f: json.dump(EQ, f, indent=1, default=str)
    def _mem_settle():
        """Settle the allocator before / after a memory measurement (never inside a measured arm): collect until no
        unreachable object is left -- a torn-down session can sit in a reference cycle until the next gc.collect()
        (full A100 profile 2026-09-24, 4 of 16 frees) -- and return the settled allocated MiB."""
        k = 0
        while gc.collect() and k < 5: k += 1
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        return round(torch.cuda.memory_allocated() / 2**20, 1)
    E1_CLIP = "Tol_ch01-20240422-124516-131900.mp4"
    E1_N, E1_REBUILD, E2_N, E5_N = 900, 450, 2000, 300
    # E1 memory verdict: core.E1_MEM through core.e1_memory_verdict -- the ONE definition gate_v5 uses
    # (review r1: this worker once had its own, wider band). Arm A keeps one bf16 [3,1008,1008] processed frame per
    # tracked frame = 5.814 MiB; with pop nothing stays per frame.
    if core.E1_MEM["prunes"] != E1_N // PRUNE_EVERY:
        raise RuntimeError(f"core.E1_MEM prunes {core.E1_MEM['prunes']} != E1_N // PRUNE_EVERY = {E1_N // PRUNE_EVERY}")
    E5_DETS = osp.join(TOL, "Detector_Eval", "replace_v1", "dets_rfdetr.json.gz")
    E5_CLIPS = ["Tol_ch01-20240422-124516-131900", "Tol_ch01-20240423-115209-122549", "Tol_ch01-20240423-220030-223435"]
    NEED_VIDEO = any(t in EQ_TESTS for t in ("E1", "E2", "E3", "E4", "E5"))
    if NEED_VIDEO:
        E1_DST = osp.join(LOCAL, "e1.mp4"); stage_clip(E1_CLIP, E1_DST)
        _c = cv2.VideoCapture(E1_DST)
        if not _c.isOpened(): raise RuntimeError("cv2 cannot open the E1 clip")
        E1_NATIVE = _c.get(cv2.CAP_PROP_FPS) or 25.0; E1_TOTAL = int(_c.get(cv2.CAP_PROP_FRAME_COUNT) or 0); _c.release()
        E1_STRIDE = max(1, round(E1_NATIVE / TRACK_FPS)); E1_W0 = parse(E1_CLIP)["start"]

    def _e1_frames(n, mode="seek"):
        """(fidx, pil) of the first n decodable E1 frames at the tracker's stride, through the tracker's decode path."""
        cap_ = cv2.VideoCapture(E1_DST)
        it = core.iter_frames(cap_, range(0, E1_TOTAL, E1_STRIDE), mode=mode, pos_prop=cv2.CAP_PROP_POS_FRAMES)
        k = 0
        try:
            for fidx, ok, fr in it:
                if not ok: continue
                yield fidx, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                k += 1
                if k >= n: break
        finally:
            it.close(); cap_.release()

    def _set_cur(fidx):
        CUR.update(clip=E1_CLIP, fidx=fidx, t_sec=fidx / E1_NATIVE, wall=E1_W0 + timedelta(seconds=fidx / E1_NATIVE))

    def _first_frame():
        g = _e1_frames(1); f0, p0 = next(g); g.close(); return f0, p0

    # ---- E3 first: it must see the process BEFORE rfdetr is imported ----
    if "E3" in EQ_TESTS:
        if BACKBONE != "student": raise RuntimeError("E3 needs backbone=student")
        if "rfdetr" in sys.modules: raise RuntimeError("E3 void: rfdetr is already imported")
        _f0, _p0 = _first_frame()
        def _feat():
            with torch.no_grad(): return student(STUDENT_INPUT(_p0)).float().cpu()
        f0, f0b = _feat(), _feat()
        load_rfdetr()
        _p_imp = torch.get_float32_matmul_precision()
        f1 = _feat()
        _set_cur(_f0); detect(_p0, topk=N_CALVES + 2, kind="equiv"); core.CACHE.clear()   # detect() fills the cache
        _p_det = torch.get_float32_matmul_precision()
        f2 = _feat()
        r = {"aa_equal": bool(torch.equal(f0, f0b)), "f0_eq_f1": bool(torch.equal(f0, f1)),
             "f0_eq_f2": bool(torch.equal(f0, f2)), "precision_after_import": _p_imp, "precision_after_detect": _p_det,
             "max_abs_f0_f1": float((f0 - f1).abs().max()), "max_abs_f0_f2": float((f0 - f2).abs().max())}
        r["pass"] = bool(r["aa_equal"] and r["f0_eq_f1"] and r["f0_eq_f2"] and _p_det == "highest")
        EQ["results"]["E3"] = r; _eq_save(); print("E3", r, flush=True)
    if NEED_VIDEO and RF is None:
        load_rfdetr()
    SHAS["detector"] = DET_SHA; ECHO["rf_max_memory_allocated_gib"] = RF_PEAK_GIB   # known only now (ECHO shares SHAS)
    EQ_BASE_MIB = _mem_settle() if NEED_VIDEO else None    # settled allocation, every model loaded, before E1/E4/E5
    EQ["base_mib"] = EQ_BASE_MIB

    # ---- E1 (+ E4 inline on arm B) ----
    E4_STATS = {"rle": [0, 0], "iou": [0, 0], "ios": [0, 0], "frames": 0}
    def _e4_check(masks_now, masks_prev, extra):
        ms = [m for m in list(masks_now) + list(extra) if m is not None]
        pv = [m for m in masks_prev if m is not None]
        core.CACHE.clear()
        for m in ms:
            E4_STATS["rle"][1] += 1; E4_STATS["rle"][0] += int(core.rle_fast(m) == _rle_encode_v4(m))
        for a, b in [(a, b) for i, a in enumerate(ms) for b in ms[i + 1:]] + [(a, b) for a in ms for b in pv]:
            E4_STATS["iou"][1] += 1; E4_STATS["iou"][0] += int(core.m_iou_fast(a, b) == _m_iou_v4(a, b))
            E4_STATS["ios"][1] += 1; E4_STATS["ios"][0] += int(core.m_ios_fast(a, b) == _m_ios_v4(a, b))
        core.CACHE.clear(); E4_STATS["frames"] += 1

    def _slope(ys):
        return None if len(ys) < 3 else round(float(np.polyfit(np.arange(len(ys), dtype=float), np.asarray(ys, float), 1)[0]), 4)

    def _floor_slope(alloc, lo, hi):
        """MiB/frame as the MEDIAN step between consecutive POST-PRUNE floors in frames [lo, hi): one delayed free (a
        single step down) can neither fake nor hide a per-frame growth, unlike the least-squares _slope."""
        fl = [(i, alloc[i]) for i in range(lo, min(hi, len(alloc))) if (i + 1) % PRUNE_EVERY == 0]
        if len(fl) < 3: return None
        return round(float(np.median([(b - a) / (j - i) for (i, a), (j, b) in zip(fl, fl[1:])])), 4)

    def _e1_arm(pop, seeds, e4=False, n=E1_N if "E1" in EQ_TESTS else 300, rebuild_at=E1_REBUILD):
        out = {"pred": [], "score": [], "rle": [], "alloc_mib": [], "npf": [], "fidx": [], "sam3_idx": [],
               "teardown": [], "prunes": 0}
        out["base_mib"] = _mem_settle()
        sess = slots = o = None; prev, prev_pil, prev_list = {}, None, []
        for i, (fidx, pil) in enumerate(_e1_frames(n)):
            if i == 0:
                sess, slots = open_session_v4(pil, seeds)
            elif i == rebuild_at:          # Tol_03 rebuild: seeds = this arm's own frame-449 masks, frame 449's image
                _sr = [(s, prev[s]) for s in slots if prev.get(s) is not None]
                o = None                   # the last SAM3 output must not keep the old session's tensors alive
                _a0 = torch.cuda.memory_allocated()
                del sess; gc.collect(); torch.cuda.empty_cache()             # = the tracker's rebuild teardown (one gc)
                out["teardown"].append({"i": i, "freed_mib": round((_a0 - torch.cuda.memory_allocated()) / 2**20, 1)})
                sess, slots = open_session_v4(prev_pil, _sr)
            res, o = track_frame(sess, slots, pil, pop=pop, raw=True)
            out["pred"].append(o.pred_masks.detach().to("cpu", copy=True))
            out["score"].append(o.object_score_logits.detach().to("cpu", copy=True))
            out["sam3_idx"].append(o.frame_idx)
            out["rle"].append({s: (None if m is None else _rle_encode_v4(m)) for s, m in res.items()})
            if (i + 1) % PRUNE_EVERY == 0:  # the tracker's own prune cadence (main loop si % 25), so
                prune(sess, 8); out["prunes"] += 1        # E1 tests the regime production runs (36 cycles / 900 frames)
                if not EXACT_FAST: gc.collect(); torch.cuda.empty_cache()
            out["alloc_mib"].append(torch.cuda.memory_allocated() / 2**20); out["npf"].append(core.n_pf(sess))
            out["fidx"].append(fidx)
            if e4: _e4_check(list(res.values()), prev_list, [m for m, _s in E1_SEEDS_KEPT])
            prev, prev_pil, prev_list = res, pil, list(res.values())
        o = res = None
        del sess; gc.collect(); torch.cuda.empty_cache()
        out["end_mib"] = _mem_settle()
        return out

    def _cmp(X, Y):
        n = min(len(X["pred"]), len(Y["pred"]))
        same = [bool(torch.equal(X["pred"][i], Y["pred"][i]) and torch.equal(X["score"][i], Y["score"][i])
                     and X["rle"][i] == Y["rle"][i]) for i in range(n)]
        return {"frames": n, "same_length": len(X["pred"]) == len(Y["pred"]), "equal_frames": int(sum(same)),
                "pred_equal": int(sum(bool(torch.equal(X["pred"][i], Y["pred"][i])) for i in range(n))),
                "score_equal": int(sum(bool(torch.equal(X["score"][i], Y["score"][i])) for i in range(n))),
                "rle_equal": int(sum(X["rle"][i] == Y["rle"][i] for i in range(n))),
                "first_diff": next((i for i in range(n) if not same[i]), None)}

    if "E1" in EQ_TESTS or "E4" in EQ_TESTS:
        _f0, _p0 = _first_frame(); _set_cur(_f0)
        E1_SEEDS_KEPT = detect(_p0, topk=N_CALVES + 2, kind="equiv")[:N_CALVES]      # one detect(), top-4, shared
        core.CACHE.clear()                  # review r1: detect() leaves every raw RF-DETR mask in the per-frame cache
        _seeds = [(k, m) for k, (m, _s) in enumerate(E1_SEEDS_KEPT)]
        if "E1" in EQ_TESTS:
            ARM_A = _e1_arm(False, _seeds)
        ARM_B = _e1_arm(True, _seeds, e4=("E4" in EQ_TESTS))
        if "E1" in EQ_TESTS:
            ARM_A2 = _e1_arm(False, _seeds)
            ab, aa = _cmp(ARM_A, ARM_B), _cmp(ARM_A, ARM_A2)
            ab_ok = ab["same_length"] and ab["equal_frames"] == ab["frames"] == E1_N
            aa_ok = aa["same_length"] and aa["equal_frames"] == aa["frames"] == E1_N
            # memory verdict, robust to a delayed free: slopes are medians of post-prune floor steps, each
            # arm starts and ends on a settled allocator, and the rebuild teardown drops every stale output reference
            def _mem_of(X):
                return {"base_mib": X["base_mib"], "end_mib": X["end_mib"], "prunes": X["prunes"],
                        "teardown": X["teardown"], "peak_mib": round(max(X["alloc_mib"]), 1),
                        "max_processed_frames": max(X["npf"]),
                        "floor_slope_mib_per_frame": [_floor_slope(X["alloc_mib"], 0, E1_REBUILD),
                                                      _floor_slope(X["alloc_mib"], E1_REBUILD, E1_N)]}
            mem = {k: _mem_of(X) for k, X in (("A", ARM_A), ("B", ARM_B), ("A2", ARM_A2))}
            _fa, _fb = mem["A"]["floor_slope_mib_per_frame"], mem["B"]["floor_slope_mib_per_frame"]
            mem["baseline_spread_mib"] = round(max(mem[k]["base_mib"] for k in ("A", "B", "A2"))
                                               - min(mem[k]["base_mib"] for k in ("A", "B", "A2")), 1)
            mem["pass"], mem["criteria"], mem["failures"] = core.e1_memory_verdict(mem, MEM_ASSERT_SESSION_FRAMES)
            r = {"seeds": len(_seeds), "A_vs_B": ab, "A_vs_A2": aa,
                 "sam3_frame_idx_equal": ARM_A["sam3_idx"] == ARM_B["sam3_idx"],
                 "alloc_slope_mib_per_frame": {k: [_slope(X["alloc_mib"][:E1_REBUILD]), _slope(X["alloc_mib"][E1_REBUILD:])]
                                               for k, X in (("A", ARM_A), ("B", ARM_B), ("A2", ARM_A2))},
                 "max_processed_frames": {"A": max(ARM_A["npf"]), "B": max(ARM_B["npf"])},
                 "memory": mem, "determinism_floor": None if aa_ok else aa}
            r["pass"] = bool(ab_ok and aa_ok and r["sam3_frame_idx_equal"] and mem["pass"])
            if not aa_ok: print("⚠⚠ E1: A != A' -- the determinism floor is reported; STOP FOR REVIEW", flush=True)
            if not mem["pass"]:
                print(f"⚠⚠ E1 memory verdict FAILED: floor slopes A {_fa} B {_fb}: {mem['failures']}", flush=True)
            EQ["results"]["E1"] = r; _eq_save(); print("E1", {k: v for k, v in r.items() if k != "A_vs_A2"}, flush=True)
            del ARM_A, ARM_A2
        del ARM_B; gc.collect()

    # ---- E2 ----
    if "E2" in EQ_TESTS:
        def _hashes(mode, n=E2_N):
            cap_ = cv2.VideoCapture(E1_DST); out = []
            it = core.iter_frames(cap_, range(0, E1_TOTAL, E1_STRIDE), mode=mode, pos_prop=cv2.CAP_PROP_POS_FRAMES)
            for fidx, ok, fr in it:
                out.append((fidx, hashlib.sha1(fr.tobytes()).hexdigest() if ok else None))
                if len(out) >= n: break
            it.close(); cap_.release()
            return out
        _t = time.time(); hs = _hashes("seek"); t_seek = time.time() - _t
        _t = time.time(); ht = _hashes("seek_thread"); t_thr = time.time() - _t
        _t = time.time(); hq = _hashes("sequential"); t_seq = time.time() - _t
        eq = sum(1 for a, b in zip(hs, ht) if a == b); eqq = sum(1 for a, b in zip(hs, hq) if a == b)
        r = {"frames": len(hs), "seek_eq_seek_thread": eq, "seek_eq_sequential_report_only": eqq,
             "s_per_1000": {"seek": round(1000 * t_seek / max(len(hs), 1), 2),
                            "seek_thread": round(1000 * t_thr / max(len(ht), 1), 2),
                            "sequential": round(1000 * t_seq / max(len(hq), 1), 2)}}
        r["pass"] = bool(len(hs) == len(ht) == E2_N and eq == E2_N)
        EQ["results"]["E2"] = r; _eq_save(); print("E2", r, flush=True)

    # ---- E5 (report only) ----
    if "E5" in EQ_TESTS:
        _e5_base = _mem_settle()            # a lingering E1 session must not inflate the memory-fit numbers
        torch.cuda.reset_peak_memory_stats(); lat = []; n_emb = 0; _e5_prunes = 0
        _f0, _p0 = _first_frame(); _set_cur(_f0)
        _kept0 = detect(_p0, topk=N_CALVES + 2, kind="equiv")[:N_CALVES]
        _sd = [(k, m) for k, (m, _s) in enumerate(_kept0)]
        sess = slots = None
        for i, (fidx, pil) in enumerate(_e1_frames(E5_N)):
            _set_cur(fidx)
            if i == 0: sess, slots = open_session_v4(pil, _sd)
            res = track_frame(sess, slots, pil)
            kept = detect(pil, topk=N_CALVES + 2, kind="equiv"); lat.append(DET_LOG[-1]["det_ms"])
            if "E4" in EQ_TESTS: _e4_check([m for m, _s in kept], list(res.values()), [])
            if i % 10 == 0:
                for m in res.values():
                    if m is not None and embed_robust(pil, m) is not None: n_emb += 1; break
            if (i + 1) % PRUNE_EVERY == 0:  # the tracker's prune cadence, so the peak is production's
                prune(sess, 8); _e5_prunes += 1
                if not EXACT_FAST: gc.collect(); torch.cuda.empty_cache()
            core.CACHE.clear()              # review r1: per frame, like the tracker -- detect() caches every raw RF-DETR
            #                                 mask (~3 MB); without E4 (whose check clears it) 300 frames kept GBs resident
        _pk_a, _pk_r = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
        res = kept = None
        del sess; gc.collect(); torch.cuda.empty_cache()
        lat_s = sorted(lat[3:]) or sorted(lat)
        coex = {"frames": E5_N, "rf_ms_median": lat_s[len(lat_s) // 2], "rf_ms_p95": lat_s[int(0.95 * (len(lat_s) - 1))],
                "peak_allocated_gib": round(_pk_a, 3), "peak_reserved_gib": round(_pk_r, 3),
                "baseline_gib": round(_e5_base / 1024, 3), "peak_above_baseline_gib": round(_pk_a - _e5_base / 1024, 3),
                "baseline_vs_pre_e1_mib": None if EQ_BASE_MIB is None else round(_e5_base - EQ_BASE_MIB, 1),
                "prunes": _e5_prunes, "pop_frames": POP_FRAMES, "embeds": n_emb}
        per = []
        try:
            _loc = _stage_small(E5_DETS, osp.join(LOCAL, "e5_dets_rfdetr.json.gz"))
            with gzip.open(_loc, "rt") as f: SD = json.load(f)
            for stem in E5_CLIPS:
                keys = sorted(k for k in SD["rows"] if k.startswith(stem + ":"))
                if not keys: continue
                vn = stem + ".mp4"
                dst = E1_DST if vn == E1_CLIP else osp.join(LOCAL, "e5.mp4")
                if dst != E1_DST: stage_clip(vn, dst)
                cap_ = cv2.VideoCapture(dst); nat = cap_.get(cv2.CAP_PROP_FPS) or 25.0; w0 = parse(vn)["start"]
                for k in keys:
                    fi = int(k.split(":")[1]); cap_.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap_.read()
                    if not ok: per.append({"key": k, "ok": False}); continue
                    pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    CUR.update(clip=vn, fidx=fi, t_sec=fi / nat, wall=w0 + timedelta(seconds=fi / nat))
                    kept = detect(pil, topk=N_CALVES + 2, kind="equiv")
                    stored = [rle_decode(x["rle"]) for x in SD["rows"][k]["kept"]]
                    if "E4" in EQ_TESTS: _e4_check([m for m, _s in kept], stored, [])
                    ious = []
                    if kept and stored:
                        M = np.array([[m_iou(a, b) for b in stored] for a, _s in kept])
                        ri, cj = linear_sum_assignment(-M); ious = [float(M[a, b]) for a, b in zip(ri, cj)]
                    per.append({"key": k, "ok": True, "phase4": core.phase4(CUR["wall"]), "n_kept": len(kept),
                                "n_stored": len(stored), "n_agree": len(kept) == len(stored),
                                "iou_mean": round(float(np.mean(ious)), 4) if ious else None,
                                "iou_min": round(float(np.min(ious)), 4) if ious else None})
                    core.CACHE.clear()
                cap_.release()
        except Exception as e:
            per.append({"error": repr(e)})
        okp = [p for p in per if p.get("ok")]
        dshift = {"calls": len(okp), "n_kept_agree": sum(p["n_agree"] for p in okp),
                  "night_calls": sum(1 for p in okp if p["phase4"] in ("night", "dusk")),
                  "iou_mean_median": (sorted(p["iou_mean"] for p in okp if p["iou_mean"] is not None) or [None])[
                      len([p for p in okp if p["iou_mean"] is not None]) // 2],
                  "per_call": per}
        EQ["results"]["E5"] = {"coexistence": coex, "domain_shift": dshift, "pass": None, "report_only": True}
        _eq_save(); print("E5", coex, {k: v for k, v in dshift.items() if k != "per_call"}, flush=True)

    if "E4" in EQ_TESTS:
        r = {k: {"equal": v[0], "n": v[1]} for k, v in E4_STATS.items() if k != "frames"}
        r["frames"] = E4_STATS["frames"]
        r["pass"] = bool(all(v[0] == v[1] and v[1] > 0 for k, v in E4_STATS.items() if k != "frames"))
        EQ["results"]["E4"] = r; _eq_save(); print("E4", r, flush=True)

    # ---- E6: Tol_03 reference root vs Tol_47 regression root (both run by Tol_47b in the v5 environment) ----
    if "E6" in EQ_TESTS:
        _ref = osp.join(TOL, _V["e6_ref_root"], STAGE + WSUF); _reg = osp.join(TOL, _V["e6_reg_root"], STAGE + WSUF)
        r = core.e6_compare_day(_ref, _reg); r["pass"] = bool(r["equal"])
        EQ["results"]["E6"] = r; _eq_save()
        print("E6", {k: v for k, v in r.items() if k not in ("masks",)}, flush=True)

    _req = [t for t in EQ_TESTS if t != "E5"]
    EQ["pass"] = bool(all(EQ["results"].get(t, {}).get("pass") for t in _req))
    EQ["wall_min"] = round((time.time() - EQ_T0) / 60, 1); _eq_save()
    print(f"\nEQUIV {'PASS' if EQ['pass'] else 'FAIL'} on {GPU_NAME}: " +
          ", ".join(f"{t} {EQ['results'].get(t, {}).get('pass')}" for t in EQ_TESTS) + f" -> {EQ_PATH}", flush=True)
    if not EQ["pass"]:
        raise RuntimeError(f"mode=equiv FAILED -- see {EQ_PATH}")
    if IS_NOTEBOOK:
        dbutils.notebook.exit(json.dumps({"equiv_pass": True, "metrics": EQ_PATH}))
    sys.exit(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main loop

# COMMAND ----------

t_run = time.time(); total_frames = 0
_RESUMED = []                                                             # [v5-C1] done clips restored this run
for ci, (vname, meta) in enumerate(CLIPS):
    stem = vname[:-4].replace(" ", "_")
    done = osp.join(TRACK_DIR, f".done_{stem}")
    if osp.exists(done) and not SMOKE:
        restore_after(stem, meta)
        if session is not None:
            del session; session = None; gc.collect(); torch.cuda.empty_cache()   # [v5-C7] one gc per teardown
        print(f"[{ci+1}/{len(CLIPS)}] {stem}: done — skip (state restored)")
        _RESUMED.append(stem)                                             # [v5-C1]
        continue
    if MAX_FRAMES and total_frames >= MAX_FRAMES:
        # GLOBAL cap: without this, max_frames limited frames PER CLIP and a "300-frame smoke"
        # marched through every clip of the day, staging ~1.3 GB each (burned a smoke run)
        print(f"MAX_FRAMES {MAX_FRAMES} reached across clips — stopping clip loop")
        break

    dst = osp.join(LOCAL, "cur.mp4")
    t_stage = stage_clip(vname, dst)
    cap = cv2.VideoCapture(dst)
    if not cap.isOpened(): raise RuntimeError(f"cv2.VideoCapture cannot open {dst} (GUI OpenCV shadowing?)")   # [v5-C2]
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(native/TRACK_FPS))
    wall0 = meta["start"]
    if total <= 0:
        print(f"  ⚠ {stem}: no frame count (corrupt?) — skipping, NOT marked done")
        log_event(type="clip_unreadable", clip=vname); cap.release(); os.remove(dst); continue

    rows = []; si = 0; t_clip = time.time()
    CLIP_PASS = {"passes": 0, "assignable": 0, "matched": 0, "accepted": [],
                 "strikes": 0, "opens": 0, "closes": 0, "merges": 0, "repairs": 0}
    for s in range(N_CALVES): repairs_clip[s] = 0
    for s in range(N_CALVES): dark_actions_clip[s] = 0; dark_suspended_logged[s] = False   # [v5-C11] per clip
    next_dark = 0.0                                                       # [v5-C12] clip-local
    next_embed, next_crop, next_seed_try = 0.0, 0.0, 0.0
    next_detect = 15.0                      # early pass validates any carried seeds
    pending_seed = None
    if session is not None:                 # sessions never span clips
        del session; session = None
        gc.collect(); torch.cuda.empty_cache()   # [v5-C7] once per session teardown: exact_fast dropped the prune-time
        #   gc, and a torn-down session can sit in a reference cycle until the next collect (full A100 profile 2026-09-24)
    # [v5-C14] per-clip counters only AFTER the previous clip's session is gone: reset_peak_memory_stats() sets the peak
    # to the CURRENT allocation, so resetting first would carry the old session into this clip's peak (and RSS start)
    _v5c_reset(vname)
    if _RESUMED and not V5C["resume_logged"]:                             # [v5-C1] new state starts EMPTY
        log_event(**core.ev_resume(vname, [s for s in range(N_CALVES) if state[s] != ST_CLOSED]))
        V5C["resume_logged"] = True; RESUME_INFO.update(resumed=True, restored_clips=list(_RESUMED))

    # [v5-C8] decode: seek_thread = the SAME cap.set/cap.read calls in the SAME order on one producer thread
    _frames = core.iter_frames(cap, range(0, total, stride), mode=DECODE, pos_prop=cv2.CAP_PROP_POS_FRAMES)
    _lap_reset()
    for fidx, ok, fr in _frames:                                          # [v5-C8]
        _lap("decode_wait")                                               # [v5-C14]
        if MAX_FRAMES and si >= MAX_FRAMES: break
        if not ok: continue
        pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        t_sec = fidx/native; wall = wall0 + timedelta(seconds=t_sec)
        is_day = tod_of_wall(wall) == "day"
        # [v5-C3/C7/C14] per-frame resets: one-detect guard, mask cache, detection-log context, CPU clock
        DET_CALLED = False; DET_N_FRAME = 0
        core.CACHE.clear()
        CUR.update(clip=vname, fidx=fidx, t_sec=t_sec, wall=wall)
        _cpu0 = time.process_time()
        _lap("decode")

        # ---- (A) session: carry fresh slots across the clip boundary (INV-1 exception,
        #      <= CLIP_CARRY_MAX, validated by the pass at +15 s) --------------------------------
        if session is None:
            carry = [(s, last_gen_mask[s]) for s in range(N_CALVES)
                     if state[s] in (ST_TRACK, ST_OCC) and last_gen_wall[s] is not None
                     and last_gen_mask[s] is not None
                     and (wall - last_gen_wall[s]).total_seconds() <= CLIP_CARRY_MAX]
            carried = {s for s, _ in carry}
            for s in range(N_CALVES):
                # A slot that cannot be carried has no session to self-recover in — keeping it
                # OCCLUDED would only block the seeding path from reopening its slot (bit us in
                # review: after a 34-min intra-day gap NOTHING could seed for 10 min). Close it.
                if state[s] in (ST_TRACK, ST_OCC) and s not in carried:
                    close_segment(s, last_gen_wall[s] or wall, "clip_boundary_stale")
            if carry:
                session, session_slots = open_session_v4(pil, carry)
                for s, _ in carry: state[s] = ST_TRACK
                log_event(type="session_carry", clip=vname, t=round(t_sec, 2),
                          slots=sorted(carried))

        # ---- (B) seeding when nothing is live --------------------------------------------------
        if session is None:
            if t_sec >= next_seed_try:
                dets = detect(pil, topk=N_CALVES + 2, kind="seed")               # [v5-C3/C14]
                next_dark = max(next_dark, t_sec + DARK_REPEAT_SEC)              # [v5-C12] one detect per frame
                _seed_refused = False
                if V5_FIXES and dets:                                            # [v5-C13] seeding guard
                    _dets0 = dets
                    dets, _seed_refused = core.seed_filter(dets, core.pool_of_wall(wall), BAND, area=_area_px)
                    if _seed_refused:
                        V5C["pen_count_refused"]["seed"] += 1
                        log_event(**core.ev_pen_count_refused("seed", DET_LOG[-1]["n_inband"],
                                                              [_area_px(m) for m, _s in _dets0],
                                                              [_s for _m, _s in _dets0], vname, t_sec))
                openable = []
                if not dets:
                    # detections flickered off: quick retry while a stability pending is live,
                    # else the full interval — without this the loop called detect() on EVERY
                    # frame for up to 25 s (review: night-seeding thrash)
                    next_seed_try = t_sec + (2.0 if pending_seed is not None else SEED_RETRY_SEC)
                elif is_day:
                    openable = dets[:N_CALVES]
                else:
                    if pending_seed is not None and (t_sec - pending_seed["t"]) <= 25:
                        stable = [(m, s) for m, s in dets
                                  if any(m_iou(m, pm) >= NIGHT_STABLE_IOU for pm in pending_seed["masks"])]
                        if stable: openable = stable[:N_CALVES]
                        else:
                            pending_seed = {"masks": [m for m, _ in dets], "t": t_sec}
                            next_seed_try = t_sec + 10.0
                    else:
                        pending_seed = {"masks": [m for m, _ in dets], "t": t_sec}
                        next_seed_try = t_sec + 10.0
                if openable:
                    if (DET_LOG[-1]["n_inband"] or 0) > N_CALVES: V5C["pen_guard_violations"] += 1   # [v5-C14] G2 audit
                    free = [s for s in range(N_CALVES) if state[s] == ST_CLOSED]
                    seeds = []
                    for (dm, _sc), slot in zip(openable, free):
                        open_segment(slot, wall, "seed_day" if is_day else "seed_night")
                        last_gen_wall[slot] = wall; last_gen_mask[slot] = dm
                        if DARK_FIRE: _note_healthy(slot, dm, wall)              # [v5-C11]
                        seeds.append((slot, dm)); CLIP_PASS["opens"] += 1
                    if seeds:
                        session, session_slots = open_session_v4(pil, seeds)
                        pending_seed = None
                        next_detect = t_sec + (DETECT_DAY_SEC if is_day else DETECT_NIGHT_SEC)
                        log_event(type="seeded", clip=vname, t=round(t_sec, 2),
                                  slots=[s for s, _ in seeds], day=is_day)
                if session is None and (pending_seed is None or t_sec > pending_seed["t"] + 25):
                    next_seed_try = t_sec + SEED_RETRY_SEC
                if _seed_refused: next_seed_try = t_sec + SEED_RETRY_SEC         # [v5-C13] refused: full retry
        _lap("session")                                                          # [v5-C14]

        # ---- track ----------------------------------------------------------------------------
        masks = track_frame(session, session_slots, pil) if session is not None \
                else {s: None for s in range(N_CALVES)}

        # ---- (C) per-slot state updates from the REAL mask stream ------------------------------
        for s in range(N_CALVES):
            if state[s] == ST_CLOSED:
                continue                      # outputs of closed slots are unattributed: ignored
            m = masks.get(s)
            # ---- area-collapse guard: a sliver is not a genuine observation ----
            # (baseline floored by the day reference: a step-then-stable shrink drags the
            #  rolling median down with it — 04-23 had the median at 0.31x day ref by 23:00,
            #  leaving the guard blind through a 2-hour partial plateau)
            if m is not None and len(area_hist[s]) >= AREA_HIST_MIN:
                a_now = int(m.sum())
                med = float(np.median(area_hist[s]))
                drf = day_ref_of(s) if PARTIAL_REPAIR else None
                med_eff = max(med, GUARD_FLOOR_FRAC * drf) if drf is not None else med
                if a_now < AREA_COLLAPSE_FRAC * med_eff:
                    collapse_run[s] += 1
                    if collapse_run[s] == AREA_COLLAPSE_N:
                        _ck = dict(type="mask_area_collapse", slot=s,
                                   seg_id=seg[s]["seg_id"] if seg[s] else None,
                                   clip=vname, t=round(t_sec, 2), area=a_now, median=int(med))
                        if drf is not None:
                            _ck["day_ref"] = int(drf); _ck["floor_active"] = bool(med_eff > med)
                        log_event(**_ck)
                    if collapse_run[s] >= AREA_COLLAPSE_N:
                        masks[s] = None; m = None       # honest absence -> OCCLUDED -> reopen
                else:
                    collapse_run[s] = 0
            if m is not None:
                a_now = int(m.sum())
                area_hist[s].append(a_now)
                if PARTIAL_REPAIR:
                    # centroid history (from the 4x-downsampled mask: ~1 ms) for the settled-scene gate
                    _sm = m[::4, ::4]; _ys, _xs = np.nonzero(_sm)
                    if len(_xs): cent_hist[s].append((wall, float(_xs.mean()) * 4, float(_ys.mean()) * 4))
                    # slow day-reference learning: daylight, not collapsing, and NOT in a partial
                    # or grown episode (review: without the gate the anchor learns the very
                    # fragments it exists to catch and silently disarms itself)
                    if is_day and collapse_run[s] == 0 and partial_run[s] == 0 \
                       and grown_run[s] == 0 and si % DAYREF_STEP == 0:
                        day_area_hist[s].append(a_now); DAYREF_VER[0] += 1
                    # partial-band / growth tracking: a sustained excursion arms a repair
                    drf = day_ref_of(s)
                    if drf is not None and a_now < PARTIAL_FRAC * drf:
                        partial_run[s] += 1; grown_run[s] = 0
                        if partial_run[s] == PARTIAL_N:
                            partial_armed[s] = True     # consumed by ONE early pass (section D)
                            log_event(type="partial_state", slot=s,
                                      seg_id=seg[s]["seg_id"] if seg[s] else None, clip=vname,
                                      t=round(t_sec, 2), area=a_now, day_ref=int(drf))
                    elif drf is not None and a_now > GROW_FRAC * drf:
                        grown_run[s] += 1; partial_run[s] = 0
                        if grown_run[s] == GROW_N:
                            grown_armed[s] = True
                            log_event(type="grown_state", slot=s,
                                      seg_id=seg[s]["seg_id"] if seg[s] else None, clip=vname,
                                      t=round(t_sec, 2), area=a_now, day_ref=int(drf))
                    else:
                        partial_run[s] = 0; grown_run[s] = 0
                if state[s] == ST_OCC:
                    absence = (wall - last_gen_wall[s]).total_seconds() if last_gen_wall[s] else 1e9
                    v = m_iou(last_gen_mask[s], m)
                    if absence >= SEG_BREAK_ABSENCE or \
                       (v < LOW_IOU_REAPPEAR and absence >= MIN_REAPPEAR_ABSENCE):
                        # SAM3's own reacquisition is an appearance guess -> segment boundary
                        close_segment(s, last_gen_wall[s] or wall,
                                      "absence_break" if absence >= SEG_BREAK_ABSENCE else "low_iou_reappear",
                                      drop_mask=False)
                        open_segment(s, wall, "self_recovery")
                    else:
                        state[s] = ST_TRACK; occ_logged[s] = False
                if seg[s] is not None: seg[s]["n_frames"] += 1
                last_gen_wall[s] = wall; last_gen_mask[s] = m
                hist[s].append((wall, m[::4, ::4].copy()))
                if DARK_FIRE: _note_healthy(s, m, wall, a=a_now)                  # [v5-C11] healthy mask
            else:
                if state[s] == ST_TRACK and last_gen_wall[s] is not None and \
                   (wall - last_gen_wall[s]).total_seconds() > FRESH_SEC:
                    state[s] = ST_OCC
                    if DARK_FIRE:                    # [v5-C11] dark onset: cause collapse | no_mask
                        dark_onset[s] = wall; dark_onset_pending[s] = True
                        dark_seg[s] = seg[s]["seg_id"] if seg[s] else None
                        dark_cause[s] = "collapse" if collapse_run[s] >= AREA_COLLAPSE_N else "no_mask"
                    note_scene_event(wall)           # >= 2 s dark anywhere = the pen is not settled
                    if not occ_logged[s]:
                        log_event(type="occluded", slot=s, clip=vname, t=round(t_sec, 2),
                                  seg_id=seg[s]["seg_id"] if seg[s] else None)
                        occ_logged[s] = True
                elif state[s] == ST_OCC and last_gen_wall[s] is not None and \
                     (wall - last_gen_wall[s]).total_seconds() > OCCLUDED_TIMEOUT:
                    close_segment(s, wall, "occluded_timeout"); CLIP_PASS["closes"] += 1

        # ---- (D) merge detection over genuine TRACKING masks ----------------------------------
        forced_pass = None
        gm = {s: masks.get(s) for s in range(N_CALVES)
              if state[s] == ST_TRACK and masks.get(s) is not None}
        gl = sorted(gm)
        for ai in range(len(gl)):
            for aj in range(ai+1, len(gl)):
                i, j = gl[ai], gl[aj]
                if m_iou(gm[i], gm[j]) > MERGE_IOU or m_ios(gm[i], gm[j]) > MERGE_IOS:
                    if merge_run[(i, j)] == 0: merge_onset[(i, j)] = wall
                    merge_run[(i, j)] += 1
                    if DARK_FIRE: last_merge_touch[i] = last_merge_touch[j] = wall   # [v5-C11]
                    if merge_run[(i, j)] >= MERGE_N:
                        CLIP_PASS["merges"] += 1
                        resolve_merge(i, j, wall)
                        forced_pass = f"merge_{i}_{j}"
                else:
                    merge_run[(i, j)] = 0; merge_onset.pop((i, j), None)
        # a partial episode crossing its arming threshold earns one early detector pass
        # (repair retries thereafter ride the normal 120/300 s schedule)
        if PARTIAL_REPAIR and (any(partial_armed) or any(grown_armed)):
            _any_grown = any(grown_armed)
            for s in range(N_CALVES): partial_armed[s] = False; grown_armed[s] = False
            if forced_pass is None and not (REPAIR_NIGHT_ONLY and is_day):
                forced_pass = "growth_band" if _any_grown else "partial_band"
        _lap("guards")                                                   # [v5-C14]

        # ---- (E) observations + verification crops (anonymous, gate fields attached) ----------
        if t_sec >= next_embed:
            next_embed = t_sec + EMBED_EVERY_SEC
            cur = {s: m for s, m in gm.items() if state[s] == ST_TRACK}   # post-merge states
            for s, m in cur.items():
                if seg[s] is None: continue
                nb = max([m_ios(m, m2) for s2, m2 in cur.items() if s2 != s], default=0.0)
                e = embed_robust(pil, m)
                if e is None: continue
                obs.append({"clip": vname, "t_sec": round(t_sec, 2), "slot": s,
                            "seg_id": seg[s]["seg_id"], "wall": wall.isoformat(),
                            "tod": tod_of_wall(wall), "area": int(m.sum()),
                            "max_neighbor_ios": round(nb, 4),
                            "emb": e.numpy().astype(np.float16)})
        _lap("embed")                                                    # [v5-C14]
        if t_sec >= next_crop:
            next_crop = t_sec + CROP_EVERY_SEC
            for s, m in gm.items():
                if state[s] != ST_TRACK or seg[s] is None: continue
                cc = canonical_crop(pil, m)
                if cc is None: continue
                cd = osp.join(CROP_DIR, seg[s]["seg_id"]); os.makedirs(cd, exist_ok=True)
                cc.save(osp.join(cd, f"{wall:%H%M%S}.jpg"), quality=88)

        _lap("crops")                                                    # [v5-C14]

        # ---- (F) detector pass ----------------------------------------------------------------
        # [v5-C12] a scheduled pass (unchanged cadence / forced) and, with dark_fire, a dark-only pass every
        # DARK_REPEAT_SEC while some dark slot is actionable; at most one detect() per frame from v5 paths.
        sched_due = bool(forced_pass) or t_sec >= next_detect
        dark_due = (DARK_FIRE and session is not None and not DET_CALLED and t_sec >= next_dark
                    and _any_actionable(wall))
        if session is not None and (sched_due or dark_due):
            if sched_due: next_detect = t_sec + (DETECT_DAY_SEC if is_day else DETECT_NIGHT_SEC)
            ran_dark = detector_pass(pil, wall, vname, t_sec, forced_by=forced_pass, scheduled=sched_due)
            if DARK_FIRE and ran_dark: next_dark = t_sec + DARK_REPEAT_SEC
        for s_r, dm_r in ROW_OVERRIDE.items(): masks[s_r] = dm_r     # [v5-C12] generalises LAST_REPAIRS
        ROW_OVERRIDE.clear(); LAST_REPAIRS.clear()
        _lap("detector")                                                 # [v5-C14]

        # all slots dead -> drop the session so seeding re-engages
        if session is not None and all(state[s] == ST_CLOSED for s in range(N_CALVES)):
            del session; session = None
            gc.collect(); torch.cuda.empty_cache()
            next_seed_try = t_sec + 5.0
            log_event(type="all_closed", clip=vname, t=round(t_sec, 2))
        if DARK_FIRE: _flush_dark_onsets(wall, vname, t_sec)                  # [v5-C11]
        V5C["max_det_calls_per_frame"] = max(V5C["max_det_calls_per_frame"], DET_N_FRAME)   # [v5-C14]
        _lap("other")

        # ---- (G) record the row from post-update state (INV-3: absence is honest) -------------
        rmask, n_gen = {}, 0
        for s in range(N_CALVES):
            m = masks.get(s)
            if state[s] == ST_TRACK and m is not None:
                rmask[str(s)] = rle_encode(m); n_gen += 1
            else:
                rmask[str(s)] = None
        rows.append({"frame_idx": fidx, "t_sec": round(t_sec, 2), "masks": rmask,
                     "_n_open": sum(1 for s in range(N_CALVES) if state[s] != ST_CLOSED),
                     "_n_gen": n_gen})
        si += 1
        _lap("rle")                                                      # [v5-C14]
        if session is not None and si % PRUNE_EVERY == 0:
            prune(session, 8)
            if not EXACT_FAST: gc.collect(); torch.cuda.empty_cache()     # [v5-C7] dropped with exact_fast (12 ms/frame)
        _lap("prune")
        if si % core.MEM_CHECK_EVERY == 0: _mem_check(si, vname)          # [v5-C6] in-run memory tripwires
        V5C["cpu_s"] += time.process_time() - _cpu0                        # [v5-C14]
        core.CACHE.clear()                                                # [v5-C7] per-frame cache
        _lap("other")
        if si % 2000 == 0:
            print(f"  [{ci+1}/{len(CLIPS)}] {si} frames  {si/(time.time()-t_clip):.2f} fps  "
                  f"states={[state[s][0] for s in range(N_CALVES)]}")

    _frames.close()                         # [v5-C8] stop + drain + JOIN the producer before cap.release()
    cap.release()
    _mem_check(si, vname, where="clip_end")                               # [v5-C6]
    out_rows = [{"frame_idx": r["frame_idx"], "t_sec": r["t_sec"], "masks": r["masks"]}
                for r in rows]
    with gzip.open(osp.join(TRACK_DIR, f"masks_{stem}.json.gz"), "wt") as f:
        json.dump({"clip": vname, "track_fps": TRACK_FPS, "mask_format": "rle_column_major",
                   "keyed_by": "slot (anonymous; identity is offline)",
                   "backbone": BACKBONE, "sam3_model": SAM3_ID,
                   "frames": out_rows, "smoke": SMOKE, "v5": ECHO}, f)          # [v5-C14] full config echo
    write_segments(); save_obs(); sync_events()
    _write_detections(stem)                 # [v5-C14] detections_<clip>.json.gz BEFORE the .done_ marker
    total_frames += len(rows)
    os.remove(dst)
    cov = {s: sum(1 for r in rows if r["masks"].get(str(s))) for s in range(N_CALVES)}
    print(f"[{ci+1}/{len(CLIPS)}] {stem}: {len(rows)} frames, "
          f"genuine={{{', '.join(f'{s}:{100*c/max(len(rows),1):.0f}%' for s, c in cov.items())}}}, "
          f"track {(time.time()-t_clip)/60:.1f} min + stage {t_stage/60:.1f} min")
    # INV-7 checks run BEFORE the done marker: an aborted clip must re-run after a fix,
    # not be skipped as complete.
    finish_clip_telemetry(stem, rows, time.time()-t_clip, t_stage)
    if not SMOKE: open(done, "w").close()

# COMMAND ----------

for s in range(N_CALVES):
    if seg[s] is not None:
        close_segment(s, last_gen_wall[s] or CLIPS[-1][1]["end"], "day_end")
write_segments(); save_obs(); sync_events()
n_seg = len(segments_done)
per_slot = {s: sum(1 for r in segments_done if r["slot"] == s) for s in range(N_CALVES)}
print(f"\nstage {STAGE} complete | {total_frames} frames | {n_seg} segments {per_slot} | "
      f"{len(obs)} observations | {(time.time()-t_run)/3600:.2f} h")
if max(per_slot.values() or [0]) > 30:
    print("⚠ >30 segments on a slot — fragmentation; check events for the cause")
