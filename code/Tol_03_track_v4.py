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

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "opencv-python-headless==4.11.0.86" "Pillow==11.1.0" "timm" "huggingface_hub" "accelerate" "scipy"

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass   # subprocess-worker mode (multiplex launcher): plain python, deps inherited via sys.executable

# COMMAND ----------

import os, os.path as osp, re, json, gzip, time, gc, glob, shutil
from collections import defaultdict, deque
from datetime import datetime, timedelta
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

def P(name, default):
    if IS_NOTEBOOK:
        try: dbutils.widgets.text(name, default)
        except Exception: pass
        return (dbutils.widgets.get(name) or default)
    return os.environ.get("TOLV4_" + name.upper(), default)

STAGE      = P("stage", "20240422").strip()
MAX_CLIPS  = int(P("max_clips", "0") or 0)
MAX_FRAMES = int(P("max_frames", "0") or 0)
FROM_HHMM  = (P("from_hhmm", "") or "").strip()
TO_HHMM    = (P("to_hhmm", "") or "").strip()
TRACK_ROOT = (P("track_root", "Tracking_v4") or "Tracking_v4").strip()
TRACK_FPS  = float(P("track_fps", "5.0") or 5.0)
ALLOW_NONSTD_FPS = (P("allow_nonstandard_fps", "false") or "false").strip().lower() == "true"
DETECT_DAY_SEC   = float(P("detect_day_sec", "120") or 120)
DETECT_NIGHT_SEC = float(P("detect_night_sec", "300") or 300)
REFRESH_MIN_SEC  = float(P("refresh_min_sec", "600") or 600)
EMBED_EVERY_SEC  = float(P("embed_every_sec", "10") or 10)
CROP_EVERY_SEC   = float(P("crop_every_sec", "60") or 60)
STAGE_SVC  = os.environ.get("TOLV4_STAGE_SVC", "")     # set by the launcher: staging-service dir
# backbone=teacher runs SAM3's ORIGINAL vision backbone (no student swap) — used to produce the
# visually-verified ground-truth track of 04-22 against which every distilled student is scored.
BACKBONE   = (P("backbone", "student") or "student").strip().lower()
STUDENT_CKPT = (P("student_ckpt", "best_backbone_student.pt") or "best_backbone_student.pt").strip()
SAM3_ID    = (P("sam3_model", "facebook/sam3") or "facebook/sam3").strip()
# partial_repair=false runs the pre-2026-09-02 tracking behavior identically (masks, segments,
# state transitions; events/telemetry carry only the flag itself). Teacher GT runs MUST use
# false so their provenance matches the frozen 04-22 GT; student production runs use true.
PARTIAL_REPAIR = (P("partial_repair", "true") or "true").strip().lower() == "true"
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
device = torch.device("cuda")
DTYPE = torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16
print(f"GPU {torch.cuda.get_device_name(0)} -> tracking dtype {DTYPE}")

VIDEO_DIR = "/dbfs/mnt/playbehavior/Tol/Videos"
DBFS_VIDEO = "dbfs:/mnt/playbehavior/Tol/Videos"
VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
WSUF = f"_w{FROM_HHMM or '0000'}-{TO_HHMM or '2400'}" if (FROM_HHMM or TO_HHMM) else ""
TRACK_DIR = osp.join(TOL, TRACK_ROOT, STAGE + WSUF)
CROP_DIR  = osp.join(TRACK_DIR, "crops")
# Per-stage local scratch: multiple day-tasks share one node under the consolidated plan, and a
# shared dir would have them overwriting each other's staged clip and obs.npz mid-read.
LOCAL = f"/local_disk0/tmp/tol_v4_{STAGE}{WSUF}"
for d in (TRACK_DIR, CROP_DIR, LOCAL): os.makedirs(d, exist_ok=True)

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

if BACKBONE == "student":
    ck = torch.load(osp.join(TOL, "Checkpoints", STUDENT_CKPT), map_location='cpu')
    student = StudentEncoderTinyViT()
    sd = student.state_dict(); sd.update(ck['model_state_dict'])
    for k, v in ck['ema_shadow'].items(): sd[k] = v
    student.load_state_dict(sd); student = student.to(device).eval()
    _t = timm.create_model('tiny_vit_21m_512', pretrained=False)
    _cfg = resolve_data_config(_t.pretrained_cfg); del _t
    STUDENT_TF = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                     transforms.Normalize(mean=_cfg['mean'], std=_cfg['std'])])
    print(f"Tol student loaded: {STUDENT_CKPT} (epoch {ck['epoch']+1}, cosine {ck.get('cosine', float('nan')):.4f})")
else:
    student, STUDENT_TF = None, None
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
    sam3.vision_encoder.backbone.cpu(); sam3.vision_encoder.backbone = dummy
    gc.collect(); torch.cuda.empty_cache()

dproc = Sam3Processor.from_pretrained(SAM3_ID)
det = Sam3Model.from_pretrained(SAM3_ID, torch_dtype=DTYPE).to(device).eval()
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

def open_segment(slot, wall, reason):
    seg_counter[slot] += 1
    sid = f"{slot}_{seg_counter[slot]:03d}"
    seg[slot] = {"seg_id": sid, "slot": slot, "t_start_wall": wall.isoformat(),
                 "open_reason": reason, "n_frames": 0}
    state[slot] = ST_TRACK
    strikes[slot] = 0; occ_logged[slot] = False
    area_hist[slot].clear(); collapse_run[slot] = 0
    partial_run[slot] = 0; partial_armed[slot] = False
    grown_run[slot] = 0; grown_armed[slot] = False; cent_hist[slot].clear()
    if not reason.endswith("_repair"): note_scene_event(wall)   # a repair boundary is not chaos
    log_event(type="segment_open", seg_id=sid, slot=slot, wall=wall.isoformat(), reason=reason)
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
    return sess, slots

def track_frame(sess, slots, pil):
    inputs = tproc(images=pil, device=device, return_tensors="pt")
    with torch.no_grad():
        if BACKBONE == "student":
            feat = student(STUDENT_TF(pil).unsqueeze(0).to(device))
            if DTYPE == torch.float16:
                mx = feat.abs().max().item()
                if not np.isfinite(mx) or mx > 60000:
                    raise RuntimeError(f"student feature magnitude {mx:.1f} unsafe for fp16")
            dummy.student_features = feat
        out = sam3(inference_session=sess, frame=inputs.pixel_values[0])
        if BACKBONE == "student":
            dummy.student_features = None
        if not torch.isfinite(out.pred_masks).all():
            raise RuntimeError("non-finite SAM3 mask logits — dtype unsafe on this GPU")
    mm = tproc.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes,
                                  binarize=False)[0]
    res = {}
    for k, slot in enumerate(slots):
        ml = mm[k, -1] if mm.dim() == 4 else mm[k]
        b = (torch.sigmoid(ml) > 0.5).cpu().numpy().astype(bool)
        res[slot] = b if b.sum() >= 200 else None
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

def detector_pass(pil, wall, clip, t_sec, forced_by=None):
    """One measurement + correction cycle. Spatial only (INV-2/5). Returns True if session rebuilt."""
    global session, session_slots, pending_open, last_refresh_wall
    is_day = tod_of_wall(wall) == "day"
    dets = detect(pil, topk=N_CALVES + 2)
    CLIP_PASS["passes"] += 1
    if not dets:
        log_event(type="pass_no_detections", clip=clip, t=round(t_sec, 2), forced=forced_by)
        return False
    trk = [s for s in range(N_CALVES) if state[s] == ST_TRACK and last_gen_mask[s] is not None]
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
            if strikes[s] >= (STRIKES_TO_CLOSE_DAY if is_day else STRIKES_TO_CLOSE_NIGHT):
                close_segment(s, wall, "detector_contradiction")
                closed_now.append(s); CLIP_PASS["closes"] += 1
        # 0.10..0.30: ambiguous — neither match nor strike; counter unchanged
    # ---- unmatched detections -> reopen / displacement (with night stability) ----------------
    un = [dets[b][0] for b in range(len(dets)) if b not in matched_det]
    survivors = []
    for p in pending_open:
        if (wall - p["wall"]).total_seconds() <= 2.5 * (DETECT_DAY_SEC if is_day else DETECT_NIGHT_SEC):
            survivors.append(p)
    pending_open = survivors
    opens = []
    for dm in un:
        hit = None
        for p in pending_open:
            if m_iou(p["mask"], dm) >= NIGHT_STABLE_IOU or m_ios(p["mask"], dm) >= NIGHT_STABLE_IOU:
                hit = p; break
        persisted = hit is not None
        if hit: hit["mask"], hit["wall"], hit["seen"] = dm, wall, hit["seen"] + 1
        else: pending_open.append({"mask": dm, "wall": wall, "seen": 1})
        free = [s for s in range(N_CALVES) if state[s] == ST_CLOSED and s not in [o[0] for o in opens]]
        if is_day and free:
            opens.append((free[0], dm, "reopen"))
        elif (not is_day) and persisted and free:
            opens.append((free[0], dm, "reopen"))
        elif persisted and not free:
            occ = [(s, (wall - last_gen_wall[s]).total_seconds())
                   for s in range(N_CALVES) if state[s] == ST_OCC and last_gen_wall[s] is not None
                   and s not in [o[0] for o in opens]]
            if occ:
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
                          note="pen holds exactly 4; all slots tracking")
    pending_open = [p for p in pending_open
                    if not any(m_iou(p["mask"], o[1]) >= NIGHT_STABLE_IOU for o in opens)]
    # ---- optional conservative refresh (day only, overlap-gated, rate-limited) ---------------
    refreshed = {}
    if (REFRESH_MIN_SEC > 0 and is_day and matched
            and (last_refresh_wall is None or (wall - last_refresh_wall).total_seconds() >= REFRESH_MIN_SEC)
            and not any(state[s] == ST_OCC and last_gen_wall[s] is not None
                        and (wall - last_gen_wall[s]).total_seconds() < SEG_BREAK_ABSENCE
                        for s in range(N_CALVES))):
        for s, (dm, sc) in matched.items():
            if sc >= REFRESH_IOU: refreshed[s] = dm
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
    if PARTIAL_REPAIR:
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
            open_segment(slot, wall, reason)
            last_gen_wall[slot] = wall; last_gen_mask[slot] = dm
            seeds.append((slot, dm)); CLIP_PASS["opens"] += 1
        if seeds:
            if session is not None: del session
            gc.collect(); torch.cuda.empty_cache()
            session, session_slots = open_session_v4(pil, seeds)
            rebuilt = True
            merge_run.clear(); merge_onset.clear()
    _pk = dict(type="pass", clip=clip, t=round(t_sec, 2), forced=forced_by, n_det=len(dets),
               tracked=len(trk), matched={s: round(v[1], 3) for s, v in matched.items()},
               struck=closed_now, opened=[o[0] for o in opens],
               refreshed=list(refreshed.keys()), rebuilt=rebuilt)
    if PARTIAL_REPAIR: _pk["repaired"] = list(repaired.keys())
    log_event(**_pk)
    return rebuilt

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
        req = osp.join(STAGE_SVC, f"req__{STAGE}__{vname}")
        rdy = osp.join(STAGE_SVC, f"ready__{STAGE}__{vname}")
        err = osp.join(STAGE_SVC, f"error__{STAGE}__{vname}")
        tmp = osp.join(STAGE_SVC, f"tmp__{STAGE}__{vname}")   # outside the launcher's req__* glob
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
# track_root in the name: a teacher run and a student run of the SAME day must not share a
# telemetry file (the teacher GT run silently overwrote the student day's summary once)
TELEM_PATH = osp.join(TOL, f"Metrics_track_v4_{TRACK_ROOT}_{STAGE}{WSUF}.json")
if osp.exists(TELEM_PATH):
    # resume: keep completed clips' telemetry entries — they are the day's acceptance evidence
    try:
        _old = json.load(open(TELEM_PATH))
        _kept = [c for c in _old.get("clips", [])
                 if osp.exists(osp.join(TRACK_DIR, f".done_{c['clip']}"))]
        if _kept:
            _op = _old.get("params", {}).get("partial_repair")
            if _op is not None and bool(_op) != PARTIAL_REPAIR:
                raise RuntimeError(f"resume with partial_repair={PARTIAL_REPAIR} but {len(_kept)} done "
                                   f"clips were tracked with {_op} -- a day must not mix behaviors")
            TELEMETRY["clips"] = _kept
            print(f"resume: preloaded {len(_kept)} clip telemetry entries")
    except Exception as _e:
        print(f"⚠ telemetry preload failed ({_e})")

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
    TELEMETRY["clips"].append(entry)
    with open(TELEM_PATH, "w") as f: json.dump(TELEMETRY, f, indent=1)
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
# MAGIC ## Main loop

# COMMAND ----------

t_run = time.time(); total_frames = 0
for ci, (vname, meta) in enumerate(CLIPS):
    stem = vname[:-4].replace(" ", "_")
    done = osp.join(TRACK_DIR, f".done_{stem}")
    if osp.exists(done) and not SMOKE:
        restore_after(stem, meta)
        if session is not None: del session; session = None
        print(f"[{ci+1}/{len(CLIPS)}] {stem}: done — skip (state restored)")
        continue
    if MAX_FRAMES and total_frames >= MAX_FRAMES:
        # GLOBAL cap: without this, max_frames limited frames PER CLIP and a "300-frame smoke"
        # marched through every clip of the day, staging ~1.3 GB each (burned a smoke run)
        print(f"MAX_FRAMES {MAX_FRAMES} reached across clips — stopping clip loop")
        break

    dst = osp.join(LOCAL, "cur.mp4")
    t_stage = stage_clip(vname, dst)
    cap = cv2.VideoCapture(dst)
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
    next_embed, next_crop, next_seed_try = 0.0, 0.0, 0.0
    next_detect = 15.0                      # early pass validates any carried seeds
    pending_seed = None
    if session is not None: del session; session = None   # sessions never span clips

    for fidx in range(0, total, stride):
        if MAX_FRAMES and si >= MAX_FRAMES: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, fr = cap.read()
        if not ok: continue
        pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        t_sec = fidx/native; wall = wall0 + timedelta(seconds=t_sec)
        is_day = tod_of_wall(wall) == "day"

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
                dets = detect(pil, topk=N_CALVES + 2)
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
                    free = [s for s in range(N_CALVES) if state[s] == ST_CLOSED]
                    seeds = []
                    for (dm, _sc), slot in zip(openable, free):
                        open_segment(slot, wall, "seed_day" if is_day else "seed_night")
                        last_gen_wall[slot] = wall; last_gen_mask[slot] = dm
                        seeds.append((slot, dm)); CLIP_PASS["opens"] += 1
                    if seeds:
                        session, session_slots = open_session_v4(pil, seeds)
                        pending_seed = None
                        next_detect = t_sec + (DETECT_DAY_SEC if is_day else DETECT_NIGHT_SEC)
                        log_event(type="seeded", clip=vname, t=round(t_sec, 2),
                                  slots=[s for s, _ in seeds], day=is_day)
                if session is None and (pending_seed is None or t_sec > pending_seed["t"] + 25):
                    next_seed_try = t_sec + SEED_RETRY_SEC

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
            else:
                if state[s] == ST_TRACK and last_gen_wall[s] is not None and \
                   (wall - last_gen_wall[s]).total_seconds() > FRESH_SEC:
                    state[s] = ST_OCC
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
        if t_sec >= next_crop:
            next_crop = t_sec + CROP_EVERY_SEC
            for s, m in gm.items():
                if state[s] != ST_TRACK or seg[s] is None: continue
                cc = canonical_crop(pil, m)
                if cc is None: continue
                cd = osp.join(CROP_DIR, seg[s]["seg_id"]); os.makedirs(cd, exist_ok=True)
                cc.save(osp.join(cd, f"{wall:%H%M%S}.jpg"), quality=88)

        # ---- (F) detector pass ----------------------------------------------------------------
        if session is not None and (forced_pass or t_sec >= next_detect):
            next_detect = t_sec + (DETECT_DAY_SEC if is_day else DETECT_NIGHT_SEC)
            detector_pass(pil, wall, vname, t_sec, forced_by=forced_pass)
            if LAST_REPAIRS:
                for s_r, dm_r in LAST_REPAIRS.items(): masks[s_r] = dm_r
                LAST_REPAIRS.clear()

        # all slots dead -> drop the session so seeding re-engages
        if session is not None and all(state[s] == ST_CLOSED for s in range(N_CALVES)):
            del session; session = None
            gc.collect(); torch.cuda.empty_cache()
            next_seed_try = t_sec + 5.0
            log_event(type="all_closed", clip=vname, t=round(t_sec, 2))

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
        if session is not None and si % PRUNE_EVERY == 0:
            prune(session, 8); gc.collect(); torch.cuda.empty_cache()
        if si % 2000 == 0:
            print(f"  [{ci+1}/{len(CLIPS)}] {si} frames  {si/(time.time()-t_clip):.2f} fps  "
                  f"states={[state[s][0] for s in range(N_CALVES)]}")

    cap.release()
    out_rows = [{"frame_idx": r["frame_idx"], "t_sec": r["t_sec"], "masks": r["masks"]}
                for r in rows]
    with gzip.open(osp.join(TRACK_DIR, f"masks_{stem}.json.gz"), "wt") as f:
        json.dump({"clip": vname, "track_fps": TRACK_FPS, "mask_format": "rle_column_major",
                   "keyed_by": "slot (anonymous; identity is offline)",
                   "backbone": BACKBONE, "sam3_model": SAM3_ID,
                   "frames": out_rows, "smoke": SMOKE}, f)
    write_segments(); save_obs(); sync_events()
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
