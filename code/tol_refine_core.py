#!/usr/bin/env python
"""tol_refine_core -- every DECISION of the offline mask refinement (design v1, 2026-10-02).

Plain module: numpy, scipy and the standard library (cv2 is used only for a faster but IDENTICAL dilation when it is
importable). `Tol_55_mask_refine` (Databricks driver) and `Tol_56_mask_refine_eval` (local evaluator) import it, so the
cluster, the evaluator and the local SELFTEST run the identical gate code. Deployed to the Workspace as a FILE beside
Tol_55 (`--format RAW`), exactly like tol_v5_core.py, which it imports for the production functions it must not copy
(phase4 / pool_of_wall / band / in_band / post_chain_idx / rle_encode / rle_fast / iter_frames / write_json_if_absent).

Scope: only the RLE of (slot, frame) rows coded d / c / g changes. Frames, slot keys, segments, events and
identity are never touched; refinement never reads identity (I7).

Contents
  §A  constants (READ from train_summary, DERIVED file schema, DESIGN), codes and reasons
  §B  exact run-length masks: Tol_48 runs_of / inter / iou_r / ios_r (verbatim) + exact set operations, RLE <-> runs
  §C  I/O: json.gz, FrameStream (verbatim Tol_48), SegIndex (verbatim Tol_48 semantics), deterministic npz, frozen files
  §D  READ constants + the frozen refine_constants file (load / validate)
  §E  clips, walls, phases; trigger; census (S0); sampling
  §F  (A) gates + per-frame pixel resolution; double-claims and orphans
  §G  apply_clip: first pass, carry, output frames + sidecar + asserts
  §H  detect stage helpers: call plan, DET_LOG record (production schema), E0 parity comparison
  §I  colour model: HSV saturation, LUT, chroma, masked-median backgrounds, STATIC / DYN, shift check
  §J  (B) grow + grow_clip
  §K  derive statistics: trusted rows, matching, synthetic fragments, boundary offset, carry window, shape, ring
      threshold, ship gate
  §L  selftest()

CLI:  python tol_refine_core.py --selftest
"""
from __future__ import annotations

import os, sys, io, re, json, gzip, math, time, bisect, hashlib, zipfile, itertools
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import tol_v5_core as V5                                    # noqa: E402  production decisions (never copied)

try:                                                         # cluster: faster dilation, identical result (§I dilate)
    import cv2 as _cv2                                       # noqa
except Exception:                                            # local / no OpenCV: scipy path
    _cv2 = None

REFINE_CORE_VERSION = "1.0.1 2026-10-02 (adversarial review fixes)"
REFINE_SPEC = "SPEC_mask_refine.md v1 (2026-10-02)"

# =====================================================================================================================
# §A  constants
# =====================================================================================================================
H, W = 1296, 2304
N_PIX = H * W
N_SLOTS = 4
SLOT_KEYS = tuple(str(s) for s in range(N_SLOTS))
TRAIN_DAYS = ("20240418", "20240419")
TEST_DAYS = ("20240422", "20240423", "20240424", "20240425", "20240426", "20240427", "20240428")
_TEST_RX = re.compile(r"2024042[2-8]")
DET_RUN_TAG = "segv1d_tracked_seg-small_r624"
DET_TOPK = V5.DET_TOPK                                       # 6 = production detect(topk=N_CALVES + 2)

CODES = ("k", "r", "d", "c", "g")
REASONS = ("nodet", "straddle", "ambiguous", "small", "big", "conflict", "pair_conflict", "post_shrink", "shrink", "carry_fail",
           "unsampled", "decode_fail", "grow_ineligible_phase", "grow_ineligible_chroma", "grow_short", "grow_leak",
           "grow_shape")
# refusals carry may bridge (transient evidence). unsampled / decode_fail = the row had no call at all (1 Hz +
# carry). The detector's current evidence and the identity conflict are NEVER overridden by carry:
CARRY_ELIGIBLE = ("nodet", "straddle", "post_shrink", "unsampled", "decode_fail")
NEVER_CARRY = ("small", "big", "conflict", "ambiguous", "pair_conflict", "shrink")
# NO_SHRINK (addition to the v1 design, needs explicit sign-off, widget no_shrink): a refined mask must be LARGER
# than the original. The 2026-10-02 dry run on real 04-22 rows found every harmful replacement to be a near-healthy mask
# (0.65-0.68 x A_ref, IoU 0.84-0.92 with the teacher) swapped for a SMALLER in-band detection -- a shrink, not a fragment
# repair. Off reproduces the v1 design literally. It has no numeric constant, but it was CHOSEN after looking at 04-22 (an
# evaluation day): the note below is written into every refine block / REFINE_CONFIG so the report can say so.
NO_SHRINK_NOTE = ("no_shrink is a parameter-free rule added after the 2026-10-02 dry run on 04-22 (evaluation day): "
                  "04-22 is not held-out for this rule; 04-23 is")

# DESIGN (pre-declared). Nothing here is fitted on any day.
DESIGN = {
    "CARRY_Q": 0.85,                # carry window quality floor
    "CARRY_GRID": [round(0.2 * k, 1) for k in range(1, 11)],   # Δ ∈ {0.2, ..., 2.0} s
    "RING_FP_MAX": 0.05,            # ring false-positive ceiling for the LUT threshold
    "CLOSE_R": 5, "STRUCT_DIL": 3, "BG_EXCL_DIL": 15, "N_BG_FRAMES": 60, "N_BG_MIN": 10,
    "DOUBLECLAIM_MIN_S": 5.0,       # doubleclaim interval length
    "DOUBLECLAIM_GAP_S": 1.0,       # rows of one interval are <= 1 s apart (dense calls are 0.2 s apart)
    "LUT_BITS": 4,                  # 16 x 16 x 16 RGB bins
    "SYN_FRACS": [0.2, 0.3, 0.4, 0.5], "SYN_LATERAL": 0.4,     # 8 perpendicular + 2 lateral = 10 fragments
    "SYN_K_ABS_MAX": 15,            # sanity clip of the sampled boundary-offset dilation |k| (px)
    "DERIVE_EVERY_S": 2.0,          # rows sampled at 0.5 Hz (detector pass, synthetic fragments)
    "COLOUR_EVERY_S": 60.0,         # colour statistics: 1 frame per minute
    "B_GATE_EVERY_S": 240.0,        # B ship-gate frames (cost: 10 grows per calf per frame, ~240 frames over 16 h)
    "STATIC_HOUR_FRAC": 0.5, "STATIC_MIN_SAMPLES": 10,          # STATIC background mask
    "CHROMA_ETA_MIN": 0.8,          # bimodality assert on the Otsu split
    "SUB": 2,                       # colour statistics (chroma, backgrounds) at half resolution; radii halved (ceil)
    "SHIFT_TOL_PX": 2, "SHIFT_MAX_PX": 8,                      # camera-shift check
    "TRUST_PAIR_IOS": 0.50,         # trusted teacher frame: pairwise ios < 0.50
    "MATCH_IOU": 0.50,              # d* = Hungarian match to T with IoU >= 0.5
    "SHEET_PER_CAT_PER_CLIP": 1,    # E8 crops kept per (code x phase) / refusal reason / clip (Tol_56 subsamples)
    "SHEET_MARGIN_PX": 80,
}
# sanity bands: a derived constant outside its band aborts the derivation (never a default)
SANITY_TAU_CONTAIN = (0.60, 0.95)
# pre-registered acceptance (A + carry, and B); thresholds are pre-declared
ACCEPT_A = {"d_iou_changed_min": 0.30, "harm_frac_max": 0.01, "catastrophic_max": 0, "calf_fill_frac_max": 0.002,
            "below_frag_lo_day_reduction_min": 0.50, "harm_delta": 0.02, "calf_fill_px_frac": 0.05}
ACCEPT_B = {"d_iou_median_min": 0.25, "false_fill_p95_frac_max": 0.05, "other_fill_frac_max": 0.005,
            "harm_frac_max": 0.01, "other_fill_px_frac": 0.05, "min_slot_hours_per_day": 0.25}

def floor_2dp(x): return math.floor(float(x) * 100.0 + 1e-9) / 100.0
def ceil_2dp(x): return math.ceil(float(x) * 100.0 - 1e-9) / 100.0

def pct(v, q):
    v = np.asarray(v, float)
    return float(np.percentile(v, q)) if len(v) else None

def pcts(v, qs=(0.1, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9)):
    v = np.asarray(v, float)
    return {str(q): (round(float(np.percentile(v, q)), 5) if len(v) else None) for q in qs} | {"n": int(len(v))}

# =====================================================================================================================
# §B  exact run-length masks. runs_of / _cover / inter / iou_r / ios_r are VERBATIM Tol_48_score_retrack.py:225-273
#     (integer-exact on the stored column-major RLE, no decode). The set operations below are new and exact too.
# =====================================================================================================================
class Runs:
    __slots__ = ("s", "e", "cum", "area", "lo", "hi", "c0", "c1", "r0", "r1", "h")

def runs_of(rle):
    """foreground runs [s, e) of a column-major RLE + prefix lengths + a (column, row) bounding box"""
    h = int(rle["size"][0])
    c = np.asarray(rle["counts"], dtype=np.int64)
    ends = np.cumsum(c); starts = ends - c
    s = starts[1::2]; e = ends[1::2]
    k = e > s
    if not k.all(): s = s[k]; e = e[k]
    R = Runs(); R.h = h; R.s = s; R.e = e
    R.cum = np.concatenate(([0], np.cumsum(e - s)))
    R.area = int(R.cum[-1])
    if R.area:
        R.lo = int(s[0]); R.hi = int(e[-1]); R.c0 = R.lo // h; R.c1 = (R.hi - 1) // h
        if ((s // h) != ((e - 1) // h)).any(): R.r0, R.r1 = 0, h - 1        # a run wraps a column: no row bound
        else: R.r0 = int((s % h).min()); R.r1 = int(((e - 1) % h).max())
    else:
        R.lo = R.hi = R.c0 = R.c1 = R.r0 = R.r1 = 0
    return R

def _cover(B, x):
    """total foreground of B in [0, x) for every x (vectorised)"""
    j = np.searchsorted(B.s, x, side="right") - 1
    jj = np.maximum(j, 0)
    v = B.cum[jj] + np.minimum(x, B.e[jj]) - B.s[jj]
    return np.where(j >= 0, v, 0)

def inter(A, B):
    if A is None or B is None or A.area == 0 or B.area == 0: return 0
    if A is B: return A.area                      # the same RLE object (a root scored against itself)
    if A.lo >= B.hi or B.lo >= A.hi: return 0
    if A.c1 < B.c0 or B.c1 < A.c0 or A.r1 < B.r0 or B.r1 < A.r0: return 0
    if len(A.s) > len(B.s): A, B = B, A
    return int((_cover(B, A.e) - _cover(B, A.s)).sum())

def iou_r(A, B):
    if A is None or B is None: return 0.0
    i = inter(A, B); u = A.area + B.area - i
    return i / u if u else 0.0

def ios_r(A, B):
    if A is None or B is None: return 0.0
    m = min(A.area, B.area)
    return inter(A, B) / m if m else 0.0
# ---------------------------------------------------------------- end of the Tol_48 verbatim block

def cover(X, D):
    """|X ∩ D| / |X| (notation cover(X, d)); 0 for an empty / absent X."""
    if X is None or X.area == 0: return 0.0
    return inter(X, D) / X.area

def runs_from_se(s, e, h=H):
    """Runs object from sorted, disjoint [s, e) arrays (merged here: touching runs become one)."""
    s = np.asarray(s, np.int64); e = np.asarray(e, np.int64)
    if len(s):
        keep = e > s
        s, e = s[keep], e[keep]
    if len(s) > 1:
        brk = s[1:] != e[:-1]
        s, e = s[np.concatenate(([True], brk))], e[np.concatenate((brk, [True]))]
    R = Runs(); R.h = h; R.s = s; R.e = e
    R.cum = np.concatenate(([0], np.cumsum(e - s))).astype(np.int64)
    R.area = int(R.cum[-1])
    if R.area:
        R.lo = int(s[0]); R.hi = int(e[-1]); R.c0 = R.lo // h; R.c1 = (R.hi - 1) // h
        if ((s // h) != ((e - 1) // h)).any(): R.r0, R.r1 = 0, h - 1
        else: R.r0 = int((s % h).min()); R.r1 = int(((e - 1) % h).max())
    else:
        R.lo = R.hi = R.c0 = R.c1 = R.r0 = R.r1 = 0
    return R

EMPTY = runs_from_se([], [])

def _member(R, x):
    if R.area == 0: return np.zeros(len(x), bool)
    j = np.searchsorted(R.s, x, side="right") - 1
    jj = np.maximum(j, 0)
    return (j >= 0) & (x < R.e[jj])

def _setop(A, B, op):
    h = A.h if A is not None else (B.h if B is not None else H)
    A = A if A is not None else EMPTY; B = B if B is not None else EMPTY
    if A.area == 0 and B.area == 0: return runs_from_se([], [], h)
    pts = np.unique(np.concatenate((A.s, A.e, B.s, B.e)))
    left, right = pts[:-1], pts[1:]
    keep = op(_member(A, left), _member(B, left))
    return runs_from_se(left[keep], right[keep], h)

def r_and(A, B):
    if A is None or B is None or A.area == 0 or B.area == 0 or inter(A, B) == 0: return runs_from_se([], [], (A or B or EMPTY).h)
    return _setop(A, B, np.logical_and)

def r_or(A, B):
    if A is None or A.area == 0: return B if B is not None else EMPTY
    if B is None or B.area == 0: return A
    return _setop(A, B, np.logical_or)

def r_sub(A, B):
    """A \\ B"""
    if A is None: return EMPTY
    if B is None or B.area == 0 or A.area == 0 or inter(A, B) == 0: return A
    return _setop(A, B, lambda a, b: a & ~b)

def r_union(seq):
    out = None
    for X in seq:
        if X is None or X.area == 0: continue
        out = X if out is None else r_or(out, X)
    return out if out is not None else EMPTY

def rle_from_runs(R, size=(H, W)):
    """The column-major RLE production's rle_encode would write for this mask (identical counts list: leading 0 when
    the first pixel is set, no trailing background run when the mask ends in foreground, [N] when empty)."""
    h, w = int(size[0]), int(size[1]); n = h * w
    if R is None or R.area == 0: return {"size": [h, w], "counts": [n]}
    s, e = R.s, R.e
    if int(e[-1]) > n or int(s[0]) < 0: raise ValueError("runs exceed the frame")
    gaps = np.empty(2 * len(s), np.int64)
    gaps[0::2] = s - np.concatenate(([0], e[:-1]))
    gaps[1::2] = e - s
    counts = gaps.tolist()
    if int(e[-1]) < n: counts.append(n - int(e[-1]))
    return {"size": [h, w], "counts": counts}

def runs_to_dense(R, shape=(H, W)):
    h, w = shape
    if R is None or R.area == 0: return np.zeros((h, w), bool)
    acc = np.zeros(h * w + 1, np.int32)
    np.add.at(acc, R.s, 1); np.add.at(acc, R.e, -1)
    return (np.cumsum(acc[:-1]) > 0).reshape((h, w), order="F")

def runs_to_crop(R, y0, y1, x0, x1, h=H):
    """dense bool [y1-y0, x1-x0] of the runs inside a crop (decodes only the column block x0..x1)."""
    hc, wc = y1 - y0, x1 - x0
    if R is None or R.area == 0 or R.c1 < x0 or R.c0 >= x1: return np.zeros((hc, wc), bool)
    lo, hi = x0 * h, x1 * h
    s = np.clip(R.s, lo, hi) - lo; e = np.clip(R.e, lo, hi) - lo
    k = e > s
    acc = np.zeros(h * wc + 1, np.int32)
    np.add.at(acc, s[k], 1); np.add.at(acc, e[k], -1)
    blk = (np.cumsum(acc[:-1]) > 0).reshape((h, wc), order="F")
    return blk[y0:y1]

def runs_from_crop(mask, y0, x0, h=H, w=W):
    """Runs of a crop mask placed at (y0, x0) in an h x w frame (column-major), exact."""
    m = np.asarray(mask, bool)
    hc, wc = m.shape
    if not m.any(): return runs_from_se([], [], h)
    pad = np.zeros((wc, hc + 2), np.int8)
    pad[:, 1:-1] = m.T
    d = np.diff(pad.ravel())
    ps = np.flatnonzero(d == 1) + 1; pe = np.flatnonzero(d == -1) + 1
    cs, rs = np.divmod(ps, hc + 2); ce, re_ = np.divmod(pe, hc + 2)
    gs = (x0 + cs).astype(np.int64) * h + (y0 + rs - 1)
    ge = (x0 + ce).astype(np.int64) * h + (y0 + re_ - 1)
    return runs_from_se(gs, ge, h)

def runs_bbox(R):
    """(y0, y1, x0, x1) half-open bounding box (rows exact even when runs wrap columns)."""
    if R is None or R.area == 0: return None
    if R.r0 == 0 and R.r1 == R.h - 1:                     # wrap or full height: compute rows exactly
        rows_s = R.s % R.h; cols_s = R.s // R.h; cols_e = (R.e - 1) // R.h
        r0 = int(np.where(cols_s == cols_e, rows_s, 0).min())
        r1 = int(np.where(cols_s == cols_e, (R.e - 1) % R.h, R.h - 1).max())
        return r0, r1 + 1, R.c0, R.c1 + 1
    return R.r0, R.r1 + 1, R.c0, R.c1 + 1

# =====================================================================================================================
# §C  I/O
# =====================================================================================================================
def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def load_json_gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f: return json.load(f)

GZ_LEVEL = 5            # outputs only (readers are level-agnostic); level 9 costs ~1 min per clip of masks

def dump_json_gz(p, obj, level=GZ_LEVEL):
    """one C-encoder pass + one write (json.dump streams thousands of small chunks through gzip: ~4x slower)."""
    data = json.dumps(obj).encode("utf-8")
    with gzip.open(p, "wb", compresslevel=level) as f: f.write(data)

def sha256_file(p):
    return V5.sha256_file(p)

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_json(obj):
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))

class FrameStream:
    """VERBATIM Tol_48_score_retrack.py:563-612. Streaming reader of one masks_<clip>.json.gz: the header fields plus
    ONE frame dict at a time (a 34-min clip is ~230 MB of JSON, ~1.5 GB as Python objects)."""
    CHUNK = 1 << 20
    TRIM = 2 << 20
    def __init__(self, path):
        self.path = path; self.f = gzip.open(path, "rt", encoding="utf-8")
        self.dec = json.JSONDecoder(); self.buf = ""; self.pos = 0; self.eof = False; self.done = False
        while True:
            k = self.buf.find('"frames"')
            b = self.buf.find("[", k) if k >= 0 else -1
            if b >= 0: break
            if not self._more(): raise RuntimeError(f"{path}: no 'frames' array")
        try: self.header = json.loads(self.buf[:k].rstrip().rstrip(",") + "}")
        except Exception: self.header = {}
        self.pos = b + 1
    def _more(self):
        if self.eof: return False
        s = self.f.read(self.CHUNK)
        if not s: self.eof = True; return False
        if self.pos > self.TRIM: self.buf = self.buf[self.pos:]; self.pos = 0
        self.buf += s
        return True
    def __iter__(self): return self
    def __next__(self):
        if self.done: raise StopIteration
        while True:
            buf, p = self.buf, self.pos; n = len(buf)
            while p < n and buf[p] in " \t\r\n,": p += 1
            self.pos = p
            if p >= n:
                if not self._more(): raise RuntimeError(f"{self.path}: truncated 'frames' array")
                continue
            if buf[p] == "]":
                self.pos = p + 1; self._tail(); self.done = True
                raise StopIteration
            try:
                obj, end = self.dec.raw_decode(buf, p)
            except json.JSONDecodeError:
                if not self._more(): raise
                continue
            self.pos = end
            return obj
    def _tail(self):
        while self._more(): pass
        rest = self.buf[self.pos:].strip().lstrip(",").strip()
        if len(rest) > 1 and rest.endswith("}"):
            try: self.header.update(json.loads("{" + rest))
            except Exception: pass
        self.f.close()

EPOCH = datetime(1970, 1, 1)
def ts_of(w): return (w - EPOCH).total_seconds()
SEG_TOL_S = 0.05            # = Tol_48 SEG_TOL_S: row wall = clip start + t_sec (rounded 0.01 s); segment walls exact

class SegIndex:
    """segments.json of one day dir; row -> segment index exactly as Tol_48.SegIndex (the latest segment of that slot
    starting at or before the row, tolerance SEG_TOL_S, whose end is at or after it). -1 = no segment."""
    def __init__(self, segs_json):
        J = segs_json if isinstance(segs_json, dict) else load_json(segs_json)
        self.recs = []
        for i, r in enumerate(J["segments"]):
            t0 = ts_of(datetime.fromisoformat(r["t_start_wall"]))
            t1 = ts_of(datetime.fromisoformat(r["t_end_wall"])) if r.get("t_end_wall") else float("inf")
            self.recs.append(dict(idx=i, sid=r["seg_id"], slot=int(r["slot"]), t0=t0, t1=t1))
        self.per_slot = defaultdict(list)
        for r in sorted(self.recs, key=lambda z: (z["t0"], z["idx"])): self.per_slot[r["slot"]].append(r)
        self.starts = {s: [r["t0"] for r in v] for s, v in self.per_slot.items()}
    def at(self, slot, t):
        st = self.starts.get(slot)
        if not st: return -1
        i = bisect.bisect_right(st, t + SEG_TOL_S) - 1
        if i < 0: return -1
        r = self.per_slot[slot][i]
        return r["idx"] if t <= r["t1"] + SEG_TOL_S else -1
    def sid(self, idx):
        return self.recs[idx]["sid"] if 0 <= idx < len(self.recs) else None

def save_npz_det(path, **arrays):
    """np.savez with fixed zip metadata (date 1980-01-01, sorted names): byte-deterministic for identical arrays."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            b = io.BytesIO(); np.lib.format.write_array(b, np.asarray(arrays[name]), allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
            zf.writestr(zi, b.getvalue())
    data = buf.getvalue()
    with open(path, "wb") as f: f.write(data)
    return sha256_bytes(data)

def png_bytes(mask_or_img):
    """deterministic PNG (PIL, no metadata) of a bool mask (0/255) or a uint8 image."""
    from PIL import Image
    a = np.asarray(mask_or_img)
    if a.dtype == bool: a = a.astype(np.uint8) * 255
    b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG", optimize=False)
    return b.getvalue()

def png_read(p):
    from PIL import Image
    return np.asarray(Image.open(p))

def write_bytes_frozen(path, data):
    """Frozen artefact: write if absent; if present it must hold exactly these bytes, else raise."""
    if os.path.exists(path):
        with open(path, "rb") as f: old = f.read()
        if old != data:
            raise RuntimeError(f"frozen artefact {path} exists with DIFFERENT content (sha {sha256_bytes(old)[:12]} != "
                               f"{sha256_bytes(data)[:12]}): never overwritten -- inspect, then delete it by hand if the "
                               "new derivation is intended")
        return False
    tmp = f"{path}.tmp_{os.getpid()}_{os.urandom(3).hex()}"
    with open(tmp, "wb") as f: f.write(data)
    try:
        os.replace(tmp, path)
    except OSError:
        with open(path, "wb") as f: f.write(data)
        if os.path.exists(tmp): os.remove(tmp)
    return True

def assert_not_test_path(p):
    """Derivation never opens a test-day path (04-22..28)."""
    if _TEST_RX.search(str(p)):
        raise RuntimeError(f"derivation tried to open a TEST-DAY path: {p}")
    return p

# =====================================================================================================================
# §D  READ constants and the frozen refine-constants file
# =====================================================================================================================
def read_constants(ts, det_run_tag=DET_RUN_TAG):
    """READ: the production band through tol_v5_core.band_from_train_summary (identical code path to the
    tracker) plus the other training-day constants the refinement uses. Raises on any provenance deviation."""
    B = V5.band_from_train_summary(ts, det_run_tag)
    if (B["A_REF_TRAIN"], B["FRAG_LO"], B["OVER_HI"]) != (V5.TRAIN_BAND_REFERENCE["A_REF_TRAIN"],
                                                        V5.TRAIN_BAND_REFERENCE["FRAG_LO"], V5.TRAIN_BAND_REFERENCE["OVER_HI"]):
        raise RuntimeError(f"band {B['A_REF_TRAIN']} {B['FRAG_LO']} {B['OVER_HI']} != tol_v5_core.TRAIN_BAND_REFERENCE")
    C = ts.get("constants") or {}
    prov = {}
    def need(k, literal=False):
        if k not in C: raise RuntimeError(f"train_summary constants lack {k!r}")
        df = set(C[k].get("derived_from") or [])
        if not literal and (not df or not df <= set(TRAIN_DAYS)):
            raise RuntimeError(f"{k} derived_from {sorted(df)} is not within {TRAIN_DAYS}")
        if literal and df and not df <= set(TRAIN_DAYS):
            raise RuntimeError(f"{k} derived_from {sorted(df)} is not within {TRAIN_DAYS}")
        nf = set(C[k].get("not_from") or [])
        if not set(TEST_DAYS) <= nf:
            raise RuntimeError(f"{k} not_from {sorted(nf)} does not exclude every test day {TEST_DAYS}")
        prov[k] = {"value": C[k]["value"], "provenance": C[k].get("provenance"), "derived_from": sorted(df)}
        return C[k]["value"]
    R = {"A_REF": float(B["A_REF_TRAIN"]), "FRAG_LO": dict(B["FRAG_LO"]), "OVER_HI": float(B["OVER_HI"]),
         "TRIG_LO": {"dayl": float(need("S_lo[dayl]")), "dark": float(need("S_lo[dark]"))},
         "LINK_IOU_MIN": float(need("LINK_IOU_MIN")), "BODY_LEN_PX": float(need("body_len_px")),
         "ACTIVE_THR": {d: float(need(f"ACTIVE_THR[{d}]")) for d in TRAIN_DAYS},
         "DISJ_IOS": float(need("DISJ_IOS", literal=True)), "VETO_IOS": float(need("VETO_IOS", literal=True)),
         "op_threshold": B["op_threshold"], "resolution": B["resolution"],
         "rf_checkpoint_sha256": ts.get("checkpoint_best_total_sha256"), "det_run_tag": det_run_tag,
         "faults_json_sha256": (ts.get("provenance") or {}).get("faults_json_sha256"),
         "train_clips": list((ts.get("provenance") or {}).get("train_clips") or ts.get("train_clips") or []),
         "valid_clips": list((ts.get("provenance") or {}).get("valid_clips") or ts.get("valid_clips") or [])}
    R["provenance"] = {**{k: v for k, v in B["provenance"].items()}, **prov}
    if not R["train_clips"] or not R["valid_clips"]:
        raise RuntimeError("train_summary has no train_clips / valid_clips (needed for the UNSEEN ∪ VALID derivation set)")
    if not R["FRAG_LO"]["dayl"] < R["TRIG_LO"]["dayl"] or not R["FRAG_LO"]["dark"] < R["TRIG_LO"]["dark"]:
        raise RuntimeError(f"TRIG_LO {R['TRIG_LO']} must exceed FRAG_LO {R['FRAG_LO']}")
    return R

def px(C, key, pool=None):
    """constant x A_REF in px."""
    v = C[key][pool] if pool is not None else C[key]
    return float(v) * C["A_REF"]

DERIVED_KEYS = ("TAU_CONTAIN", "TAU_CONFLICT", "CARRY_MAX_S", "CHROMA_MIN", "LLR_MIN", "BL_P99", "SOL_P1")
APPLY_KEYS = ("TAU_CONTAIN", "TAU_CONFLICT", "CARRY_MAX_S")

def load_refine_constants(path, read_c):
    """The frozen derivation output merged with the READ constants -> the dict every stage uses. Raises if it
    is not a full (non-smoke) derivation, if a value lacks its provenance, or if READ values differ from today's READ."""
    J = load_json(path)
    if J.get("version") != "refine_constants_v1": raise RuntimeError(f"{path}: version {J.get('version')!r}")
    D = J.get("DERIVED") or {}
    out = dict(read_c)
    for k in DERIVED_KEYS:
        if k not in D: raise RuntimeError(f"{path}: DERIVED lacks {k}")
        rec = D[k]
        nf = set(rec.get("not_from") or [])
        if not set(TEST_DAYS) <= nf: raise RuntimeError(f"{path}: {k}.not_from does not list every test day")
        df = set(rec.get("derived_from_days") or [])
        if not df or not df <= set(TRAIN_DAYS): raise RuntimeError(f"{path}: {k}.derived_from_days {sorted(df)}")
        out[k] = rec["value"]
    for k in ("A_REF", "FRAG_LO", "OVER_HI", "TRIG_LO", "LINK_IOU_MIN", "BODY_LEN_PX"):
        if J["READ"][k] != read_c[k]: raise RuntimeError(f"{path}: READ {k} {J['READ'][k]} != today's {read_c[k]}")
    out["B_SHIP"] = J.get("b_ship_gate") or {}
    out["constants_sha256"] = sha256_file(path)
    out["smoke"] = bool(J.get("smoke"))
    return out

def check_apply_constants(C):
    for p in ("dayl", "dark"):
        lo, hi = SANITY_TAU_CONTAIN
        if not lo <= C["TAU_CONTAIN"][p] <= hi: raise RuntimeError(f"TAU_CONTAIN[{p}] {C['TAU_CONTAIN'][p]} outside {SANITY_TAU_CONTAIN}")
        if not C["DISJ_IOS"] <= C["TAU_CONFLICT"][p] <= C["VETO_IOS"]:
            raise RuntimeError(f"TAU_CONFLICT[{p}] {C['TAU_CONFLICT'][p]} outside [{C['DISJ_IOS']}, {C['VETO_IOS']}]")
        if not C["TAU_CONFLICT"][p] < C["TAU_CONTAIN"][p]: raise RuntimeError("TAU_CONFLICT must be < TAU_CONTAIN")
    if not 0.0 <= float(C["CARRY_MAX_S"]) <= 2.0: raise RuntimeError(f"CARRY_MAX_S {C['CARRY_MAX_S']}")

# =====================================================================================================================
# §E  clips, walls, phases, trigger, census, sampling
# =====================================================================================================================
_RXC = re.compile(r"(\d{8})-(\d{6})-(\d{6})")

def clip_start(name):
    m = _RXC.search(os.path.basename(str(name)).replace("_", "-"))
    if m is None: raise RuntimeError(f"cannot parse a clip start from {name!r}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

def clip_stem(name):
    b = os.path.basename(str(name))
    if b.endswith(".mp4"): b = b[:-4]
    if b.startswith("masks_") and b.endswith(".json.gz"): b = b[len("masks_"):-len(".json.gz")]
    return b.replace(" ", "_")

def wall_of(w0, t_sec): return w0 + timedelta(seconds=float(t_sec))
def pool_of(w): return V5.pool_of_wall(w)
def phase_of(w): return V5.phase4(w)

def rle_area(rle): return int(sum(rle["counts"][1::2])) if rle else 0

def trig_lo_px(pool, C): return float(C["TRIG_LO"][pool]) * C["A_REF"]

def is_trigger(area, pool, C):
    """Trigger: the slot has a mask and 0 < area < TRIG_LO[pool] x A_ref (identity-agnostic, I7)."""
    return 0 < area < trig_lo_px(pool, C)

def frame_triggers(fr, w0, C):
    w = wall_of(w0, fr["t_sec"]); pool = pool_of(w)
    ms = fr.get("masks") or {}
    return [s for s in range(N_SLOTS) if ms.get(str(s)) and is_trigger(rle_area(ms[str(s)]), pool, C)], w, pool

RENDER_SAMPLE_SEC = 2.0     # Tol_17 sample_sec default (the 2-s render cadence)

def sample_selection(frames, sample):
    """Row indices eligible for a NEW detector call. '1' = every 5-fps row; 'render' = exactly the rows Tol_17
    renders at its default 2.0 s (`last_t is None or t - last_t >= SAMPLE_SEC`, per clip); an integer k = every k-th row."""
    s = str(sample).strip().lower()
    if s in ("1", ""): return None                                  # None = all rows
    if s == "render":
        sel, last = set(), None
        for i, r in enumerate(frames):
            if last is None or r["t_sec"] - last >= RENDER_SAMPLE_SEC:
                sel.add(i); last = r["t_sec"]
        return sel
    k = int(s)
    if k < 1: raise ValueError(f"sample {sample!r}")
    return None if k == 1 else set(range(0, len(frames), k))

def census_clip(frames, clip_name, C, stored_fidx=(), sample="1"):
    """S0: exact trigger counts of one clip from the masks alone (the GPU budget)."""
    w0 = clip_start(clip_name); stored = set(int(f) for f in stored_fidx)
    sel = sample_selection(frames, sample)
    out = {"clip": clip_stem(clip_name), "frames": len(frames), "trigger_rows": 0, "trigger_frames": 0,
           "trigger_frames_stored_call": 0, "new_calls": 0, "rows_by_pool": Counter(), "rows_by_phase": Counter(),
           "rows_by_slot": Counter(), "below_frag_lo_rows": 0, "mask_rows": 0}
    for i, fr in enumerate(frames):
        trig, w, pool = frame_triggers(fr, w0, C)
        ms = fr.get("masks") or {}
        for s in range(N_SLOTS):
            rl = ms.get(str(s))
            if rl:
                out["mask_rows"] += 1
                if rle_area(rl) < px(C, "FRAG_LO", pool): out["below_frag_lo_rows"] += 1
        if not trig: continue
        out["trigger_rows"] += len(trig); out["trigger_frames"] += 1
        out["rows_by_pool"][pool] += len(trig); out["rows_by_phase"][phase_of(w)] += len(trig)
        for s in trig: out["rows_by_slot"][str(s)] += 1
        if int(fr["frame_idx"]) in stored: out["trigger_frames_stored_call"] += 1
        elif sel is None or i in sel: out["new_calls"] += 1
    for k in ("rows_by_pool", "rows_by_phase", "rows_by_slot"): out[k] = dict(out[k])
    return out

# =====================================================================================================================
# §F  (A) gates + pixel resolution, double-claims, orphans
# =====================================================================================================================
def kept_runs(rec):
    """kept detections of one DET_LOG record (production post-chain order) as Runs."""
    if rec is None: return None
    raw = rec.get("raw") or []
    return [runs_of(raw[k]["rle"]) for k in rec.get("kept_idx") or []]

def gate_frame(M, D, pool, C, trig):
    """The (A) gates for every trigger slot of one frame + pixel resolution (+ the NO_SHRINK gate when
    C['NO_SHRINK'] is true: refused 'shrink' when area(R_s) <= area(M_s).
    M: list of N_SLOTS Runs|None (original v5 masks); D: kept detections (Runs list); trig: trigger slots.
    Returns {slot: decision}; decision = {code ('d'|'r'), reason, det_idx, cover, cover2, n_cand, ios_conflict,
    conflict_slot, det_area, R (Runs for 'd')}."""
    tc = float(C["TAU_CONTAIN"][pool]); tf = float(C["TAU_CONFLICT"][pool])
    lo, hi = px(C, "FRAG_LO", pool), float(C["OVER_HI"]) * C["A_REF"]
    dec = {}
    for s in sorted(trig):
        m = M[s]
        covs = [cover(m, d) for d in D]
        srt = sorted(covs, reverse=True)
        cand = [i for i, c in enumerate(covs) if c >= tc]
        rec = {"code": "r", "reason": None, "det_idx": None, "cover": round(srt[0], 4) if srt else 0.0,
               "cover2": round(srt[1], 4) if len(srt) > 1 else 0.0, "n_cand": len(cand), "ios_conflict": None,
               "conflict_slot": None, "det_area": None, "R": None, "n_det": len(D)}
        if not cand:
            rec["reason"] = "nodet" if (srt[0] if srt else 0.0) < tf else "straddle"
        elif len(cand) >= 2:
            rec["reason"] = "ambiguous"
        else:
            i = cand[0]; d = D[i]; rec["det_idx"] = i; rec["det_area"] = d.area
            if d.area < lo: rec["reason"] = "small"
            elif d.area > hi: rec["reason"] = "big"
            else:
                conf = [(ios_r(d, M[t]), t) for t in range(N_SLOTS) if t != s and M[t] is not None]
                mx, ct = max(conf) if conf else (0.0, None)
                rec["ios_conflict"] = round(mx, 4); rec["conflict_slot"] = ct
                if mx >= tf: rec["reason"] = "conflict"
                else: rec["code"] = "d"
        dec[s] = rec
    acc = [s for s in sorted(dec) if dec[s]["code"] == "d"]
    if not acc: return dec
    # step 1: other slots' ORIGINAL pixels always win
    R1 = {}
    for s in acc:
        oth = r_union(M[t] for t in range(N_SLOTS) if t != s and M[t] is not None)
        R1[s] = r_sub(D[dec[s]["det_idx"]], oth)
    # step 2: two accepted slots: a real overlap reverts both, a small one is removed from both
    bad = set()
    for a, b in itertools.combinations(acc, 2):
        if ios_r(R1[a], R1[b]) >= tf:
            bad |= {a, b}
            for x, y in ((a, b), (b, a)):
                dec[x]["pair_slot"] = y
    keep = [s for s in acc if s not in bad]
    R2 = {}
    for s in keep:
        R2[s] = r_sub(R1[s], r_union(R1[t] for t in keep if t != s))
    for s in bad:
        dec[s]["code"] = "r"; dec[s]["reason"] = "pair_conflict"
    # step 3: post-shrink keeps I5 (containment) and the band floor
    for s in keep:
        R = R2[s]
        if R.area < lo or cover(M[s], R) < tc:
            dec[s]["code"] = "r"; dec[s]["reason"] = "post_shrink"; dec[s]["post_area"] = R.area
        elif C.get("NO_SHRINK", True) and R.area <= M[s].area:
            dec[s]["code"] = "r"; dec[s]["reason"] = "shrink"; dec[s]["post_area"] = R.area
        else:
            dec[s]["R"] = R                                   # step 4: the PURE detection geometry
    return dec

def claims_frame(M, D, pool, C):
    """By-products at any call frame: double-claims (one in-band kept detection contains >= TAU_CONTAIN of two
    slots' masks) and orphans (an in-band detection overlapping no slot: ios < TAU_CONFLICT with every mask)."""
    tc = float(C["TAU_CONTAIN"][pool]) if "TAU_CONTAIN" in C else None
    tf = float(C["TAU_CONFLICT"][pool]) if "TAU_CONFLICT" in C else None
    dcl, orph = [], []
    if tc is None or not D: return dcl, orph
    for i, d in enumerate(D):
        if not V5.in_band(d.area, pool, {"FRAG_LO": C["FRAG_LO"], "OVER_HI": C["OVER_HI"], "A_REF_TRAIN": C["A_REF"]}):
            continue
        hold = [s for s in range(N_SLOTS) if M[s] is not None and cover(M[s], d) >= tc]
        if len(hold) >= 2: dcl.append({"det_idx": i, "slots": hold, "det_area": d.area})
        if all(M[s] is None or ios_r(d, M[s]) < tf for s in range(N_SLOTS)): orph.append({"det_idx": i, "det_area": d.area})
    return dcl, orph

def doubleclaim_intervals(rows, min_s=None, gap_s=None):
    """rows: [{wall_ts, slots, ...}] of one day -> intervals per slot pair lasting >= DOUBLECLAIM_MIN_S."""
    min_s = DESIGN["DOUBLECLAIM_MIN_S"] if min_s is None else min_s
    gap_s = DESIGN["DOUBLECLAIM_GAP_S"] if gap_s is None else gap_s
    by = defaultdict(list)
    for r in rows:
        for a, b in itertools.combinations(sorted(r["slots"]), 2): by[(a, b)].append(r)
    out = []
    for (a, b), rr in sorted(by.items()):
        rr = sorted(rr, key=lambda z: z["ts"])
        cur = [rr[0]]
        for r in rr[1:] + [None]:
            if r is not None and r["ts"] - cur[-1]["ts"] <= gap_s:
                cur.append(r); continue
            dur = cur[-1]["ts"] - cur[0]["ts"]
            if dur >= min_s:
                out.append({"slots": [a, b], "from": (EPOCH + timedelta(seconds=cur[0]["ts"])).isoformat(timespec="milliseconds"),
                            "to": (EPOCH + timedelta(seconds=cur[-1]["ts"])).isoformat(timespec="milliseconds"),
                            "dur_s": round(dur, 2), "n_rows": len(cur), "clips": sorted({z["clip"] for z in cur})})
            if r is not None: cur = [r]
    return out

# =====================================================================================================================
# §G  apply_clip: (A) first pass + carry + output + sidecar + asserts
# =====================================================================================================================
def _mask_runs(fr):
    ms = fr.get("masks") or {}
    return [runs_of(ms[str(s)]) if ms.get(str(s)) else None for s in range(N_SLOTS)]

def apply_clip(src, stored, new, segs, C, sample_sel=None, keep_rows=False):
    """S3 on one clip. src = the source masks JSON (header + frames); stored / new = {frame_idx: DET_LOG record}
    (production calls / refine calls); segs = SegIndex of the day. Returns (out, side, extra):
      out   -- the refined masks JSON: header verbatim + 'refine' block (filled by the caller), frames with the RLE
               replaced for d / c rows and a 'prov' key per frame;
      side  -- one record per trigger row (refine_<clip>);
      extra -- doubleclaim rows, orphans, counts."""
    frames = src["frames"]; clip = src.get("clip"); w0 = clip_start(clip)
    size = None
    n = len(frames)
    rows = []                            # per frame: (t_sec, wall_ts, pool, phase, M, trig, D_src, D, dec)
    dcl_rows, orphans = [], []
    n_calls = Counter()
    for i, fr in enumerate(frames):
        trig, w, pool = frame_triggers(fr, w0, C)
        fidx = int(fr["frame_idx"])
        rec, src_kind = (stored.get(fidx), "stored") if fidx in stored else (new.get(fidx), "new")
        if rec is None: src_kind = None
        D = None
        if rec is not None and rec.get("decode_ok", True):
            D = kept_runs(rec); n_calls[src_kind] += 1
        need_M = bool(trig) or D is not None
        M = _mask_runs(fr) if need_M else None
        if size is None:
            for v in (fr.get("masks") or {}).values():
                if v: size = tuple(v["size"]); break
        dec = {}
        if trig:
            if D is not None:
                dec = gate_frame(M, D, pool, C, trig)
            else:
                why = "decode_fail" if rec is not None else ("unsampled" if (sample_sel is not None and i not in sample_sel)
                                                             else None)
                if why is None:
                    raise RuntimeError(f"{clip_stem(clip)} frame {fidx}: trigger row without a detector call "
                                       "(detect stage incomplete) and the row is inside the sample")
                for s in trig:
                    dec[s] = {"code": "r", "reason": why, "det_idx": None, "cover": None, "cover2": None, "n_cand": None,
                              "ios_conflict": None, "conflict_slot": None, "det_area": None, "R": None, "n_det": None}
        if D is not None:
            dc, orp = claims_frame(M, D, pool, C)
            for z in dc: dcl_rows.append(dict(z, ts=ts_of(w), fidx=fidx, clip=clip_stem(clip), src=src_kind))
            for z in orp: orphans.append(dict(z, ts=ts_of(w), fidx=fidx, src=src_kind))
        rows.append({"t": float(fr["t_sec"]), "ts": ts_of(w), "pool": pool, "phase": phase_of(w), "M": M,
                     "trig": trig, "src": src_kind if D is not None else None, "dec": dec, "fidx": fidx})
    if size is None: size = (H, W)
    # ---- carry: second pass over the clip --------------------------------------------------------------------------
    cmax = float(C["CARRY_MAX_S"])
    d_rows = {s: [] for s in range(N_SLOTS)}                   # slot -> sorted [(t, i)] of 'd' rows
    for i, r in enumerate(rows):
        for s, dd in r["dec"].items():
            if dd["code"] == "d": d_rows[s].append((r["t"], i))
    d_t = {s: [t for t, _ in v] for s, v in d_rows.items()}
    seg_cache = {}
    def seg_at(s, i):
        k = (s, i)
        if k not in seg_cache: seg_cache[k] = segs.at(s, rows[i]["ts"]) if segs is not None else -1
        return seg_cache[k]
    if cmax > 0:
        for i, r in enumerate(rows):
            for s in sorted(r["dec"]):
                dd = r["dec"][s]
                if dd["code"] != "r" or dd["reason"] not in CARRY_ELIGIBLE: continue
                ts_ = d_t[s]
                if not ts_: continue
                j0 = bisect.bisect_left(ts_, r["t"] - cmax - 1e-6); j1 = bisect.bisect_right(ts_, r["t"] + cmax + 1e-6)
                sg = seg_at(s, i)
                if sg < 0: continue
                cands = [(abs(ts_[j] - r["t"]), ts_[j] - r["t"], d_rows[s][j][1]) for j in range(j0, j1)
                         if seg_at(s, d_rows[s][j][1]) == sg]
                if not cands: continue
                _adt, dt, a = min(cands, key=lambda z: (z[0], z[1]))          # nearest; tie -> the earlier row
                ra = rows[a]; Ra = ra["dec"][s]["R"]; Mr = r["M"]
                tc = float(C["TAU_CONTAIN"][r["pool"]]); tf = float(C["TAU_CONFLICT"][r["pool"]])
                lo = px(C, "FRAG_LO", r["pool"])
                why = None
                link = iou_r(Mr[s], ra["M"][s])
                if link < float(C["LINK_IOU_MIN"]): why = "link"
                elif cover(Mr[s], Ra) < tc: why = "contain"
                elif max([ios_r(Ra, Mr[t]) for t in range(N_SLOTS) if t != s and Mr[t] is not None] or [0.0]) >= tf:
                    why = "conflict"
                R = None
                if why is None:
                    R0 = r_sub(Ra, r_union(Mr[t] for t in range(N_SLOTS) if t != s and Mr[t] is not None))
                    fin = [r["dec"][t]["R"] for t in r["dec"] if t != s and r["dec"][t]["code"] in ("d", "c")]
                    if any(ios_r(R0, F) >= tf for F in fin): why = "pair_conflict"
                    else:
                        R = r_sub(R0, r_union(fin))
                        if R.area < lo or cover(Mr[s], R) < tc: why = "post_shrink"
                        elif C.get("NO_SHRINK", True) and R.area <= Mr[s].area: why = "shrink"
                dd["carry_src_fidx"] = rows[a]["fidx"]; dd["carry_dt"] = round(dt, 3); dd["carry_link_iou"] = round(link, 4)
                if why is None:
                    dd["reason0"] = dd["reason"]; dd["code"] = "c"; dd["reason"] = None; dd["R"] = R
                else:
                    dd["reason0"] = dd["reason"]; dd["reason"] = "carry_fail"; dd["carry_why"] = why
    # ---- output frames, sidecar --------------------------------------------------------------------------------
    out_frames, side = [], []
    counts = Counter()
    for i, (fr, r) in enumerate(zip(frames, rows)):
        ms = fr.get("masks") or {}
        nm = dict(ms)
        prov = {k: "k" for k in ms}
        for s, dd in r["dec"].items():
            prov[str(s)] = dd["code"]
            counts[dd["code"]] += 1
            if dd["code"] in ("d", "c"):
                nm[str(s)] = rle_from_runs(dd["R"], size)
            rec = {"slot": s, "frame_idx": r["fidx"], "t_sec": fr["t_sec"], "code": dd["code"], "reason": dd["reason"],
                   "pool": r["pool"], "phase": r["phase"], "det_src": r["src"], "det_idx": dd.get("det_idx"),
                   "n_det": dd.get("n_det"), "cover": dd.get("cover"), "cover2": dd.get("cover2"),
                   "n_cand": dd.get("n_cand"), "ios_conflict": dd.get("ios_conflict"),
                   "conflict_slot": dd.get("conflict_slot"), "det_area": dd.get("det_area"),
                   "area_before": r["M"][s].area if r["M"] is not None and r["M"][s] is not None else None,
                   "area_after": dd["R"].area if dd["code"] in ("d", "c") else None,
                   "seg": seg_at(s, i) if segs is not None else None}
            for k in ("reason0", "carry_src_fidx", "carry_dt", "carry_why", "carry_link_iou", "pair_slot", "post_area"):
                if k in dd: rec[k] = dd[k]
            side.append(rec)
        nf = {}
        for k, v in fr.items():
            nf[k] = nm if k == "masks" else v
        nf["prov"] = prov
        out_frames.append(nf)
        for k in ms:
            if prov[k] == "k" and ms[k]: counts["k"] += 1
    out = {}
    for k, v in src.items():
        out[k] = out_frames if k == "frames" else v
    extra = {"doubleclaim_rows": dcl_rows, "orphans": orphans, "counts": dict(counts), "n_calls": dict(n_calls),
             "n_frames": n, "n_trigger_rows": len(side)}
    if keep_rows: extra["rows"] = rows
    return out, side, extra

def verify_clip_output(src, out, side, C, allowed_codes=("k", "r", "d", "c"), grow_floor_trig=False):
    """Invariant asserts on one published clip. Raises AssertionError with the first violation."""
    fs, fo = src["frames"], out["frames"]
    w0 = clip_start(src.get("clip"))
    if len(fs) != len(fo): raise AssertionError(f"I1: {len(fs)} source frames != {len(fo)} output frames")
    for k in src:
        if k != "frames" and out.get(k) != src[k]: raise AssertionError(f"header key {k!r} changed")
    n_trig = 0
    for i, (a, b) in enumerate(zip(fs, fo)):
        if a["frame_idx"] != b["frame_idx"] or a["t_sec"] != b["t_sec"]:
            raise AssertionError(f"I1: frame {i} frame_idx/t_sec changed")
        ma, mb = a.get("masks") or {}, b.get("masks") or {}
        if set(ma) != set(mb): raise AssertionError(f"I2: frame {a['frame_idx']} slot keys changed")
        prov = b.get("prov") or {}
        if set(prov) != set(ma): raise AssertionError(f"prov keys != slot keys at frame {a['frame_idx']}")
        trig, _w, pool = frame_triggers(a, w0, C)
        n_trig += len(trig)
        tc = float(C["TAU_CONTAIN"][pool]); lo = px(C, "FRAG_LO", pool); hi = float(C["OVER_HI"]) * C["A_REF"]
        if grow_floor_trig: lo_g = trig_lo_px("dayl", C)
        fin = {k: (runs_of(mb[k]) if mb[k] else None) for k in mb}
        for k in ma:
            code = prov[k]
            if code not in allowed_codes: raise AssertionError(f"code {code!r} not allowed here")
            if (ma[k] is None) != (mb[k] is None): raise AssertionError(f"I2: slot {k} presence changed at {a['frame_idx']}")
            if code in ("k", "r"):
                if ma[k] != mb[k]: raise AssertionError(f"I3: slot {k} row {a['frame_idx']} code {code} not byte-identical")
                if code == "k" and int(k) in trig: raise AssertionError(f"trigger row coded k at {a['frame_idx']} slot {k}")
                if code == "r" and int(k) not in trig: raise AssertionError(f"non-trigger row coded r at {a['frame_idx']}")
            else:
                if int(k) not in trig: raise AssertionError(f"non-trigger row refined at {a['frame_idx']} slot {k}")
                Mo, R = runs_of(ma[k]), fin[k]
                if cover(Mo, R) < tc - 1e-12: raise AssertionError(f"I5: cover {cover(Mo, R):.4f} < {tc} at {a['frame_idx']} slot {k}")
                flo = lo_g if (code == "g" and grow_floor_trig) else lo
                if not flo <= R.area <= hi: raise AssertionError(f"band: area {R.area} at {a['frame_idx']} slot {k} code {code}")
                for k2, R2 in fin.items():
                    if k2 != k and R2 is not None and inter(R, R2):
                        raise AssertionError(f"I6: slot {k} overlaps slot {k2} at {a['frame_idx']} ({inter(R, R2)} px)")
    if n_trig != len(side): raise AssertionError(f"sidecar has {len(side)} records for {n_trig} trigger rows")
    return n_trig

# =====================================================================================================================
# §H  detect-stage helpers (S2) and E0 parity
# =====================================================================================================================
def stored_by_fidx(det_json):
    """{frame_idx: record} of a production detections_<clip>.json.gz; two calls at one frame would be a contract
    violation (production: <= 1 detect() per frame) and raise."""
    out = {}
    for r in (det_json or {}).get("records") or []:
        f = int(r["frame_idx"])
        if f in out: raise RuntimeError(f"two stored detector calls at frame {f}")
        out[f] = r
    return out

def plan_new_calls(frames, clip_name, C, stored_fidx, sample_sel):
    """Trigger frames inside the sample without a stored production call, ascending (the decode order)."""
    w0 = clip_start(clip_name); stored = set(stored_fidx); out = []
    for i, fr in enumerate(frames):
        trig, _w, _p = frame_triggers(fr, w0, C)
        if not trig: continue
        f = int(fr["frame_idx"])
        if f in stored: continue
        if sample_sel is not None and i not in sample_sel: continue
        out.append(f)
    return out

def det_record(clip_name, fidx, t_sec, rows_rle, kept_idx, det_ms, pool, C, kind="refine", decode_ok=True):
    """One record in the production DET_LOG schema (Tol_47:744-749) with kind='refine'. rows_rle: [(rle, score, area)]."""
    B = {"FRAG_LO": C["FRAG_LO"], "OVER_HI": C["OVER_HI"], "A_REF_TRAIN": C["A_REF"]}
    n_in = sum(1 for k in kept_idx if V5.in_band(rows_rle[k][2], pool, B)) if decode_ok else None
    return {"clip": clip_name, "frame_idx": int(fidx), "t_sec": None if t_sec is None else round(float(t_sec), 2),
            "kind": kind, "n_raw": len(rows_rle) if decode_ok else None, "det_ms": round(float(det_ms), 1),
            "n_inband": n_in, "raw": [{"rle": r, "score": round(float(s), 4), "area": int(a)} for r, s, a in rows_rle],
            "kept_idx": list(kept_idx), **({} if decode_ok else {"decode_ok": False})}

PARITY_IOU_MIN = 0.995      # tolerance (another GPU class)
PARITY_DSCORE_MAX = 1e-3

def parity_compare(stored, new, exact):
    """E0. exact (same GPU class as the production root): same n_raw, identical kept_idx order, per raw mask an
    identical RLE and score -- positional, because identical numerics give identical order.
    tolerance (another GPU class): same n_raw; every stored raw mask matched one-to-one (Hungarian on IoU) to a new raw
    mask with IoU >= 0.995 and |Δscore| <= 1e-3; the stored kept SET maps onto the new kept set through that matching.
    The raw list is score-ranked (RF-DETR top-k) and the kept list area-ranked, and the stored production calls hold
    adjacent raw scores <= 1e-4 apart (12 on 04-22, 26 on 04-23) and kept areas < 0.1 % apart (7 / 10): numerics inside
    the tolerance can reorder them, so a positional comparison would report a contract failure for an equivalent
    output. Order carries no information downstream (gates, carry and claims use the kept SET; dedupe outcomes between
    overlapping masks are captured by the set), so in tolerance mode a reorder is recorded ('order_differs') only."""
    why = []
    ra, rb = stored.get("raw") or [], new.get("raw") or []
    ka, kb = list(stored.get("kept_idx") or []), list(new.get("kept_idx") or [])
    if stored.get("n_raw") != new.get("n_raw"): why.append(f"n_raw {stored.get('n_raw')} != {new.get('n_raw')}")
    min_iou, max_ds, order_differs = 1.0, 0.0, False
    if exact:
        if ka != kb: why.append(f"kept_idx {ka} != {kb}")
        for a, b in zip(ra, rb):
            ds = abs(float(a["score"]) - float(b["score"])); max_ds = max(max_ds, ds)
            if a["rle"]["counts"] != b["rle"]["counts"]:
                why.append("raw RLE differs"); min_iou = min(min_iou, iou_r(runs_of(a["rle"]), runs_of(b["rle"])))
            if round(float(b["score"]), 4) != float(a["score"]): why.append(f"score {a['score']} != {b['score']}")
    elif len(ra) != len(rb):
        pass                                                    # n_raw mismatch already recorded: no matching
    elif ra:
        A = [runs_of(x["rle"]) for x in ra]; B = [runs_of(x["rle"]) for x in rb]
        IoU = np.array([[iou_r(a, b) for b in B] for a in A])
        ri, ci = linear_sum_assignment(-IoU)
        perm = {int(i): int(j) for i, j in zip(ri, ci)}         # stored raw index -> new raw index
        for i, j in perm.items():
            v = float(IoU[i, j]); min_iou = min(min_iou, v)
            ds = abs(float(ra[i]["score"]) - float(rb[j]["score"])); max_ds = max(max_ds, ds)
            if v < PARITY_IOU_MIN: why.append(f"raw {i}->{j} IoU {v:.4f} < {PARITY_IOU_MIN}")
            if ds > PARITY_DSCORE_MAX: why.append(f"raw {i}->{j} |Δscore| {ds:.5f} > {PARITY_DSCORE_MAX}")
        mapped = [perm[k] for k in ka if k in perm]
        if len(mapped) != len(ka) or set(mapped) != set(kb):
            why.append(f"kept set {ka} (-> {mapped}) != {kb}")
        order_differs = (any(perm[i] != i for i in perm) or mapped != kb)
    return {"ok": not why, "why": why[:6], "min_iou": round(min_iou, 6), "max_dscore": round(max_ds, 6),
            "frame_idx": int(stored["frame_idx"]), "kind": stored.get("kind"), "exact": bool(exact),
            "order_differs": bool(order_differs)}

# =====================================================================================================================
# §I  colour model
# =====================================================================================================================
def disk(r):
    r = int(r)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r

def dilate(mask, r):
    """exact binary dilation by a Euclidean disk of radius r (cv2 and scipy give the identical result)."""
    m = np.asarray(mask, bool)
    r = int(r)
    if r <= 0 or not m.any(): return m.copy()
    if _cv2 is not None:
        return _cv2.dilate(m.astype(np.uint8), disk(r).astype(np.uint8), borderType=_cv2.BORDER_CONSTANT, borderValue=0) > 0
    return ndi.binary_dilation(m, structure=disk(r))

def erode(mask, r):
    m = np.asarray(mask, bool)
    r = int(r)
    if r <= 0 or not m.any(): return m.copy()
    if _cv2 is not None:
        return _cv2.erode(m.astype(np.uint8), disk(r).astype(np.uint8), borderType=_cv2.BORDER_CONSTANT, borderValue=0) > 0
    return ndi.binary_erosion(m, structure=disk(r), border_value=0)

def hsv_sat(rgb):
    """HSV saturation, OpenCV 8-bit convention S = round(255 (max - min) / max), 0 when max = 0 (numpy)."""
    a = np.asarray(rgb)
    mx = a.max(axis=-1).astype(np.int32); mn = a.min(axis=-1).astype(np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(mx > 0, ((mx - mn) * 255 + mx // 2) // np.maximum(mx, 1), 0)
    return s.astype(np.uint8)

def lut_index(rgb, bits=None):
    b = DESIGN["LUT_BITS"] if bits is None else bits
    a = np.asarray(rgb).astype(np.uint16); sh = 8 - b
    return ((a[..., 0] >> sh) << (2 * b)) | ((a[..., 1] >> sh) << b) | (a[..., 2] >> sh)

def lut_hist(rgb, sel):
    nb = 1 << (3 * DESIGN["LUT_BITS"])
    return np.bincount(lut_index(rgb)[sel].ravel(), minlength=nb).astype(np.int64)

def build_lut(hp, hn):
    """log P(c | calf) / P(c | background), Laplace (+1) smoothing over the 4096 bins."""
    hp = np.asarray(hp, np.float64); hn = np.asarray(hn, np.float64); K = len(hp)
    return (np.log((hp + 1.0) / (hp.sum() + K)) - np.log((hn + 1.0) / (hn.sum() + K))).astype(np.float32)

def calf_like(rgb, lut, llr_min):
    return lut[lut_index(rgb)] >= llr_min

def subsample(a, sub=None):
    k = DESIGN["SUB"] if sub is None else sub
    return a[::k, ::k]

def chroma_of(rgb, excl_full, sub=None):
    """Colour-frame statistic: median HSV saturation of the pixels outside excl (slot masks + detections),
    excl dilated by BG_EXCL_DIL; at 1/SUB resolution (radius ceil(BG_EXCL_DIL / SUB))."""
    k = DESIGN["SUB"] if sub is None else sub
    S = hsv_sat(subsample(rgb, k))
    ex = dilate(subsample(np.asarray(excl_full, bool), k), math.ceil(DESIGN["BG_EXCL_DIL"] / k))
    v = S[~ex]
    return float(np.median(v)) if v.size else None

def otsu(values):
    """Otsu on integer values 0..255 -> (threshold k: class 0 = v <= k, eta = sigma_B^2 / sigma_T^2)."""
    v = np.clip(np.round(np.asarray(values, float)), 0, 255).astype(np.int64)
    h = np.bincount(v, minlength=256).astype(np.float64); p = h / max(h.sum(), 1)
    om = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * om - mu) ** 2 / (om * (1 - om))
    sb = np.nan_to_num(sb, nan=0.0, posinf=0.0)
    k = int(np.argmax(sb)); st = float((p * (np.arange(256) - mt) ** 2).sum())
    return k, (float(sb[k]) / st if st > 0 else 0.0)

def masked_median(frames, excl):
    """per-pixel LOWER median of each channel over the non-excluded samples; frames [n, h, w, 3] uint8, excl
    [n, h, w] bool. Returns (median uint8 [h, w, 3], count int [h, w]); pixels with count 0 hold 0."""
    F = np.asarray(frames, np.uint16); E = np.asarray(excl, bool)
    n = F.shape[0]
    F = np.where(E[..., None], np.uint16(999), F)
    F.sort(axis=0)
    cnt = (~E).sum(axis=0)
    idx = np.clip((cnt - 1) // 2, 0, n - 1)
    med = np.take_along_axis(F, idx[None, ..., None].repeat(3, axis=-1), axis=0)[0]
    med = np.where(cnt[..., None] > 0, med, 0).astype(np.uint8)
    return med, cnt

def structure_from_background(med, cnt, lut, llr_min, min_n=None):
    mn = DESIGN["N_BG_MIN"] if min_n is None else min_n
    return calf_like(med, lut, llr_min) & (cnt >= mn)

def static_from_hours(hour_meds, lut, llr_min):
    """STATIC at 1/SUB resolution: calf-like median in >= STATIC_HOUR_FRAC of the hours with >= STATIC_MIN_SAMPLES."""
    n_el = None; n_cl = None
    for med, cnt in hour_meds:
        el = cnt >= DESIGN["STATIC_MIN_SAMPLES"]; cl = calf_like(med, lut, llr_min) & el
        n_el = el.astype(np.int32) if n_el is None else n_el + el
        n_cl = cl.astype(np.int32) if n_cl is None else n_cl + cl
    if n_el is None: raise RuntimeError("STATIC: no hour medians")
    return (n_el > 0) & (n_cl >= DESIGN["STATIC_HOUR_FRAC"] * n_el)

def upsample_mask(m_sub, shape=(H, W), sub=None, dil=None):
    k = DESIGN["SUB"] if sub is None else sub
    up = np.repeat(np.repeat(np.asarray(m_sub, bool), k, axis=0), k, axis=1)[:shape[0], :shape[1]]
    if up.shape != tuple(shape):
        pad = np.zeros(shape, bool); pad[:up.shape[0], :up.shape[1]] = up; up = pad
    return dilate(up, DESIGN["STRUCT_DIL"] if dil is None else dil)

def grey(rgb):
    a = np.asarray(rgb, np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

def phase_shift(a, b):
    """integer (dy, dx) that best aligns b to a (phase correlation, Hann window). a, b: 2-D float."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    wy = np.hanning(a.shape[0])[:, None]; wx = np.hanning(a.shape[1])[None, :]
    A = np.fft.fft2((a - a.mean()) * wy * wx); Bf = np.fft.fft2((b - b.mean()) * wy * wx)
    Rr = A * np.conj(Bf); Rr /= np.abs(Rr) + 1e-12
    r = np.fft.ifft2(Rr).real
    dy, dx = np.unravel_index(int(np.argmax(r)), r.shape)
    if dy > a.shape[0] // 2: dy -= a.shape[0]
    if dx > a.shape[1] // 2: dx -= a.shape[1]
    return int(dy), int(dx)

def shift_mask(m, dy, dx):
    out = np.zeros_like(m)
    h, w = m.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = m[ys, xs]
    return out

def bg_shift_check(train_bg_sub, test_bg_sub, sub=None):
    """Phase-correlate the test day's first daylight background against the training one (at 1/SUB resolution).
    Returns (dy_px, dx_px, action) with action 'none' | 'shift' (> SHIFT_TOL_PX) | raises (> SHIFT_MAX_PX)."""
    k = DESIGN["SUB"] if sub is None else sub
    dy, dx = phase_shift(grey(train_bg_sub), grey(test_bg_sub))
    dy, dx = -dy * k, -dx * k                                   # shift that moves the training mask onto the test view
    m = max(abs(dy), abs(dx))
    if m > DESIGN["SHIFT_MAX_PX"]:
        raise RuntimeError(f"camera shift {dy},{dx} px > {DESIGN['SHIFT_MAX_PX']} px: STATIC is invalid for this day -- stop")
    return dy, dx, ("shift" if m > DESIGN["SHIFT_TOL_PX"] else "none")

# =====================================================================================================================
# §J  (B) colour grow
# =====================================================================================================================
def body_len(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 3: return 0.0
    c = np.cov(np.vstack((xs, ys)).astype(np.float64))
    return float(4.0 * math.sqrt(max(np.linalg.eigvalsh(c)[-1], 0.0)))

def solidity(mask):
    """pixel area / area of the convex hull of the boundary pixels' CORNERS (<= 1)."""
    from scipy.spatial import ConvexHull
    m = np.asarray(mask, bool)
    a = int(m.sum())
    if a < 3: return 0.0
    edge = m & ~ndi.binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)
    ys, xs = np.nonzero(edge)
    pts = np.concatenate([np.stack((xs + dx, ys + dy), 1) for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)]).astype(np.float64)
    try: hv = ConvexHull(pts).volume
    except Exception: return 0.0
    return float(a / hv) if hv > 0 else 0.0

def grow_row(rgb, Ms, others, dets, static, dyn, lut, llr_min, C, pool="dayl"):
    """(B) colour grow on one eligible row. rgb: HxWx3 uint8 frame; Ms: Runs of the fragment; others: Runs list (other slots'
    original AND final masks); dets: Runs list (every kept detection at f); static/dyn: full-res bool or None.
    Returns (G Runs | None, info)."""
    R = float(C["BODY_LEN_PX"]); Ri = int(math.ceil(R))
    y0, y1, x0, x1 = runs_bbox(Ms)
    Y0, Y1, X0, X1 = max(0, y0 - Ri), min(H, y1 + Ri), max(0, x0 - Ri), min(W, x1 + Ri)
    Mc = runs_to_crop(Ms, Y0, Y1, X0, X1)
    Oc = np.zeros_like(Mc); Dc = np.zeros_like(Mc)
    for O in others: Oc |= runs_to_crop(O, Y0, Y1, X0, X1)
    for Dd in dets: Dc |= runs_to_crop(Dd, Y0, Y1, X0, X1)
    sd = DESIGN["STRUCT_DIL"]
    F = dilate(Oc, sd) | dilate(Dc, sd)
    if static is not None: F |= static[Y0:Y1, X0:X1]
    if dyn is not None: F |= dyn[Y0:Y1, X0:X1]
    cl = calf_like(rgb[Y0:Y1, X0:X1], lut, llr_min)
    allowed = (cl & ~F) | Mc
    within = ndi.distance_transform_edt(~Mc) <= R
    lab, nl = ndi.label(allowed & within, structure=np.ones((3, 3), int))
    keep = np.unique(lab[Mc & (lab > 0)])
    G = np.isin(lab, keep[keep > 0])
    cr = DESIGN["CLOSE_R"]
    G = (erode(dilate(G, cr), cr) & ~F) | Mc
    G = ndi.binary_fill_holes(G)
    G = (G & ~F) | Mc                                          # a hole may enclose STATIC / DYN / a detection / another
    G &= ~Oc                                                   # slot: never painted (only the fragment's own pixels stay)
                                                               # other slots win
    a = int(G.sum())
    info = {"crop": [int(Y0), int(Y1), int(X0), int(X1)], "grown_area": a, "frag_area": int(Mc.sum()),
            "calf_like_frac_crop": round(float(cl.mean()), 4), "forbidden_frac_crop": round(float(F.mean()), 4)}
    lo, hi = trig_lo_px("dayl", C), float(C["OVER_HI"]) * C["A_REF"]
    if a < lo: info["why"] = "grow_short"; return None, info
    if a > hi: info["why"] = "grow_leak"; return None, info
    bl = body_len(G); so = solidity(G)
    info["body_len"] = round(bl, 1); info["solidity"] = round(so, 4)
    if bl > float(C["BL_P99"]) or so < float(C["SOL_P1"]): info["why"] = "grow_shape"; return None, info
    Gr = runs_from_crop(G, Y0, X0)
    if cover(Ms, Gr) < float(C["TAU_CONTAIN"][pool]): info["why"] = "grow_shape"; info["contain"] = False; return None, info
    info["why"] = None
    return Gr, info

def grow_eligible(rec):
    return rec["code"] == "r" and rec.get("reason") == "nodet" and "reason0" not in rec

def dyn_structure(frames_rgb, excl_list, lut, llr_min):
    """DYN(clip) at 1/SUB resolution from <= N_BG_FRAMES day colour frames; excl_list: full-res bool exclusion masks."""
    k = DESIGN["SUB"]; r = math.ceil(DESIGN["BG_EXCL_DIL"] / k)
    F = np.stack([subsample(f, k) for f in frames_rgb]); E = np.stack([dilate(subsample(e, k), r) for e in excl_list])
    med, cnt = masked_median(F, E)
    return structure_from_background(med, cnt, lut, llr_min), med, cnt

# =====================================================================================================================
# §K  derive statistics (training days only)
# =====================================================================================================================
def parse_train_relabel(J):
    """A training relabel_week_<window>.json -> (rules {slot: [(t0, t1)]}, hidden [(slot, t0, t1)], exclusions
    [(slots set, t0, t1)]) in epoch seconds. exclusions = merge episodes where identity is undefined (Tol_17 renders them
    grey): a frame inside one is never trusted. Unknown entry shapes raise (never silently ignored)."""
    tsx = lambda s: ts_of(datetime.fromisoformat(s))
    R = {int(k): [(tsx(a) if a else -1e18, tsx(b) if b else 1e18) for a, b, _c in v] for k, v in (J.get("rules") or {}).items()}
    hid = [(int(h["slot"]), tsx(h["from"]), tsx(h["to"])) for h in (J.get("hidden") or [])]
    exc = []
    for e in (J.get("exclusions") or []):
        if not isinstance(e, dict) or not {"from", "to", "slots"} <= set(e):
            raise RuntimeError(f"relabel exclusion entry of unknown shape: {e!r}")
        exc.append((set(int(s) for s in e["slots"]), tsx(e["from"]), tsx(e["to"])))
    return R, hid, exc

def relabel_trusted(R, hid, exc, ts, slots=range(N_SLOTS)):
    """every slot is assigned by a rule at ts, none is hidden ([from, to] closed) and no exclusion covers ts."""
    if any(a <= ts < b for _ss, a, b in exc): return False
    for s in slots:
        if not any(a <= ts < b for a, b in R.get(s, [])): return False
        if any(hs == s and a <= ts <= b for hs, a, b in hid): return False
    return True

def smoke_clip_subset(clips, k, pool_of_clip):
    """SMOKE derive: k clips that exercise every derivation path -- round-robin over the groups (unseen, dayl),
    (valid, dark), (train, dayl), (unseen, dark), (valid, dayl), (train, dark), earliest clip first in each group.
    clips: [(window, stem, role)]; pool_of_clip(stem) -> 'dayl' | 'dark'. The first k clips by time are all TRAIN
    (04-18 09:27-11:33), which have no detector frames, so a time-ordered smoke could never reach the constants."""
    order = [("unseen", "dayl"), ("valid", "dark"), ("train", "dayl"), ("unseen", "dark"), ("valid", "dayl"), ("train", "dark")]
    groups = {g: [c for c in clips if (c[2], pool_of_clip(c[1])) == g] for g in order}
    out = []
    while len(out) < k and any(groups.values()):
        for g in order:
            if groups[g] and len(out) < k: out.append(groups[g].pop(0))
    return sorted(out, key=lambda z: clip_start(z[1]))

def trusted_frame(M, pool, C):
    """Trusted teacher frame (the per-mask fault / relabel checks are applied by the caller)."""
    if any(m is None for m in M) or len(M) != N_SLOTS: return False
    lo, hi = px(C, "FRAG_LO", pool), float(C["OVER_HI"]) * C["A_REF"]
    if any(not lo <= m.area <= hi for m in M): return False
    for a, b in itertools.combinations(range(N_SLOTS), 2):
        if ios_r(M[a], M[b]) >= DESIGN["TRUST_PAIR_IOS"]: return False
    return True

def hungarian(T, D, min_iou=None):
    """{t_index: d_index} maximising IoU, pairs with IoU >= MATCH_IOU only."""
    mi = DESIGN["MATCH_IOU"] if min_iou is None else min_iou
    if not T or not D: return {}
    A = np.array([[iou_r(t, d) for d in D] for t in T])
    ri, ci = linear_sum_assignment(-A)
    return {int(i): int(j) for i, j in zip(ri, ci) if A[i, j] >= mi}

def signed_offset(T_crop, d_crop):
    """median over d's contour pixels of the signed pixel offset from T's contour: + = outside T (EDT to the nearest
    T pixel), 0 on T's own boundary pixels, - = inside (1 - EDT to the nearest background pixel). d = T gives 0."""
    if not T_crop.any() or not d_crop.any(): return None
    din = ndi.distance_transform_edt(T_crop); dout = ndi.distance_transform_edt(~T_crop)
    sgn = np.where(T_crop, -(din - 1.0), dout)
    edge = d_crop & ~ndi.binary_erosion(d_crop, structure=np.ones((3, 3), bool), border_value=0)
    v = sgn[edge]
    return float(np.median(v)) if v.size else None

def synth_fragments(Tc):
    """The 10 body-part cuts of one teacher mask crop: PCA spine axis; perpendicular cuts keeping the head end /
    the rear end at area fractions SYN_FRACS; one lateral cut keeping each side at SYN_LATERAL. -> [(name, bool crop)]."""
    ys, xs = np.nonzero(Tc)
    if len(xs) < 50: return []
    P = np.stack((xs, ys), 1).astype(np.float64); P -= P.mean(0)
    w, V = np.linalg.eigh(np.cov(P.T))
    u, v = V[:, -1], V[:, 0]
    pu, pv = P @ u, P @ v
    out = []
    for f in DESIGN["SYN_FRACS"]:
        for name, sel in ((f"rear{f}", pu <= np.quantile(pu, f)), (f"head{f}", pu >= np.quantile(pu, 1.0 - f))):
            m = np.zeros_like(Tc); m[ys[sel], xs[sel]] = True; out.append((name, m))
    q = DESIGN["SYN_LATERAL"]
    for name, sel in (("latA", pv <= np.quantile(pv, q)), ("latB", pv >= np.quantile(pv, 1.0 - q))):
        m = np.zeros_like(Tc); m[ys[sel], xs[sel]] = True; out.append((name, m))
    return out

def bleed(m, k):
    k = int(max(-DESIGN["SYN_K_ABS_MAX"], min(DESIGN["SYN_K_ABS_MAX"], int(round(k)))))
    return dilate(m, k) if k > 0 else (erode(m, -k) if k < 0 else m)

def crop_box(Rs, margin, extra=()):
    bbs = [runs_bbox(x) for x in list(Rs) + list(extra) if x is not None and x.area]
    y0 = min(b[0] for b in bbs); y1 = max(b[1] for b in bbs); x0 = min(b[2] for b in bbs); x1 = max(b[3] for b in bbs)
    return max(0, y0 - margin), min(H, y1 + margin), max(0, x0 - margin), min(W, x1 + margin)

def syn_cover_samples(T, D, match, offsets, rng, trig_lo):
    """For each matched teacher mask: 10 synthetic fragments, each bled by a k drawn from the boundary-offset
    distribution; returns (cover(F, d*) list, cover(F, d_other) list). T, D: Runs lists; match: {t: d}."""
    cs, co = [], []
    kmax = DESIGN["SYN_K_ABS_MAX"]
    for ti, di in sorted(match.items()):
        y0, y1, x0, x1 = crop_box([T[ti]], kmax + 2)
        Tc = runs_to_crop(T[ti], y0, y1, x0, x1)
        dstar = runs_to_crop(D[di], y0, y1, x0, x1)
        dothers = [runs_to_crop(D[j], y0, y1, x0, x1) for j in range(len(D)) if j != di]
        for _name, F in synth_fragments(Tc):
            k = offsets[int(rng.integers(len(offsets)))] if len(offsets) else 0.0
            Fb = bleed(F, k)
            a = int(Fb.sum())
            if a == 0 or a >= trig_lo: continue
            cs.append(float((Fb & dstar).sum()) / a)
            for dm in dothers:
                if dm.any(): co.append(float((Fb & dm).sum()) / a)
    return cs, co

def carry_window(iou_by_dt, q=None):
    """The largest Δ on the grid whose P5 of IoU(T_t, T_t+Δ) >= CARRY_Q (0.0 = carry off)."""
    qq = DESIGN["CARRY_Q"] if q is None else q
    best = 0.0; table = {}
    for dt in DESIGN["CARRY_GRID"]:
        v = iou_by_dt.get(dt) or []
        p5 = pct(v, 5); table[str(dt)] = {"p5": None if p5 is None else round(p5, 4), "n": len(v)}
        if p5 is not None and p5 >= qq: best = dt
        else: break                                       # monotone requirement: the window is contiguous from 0
    return best, table

def ring_threshold(llr_vals_nonstatic, n_ring, lut_values, fp_max=None):
    """Smallest LUT threshold at which (# ring pixels calf-like and not structure) / (# ring pixels) <= fp_max."""
    fm = DESIGN["RING_FP_MAX"] if fp_max is None else fp_max
    v = np.sort(np.asarray(llr_vals_nonstatic, np.float64))
    for thr in np.unique(np.asarray(lut_values, np.float64)):
        n_ge = len(v) - int(np.searchsorted(v, thr, side="left"))
        if n_ring > 0 and n_ge / n_ring <= fm: return float(thr), n_ge / n_ring
    return float(np.max(lut_values)) + 1.0, 0.0

def b_gate_eval(rows):
    """(B) ship gate over synthetic-fragment grow rows: rows = [{accepted, iou_before, iou_after, false_fill_px,
    other_fill_px}]. Returns (pass, values, criteria)."""
    acc = [r for r in rows if r["accepted"]]
    A = 87137.0 if not rows else rows[0].get("a_ref", 87137.0)
    vals = {"n_rows": len(rows), "n_accepted": len(acc), "accept_frac": round(len(acc) / len(rows), 4) if rows else None}
    crit = dict(ACCEPT_B)
    if not acc:
        return False, dict(vals, note="no accepted grow row"), crit
    d = np.array([r["iou_after"] - r["iou_before"] for r in acc])
    ff = np.array([r["false_fill_px"] for r in acc]); of = np.array([r["other_fill_px"] for r in acc])
    vals.update(d_iou_median=round(float(np.median(d)), 4), false_fill_p95_px=round(float(np.percentile(ff, 95)), 1),
                other_fill_frac=round(float((of >= ACCEPT_B["other_fill_px_frac"] * A).mean()), 5),
                harm_frac=round(float((d < 0).mean()), 5))
    ok = (vals["d_iou_median"] >= ACCEPT_B["d_iou_median_min"]
          and vals["false_fill_p95_px"] <= ACCEPT_B["false_fill_p95_frac_max"] * A
          and vals["other_fill_frac"] <= ACCEPT_B["other_fill_frac_max"]
          and vals["harm_frac"] <= ACCEPT_B["harm_frac_max"])
    return bool(ok), vals, crit

def activity_scores(cent_rows, half_s=5.0):
    """Tol_42 add_temporal activity: mean centroid displacement between t-5 s and t+5 s over slots present in both.
    cent_rows: [(t, [centroid|None]*4)] 1-Hz rows of one clip -> list of act|None."""
    ts = np.array([t for t, _ in cent_rows], float); out = []
    for k, (t, _c) in enumerate(cent_rows):
        jm = int(np.searchsorted(ts, t - half_s)); jp = int(np.searchsorted(ts, t + half_s))
        act = None
        if jm < len(ts) and jp < len(ts) and abs(ts[jm] - (t - half_s)) <= 1.0 and abs(ts[jp] - (t + half_s)) <= 1.0:
            d = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(cent_rows[jm][1], cent_rows[jp][1])
                 if a is not None and b is not None]
            if d: act = float(np.mean(d))
        out.append(act)
    return out

def centroid_runs(R):
    """(x, y) centroid of a Runs mask (column-major: x = index // h, y = index % h), exact."""
    if R is None or R.area == 0: return None
    h = R.h; sx = 0.0; sy = 0.0
    s, e = R.s, R.e
    c0, c1 = s // h, (e - 1) // h
    one = c0 == c1
    if one.any():
        ss, ee = s[one], e[one]; ln = (ee - ss).astype(np.float64)
        sx += float(((ss // h) * ln).sum()); sy += float((((ss % h) + ((ee - 1) % h)) * ln / 2.0).sum())
    for a, b in zip(s[~one], e[~one]):                      # rare: runs that wrap a column
        for p0 in range(int(a), int(b), 1):
            sx += p0 // h; sy += p0 % h
    return sx / R.area, sy / R.area

# =====================================================================================================================
# §L  selftest
# =====================================================================================================================
class _Check:
    def __init__(self): self.n = 0; self.fail = []
    def __call__(self, ok, msg):
        self.n += 1
        if not ok: self.fail.append(msg); print(f"  FAIL: {msg}")

def _ellipse(cy, cx, ry, rx, ang=0.0, shape=(H, W)):
    y, x = np.ogrid[:shape[0], :shape[1]]
    ca, sa = math.cos(ang), math.sin(ang)
    u = (x - cx) * ca + (y - cy) * sa; v = -(x - cx) * sa + (y - cy) * ca
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0

def selftest_constants():
    """READ-like + DERIVED-like constants for synthetic scenes (values in the documented bands, NOT derived)."""
    C = {"A_REF": 87137.0, "FRAG_LO": {"dayl": 0.44, "dark": 0.43}, "OVER_HI": 1.32,
         "TRIG_LO": {"dayl": 0.68, "dark": 0.65}, "LINK_IOU_MIN": 0.5, "BODY_LEN_PX": 467.6,
         "DISJ_IOS": 0.05, "VETO_IOS": 0.30, "TAU_CONTAIN": {"dayl": 0.80, "dark": 0.80},
         "TAU_CONFLICT": {"dayl": 0.15, "dark": 0.15}, "CARRY_MAX_S": 1.0, "CHROMA_MIN": 30.0, "LLR_MIN": 0.5,
         "BL_P99": 700.0, "SOL_P1": 0.70, "op_threshold": 0.35, "resolution": 624}
    return C

def _scene():
    """4 synthetic calves (~84-90k px ellipses), one per slot."""
    calves = [_ellipse(300, 400, 140, 200, 0.2), _ellipse(320, 1200, 130, 210, -0.3),
              _ellipse(900, 600, 135, 205, 1.2), _ellipse(950, 1700, 140, 200, 0.6)]
    return calves

def _st_runs(ck, rng):
    shapes = [(37, 53), (H, W)]
    for (h, w) in shapes:
        for trial in range(30 if h < 100 else 4):
            if h < 100:
                a = rng.random((h, w)) < rng.random(); b = rng.random((h, w)) < rng.random()
                if trial == 0: a[:] = False
                if trial == 1: a[:] = True
                if trial == 2: a[0, 0] = True; a[-1, -1] = True
            else:
                a = _ellipse(500, 900, 200, 300, 0.4); b = _ellipse(600, 1000, 150, 260, -0.2)
                if trial == 1: a = np.zeros((h, w), bool)
                if trial == 2: a[:, 5] = True                      # a run spanning a full column
                if trial == 3: b = np.ones((h, w), bool)
            ra, rb = V5.rle_encode(a), V5.rle_encode(b)
            A, B = runs_of(ra), runs_of(rb)
            ck(rle_from_runs(A, (h, w)) == ra, f"rle_from_runs != rle_encode ({h}x{w}, trial {trial})")
            ck(np.array_equal(runs_to_dense(A, (h, w)), a), "runs_to_dense round trip")
            for nm, op, dn in (("and", r_and, a & b), ("or", r_or, a | b), ("sub", r_sub, a & ~b)):
                X = op(A, B)
                ck(rle_from_runs(X, (h, w)) == V5.rle_encode(dn), f"set op {nm} != dense ({h}x{w}, trial {trial})")
            ck(inter(A, B) == int((a & b).sum()), "inter")
            ck(abs(cover(A, B) - (float((a & b).sum()) / a.sum() if a.sum() else 0.0)) < 1e-12, "cover")
            if a.any():
                bb = runs_bbox(A); ys, xs = np.nonzero(a)
                ck(bb == (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1), f"runs_bbox {bb}")
                cr = (max(0, bb[0] - 3), min(h, bb[1] + 2), max(0, bb[2] - 1), min(w, bb[3] + 4))
                crop = runs_to_crop(A, *cr, h=h)
                ck(np.array_equal(crop, a[cr[0]:cr[1], cr[2]:cr[3]]), "runs_to_crop")
                back = runs_from_crop(crop, cr[0], cr[2], h=h, w=w)
                ck(rle_from_runs(back, (h, w)) == ra, "runs_from_crop round trip")
                cx, cy = centroid_runs(A)
                ck(abs(cx - xs.mean()) < 1e-6 and abs(cy - ys.mean()) < 1e-6, "centroid_runs")
    ck(rle_from_runs(None) == {"size": [H, W], "counts": [N_PIX]}, "empty RLE")

def _det_rec(masks, fidx, t, pool, C, kept=None, kind="pass"):
    rows = [(V5.rle_encode(m), 0.9, int(m.sum())) for m in masks]
    if kept is None:
        kidx, _ = V5.post_chain_idx([(m, 0.9) for m in masks], topk=DET_TOPK)
    else:
        kidx = kept
    return det_record("x.mp4", fidx, t, rows, kidx, 10.0, pool, C, kind=kind)

def _st_gates(ck):
    C = selftest_constants(); calves = _scene(); pool = "dayl"
    Mfull = [runs_of(V5.rle_encode(m)) for m in calves]
    # fragment = front 30 % of calf 0 (body-part lock)
    ys, xs = np.nonzero(calves[0]); frag0 = np.zeros((H, W), bool); sel = xs < np.quantile(xs, 0.3); frag0[ys[sel], xs[sel]] = True
    M = [runs_of(V5.rle_encode(frag0))] + Mfull[1:]
    D = [Mfull[0], Mfull[1], Mfull[2], Mfull[3]]
    dec = gate_frame(M, D, pool, C, [0])
    ck(dec[0]["code"] == "d" and dec[0]["det_idx"] == 0 and dec[0]["R"].area == Mfull[0].area, f"body-part lock -> d: {dec[0]}")
    # pure detection minus other slots' pixels: make slot 1 overlap calf 0's detection slightly
    m1 = calves[1] | _ellipse(300, 610, 30, 30); M2 = [M[0], runs_of(V5.rle_encode(m1)), M[2], M[3]]
    dec = gate_frame(M2, D, pool, C, [0])
    ck(dec[0]["code"] == "d" and inter(dec[0]["R"], M2[1]) == 0, "R_s excludes other slots' original pixels")
    # no detection on the fragment -> nodet ; detection covering half -> straddle
    dec = gate_frame(M, D[1:], pool, C, [0]); ck(dec[0]["reason"] == "nodet", f"nodet: {dec[0]['reason']}")
    half = np.zeros((H, W), bool); hs = xs < np.quantile(xs, 0.15); half[ys[hs], xs[hs]] = True
    dec = gate_frame(M, [runs_of(V5.rle_encode(half | calves[1]))], pool, C, [0])
    ck(dec[0]["reason"] in ("straddle", "big"), f"straddle: {dec[0]['reason']}")
    dec = gate_frame(M, [runs_of(V5.rle_encode(half))] + D[1:], pool, C, [0]); ck(dec[0]["reason"] == "straddle", f"straddle2 {dec[0]['reason']}")
    # small (edge-cut calf: the only containing detection is small)
    small = _ellipse(300, 330, 100, 110) & calves[0]
    fr_s = np.zeros((H, W), bool); ys2, xs2 = np.nonzero(small); s2 = xs2 < np.quantile(xs2, 0.5); fr_s[ys2[s2], xs2[s2]] = True
    dec = gate_frame([runs_of(V5.rle_encode(fr_s))] + Mfull[1:], [runs_of(V5.rle_encode(small))] + D[1:], pool, C, [0])
    ck(dec[0]["reason"] == "small", f"small: {dec[0]['reason']}")
    # big (merged blob of two calves)
    dec = gate_frame(M, [runs_of(V5.rle_encode(calves[0] | calves[1]))] + D[2:], pool, C, [0])
    ck(dec[0]["reason"] == "big", f"big: {dec[0]['reason']}")
    # ambiguous: two kept detections both contain the fragment
    dup = calves[0] | _ellipse(300, 150, 60, 60)
    dec = gate_frame(M, [Mfull[0], runs_of(V5.rle_encode(dup))] + D[1:], pool, C, [0])
    ck(dec[0]["reason"] == "ambiguous", f"ambiguous: {dec[0]['reason']}")
    # conflict (two identities on one cow): slot 1's mask = the rear of calf 0; slot 0 = front fragment
    rear = np.zeros((H, W), bool); sr = xs > np.quantile(xs, 0.6); rear[ys[sr], xs[sr]] = True
    Mc = [M[0], runs_of(V5.rle_encode(rear)), Mfull[2], Mfull[3]]
    dec = gate_frame(Mc, D, pool, C, [0, 1])
    ck(dec[0]["reason"] == "conflict" and dec[1]["reason"] == "conflict", f"double claim -> conflict: {dec[0]['reason']} {dec[1]['reason']}")
    dcl, _orp = claims_frame(Mc, D, pool, C)
    ck(any(sorted(z["slots"]) == [0, 1] for z in dcl), "double-claim row recorded")
    _dcl2, orp = claims_frame(Mfull[:3] + [None], D, pool, C)
    ck(len(orp) == 1 and orp[0]["det_idx"] == 3, "orphan detection on an unmasked calf")
    # NO_SHRINK: a near-healthy trigger mask (0.66 A_ref) whose only containing detection is SMALLER -> refused 'shrink'
    m66 = _ellipse(950, 1700, 115, 160, 0.6); d60 = m66 & _ellipse(950, 1690, 112, 150, 0.6)
    Ms = [Mfull[0], Mfull[1], Mfull[2], runs_of(V5.rle_encode(m66))]
    dec = gate_frame(Ms, [Mfull[0], Mfull[1], Mfull[2], runs_of(V5.rle_encode(d60))], pool, C, [3])
    ck(dec[3]["code"] == "r" and dec[3]["reason"] == "shrink", f"no_shrink: {dec[3]['reason']} ({Ms[3].area} -> {int(d60.sum())})")
    dec = gate_frame(Ms, [Mfull[0], Mfull[1], Mfull[2], runs_of(V5.rle_encode(d60))], pool, dict(C, NO_SHRINK=False), [3])
    ck(dec[3]["code"] == "d", f"no_shrink off = v1 rule: {dec[3]['reason']}")
    # post_shrink: the other slots' originals eat most of the detection
    big1 = runs_of(V5.rle_encode(calves[1] | (calves[0] & ~frag0 & (np.arange(W)[None, :] < 560))))
    dec = gate_frame([M[0], big1, M[2], M[3]], D, pool, C, [0])
    ck(dec[0]["code"] == "r" and dec[0]["reason"] in ("conflict", "post_shrink"), f"eaten detection refused: {dec[0]['reason']}")
    return C

def _mk_src(clip, frames_masks, t0=0.0, fps=5.0):
    frames = []
    for i, ms in enumerate(frames_masks):
        frames.append({"frame_idx": i * 5, "t_sec": round(t0 + i / fps, 2),
                       "masks": {str(s): (V5.rle_encode(m) if m is not None else None) for s, m in enumerate(ms)}})
    return {"clip": clip, "track_fps": fps, "mask_format": "rle_column_major", "keyed_by": "slot",
            "backbone": "student", "sam3_model": "facebook/sam3", "frames": frames, "smoke": False, "v5": {"x": 1}}

def _st_apply(ck):
    C = selftest_constants(); calves = _scene()
    clip = "Tol_ch01-20240422-140000-143000.mp4"
    ys, xs = np.nonzero(calves[0])
    frag = np.zeros((H, W), bool); sel = xs < np.quantile(xs, 0.3); frag[ys[sel], xs[sel]] = True
    seq = []
    for i in range(30):
        ms = [frag if 5 <= i < 25 else calves[0], calves[1], calves[2], calves[3]]
        seq.append(ms)
    src = _mk_src(clip, seq)
    segs = {"segments": [{"seg_id": f"{s}_001", "slot": s, "t_start_wall": "2024-04-22T14:00:00",
                          "t_end_wall": "2024-04-22T14:00:03.000000"} for s in range(4)]
                       + [{"seg_id": "0_002", "slot": 0, "t_start_wall": "2024-04-22T14:00:03.100000",
                           "t_end_wall": "2024-04-22T14:00:06"}]}
    SI = SegIndex(segs)
    stored = {}; new = {}
    for i in range(5, 25):
        f = i * 5; t = round(i / 5.0, 2)
        if i == 10: stored[f] = _det_rec(calves, f, t, "dayl", C, kind="pass")
        elif i in (12, 13): new[f] = _det_rec(calves[1:], f, t, "dayl", C)            # detector misses calf 0 -> nodet
        elif i == 17: new[f] = _det_rec([calves[0] | calves[1]] + calves[2:], f, t, "dayl", C)   # merged -> big
        elif i == 20: new[f] = det_record(clip, f, t, [], [], 0.0, "dayl", C, decode_ok=False)
        else: new[f] = _det_rec(calves, f, t, "dayl", C)
    out, side, extra = apply_clip(src, stored, new, SI, C)
    n = verify_clip_output(src, out, side, C)
    ck(n == 20 and len(side) == 20, f"one sidecar record per trigger row ({n}, {len(side)})")
    code = {r["frame_idx"] // 5: (r["code"], r["reason"], r.get("reason0")) for r in side}
    ck(code[10][0] == "d", f"stored call reused at frame 10: {code[10]}")
    ck(code[6][0] == "d" and code[12][0] == "c" and code[13][0] == "c", f"carry bridges nodet: {code[12]} {code[13]}")
    ck(code[17][0] == "r" and code[17][1] == "big", f"big is never carried: {code[17]}")
    ck(code[20][0] == "c" and code[20][2] == "decode_fail", f"decode_fail carried: {code[20]}")
    ck(all(code[i][0] == "d" for i in (5, 9, 11, 14, 16, 18, 24)), "d rows")
    # carry never crosses a segment boundary: frame 15 (t=3.0) is the last row of seg 0_001, frame 16 opens 0_002.
    # With frames 12-15 nodet, frame 15's nearest d row is frame 16 (0.2 s, other segment) -> it must take frame 11.
    new2 = {f: (_det_rec(calves[1:], f, f / 25.0, "dayl", C) if f in (60, 65, 70, 75) else r) for f, r in new.items()}
    out2, side2, _ = apply_clip(src, stored, new2, SI, C)
    verify_clip_output(src, out2, side2, C)
    c15 = [r for r in side2 if r["frame_idx"] == 75][0]; c14 = [r for r in side2 if r["frame_idx"] == 70][0]
    ck(c15["code"] == "c" and c15["carry_src_fidx"] == 55, f"carry stays in its segment: {c15}")
    ck(c14["code"] == "c" and c14["carry_src_fidx"] == 55, f"carry nearest same-segment source: {c14}")
    # prov and byte identity of untouched rows
    ck(all(fr["prov"]["1"] == "k" for fr in out["frames"]), "untouched slots coded k")
    ck(out["frames"][0]["masks"] == src["frames"][0]["masks"], "non-trigger frame byte-identical")
    ck(json.dumps(out["frames"][17]["masks"]["0"]) == json.dumps(src["frames"][17]["masks"]["0"]), "refused row byte-identical")
    # FrameStream reads the refined file (header before frames + tail keys after)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "masks_x.json.gz")
        out["refine"] = {"spec": REFINE_SPEC}
        dump_json_gz(p, out)
        fsr = FrameStream(p); got = list(fsr)
        ck(len(got) == 30 and fsr.header.get("refine", {}).get("spec") == REFINE_SPEC and fsr.header.get("clip") == clip,
           "FrameStream on the refined file (header + tail)")
    # a trigger row without any call inside the sample must raise
    try:
        apply_clip(src, stored, {k: v for k, v in new.items() if k != 30}, SI, C); ck(False, "missing call accepted")
    except RuntimeError:
        pass
    # sampling: render cadence and unsampled rows carried
    sel = sample_selection(src["frames"], "render")
    ck(sel == set(range(0, 30, 10)), f"render selection {sorted(sel)}")
    new_s = {f: r for f, r in new.items() if (f // 5) % 5 == 0}
    out3, side3, _ = apply_clip(src, stored, new_s, SI, C, sample_sel=set(range(0, 30, 5)))
    verify_clip_output(src, out3, side3, C)
    c3 = Counter(r["code"] for r in side3)
    # rows 16-24 sit in segment 0_002, whose only call (frame 20) failed to decode: no carry source there
    ck(c3 == Counter({"d": 3, "c": 8, "r": 9}) and all(r["reason"] in ("unsampled", "decode_fail") for r in side3 if r["code"] == "r"),
       f"1-in-5 sample + carry: {dict(c3)}")
    # verify catches a forged change
    bad = json.loads(json.dumps(out)); bad["frames"][2]["masks"]["1"] = bad["frames"][2]["masks"]["2"]
    try:
        verify_clip_output(src, bad, side, C); ck(False, "verify missed an I3 violation")
    except AssertionError:
        pass
    # double-claim intervals
    rows = [{"ts": 100.0 + 0.2 * i, "slots": [2, 3], "clip": "c"} for i in range(40)] + [{"ts": 300.0, "slots": [0, 1], "clip": "c"}]
    iv = doubleclaim_intervals(rows)
    ck(len(iv) == 1 and iv[0]["slots"] == [2, 3] and abs(iv[0]["dur_s"] - 7.8) < 1e-6, f"doubleclaim intervals {iv}")
    # census
    cen = census_clip(src["frames"], clip, C, stored_fidx=stored.keys())
    ck(cen["trigger_rows"] == 20 and cen["trigger_frames_stored_call"] == 1 and cen["new_calls"] == 19, f"census {cen}")
    ck(plan_new_calls(src["frames"], clip, C, stored.keys(), None) == [i * 5 for i in range(5, 25) if i != 10], "call plan")

def _st_parity(ck):
    C = selftest_constants(); calves = _scene()
    a = _det_rec(calves, 0, 0.0, "dayl", C); b = json.loads(json.dumps(a))
    ck(parity_compare(a, b, True)["ok"], "parity identical")
    b["raw"][0]["score"] = 0.9001
    ck(not parity_compare(a, b, True)["ok"] and parity_compare(a, b, False)["ok"], "parity score tolerance by GPU class")
    c = json.loads(json.dumps(a)); c["kept_idx"] = c["kept_idx"][::-1]
    ck(not parity_compare(a, c, True)["ok"], "exact parity: kept_idx order")
    r = parity_compare(a, c, False)
    ck(r["ok"] and r["order_differs"], f"tolerance parity: a kept-order swap of the same set is equivalent {r}")
    # near-equal scores swap the score-ranked raw list on another GPU class: same masks, permuted indices
    sw = json.loads(json.dumps(a)); p = [1, 0, 2, 3]
    sw["raw"] = [a["raw"][j] for j in p]; sw["raw"][0]["score"] = round(sw["raw"][0]["score"] + 4e-4, 4)
    sw["kept_idx"] = [p.index(k) for k in a["kept_idx"]]
    ck(not parity_compare(a, sw, True)["ok"], "exact parity: a raw permutation fails")
    r = parity_compare(a, sw, False)
    ck(r["ok"] and r["order_differs"] and r["min_iou"] == 1.0, f"tolerance parity: raw permutation matched {r}")
    d = json.loads(json.dumps(sw)); d["kept_idx"] = d["kept_idx"][:-1]
    ck(not parity_compare(a, d, False)["ok"], "tolerance parity: a different kept set fails")
    e = json.loads(json.dumps(sw)); e["raw"][2]["score"] = round(e["raw"][2]["score"] + 0.002, 4)
    ck(not parity_compare(a, e, False)["ok"], "tolerance parity: |Δscore| > 1e-3 fails")
    g = json.loads(json.dumps(sw)); g["raw"][3]["rle"] = V5.rle_encode(dilate(calves[3], 12))
    ck(not parity_compare(a, g, False)["ok"], "tolerance parity: a mask with IoU < 0.995 fails")
    h = json.loads(json.dumps(sw)); h["raw"] = h["raw"][:3]; h["n_raw"] = 3; h["kept_idx"] = [k for k in h["kept_idx"] if k < 3]
    ck(not parity_compare(a, h, False)["ok"], "tolerance parity: n_raw differs")

def _st_colour_grow(ck, rng):
    C = selftest_constants(); calves = _scene()
    img = np.zeros((H, W, 3), np.uint8); img[...] = (190, 160, 90)              # yellowish straw
    img += rng.integers(0, 12, (H, W, 3), dtype=np.uint8)
    bar = np.zeros((H, W), bool); bar[:, 820:835] = True                        # a black pen bar
    img[bar] = (25, 25, 25)
    patt = (np.add.outer(np.arange(H) // 40, np.arange(W) // 40) % 2).astype(bool)
    for m in calves: img[m & patt] = (20, 20, 20); img[m & ~patt] = (235, 235, 235)
    pos = np.zeros((H, W), bool)
    for m in calves: pos |= erode(m, 3)
    neg = ~dilate(np.any(calves, axis=0), 15) & ~bar
    lut = build_lut(lut_hist(img, pos), lut_hist(img, neg))
    llr = 0.5
    ck(calf_like(img, lut, llr)[calves[0]].mean() > 0.95 and calf_like(img, lut, llr)[neg].mean() < 0.02, "LUT separates coat / straw")
    st = calf_like(img, lut, llr) & bar                                          # STATIC: the bar
    ys, xs = np.nonzero(calves[0])
    frag = np.zeros((H, W), bool); sel = xs < np.quantile(xs, 0.35); frag[ys[sel], xs[sel]] = True
    Ms = runs_of(V5.rle_encode(frag)); others = [runs_of(V5.rle_encode(m)) for m in calves[1:]]
    G, info = grow_row(img, Ms, others, [], dilate(st, 3), None, lut, llr, C)
    ck(G is not None, f"grow accepted on a clean scene: {info}")
    if G is not None:
        Gd = runs_to_dense(G)
        iou = float((Gd & calves[0]).sum()) / float((Gd | calves[0]).sum())
        ck(iou > 0.9 and not (Gd & bar).any(), f"grow IoU {iou:.3f}, bar painted {(Gd & bar).sum()}")
    # a STATIC blob fully enclosed by the grown calf (e.g. a drain under the lying calf) is a hole of G: hole filling
    # must not paint it (review 2026-10-02)
    hole = _ellipse(300, 470, 12, 12)
    ck(bool(np.array_equal(hole & calves[0] & ~frag, hole)), "selftest geometry: the static blob lies inside calf 0, outside the fragment")
    Gh, infoh = grow_row(img, Ms, others, [], dilate(st, 3) | hole, None, lut, llr, C)
    ck(Gh is not None and not (runs_to_dense(Gh) & hole).any(),
       f"grow never paints an enclosed STATIC blob: {None if Gh is None else int((runs_to_dense(Gh) & hole).sum())} px {infoh}")
    # a neighbouring unmasked calf adjacent to the fragment: blocked by its detection (dilated)
    img2 = img.copy(); nb = _ellipse(300, 700, 120, 150) & ~calves[0]; img2[nb & patt] = (20, 20, 20); img2[nb & ~patt] = (235, 235, 235)
    G2, info2 = grow_row(img2, Ms, others, [runs_of(V5.rle_encode(nb))], None, None, lut, llr, C)
    if G2 is not None: ck(not (runs_to_dense(G2) & nb).any(), "grow never paints a detected neighbour")
    G3, info3 = grow_row(img2, Ms, others, [], None, None, lut, llr, C)
    ck(G3 is None and info3["why"] in ("grow_leak", "grow_shape"), f"without the neighbour's detection the leak is refused: {info3}")
    # chroma: a greyscale frame has saturation ~0
    ch_col = chroma_of(img, np.any(calves, axis=0)); ch_ir = chroma_of(np.repeat(grey(img)[..., None].astype(np.uint8), 3, -1), np.any(calves, axis=0))
    ck(ch_col > 60 and ch_ir < 2, f"chroma colour {ch_col} vs IR {ch_ir}")
    k, eta = otsu([0] * 50 + [1] * 30 + [90] * 40 + [100] * 50)
    ck(1 <= k < 90 and eta > 0.9, f"otsu {k} {eta}")
    # masked median + STATIC + upsample; phase shift
    fr = np.stack([img[::2, ::2]] * 12); ex = np.zeros(fr.shape[:3], bool); ex[:6, :100, :100] = True
    med, cnt = masked_median(fr, ex)
    ck(np.array_equal(med, img[::2, ::2]) and cnt[0, 0] == 6 and cnt[-1, -1] == 12, "masked median")
    ck(static_from_hours([(med, cnt)], lut, llr)[:, 410:417].any(), "STATIC finds the bar")
    s_hi, s_lo = static_from_hours([(med, cnt)], lut, 2.0), static_from_hours([(med, cnt)], lut, -2.0)
    ck(bool((s_hi <= s_lo).all()) and s_lo.sum() >= s_hi.sum(), "STATIC is monotone: a lower LLR_MIN gives a superset")
    shifted = np.roll(np.roll(img[::2, ::2], 3, axis=0), -2, axis=1)
    dy, dx = phase_shift(grey(shifted), grey(img[::2, ::2]))
    ck((dy, dx) == (3, -2), f"phase shift {(dy, dx)}")
    ck(body_len(calves[0]) > 300 and 0.9 < solidity(calves[0]) <= 1.0, f"shape stats {body_len(calves[0])} {solidity(calves[0])}")

def _st_derive(ck, rng):
    C = selftest_constants(); calves = _scene()
    T = [runs_of(V5.rle_encode(m)) for m in calves]
    D = [runs_of(V5.rle_encode(dilate(m, 2))) for m in calves] + [runs_of(V5.rle_encode(_ellipse(1200, 2200, 40, 40)))]
    mt = hungarian(T, D); ck(mt == {0: 0, 1: 1, 2: 2, 3: 3}, f"hungarian {mt}")
    ck(trusted_frame(T, "dayl", C) and not trusted_frame(T[:3] + [None], "dayl", C), "trusted frame")
    y0, y1, x0, x1 = crop_box([T[0]], 20)
    off = signed_offset(runs_to_crop(T[0], y0, y1, x0, x1), runs_to_crop(D[0], y0, y1, x0, x1))
    off0 = signed_offset(runs_to_crop(T[0], y0, y1, x0, x1), runs_to_crop(T[0], y0, y1, x0, x1))
    ck(off is not None and 1.4 <= off <= 2.0 and off0 == 0.0, f"signed offset of a 2-px dilation {off}, identity {off0}")
    frs = synth_fragments(runs_to_crop(T[0], y0, y1, x0, x1))
    ck(len(frs) == 10 and all(0.15 < f.sum() / T[0].area < 0.55 for _n, f in frs), "10 synthetic fragments")
    cs, co = syn_cover_samples(T, D, mt, [2.0, 3.0], rng, trig_lo_px("dayl", C))
    ck(len(cs) == 40 and min(cs) > 0.9 and (max(co) if co else 0.0) < 0.05, f"synthetic covers {min(cs):.3f} other {max(co) if co else 0}")
    best, tab = carry_window({0.2: [0.95] * 100, 0.4: [0.9] * 100, 0.6: [0.8] * 100})
    ck(best == 0.4, f"carry window {best} {tab}")
    thr, share = ring_threshold(np.array([0.1, 0.2, 0.3, 2.0, 3.0]), 80, np.array([0.0, 0.5, 1.0, 2.5]))
    ck(thr == 0.5 and share == 0.025, f"ring threshold {thr} {share}")
    ok, vals, _ = b_gate_eval([{"accepted": True, "iou_before": 0.35, "iou_after": 0.9, "false_fill_px": 100, "other_fill_px": 0}] * 50)
    ck(ok, f"b gate pass {vals}")
    ok2, _v2, _ = b_gate_eval([{"accepted": True, "iou_before": 0.35, "iou_after": 0.3, "false_fill_px": 100, "other_fill_px": 0}] * 50)
    ck(not ok2, "b gate fails on harm")
    acts = activity_scores([(float(t), [(100.0 + 10 * t, 50.0)] * 4) for t in range(20)])
    ck(acts[0] is None and abs(acts[10] - 100.0) < 1e-9, f"activity {acts[10]}")
    ck(floor_2dp(0.8799999) == 0.87 and floor_2dp(0.88) == 0.88 and ceil_2dp(0.1201) == 0.13 and ceil_2dp(0.12) == 0.12, "2dp rounding")
    # training relabel parsing: rules / hidden / exclusions
    J = {"rules": {"0": [["2024-04-18T10:00:00", "2024-04-18T11:00:00", 2]], "1": [[None, None, 0]], "2": [[None, None, 1]],
                   "3": [[None, None, 3]]},
         "hidden": [{"slot": 2, "from": "2024-04-18T10:20:00", "to": "2024-04-18T10:21:00"}],
         "exclusions": [{"from": "2024-04-18T10:40:00", "to": "2024-04-18T10:41:00", "slots": [0, 1]}]}
    R_, hid, exc = parse_train_relabel(J)
    t = lambda s: ts_of(datetime.fromisoformat("2024-04-18T" + s))
    ck(relabel_trusted(R_, hid, exc, t("10:10:00")) and not relabel_trusted(R_, hid, exc, t("09:59:59"))
       and not relabel_trusted(R_, hid, exc, t("10:20:30")) and not relabel_trusted(R_, hid, exc, t("10:40:30"))
       and relabel_trusted(R_, hid, exc, t("10:41:00")), "relabel trust: rules, hidden (closed), exclusions")
    try:
        parse_train_relabel(dict(J, exclusions=[["x"]])); ck(False, "unknown exclusion shape accepted")
    except RuntimeError:
        pass
    # SMOKE derive subset covers unseen / valid / train and both pools
    cl = ([("w", f"Tol_ch01_20240418-{h:02d}0000-{h:02d}3000", "train") for h in (9, 10, 11)]
          + [("w", "Tol_ch01_20240418-121506-125711", "unseen"), ("w", "Tol_ch01_20240418-220618-224904", "valid"),
             ("w", "Tol_ch01_20240418-212330-220618", "unseen")])
    pc = lambda stem: V5.pool_of_wall(clip_start(stem) + timedelta(minutes=10))
    sub = smoke_clip_subset(cl, 3, pc)
    ck({r for _w, _s, r in sub} == {"train", "unseen", "valid"} and {pc(s) for _w, s, _r in sub} == {"dayl", "dark"}
       and sub == sorted(sub, key=lambda z: clip_start(z[1])), f"smoke derive subset {sub}")

def _st_files(ck):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = os.path.join(td, "a.npz"), os.path.join(td, "b.npz")
        s1 = save_npz_det(p1, lut=np.arange(10, dtype=np.float32), x=np.ones(3)); time.sleep(1.1)
        s2 = save_npz_det(p2, x=np.ones(3), lut=np.arange(10, dtype=np.float32))
        ck(s1 == s2 and np.array_equal(np.load(p1)["lut"], np.arange(10)), "deterministic npz")
        pf = os.path.join(td, "c.json")
        ck(write_bytes_frozen(pf, b"abc") and not write_bytes_frozen(pf, b"abc"), "frozen write-if-absent")
        try:
            write_bytes_frozen(pf, b"abd"); ck(False, "frozen overwrite allowed")
        except RuntimeError:
            pass
        b1 = png_bytes(np.eye(5, dtype=bool)); b2 = png_bytes(np.eye(5, dtype=bool)); ck(b1 == b2, "deterministic png")
    try:
        assert_not_test_path("/x/Tracking_v5a10/20240423/masks.json.gz"); ck(False, "test-day path guard")
    except RuntimeError:
        pass
    ck(assert_not_test_path("/x/Tracking_train/20240418_w0927-1051/m.json.gz").endswith("m.json.gz"), "train path allowed")

def _st_read_constants(ck, ts_path):
    if not ts_path or not os.path.exists(ts_path):
        print("  (train_summary not given: READ-constant checks SKIPPED)"); return
    ts = load_json(ts_path)
    R = read_constants(ts)
    ck(R["TRIG_LO"] == {"dayl": 0.68, "dark": 0.65} and R["A_REF"] == 87137.0 and R["BODY_LEN_PX"] == 467.6
       and R["LINK_IOU_MIN"] == 0.5 and R["DISJ_IOS"] == 0.05 and R["VETO_IOS"] == 0.30, f"READ constants {R['TRIG_LO']}")
    ck(len(R["train_clips"]) == 18 and len(R["valid_clips"]) == 4, "train / valid clip lists")
    bad = json.loads(json.dumps(ts)); bad["constants"]["S_lo[dayl]"]["derived_from"] = ["20240422"]
    try:
        read_constants(bad); ck(False, "READ provenance guard")
    except RuntimeError:
        pass
    lo, hi = trig_lo_px("dayl", R), trig_lo_px("dark", R)
    ck(round(lo) == 59253 and round(hi) == 56639, f"TRIG_LO px {lo:.0f} {hi:.0f} (expected 59,253 / 56,639)")

def selftest(ts_path=None, quick=False):
    t0 = time.time()
    ck = _Check(); rng = np.random.default_rng(0)
    print(f"tol_refine_core {REFINE_CORE_VERSION} selftest (cv2 {'present' if _cv2 is not None else 'absent: scipy morphology'})")
    for name, fn in (("runs", lambda: _st_runs(ck, rng)), ("gates", lambda: _st_gates(ck)), ("apply", lambda: _st_apply(ck)),
                     ("parity", lambda: _st_parity(ck)), ("files", lambda: _st_files(ck)),
                     ("read", lambda: _st_read_constants(ck, ts_path)), ("derive", lambda: _st_derive(ck, rng)),
                     ("colour_grow", lambda: _st_colour_grow(ck, rng))):
        t1 = time.time(); fn(); print(f"  {name}: done in {time.time() - t1:.1f} s")
    print(f"tol_refine_core selftest: {ck.n} checks, {len(ck.fail)} failed, {time.time() - t0:.1f} s")
    if ck.fail: raise SystemExit("SELFTEST FAILED:\n  " + "\n  ".join(ck.fail))
    return True

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train_summary", default=None)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.selftest: selftest(a.train_summary)
