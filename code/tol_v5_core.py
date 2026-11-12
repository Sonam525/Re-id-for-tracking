#!/usr/bin/env python
"""tol_v5_core -- the single source of truth for every DECISION tracker v5 adds (design revision 2026-09-24).

Plain module: numpy, scipy and the standard library only (no torch, no cv2), so the tracker (`Tol_47_track_v5`), the
scorer and the local SELFTEST all run the identical gate code. Deployed to the Workspace as a FILE (`--format RAW`);
`Tol_47` imports it and echoes its sha256.

Contents
  * verbatim copies of Tol_03_track_v4.py:280-303 (rle_encode / rle_decode / m_iou / m_ios) -- test references
  * C7  exact fast twins: MaskCache (per-frame, strong refs), m_iou_fast / m_ios_fast / area_fast / rle_fast
  * C3  post_chain (verbatim Tol_44:702-711) + post_chain_idx (same order, returns indices)
  * phases, pools, the TRAINING-DAY calf band (read + provenance-asserted from train_summary.json)
  * C10 strikes_to_close / pending_window
  * C1  widget table, config_from_flags, gpu_class, resume_guard, ROOT_CONFIG record/check
  * C3/C5/C8 precision toggle, pop-frames forward, n_pf, decode iterator (seek / seek_thread / sequential)
  * C11-C13 slot_actionable, dark_step_decide (pure), reopen_filter, seed_filter, refresh_ok, v4_step4_opens
    (dry run for the young-OCC deferral), row_overrides, events_for_call
  * C14 event builders + schema, v4 projection; C15 E6 comparator (masks / segments / projected events + coverage)
  * C1  source_norm_sha256 / source_shas / read_json_retry / write_json_if_absent
  * C6/C14 E1_MEM + e1_memory_verdict (the ONE definition of the E1 memory criteria), resume_telemetry, publish_file
  * selftest(): S1-S18 (S5a/S5b need --data; printed SKIPPED otherwise; acceptance requires one run WITH data)

CLI:  python tol_v5_core.py --selftest [--data DIR] [--quick] [--report JSON]
"""
from __future__ import annotations

import os, sys, json, gzip, glob, math, re, time, hashlib, threading, queue, argparse, shutil
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from fractions import Fraction

import numpy as np
from scipy.optimize import linear_sum_assignment

CORE_VERSION = "v5.0.2 2026-09-24"   # .1 = finisher r1: source digest, atomic ROOT_CONFIG, E6 obs/new-branch, S17
# .2 = fixer r1: raw `source` + guarded `source_norm` shas, E1_MEM / e1_memory_verdict, reopen veto flags (the v4
#      unmatched loop sees every detection), resume_telemetry / publish_file, S18

# =====================================================================================================================
# Constants. Provenance tags: [code] existing constant, [train] 04-18/19 train_summary.json,
# [binding] design decision, [definition] 2026-09-23 investigation, [design] D2-D4, [runtime] tripwire only.
# NO constant here is fitted on 04-22..28; the 04-23-derived REPAIR_* reuse is flagged in-sample.
# =====================================================================================================================
N_CALVES = 4                          # [code] Tol_03:136
ST_TRACK, ST_OCC, ST_CLOSED = "TRACKING", "OCCLUDED", "CLOSED"   # [code] Tol_03:231
FRESH_SEC = 2.0                       # [code] Tol_03:138
MIN_MASK_PX = 8000                    # [code] Tol_03:225 (incumbent post-chain floor)
DEDUPE_IOS = 0.50                     # [code] Tol_03:227
DET_TOPK = N_CALVES + 2               # [code] Tol_03:727 / :1167
ASSOC_MIN = 0.30                      # [code] Tol_03:144
REFRESH_IOU = 0.60                    # [code] Tol_03:147
NIGHT_STABLE_IOU = 0.50               # [code] Tol_03:229
PENDING_FACTOR = 2.5                  # [code] Tol_03:764 (pending window = 2.5 x current interval, UNCHANGED)
STRIKES_V4 = {"day": (2, 120.0), "night": (3, 300.0)}   # [code] Tol_03:146 with its cadences Tol_03:86-87
# ---- constants of the new logic ----
LINK_CONTAIN_IOS = 0.60               # [binding] = REPAIR_CONTAIN_IOS [code Tol_03:173; set 2026-09-02 from the 04-23 diagnosis]
LINK_CONFLICT_IOS = 0.10              # [binding] = REPAIR_CONFLICT_IOS [code Tol_03:181; 04-23-derived]
LINK_MAX_SEC = 60.0                   # = SEG_BREAK_ABSENCE [code Tol_03:140]
UNCLAIMED_MAX = 0.30                  # = ASSOC_MIN [code Tol_03:144]
COVER_IOS = 0.50                      # [definition] 2026-09-23 gaps investigation; numerically = DEDUPE_IOS [code Tol_03:227]
MERGE_GUARD_SEC = 5.0                 # = Tol_30 MERGE_GUARD [code Tol_30:68]
DARK_MAX_PER_CLIP = 6                 # per slot, links + opens = REPAIR_MAX_PER_CLIP [code Tol_03:184]
DARK_MIN_GAP_SEC = 120.0              # per slot between dark-step actions = REPAIR_MIN_SEC [code Tol_03:183]
DARK_REPEAT_SEC_DEFAULT = 10.0        # [design] D3 (widget dark_repeat_sec); DARK_FIRST_SEC = DARK_REPEAT_SEC
LINK_CAUSES = ("collapse", "no_mask")  # the episode began as an OCC, not as a close
MEM_CHECK_EVERY = 250                 # [runtime] memory tripwire cadence
# E1 / E5 memory criteria -- the ONE definition: Tol_47's on-node E1 verdict evaluates it through
# e1_memory_verdict() and gate_v5 must use the same numbers (review r1: the worker's band was (5.0, 6.8), the gate's
# (5.3, 6.4), so an on-node EQUIV PASS could sit next to a gate FAIL). Arm A (pop off) keeps one bf16 [3,1008,1008]
# processed frame per tracked frame = 5.814 MiB; arm B (pop on) keeps none. Slopes are medians of the steps
# between post-prune floors (robust to a delayed free), one per half (rebuild at frame 450); 36 prunes = 900 / 25.
E1_MEM = {"A_band": (5.3, 6.4), "B_abs": 0.25, "AB_band": (5.3, 6.4), "A_npf_min": 400, "prunes": 36}
E5_PRUNES = 12                        # E5: 300 frames / PRUNE_EVERY 25 (gate_v5 E5_PRUNES)
TRAIN_DAYS = frozenset({"20240418", "20240419"})   # [train] the only admissible provenance of the band
RF_OP_THRESHOLD = 0.35                # [train] train_summary.op_threshold (asserted, never tuned)
RF_RESOLUTION = 624                   # [train] train_summary.resolution (pinned at load: strip_checkpoint drops model_config)
RF_CKPT_SHA_PREFIX = "163b73b1c3b0c2cb"   # pinned detector checkpoint (checkpoint_best_total.pth)
RF_CKPT_BYTES = 136459491             # pinned detector checkpoint size
TRAIN_BAND_REFERENCE = {"A_REF_TRAIN": 87137.0, "FRAG_LO": {"dayl": 0.44, "dark": 0.43}, "OVER_HI": 1.32}  # the
# documented reference values (mean(86,500, 87,774); frag_lo user_override; over_hi Tol_41). The tracker NEVER uses this
# dict -- it reads train_summary.json through band_from_train_summary(); the SELFTEST asserts the file equals it.

# =====================================================================================================================
# VERBATIM Tol_03_track_v4.py:280-303 -- the test references for every fast twin (never edit)
# =====================================================================================================================
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

# =====================================================================================================================
# C7 -- exact fast twins. Same integer counts, same true division => bit-identical floats (S1, S2, E4).
# =====================================================================================================================
def measure(arr):
    """(area, bbox) of a 2-D bool mask; bbox = (y0, y1, x0, x1) half-open, None when empty."""
    rows = np.flatnonzero(arr.any(axis=1))
    if rows.size == 0: return 0, None
    cols = np.flatnonzero(arr.any(axis=0))
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return int(np.count_nonzero(arr[y0:y1, x0:x1])), (y0, y1, x0, x1)

class MaskCache:
    """Per-frame (area, bbox) cache keyed by id(arr). Each entry HOLDS a strong reference to
    its array, so an id cannot be recycled while it is cached; every lookup asserts entry.arr is arr. The tracker
    clears it once per frame, so masks older than a frame (healthy_mask, last_gen_mask, overrides) are re-measured."""
    __slots__ = ("_d", "hits", "misses")
    def __init__(self):
        self._d = {}; self.hits = 0; self.misses = 0
    def stats(self, arr):
        k = id(arr)
        e = self._d.get(k)
        if e is not None:
            if e[0] is not arr:
                raise RuntimeError("MaskCache: the entry for this id() holds a different array (stale cache)")
            self.hits += 1
            return e[1], e[2]
        if arr.dtype != np.bool_ or arr.ndim != 2:
            raise TypeError(f"MaskCache: 2-D bool masks only, got {arr.dtype} ndim={arr.ndim}")
        a, bb = measure(arr)
        self._d[k] = (arr, a, bb); self.misses += 1
        return a, bb
    def clear(self):
        self._d.clear()
    def __len__(self):
        return len(self._d)

CACHE = MaskCache()

def _fast_ok(a, b):
    return (a.shape == b.shape and a.ndim == 2 and a.dtype == np.bool_ and b.dtype == np.bool_)

def _inter(a, A, b, B):
    if A is None or B is None: return 0
    y0 = max(A[0], B[0]); y1 = min(A[1], B[1]); x0 = max(A[2], B[2]); x1 = min(A[3], B[3])
    if y0 >= y1 or x0 >= x1: return 0
    return int(np.count_nonzero(np.logical_and(a[y0:y1, x0:x1], b[y0:y1, x0:x1])))

def m_iou_fast(a, b, cache=None):
    if a is None or b is None: return 0.0
    if not _fast_ok(a, b): return m_iou(a, b)            # anything unusual takes the verbatim path
    c = CACHE if cache is None else cache
    aa, A = c.stats(a); ab, B = c.stats(b)
    i = _inter(a, A, b, B); u = aa + ab - i
    return float(i / u) if u else 0.0

def m_ios_fast(a, b, cache=None):
    if a is None or b is None: return 0.0
    if not _fast_ok(a, b): return m_ios(a, b)
    c = CACHE if cache is None else cache
    aa, A = c.stats(a); ab, B = c.stats(b)
    m = min(aa, ab)
    return float(_inter(a, A, b, B) / m) if m else 0.0

def area_fast(a, cache=None):
    if a is None: return 0
    if a.dtype != np.bool_ or a.ndim != 2: return int(a.sum())
    return (CACHE if cache is None else cache).stats(a)[0]

def rle_fast(mask, cache=None):
    """rle_encode() on the mask's bounding block only. Runs are found per column with a zero separator so none can
    cross a column inside the block; runs that touch across a column seam (block spans the full height) are merged.
    Output is identical to rle_encode (S1, E4), including the leading 0 when the first pixel is set."""
    if mask.dtype != np.bool_ or mask.ndim != 2:
        return rle_encode(mask)
    h, w = mask.shape
    n = h * w
    _a, bb = (CACHE if cache is None else cache).stats(mask)
    if bb is None:
        return {"size": [h, w], "counts": [n]}
    y0, y1, x0, x1 = bb
    hb, wb = y1 - y0, x1 - x0
    pad = np.zeros((wb, hb + 2), np.int8)
    pad[:, 1:-1] = mask[y0:y1, x0:x1].T
    d = np.diff(pad.ravel())
    ps = np.flatnonzero(d == 1) + 1
    pe = np.flatnonzero(d == -1) + 1
    cs, rs = np.divmod(ps, hb + 2)
    ce, re_ = np.divmod(pe, hb + 2)
    gs = (x0 + cs).astype(np.int64) * h + (y0 + rs - 1)
    ge = (x0 + ce).astype(np.int64) * h + (y0 + re_ - 1)
    if gs.size > 1:
        brk = ge[:-1] != gs[1:]
        gs = gs[np.concatenate(([True], brk))]
        ge = ge[np.concatenate((brk, [True]))]
    bnd = np.empty(2 * gs.size + 2, np.int64)
    bnd[0] = 0; bnd[1:-1:2] = gs; bnd[2:-1:2] = ge; bnd[-1] = n
    counts = np.diff(bnd)
    if counts[-1] == 0: counts = counts[:-1]
    return {"size": [h, w], "counts": counts.tolist()}

# =====================================================================================================================
# C3 -- the incumbent post-chain. post_chain is VERBATIM Tol_44_detector_eval.py:702-711 (= Tol_03.detect() ordering).
# =====================================================================================================================
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

def post_chain_idx(cands, topk=DET_TOPK, dedupe=DEDUPE_IOS, min_px=MIN_MASK_PX, iou=None, ios=None, area=None):
    """post_chain() returning INDICES into cands (for detections_<clip>.json.gz kept_idx). Same floor, same stable
    area-descending order, same greedy dedupe, same truncation (S3 asserts equality with post_chain)."""
    iou = iou or m_iou_fast; ios = ios or m_ios_fast; area = area or area_fast
    idx = [i for i, (m, _s) in enumerate(cands) if area(m) >= min_px]
    n_floor = len(cands) - len(idx)
    idx.sort(key=lambda i: -area(cands[i][0]))
    keep = []
    for i in idx:
        m = cands[i][0]
        if all(ios(m, cands[k][0]) < dedupe and iou(m, cands[k][0]) < dedupe for k in keep):
            keep.append(i)
    return (keep[:topk] if topk else keep), n_floor

# =====================================================================================================================
# Phases, pools and the training-day band
# =====================================================================================================================
def tod_of_wall(w):                   # VERBATIM Tol_03:647-648 (v4's own logic keeps it)
    return "night" if (w.hour >= 21 or w.hour < 6) else "day"

def phase4(w):                        # VERBATIM Tol_44:290-294 (the new logic's phase function)
    if w.hour >= 21 or w.hour < 6: return "night"
    if w.hour < 8:                 return "dawn"
    if w.hour == 20:               return "dusk"
    return "day"

POOL = {"dayl": ("day", "dawn"), "dark": ("dusk", "night")}   # dusk counts as dark

def pool_of_wall(w):
    return "dayl" if phase4(w) in POOL["dayl"] else "dark"

BAND_KEYS = ("A_ref[20240418]", "A_ref[20240419]", "frag_lo[dayl]", "frag_lo[dark]", "over_hi")

def band_from_train_summary(ts, det_run_tag):
    """Assert the detector run is the training-day run and READ the band constants with their
    provenance. Raises on any deviation (never falls back to a default)."""
    def bad(msg): raise RuntimeError(f"train_summary.json: {msg}")
    if ts.get("run_tag") != det_run_tag: bad(f"run_tag {ts.get('run_tag')!r} != det_run_tag {det_run_tag!r}")
    if ts.get("mode") not in ("full", "resume") or ts.get("smoke"): bad(f"mode {ts.get('mode')!r} is not a full run")
    if set(ts.get("source_days") or []) != set(TRAIN_DAYS): bad(f"source_days {ts.get('source_days')} != {sorted(TRAIN_DAYS)}")
    if float(ts.get("op_threshold")) != RF_OP_THRESHOLD: bad(f"op_threshold {ts.get('op_threshold')} != {RF_OP_THRESHOLD}")
    if int(ts.get("resolution")) != RF_RESOLUTION: bad(f"resolution {ts.get('resolution')} != {RF_RESOLUTION}")
    C = ts.get("constants") or {}
    prov = {}
    for k in BAND_KEYS:
        if k not in C: bad(f"constants lacks {k!r}")
        df = set(C[k].get("derived_from") or [])
        if not df or not df <= set(TRAIN_DAYS): bad(f"{k} derived_from {sorted(df)} is not within {sorted(TRAIN_DAYS)}")
        prov[k] = {"value": C[k]["value"], "provenance": C[k].get("provenance"), "derived_from": sorted(df)}
    a_ref = (float(C["A_ref[20240418]"]["value"]) + float(C["A_ref[20240419]"]["value"])) / 2.0
    return {"A_REF_TRAIN": a_ref,
            "FRAG_LO": {"dayl": float(C["frag_lo[dayl]"]["value"]), "dark": float(C["frag_lo[dark]"]["value"])},
            "OVER_HI": float(C["over_hi"]["value"]),
            "op_threshold": float(ts["op_threshold"]), "resolution": int(ts["resolution"]),
            "rf_checkpoint_sha256": ts.get("checkpoint_best_total_sha256"),
            "det_run_tag": det_run_tag, "provenance": prov,
            "A_REF_TRAIN_rule": "mean(A_ref[20240418], A_ref[20240419]); median largest daylight mask at 1 Hz (Tol_40:141-142)"}

def band(pool, B):
    """[FRAG_LO[pool] x A_REF_TRAIN, OVER_HI x A_REF_TRAIN] in px -- constants only, no runtime area history (S4 t)."""
    return B["FRAG_LO"][pool] * B["A_REF_TRAIN"], B["OVER_HI"] * B["A_REF_TRAIN"]

def in_band(area, pool, B):
    lo, hi = band(pool, B)
    return lo <= area <= hi

def pen_count(dets, pool, B, area=None):
    area = area or area_fast
    return sum(1 for m, _s in dets if in_band(area(m), pool, B))

def pen_count_ok(n_inband):
    return n_inband <= N_CALVES

def healthy_ok(area, collapse_run, pool, B):
    """The slot's latest genuine mask with collapse_run == 0 AND area in band(pool)."""
    return collapse_run == 0 and in_band(area, pool, B)

# =====================================================================================================================
# C10 -- cadence
# =====================================================================================================================
def strikes_to_close(is_day, interval_s):
    """1 + ceil((n_v4 - 1) x I_v4 / I): holds v4's time from first strike to close.
    day@120 -> 2, night@300 -> 3, night@120 -> 6. Exact rational arithmetic (no float ceil drift)."""
    n_v4, i_v4 = STRIKES_V4["day" if is_day else "night"]
    q = Fraction(n_v4 - 1) * Fraction(str(float(i_v4))) / Fraction(str(float(interval_s)))
    return 1 + math.ceil(q)

def pending_window(interval_s):
    return PENDING_FACTOR * interval_s          # Tol_03:764 formula, unchanged (night@120 -> 300 s)

# =====================================================================================================================
# C1 -- widgets, config, guards
# =====================================================================================================================
# name: (v5 default, kind, choices). Kinds: str | int | float | bool (true/false) | onoff (on/off) | enum.
WIDGETS = {
    "mode": ("track", "enum", ("track", "equiv")),
    "stage": ("20240422", "str", None),
    "max_clips": ("0", "int", None),
    "max_frames": ("0", "int", None),
    "from_hhmm": ("", "hhmm", None),
    "to_hhmm": ("", "hhmm", None),
    "track_root": ("Tracking_v5a10", "str", None),
    "track_fps": ("5.0", "float", None),
    "allow_nonstandard_fps": ("false", "bool", None),
    "detect_day_sec": ("120", "float", None),
    "detect_night_sec": ("120", "float", None),        # D2 (Tol_03:87 default was 300)
    "refresh_min_sec": ("600", "float", None),
    "embed_every_sec": ("10", "float", None),
    "crop_every_sec": ("60", "float", None),
    "backbone": ("student", "enum", ("student", "teacher")),
    "student_ckpt": ("best_backbone_student_v2.pt", "str", None),   # Tol_03:95 default was v1
    "sam3_model": ("facebook/sam3", "str", None),
    "partial_repair": ("true", "bool", None),           # D1
    "detector": ("rfdetr", "enum", ("rfdetr", "sam3")),
    "det_run_tag": ("segv1d_tracked_seg-small_r624", "str", None),
    "dark_fire": ("on", "onoff", None),
    "dark_link": ("on", "onoff", None),
    "dark_open": ("on", "onoff", None),
    "dark_repeat_sec": ("10", "float", None),           # D3; also DARK_FIRST_SEC
    "v5_fixes": ("on", "onoff", None),
    "pop_frames": ("on", "onoff", None),
    "exact_fast": ("on", "onoff", None),
    "decode": ("seek_thread", "enum", ("seek", "seek_thread", "sequential")),
    "gpu_tf": ("off", "onoff", None),
    "allow_numeric_change": ("false", "bool", None),
    "mem_cap_gib": ("0", "float", None),
    "mem_assert_alloc_gib": ("6", "float", None),
    "mem_assert_reserved_gib": ("10", "float", None),
    "mem_assert_session_frames": ("2", "int", None),
    "cuda_probe_retries": ("3", "int", None),
    "cuda_probe_sleep_s": ("90", "float", None),
    "allow_mixed_gpu": ("false", "bool", None),
    # mode=equiv only (C15). Not a tracking flag: needed so the same notebook can run the E6 comparison.
    "equiv_tests": ("E1,E2,E3,E4,E5", "str", None),
    "e6_ref_root": ("Tracking_v5eq_ref", "str", None),
    "e6_reg_root": ("Tracking_v5eq_reg", "str", None),
}
# The "regression value" of each flag (plus partial_repair / student_ckpt set to match the reference run).
REGRESSION_FLAGS = {"detector": "sam3", "dark_fire": "off", "detect_night_sec": "300", "v5_fixes": "off",
                    "pop_frames": "on", "exact_fast": "on", "decode": "seek_thread", "gpu_tf": "off",
                    "allow_numeric_change": "false"}
EQUIV_ONLY = ("equiv_tests", "e6_ref_root", "e6_reg_root")
ROOT_EXCLUDE = ("stage", "from_hhmm", "to_hhmm", "max_clips", "max_frames", "allow_mixed_gpu", "mode") + EQUIV_ONLY
ENV_PREFIX = "TOLV5_"

def _parse(name, s, kind, choices):
    s = "" if s is None else str(s).strip()
    if kind == "str":
        return s
    if kind == "hhmm":
        if s and not re.fullmatch(r"\d{4}", s): raise ValueError(f"widget {name}={s!r}: HHMM or empty")
        return s
    if kind == "int":
        try: return int(s)
        except ValueError: raise ValueError(f"widget {name}={s!r}: not an int")
    if kind == "float":
        try: return float(s)
        except ValueError: raise ValueError(f"widget {name}={s!r}: not a number")
    if kind == "bool":
        if s.lower() not in ("true", "false"): raise ValueError(f"widget {name}={s!r}: true|false")
        return s.lower() == "true"
    if kind == "onoff":
        if s.lower() not in ("on", "off"): raise ValueError(f"widget {name}={s!r}: on|off")
        return s.lower() == "on"
    if kind == "enum":
        if s.lower() not in choices: raise ValueError(f"widget {name}={s!r}: one of {choices}")
        return s.lower()
    raise ValueError(f"unknown widget kind {kind}")

def config_from_flags(raw):
    """raw: {widget: string} (missing/empty -> v5 default). Unknown names or values raise. Returns
    {"values": typed widget values, "raw": canonical strings, derived fields ...} (S7)."""
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown: raise ValueError(f"unknown widget(s) {unknown}")
    vals, canon = {}, {}
    for name, (dflt, kind, choices) in WIDGETS.items():
        s = raw.get(name)
        s = dflt if (s is None or str(s).strip() == "") else s
        vals[name] = _parse(name, s, kind, choices)
        canon[name] = str(s).strip()
    numeric_change = vals["decode"] == "sequential" or vals["gpu_tf"]
    if numeric_change and not vals["allow_numeric_change"]:
        raise ValueError("decode=sequential / gpu_tf=on change pixels or features: set allow_numeric_change=true "
                         "explicitly (deferred in this version)")
    for k in ("detect_day_sec", "detect_night_sec", "dark_repeat_sec"):
        if not vals[k] > 0: raise ValueError(f"{k} must be > 0")
    cfg = {"values": vals, "raw": canon, "numeric_change": bool(numeric_change),
           "dark_fire": vals["dark_fire"],
           "dark_link": bool(vals["dark_fire"] and vals["dark_link"]),     # ignored when dark_fire=off (S7)
           "dark_open": bool(vals["dark_fire"] and vals["dark_open"]),
           "dark_repeat_sec": vals["dark_repeat_sec"], "dark_first_sec": vals["dark_repeat_sec"],
           "strikes_day": strikes_to_close(True, vals["detect_day_sec"]),
           "strikes_night": strikes_to_close(False, vals["detect_night_sec"]),
           "pending_day": pending_window(vals["detect_day_sec"]),
           "pending_night": pending_window(vals["detect_night_sec"]),
           "is_regression": all(canon[k] == v or vals[k] == _parse(k, v, WIDGETS[k][1], WIDGETS[k][2])
                                for k, v in REGRESSION_FLAGS.items())}
    return cfg

def dark_cfg(cfg):
    """The dark-step parameter dict dark_step_decide / slot_actionable take."""
    return {"dark_link": cfg["dark_link"], "dark_open": cfg["dark_open"], "dark_first_sec": cfg["dark_first_sec"],
            "link_max_sec": LINK_MAX_SEC, "max_per_clip": DARK_MAX_PER_CLIP, "min_gap_sec": DARK_MIN_GAP_SEC,
            "unclaimed_max": UNCLAIMED_MAX, "link_contain_ios": LINK_CONTAIN_IOS,
            "link_conflict_ios": LINK_CONFLICT_IOS, "cover_ios": COVER_IOS}

def gpu_class(name):
    n = (name or "").upper()
    if "A100" in n: return "A100"
    if re.search(r"A10(?!\d)", n): return "A10"
    return n or "UNKNOWN"

def decode_class(d):
    return "seek" if d in ("seek", "seek_thread") else d      # seek == seek_thread (same decoder calls, same order)

RESUME_KEYS = ("detector", "detector_sha256", "op_threshold", "dark_fire", "dark_link", "dark_open", "dark_repeat_sec",
               "detect_day_sec", "detect_night_sec", "v5_fixes", "partial_repair", "numeric_change", "decode_class",
               "gpu_tf", "student_sha256", "core_sha256", "source_norm_sha256", "gpu_class")

def guard_params(cfg, shas, gpu_cls, op_threshold):
    """The flat record the resume guard compares. shas: {detector, student, core, source, source_norm}; the
    Tol_47 source is guarded by its NORMALISED digest `source_norm` (the raw `source` depends on how a notebook-mode
    task read its own bytes -- workspace FUSE or SDK export -- and is only pin-compared, in the equiv echo)."""
    v = cfg["values"]
    return {"detector": v["detector"], "detector_sha256": shas.get("detector"), "op_threshold": op_threshold,
            "dark_fire": cfg["dark_fire"], "dark_link": cfg["dark_link"], "dark_open": cfg["dark_open"],
            "dark_repeat_sec": v["dark_repeat_sec"], "detect_day_sec": v["detect_day_sec"],
            "detect_night_sec": v["detect_night_sec"], "v5_fixes": v["v5_fixes"], "partial_repair": v["partial_repair"],
            "numeric_change": cfg["numeric_change"], "decode_class": decode_class(v["decode"]), "gpu_tf": v["gpu_tf"],
            "student_sha256": shas.get("student"), "core_sha256": shas.get("core"),
            "source_norm_sha256": shas.get("source_norm"),
            "gpu_class": gpu_cls}

def resume_guard(stored, cur, allow_mixed_gpu=False):
    """Raise if a resumed day would mix behaviours (S10). Returns the tolerated diffs (GPU class only, and
    only with allow_mixed_gpu)."""
    diffs = [(k, stored.get(k), cur.get(k)) for k in RESUME_KEYS if stored.get(k) != cur.get(k)]
    if not diffs: return []
    if allow_mixed_gpu and all(k == "gpu_class" for k, _a, _b in diffs): return diffs
    raise RuntimeError("resume guard: done clips were tracked with a different configuration -- a day must not mix "
                       "behaviours: " + "; ".join(f"{k}: stored {a!r} != now {b!r}" for k, a, b in diffs))

def root_config_record(cfg, shas, gpu_cls, B):
    return {"widgets": {k: cfg["values"][k] for k in WIDGETS if k not in ROOT_EXCLUDE},
            "shas": {k: shas.get(k) for k in ("detector", "student", "core", "source_norm")},   # source_norm: see guard_params
            "gpu_class": gpu_cls,
            "band": {"A_REF_TRAIN": B["A_REF_TRAIN"], "FRAG_LO": dict(B["FRAG_LO"]), "OVER_HI": B["OVER_HI"]}}

def root_config_check(stored, cur, allow_mixed_gpu=False):
    """S15: every later day / resume in a root must match its ROOT_CONFIG.json."""
    diffs = []
    for k in sorted(set(stored.get("widgets", {})) | set(cur.get("widgets", {}))):
        a, b = stored.get("widgets", {}).get(k), cur.get("widgets", {}).get(k)
        if a != b: diffs.append((f"widget:{k}", a, b))
    for k in ("shas", "band"):
        if stored.get(k) != cur.get(k): diffs.append((k, stored.get(k), cur.get(k)))
    if stored.get("gpu_class") != cur.get("gpu_class"):
        diffs.append(("gpu_class", stored.get("gpu_class"), cur.get("gpu_class")))
    if not diffs: return []
    if allow_mixed_gpu and all(k == "gpu_class" for k, _a, _b in diffs): return diffs
    raise RuntimeError("ROOT_CONFIG.json mismatch -- all days of a root share one config: "
                       + "; ".join(f"{k}: root {a!r} != now {b!r}" for k, a, b in diffs))

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def source_norm_sha256(data):
    """sha256 of a notebook source NORMALISED for transport: BOM dropped, CRLF / CR -> LF, trailing
    whitespace stripped, blank lines dropped. The local .py (CRLF), the Volume copy the launcher stages and a Databricks
    SOURCE export of the imported notebook differ in exactly these respects, so all three give ONE value, while any
    change to a code or comment character changes it. Tol_47 guards this value (ROOT_CONFIG.json shas.source_norm and
    the resume guard's source_norm_sha256); the local tools reproduce it with this function."""
    t = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
    t = t.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256("\n".join(l.rstrip() for l in t.split("\n") if l.strip()).encode("utf-8")).hexdigest()

def source_shas(data):
    """The two Tol_47 source digests, one name = one meaning (an earlier revision had put the normalised
    digest under `shas.source`, which the launcher and gate_v5 compare with the RAW payload pin):
      source      = sha256 of the RAW bytes -- the pin-comparable value (subprocess mode: the staged, pinned copy);
      source_norm = source_norm_sha256 -- the transport-invariant value ROOT_CONFIG.json / the resume guard hold."""
    b = bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
    return {"source": hashlib.sha256(b).hexdigest(), "source_norm": source_norm_sha256(b)}

def publish_file(src, dst):
    """Put a finished LOCAL file at dst (a UC Volume path) without leaving a truncated dst behind (review r1: telemetry
    was json.dump-ed in place on the Volume, so a crash mid-write left a file the resume could not read): copy to
    dst + '.part' and os.replace it over dst; where the mount cannot rename, fall back to a plain sequential copy
    (Tol_03's behaviour, still never an append or a seek-write). Returns 'replace' or 'copy'."""
    part = dst + ".part"
    try:
        shutil.copyfile(src, part)
        os.replace(part, dst)
        return "replace"
    except OSError:
        try:
            if os.path.exists(part): os.remove(part)
        except OSError:
            pass
        shutil.copyfile(src, dst)
        return "copy"

def resume_telemetry(path, done_stems, read=None):
    """(old, kept) for a resumed day. kept = the telemetry entries of the .done_ clips, the day's
    acceptance evidence (G2 / G3 read it). With done clips present, a MISSING, UNREADABLE (after read_json_retry) or
    INCOMPLETE file raises: starting from an empty clip list would let the next write erase that evidence and would
    silently skip the resume guard. Without done clips a missing / unreadable file is only reported (nothing to keep)."""
    read = read or read_json_retry
    done = sorted(set(done_stems))
    fix = ("restore the file, or delete those .done_ markers so the clips are re-tracked -- never resume without the "
           "done clips' evidence")
    if not os.path.exists(path):
        if done:
            raise RuntimeError(f"resume: {len(done)} .done_ clip(s) but no telemetry file {path}: {fix}")
        return None, []
    try:
        old = read(path)
        if not isinstance(old, dict): raise ValueError(f"top level is {type(old).__name__}, not an object")
    except Exception as e:                                     # noqa: BLE001 -- reported / raised below
        if done:
            raise RuntimeError(f"resume: telemetry {path} unreadable ({e}) with {len(done)} .done_ clip(s): {fix}") from e
        print(f"⚠ telemetry preload failed ({e}); no .done_ clips, starting a fresh file")
        return None, []
    ds = set(done)
    kept = [c for c in (old.get("clips") or []) if isinstance(c, dict) and c.get("clip") in ds]
    missing = sorted(ds - {c["clip"] for c in kept})
    if missing:
        raise RuntimeError(f"resume: telemetry {path} has no entry for .done_ clip(s) {missing[:5]}"
                           f"{' ...' if len(missing) > 5 else ''}: {fix}")
    return old, kept

def e1_memory_verdict(mem, npf_max, crit=None):
    """Memory verdict of the PRUNED E1 (E1_MEM), on Tol_47's results.E1.memory {A, B, A2: {floor_slope_mib_
    per_frame [half1, half2], max_processed_frames, prunes}}. npf_max = the mem_assert_session_frames widget (2).
    Returns (pass, criteria, failures)."""
    c = dict(E1_MEM if crit is None else crit)
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    arm = lambda k: mem.get(k) or {}
    fa = list(arm("A").get("floor_slope_mib_per_frame") or [])
    fb = list(arm("B").get("floor_slope_mib_per_frame") or [])
    fails = []
    if not (len(fa) == 2 and all(num(x) and c["A_band"][0] <= x <= c["A_band"][1] for x in fa)):
        fails.append(f"arm A post-prune floor slopes {fa} not both in {list(c['A_band'])} MiB/frame")
    if not (len(fb) == 2 and all(num(x) and abs(x) <= c["B_abs"] for x in fb)):
        fails.append(f"arm B post-prune floor slopes {fb} not both within +-{c['B_abs']} MiB/frame")
    dif = [round(a - b, 4) if num(a) and num(b) else None for a, b in zip(fa, fb)]
    if not (len(dif) == 2 and all(num(x) and c["AB_band"][0] <= x <= c["AB_band"][1] for x in dif)):
        fails.append(f"A - B {dif} not both in {list(c['AB_band'])} MiB/frame (one frame = 5.814 MiB)")
    npa, npb = arm("A").get("max_processed_frames"), arm("B").get("max_processed_frames")
    if not (num(npa) and npa >= c["A_npf_min"]):
        fails.append(f"arm A max processed_frames {npa} < {c['A_npf_min']} (pop off must accumulate)")
    if not (num(npb) and npb <= npf_max):
        fails.append(f"arm B max processed_frames {npb} > {npf_max}")
    pr = {k: arm(k).get("prunes") for k in ("A", "B", "A2")}
    if not all(v == c["prunes"] for v in pr.values()):
        fails.append(f"prune cycles {pr} != {c['prunes']} per arm")
    crit_out = {"A_floor_slope_in": list(c["A_band"]), "B_floor_slope_abs_le": c["B_abs"],
                "A_minus_B_in": list(c["AB_band"]), "A_max_processed_frames_ge": c["A_npf_min"],
                "B_max_processed_frames_le": npf_max, "prunes_per_arm": c["prunes"], "source": "tol_v5_core.E1_MEM"}
    return (not fails), crit_out, fails

def read_json_retry(path, tries=6, sleep_s=5.0, _sleep=time.sleep):
    """json.load with a retry on a half-visible file (a concurrent writer on a FUSE mount).
    Raises RuntimeError after `tries` unreadable reads; a missing file raises FileNotFoundError at once."""
    last = None
    for k in range(tries):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (ValueError, UnicodeDecodeError) as e:          # json.JSONDecodeError is a ValueError
            last = e
            if k + 1 < tries: _sleep(sleep_s)
    raise RuntimeError(f"{path}: not valid JSON after {tries} reads ({last})")

def write_json_if_absent(path, obj, tag="", _link=None):
    """Atomic write-if-absent (two tasks of one root may start together). The record is written to a
    unique temp file beside the target and published with os.link -- create-if-absent in one step, FileExistsError
    when another task won -- so no reader ever sees half a file and at most one writer publishes. Where hard links are
    unsupported (some FUSE mounts) it falls back to os.replace (still never half a file; two simultaneous FIRST writers
    may both publish and the last one wins -- harmless, because every caller re-reads the file and checks it against
    its own record), and as a last resort to a plain sequential write (readers use read_json_retry).
    Returns True when this call published the file. _link: injectable for the SELFTEST."""
    if os.path.exists(path): return False
    link = _link if _link is not None else getattr(os, "link", None)
    tmp = (f"{path}.tmp_{re.sub(r'[^A-Za-z0-9_.-]', '_', str(tag))}_{os.getpid()}_{threading.get_ident()}_"
           f"{os.urandom(4).hex()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    try:
        try:
            if link is None: raise OSError("no os.link on this platform")
            link(tmp, path)
            return True
        except FileExistsError:
            return False
        except OSError:
            if os.path.exists(path): return False
            try:
                os.replace(tmp, path)
                return True
            except OSError:                                    # no rename either: plain write, readers retry
                if os.path.exists(path): return False
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(obj, f, indent=1, default=str)
                return True
    finally:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except OSError:
            pass

# =====================================================================================================================
# C3 precision toggle, C5 pop-frames forward, C8 decode iterator
# =====================================================================================================================
def run_with_precision(set_prec, get_prec, fn, inner="high", outer="highest"):
    """TF32 only inside fn (RF-DETR predict); 'highest' restored on EVERY path, then asserted (S9, E3)."""
    set_prec(inner)
    try:
        return fn()
    finally:
        set_prec(outer)
        if get_prec() != outer:
            raise RuntimeError(f"float32 matmul precision is {get_prec()!r} after the detector, expected {outer!r}")

def sam3_forward(sam3_call, sess, pixel, pop):
    """C5. pop=False: Tol_03:633 verbatim (implicit index = len(processed_frames)). pop=True: pass the
    explicit index 0,1,2,... (reset to 0 in open_session_v4) and drop that frame's pixels after its own forward
    (the only pixel read is inside that forward, TV:1910). Popping WITHOUT the explicit index would repeat index 0,
    hit the cond-frame reuse branch (TV:1781-1784) and emit the seed mask forever (S8 demonstrates)."""
    if pop:
        fi = sess._tol_fidx
        out = sam3_call(inference_session=sess, frame_idx=fi, frame=pixel)
        (sess.processed_frames or {}).pop(fi, None)
        sess._tol_fidx = fi + 1
        return out
    return sam3_call(inference_session=sess, frame=pixel)

def n_pf(session):
    """processed_frames is None until the first forward (TV:144, :313); session may be None."""
    return len(session.processed_frames or {}) if session is not None else 0

_END, _ERR = object(), object()

def iter_frames(cap, fidxs, mode="seek_thread", pos_prop=1, qsize=8, poll_s=5.0):
    """C8. Yields (fidx, ok, frame) in fidx order.
    seek        -- Tol_03:1137-1138 inline: cap.set(POS_FRAMES, f); cap.read().
    seek_thread -- the SAME calls in the SAME order on ONE producer thread (owns cap, never touches torch), bounded
                   queue; ok=False is passed through; a producer exception is re-raised in the consumer; on early
                   exit / close() the consumer sets stop, drains the queue and JOINS the producer (call close()
                   before cap.release()).
    sequential  -- grab() every native frame, retrieve() the wanted ones: CHANGES PIXELS (needs allow_numeric_change)."""
    fidxs = list(fidxs)
    if mode == "seek":
        for f in fidxs:
            cap.set(pos_prop, f)
            ok, fr = cap.read()
            yield f, ok, fr
        return
    if mode == "sequential":
        want = set(fidxs); last = fidxs[-1] if fidxs else -1; i = 0
        while i <= last:
            if not cap.grab():
                for f in fidxs:
                    if f >= i: yield f, False, None
                return
            if i in want:
                ok, fr = cap.retrieve()
                yield i, ok, fr
            i += 1
        return
    if mode != "seek_thread":
        raise ValueError(f"decode mode {mode!r}")
    q = queue.Queue(maxsize=qsize)
    stop = threading.Event()
    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2); return True
            except queue.Full:
                continue
        return False
    def producer():
        try:
            for f in fidxs:
                if stop.is_set(): return
                cap.set(pos_prop, f)
                ok, fr = cap.read()
                if not put((f, ok, fr)): return
            put((_END, None, None))
        except BaseException as e:            # noqa: BLE001 -- re-raised in the consumer
            put((_ERR, e, None))
    th = threading.Thread(target=producer, name="tol_v5_decode", daemon=True)
    th.start()
    try:
        while True:
            try:
                item = q.get(timeout=poll_s)
            except queue.Empty:
                if not th.is_alive() and q.empty():
                    raise RuntimeError("decode producer thread exited without a sentinel")
                continue
            if item[0] is _END: return
            if item[0] is _ERR: raise item[1]
            yield item
    finally:
        stop.set()
        while True:
            try: q.get_nowait()
            except queue.Empty: break
        th.join()

# =====================================================================================================================
# C11-C13 -- the dark step and the v5_fixes guards (pure; the tracker applies the decisions)
# =====================================================================================================================
def slot_actionable(v, pool, cfg):
    """Actionability of one DARK slot view. Returns {link, open, why}; why = the first reason the slot is not
    link-actionable (None when it is). Order: young, suspended, rate_limited, link_off, no_healthy, healthy_stale,
    cause, covered, merge_near."""
    st = v["state"]
    if st not in (ST_OCC, ST_CLOSED):
        return {"link": False, "open": False, "why": "live"}
    age = v.get("dark_age_s")
    aged = st == ST_CLOSED or (age is not None and age >= cfg["dark_first_sec"])
    since = v.get("since_action_s")
    base = None
    if not aged: base = "young"
    elif v.get("actions_clip", 0) >= cfg["max_per_clip"]: base = "suspended"
    elif since is not None and since < cfg["min_gap_sec"]: base = "rate_limited"
    why = base
    if why is None:
        if not cfg["dark_link"]: why = "link_off"
        elif v.get("healthy_mask") is None: why = "no_healthy"
        elif v.get("healthy_age_s") is None or v["healthy_age_s"] > cfg["link_max_sec"]: why = "healthy_stale"
        elif v.get("cause") not in LINK_CAUSES: why = "cause"
        elif v.get("covered"): why = "covered"
        elif v.get("merge_near"): why = "merge_near"
    return {"link": why is None, "open": bool(base is None and cfg["dark_open"] and pool == "dayl"), "why": why}

def dark_step_decide(dets, claimed, slots, pool, B, cfg, scheduled=False, sched_rebuilds=False,
                     iou=None, ios=None, area=None):
    """The dark step. Pure: returns the decisions, the tracker applies them.

    dets      [(bool mask, score)] as kept by the post-chain (area-first order, never reordered here)
    claimed   det indices matched by the scheduled step 2 (empty on a dark-only pass)
    slots     N_CALVES views: state, live_mask (TRACK slots' last genuine mask), dark_age_s, healthy_mask,
              healthy_age_s, healthy_seg, cause, covered, merge_near, actions_clip, since_action_s
    scheduled True on a scheduled pass: links only (fast-open is the v4 unmatched loop's job there)
    sched_rebuilds  the scheduled part will rebuild anyway (young-OCC deferral does not apply)
    """
    iou = iou or m_iou_fast; ios = ios or m_ios_fast; area = area or area_fast
    lo, hi = band(pool, B)
    n = len(slots)
    live = [s for s in range(n) if slots[s]["state"] == ST_TRACK and slots[s].get("live_mask") is not None]
    dark = [s for s in range(n) if slots[s]["state"] in (ST_OCC, ST_CLOSED)]
    areas = [int(area(m)) for m, _s in dets]
    inb = [lo <= a <= hi for a in areas]
    res = {"pool": pool, "band": [lo, hi], "n_det": len(dets), "n_inband": int(sum(inb)), "areas": areas,
           "scores": [round(float(s), 4) for _m, s in dets], "live": live, "dark": dark, "n_dark": len(dark),
           "act": {}, "unclaimed": [], "candidates": [], "det_reason": {}, "links": {}, "opens": {},
           "rejected": {}, "refused": None, "deferred": None, "open_refused": None, "young": [], "ran": False,
           "used": []}
    act = {s: slot_actionable(slots[s], pool, cfg) for s in dark}
    res["act"] = act
    link_act = [s for s in dark if act[s]["link"]]
    open_act = [s for s in dark if act[s]["open"]]
    for s in dark:
        if not act[s]["link"]: res["rejected"][s] = [[None, act[s]["why"]]]
    if not link_act and not open_act:
        return res
    res["ran"] = True
    # ---- A. unclaimed candidates ----
    conflict = {}
    for b, (dm, _sc) in enumerate(dets):
        if b in claimed:
            res["det_reason"][b] = "matched"; continue
        clm = max([max(iou(dm, slots[l]["live_mask"]), ios(slots[l]["live_mask"], dm)) for l in live], default=0.0)
        if clm >= cfg["unclaimed_max"]:
            res["det_reason"][b] = "claimed"; continue
        res["unclaimed"].append(b)
        if not inb[b]:
            res["det_reason"][b] = "area"; continue
        cf = max([ios(dm, slots[l]["live_mask"]) for l in live], default=0.0)
        conflict[b] = cf
        if cf >= cfg["link_conflict_ios"]:
            res["det_reason"][b] = "conflict"; continue
        res["det_reason"][b] = "candidate"
        res["candidates"].append(b)
    # ---- B. refusals, evaluated first ----
    if res["n_inband"] > N_CALVES:
        res["refused"] = "pen_count"
        for s in link_act: res["rejected"][s] = [[None, "pen_count"]]
        return res
    young = [s for s in range(n) if slots[s]["state"] == ST_OCC and slots[s].get("dark_age_s") is not None
             and slots[s]["dark_age_s"] < cfg["dark_first_sec"]]
    res["young"] = young
    if young and not sched_rebuilds:
        res["deferred"] = "young_occ"
        for s in link_act: res["rejected"][s] = [[None, "young_occ"]]
        return res
    # ---- C/D. link gate: contained, then two-way unique ----
    cset = set(res["candidates"])
    passes = {s: [b for b in res["candidates"]
                  if ios(slots[s]["healthy_mask"], dets[b][0]) >= cfg["link_contain_ios"]] for s in link_act}
    for s in link_act:
        rej = []
        for b in range(len(dets)):
            if b not in cset:
                rej.append([b, res["det_reason"][b]]); continue
            if b not in passes[s]:
                rej.append([b, "not_contained"]); continue
            if len(passes[s]) > 1 or any(b in passes[s2] for s2 in link_act if s2 != s):
                rej.append([b, "ambiguous"]); continue
            hm = slots[s]["healthy_mask"]
            res["links"][s] = {"det": b, "ios": round(ios(hm, dets[b][0]), 4), "iou": round(iou(hm, dets[b][0]), 4),
                               "det_area": areas[b], "conflict": round(conflict.get(b, 0.0), 4),
                               "cause": slots[s].get("cause"), "gap_s": slots[s].get("healthy_age_s"),
                               "prev_seg": slots[s].get("healthy_seg"), "closed_occ": slots[s]["state"] == ST_OCC,
                               "n_dark": len(dark), "sole_dark": len(dark) == 1}
        if s not in res["links"]:
            res["rejected"][s] = rej if rej else [[None, "no_candidate"]]
        else:
            res["rejected"].pop(s, None)
    # ---- E. daylight fast-open (dark-only passes; the dark pool never opens a moved calf, D4) ----
    if not scheduled and pool == "dayl" and cfg["dark_open"]:
        linked = [L["det"] for L in res["links"].values()]
        t_closed = [s for s in open_act if slots[s]["state"] == ST_CLOSED and s not in res["links"]]
        t_occ = sorted([s for s in open_act if slots[s]["state"] == ST_OCC and s not in res["links"]],
                       key=lambda s: -(slots[s].get("dark_age_s") or 0.0))
        targets = t_closed + t_occ
        cands = [b for b in res["candidates"] if b not in linked
                 and all(ios(dets[b][0], dets[l][0]) < cfg["cover_ios"] for l in linked)]
        if cands and targets:
            if len(cands) > len(targets):
                res["open_refused"] = {"reason": "excess_candidates", "n_cand": len(cands), "n_targets": len(targets)}
                if res["refused"] is None: res["refused"] = "excess_candidates"
            else:
                for b, s in zip(cands, targets):
                    closed = slots[s]["state"] == ST_CLOSED
                    res["opens"][s] = {"det": b, "reason": "reopen" if closed else "displaced",
                                       "closed_occ": not closed, "det_area": areas[b],
                                       "n_cand": len(cands), "n_targets": len(targets)}
                    res["rejected"].pop(s, None)
    res["used"] = sorted([L["det"] for L in res["links"].values()] + [o["det"] for o in res["opens"].values()])
    return res

def reopen_considered(un_masks, live_masks, pool, B, n_inband, ios=None, area=None):
    """The v5_fixes reopen guard as a per-detection VETO on the scheduled unmatched loop (review r1). Returns (ok, why, refused):
    ok[i] = may unmatched detection i open or displace; why[i] = None | 'pen_count' (more than N_CALVES in-band
    detections on the call: no detection may) | 'band' (area outside band(pool)) | 'conflict' (IoS >= LINK_CONFLICT_IOS
    with a live mask). The loop itself -- pending bookkeeping, the two-sighting rule and the ignored_detection branch
    with its scene event -- still runs on EVERY unmatched detection, as in v4 ('the ignored_detection branch is
    unchanged'): filtering the list, as reopen_filter does, hid v4's whole-animal signal that gates the night repairs."""
    ios = ios or m_ios_fast; area = area or area_fast
    if n_inband > N_CALVES: return [False] * len(un_masks), ["pen_count"] * len(un_masks), True
    ok, why = [], []
    for dm in un_masks:
        if not in_band(area(dm), pool, B):
            ok.append(False); why.append("band")
        elif max([ios(dm, L) for L in live_masks], default=0.0) >= LINK_CONFLICT_IOS:
            ok.append(False); why.append("conflict")
        else:
            ok.append(True); why.append(None)
    return ok, why, False

def reopen_filter(un_masks, live_masks, pool, B, n_inband, ios=None, area=None):
    """The considered subset of reopen_considered, in the original order (filters, never reorders). Returns
    (considered list, refused bool). Tol_47 uses reopen_considered (veto flags), not this list."""
    ok, _why, refused = reopen_considered(un_masks, live_masks, pool, B, n_inband, ios=ios, area=area)
    return [dm for dm, k in zip(un_masks, ok) if k], refused

def seed_filter(dets, pool, B, area=None):
    """v5_fixes: seeding candidates filtered to band(pool) keeping area order; > N_CALVES left = refused."""
    area = area or area_fast
    inb = [(m, s) for m, s in dets if in_band(area(m), pool, B)]
    if len(inb) > N_CALVES: return [], True
    return inb, False

def refresh_ok(last, dm, other_lasts, pool, B, iou=None, ios=None, area=None):
    """v5_fixes: refresh needs IoU >= REFRESH_IOU (not max(IoU, IoS): a calf inside a merged blob has IoS 1),
    no conflict with any other live mask, and a calf-sized detection."""
    iou = iou or m_iou_fast; ios = ios or m_ios_fast; area = area or area_fast
    return (iou(last, dm) >= REFRESH_IOU
            and max([ios(dm, L) for L in other_lasts], default=0.0) < LINK_CONFLICT_IOS
            and in_band(area(dm), pool, B))

def v4_step4_opens(un_masks, pending, states, occ_has_lgw, occ_dark_s, is_day, now, window_s, iou=None, ios=None,
                   allow=None, ignored=None):
    """Side-effect-free replica of the v4 unmatched loop (Tol_03:762-788) on COPIES: which slots would it open?
    Used as the dry run behind the young-OCC deferral ('the scheduled part will rebuild anyway'), and by S4(q).
    allow (v5_fixes, reopen_considered's ok flags): a vetoed detection still does the pending bookkeeping and can still
    reach the ignored_detection branch, but never opens or displaces -- exactly Tol_47's loop. ignored: a list that
    collects the indices reaching the ignored_detection branch (persisted, no free slot, no OCC slot, all TRACKING)."""
    iou = iou or m_iou_fast; ios = ios or m_ios_fast
    pend = [dict(p) for p in pending if (now - p["wall"]).total_seconds() <= window_s]
    opens = []
    for di, dm in enumerate(un_masks):
        hit = None
        for p in pend:
            if iou(p["mask"], dm) >= NIGHT_STABLE_IOU or ios(p["mask"], dm) >= NIGHT_STABLE_IOU:
                hit = p; break
        persisted = hit is not None
        if hit: hit["mask"], hit["wall"], hit["seen"] = dm, now, hit.get("seen", 1) + 1
        else: pend.append({"mask": dm, "wall": now, "seen": 1})
        taken = [o[0] for o in opens]
        free = [s for s in range(len(states)) if states[s] == ST_CLOSED and s not in taken]
        ok = True if allow is None else bool(allow[di])
        if is_day and free:
            if ok: opens.append((free[0], di, "reopen"))
        elif (not is_day) and persisted and free:
            if ok: opens.append((free[0], di, "reopen"))
        elif persisted and not free:
            occ = [(s, occ_dark_s[s]) for s in range(len(states))
                   if states[s] == ST_OCC and occ_has_lgw[s] and s not in taken]
            if occ:
                if ok: opens.append((max(occ, key=lambda z: z[1])[0], di, "displaced"))
            elif ignored is not None and all(st == ST_TRACK for st in states):
                ignored.append(di)
    return opens

def row_overrides(opens, open_extra, repaired, v5_fixes):
    """C12/C13. Slots whose row on THIS frame must carry the seed detection (and whose new segment
    starts with n_frames = 1): every dark-step open, every scheduled open under v5_fixes, every accepted repair
    (Tol_03 LAST_REPAIRS). With v5_fixes off a v4 scheduled reopen keeps the v4 row (S11)."""
    out = {}
    for slot, dm, reason in opens:
        ex = open_extra.get(slot) or {}
        if v5_fixes or reason == "dark_link" or ex.get("via") == "dark_pass":
            out[slot] = dm
    for s, dm in repaired.items():
        out[s] = dm
    return out

def events_for_call(scheduled, n_det, dark_ran):
    """Which event types one detect() call emits. A dark-only call emits ONLY dark_pass
    (never pass / pass_no_detections) and never touches CLIP_PASS (see counts_clip_pass)."""
    if scheduled:
        if n_det == 0: return ["pass_no_detections"]
        return ["pass", "dark_pass"] if dark_ran else ["pass"]
    return ["dark_pass"]

def counts_clip_pass(scheduled):
    return bool(scheduled)          # CLIP_PASS["passes"] and the INV-7 counters count scheduled calls only

# =====================================================================================================================
# C14 -- events: builders, schema, v4 projection
# =====================================================================================================================
V4_PASS_KEYS = ("type", "clip", "t", "forced", "n_det", "tracked", "matched", "struck", "opened", "refreshed", "rebuilt")
V5_PASS_FIELDS = ("dark", "n_raw", "n_inband", "det_ms", "detector", "pen_refused")
V5_SEG_FIELDS = ("via", "link_from", "link_gap_s", "link_ios", "link_iou", "det_area", "n_dark", "sole_dark", "cause")
EVENT_SCHEMA = {
    "dark_onset": ("slot", "seg_id", "clip", "t", "wall", "cause", "covered", "merge_near", "healthy_age_s"),
    "dark_pass": ("clip", "t", "wall", "pool", "kind", "n_raw", "n_det", "n_inband", "det_ms", "dark_slots",
                  "unclaimed", "candidates", "links", "opens", "rejected", "refused", "deferred", "rebuilt"),
    "dark_link": ("slot", "seg_id", "prev_seg_id", "closed_occ", "gap_s", "ios", "iou", "det_area", "conflict",
                  "cause", "n_dark", "sole_dark"),
    "dark_open": ("slot", "seg_id", "reason", "det_area", "n_cand", "n_targets"),
    "dark_open_refused": ("clip", "t", "reason", "n_cand", "n_targets"),
    "pen_count_refused": ("path", "n_inband", "areas", "scores"),
    "dark_suspended": ("slot", "clip", "n"),
    "mem_check": ("si", "allocated_gib", "reserved_gib", "processed_frames"),
    "mem_assert_fail": ("si", "allocated_gib", "reserved_gib", "processed_frames"),
    "resume": ("clip", "restored_slots"),
    "pass": V4_PASS_KEYS[1:] + V5_PASS_FIELDS,
}
V5_ONLY_TYPES = frozenset(k for k in EVENT_SCHEMA if k != "pass")

def check_event(ev):
    req = EVENT_SCHEMA.get(ev.get("type"))
    if req is None: return []
    return [k for k in req if k not in ev]

def _r(x, nd=2):
    return None if x is None else round(float(x), nd)

def ev_dark_onset(slot, seg_id, clip, t, wall, cause, covered, merge_near, healthy_age_s):
    return {"type": "dark_onset", "slot": slot, "seg_id": seg_id, "clip": clip, "t": _r(t), "wall": wall,
            "cause": cause, "covered": bool(covered), "merge_near": bool(merge_near), "healthy_age_s": _r(healthy_age_s)}

def ev_dark_pass(clip, t, wall, res, kind, n_raw, det_ms, dark_slots, rebuilt):
    """dark_pass from a dark_step_decide() result. kind: dark_only | combined."""
    return {"type": "dark_pass", "clip": clip, "t": _r(t), "wall": wall, "pool": res["pool"], "kind": kind,
            "n_raw": n_raw, "n_det": res["n_det"], "n_inband": res["n_inband"], "det_ms": _r(det_ms, 1),
            "dark_slots": dark_slots, "unclaimed": list(res["unclaimed"]), "candidates": list(res["candidates"]),
            "links": {str(s): v for s, v in res["links"].items()}, "opens": {str(s): v for s, v in res["opens"].items()},
            "rejected": {str(s): v for s, v in res["rejected"].items()}, "refused": res["refused"],
            "deferred": res["deferred"], "rebuilt": bool(rebuilt), "areas": res["areas"], "scores": res["scores"]}

def ev_dark_link(slot, seg_id, link, clip=None, t=None):
    return {"type": "dark_link", "slot": slot, "seg_id": seg_id, "prev_seg_id": link.get("prev_seg"),
            "closed_occ": bool(link["closed_occ"]), "gap_s": _r(link.get("gap_s")), "ios": link["ios"],
            "iou": link["iou"], "det_area": link["det_area"], "conflict": link["conflict"], "cause": link["cause"],
            "n_dark": link["n_dark"], "sole_dark": bool(link["sole_dark"]), "clip": clip, "t": _r(t)}

def ev_dark_open(slot, seg_id, op, clip=None, t=None):
    return {"type": "dark_open", "slot": slot, "seg_id": seg_id, "reason": op["reason"], "det_area": op["det_area"],
            "n_cand": op["n_cand"], "n_targets": op["n_targets"], "via": "dark_pass", "clip": clip, "t": _r(t)}

def ev_dark_open_refused(clip, t, info):
    return {"type": "dark_open_refused", "clip": clip, "t": _r(t), "reason": info["reason"],
            "n_cand": info["n_cand"], "n_targets": info["n_targets"]}

def ev_pen_count_refused(path, n_inband, areas, scores, clip=None, t=None):
    return {"type": "pen_count_refused", "path": path, "n_inband": int(n_inband), "areas": [int(a) for a in areas],
            "scores": [round(float(s), 4) for s in scores], "clip": clip, "t": _r(t)}

def ev_dark_suspended(slot, clip, n, t=None):
    return {"type": "dark_suspended", "slot": slot, "clip": clip, "n": int(n), "t": _r(t)}

def ev_mem(kind, si, allocated_gib, reserved_gib, processed_frames, clip=None, **extra):
    e = {"type": kind, "si": int(si), "allocated_gib": _r(allocated_gib, 3), "reserved_gib": _r(reserved_gib, 3),
         "processed_frames": int(processed_frames), "clip": clip}
    e.update(extra)
    return e

def ev_resume(clip, restored_slots):
    return {"type": "resume", "clip": clip, "restored_slots": list(restored_slots)}

def project_v4(ev):
    """Project a v5 event onto the v4 schema for E6: v5-only types dropped, v5 fields removed."""
    if ev.get("type") in V5_ONLY_TYPES: return None
    e = dict(ev)
    if e.get("type") == "pass":
        for k in V5_PASS_FIELDS: e.pop(k, None)
    if e.get("type") in ("segment_open", "segment_close"):
        for k in V5_SEG_FIELDS: e.pop(k, None)
    return e

# =====================================================================================================================
# C15 -- E6 comparator (reference arm = Tol_03, regression arm = Tol_47 regression flags) + branch coverage
# =====================================================================================================================
COVERAGE_BRANCHES = ("strike", "night_pending_reopen", "partial_repair*", "session_carry", "displaced_by_detection",
                     "not_seedable_at_rebuild", "merge_resolved", "occluded_timeout", "all_closed")

def branch_coverage(events):
    c = Counter()
    for e in events:
        t = e.get("type")
        if t in ("strike", "session_carry", "merge_resolved", "all_closed"): c[t] += 1
        if t and (t.startswith("partial_repair") or t.startswith("growth_repair")): c["partial_repair*"] += 1
        if t == "segment_close" and e.get("reason") in ("displaced_by_detection", "not_seedable_at_rebuild",
                                                        "occluded_timeout"):
            c[e["reason"]] += 1
        if t == "segment_open" and e.get("reason") == "reopen" and e.get("wall"):
            try:
                if tod_of_wall(datetime.fromisoformat(e["wall"])) == "night": c["night_pending_reopen"] += 1
            except ValueError:
                pass
    return {k: c.get(k, 0) for k in COVERAGE_BRANCHES}

def compare_frames(fa, fb):
    """Exact comparison of two masks_<clip> 'frames' lists. Returns (n_equal, n_total, first difference)."""
    first = None; eq = 0
    n = max(len(fa), len(fb))
    for i in range(n):
        a = fa[i] if i < len(fa) else None; b = fb[i] if i < len(fb) else None
        if a == b: eq += 1
        elif first is None:
            first = {"i": i, "ref": None if a is None else {"frame_idx": a.get("frame_idx"), "t_sec": a.get("t_sec")},
                     "reg": None if b is None else {"frame_idx": b.get("frame_idx"), "t_sec": b.get("t_sec")},
                     "slots": None if (a is None or b is None) else
                     [s for s in sorted(set(a["masks"]) | set(b["masks"])) if a["masks"].get(s) != b["masks"].get(s)]}
    return eq, n, first

def compare_events(ea, eb):
    pa = [p for p in (project_v4(e) for e in ea) if p is not None]
    pb = [p for p in (project_v4(e) for e in eb) if p is not None]
    first = None
    for i in range(max(len(pa), len(pb))):
        a = pa[i] if i < len(pa) else None; b = pb[i] if i < len(pb) else None
        if a != b:
            first = {"i": i, "ref": a, "reg": b}; break
    return first is None, len(pa), len(pb), first

def _read_json_any(p):
    if p.endswith(".gz"):
        with gzip.open(p, "rt") as f: return json.load(f)
    with open(p) as f: return json.load(f)

def _obs_pair(d):
    """(meta, embeddings) of a day dir's observations; None when the day has none; ("incomplete", None) when only one
    of the two files exists."""
    mp, zp = os.path.join(d, "observations_meta.json"), os.path.join(d, "observations.npz")
    if not (os.path.exists(mp) and os.path.exists(zp)):
        return None if not (os.path.exists(mp) or os.path.exists(zp)) else ("incomplete", None)
    with open(mp) as f: meta = json.load(f)
    with np.load(zp) as z: emb = np.array(z["emb"])
    return meta, emb

def obs_equal(a, b):
    if a is None or b is None: return a is None and b is None
    if a[1] is None or b[1] is None: return False
    return (a[0] == b[0] and a[1].shape == b[1].shape and a[1].dtype == b[1].dtype
            and bool(np.array_equal(a[1], b[1])))

def e6_compare_day(ref_dir, reg_dir):
    """E6: the regression arm must equal the reference arm on masks, segments.json and (projected) events."""
    rep = {"ref_dir": ref_dir, "reg_dir": reg_dir, "masks": {}, "equal": True}
    ra = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ref_dir, "masks_*.json.gz")))
    rb = sorted(os.path.basename(p) for p in glob.glob(os.path.join(reg_dir, "masks_*.json.gz")))
    if ra != rb:
        rep["equal"] = False; rep["mask_files"] = {"ref_only": sorted(set(ra) - set(rb)), "reg_only": sorted(set(rb) - set(ra))}
    for nm in sorted(set(ra) & set(rb)):
        A = _read_json_any(os.path.join(ref_dir, nm)); Bm = _read_json_any(os.path.join(reg_dir, nm))
        eq, tot, first = compare_frames(A.get("frames", []), Bm.get("frames", []))
        hdr = {k: (A.get(k), Bm.get(k)) for k in ("clip", "track_fps", "mask_format") if A.get(k) != Bm.get(k)}
        rep["masks"][nm] = {"frames_equal": eq, "frames": tot, "first_diff": first, "header_diff": hdr}
        if eq != tot or hdr: rep["equal"] = False
    sa = _read_json_any(os.path.join(ref_dir, "segments.json")) if os.path.exists(os.path.join(ref_dir, "segments.json")) else None
    sb = _read_json_any(os.path.join(reg_dir, "segments.json")) if os.path.exists(os.path.join(reg_dir, "segments.json")) else None
    seg_eq = sa is not None and sb is not None and sa.get("segments") == sb.get("segments")
    rep["segments_equal"] = seg_eq
    if not seg_eq: rep["equal"] = False
    def rd(p):
        if not os.path.exists(p): return []
        with open(p) as f: return [json.loads(l) for l in f if l.strip()]
    ea, eb = rd(os.path.join(ref_dir, "events.jsonl")), rd(os.path.join(reg_dir, "events.jsonl"))
    ok, na, nb, first = compare_events(ea, eb)
    rep["events"] = {"equal": ok, "n_ref": na, "n_reg": nb, "first_diff": first}
    if not ok: rep["equal"] = False
    # the regression arm must reach NO new branch (no v5-only event type other than the memory
    # tripwire's mem_check / a resume; every pass with dark None and pen_refused False), and the observations (meta +
    # embeddings) must be identical too -- the PASS definition, as the independent e6_compare.py checks it
    new_types = sorted({e.get("type") for e in eb if e.get("type") in V5_ONLY_TYPES} - {"mem_check", "resume"})
    new_pass = sum(1 for e in eb if e.get("type") == "pass" and (e.get("dark") is not None or e.get("pen_refused")))
    rep["reg_new_branch"] = {"v5_only_types": new_types, "passes_with_dark_or_pen_refused": new_pass}
    if new_types or new_pass: rep["equal"] = False
    rep["observations_equal"] = obs_equal(_obs_pair(ref_dir), _obs_pair(reg_dir))
    if not rep["observations_equal"]: rep["equal"] = False
    da = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ref_dir, ".done_*")))
    db = sorted(os.path.basename(p) for p in glob.glob(os.path.join(reg_dir, ".done_*")))
    rep["done_equal"] = da == db
    if da != db: rep["equal"] = False
    rep["coverage"] = {"ref": branch_coverage(ea), "reg": branch_coverage(eb)}
    rep["uncovered"] = [k for k in COVERAGE_BRANCHES if rep["coverage"]["ref"][k] == 0 and rep["coverage"]["reg"][k] == 0]
    return rep

# =====================================================================================================================
# Static audit of Tol_47 (review checklist items 1, 3, 4, 5, 6, 8) -- run by Tol_47's TOL_SELFTEST=1 path
# =====================================================================================================================
T03_SHA256 = "d7fe1cf4f87634f4951deb2e868f7b74dd8012311ad3a35065d7d27f2d372181"
APPEARANCE_NAMES = frozenset({"embed_robust", "dino", "dino_proc", "obs", "canonical_crop", "_augment"})
DARK_FUNCS = ("detector_pass", "_dark_step", "_slot_views", "_any_actionable", "_flush_dark_onsets", "_note_healthy",
              "_sched_would_rebuild", "_dark_action", "_dark_slots_view", "_live")

def hunks_without_marker(t03_text, t47_text, n=3):
    """Every changed / added hunk of the Tol_03 -> Tol_47 unified diff must carry a [v5-Cn] marker (in its added lines,
    or in its context when it only deletes). Returns (n_hunks, offending hunk headers, markers seen)."""
    import difflib
    d = list(difflib.unified_diff(t03_text.splitlines(), t47_text.splitlines(), "Tol_03_track_v4.py",
                                  "Tol_47_track_v5.py", n=n, lineterm=""))
    hunks, cur = [], None
    for line in d[2:]:
        if line.startswith("@@"):
            cur = [line]; hunks.append(cur)
        elif cur is not None:
            cur.append(line)
    bad, seen = [], set()
    for h in hunks:
        plus = [l for l in h[1:] if l.startswith("+")]
        ctx = [l for l in h[1:] if l.startswith(" ")]
        pool = plus if plus else ctx
        mk = set(re.findall(r"\[v5-C(\d+)", "\n".join(pool)))
        seen |= set(re.findall(r"\[v5-C(\d+)", "\n".join(plus)))
        if not mk: bad.append(h[0])
    return len(hunks), bad, sorted(int(x) for x in seen)

def audit_tol47(path, t03_path=None):
    """Static audit of the Tol_47 source; raises AssertionError listing every problem."""
    import ast
    src = open(path, encoding="utf-8").read()
    stripped = "\n".join(l for l in src.splitlines() if not l.startswith("# MAGIC"))
    compile(stripped, path, "exec")
    tree = ast.parse(stripped)
    probs = []
    funcs = {}
    parent_fn = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        funcs.setdefault(fn.name, fn)
        for sub in ast.walk(fn):
            parent_fn.setdefault(id(sub), fn.name) if sub is not fn else None
    seg = lambda node: ast.get_source_segment(stripped, node) or ""
    # checklist 3 / INV-1: the healthy mask is a gate reference only -- never a seed
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fname = getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            if fname == "open_session_v4" and "healthy" in seg(n):
                probs.append(f"INV-1: open_session_v4 call mentions healthy: {seg(n)[:80]}")
            if (isinstance(n.func, ast.Attribute) and n.func.attr == "append"
                    and getattr(n.func.value, "id", None) in ("seeds", "opens", "carry") and "healthy" in seg(n)):
                probs.append(f"INV-1: {seg(n)[:80]}")
    # checklist 4 / INV-2: no appearance signal in the pass or the dark step
    for fname in DARK_FUNCS:
        if fname not in funcs:
            probs.append(f"missing function {fname}"); continue
        used = {x.id for x in ast.walk(funcs[fname]) if isinstance(x, ast.Name)}
        if used & APPEARANCE_NAMES: probs.append(f"INV-2: {fname} references {sorted(used & APPEARANCE_NAMES)}")
    # checklist 5/6: the dark step never touches strikes / pending / refresh / repair / INV-7 counters; no continue-links
    if "_dark_step" in funcs:
        f = funcs["_dark_step"]
        used = {x.id for x in ast.walk(f) if isinstance(x, ast.Name)}
        for forbidden in ("strikes", "pending_open", "last_refresh_wall", "repairs_clip", "last_repair_wall"):
            if forbidden in used: probs.append(f"_dark_step touches {forbidden}")
        calls = {getattr(c.func, "id", None) for c in ast.walk(f) if isinstance(c, ast.Call)}
        for forbidden in ("open_segment", "open_session_v4", "detect", "log_event"):
            if forbidden in calls: probs.append(f"_dark_step calls {forbidden} (links must be ordinary opens)")
        for sub in ast.walk(f):
            if isinstance(sub, ast.Subscript) and getattr(sub.value, "id", None) == "CLIP_PASS":
                key = sub.slice.value if isinstance(sub.slice, ast.Constant) else None
                if key != "closes": probs.append(f"_dark_step touches CLIP_PASS[{key!r}]")
    # checklist 8: _tol_fidx reset only in open_session_v4; every tracker forward through core.sam3_forward
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "_tol_fidx" for t in n.targets):
            if parent_fn.get(id(n)) != "open_session_v4": probs.append("_tol_fidx assigned outside open_session_v4")
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "sam3":
            probs.append("a direct sam3(...) forward bypasses core.sam3_forward")
    # checklist 7: rfdetr imported only inside load_rfdetr
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            if any(m.split(".")[0] == "rfdetr" for m in mods) and parent_fn.get(id(n)) != "load_rfdetr":
                probs.append("rfdetr imported outside load_rfdetr")
    # checklist 5: every v5 detect() call site names its kind (pass | dark | seed | equiv)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "detect":
            if not any(k.arg == "kind" for k in n.keywords): probs.append(f"detect() without kind=: {seg(n)[:60]}")
    # finisher r1 (2026-09-24): gc.collect() once right after EVERY session teardown -- the full A100 profile showed a
    # torn-down session can sit in a reference cycle until the next collect, and exact_fast removed the prune-time gc
    lines = stripped.splitlines()
    for i, l in enumerate(lines):
        if re.search(r"\bdel (session|sess)\b", l.split("#", 1)[0]):
            if "gc.collect()" not in " ".join(x.split("#", 1)[0] for x in lines[i:i + 3]):
                probs.append(f"session teardown without gc.collect() (stripped line {i + 1}): {l.strip()[:70]}")
    # E1 and E5 prune like the tracker (every PRUNE_EVERY frames, 36 cycles at 900 frames)
    if "_e1_arm" in funcs and not any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "prune"
                                      for c in ast.walk(funcs["_e1_arm"])):
        probs.append("E1 (_e1_arm) never prunes")
    if stripped.count("prune(sess, 8)") < 2:
        probs.append("the E1 and E5 loops must both prune(sess, 8) like the tracker")
    # exactly one admissible tol_v5_core directory, asserted after the import
    if "_core_dirs" in funcs or "sys.path.insert(0, CORE_DIR)" not in stripped or "resolved to" not in stripped:
        probs.append("tol_v5_core must be imported from exactly one asserted directory")
    # review r1: raw `source` (pin-compared) + guarded `source_norm`, both from core.source_shas
    if "core.source_shas(" not in stripped or re.search(r"[\"']source[\"']\s*:\s*SOURCE_SHA\b", stripped):
        probs.append("the Tol_47 source shas are not core.source_shas() (raw `source` = the pin, `source_norm` guarded)")
    # review r1 (fixer): the clip entry gets ELAPSED seconds; the gc recheck never empty_cache()s; E1 verdict from core;
    # telemetry resumed / published through core; the scheduled unmatched loop vetoes instead of filtering
    calls_in = lambda f, attr: any(isinstance(c, ast.Call) and getattr(c.func, "attr", None) == attr
                                   for c in ast.walk(funcs[f])) if f in funcs else False
    if calls_in("_v5_clip_entry", "time"):
        probs.append("_v5_clip_entry must take the elapsed tracking seconds, not a start time (review r1)")
    if calls_in("_mem_check", "empty_cache"):
        probs.append("_mem_check's recheck must not empty_cache(): it would mask the reserved tripwire (review r1)")
    for need, why in (("core.e1_memory_verdict(", "E1 memory verdict from core.E1_MEM"),
                      ("core.resume_telemetry(", "telemetry preload raises instead of erasing done-clip evidence"),
                      ("core.publish_file(", "telemetry built locally, then published to the Volume"),
                      ("core.reopen_considered(", "the scheduled unmatched loop vetoes (reopen_considered)")):
        if need not in stripped: probs.append(f"missing {need[:-1]}: {why} (review r1)")
    if "core.reopen_filter(" in stripped:
        probs.append("core.reopen_filter hides detections from the v4 unmatched loop; use reopen_considered (review r1)")
    if "core.write_json_if_absent(" not in stripped or "core.read_json_retry(" not in stripped:
        probs.append("ROOT_CONFIG.json is not published by an atomic write-if-absent + read retry")
    # checklist 1: T03 byte-identical + every hunk marked
    t03_path = t03_path or os.path.join(os.path.dirname(path), "Tol_03_track_v4.py")
    info = {}
    if os.path.exists(t03_path):
        t03 = open(t03_path, "rb").read()
        if hashlib.sha256(t03).hexdigest() != T03_SHA256: probs.append("Tol_03_track_v4.py sha256 changed")
        nh, bad, seen = hunks_without_marker(t03.decode("utf-8"), src)
        if bad: probs.append(f"{len(bad)} hunk(s) without a [v5-Cn] marker: {bad[:6]}")
        if [c for c in seen if not 1 <= c <= 15]: probs.append(f"unknown change ids {seen}")
        info = {"hunks": nh, "markers": seen}
    else:
        info = {"hunks": None, "note": "Tol_03 not beside Tol_47: hunk-marker check skipped"}
    print(f"  audit_tol47: {len(funcs)} functions, {info}")
    if probs:
        raise AssertionError("Tol_47 static audit FAILED:\n  " + "\n  ".join(probs))
    print("  audit_tol47: INV-1 / INV-2 / dark-step isolation / pop index / rfdetr import / detect kinds / "
          "T03 sha / hunk markers -- all clean")
    return info

# =====================================================================================================================
# SELFTEST
# =====================================================================================================================
def _default_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("TOL_V5_DATA", ""),
             r"<LOCAL_RUN_DIR>"
             r"", os.path.join(here, "_v5_data")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "masks")): return c
    return None

class _Check:
    def __init__(self): self.n = 0; self.failed = []
    def __call__(self, cond, msg):
        self.n += 1
        if not cond: self.failed.append(msg)
        return cond

_H, _W = 648, 1152                     # quarter-size canvas for synthetic scenes (real-scale areas)

def _rect(y0, x0, h, w, H=_H, W=_W):
    m = np.zeros((H, W), bool); m[y0:y0 + h, x0:x0 + w] = True; return m

def _view(state, live=None, dark_age=None, healthy=None, healthy_age=None, cause=None, covered=False,
          merge_near=False, actions=0, since=None, seg="s_001"):
    return {"state": state, "live_mask": live, "dark_age_s": dark_age, "healthy_mask": healthy,
            "healthy_age_s": healthy_age, "healthy_seg": seg, "cause": cause, "covered": covered,
            "merge_near": merge_near, "actions_clip": actions, "since_action_s": since}

def _s1_s2(ck, rng, real_masks):
    # ---- S1: rle_fast == rle_encode, adversarial + real, round trip ----
    H, W = 37, 53
    synth = [np.zeros((H, W), bool), np.ones((H, W), bool)]
    for (y, x) in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        m = np.zeros((H, W), bool); m[y, x] = True; synth.append(m)
    m = np.zeros((H, W), bool); m[:, 0] = True; synth.append(m)
    m = np.zeros((H, W), bool); m[:, -1] = True; synth.append(m)
    m = np.zeros((H, W), bool); m[5, :] = True; synth.append(m)
    synth.append((np.indices((H, W)).sum(0) % 2).astype(bool))
    m = np.zeros((H, W), bool); m[:, 3:9] = True; synth.append(m)          # full-height block: runs merge across columns
    m = np.zeros((H, W), bool); m[H - 1, 4] = True; m[0, 5] = True; synth.append(m)   # seam-adjacent single pixels
    m = np.ones((H, W), bool); m[10, 10] = False; synth.append(m)
    for _ in range(60):
        synth.append(rng.random((H, W)) < rng.uniform(0.01, 0.99))
    for _ in range(20):
        mm = np.zeros((H, W), bool)
        y0, x0 = rng.integers(0, H), rng.integers(0, W)
        mm[y0:y0 + rng.integers(1, H), x0:x0 + rng.integers(1, W)] = True; synth.append(mm)
    allm = synth + list(real_masks)
    bad = 0
    for mm in allm:
        CACHE.clear()
        a, b = rle_encode(mm), rle_fast(mm)
        if a != b or not np.array_equal(rle_decode(b), mm) or type(b["counts"][0]) is not int: bad += 1
    CACHE.clear()
    ck(bad == 0, f"S1 rle_fast != rle_encode on {bad}/{len(allm)} masks")
    print(f"  S1 rle_fast == rle_encode: {len(allm) - bad}/{len(allm)} ({len(synth)} synthetic, {len(real_masks)} real)")
    # ---- S2: fast IoU/IoS exactly == verbatim ----
    pool = [mm for mm in synth[:40]] + [mm for mm in list(real_masks)[:: max(1, len(real_masks) // 24)][:24]]
    nbad = npair = 0
    for i in range(len(pool)):
        for j in range(len(pool)):
            if pool[i].shape != pool[j].shape: continue
            npair += 1
            if (m_iou_fast(pool[i], pool[j]) != m_iou(pool[i], pool[j]) or
                    m_ios_fast(pool[i], pool[j]) != m_ios(pool[i], pool[j])):
                nbad += 1
        CACHE.clear()
    a, b = _rect(0, 0, 10, 10, 50, 50), _rect(30, 30, 10, 10, 50, 50)
    ck(m_iou_fast(a, b) == m_iou(a, b) == 0.0 and m_ios_fast(a, b) == 0.0, "S2 disjoint bbox")
    ck(m_iou_fast(None, a) == 0.0 and m_ios_fast(a, None) == 0.0, "S2 None")
    e = np.zeros((50, 50), bool)
    ck(m_iou_fast(e, e) == m_iou(e, e) and m_ios_fast(e, a) == m_ios(e, a), "S2 empty masks")
    u8 = a.astype(np.uint8) * 255
    ck(m_iou_fast(u8, u8) == m_iou(u8, u8), "S2 non-bool takes the verbatim path")
    CACHE.clear()
    ck(nbad == 0, f"S2 fast IoU/IoS differ on {nbad}/{npair} pairs")
    print(f"  S2 fast IoU/IoS == verbatim on {npair - nbad}/{npair} pairs (+ disjoint / None / empty / non-bool)")

def _s3(ck):
    def t03_chain(cand_raw, topk):                          # re-implementation of Tol_03:463-472
        cand = []
        for m, s in cand_raw:
            if m.sum() >= MIN_MASK_PX: cand.append((m, float(s)))
        cand.sort(key=lambda z: -z[0].sum())
        keep = []
        for m, s in cand:
            if all(m_ios(m, k) < DEDUPE_IOS and m_iou(m, k) < 0.5 for k, _ in keep):
                keep.append((m, s))
        return keep[:topk] if topk else keep
    A = _rect(100, 100, 250, 300); Bm = _rect(100, 420, 250, 300)
    blob = A | Bm | _rect(100, 390, 250, 40)                       # merged two-calf blob, larger than either
    frag1 = _rect(400, 100, 60, 150); frag2 = _rect(430, 180, 60, 150)   # overlapping night fragments of one calf
    tiny = _rect(600, 900, 20, 20)
    worker = _rect(300, 800, 280, 300)
    scenes = [
        [(A, .9), (Bm, .8), (blob, .7)],                           # area-first keeps the blob, drops both inner calves
        [(frag1, .6), (frag2, .7), (A, .9), (tiny, .99)],          # fragments dedupe to the larger; tiny floored
        [(A, .9), (A.copy(), .95), (Bm, .5), (worker, .4), (frag1, .3), (frag2, .3), (tiny, .3), (_rect(0, 1000, 100, 100), .5)],
        [], [(tiny, .9)],
    ]
    ok = True
    for sc in scenes:
        for topk in (DET_TOPK, 0, 2):
            ref = t03_chain(sc, topk)
            k1, _ = post_chain(sc, topk=topk)
            ki, _ = post_chain_idx(sc, topk=topk)
            k2 = [(sc[i][0], float(sc[i][1])) for i in ki]
            same = (len(ref) == len(k1) == len(k2) and all(r[0] is a[0] and r[1] == a[1] for r, a in zip(ref, k1))
                    and all(np.array_equal(r[0], a[0]) and r[1] == a[1] for r, a in zip(ref, k2)))
            ok &= same
        CACHE.clear()
    k, _ = post_chain(scenes[0])
    ck(len(k) == 1 and k[0][0] is blob, "S3 area-first must keep the merged blob and drop the inner calves")
    ck(ok, "S3 post_chain / post_chain_idx != Tol_03:463-472 re-implementation")
    print("  S3 post_chain == post_chain_idx == Tol_03:463-472 re-implementation (blob / fragments / floor / topk)")

def _s4(ck, B):
    cfg = {"dark_link": True, "dark_open": True, "dark_first_sec": 10.0, "link_max_sec": LINK_MAX_SEC,
           "max_per_clip": DARK_MAX_PER_CLIP, "min_gap_sec": DARK_MIN_GAP_SEC, "unclaimed_max": UNCLAIMED_MAX,
           "link_contain_ios": LINK_CONTAIN_IOS, "link_conflict_ios": LINK_CONFLICT_IOS, "cover_ios": COVER_IOS}
    A = _rect(20, 20, 280, 300); Bm = _rect(20, 400, 280, 300); C = _rect(340, 20, 280, 300); D = _rect(340, 400, 280, 300)
    A2 = _rect(24, 26, 280, 300)                   # A, nudged: IoS ~0.96
    far = _rect(340, 780, 280, 300)                # the same calf after moving 900 px
    assert all(in_band(int(x.sum()), "dayl", B) for x in (A, Bm, C, D, A2, far))
    def scene(extra_dets, s0=None, others=None):
        sl = [s0 or _view(ST_OCC, dark_age=12.0, healthy=A, healthy_age=12.0, cause="collapse")]
        sl += others if others is not None else [_view(ST_TRACK, live=Bm), _view(ST_TRACK, live=C), _view(ST_TRACK, live=D)]
        live_dets = [(x["live_mask"], .9) for x in sl if x["state"] == ST_TRACK]
        return extra_dets + live_dets, sl
    out = {}
    def run(tag, dets, sl, pool="dark", scheduled=False, sched_rebuilds=False, c=None):
        r = dark_step_decide(dets, set(), sl, pool, B, c or cfg, scheduled=scheduled, sched_rebuilds=sched_rebuilds)
        out[tag] = r; CACHE.clear(); return r
    # (a) stationary, in band, sole dark -> link with sole_dark
    d, sl = scene([(A2, .9)]); r = run("a", d, sl)
    ck(0 in r["links"] and r["links"][0]["det"] == 0 and r["links"][0]["sole_dark"] and r["links"][0]["n_dark"] == 1,
       f"S4a stationary sole-dark link: {r['links']} {r['rejected']}")
    r2 = run("a_dayl", d, sl, pool="dayl")
    ck(0 in r2["links"] and not r2["opens"], "S4a dayl: link precedes open")
    # (b) moved -> no link; dayl -> open (displaced); dark pool incl. 20:30 dusk -> nothing
    d, sl = scene([(far, .9)]); r = run("b_dark", d, sl)
    ck(not r["links"] and not r["opens"] and r["rejected"][0][0] == [0, "not_contained"], f"S4b dark: {r['rejected']}")
    r = run("b_dayl", d, sl, pool="dayl")
    ck(not r["links"] and r["opens"].get(0, {}).get("reason") == "displaced", f"S4b dayl open: {r['opens']}")
    ck(pool_of_wall(datetime(2024, 4, 23, 20, 30)) == "dark" and pool_of_wall(datetime(2024, 4, 23, 7, 30)) == "dayl",
       "S4b 20:30 dusk is the dark pool, 07:30 dawn is dayl")
    r = run("b_scheduled", d, sl, pool="dayl", scheduled=True)
    ck(not r["opens"], "S4b a scheduled pass never fast-opens (v4's loop does)")
    # (c) merged 1.4 x A_REF blob -> area ; (d) 0.3 x A_REF fragment -> area
    big = _rect(10, 10, 360, 340)          # 122,400 px = 1.405 A_REF, contains A
    d, sl = scene([(big, .9)]); r = run("c", d, sl)
    ck(not r["links"] and r["det_reason"][0] == "area" and r["rejected"][0][0] == [0, "area"], f"S4c: {r['rejected']}")
    frag = _rect(60, 60, 100, 260)         # 26,000 px = 0.30 A_REF, inside A
    d, sl = scene([(frag, .9)]); r = run("d", d, sl)
    ck(not r["links"] and r["det_reason"][0] == "area", f"S4d: {r['det_reason']}")
    # (e) covered ; (f) close causes -> not link-actionable (dark pool: nothing actionable at all)
    d, sl = scene([(A2, .9)], s0=_view(ST_OCC, dark_age=12.0, healthy=A, healthy_age=12.0, cause="collapse", covered=True))
    r = run("e", d, sl)
    ck(not r["ran"] and r["act"][0]["why"] == "covered", f"S4e: {r['act']}")
    r = run("e_dayl", d, sl, pool="dayl")
    ck(not r["links"] and r["act"][0]["open"], "S4e dayl: covered is not link-actionable but still open-actionable")
    for cz in ("merge_jumper", "detector_contradiction", "occluded_timeout", "merge_ambiguous"):
        d, sl = scene([(A2, .9)], s0=_view(ST_CLOSED, dark_age=3.0, healthy=A, healthy_age=3.0, cause=cz))
        r = run("f_" + cz, d, sl)
        ck(not r["ran"] and r["act"][0]["why"] == "cause", f"S4f {cz}: {r['act']}")
    # (g) one detection containing two healthy masks -> ambiguous (two dark slots)
    hA = _rect(100, 100, 200, 240); hB = _rect(100, 345, 200, 240)       # 48,000 px each
    blob2 = _rect(95, 95, 210, 495)                                       # 103,950 px: in band, contains both
    sl = [_view(ST_OCC, dark_age=15.0, healthy=hA, healthy_age=15.0, cause="collapse"),
          _view(ST_OCC, dark_age=15.0, healthy=hB, healthy_age=15.0, cause="no_mask"),
          _view(ST_TRACK, live=C), _view(ST_TRACK, live=D)]
    r = run("g", [(blob2, .9), (C, .9), (D, .9)], sl)
    ck(not r["links"] and r["rejected"][0][0] == [0, "ambiguous"] and r["rejected"][1][0] == [0, "ambiguous"],
       f"S4g: {r['rejected']}")
    # (h) two detections inside one healthy mask -> ambiguous
    hBig = _rect(20, 20, 330, 330)                    # 108,900 px healthy
    p1 = _rect(25, 25, 160, 300); p2 = _rect(190, 25, 155, 300)   # 48,000 / 46,500 px, both inside hBig
    sl = [_view(ST_OCC, dark_age=12.0, healthy=hBig, healthy_age=12.0, cause="collapse"),
          _view(ST_CLOSED, dark_age=50.0, cause="merge_ambiguous"), _view(ST_TRACK, live=C), _view(ST_TRACK, live=D)]
    r = run("h", [(p1, .9), (p2, .8), (C, .9), (D, .9)], sl)
    ck(not r["links"] and [0, "ambiguous"] in r["rejected"][0] and [1, "ambiguous"] in r["rejected"][0], f"S4h: {r['rejected']}")
    # (i) conflict 0.12 -> not a candidate
    Bn = _rect(20, 600, 280, 300)                     # live calf
    cfl = _rect(20, 336, 280, 300)                    # overlaps Bn by 280 x 36 = 10,080 px = 0.12 x 84,000
    sl = [_view(ST_OCC, dark_age=12.0, healthy=_rect(20, 330, 280, 300), healthy_age=12.0, cause="collapse"),
          _view(ST_TRACK, live=Bn), _view(ST_TRACK, live=C), _view(ST_TRACK, live=D)]
    r = run("i", [(cfl, .9), (Bn, .9), (C, .9), (D, .9)], sl)
    ck(not r["links"] and r["det_reason"][0] == "conflict", f"S4i: {r['det_reason']}")
    # (j) cap 6 -> suspended ; (s) second action within 120 s -> rate_limited ; (k) healthy 61 s -> stale
    d, sl = scene([(A2, .9)], s0=_view(ST_OCC, dark_age=12.0, healthy=A, healthy_age=12.0, cause="collapse", actions=6))
    r = run("j", d, sl, pool="dayl")
    ck(not r["ran"] and r["act"][0]["why"] == "suspended" and not r["act"][0]["open"], f"S4j: {r['act']}")
    d, sl = scene([(A2, .9)], s0=_view(ST_OCC, dark_age=12.0, healthy=A, healthy_age=12.0, cause="collapse", actions=1, since=60.0))
    r = run("s", d, sl, pool="dayl")
    ck(not r["ran"] and r["act"][0]["why"] == "rate_limited", f"S4s: {r['act']}")
    d, sl = scene([(A2, .9)], s0=_view(ST_OCC, dark_age=61.0, healthy=A, healthy_age=61.0, cause="collapse"))
    r = run("k", d, sl)
    ck(not r["ran"] and r["act"][0]["why"] == "healthy_stale", f"S4k: {r['act']}")
    # (l) dayl, 1 target and 2 candidates -> excess_candidates
    E1 = _rect(340, 780, 250, 300); F1 = _rect(20, 780, 250, 300)
    sl = [_view(ST_CLOSED, dark_age=5.0, cause="merge_ambiguous"), _view(ST_TRACK, live=Bm), _view(ST_TRACK, live=C),
          _view(ST_CLOSED, dark_age=5.0, cause="merge_ambiguous", actions=1, since=30.0)]
    r = run("l", [(E1, .9), (F1, .8), (Bm, .9), (C, .9)], sl, pool="dayl")
    ck(not r["opens"] and r["refused"] == "excess_candidates" and r["open_refused"]["n_cand"] == 2
       and r["open_refused"]["n_targets"] == 1, f"S4l: {r['refused']} {r['open_refused']}")
    # (m) OCC at 5 s: not a target, another slot's link deferred unless the scheduled part rebuilds anyway
    sl = [_view(ST_OCC, dark_age=15.0, healthy=A, healthy_age=15.0, cause="collapse"),
          _view(ST_OCC, dark_age=5.0, healthy=Bm, healthy_age=5.0, cause="collapse"),
          _view(ST_TRACK, live=C), _view(ST_TRACK, live=D)]
    dets_m = [(A2, .9), (C, .9), (D, .9)]
    r = run("m", dets_m, sl)
    ck(r["deferred"] == "young_occ" and not r["links"] and r["act"][1]["why"] == "young" and 1 in r["young"],
       f"S4m deferral: {r['deferred']} {r['act']}")
    r = run("m_rebuild", dets_m, sl, scheduled=True, sched_rebuilds=True)
    ck(r["deferred"] is None and 0 in r["links"] and 1 not in r["links"], f"S4m rebuild-anyway: {r['links']}")
    # (n) OCC at 12 s -> link allowed (stationary) / open allowed (dayl, moved)
    d, sl = scene([(A2, .9)], s0=_view(ST_OCC, dark_age=12.0, healthy=A, healthy_age=12.0, cause="no_mask"))
    ck(0 in run("n_link", d, sl)["links"], "S4n link at 12 s")
    d, sl = scene([(far, .9)], s0=_view(ST_OCC, dark_age=12.0, healthy=A, healthy_age=12.0, cause="no_mask"))
    ck(0 in run("n_open", d, sl, pool="dayl")["opens"], "S4n open at 12 s")
    # (o) 5 in-band detections -> pen_count refused
    d, sl = scene([(A2, .9), (far, .8)])
    r = run("o", d, sl, pool="dayl")
    ck(r["refused"] == "pen_count" and not r["links"] and not r["opens"] and r["n_inband"] == 5, f"S4o: {r['refused']}")
    # (p) worker-sized in-band detection on the calf's healthy footprint, calf undetected -> ACCEPTED (known gap)
    worker = _rect(40, 40, 260, 280)
    d, sl = scene([(worker, .6)]); r = run("p", d, sl)
    ck(0 in r["links"], "S4p the worker-over-a-missed-calf case is accepted (documents the known gap)")
    # (q) blob unmatched at a scheduled pass: v5_fixes filters it; v5_fixes off -> v4 opens it
    now = datetime(2024, 4, 23, 12, 0, 0)
    states = [ST_CLOSED, ST_TRACK, ST_TRACK, ST_TRACK]
    op_v4 = v4_step4_opens([big], [], states, [False] * 4, [0.0] * 4, True, now, 300.0)
    cons, refused = reopen_filter([big], [Bm, C, D], "dayl", B, n_inband=3)
    ok, why, refused2 = reopen_considered([big], [Bm, C, D], "dayl", B, n_inband=3)
    op_v5 = v4_step4_opens([big], [], states, [False] * 4, [0.0] * 4, True, now, 300.0, allow=ok)
    ck(op_v4 == [(0, 0, "reopen")] and cons == [] and not refused and op_v5 == [] and ok == [False]
       and why == ["band"] and not refused2, f"S4q: v4 {op_v4} v5 {op_v5} {why}")
    cons, refused = reopen_filter([A, far], [Bm, C, D], "dayl", B, n_inband=5)
    ok, why, refused2 = reopen_considered([A, far], [Bm, C, D], "dayl", B, n_inband=5)
    ck(refused and cons == [] and refused2 and ok == [False, False] and why == ["pen_count"] * 2,
       "S4q pen-count refusal on the scheduled path")
    ok, why, _r = reopen_considered([A2, far], [A, Bm, C], "dayl", B, n_inband=4)
    ck(ok == [False, True] and why == ["conflict", None], f"S4q conflict veto, in-band far calf allowed: {why}")
    # (q2) review r1: a vetoed detection still runs v4's loop -- pending bookkeeping and the ignored_detection branch
    # (its whole-animal scene event gates the night repairs) -- it only never opens / displaces
    allt = [ST_TRACK] * 4; ign = []; pend = [{"mask": big, "wall": now - timedelta(seconds=120), "seen": 1}]
    ok, why, _r = reopen_considered([big], [A, Bm, C, D], "dark", B, n_inband=4)
    op = v4_step4_opens([big], pend, allt, [True] * 4, [0.0] * 4, False, now, 300.0, allow=ok, ignored=ign)
    ck(op == [] and ign == [0] and ok == [False], f"S4q2 a filtered persisted blob reaches ignored_detection: {op} {ign}")
    ign = []
    op = v4_step4_opens([big], [], allt, [True] * 4, [0.0] * 4, False, now, 300.0, allow=ok, ignored=ign)
    ck(op == [] and ign == [], "S4q2 first sighting: pending only (v4: ignored needs a persisted detection)")
    occ4 = [ST_OCC, ST_TRACK, ST_TRACK, ST_TRACK]
    op_v4 = v4_step4_opens([big], pend, occ4, [True] * 4, [30.0, 0, 0, 0], False, now, 300.0)
    op_v5 = v4_step4_opens([big], pend, occ4, [True] * 4, [30.0, 0, 0, 0], False, now, 300.0, allow=ok)
    ck(op_v4 == [(0, 0, "displaced")] and op_v5 == [], f"S4q2 displacement vetoed: v4 {op_v4} v5 {op_v5}")
    cl1 = [ST_CLOSED, ST_TRACK, ST_TRACK, ST_TRACK]          # night: a vetoed first sighting still counts as pending
    pend2 = []
    for di_, (msk, al) in enumerate(((big, [False]), (A2, [True]))):
        o_ = v4_step4_opens([msk], pend2, cl1, [True] * 4, [0.0] * 4, False, now, 300.0, allow=al)
        if di_ == 0:
            ck(o_ == [], "S4q2 night vetoed sighting opens nothing")
            pend2 = [{"mask": big, "wall": now, "seen": 1}]
        else:
            ck(o_ == [(0, 0, "reopen")], f"S4q2 v4 pending semantics kept: second (allowed) sighting opens {o_}")
    # (r) seeding: a blob in the day seed set is filtered; 5 in-band seeds refused
    seeds, refused = seed_filter([(big, .9), (A, .9), (Bm, .9), (C, .9), (D, .9)], "dayl", B)
    ck(not refused and len(seeds) == 4 and all(s[0] is not big for s in seeds), "S4r blob filtered from seeds")
    seeds, refused = seed_filter([(A, .9), (Bm, .9), (C, .9), (D, .9), (far, .9)], "dayl", B)
    ck(refused and seeds == [], "S4r 5 in-band seeds refused")
    # (t) the band comes from constants only: adding an area history changes nothing
    d, sl = scene([(A2, .9)])
    r1 = run("t1", d, sl)
    for v in sl: v["area_hist"] = [1000] * 100; v["day_ref"] = 20000.0
    r2 = run("t2", d, sl)
    ck(r1["links"] == r2["links"] and r1["det_reason"] == r2["det_reason"] and r1["band"] == r2["band"],
       "S4t decision independent of any area history")
    # (u) largest calf at 1.18 x A_REF passes; a 118,440 px merge fails
    ck(in_band(1.18 * B["A_REF_TRAIN"], "dayl", B) and in_band(1.18 * B["A_REF_TRAIN"], "dark", B)
       and not in_band(118440, "dayl", B) and not in_band(118440, "dark", B), "S4u band edges")
    # (v) two dark slots, one link -> sole_dark False, n_dark 2
    sl = [_view(ST_OCC, dark_age=15.0, healthy=A, healthy_age=15.0, cause="collapse"),
          _view(ST_CLOSED, dark_age=40.0, cause="merge_ambiguous"),
          _view(ST_TRACK, live=C), _view(ST_TRACK, live=D)]
    r = run("v", [(A2, .9), (C, .9), (D, .9)], sl)
    ck(0 in r["links"] and r["links"][0]["n_dark"] == 2 and r["links"][0]["sole_dark"] is False, f"S4v: {r['links']}")
    print(f"  S4 dark_step_decide scenes a-v: {sum(1 for _ in out)} runs")
    return out

def _s6_s7(ck):
    ck(strikes_to_close(True, 120) == 2 and strikes_to_close(False, 300) == 3 and strikes_to_close(False, 120) == 6,
       "S6 strikes_to_close day@120=2 night@300=3 night@120=6")
    ck(pending_window(120) == 300.0 and pending_window(300) == 750.0, "S6 pending window 2.5 x interval")
    reg = config_from_flags(dict(REGRESSION_FLAGS))
    ck(reg["strikes_day"] == 2 and reg["strikes_night"] == 3 and reg["pending_night"] == 750.0
       and reg["pending_day"] == 300.0 and not reg["dark_link"] and not reg["dark_open"] and not reg["dark_fire"]
       and reg["is_regression"] and not reg["numeric_change"], "S7 regression flags give v4 values")
    v5 = config_from_flags({})
    ck(v5["strikes_night"] == 6 and v5["dark_link"] and v5["dark_open"] and v5["values"]["detector"] == "rfdetr"
       and v5["values"]["student_ckpt"] == "best_backbone_student_v2.pt" and not v5["is_regression"], "S7 v5 defaults")
    off = config_from_flags({"dark_fire": "off", "dark_link": "on", "dark_open": "on"})
    ck(not off["dark_link"] and not off["dark_open"], "S7 dark_link/dark_open ignored when dark_fire=off")
    for bad in ({"decode": "sequential"}, {"gpu_tf": "on"}, {"detector": "yolo"}, {"dark_fire": "yes"},
                {"nonsense": "1"}, {"from_hhmm": "7am"}, {"partial_repair": "maybe"}):
        try:
            config_from_flags(bad); ck(False, f"S7 accepted {bad}")
        except ValueError:
            pass
    ok = config_from_flags({"decode": "sequential", "allow_numeric_change": "true"})
    ck(ok["numeric_change"], "S7 sequential allowed with allow_numeric_change")
    print("  S6/S7 cadence + flags -> config")

class _FakeSession:
    """TV:130-318 semantics: processed_frames None until the first forward; add_new_frame derives the index from
    len(processed_frames) when none is given; the cond frame (0) is REUSED when its index repeats."""
    def __init__(self):
        self.processed_frames = None; self.cond = {}; self.calls = []
    def add_new_frame(self, pixel_values, frame_idx=None):
        if frame_idx is None:
            frame_idx = len(self.processed_frames) if self.processed_frames is not None else 0
        if self.processed_frames is None: self.processed_frames = {frame_idx: pixel_values}
        else: self.processed_frames[frame_idx] = pixel_values
        return frame_idx

def _fake_sam3(inference_session=None, frame_idx=None, frame=None):
    s = inference_session
    fi = s.add_new_frame(frame, frame_idx)
    if fi == 0 and 0 in s.cond:
        out = ("seed", s.cond[0])                # cond-frame reuse branch (TV:1781-1784)
    else:
        out = ("track", frame)
        if fi == 0: s.cond[0] = frame
    s.calls.append((fi, len(s.processed_frames)))
    return out

def _s8(ck):
    def run(pop, n=30, reopen_at=12):
        seq, stored, outs = [], [], []
        sess = _FakeSession(); sess._tol_fidx = 0
        for k in range(n):
            if k == reopen_at:
                sess = _FakeSession(); sess._tol_fidx = 0
            o = sam3_forward(_fake_sam3, sess, f"px{k}", pop)
            seq.append(sess.calls[-1][0]); stored.append(n_pf(sess)); outs.append(o)
        return seq, stored, outs
    s_pop, st_pop, o_pop = run(True)
    s_off, st_off, o_off = run(False)
    exp = list(range(12)) + list(range(18))
    ck(s_pop == exp and s_off == exp, f"S8 index sequence 0..N-1 reset on reopen: pop {s_pop[:14]} off {s_off[:14]}")
    ck(max(st_pop) == 0 and max(st_off) == 18, f"S8 stored frames: pop max {max(st_pop)} off max {max(st_off)}")
    ck(o_pop == o_off and all(o[0] == "track" for o in o_pop), "S8 outputs identical with pop on/off")
    sess = _FakeSession()                          # the trap: pop WITHOUT an explicit index repeats index 0
    trap = []
    for k in range(5):
        trap.append(_fake_sam3(inference_session=sess, frame=f"px{k}"))
        sess.processed_frames.pop(sess.calls[-1][0], None)
    ck([t[0] for t in trap] == ["track", "seed", "seed", "seed", "seed"], "S8 demonstrates the pop-without-index trap")
    ck(n_pf(None) == 0 and n_pf(_FakeSession()) == 0, "S8 n_pf None-safe")
    print("  S8 pop/index logic with a fake session (TV add_new_frame semantics)")

def _s9(ck):
    state = {"p": "highest"}
    def setp(v): state["p"] = v
    def getp(): return state["p"]
    seen = []
    ck(run_with_precision(setp, getp, lambda: seen.append(getp()) or 7) == 7 and seen == ["high"]
       and getp() == "highest", "S9 inner high, restored highest")
    try:
        run_with_precision(setp, getp, lambda: (_ for _ in ()).throw(ValueError("boom")))
        ck(False, "S9 predictor exception swallowed")
    except ValueError:
        ck(getp() == "highest", "S9 highest restored after a raising predictor")
    def bad_set(v): state["p"] = "high"            # a setter that silently ignores 'highest'
    try:
        run_with_precision(bad_set, getp, lambda: 1); ck(False, "S9 precision drift not detected")
    except RuntimeError:
        pass
    print("  S9 precision toggle restores 'highest' on every path")

def _s10(ck):
    cfg = config_from_flags({})
    shas = {"detector": "d" * 64, "student": "s" * 64, "core": "c" * 64, "source": "x" * 64, "source_norm": "n" * 64}
    base = guard_params(cfg, shas, "A10", 0.35)
    ck(base["source_norm_sha256"] == "n" * 64 and "source_sha256" not in base,
       "S10 the resume guard holds the NORMALISED source digest")
    ck(resume_guard(base, dict(base)) == [], "S10 identical passes")
    alt = {"detector": "sam3", "detector_sha256": "e", "op_threshold": 0.4, "dark_fire": False, "dark_link": False,
           "dark_open": False, "dark_repeat_sec": 30.0, "detect_day_sec": 60.0, "detect_night_sec": 300.0,
           "v5_fixes": False, "partial_repair": False, "numeric_change": True, "decode_class": "sequential",
           "gpu_tf": True, "student_sha256": "t", "core_sha256": "k", "source_norm_sha256": "y", "gpu_class": "A100"}
    miss = []
    for k in RESUME_KEYS:
        cur = dict(base); cur[k] = alt[k]
        try:
            resume_guard(base, cur); miss.append(k)
        except RuntimeError:
            pass
    ck(not miss, f"S10 guard misses {miss}")
    cur = dict(base); cur["gpu_class"] = "A100"
    ck(len(resume_guard(base, cur, allow_mixed_gpu=True)) == 1, "S10 allow_mixed_gpu tolerates the GPU class only")
    cur["v5_fixes"] = False
    try:
        resume_guard(base, cur, allow_mixed_gpu=True); ck(False, "S10 mixed GPU + other diff passed")
    except RuntimeError:
        pass
    s2 = guard_params(config_from_flags({"decode": "seek"}), shas, "A10", 0.35)
    ck(resume_guard(base, s2) == [], "S10 seek == seek_thread")
    print("  S10 resume guard: every guarded key raises; seek == seek_thread; mixed GPU only with the flag")

def _s11(ck):
    dm0, dm1, dm2, rep = (np.zeros((4, 4), bool) for _ in range(4))
    opens = [(0, dm0, "reopen"), (1, dm1, "dark_link"), (2, dm2, "displaced")]
    extra = {1: {"via": "dark_pass"}, 2: {"via": "dark_pass"}}
    on = row_overrides(opens, extra, {3: rep}, True)
    off = row_overrides(opens, extra, {3: rep}, False)
    ck(on[0] is dm0 and on[1] is dm1 and on[2] is dm2 and on[3] is rep, "S11 v5_fixes on: every open + repair overrides")
    ck(0 not in off and off[1] is dm1 and off[2] is dm2 and off[3] is rep,
       "S11 v5_fixes off: the v4 reopen keeps the v4 row; dark actions and repairs still override")
    ck(row_overrides([(0, dm0, "reopen")], {}, {}, False) == {}, "S11 pure v4 pass: no override")
    print("  S11 row overrides (v4 reopen on/off, dark link on OCC, dark open, repair)")

def _s12(ck, s4out):
    ck(events_for_call(True, 0, False) == ["pass_no_detections"] and events_for_call(True, 3, False) == ["pass"]
       and events_for_call(True, 3, True) == ["pass", "dark_pass"] and events_for_call(False, 0, True) == ["dark_pass"]
       and events_for_call(False, 4, True) == ["dark_pass"], "S12 call -> event types")
    ck(counts_clip_pass(True) and not counts_clip_pass(False), "S12 CLIP_PASS counts scheduled calls only")
    r = s4out["a"]
    evs = [ev_dark_onset(0, "0_001", "c.mp4", 1.0, "2024-04-23T12:00:00", "collapse", False, False, 3.2),
           ev_dark_pass("c.mp4", 12.0, "2024-04-23T12:00:10", r, "dark_only", 5, 88.123, {"0": {"state": ST_OCC}}, True),
           ev_dark_link(0, "0_002", r["links"][0], "c.mp4", 12.0),
           ev_dark_open(0, "0_003", {"reason": "reopen", "det_area": 80000, "n_cand": 1, "n_targets": 1}),
           ev_pen_count_refused("dark", 5, [1, 2], [0.5, 0.6]), ev_dark_suspended(0, "c.mp4", 6),
           ev_mem("mem_check", 250, 2.1, 3.0, 1), ev_mem("mem_assert_fail", 500, 7.0, 3.0, 1), ev_resume("c.mp4", [0, 2]),
           ev_dark_open_refused("c.mp4", 3.0, {"reason": "excess_candidates", "n_cand": 2, "n_targets": 1})]
    bad = {e["type"]: check_event(e) for e in evs if check_event(e)}
    ck(not bad, f"S12 builders miss fields {bad}")
    js = [json.loads(json.dumps(e)) for e in evs]
    ck(all(project_v4(e) is None for e in js), "S12 v5-only events drop out of the v4 projection")
    v5pass = {"type": "pass", "clip": "c", "t": 1.0, "forced": None, "n_det": 4, "tracked": 4, "matched": {}, "struck": [],
              "opened": [], "refreshed": [], "rebuilt": False, "dark": None, "n_raw": 4, "n_inband": 4, "det_ms": 9.0,
              "detector": "rfdetr", "pen_refused": False}
    ck(not check_event(v5pass) and tuple(project_v4(v5pass)) == V4_PASS_KEYS, "S12 the scheduled pass keeps every v4 field")
    print("  S12 event schema / call-type rules / v4 projection")

class _FakeCap:
    def __init__(self, n, bad=(), boom_at=None):
        self.n = n; self.bad = set(bad); self.boom = boom_at; self.pos = 0; self.released = False
        self.calls = []; self.lock = threading.Lock(); self.reads_after_release = 0
    def set(self, prop, v):
        with self.lock: self.calls.append(("set", int(v))); self.pos = int(v)
    def read(self):
        if self.released: self.reads_after_release += 1
        with self.lock:
            f = self.pos; self.calls.append(("read", f)); self.pos += 1
        if self.boom is not None and f == self.boom: raise IOError(f"decoder error at {f}")
        time.sleep(0.001)
        if f in self.bad or f >= self.n: return False, None
        return True, np.full((2, 2), f % 251, np.uint8)
    def grab(self):
        with self.lock:
            f = self.pos; self.pos += 1
        return f < self.n
    def retrieve(self):
        return True, np.full((2, 2), (self.pos - 1) % 251, np.uint8)
    def release(self):
        self.released = True

def _s13(ck):
    fidx = list(range(0, 200, 5))
    c1, c2 = _FakeCap(200, bad={25, 100}), _FakeCap(200, bad={25, 100})
    a = [(f, ok, None if fr is None else int(fr[0, 0])) for f, ok, fr in iter_frames(c1, fidx, "seek")]
    b = [(f, ok, None if fr is None else int(fr[0, 0])) for f, ok, fr in iter_frames(c2, fidx, "seek_thread")]
    ck(a == b and c1.calls == c2.calls and (25, False, None) in b, "S13 seek_thread == seek, same calls, ok=False passes")
    c3 = _FakeCap(10 ** 6)
    it = iter_frames(c3, range(0, 10 ** 6, 5), "seek_thread")
    got = 0
    for f, ok, fr in it:
        got += 1
        if got >= 7: break                        # MAX_FRAMES-style early stop
    it.close(); th_alive = any(t.name == "tol_v5_decode" and t.is_alive() for t in threading.enumerate())
    c3.release()
    ck(got == 7 and not th_alive and c3.reads_after_release == 0, "S13 early stop: producer joined before release")
    c4 = _FakeCap(200, boom_at=50)
    try:
        for _ in iter_frames(c4, fidx, "seek_thread"): pass
        ck(False, "S13 producer exception swallowed")
    except IOError:
        pass
    ck(not any(t.name == "tol_v5_decode" and t.is_alive() for t in threading.enumerate()), "S13 no producer left after error")
    c5 = _FakeCap(40)
    sq = [(f, ok) for f, ok, _fr in iter_frames(c5, range(0, 60, 5), "sequential")]
    ck(sq[:8] == [(f, True) for f in range(0, 40, 5)] and all(not ok for f, ok in sq[8:]), f"S13 sequential {sq}")
    print("  S13 decode-thread protocol with a fake capture")

def _s14(ck):
    c = MaskCache()
    a = _rect(10, 10, 20, 20, 64, 64); b = a.copy()
    ck(c.stats(a) == (400, (10, 30, 10, 30)) and len(c) == 1, "S14 measure")
    c.stats(a); ck(c.hits == 1, "S14 hit")
    ids = id(a)
    c._d[ids] = (b, 1, None)                       # simulate a recycled id with a different array
    try:
        c.stats(a); ck(False, "S14 stale entry not detected")
    except RuntimeError:
        pass
    c.clear(); ck(len(c) == 0, "S14 cleared per frame")
    try:
        c.stats(a.astype(np.uint8)); ck(False, "S14 non-bool accepted")
    except TypeError:
        pass
    # an entry keeps its array alive: its id cannot be reused while cached
    c2 = MaskCache(); tmp = _rect(1, 1, 3, 3, 16, 16); k = id(tmp); c2.stats(tmp); del tmp
    others = [np.zeros((16, 16), bool) for _ in range(200)]
    ck(all(id(o) != k for o in others) and c2._d[k][0].sum() == 9, "S14 entry holds a strong reference")
    print("  S14 mask cache: strong reference, identity assert, per-frame clear, bool only")

def _s15(ck):
    B = dict(TRAIN_BAND_REFERENCE)
    cfg = config_from_flags({})
    shas = {"detector": "d", "student": "s", "core": "c", "source": "x", "source_norm": "n"}
    rc = root_config_record(cfg, shas, "A10", B)
    ck(root_config_check(rc, json.loads(json.dumps(rc))) == [], "S15 same config passes")
    ck(set(rc["shas"]) == {"detector", "student", "core", "source_norm"},
       f"S15 ROOT_CONFIG pins the normalised source digest only: {sorted(rc['shas'])}")
    ck(root_config_check(rc, root_config_record(cfg, dict(shas, source="other read basis"), "A10", B)) == [],
       "S15 a raw-only source difference (FUSE read vs SDK export) is not a config change")
    try:
        root_config_check(rc, root_config_record(cfg, dict(shas, source_norm="m"), "A10", B))
        ck(False, "S15 a source code change not detected")
    except RuntimeError:
        pass
    rc_day2 = root_config_record(config_from_flags({"stage": "20240423", "from_hhmm": "1152", "max_frames": "10"}),
                                 shas, "A10", B)
    ck(root_config_check(rc, rc_day2) == [], "S15 stage / window / max_* are exempt")
    for mut in ({"detect_night_sec": "300"}, {"dark_link": "off"}, {"v5_fixes": "off"}, {"student_ckpt": "v1.pt"},
                {"mem_cap_gib": "21"}):
        rc2 = root_config_record(config_from_flags(mut), shas, "A10", B)
        try:
            root_config_check(rc, rc2); ck(False, f"S15 {mut} not detected")
        except RuntimeError:
            pass
    rc3 = root_config_record(cfg, dict(shas, core="other"), "A10", B)
    try:
        root_config_check(rc, rc3); ck(False, "S15 core sha change not detected")
    except RuntimeError:
        pass
    rc4 = root_config_record(cfg, shas, "A100", B)
    ck(len(root_config_check(rc, rc4, allow_mixed_gpu=True)) == 1, "S15 GPU class tolerated with allow_mixed_gpu")
    try:
        root_config_check(rc, rc4); ck(False, "S15 GPU class change not detected")
    except RuntimeError:
        pass
    B2 = dict(B, OVER_HI=1.4)
    try:
        root_config_check(rc, root_config_record(cfg, shas, "A10", B2)); ck(False, "S15 band change not detected")
    except RuntimeError:
        pass
    print("  S15 ROOT_CONFIG guard across days")

def _s16(ck):
    fr = [{"frame_idx": 0, "t_sec": 0.0, "masks": {"0": {"size": [2, 2], "counts": [4]}, "1": None}},
          {"frame_idx": 5, "t_sec": 0.2, "masks": {"0": None, "1": None}}]
    fr2 = json.loads(json.dumps(fr)); fr2[1]["masks"]["1"] = {"size": [2, 2], "counts": [0, 4]}
    eq, n, first = compare_frames(fr, json.loads(json.dumps(fr)))
    ck(eq == n == 2 and first is None, "S16 identical frames")
    eq, n, first = compare_frames(fr, fr2)
    ck(eq == 1 and first["i"] == 1 and first["slots"] == ["1"], f"S16 frame diff located {first}")
    ref = [{"type": "pass", "clip": "c", "t": 1.0, "forced": None, "n_det": 4, "tracked": 4, "matched": {"0": 0.9},
            "struck": [], "opened": [], "refreshed": [], "rebuilt": False},
           {"type": "segment_close", "seg_id": "0_001", "slot": 0, "wall": "2024-04-23T22:00:00", "reason": "occluded_timeout", "n_frames": 3}]
    reg = [dict(ref[0], dark=None, n_raw=None, n_inband=4, det_ms=5.0, detector="sam3", pen_refused=False),
           {"type": "mem_check", "si": 250, "allocated_gib": 1, "reserved_gib": 2, "processed_frames": 1}, ref[1]]
    ok, na, nb, _f = compare_events(ref, reg)
    ck(ok and na == nb == 2, "S16 projected events equal")
    reg2 = [dict(reg[0], struck=[1])] + reg[1:]
    ok, *_ = compare_events(ref, reg2)
    ck(not ok, "S16 event difference detected")
    cov = branch_coverage(ref + [{"type": "segment_open", "reason": "reopen", "wall": "2024-04-23T22:10:00"},
                                 {"type": "partial_repair_skipped"}, {"type": "all_closed"}])
    ck(cov["occluded_timeout"] == 1 and cov["night_pending_reopen"] == 1 and cov["partial_repair*"] == 1
       and cov["all_closed"] == 1 and cov["strike"] == 0, f"S16 coverage {cov}")
    import tempfile                                    # e6_compare_day on real day dirs
    with tempfile.TemporaryDirectory() as td:
        ra, rb = os.path.join(td, "ref"), os.path.join(td, "reg")
        for d, evs in ((ra, ref), (rb, reg)):
            os.makedirs(d)
            with gzip.open(os.path.join(d, "masks_c.json.gz"), "wt") as f:
                json.dump({"clip": "c.mp4", "track_fps": 5.0, "mask_format": "rle_column_major", "frames": fr}, f)
            with open(os.path.join(d, "segments.json"), "w") as f: json.dump({"segments": [{"seg_id": "0_001"}]}, f)
            with open(os.path.join(d, "events.jsonl"), "w") as f: f.write("".join(json.dumps(e) + "\n" for e in evs))
            np.savez_compressed(os.path.join(d, "observations.npz"), emb=np.ones((2, 3), np.float16))
            with open(os.path.join(d, "observations_meta.json"), "w") as f:
                json.dump({"n": 2, "obs": [{"slot": 0}, {"slot": 1}]}, f)
            open(os.path.join(d, ".done_c"), "w").close()
        rep = e6_compare_day(ra, rb)
        ck(rep["equal"] and rep["observations_equal"] and not rep["reg_new_branch"]["v5_only_types"],
           f"S16 e6_compare_day on equal day dirs: {rep.get('reg_new_branch')} obs {rep.get('observations_equal')}")
        with open(os.path.join(rb, "events.jsonl"), "a") as f: f.write(json.dumps({"type": "dark_pass", "clip": "c"}) + "\n")
        rep = e6_compare_day(ra, rb)
        ck(not rep["equal"] and rep["reg_new_branch"]["v5_only_types"] == ["dark_pass"],
           "S16 a v5-only event type in the regression arm fails E6")
        with open(os.path.join(rb, "events.jsonl"), "w") as f: f.write("".join(json.dumps(e) + "\n" for e in reg))
        np.savez_compressed(os.path.join(rb, "observations.npz"), emb=np.zeros((2, 3), np.float16))
        rep = e6_compare_day(ra, rb)
        ck(not rep["equal"] and not rep["observations_equal"], "S16 different observations fail E6")
    print("  S16 E6 comparator (frames / projected events / coverage / obs / new-branch) on synthetic outputs")

def _s17(ck):
    """The normalised source digest, the atomic write-if-absent and the read retry."""
    import tempfile
    base = "# Databricks notebook source\nimport os\n\n# COMMAND ----------\n\nx = 1  # [v5-C1]\n"
    variants = [base, base.replace("\n", "\r\n"), base + "\n\n", base.replace("[v5-C1]\n", "[v5-C1]   \t\n"),
                "\ufeff" + base, base.replace("\n\n", "\n\n\n"), base.encode("utf-8"),
                base.replace("\n", "\r\n").encode("utf-8")]
    ref = source_norm_sha256(base)
    ck(all(source_norm_sha256(v) == ref for v in variants),
       "S17 source digest must ignore CRLF / trailing whitespace / blank lines / BOM")
    ck(source_norm_sha256(base.replace("x = 1", "x = 2")) != ref
       and source_norm_sha256(base.replace("[v5-C1]", "[v5-C2]")) != ref,
       "S17 source digest must change with any code or comment character")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ROOT_CONFIG.json")
        n = 8; bar = threading.Barrier(n); won, errs = [], []
        def writer(i):
            try:
                bar.wait()
                won.append((i, write_json_if_absent(p, {"who": i, "pad": "x" * 20000}, tag=f"t{i}")))
            except Exception as e:                            # noqa: BLE001 -- reported by the check below
                errs.append(repr(e))
        ths = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
        for t in ths: t.start()
        for t in ths: t.join()
        d = read_json_retry(p, tries=2, sleep_s=0.01)
        winners = [i for i, ok in won if ok]
        ck(not errs and len(winners) == 1 and d["who"] == winners[0] and len(d["pad"]) == 20000,
           f"S17 exactly one of {n} racing writers publishes a whole file: winners {winners} errors {errs}")
        ck(sorted(os.listdir(td)) == ["ROOT_CONFIG.json"], f"S17 no temp file left: {sorted(os.listdir(td))}")
        q = os.path.join(td, "nolink.json")                    # FUSE without hard links: the rename fallback
        def nolink(a, b): raise OSError(95, "Operation not supported")
        ck(write_json_if_absent(q, {"a": 1}, tag="x", _link=nolink) is True
           and write_json_if_absent(q, {"a": 2}, _link=nolink) is False and read_json_retry(q)["a"] == 1,
           "S17 rename fallback publishes once without hard links")
        ck(sorted(os.listdir(td)) == ["ROOT_CONFIG.json", "nolink.json"], "S17 the fallback leaves no temp file")
        h = os.path.join(td, "half.json")                      # a reader landing on a half-written file retries
        with open(h, "w") as f: f.write('{"who": ')
        def finish():
            with open(h, "w") as f: f.write('{"who": 7}')
        tm = threading.Timer(0.15, finish); tm.start()
        got = read_json_retry(h, tries=60, sleep_s=0.02); tm.join()
        ck(got == {"who": 7}, f"S17 read retry on a half-visible file: {got}")
        with open(h, "w") as f: f.write("{")
        try:
            read_json_retry(h, tries=3, sleep_s=0.0); ck(False, "S17 a permanently broken file must raise")
        except RuntimeError:
            pass
    print("  S17 source digest; atomic write-if-absent (8 racing writers, rename fallback); read retry")

def _s18(ck):
    """review r1 (fixer): raw vs normalised source shas, telemetry resume / publish, the E1 memory verdict."""
    import tempfile
    crlf = "# Databricks notebook source\r\nx = 1  # [v5-C1]\r\n".encode("utf-8")
    s = source_shas(crlf)
    ck(s["source"] == hashlib.sha256(crlf).hexdigest() and s["source_norm"] == source_norm_sha256(crlf)
       and s["source"] != s["source_norm"] and source_shas(crlf.replace(b"\r\n", b"\n"))["source_norm"] == s["source_norm"],
       "S18 source_shas: `source` = raw sha256 (the pin), `source_norm` EOL-invariant, and the two differ")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "Metrics_track_v5_x.json")
        ck(resume_telemetry(p, []) == (None, []), "S18 no file, no done clips: fresh start")
        try:
            resume_telemetry(p, ["c1"]); ck(False, "S18 done clips without a telemetry file must raise")
        except RuntimeError:
            pass
        with open(p, "w") as f: f.write('{"clips": [{"clip": "c1"}, {"clip": "c2"}')        # truncated
        fast = lambda q: read_json_retry(q, tries=2, sleep_s=0.0)
        try:
            resume_telemetry(p, ["c1"], read=fast); ck(False, "S18 a truncated file with done clips must raise")
        except RuntimeError:
            pass
        ck(resume_telemetry(p, [], read=fast) == (None, []), "S18 a truncated file without done clips: fresh start")
        loc = os.path.join(td, "local.json")
        with open(loc, "w") as f: json.dump({"params": {"x": 1}, "clips": [{"clip": "c1"}, {"clip": "c2"}]}, f)
        how = publish_file(loc, p)
        ck(how in ("replace", "copy") and sorted(os.listdir(td)) == ["Metrics_track_v5_x.json", "local.json"],
           f"S18 publish_file leaves no .part file ({how}): {sorted(os.listdir(td))}")
        old, kept = resume_telemetry(p, ["c2"], read=fast)
        ck(old["params"] == {"x": 1} and kept == [{"clip": "c2"}], f"S18 kept = the done clips' entries: {kept}")
        try:
            resume_telemetry(p, ["c2", "c3"], read=fast); ck(False, "S18 a done clip without an entry must raise")
        except RuntimeError:
            pass
        _rep = os.replace
        def _norename(a, b): raise OSError(95, "Operation not supported")
        try:
            os.replace = _norename
            with open(loc, "w") as f: json.dump({"clips": []}, f)
            how = publish_file(loc, p)
        finally:
            os.replace = _rep
        ck(how == "copy" and read_json_retry(p) == {"clips": []} and not os.path.exists(p + ".part"),
           "S18 publish_file falls back to a plain copy where the mount cannot rename")
    good = {"A": {"floor_slope_mib_per_frame": [5.814, 5.81], "max_processed_frames": 450, "prunes": 36},
            "B": {"floor_slope_mib_per_frame": [0.0, 0.01], "max_processed_frames": 1, "prunes": 36},
            "A2": {"floor_slope_mib_per_frame": [5.814, 5.81], "max_processed_frames": 450, "prunes": 36}}
    ok, crit, fails = e1_memory_verdict(good, 2)
    ck(ok and not fails and crit["A_floor_slope_in"] == [5.3, 6.4], f"S18 E1 verdict passes the expected arms: {fails}")
    def bad(path, v):
        m = json.loads(json.dumps(good)); d = m
        for k in path[:-1]: d = d[k]
        d[path[-1]] = v
        return e1_memory_verdict(m, 2)[0]
    for path, v in ((("A", "floor_slope_mib_per_frame"), [5.0, 5.8]), (("A", "floor_slope_mib_per_frame"), [5.8, 6.7]),
                    (("B", "floor_slope_mib_per_frame"), [0.3, 0.0]), (("B", "floor_slope_mib_per_frame"), [None, 0.0]),
                    (("B", "max_processed_frames"), 3), (("A", "max_processed_frames"), 50), (("A2", "prunes"), 35),
                    (("A", "floor_slope_mib_per_frame"), [5.8]), (("B", "floor_slope_mib_per_frame"), [0.2, float("nan")])):
        ck(not bad(path, v), f"S18 E1 verdict must fail on {path} = {v}")
    ck(not e1_memory_verdict({"A": {"floor_slope_mib_per_frame": [5.4, 5.4], "max_processed_frames": 450, "prunes": 36},
                              "B": {"floor_slope_mib_per_frame": [0.2, 0.2], "max_processed_frames": 1, "prunes": 36},
                              "A2": {"prunes": 36}}, 2)[0], "S18 E1 verdict checks A - B (5.2 < 5.3)")
    print("  S18 source shas raw/normalised; telemetry resume raises on lost evidence; publish_file; E1 memory verdict")

# ---- data tests (S1 real masks, S5a, S5b) ----
_RXT = re.compile(r"-(\d{8})-(\d{6})-")

def _clip_start(stem):
    m = _RXT.search(stem.replace("_", "-"))
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

class _DayRows:
    """Lazy per-clip loader of stored v4v2 rows (masks/<day>/masks_<clip>.json.gz); keeps <= 2 clips decoded."""
    def __init__(self, data, day):
        self.dir = os.path.join(data, "masks", day)
        fs = sorted(glob.glob(os.path.join(self.dir, "masks_*.json.gz")))
        self.clips = sorted(((_clip_start(os.path.basename(p)[6:-8]), os.path.basename(p)[6:-8], p) for p in fs))
        self.cache = {}; self.order = []
    def rows(self, stem):
        if stem in self.cache: return self.cache[stem]
        p = os.path.join(self.dir, f"masks_{stem}.json.gz")
        with gzip.open(p, "rt") as f: d = json.load(f)
        w0 = _clip_start(stem)
        rows = [(w0 + timedelta(seconds=r["t_sec"]), r) for r in d["frames"]]
        self.cache[stem] = rows; self.order.append(stem)
        while len(self.order) > 2: self.cache.pop(self.order.pop(0), None)
        return rows
    def stems_between(self, ta, tb):
        out = []
        for i, (w0, stem, _p) in enumerate(self.clips):
            w1 = self.clips[i + 1][0] if i + 1 < len(self.clips) else w0 + timedelta(hours=1)
            if w0 <= tb and w1 >= ta: out.append(stem)
        return out
    def rows_between(self, ta, tb):
        out = []
        for stem in self.stems_between(ta, tb):
            out += [(w, r, stem) for w, r in self.rows(stem) if ta <= w <= tb]
        return out

def _mask(r, s):
    rl = r["masks"].get(str(s))
    return None if rl is None else rle_decode(rl)

def _area(r, s):
    rl = r["masks"].get(str(s))
    return 0 if rl is None else int(sum(rl["counts"][1::2]))

def _load_events(data, day):
    p = os.path.join(data, "masks", day, "events.jsonl")
    ev = [json.loads(l) for l in open(p)]
    for e in ev:
        w = None
        if "wall" in e:
            w = datetime.fromisoformat(e["wall"])
        elif "clip" in e and "t" in e:
            w = _clip_start(e["clip"][:-4]) + timedelta(seconds=e["t"])
        e["_w"] = w
    return [e for e in ev if e["_w"] is not None]

def _episode_context(e, dr, events, B):
    """Rebuild what v5 would know for one stored v4v2 gap (runtime band rule; cause / cover / merge-near from events)."""
    s = e["slot"]; t0 = datetime.fromisoformat(e["t0"])
    near = dr.rows_between(t0 - timedelta(seconds=LINK_MAX_SEC + 3), t0 + timedelta(seconds=3))
    onset_i = None
    for i in range(1, len(near)):
        if near[i][1]["masks"].get(str(s)) is None and near[i - 1][1]["masks"].get(str(s)) is not None \
                and abs((near[i][0] - t0).total_seconds()) <= 1.5:
            onset_i = i; break
    if onset_i is None: return None
    w_on = near[onset_i][0]
    pool = pool_of_wall(w_on)
    healthy = None
    for w, r, _st in reversed(near[:onset_i]):
        if (w_on - w).total_seconds() > LINK_MAX_SEC: break
        a = _area(r, s)
        if a and in_band(a, pool, B):
            healthy = (w, _mask(r, s)); break
    closes = [x for x in events if x["type"] == "segment_close" and x.get("slot") == s
              and w_on - timedelta(seconds=0.4) <= x["_w"] <= w_on + timedelta(seconds=FRESH_SEC + 0.4)]
    if closes:
        cause = closes[0]["reason"]; t_on = closes[0]["_w"]
    else:
        sc = e.get("start_cause", "")
        cause = {"collapse_guard": "collapse", "sam3_no_mask": "no_mask"}.get(sc)
        if cause is None:
            cause = sc.split(":", 1)[1] if sc.startswith("segment_close:") else ((e.get("closes") or [sc])[0])
        t_on = w_on + timedelta(seconds=FRESH_SEC) if cause in LINK_CAUSES else w_on
    merge_near = any(x["type"] == "merge_resolved" and s in x.get("pair", [])
                     and datetime.fromisoformat(x["onset"]) <= t_on
                     and x["_w"] >= t_on - timedelta(seconds=MERGE_GUARD_SEC) for x in events)
    covered = False
    if healthy is not None:
        r_on = near[onset_i][1]
        covered = max([m_ios_fast(healthy[1], _mask(r_on, k)) for k in range(N_CALVES)
                       if k != s and r_on["masks"].get(str(k)) is not None], default=0.0) >= COVER_IOS
    return {"slot": s, "w_on": w_on, "t_on": t_on, "pool": pool, "healthy": healthy, "cause": cause,
            "merge_near": merge_near, "covered": covered}

def _first_inband(dr, slot, t_from, pool, B, horizon=120.0):
    for w, r, _st in dr.rows_between(t_from - timedelta(seconds=0.3), t_from + timedelta(seconds=horizon)):
        a = _area(r, slot)
        if a and in_band(a, pool, B): return w, r
    return None

def _closed_between(events, slot, ta, tb):
    return [x["reason"] for x in events if x["type"] == "segment_close" and x.get("slot") == slot and ta < x["_w"] <= tb]

def _s5a(ck, data, B, report):
    """Wiring test on the 99 GT-scored 04-22/23 episodes (S5a). NOT safety evidence."""
    E = json.load(open(os.path.join(data, "gaps", "episodes.json")))["episodes"]
    eps = [e for e in E if e["day"] in ("20240422", "20240423") and e.get("gt_pre") is not None and e.get("gt_re") is not None]
    cfg = {"dark_link": True, "dark_open": True, "dark_first_sec": DARK_REPEAT_SEC_DEFAULT, "link_max_sec": LINK_MAX_SEC,
           "max_per_clip": DARK_MAX_PER_CLIP, "min_gap_sec": DARK_MIN_GAP_SEC, "unclaimed_max": UNCLAIMED_MAX,
           "link_contain_ios": LINK_CONTAIN_IOS, "link_conflict_ios": LINK_CONFLICT_IOS, "cover_ios": COVER_IOS}
    out, no_sur, fails = [], [], []
    for day in ("20240422", "20240423"):
        dr = _DayRows(data, day); ev = _load_events(data, day)
        for e in sorted([x for x in eps if x["day"] == day], key=lambda x: x["t0"]):
            ctx = _episode_context(e, dr, ev, B)
            sentinel = not e["same_by_gt"]
            rec = {"day": day, "slot": e["slot"], "t0": e["t0"], "sentinel": sentinel, "phase": e["phase"],
                   "start_cause": e.get("start_cause"), "cls": e.get("cls")}
            if ctx is None:
                rec["result"] = "no_onset_row"; out.append(rec); continue
            s = ctx["slot"]
            rec.update(cause=ctx["cause"], covered=ctx["covered"], merge_near=ctx["merge_near"],
                       healthy=ctx["healthy"] is not None, pool=ctx["pool"])
            tr = e.get("true_re")
            if e["same_by_gt"]:
                k, t_true = e["re_slot"], datetime.fromisoformat(e["re_t"])
            elif tr:
                k, t_true = tr["slot"], datetime.fromisoformat(e["t0"]) + timedelta(seconds=tr["delay"])
            else:
                k, t_true = None, None
            probes = []
            if k is not None:
                sur = _first_inband(dr, k, t_true, ctx["pool"], B)
                if sur is None: no_sur.append((day, e["t0"], s))
                else: probes.append(("true", k, sur))
            else:
                no_sur.append((day, e["t0"], s))
            if sentinel:
                wr = _first_inband(dr, e["re_slot"], datetime.fromisoformat(e["re_t"]), ctx["pool"], B)
                if wr is not None: probes.append(("wrong", e["re_slot"], wr))
            rec["probes"] = []
            for kind, kslot, (w_s, r_s) in probes:
                t_dec = max(w_s, ctx["t_on"] + (timedelta(seconds=cfg["dark_first_sec"]) if ctx["cause"] in LINK_CAUSES else timedelta(0)))
                closed_later = _closed_between(ev, s, ctx["t_on"], t_dec)
                state = ST_CLOSED if (ctx["cause"] not in LINK_CAUSES or closed_later) else ST_OCC
                hv = ctx["healthy"]
                slots = []
                for q in range(N_CALVES):
                    if q == s:
                        slots.append(_view(state, dark_age=(t_dec - ctx["w_on"]).total_seconds() + FRESH_SEC,
                                           healthy=None if hv is None else hv[1],
                                           healthy_age=None if hv is None else (t_dec - hv[0]).total_seconds(),
                                           cause=ctx["cause"], covered=ctx["covered"], merge_near=ctx["merge_near"]))
                    elif q == kslot or r_s["masks"].get(str(q)) is None:
                        slots.append(_view(ST_CLOSED, dark_age=999.0, cause="unknown"))
                    else:
                        slots.append(_view(ST_TRACK, live=_mask(r_s, q)))
                dets = [(_mask(r_s, kslot), 0.9)] + [(v["live_mask"], 0.9) for v in slots if v["state"] == ST_TRACK]
                res = dark_step_decide(dets, set(), slots, ctx["pool"], B, cfg, scheduled=False, sched_rebuilds=False)
                reason = next((rr for b, rr in res["rejected"].get(s, []) if b in (0, None)), None)
                act = "link" if s in res["links"] else ("open" if s in res["opens"] else None)
                sur_area = int(dets[0][0].sum())
                rec["probes"].append({"kind": kind, "det_slot": kslot, "t_det": w_s.isoformat(), "action": act,
                                      "first_reason": reason, "det_area": sur_area})
                if sentinel and act == "link":
                    fails.append(f"sentinel {day} {e['t0']} slot {s}: {kind} detection LINKED")
                if sentinel and reason == "area" and not in_band(sur_area, ctx["pool"], B):
                    fails.append(f"sentinel {day} {e['t0']} slot {s}: rejected only by area on a fragment")
                CACHE.clear()
            out.append(rec)
    n_link = sum(1 for r in out for p in r.get("probes", []) if p["kind"] == "true" and p["action"] == "link")
    n_open = sum(1 for r in out for p in r.get("probes", []) if p["kind"] == "true" and p["action"] == "open")
    reasons = Counter(p["first_reason"] for r in out for p in r.get("probes", []) if p["kind"] == "true" and p["action"] is None)
    print(f"  S5a {len(eps)} GT-scored episodes: true-animal surrogate -> link {n_link}, open {n_open}, "
          f"rejected {dict(reasons)}; no whole surrogate: {len(no_sur)}")
    print("      sentinels (first rejecting reason; expected):")
    expected = {("20240422", "14:00:5"): "merge_ambiguous", ("20240422", "21:00:0"): "merge_ambiguous",
                ("20240423", "12:05:2"): "merge_near", ("20240423", "12:26:0"): "not_contained",
                ("20240423", "20:35:5"): "not_contained", ("20240423", "21:28:0"): "merge_ambiguous"}
    for r in out:
        if not r["sentinel"]: continue
        exp = next((v for (d, hm), v in expected.items() if d == r["day"] and r["t0"][11:18] == hm), "?")
        pr = "; ".join(f"{p['kind']}: {p['action'] or 'no action'} ({p['first_reason']})" for p in r.get("probes", []))
        print(f"        {r['day']} {r['t0'][11:]} slot {r['slot']} cause={r.get('cause')} covered={r.get('covered')} "
              f"merge_near={r.get('merge_near')} | {pr or 'no probe'} | expected first reason: {exp}")
    ck(not fails, "S5a " + "; ".join(fails))
    ck(sum(1 for r in out if r["sentinel"]) == 6, "S5a expects 6 sentinels")
    report["S5a"] = {"episodes": out, "no_whole_surrogate": no_sur, "fails": fails}

def _associate(trk_masks, dets):
    """Tol_03:733-745 replica: Hungarian on 1 - max(IoU, IoS); keep >= ASSOC_MIN. Returns matched det indices."""
    if not trk_masks or not dets: return set()
    S = np.zeros((len(trk_masks), len(dets)), np.float32)
    for a, tm in enumerate(trk_masks):
        for b, (dm, _s) in enumerate(dets):
            S[a, b] = max(m_iou_fast(tm, dm), m_ios_fast(tm, dm))
    ri, cj = linear_sum_assignment(1.0 - S)
    return {int(b) for a, b in zip(ri, cj) if S[a, b] >= ASSOC_MIN}

def _s5b(ck, data, B, report):
    """Wiring test on the stored RF-DETR detections (S5b): pass calls inside the first 60 s of a GT-day
    episode, full dark step (scheduled association first). Lists every link."""
    p = os.path.join(data, "tune_scout", "dets_rfdetr.json.gz")
    with gzip.open(p, "rt") as f: D = json.load(f)
    E = json.load(open(os.path.join(data, "gaps", "episodes.json")))["episodes"]
    eps = [e for e in E if e["day"] in ("20240422", "20240423")]
    calls = defaultdict(list)
    for key in D["rows"]:
        stem, fi = key.split(":"); calls[stem].append(int(fi))
    cfg = {"dark_link": True, "dark_open": True, "dark_first_sec": DARK_REPEAT_SEC_DEFAULT, "link_max_sec": LINK_MAX_SEC,
           "max_per_clip": DARK_MAX_PER_CLIP, "min_gap_sec": DARK_MIN_GAP_SEC, "unclaimed_max": UNCLAIMED_MAX,
           "link_contain_ios": LINK_CONTAIN_IOS, "link_conflict_ios": LINK_CONFLICT_IOS, "cover_ios": COVER_IOS}
    links, n_calls, n_ran = [], 0, 0
    for day in ("20240422", "20240423"):
        dr = _DayRows(data, day); ev = _load_events(data, day)
        for e in sorted([x for x in eps if x["day"] == day], key=lambda x: x["t0"]):
            ctx = _episode_context(e, dr, ev, B)
            if ctx is None: continue
            s = ctx["slot"]
            for stem in dr.stems_between(ctx["w_on"], ctx["w_on"] + timedelta(seconds=60)):
                if not calls.get(stem): continue
                rows = {r["frame_idx"]: (w, r) for w, r in dr.rows(stem)}
                for fi in sorted(calls.get(stem, [])):
                    if fi not in rows: continue
                    w, r = rows[fi]
                    if not (ctx["w_on"] <= w <= ctx["w_on"] + timedelta(seconds=60)): continue
                    n_calls += 1
                    kept = D["rows"][f"{stem}:{fi:07d}"]["kept"]
                    dets = [(rle_decode(k["rle"]), k["score"]) for k in kept]
                    hv = ctx["healthy"]
                    closed_later = _closed_between(ev, s, ctx["t_on"], w)
                    state = ST_CLOSED if (ctx["cause"] not in LINK_CAUSES or closed_later) else ST_OCC
                    slots = []
                    for q in range(N_CALVES):
                        if q == s:
                            slots.append(_view(state, dark_age=(w - ctx["w_on"]).total_seconds() + FRESH_SEC,
                                               healthy=None if hv is None else hv[1],
                                               healthy_age=None if hv is None else (w - hv[0]).total_seconds(),
                                               cause=ctx["cause"], covered=ctx["covered"], merge_near=ctx["merge_near"]))
                        elif r["masks"].get(str(q)) is None:
                            slots.append(_view(ST_OCC, dark_age=30.0, cause="unknown"))
                        else:
                            slots.append(_view(ST_TRACK, live=_mask(r, q)))
                    trk = [v["live_mask"] for v in slots if v["state"] == ST_TRACK]
                    claimed = _associate(trk, dets)
                    res = dark_step_decide(dets, claimed, slots, ctx["pool"], B, cfg, scheduled=True, sched_rebuilds=True)
                    n_ran += int(res["ran"])
                    for ls, L in res["links"].items():
                        links.append({"day": day, "clip": stem, "frame_idx": fi, "wall": w.isoformat(), "slot": ls,
                                      "episode_t0": e["t0"], "episode_slot": s, "same_by_gt": e.get("same_by_gt"),
                                      "ios": L["ios"], "iou": L["iou"], "det_area": L["det_area"], "sole_dark": L["sole_dark"]})
                    CACHE.clear()
    print(f"  S5b stored RF-DETR calls in the first 60 s of a GT-day episode: {n_calls} calls, dark step ran on "
          f"{n_ran}, links {len(links)}")
    for L in links:
        print(f"        link {L['day']} {L['wall'][11:19]} slot {L['slot']} (episode slot {L['episode_slot']} "
              f"t0 {L['episode_t0'][11:]} same_by_gt={L['same_by_gt']}) ios={L['ios']} iou={L['iou']} area={L['det_area']}")
    ck(n_calls > 0, "S5b found no stored RF-DETR call inside any episode (wiring broken?)")
    report["S5b"] = {"n_calls": n_calls, "n_ran": n_ran, "links": links}

def selftest(data=None, quick=False, report_path=None):
    t0 = time.time()
    ck = _Check(); rng = np.random.default_rng(0); report = {}
    data = data if data is not None else _default_data_dir()
    have_data = bool(data) and os.path.isdir(os.path.join(data, "masks")) and not quick
    print(f"tol_v5_core {CORE_VERSION} selftest -- data: {data if have_data else 'SKIPPED (no data)'}")
    B = dict(TRAIN_BAND_REFERENCE)
    if have_data:
        ts_p = os.path.join(data, "tune_scout", "train_summary_segv1d.json")
        if os.path.exists(ts_p):
            Bf = band_from_train_summary(json.load(open(ts_p)), "segv1d_tracked_seg-small_r624")
            ck(Bf["A_REF_TRAIN"] == 87137.0 and Bf["FRAG_LO"] == {"dayl": 0.44, "dark": 0.43} and Bf["OVER_HI"] == 1.32,
               f"band from train_summary.json != the reference values: {Bf['A_REF_TRAIN']} {Bf['FRAG_LO']} {Bf['OVER_HI']}")
            B = Bf
            lo_d, hi_ = band("dayl", B); lo_k, _ = band("dark", B)
            print(f"  band read from train_summary.json: dayl [{lo_d:,.0f}, {hi_:,.0f}] dark [{lo_k:,.0f}, {hi_:,.0f}] px")
            for bad in ({"run_tag": "x"}, {"op_threshold": 0.4}, {"source_days": ["20240418", "20240422"]}, {"resolution": 560}):
                try:
                    band_from_train_summary(dict(json.load(open(ts_p)), **bad), "segv1d_tracked_seg-small_r624")
                    ck(False, f"band provenance guard accepted {bad}")
                except RuntimeError:
                    pass
            tsb = json.load(open(ts_p)); tsb["constants"]["over_hi"]["derived_from"] = ["20240422"]
            try:
                band_from_train_summary(tsb, "segv1d_tracked_seg-small_r624"); ck(False, "band derived_from guard")
            except RuntimeError:
                pass
        else:
            print("  (train_summary_segv1d.json not in the data dir -- band from the documented reference values)")
    real = []
    if have_data:
        for day in ("20240422", "20240423"):
            fs = sorted(glob.glob(os.path.join(data, "masks", day, "masks_*.json.gz")))
            if not fs: continue
            with gzip.open(fs[len(fs) // 2], "rt") as f: d = json.load(f)
            frs = d["frames"]
            for r in frs[:: max(1, len(frs) // 40)]:
                for v in r["masks"].values():
                    if v is not None: real.append(rle_decode(v))
                if len(real) >= (130 if day == "20240422" else 260): break
    _s1_s2(ck, rng, real)
    if have_data: ck(len(real) >= 200, f"S1 needs >= 200 real masks, got {len(real)}")
    _s3(ck)
    s4 = _s4(ck, B)
    _s6_s7(ck); _s8(ck); _s9(ck); _s10(ck); _s11(ck); _s12(ck, s4); _s13(ck); _s14(ck); _s15(ck); _s16(ck); _s17(ck); _s18(ck)
    if have_data:
        _s5a(ck, data, B, report)
        if os.path.exists(os.path.join(data, "tune_scout", "dets_rfdetr.json.gz")):
            _s5b(ck, data, B, report)
        else:
            print("  S5b SKIPPED (tune_scout/dets_rfdetr.json.gz missing)")
    else:
        print("  S5a / S5b SKIPPED (no data) -- G0 requires one full run with data")
    CACHE.clear()
    if report_path:
        with open(report_path, "w") as f: json.dump(report, f, indent=1, default=str)
    print(f"{ck.n} checks, {len(ck.failed)} failed, {time.time() - t0:.0f} s")
    for m in ck.failed: print("  FAIL:", m)
    if ck.failed:
        raise AssertionError(f"tol_v5_core selftest FAILED ({len(ck.failed)} of {ck.n})")
    print("SELFTEST PASSED" + ("" if have_data else " (WITHOUT data: S5a/S5b/real-mask S1 skipped)"))
    return {"checks": ck.n, "have_data": have_data}

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", default=None, help="scratchpad dir holding masks/, gaps/, tune_scout/")
    ap.add_argument("--quick", action="store_true", help="skip the data tests (S5a/S5b, real-mask S1)")
    ap.add_argument("--report", default=None, help="write the S5a/S5b detail to this JSON")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.data, a.quick, a.report)
    else:
        ap.print_help()
