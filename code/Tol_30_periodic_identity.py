# -*- coding: utf-8 -*-
"""Tol · N30 — periodic identity layer (revised L2): within-segment re-ID every T minutes, exemplar
hygiene, reverse correction, pigeonhole-with-flags. Offline, CPU, on the tracker's stored outputs —
NO re-tracking: it consumes segments.json + crop_bands3.npz (1 canonical crop / min / slot, frozen
DINOv3-S 8-band embeddings) + observations_meta.json (10-s mask area / neighbour overlap) + events.jsonl.

Design (what changed vs Tol_15_v2, which decides once per segment at tracker breaks):
  FREQUENCY   every T-min window each live run is scored against the four identity pools with the
              frozen matcher (3 body-part cosines, orientation-max, top-5). Window decision = Hungarian
              over live runs x identities (row margin >= TAU = confident) — used for HOP detection.
  AGGREGATE   a run's LABEL is not the first confident window: every window's similarity row is
  DECISION    accumulated per run, and at every window a joint Hungarian over the concurrent runs'
              AGGREGATE rows decides the labels — with hysteresis: a change to an existing label (or a
              swap between runs) is applied only if it raises the joint similarity by >= TAU; a smaller
              disagreement marks the runs "contested" (review) and holds the labels. Relabelling a run
              applies to ALL its windows, backward and forward = reverse correction; its crops move
              pools. A label whose run never settles is marked "provisional" (never silent).
  HYGIENE     crop gate = Tol_15_v2's (area >= 20k, neighbour IoS < 0.1, merge guard, strike tails)
              + an over-extension gate (area > 1.5x the slot's rolling 2-h median). Pools are causal,
              lagged by GAP_MIN (no near-duplicate self-matches), RECENCY-LIMITED (last POOL_H hours,
              at most POOL_CAP crops) so no identity becomes a hub through a bigger pool, and fed only
              by SETTLED runs, never from contested/chaos windows, never by pigeonholed / inherited /
              hop-born runs.
  HOP         a confident window dissent whose rival identity is FREE (its holder is not live), K_HOP
              consecutive windows, outside tracker chaos = a within-segment hop: the run is split there
              (tail = rival identity, flag "hop"). Dissent toward a HELD identity never flips a window;
              it feeds the aggregate re-decision above.
  PIGEONHOLE  demoted to a tie-breaker: a run with no label takes the one identity the other three
  WITH FLAGS  slots leave free ONLY if that identity is also its appearance rank-1 in most windows
              (flag "pigeonhole"), else UNASSIGNED + "pigeonhole_conflict". Sub-window blips stay
              UNASSIGNED. self_recovery segments within SOFT_MAX_ABSENCE s inherit (flag "soft_inherit").

Scoring vs the frozen GT chains (04-22 reviewed, 04-23 verified): per crop, by phase; SILENT vs flagged
errors; coverage gained where the GT left a segment UNASSIGNED / lost where we abstain. Writes a
Tol_17-compatible relabel JSON (interval rules per slot) for rendering.

Usage: python Tol_30_periodic_identity.py --root <dir with 20240422/ 20240423/> --day 20240422
         [--T 3] [--gt segment_assignments_reviewed.json] [--out_dir ...]
"""
import os, sys, json, argparse, time, re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

N = 4
TOPK = 5
TAU = 0.015            # confident margin / hysteresis (= Tol_15_v2 hard_min_margin)
GAP_MIN = 30           # pool lag (minutes)
POOL_H = 3.0           # pool recency window (hours before the lag point); week mode overrides this —
                       # a 3 h window is EMPTY after the ~6 h overnight recording gap, which would defeat the
                       # whole point of accumulating across days.
# POOL_CAP USED TO BE A CROP COUNT (240). At 60-s crops that never bound (3 h = 180 crops); at 10-s features
# it silently truncated every pool to the last 40 min and manufactured 1,658 SILENT errors on 04-23 that were
# wrongly read as "dense whole-crop features are over-confident". Express the cap in TIME so it is
# cadence-independent, and keep it equal across identities so the anti-hub property survives.
POOL_CAP_SEC = 10800.0 # most recent exemplar span per identity (s)
POOL_MAX_N = 4000      # compute bound only; never the binding constraint at any sane cadence
MIN_Q = 2              # gated crops needed in a window to score a run
K_HOP = 3              # consecutive confident dissents toward a FREE rival, quiet tracker
SETTLE_WIN = 2         # confident agreeing windows ...
SETTLE_MIN = 6.0       # ... spanning at least this many minutes
CHAOS_EVENTS = 3       # tracker events in the last CHAOS_MIN minutes = chaos
CHAOS_MIN = 5
AREA_OVER = 1.5
GATE_AREA = 20000
GATE_NB = 0.10
MERGE_GUARD = 5.0
SOFT_MAX_ABSENCE = 120.0
SOFT_MIN_LINK_IOU = 0.20   # spatial link for a reopen to inherit (= Tol_15_v2)
SEAM_MIN_IOU = 0.50        # SEAM links (2026-09-24, training windows): a later tracking window's SEED segment whose first
SEAM_MAX_ABSENCE = 20.0    # mask sits on an earlier window's last mask across a contiguous clip boundary (mask-IoU
SEAM_MIN_ABSENCE = -10.0   # Hungarian, one-to-one) is the SAME animal. Only link rows carrying seam=true use these;
                           # without such rows every seam code path below is skipped (strict no-op).
                           # Absence bounds (v2): night seeds fire ~10 s into a clip, so the old 10 s cap had 0.34 s of
                           # headroom over the largest observed absence (+9.66 s); negative absence = the two windows'
                           # clock estimates overlap (observed down to -3.15 s). Rejected seam rows are printed.
# The chain P -> S across a seam is ONE animal, so S carries the provenance of the label it inherits: these flags on
# the source run (how ITS label was obtained, when weaker than appearance-in-context) are copied onto seam-inherited
# runs at the end of the post pass, so an uncertain origin is never laundered into a clean-looking seam label.
SEAM_CARRY_FLAGS = ("pigeonhole", "pigeonhole_conflict", "modal", "hop_in", "spatial_inherit", "soft_inherit")
MODAL_MIN_WIN = 3          # confident windows needed before a run may be decided by VOTE ...
MODAL_FRAC = 0.60          # ... and the share of them that must agree
PIGEONHOLE_MODE = "assign" # "assign" = take the pen constraint and flag when appearance dissents; "abstain" = leave UNASSIGNED
OBS_MATCH_SEC = 5.0    # tightened from 15: at a 10-20 s crop cadence a 15 s tolerance lets a crop
                       # inherit the area / neighbour-overlap gate of a DIFFERENT posture
P3, P3R = [1, 2, 3], [5, 6, 7]
CHAOS_TYPES = ("mask_area_collapse", "merge_resolved", "occluded", "segment_open", "segment_close", "strike")
REVIEW_FLAGS = ("pigeonhole", "pigeonhole_conflict", "soft_inherit", "spatial_inherit", "hop_in",
                "contested", "reassigned", "provisional", "modal", "bijection_conflict", "pool_bootstrap", "warm_pool",
                # seam-mode only (never produced without seam links): a seed that no seam links to, after a recording
                # gap / at an unlinkable seam; a seam run whose own evidence disagrees with its inherited label; a
                # seam inherit blocked by another holder of the label.
                "gap_decision", "seam_unlinked", "seam_dissent", "seam_blocked")
# "seam_inherit" itself is deliberately NOT a review flag: an accepted seam link (mask IoU >= SEAM_MIN_IOU, one-to-one,
# across a 0-20 s clip boundary) is the tracker's own continuity, as good as an unbroken segment. Its uncertainty is
# the uncertainty of the label it carries, which travels with it (SEAM_CARRY_FLAGS) or is raised by "seam_dissent".

def phase_of(w):
    if w.hour >= 21 or w.hour < 6: return "night"
    if w.hour >= 20: return "dusk"
    return "day"

# ----------------------------------------------------------------------------- inputs
def load_inputs(d, features="crops", bands="crop_bands3.npz"):
    segj = json.load(open(os.path.join(d, "segments.json")))
    segs = {}
    for r in segj["segments"]:
        r = dict(r); r["t_start"] = datetime.fromisoformat(r["t_start_wall"]); r["t_end"] = datetime.fromisoformat(r["t_end_wall"])
        segs[r["seg_id"]] = r
    merge_walls, strikes, density = [], defaultdict(list), Counter()
    evp = os.path.join(d, "events.jsonl")
    if os.path.exists(evp):
        rx = re.compile(r'[-_](\d{8})[-_](\d{6})[-_]')
        for line in open(evp, encoding="utf-8", errors="ignore"):
            try: e = json.loads(line)
            except Exception: continue
            t = e.get("type"); w = None
            if e.get("wall"):
                try: w = datetime.fromisoformat(e["wall"])
                except Exception: w = None
            if w is None and e.get("clip"):
                m = rx.search(e["clip"])
                if m: w = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") + timedelta(seconds=float(e.get("t", 0)))
            if w is None: continue
            if t == "merge_resolved": merge_walls.append(w)
            if t == "strike" and e.get("seg_id"): strikes[e["seg_id"]].append((w, int(e.get("n", 1))))
            if t in CHAOS_TYPES: density[(w.hour, w.minute)] += 1
    strike_first = {}
    for sid, evs in strikes.items():
        if segs.get(sid, {}).get("close_reason") != "detector_contradiction": continue
        evs.sort(); starts = [w for w, n in evs if n == 1]
        strike_first[sid] = starts[-1] if starts else evs[0][0]
    obs_by_seg, obs_by_slot = defaultdict(list), defaultdict(list)
    op = os.path.join(d, "observations_meta.json")
    if os.path.exists(op):
        for o in json.load(open(op))["obs"]:
            rec = (datetime.fromisoformat(o["wall"]), float(o.get("area", 0)), float(o.get("max_neighbor_ios", 0)))
            obs_by_seg[o["seg_id"]].append(rec)
            if o["seg_id"] in segs: obs_by_slot[int(segs[o["seg_id"]]["slot"])].append(rec)
        for k in obs_by_seg: obs_by_seg[k].sort()
        for k in obs_by_slot: obs_by_slot[k].sort()
    slot_t = {k: np.array([e[0].timestamp() for e in v]) for k, v in obs_by_slot.items()}
    if features == "obs":
        # the tracker's OWN 10-s masked-crop embeddings (embed_robust: same canonical_crop, augmentation-
        # averaged, whole-crop 384-d). 6x denser than the saved crops, so windows below 3 min are scorable.
        CE = np.load(os.path.join(d, "observations.npz"))["emb"].astype(np.float32)
        CE /= (np.linalg.norm(CE, axis=1, keepdims=True) + 1e-8)
        recs = json.load(open(os.path.join(d, "observations_meta.json")))["obs"]
        assert len(recs) == len(CE), f"obs/emb mismatch {len(recs)} vs {len(CE)}"
        crops = []
        for i, o in enumerate(recs):
            sid = o["seg_id"]
            if sid not in segs: continue
            w = datetime.fromisoformat(o["wall"]); slot = int(o["slot"])
            area = float(o.get("area", 0)); nb = float(o.get("max_neighbor_ios", 0))
            over = False
            if slot in slot_t:
                lo = int(np.searchsorted(slot_t[slot], (w - timedelta(hours=2)).timestamp())); hi = int(np.searchsorted(slot_t[slot], w.timestamp()))
                ref = [a_ for (_, a_, _) in obs_by_slot[slot][lo:hi] if a_ >= GATE_AREA]
                if len(ref) >= 30 and area > AREA_OVER * float(np.median(ref)): over = True
            gated = (area >= GATE_AREA and nb < GATE_NB and not over
                     and not any(abs((w - mw).total_seconds()) <= MERGE_GUARD for mw in merge_walls)
                     and not (sid in strike_first and w >= strike_first[sid]))
            crops.append(dict(i=i, sid=sid, slot=slot, w=w, area=area, over=over, gated=bool(gated), phase=phase_of(w)))
        crops.sort(key=lambda c: c["w"])
        return segs, crops, CE, density
    z = np.load(os.path.join(d, bands), allow_pickle=True)
    paths = [str(p).replace("\\", "/") for p in z["paths"]]
    CE = z["emb"].astype(np.float32); CE /= (np.linalg.norm(CE, axis=2, keepdims=True) + 1e-8)
    crops = []
    for i, p in enumerate(paths):
        sid = os.path.basename(os.path.dirname(p))
        if sid not in segs: continue
        s = segs[sid]; hh = os.path.basename(p)[:6]; slot = int(s["slot"])
        w = s["t_start"].replace(hour=int(hh[:2]), minute=int(hh[2:4]), second=int(hh[4:6]), microsecond=0)
        if w < s["t_start"] - timedelta(hours=1): w += timedelta(days=1)
        area, nb = None, None
        ol = obs_by_seg.get(sid, [])
        if ol:
            j = min(range(len(ol)), key=lambda k: abs((ol[k][0] - w).total_seconds()))
            if abs((ol[j][0] - w).total_seconds()) <= OBS_MATCH_SEC: area, nb = ol[j][1], ol[j][2]
        over = False
        if area is not None and slot in slot_t:
            lo = int(np.searchsorted(slot_t[slot], (w - timedelta(hours=2)).timestamp())); hi = int(np.searchsorted(slot_t[slot], w.timestamp()))
            ref = [a_ for (_, a_, _) in obs_by_slot[slot][lo:hi] if a_ >= GATE_AREA]
            if len(ref) >= 30 and area > AREA_OVER * float(np.median(ref)): over = True
        gated = (area is not None and area >= GATE_AREA and nb < GATE_NB and not over
                 and not any(abs((w - mw).total_seconds()) <= MERGE_GUARD for mw in merge_walls)
                 and not (sid in strike_first and w >= strike_first[sid]))
        crops.append(dict(i=i, sid=sid, slot=slot, w=w, area=area, over=over, gated=bool(gated), phase=phase_of(w)))
    crops.sort(key=lambda c: c["w"])
    return segs, crops, CE, density

VARIANT = "parts3"                     # "parts3" = frozen matcher; "whole" = band 0/4 only (single-pass)

def load_week(root, days, features, bands, week_map):
    """Merge N days into ONE continuous timeline so the exemplar pool ACCUMULATES across days.

    Why this matters: run per-day, every day starts with an empty pool and must re-bootstrap from its own
    day-start seeds. 04-26's seeding produced only THREE seeds, so identity 3 had no exemplars at all and was
    unmatchable for 11.75 h. With the week merged, that identity carries four prior days of exemplars and the
    missing seed is a non-event. It is also what the paper's section 2.5 actually specifies: the pool is
    persisted per identity across the week, not rebuilt daily.

    Identity space: each day's LOCAL identities are mapped to GLOBAL via `week_map` (verified to compose from
    the measured night-gap permutations), so the accumulated pool is per ANIMAL, not per day-slot. Segment ids
    are namespaced "<day>/<seg_id>" because they collide across days ("0_001" exists in every day).
    """
    SEGS, CROPS, CEs, DENS = {}, [], [], Counter()
    off = 0
    for day in days:
        d = os.path.join(root, day)
        segs, crops, CE, density = load_inputs(d, features, bands)
        for sid, r in segs.items():
            r = dict(r); r["day"] = day; r["local_seg"] = sid
            r["seg_id"] = f"{day}/{sid}"      # the record carries its own id; namespace it too or every
            SEGS[r["seg_id"]] = r             # lookup that goes through seg["seg_id"] collides across days
        for c in crops:
            c = dict(c); c["sid"] = f"{day}/{c['sid']}"; c["i"] = c["i"] + off; c["day"] = day
            CROPS.append(c)
        CEs.append(CE); off += len(CE)
        for k, v in density.items(): DENS[k] += v
    CE = np.concatenate(CEs, axis=0) if CEs[0].ndim == 2 else np.concatenate(CEs, axis=0)
    CROPS.sort(key=lambda c: c["w"])
    print(f"week merged: {len(days)} days, {len(SEGS)} segments, {len(CROPS)} crops, CE {CE.shape}")
    return SEGS, CROPS, CE, DENS

def pair_sim(CE, q_idx, x_idx):
    if CE.ndim == 2:                       # observation features: plain cosine (already aug-averaged)
        S = CE[q_idx] @ CE[x_idx].T
    elif VARIANT == "whole":               # isolate the FEATURE from the AUGMENTATION: single-pass whole crop
        Q = CE[q_idx][:, 0, :]
        S = np.maximum(Q @ CE[x_idx][:, 0, :].T, Q @ CE[x_idx][:, 4, :].T)
    else:                                  # frozen matcher: 3 body-part cosines, orientation-max
        Q = CE[q_idx][:, P3, :].reshape(len(q_idx), -1)
        X0 = CE[x_idx][:, P3, :].reshape(len(x_idx), -1); X1 = CE[x_idx][:, P3R, :].reshape(len(x_idx), -1)
        S = np.maximum(Q @ X0.T, Q @ X1.T) / 3.0
    k = min(TOPK, S.shape[1])
    return float(np.sort(S, axis=1)[:, -k:].mean())

# ----------------------------------------------------------------------------- ANCHOR / EVEN-POOL EXTENSION (2026-09-26)
# Anchor-pool re-ID extension (design amended 2026-09-26). EVERYTHING in this block is OFF unless one
# of --anchors / --temporal even / --boot none is passed. With none of them every code path below is the production path
# (NO-OP-1: 7-day test week, NO-OP-2: training windows with seam links — output byte-identical to the pre-patch file
# 329ccf51; proofs in train_gt/anchor_ablation/noop/, 2026-09-27). Provenance (EXT_PROV) and the per-window diagnostics
# are written to the outputs ONLY when an extension flag is on, so production output files are unchanged.
#   s(run, g) = w * s_anchor(run, g) + (1 - w) * s_temporal(run, g)       (convex; w = 1.0 is "anchors only", arm A1)
#   s_anchor  = mean over the window's query crops of the mean of each crop's top-TOPK similarities to the FROZEN 04-18/19
#               anchors of global identity g (anchors of coat c = pi^-1(g)); same frozen matcher as the temporal side
#               (parts-3, orientation-max, /3) — precomputed per crop (anchor_mean5) because it never changes.
#   s_temporal = pair_sim against the temporal pool: "recency" = production (lagged, POOL_H recency, POOL_MAX_N) or "even"
#               (even_select: lagged, whole history, hour-of-day stratified, equal cap per identity, deterministic).
# Blend windows fall back to s_anchor for ALL four identities while any identity has < TOPK visible temporal exemplars
# (window record anchor_only=True). With anchors on, every identity is always scorable, so the starved machinery
# (pool_bootstrap feed exception, pass-2 warm_pool) is disabled. Downstream logic is untouched.
ANCHOR_S = None        # [n_crops, N] float64: per-crop anchor score in GLOBAL identity space, rows aligned with CE; None = off
ANCHOR_W = None        # blend weight w in (0, 1]
TEMPORAL = "recency"   # "recency" (production) | "even"
EVEN_CAP = None        # even pool: total exemplars per identity (None -> POOL_MAX_N, production's per-identity count cap)
BOOT = "seeds"         # "seeds" (production: the first day's seeds bootstrap) | "none" (anchor arms: no seed bootstrap)
_EXT = False           # True iff any extension is active -> pair_sim_ext (bitwise-equal precomputed parts-3 path)
_CE3 = None            # (CE[:, P3] flat, CE[:, P3R] flat), float32 C-contiguous, built once when _EXT

# ----------------------------------------------------------------------------- NON-CALF FILTER (2026-09-29, --noncalf)
# Decision 2026-09-29: on 04-23 15:38:51-15:49:20 tracker v5 opened seg 1_012 on a FARM WORKER (an RF-DETR false
# positive on a person). An OFFLINE filter, no re-track: a segment whose gated crops look like NONE of the four calves is
# NON-CALF — max over identities of its ANCHOR_S rows (the frozen 04-18/19 anchors) below a crop threshold for a majority of
# its gated crops. Constants: Tol_53_noncalf_threshold.py, derived from 04-18/19 ONLY (leave-one-day-out anchors), frozen in
# noncalf_threshold_v1.json. A NON-CALF segment is removed from the identity layer BEFORE it runs (it never opens a run: no
# label, never feeds a pool, invisible to hop / aggregate / pigeonhole / modal / inherit / bijection, so it can neither
# consume nor block an identity; the window grid is pinned to the unfiltered crop span) and is written to the relabel JSON
# under "hidden" (Tol_17 draws NOTHING for that slot there). A hidden track is also a slot-continuity BARRIER (review
# 2026-09-29): a self_recovery segment right after it may not soft-inherit ACROSS it from the calf before it (without the
# barrier the filtered prev_seg would skip the hidden track; a 10-gated-crop hidden track can be < SOFT_MAX_ABSENCE long at
# the interleaved 15-s + 60-s dense cadence). OFF unless --noncalf (which requires --anchors): without it no line of this
# block runs and every output is byte-identical to 299c3686.
# COLOUR MODE (2026-09-30, decision "exclude colour except for white/black should be a strong filter"; the anchor score
# above FAILED acceptance: the worker's median 0.663 sits inside the calves' range). --noncalf also accepts the Tol_54 colour
# json (noncalf_colour_v1.json; "mode": "colour", or inferred from its statistic "coloured_frac" + "frac_above"). The per-crop
# statistic is read from <root>/<day>/<statistic.npz> (crop_colour_v1.npz, Tol_54 --score) whose 'paths' array must EQUAL
# the day's bands npz 'paths' (same strings, same order; any difference is fatal), so its rows align with CE / the crops.
# A segment is NON-CALF iff >= min_gated_crops gated crops of which >= frac_above have coloured_frac > crop_threshold (strict
# >, as the frozen json states; a NaN statistic is gated but never above) = Tol_54.colour_noncalf_segments. Everything
# downstream (removal before the layer, pinned window grid, continuity barriers, relabel "hidden", flags) is the machinery
# above, unchanged. Colour mode does not need --anchors (it reads no anchor score); with anchors its provenance also sits
# in the "extension" block. The anchor mode is byte-identical to a36946cb; without --noncalf no line of either mode runs.
NONCALF = None         # the loaded non-calf json (Tol_53 threshold / Tol_54 colour) when --noncalf; None = off
NONCALF_MODE = None    # "anchor" | "colour" when --noncalf; None = off
COLOUR_CF = None       # colour mode: [n_rows] float64 coloured_frac, rows aligned with CE (per-day npz in day order)

def noncalf_mode(J, path):
    """"anchor" (Tol_53 threshold json: frac_below + score) or "colour" (Tol_54 json: frac_above + statistic coloured_frac).
    An explicit "mode" key wins (and must agree with the fields); a json that is both or neither is refused."""
    is_c = (J.get("statistic") or {}).get("name") == "coloured_frac" and "frac_above" in J
    is_a = "frac_below" in J and "score" in J
    m = J.get("mode")
    if m is not None:
        if m not in ("anchor", "colour") or (m == "colour" and not is_c) or (m == "anchor" and not is_a):
            raise SystemExit(f"--noncalf {path}: mode {m!r} does not match its fields (colour: frac_above + statistic "
                             f"coloured_frac; anchor: frac_below + score)")
        return m
    if is_c == is_a:
        raise SystemExit(f"--noncalf {path}: neither (or both) an anchor json (frac_below + score) and a colour json "
                         f"(frac_above + statistic coloured_frac) — refused")
    return "colour" if is_c else "anchor"

def load_noncalf_colour(path):
    """the Tol_54 colour json -> dict; refused unless derived from the training days only, with sane constants and a plain
    statistic npz basename (it is read next to each day's bands npz)."""
    J = json.load(open(path, encoding="utf-8"))
    for k in ("crop_threshold", "min_gated_crops", "frac_above", "statistic"):
        if k not in J: raise SystemExit(f"--noncalf {path}: no {k!r}")
    st = J["statistic"]; npz = st.get("npz")
    if st.get("name") != "coloured_frac" or not isinstance(npz, str) or not npz.endswith(".npz") \
            or os.path.basename(npz) != npz or npz in (".npz",):
        raise SystemExit(f"--noncalf {path}: statistic {st.get('name')!r} / npz {npz!r} (need coloured_frac + a plain .npz name)")
    if not J.get("derived_from") or set(J["derived_from"]) - {"20240418", "20240419"} \
            or not {f"202404{d}" for d in range(22, 29)} <= set(J.get("not_from") or []):
        raise SystemExit(f"LEAKAGE GUARD: --noncalf {path} derived_from {J.get('derived_from')} / not_from {J.get('not_from')}")
    if not (0.0 < float(J["crop_threshold"]) < 1.0) or not (0.0 < float(J["frac_above"]) <= 1.0) or int(J["min_gated_crops"]) < 1:
        raise SystemExit(f"--noncalf {path}: crop_threshold {J['crop_threshold']} / frac_above {J['frac_above']} / "
                         f"min_gated_crops {J['min_gated_crops']} out of range")
    return J

def load_colour_frac(root, days, bands, npz_name, n_rows):
    """colour mode: (coloured_frac float64 [n_rows] aligned with CE, {day: npz sha256}). <root>/<day>/<npz_name> must hold
    EXACTLY the day's bands npz 'paths' (same strings, same order) and one coloured_frac per path; days concatenated in the
    order load_week concatenates CE. Any mismatch is fatal (a misaligned row would hide / keep the wrong track)."""
    parts, shas = [], {}
    for day in days:
        cp, bp = os.path.join(root, day, npz_name), os.path.join(root, day, bands)
        if not os.path.exists(cp):
            raise SystemExit(f"--noncalf (colour): {cp} missing — score the day's crops_dense with Tol_54 --score first")
        with np.load(cp, allow_pickle=False) as z:
            if not {"paths", "coloured_frac"} <= set(z.files): raise SystemExit(f"{cp}: needs 'paths' + 'coloured_frac', has {z.files}")
            P = [str(p) for p in z["paths"]]; cf = np.asarray(z["coloured_frac"], dtype=np.float64)
        with np.load(bp, allow_pickle=True) as z: B = [str(p) for p in z["paths"]]
        if P != B or cf.shape != (len(B),):
            bad = next((j for j, (x, y) in enumerate(zip(P, B)) if x != y), min(len(P), len(B)))
            raise SystemExit(f"--noncalf (colour): {cp} is not row-aligned with {bp} ({len(P)} vs {len(B)} paths, "
                             f"{cf.shape[0]} values; first difference at row {bad}) — rescore with Tol_54")
        parts.append(cf); shas[day] = sha256_file(cp)
    cf = np.concatenate(parts) if parts else np.zeros(0)
    if len(cf) != n_rows: raise SystemExit(f"--noncalf (colour): {len(cf)} colour rows vs {n_rows} CE rows")
    return cf, shas

def colour_noncalf_segments(crops, cf, cfg):
    """{sid: dict(n_gated, n_above, frac_above, cf_median)} for every NON-CALF segment: >= cfg["min_gated_crops"] gated crops
    of which >= cfg["frac_above"] have cf[c["i"]] > cfg["crop_threshold"] (strict >; NaN never counts as above but DOES count
    as a gated crop). = Tol_54.colour_noncalf_segments (the frozen rule's reference; prod_relabel_v5's SELFTEST cross-checks)."""
    thr = float(cfg["crop_threshold"]); mg = int(cfg["min_gated_crops"]); fr = float(cfg["frac_above"])
    by = defaultdict(list)
    for c in crops:
        if c["gated"]: by[c["sid"]].append(float(cf[c["i"]]))
    out = {}
    for sid, v in by.items():
        n = len(v); na = sum(1 for x in v if x > thr)
        if n >= mg and na >= fr * n:
            fv = [x for x in v if np.isfinite(x)]
            out[sid] = dict(n_gated=n, n_above=na, frac_above=round(na / n, 4),
                            cf_median=round(float(np.median(fv)), 5) if fv else None)
    return out

def noncalf_segments(crops, S, cfg):
    """{sid: dict(n_gated, n_below, frac_below, score_median)} for every NON-CALF segment: >= cfg["min_gated_crops"] gated
    crops of which >= cfg["frac_below"] have max_k S[i, k] < cfg["crop_threshold"] (strict <). S = ANCHOR_S, rows aligned
    with CE (the max over identities does not depend on pi). Pure; Tol_53's SELFTEST cross-checks it against its reference."""
    thr = float(cfg["crop_threshold"]); mg = int(cfg["min_gated_crops"]); fr = float(cfg["frac_below"])
    by = defaultdict(list)
    for c in crops:
        if c["gated"]: by[c["sid"]].append(float(np.max(S[c["i"]])))
    out = {}
    for sid, v in by.items():
        n = len(v); nb = sum(1 for x in v if x < thr)
        if n >= mg and nb >= fr * n:
            out[sid] = dict(n_gated=n, n_below=nb, frac_below=round(nb / n, 4), score_median=round(float(np.median(v)), 5))
    return out

def load_noncalf(path, a_sha):
    """the Tol_53 threshold json -> dict; refused unless derived from the training days only, for THIS anchor pool, with the
    frozen matcher and this run's TOPK (the score it thresholds must be the score ANCHOR_S holds)."""
    J = json.load(open(path, encoding="utf-8"))
    for k in ("crop_threshold", "min_gated_crops", "frac_below"):
        if k not in J: raise SystemExit(f"--noncalf {path}: no {k!r}")
    if J.get("anchor_pool", {}).get("sha256") != a_sha:
        raise SystemExit(f"--noncalf {path} was derived for anchor pool {str(J.get('anchor_pool', {}).get('sha256'))[:12]}, not {a_sha[:12]}")
    sc = J.get("score", {})
    if sc.get("matcher") != "parts3_orientmax_div3" or int(sc.get("topk", -1)) != TOPK:
        raise SystemExit(f"--noncalf {path}: matcher {sc.get('matcher')} / topk {sc.get('topk')} != parts3_orientmax_div3 / {TOPK}")
    if not J.get("derived_from") or set(J["derived_from"]) - {"20240418", "20240419"} \
            or not {f"202404{d}" for d in range(22, 29)} <= set(J.get("not_from") or []):
        raise SystemExit(f"LEAKAGE GUARD: --noncalf {path} derived_from {J.get('derived_from')} / not_from {J.get('not_from')}")
    if not (0.0 < float(J["frac_below"]) <= 1.0) or int(J["min_gated_crops"]) < 1:
        raise SystemExit(f"--noncalf {path}: frac_below {J['frac_below']} / min_gated_crops {J['min_gated_crops']} out of range")
    return J

def sha256_file(p, bs=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()

def even_select(entries, lag, cap):
    """EVEN temporal pool (2026-09-26, "sample evenly during the day"). Candidates = every exemplar fed so far that
    is at least GAP_MIN old (wall <= lag) — the same causal, settled-only, hygiene-filtered entries production draws from,
    but over the WHOLE history instead of the last POOL_H hours. Stratified by CLOCK HOUR (hour of day 0-23, pooled over all
    days so far): equal quota per hour bin by water-filling (a bin holding fewer exemplars than the quota gives all it has
    and the rest is shared equally by the other bins; a +1 remainder goes to the earliest open bins), so every identity
    gets min(cap, available) exemplars — the anti-hub property of an equal cap. Within a bin the exemplars are taken
    EVENLY SPACED in time order (ranks floor((j + 0.5) * n / m)). Deterministic, no randomness. Returns crop indices."""
    el = [(w_, i) for (w_, i, _, _) in entries if w_ <= lag]
    el.sort()
    if len(el) <= cap: return [i for _, i in el]
    bins = defaultdict(list)
    for w_, i in el: bins[w_.hour].append(i)
    hrs = sorted(bins); quota = {}; left = cap; open_ = list(hrs)
    while open_:
        q = left // len(open_)
        small = [h for h in open_ if len(bins[h]) <= q]
        if not small:
            r = left - q * len(open_)
            for j, h in enumerate(open_): quota[h] = q + (1 if j < r else 0)
            break
        for h in small: quota[h] = len(bins[h]); left -= len(bins[h])
        open_ = [h for h in open_ if h not in small]
    out = []
    for h in hrs:
        b = bins[h]; m = quota[h]
        out.extend(b if m >= len(b) else [b[int((j + 0.5) * len(b) / m)] for j in range(m)])
    return out

def pair_sim_ext(CE, q_idx, x_idx):
    """pair_sim with the parts-3 bands pre-flattened once (_CE3). Same arrays, same expression, same top-k => bitwise-equal
    to pair_sim (EXT-EQ, train_gt/anchor_ablation/noop/: the full layer with _EXT forced on and production settings
    reproduces the production output byte for byte); only the per-call CE[x_idx] copy of all 8 bands is avoided."""
    if _CE3 is None or CE.ndim != 3 or VARIANT != "parts3": return pair_sim(CE, q_idx, x_idx)
    Q = _CE3[0][q_idx]; X0 = _CE3[0][x_idx]; X1 = _CE3[1][x_idx]
    S = np.maximum(Q @ X0.T, Q @ X1.T) / 3.0
    k = min(TOPK, S.shape[1])
    return float(np.sort(S, axis=1)[:, -k:].mean())

def build_ce3(CE):
    n = len(CE)
    return (np.ascontiguousarray(CE[:, P3, :].reshape(n, -1)), np.ascontiguousarray(CE[:, P3R, :].reshape(n, -1)))

def load_band_matrix(path):
    """a band cache exactly as load_inputs reads it (float32, per-band L2 normalisation) -> (paths, CE)"""
    z = np.load(path, allow_pickle=True)
    paths = [str(p).replace("\\", "/") for p in z["paths"]]
    E = z["emb"].astype(np.float32); E /= (np.linalg.norm(E, axis=2, keepdims=True) + 1e-8)
    return paths, E

def anchor_mean5(Q3, A3, A3R, ident, topk=None, chunk=1024):
    """[nq, N]: per query crop (parts-3 flattened), the mean of its top-k similarities max(<q,x_P3>, <q,x_P3R>)/3 over the
    anchors of each identity column c (ident == c) = pair_sim([q], A_c) up to float summation order (CACHE-EQ <= 1e-6)."""
    k = topk or TOPK
    out = np.zeros((len(Q3), N), np.float32)
    for c in range(N):
        sel = np.flatnonzero(ident == c)
        assert len(sel) >= k, f"identity {c}: only {len(sel)} anchors"
        A0 = np.ascontiguousarray(A3[sel].T); A1 = np.ascontiguousarray(A3R[sel].T)
        for s0 in range(0, len(Q3), chunk):
            q = Q3[s0:s0 + chunk]
            S = np.maximum(q @ A0, q @ A1) / 3.0
            out[s0:s0 + chunk, c] = np.sort(np.partition(S, -k, axis=1)[:, -k:], axis=1).mean(axis=1)
    return out

def load_anchor_pool(path, load=True):
    """the frozen anchor pool (Tol_50) -> (A3, A3R float32 flattened, ident int, sha256). Refuses a file whose sha256 is not
    the one its manifest ("output": {"sha256": ...}) recorded at build time. load=False: verify only -> (None, None, None, sha)
    (the embeddings, ~0.7 GB as float32, are then read only if a per-band score cache is missing)."""
    a_sha = sha256_file(path)
    mp = path[:-4] + ".manifest.json" if path.endswith(".npz") else path + ".manifest.json"
    if not os.path.exists(mp): raise SystemExit(f"anchor pool manifest missing: {mp}")
    man = json.load(open(mp, encoding="utf-8"))
    if man.get("output", {}).get("sha256") != a_sha:
        raise SystemExit(f"anchor pool sha256 {a_sha} != manifest {man.get('output', {}).get('sha256')} — refusing a changed/unfrozen pool")
    if not load: return None, None, None, a_sha
    z = np.load(path, allow_pickle=False)
    E = z["emb"].astype(np.float32); E /= (np.linalg.norm(E, axis=2, keepdims=True) + 1e-8)
    A3, A3R = build_ce3(E)
    return A3, A3R, z["ident_train"].astype(np.int64), a_sha

_POOL_MEM = {}         # anchor sha -> loaded pool (only on a score-cache miss)

def anchor_scores_for_band(band_path, anchors_path, cache_dir, pool=None):
    """per-crop anchor scores [n, N] in COAT space (training identity key) for one band file, cached by (anchor sha, band
    sha, topk, matcher) under cache_dir. Returns (mean5, band_sha, anchor_sha, cache_path)."""
    b_sha = sha256_file(band_path)
    a_sha = pool[3] if pool is not None else sha256_file(anchors_path)
    os.makedirs(cache_dir, exist_ok=True)
    cp = os.path.join(cache_dir, f"anchor_mean5_a{a_sha[:16]}_b{b_sha[:16]}_k{TOPK}.npz")
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=False)
        assert str(z["anchor_sha256"]) == a_sha and str(z["band_sha256"]) == b_sha and int(z["topk"]) == TOPK \
            and str(z["matcher"]) == "parts3_orientmax_div3", f"stale anchor cache {cp}"
        return z["mean5"], b_sha, a_sha, cp
    if pool is None or pool[0] is None:
        pool = _POOL_MEM.get(a_sha) or load_anchor_pool(anchors_path)
        assert pool[3] == a_sha
        _POOL_MEM.clear(); _POOL_MEM[a_sha] = pool          # loaded at most once per process
    _, E = load_band_matrix(band_path)
    Q3, _ = build_ce3(E)
    m5 = anchor_mean5(Q3, pool[0], pool[1], pool[2])
    tmp = cp[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, mean5=m5, anchor_sha256=a_sha, band_sha256=b_sha, topk=TOPK, matcher="parts3_orientmax_div3", n=len(m5))
    try:
        os.replace(tmp, cp)
    except OSError:                   # Windows: a parallel arm wrote / holds the same (deterministic) cache — keep theirs
        if not os.path.exists(cp): raise
        os.remove(tmp)
    return m5, b_sha, a_sha, cp

def load_pi(pi_path, a_sha):
    """pi: training coat identity c -> global identity g (a bijection), from Tol_51 (provisional) or the reconciled key."""
    pj = json.load(open(pi_path, encoding="utf-8"))
    P = {int(c): int(g) for c, g in pj["pi"].items()}
    if sorted(P) != list(range(N)) or sorted(P.values()) != list(range(N)): raise SystemExit(f"pi is not a bijection: {P}")
    if pj.get("anchor_sha256") and pj["anchor_sha256"] != a_sha:
        raise SystemExit(f"pi was derived for anchor pool {pj['anchor_sha256'][:12]}, not {a_sha[:12]}")
    return P, pj

def ext_decision(x):
    """one scored run-window for the extension outputs: the production fields plus, with anchors on, the blend's parts
    (sims = the row the layer used; sims_a = anchor row; sims_t = temporal row or None; anchor_only = fallback taken)"""
    d = dict(sid=x["sid"], slot=x["slot"], t=x["t"].isoformat(), pred=x["pred"], margin=round(x["margin"], 4), argmax=x["argmax"],
             conf=x["conf"], status=x["status"], chaos=x["chaos"], sims=x["sims"], n_q=x["n_q"])
    for k_ in ("sims_a", "sims_t", "anchor_only"):
        if k_ in x: d[k_] = x[k_]
    return d

# ----------------------------------------------------------------------------- the layer
def run_layer(segs, crops, CE, T, density, links=None, seed_pools=None, span=None, hidden=None):
    W = timedelta(minutes=T); day0 = min(c["w"] for c in crops); end = max(c["w"] for c in crops) + timedelta(seconds=1)
    if span is not None: day0, end = span       # --noncalf: the window grid of the UNFILTERED crops (None = production)
    # --noncalf: the NON-CALF segments removed from `segs` (None = production). They open no run; they are used ONLY as
    # slot-continuity BARRIERS (prev_seg: a self_recovery right after a hidden track must not soft-inherit ACROSS it from the
    # calf before it) and for the recorded day span (seam provenance). Nothing else reads them.
    hidden = hidden or {}
    seg_order = sorted(segs.values(), key=lambda s: s["t_start"])
    # Only the FIRST day's seeds bootstrap. A later day's "seed_day"/"seed_night" segment must NOT reset an
    # identity — by then the accumulated pool is the authority, and the re-seed is just another HARD break to
    # be decided by appearance against it. (Bootstrapping every day is what made each day start cold.)
    _d0 = min((r.get("day") for r in segs.values() if r.get("day")), default=None)
    boot = {s["seg_id"]: s["slot"] for s in seg_order
            if s["open_reason"] in ("seed_day", "seed_night") and (_d0 is None or s.get("day") == _d0)}
    if BOOT == "none": boot = {}                # extension (--boot none): the first day's seeds are decided like any break
    runs = {}
    pool = {k: [] for k in range(N)}            # identity -> [(wall, crop_i, sid, run_no)]
    # PASS-2 WARM START (only for identities that were STARVED in pass 1 — see main()). The lag filter still
    # applies, so a warm-started exemplar is only visible once it is GAP_MIN old: causality is preserved and
    # this is not the self-labelling loop the periodic study warned about, which reused ALL of its own labels.
    warm = set()
    unscorable = Counter(); n_win_tot = 0        # windows in which an identity had < TOPK visible exemplars
    if seed_pools:
        ci_by = {c["i"]: c for c in crops}
        for k, idxs in seed_pools.items():
            for i in idxs:
                c = ci_by.get(i)
                if c is not None and c["gated"]: pool[k].append((c["w"], i, c["sid"], -1))
            pool[k].sort(); warm.add(k)
    crops_by_seg = defaultdict(list)
    for c in crops: crops_by_seg[c["sid"]].append(c)
    decisions, flags = [], []
    # SEAM LINKS: only rows whose tuple carries seam=True (4th element, set in main()) — empty dicts otherwise.
    seam_prev = {s_: lk_[0] for s_, lk_ in (links or {}).items()
                 if len(lk_) > 3 and lk_[3] and s_ in segs and lk_[0] in segs
                 and lk_[1] >= SEAM_MIN_IOU and SEAM_MIN_ABSENCE <= lk_[2] <= SEAM_MAX_ABSENCE}
    seam_next = {p_: s_ for s_, p_ in seam_prev.items()}
    # HAND-OVER state per seam successor S (v2): "pending" = S opened while its predecessor P was live and still
    # unlabelled — S waits (one window) for P's decision and is NOT scored, so the one-to-one Hungarian never sees
    # the same animal as two rows; "inherited" = S took P's label (and P's aggregate evidence) — from then on P is
    # not scored; "released" = no inheritance possible — S is decided like any other run, P is not scored.
    # Exactly one of P / S is scored in any window.
    seam_state = {}

    def cur_run(sid):
        return runs[sid][-1] if runs.get(sid) else None
    def open_run(sid, start, label=None, flag=None):
        r = dict(sid=sid, start=start, end=None, label=label, flags=[flag] if flag else [], win=[], settled=False, dissent=[],
                 no=len(runs.get(sid, [])), agg=np.zeros(N), agg_n=np.zeros(N), fed=set(), contested_open=False)
        runs.setdefault(sid, []).append(r); return r
    def feed(r, upto):
        if r["label"] is None: return
        # POOL BOOTSTRAP FLOOR. Normal hygiene only lets SETTLED, appearance-decided runs feed a pool. But if
        # an identity has no day-start seed, nothing is ever settled for it, its pool stays empty, and it
        # becomes permanently UNMATCHABLE — every window scores it None, everything that really is that animal
        # matches someone else, and the bijection pass then demotes the duplicates. Measured on 04-26, whose
        # day start produced only THREE seeds (slot 3 had none): identity 3 held 133 min of 4,400, its agg
        # column was None in every run, and 819 min of genuine identity-3 slot-time went UNASSIGNED.
        # So: while an identity is STARVED (< TOPK exemplars), any run holding it may seed it, flags or not.
        # Once it clears TOPK the normal hygiene resumes.
        starved = len(pool[r["label"]]) < TOPK and ANCHOR_S is None      # anchors on: never starved (extension)
        if not starved and (not r["settled"] or any(f in r["flags"] for f in ("hop_in", "pigeonhole", "soft_inherit"))): return
        if starved and "pool_bootstrap" not in r["flags"]:
            r["flags"].append("pool_bootstrap")
            flags.append(dict(kind="pool_bootstrap", sid=r["sid"], slot=segs[r["sid"]]["slot"],
                              t=r["start"].isoformat(), ident=r["label"],
                              why="identity had < TOPK exemplars; seeded from an unsettled/inherited run"))
        bad = [(x["t"], x["t"] + W) for x in r["win"] if x["status"] != "ok" or x["chaos"]]
        for c in crops_by_seg[r["sid"]]:
            if not c["gated"] or c["i"] in r["fed"] or c["w"] < r["start"] or c["w"] >= upto: continue
            if r["end"] is not None and c["w"] >= r["end"]: continue
            if any(a_ <= c["w"] < b_ for a_, b_ in bad): continue
            pool[r["label"]].append((c["w"], c["i"], r["sid"], r["no"])); r["fed"].add(c["i"])
    def unfeed(r, old_label, after=None):
        """Remove this run's crops from `old_label`'s pool (all, or those at/after `after`).
        FLOOR: never take an identity below TOPK exemplars — an empty pool cannot be scored against, so the
        identity would drop out of every future window and never be re-fed (the traced deadlock). The oldest
        entries are retained, which are the bootstrap-seeded ones."""
        if old_label is None: return 0
        keep, drop = [], []
        for e in pool[old_label]:
            (drop if (e[2] == r["sid"] and e[3] == r["no"] and (after is None or e[0] >= after)) else keep).append(e)
        if len(keep) < TOPK and drop:
            drop.sort()
            restore = drop[:max(0, TOPK - len(keep))]
            keep.extend(restore); drop = [e for e in drop if e not in restore]
        for e in drop: r["fed"].discard(e[1])
        pool[old_label] = sorted(keep); return len(drop)
    def relabel(r, new, why, t):
        old = r["label"]; unfeed(r, old); r["label"] = new; r["settled"] = False; r["dissent"] = []
        if old is not None and new != old:
            r["flags"].append("reassigned")
            flags.append(dict(kind="reassigned", sid=r["sid"], slot=segs[r["sid"]]["slot"], t=t.isoformat(), from_ident=old, to_ident=new, why=why))
    def seam_try(sid, when, t_now):
        """SEAM INHERIT (in-loop): the seam successor's FIRST run, while still unlabelled, takes its predecessor's
        CURRENT label — spatial continuity, never appearance. Tried when the segment opens and, while the successor
        is "pending" (predecessor still live and unlabelled), once more per window until the predecessor closes.
        The chain is ONE animal, so the successor also takes over the predecessor's AGGREGATE evidence (v2): without
        it, 1-3 windows of the successor's own appearance could contest a label the whole chain supports (04-18
        22:49, w2249/1_001). Guard: no OTHER run (the predecessor excepted — it legitimately holds the label until it
        closes) may hold the label live, or for > 60 s (the bijection tolerance) of the successor's span so far.
        Returns "inherited" / "wait" (predecessor unlabelled) / "blocked" / None (not applicable)."""
        p_sid = seam_prev.get(sid)
        if p_sid is None or not runs.get(sid) or not runs.get(p_sid): return None
        r = runs[sid][0]
        if len(runs[sid]) != 1 or r["label"] is not None: return None
        pr = runs[p_sid][-1]; pl = pr["label"]
        if pl is None: return "wait"
        a0 = r["start"]; b0 = max(t_now, a0)
        for sid_, rl_ in runs.items():
            if sid_ in (p_sid, sid): continue
            for rr in rl_:
                if rr["label"] != pl: continue
                if rr["end"] is None or (min(b0, rr["end"]) - max(a0, rr["start"])).total_seconds() > 60.0: return "blocked"
        lk = links[sid]
        r["label"] = pl; r["flags"].append("seam_inherit"); r["seam_src"] = [pr]
        r["agg"] = r["agg"] + pr["agg"]; r["agg_n"] = r["agg_n"] + pr["agg_n"]
        flags.append(dict(kind="seam_inherit", sid=sid, slot=segs[sid]["slot"], t=segs[sid]["t_start"].isoformat(), ident=pl,
                          prev=p_sid, link_iou=round(lk[1], 4), absence_s=lk[2], when=when, carried_agg_n=int(pr["agg_n"].max())))
        return "inherited"
    def seam_step(sid, t_now, when):
        """advance the hand-over state of seam successor `sid` (see seam_state)"""
        res = seam_try(sid, when, t_now)
        if res is None: return
        p_live = runs.get(seam_prev[sid]) and runs[seam_prev[sid]][-1]["end"] is None
        if res == "inherited": seam_state[sid] = "inherited"
        elif res == "wait" and p_live: seam_state[sid] = "pending"
        else:
            seam_state[sid] = "released"
            if res == "blocked":
                r = runs[sid][0]; pl = runs[seam_prev[sid]][-1]["label"]
                if "seam_blocked" not in r["flags"]: r["flags"].append("seam_blocked")
                flags.append(dict(kind="seam_blocked", sid=sid, slot=segs[sid]["slot"], t=t_now.isoformat(), ident=pl,
                                  prev=seam_prev[sid], when=when))

    t = day0
    while t < end:
        t1 = t + W; lag = t - timedelta(minutes=GAP_MIN)
        chaos = sum(density.get(((t - timedelta(minutes=m_)).hour, (t - timedelta(minutes=m_)).minute), 0) for m_ in range(CHAOS_MIN)) >= CHAOS_EVENTS
        win = defaultdict(list)
        for c in crops:
            if t <= c["w"] < t1 and c["gated"]: win[c["sid"]].append(c["i"])
        for s in seg_order:
            if s["t_start"] < t1 and s["t_end"] > t and cur_run(s["seg_id"]) is None:
                lab = boot.get(s["seg_id"])
                r = open_run(s["seg_id"], s["t_start"], lab, "bootstrap" if lab is not None else None)
                if lab is not None: r["settled"] = True
                # SPATIAL INHERIT (Tol_15_v2's SOFT path, extended to detector-driven reopens). A reopen whose
                # first mask still sits on the same slot's last mask is the SAME animal by continuity; deciding
                # it by appearance instead is what produced the 04-22 21:00 swap. Purely spatial — it can never
                # reproduce Tol_12. The aggregate machinery may still contest it later — and NB it CAN flip it: a joint
                # re-decision with gain >= TAU relabels an inherited run like any other (only seam runs are protected).
                elif links and s["open_reason"] in ("reopen", "displaced"):
                    lk = links.get(s["seg_id"])
                    if lk and lk[1] >= SOFT_MIN_LINK_IOU and lk[2] <= SOFT_MAX_ABSENCE and runs.get(lk[0]):
                        pl = runs[lk[0]][-1]["label"]
                        if pl is not None and not any(
                                rr["label"] == pl and rr["end"] is None and segs[sid_]["slot"] != s["slot"]
                                for sid_, rl_ in runs.items() for rr in rl_):
                            r["label"] = pl; r["flags"].append("spatial_inherit")
                            flags.append(dict(kind="spatial_inherit", sid=s["seg_id"], slot=s["slot"],
                                              t=s["t_start"].isoformat(), ident=pl, prev=lk[0],
                                              link_iou=round(lk[1], 4), absence_s=lk[2]))
                elif seam_prev and s["open_reason"] in ("seed_day", "seed_night") and s["seg_id"] in seam_prev:
                    seam_step(s["seg_id"], t, "open")
        if seam_state:
            # a PENDING successor is retried while its predecessor is live, and one last time right after it closes
            # (the predecessor may be decided in the seam window itself); then it is released — never retried for
            # the rest of the timeline (a late retry would apply the label back over a span another run held).
            for sid_s in [x for x, st_ in seam_state.items() if st_ == "pending"]:
                seam_step(sid_s, t, f"deferred@{t.strftime('%H:%M:%S')}")
        # recency-limited, capped, lagged pools
        vis = {}
        for k in range(N):
            if TEMPORAL == "even": vis[k] = even_select(pool[k], lag, EVEN_CAP or POOL_MAX_N); continue   # extension
            lo_ = lag - timedelta(seconds=min(POOL_CAP_SEC, POOL_H * 3600.0))
            recent = [i for (w_, i, _, _) in pool[k] if lo_ <= w_ <= lag]
            if len(recent) < TOPK: recent = [i for (w_, i, _, _) in sorted(pool[k]) if w_ <= lag]
            vis[k] = recent[-POOL_MAX_N:]
        live = [sid for sid in win if len(win[sid]) >= MIN_Q]
        if seam_state:
            # HAND-OVER (v2): never score one animal as two rows of the one-to-one Hungarian. A pending successor
            # waits; once the successor has inherited (or been released) its predecessor is frozen — it is not
            # scored, so it can neither be relabelled behind the seam nor compete with its own continuation.
            live = [sid for sid in live if seam_state.get(sid) != "pending"
                    and not (sid in seam_next and seam_state.get(seam_next[sid]) in ("inherited", "released"))]
        ok_ids = [k for k in range(N) if len(vis[k]) >= TOPK]
        n_win_tot += 1
        for k in range(N):
            if len(vis[k]) < TOPK: unscorable[k] += 1
        if ANCHOR_S is not None: ok_ids = list(range(N))      # extension: an anchor set is never empty
        if live and len(ok_ids) >= 2:
            M = np.full((len(live), N), np.nan)
            adiag = {}
            if _EXT:
                for a, sid in enumerate(live):                  # extension: anchor blend and/or even pool
                    if ANCHOR_S is None:
                        for k in ok_ids: M[a, k] = pair_sim_ext(CE, win[sid], vis[k])
                        continue
                    sa = ANCHOR_S[win[sid]].mean(axis=0)
                    t_ok = ANCHOR_W < 1.0 and all(len(vis[k]) >= TOPK for k in range(N))
                    st = np.array([pair_sim_ext(CE, win[sid], vis[k]) for k in range(N)]) if t_ok else None
                    M[a] = ANCHOR_W * sa + (1.0 - ANCHOR_W) * st if t_ok else sa
                    adiag[sid] = (sa, st, not t_ok)
            else:
                for a, sid in enumerate(live):
                    for k in ok_ids: M[a, k] = pair_sim(CE, win[sid], vis[k])
            Mh = np.where(np.isnan(M), -9.0, M)
            ri, cj = linear_sum_assignment(-Mh)
            dec = {live[a]: int(k) for a, k in zip(ri, cj)}
            held = {}
            for sid in live:
                r_ = cur_run(sid)
                if r_["label"] is not None: held.setdefault(r_["label"], sid)
            recs_now = {}
            for a, sid in enumerate(live):
                r = cur_run(sid)
                for k in ok_ids: r["agg"][k] += M[a, k]; r["agg_n"][k] += 1
                if sid not in dec: continue
                row = Mh[a]; pred = dec[sid]
                if row[pred] < -5: continue
                others = [row[k] for k in range(N) if k != pred and row[k] > -5]
                margin = float(row[pred] - max(others)) if others else 0.0
                argmax = int(np.nanargmax(M[a])); conf = margin >= TAU
                rec = dict(sid=sid, slot=segs[sid]["slot"], t=t, pred=pred, margin=margin, argmax=argmax, conf=conf, chaos=chaos, status="ok",
                           sims={k: (None if np.isnan(M[a, k]) else round(float(M[a, k]), 4)) for k in range(N)}, n_q=len(win[sid]))
                if sid in adiag:                                # extension diagnostics only (never read by the layer)
                    _sa, _st, _ao = adiag[sid]
                    rec.update(sims_a=[round(float(x), 5) for x in _sa], anchor_only=bool(_ao),
                               sims_t=None if _st is None else [round(float(x), 5) for x in _st])
                decisions.append(rec); r["win"].append(rec); recs_now[sid] = rec
            # ---- window-level hop detection (free rival only) ------------------------------------
            for sid, rec in recs_now.items():
                r = cur_run(sid)
                if not rec["conf"] or r["label"] is None: continue
                if rec["pred"] == r["label"]: r["dissent"] = []; continue
                j = rec["pred"]; holder = held.get(j)
                if holder is not None and holder != sid:
                    rec["status"] = "dissent_held"; r["dissent"] = []; continue
                if chaos: rec["status"] = "dissent_chaos"; r["dissent"] = []; continue
                rec["status"] = "dissent"; r["dissent"].append((t, j))
                if len(r["dissent"]) == 1:
                    flags.append(dict(kind="dissent", sid=sid, slot=segs[sid]["slot"], t=t.isoformat(), incumbent=r["label"], rival=j, margin=round(rec["margin"], 4)))
                if len(r["dissent"]) >= K_HOP and len({p_ for _, p_ in r["dissent"]}) == 1:
                    hop_t = r["dissent"][0][0]
                    n_purged = unfeed(r, r["label"], after=hop_t)
                    r["end"] = hop_t; r["flags"].append("hop_out")
                    tail = [x for x in r["win"] if x["t"] >= hop_t]; r["win"] = [x for x in r["win"] if x["t"] < hop_t]
                    r2 = open_run(sid, hop_t, j, "hop_in"); r2["win"] = tail
                    for x in tail:
                        for k in range(N):
                            if x["sims"][k] is not None: r2["agg"][k] += x["sims"][k]; r2["agg_n"][k] += 1; r["agg"][k] -= x["sims"][k]; r["agg_n"][k] -= 1
                    flags.append(dict(kind="hop", sid=sid, slot=segs[sid]["slot"], t=hop_t.isoformat(), from_ident=r["label"], to_ident=j, purged=n_purged))
            # ---- aggregate joint decision over the live runs, with hysteresis -------------------
            lr = [cur_run(sid) for sid in live if cur_run(sid)["agg_n"].max() > 0]
            if lr:
                A = np.full((len(lr), N), -9.0)
                for a, r in enumerate(lr):
                    for k in ok_ids:
                        if r["agg_n"][k] > 0: A[a, k] = r["agg"][k] / r["agg_n"][k]
                ri, cj = linear_sum_assignment(-A)
                new = {lr[a]["sid"]: int(k) for a, k in zip(ri, cj) if A[a, k] > -5}
                cur = {r["sid"]: r["label"] for r in lr}
                changed = [r for r in lr if new.get(r["sid"]) is not None and new[r["sid"]] != cur[r["sid"]]]
                if changed:
                    # improvement of the joint similarity: labelled runs pay their old row, unlabelled runs
                    # must clear the margin to their best alternative
                    delta = 0.0
                    for r in changed:
                        a = lr.index(r); row = A[a]; nk = new[r["sid"]]
                        if cur[r["sid"]] is not None and row[cur[r["sid"]]] > -5: delta += row[nk] - row[cur[r["sid"]]]
                        else: delta += row[nk] - max([row[k] for k in range(N) if k != nk and row[k] > -5] + [-9.0])
                    # GUARD (added after the T=1 deadlock trace: a +0.0556 joint gain permuted 3 of 4 runs
                    # whose OWN margins were -0.0027 / -0.0116 / -0.0507, i.e. no run individually preferred
                    # its new label; every run then unsettled, stopped feeding, and the pools never recovered).
                    # A joint re-decision may only fire if EVERY changed run's own row prefers its new label.
                    own_ok = True
                    for r in changed:
                        a = lr.index(r); row = A[a]; nk = new[r["sid"]]
                        alt = max([row[k] for k in range(N) if k != nk and row[k] > -5] + [-9.0])
                        if row[nk] - alt <= 0: own_ok = False; break
                    # SEAM PROTECTION (v2): a seam-inherited label is spatial continuity across a clip boundary; the
                    # predecessor behind the seam is closed and cannot be corrected with it, so a flip here would
                    # always manufacture a seam conflict. Appearance may CONTEST it (flag + hold), never flip it.
                    seam_hold = [r for r in changed if cur[r["sid"]] is not None and "seam_inherit" in r["flags"]] if seam_prev else []
                    if delta >= TAU * max(1, len(changed)) and own_ok and not chaos and not seam_hold:
                        for r in changed:
                            _was = r["label"]
                            relabel(r, new[r["sid"]], f"aggregate joint re-decision (+{delta:.4f})", t)
                            if seam_prev and _was is None: r["flags"].append("joint_decision"); r["decided_at"] = t
                    else:
                        for r in changed:
                            if cur[r["sid"]] is None and delta >= TAU: continue      # (cannot happen jointly; kept for clarity)
                            if not r["contested_open"] and cur[r["sid"]] is not None:
                                _ev_ = dict(kind="contested", sid=r["sid"], slot=segs[r["sid"]]["slot"], t=t.isoformat(), label=cur[r["sid"]],
                                            aggregate_prefers=new[r["sid"]], joint_gain=round(delta, 4))
                                if seam_hold: _ev_["seam_protected"] = [x["sid"] for x in seam_hold]
                                flags.append(_ev_)
                                r["contested_open"] = True
                            if cur[r["sid"]] is not None and "contested" not in r["flags"]: r["flags"].append("contested")
                        for r in changed: r["contested_at"] = t; r["settled"] = False
                for r in lr:
                    if r["sid"] not in [x["sid"] for x in changed]: r["contested_open"] = False
                # unlabelled runs whose aggregate row is clear take their identity if it is free
                for a, r in enumerate(lr):
                    if r["label"] is None and r["sid"] in new:
                        nk = new[r["sid"]]; row = A[a]
                        alt = max([row[k] for k in range(N) if k != nk and row[k] > -5] + [-9.0])
                        holders = [x for x in lr if x is not r and x["label"] == nk]
                        if row[nk] - alt >= TAU and not holders:
                            r["label"] = nk; r["decided_at"] = t; r["flags"].append("appearance")
            # ---- settle check ------------------------------------------------------------------
            for sid in live:
                r = cur_run(sid)
                if r["label"] is None or r["settled"]: continue
                agree = [x for x in r["win"] if x["conf"] and x["pred"] == r["label"] and x["status"] == "ok"]
                # a contested run may still settle, but only on agreement recorded AFTER the last contest
                since = r.get("contested_at")
                if since is not None: agree = [x for x in agree if x["t"] > since]
                if len(agree) >= SETTLE_WIN and (agree[-1]["t"] - agree[0]["t"]) >= timedelta(minutes=SETTLE_MIN - T):
                    r["settled"] = True
        for sid, rl in runs.items():
            for r in rl:
                if r["end"] is None or r["end"] > t: feed(r, t1)
        for s in seg_order:
            r = cur_run(s["seg_id"])
            if r is not None and r["end"] is None and s["t_end"] <= t1: r["end"] = s["t_end"]
        t = t1

    def enforce_bijection(tag):
            # ---- BIJECTION ENFORCEMENT ----------------------------------------------------------------
        # The pen holds four animals in four slots, so at any instant the concurrent runs must hold four
        # DISTINCT identities. The window Hungarian enforces this only among runs it actually SCORED; a run
        # that is live but has < MIN_Q crops in a window is invisible to it and keeps a stale label. Measured
        # before this pass: 1,921 duplicate-colour minutes across the week (04-26 alone 823, where identity 2
        # held 31.5 h and identity 3 only 2.2 h). A duplicate is a claim we KNOW is false, so the weaker
        # claimant must abstain rather than assert.
        def _tier0(r):
            return 3 if "bootstrap" in r["flags"] else (2 if "appearance" in r["flags"] else
                   (1 if ("modal" in r["flags"] or "spatial_inherit" in r["flags"] or "soft_inherit" in r["flags"]
                          or "seam_inherit" in r["flags"]) else 0))
        def _ev(r):
            """evidence rank for keeping a contested identity: confident agreeing windows, then how it was set"""
            agree = sum(1 for x in r["win"] if x["conf"] and x["pred"] == r["label"] and x["status"] == "ok")
            prio = _tier0(r)
            if r.get("seam_src"):
                # a seam run is as strong as the ORIGIN of its chain (v2): a chain seeded by a bootstrap / appearance
                # decision must not lose a bijection conflict to a single appearance decision just because it
                # crossed a window seam. Only runs of the same label along the recorded seam sources count.
                seen, todo = set(), [r]
                while todo:
                    x = todo.pop()
                    if id(x) in seen: continue
                    seen.add(id(x))
                    for y in x.get("seam_src", ()):
                        if y["label"] == r["label"]: prio = max(prio, _tier0(y)); todo.append(y)
            return (prio, agree, len(r["win"]))
        allruns = [(sid, r) for sid, rl in runs.items() for r in rl if r["label"] is not None]
        n_demoted = 0
        for k in range(N):
            holders = [(sid, r) for sid, r in allruns if r["label"] == k]
            for i in range(len(holders)):
                for j in range(i + 1, len(holders)):
                    (sa, ra), (sb, rb) = holders[i], holders[j]
                    if segs[sa]["slot"] == segs[sb]["slot"]: continue        # same slot cannot be concurrent
                    a0 = max(ra["start"], rb["start"]); b0 = min(ra["end"] or end, rb["end"] or end)
                    ov = (b0 - a0).total_seconds() / 60.0
                    if ov <= 1.0: continue
                    if ra["label"] is None or rb["label"] is None: continue
                    lose = rb if _ev(ra) >= _ev(rb) else ra
                    keep = ra if lose is rb else rb
                    lose_sid = sb if lose is rb else sa
                    flags.append(dict(kind="bijection_conflict", sid=lose_sid, slot=segs[lose_sid]["slot"],
                                      t=a0.isoformat(), ident=k, overlap_min=round(ov, 1),
                                      lost_to=(sa if keep is ra else sb), evidence_lost=_ev(lose), evidence_kept=_ev(keep)))
                    lose["label"] = None; lose["flags"].append("bijection_conflict"); n_demoted += 1
        if n_demoted: print(f"  bijection enforcement ({tag}): {n_demoted} runs demoted to UNASSIGNED (duplicate identity)")
        return n_demoted



    # ---- post pass: pigeonhole-with-flags, soft inheritance, provisional marks -------------------
    # Enforce the bijection BEFORE the post-pass, so the pigeonhole / modal / inherit rules can refill the
    # slot-time a demotion frees. Demoting alone lost hours: 04-27's run 1_003 (449 min, 71 confident windows)
    # preferred identity 2, lost it to a better-evidenced run, and was left UNASSIGNED even though another
    # identity was free for its whole span. Enforce again afterwards so the refill cannot re-create a duplicate.
    enforce_bijection("pre")

    def label_at(slot, w):
        for sid, rl in runs.items():
            if segs[sid]["slot"] != slot: continue
            for r in rl:
                if r["label"] is not None and r["start"] <= w < (r["end"] or end): return r["label"], r
        return None, None
    prev_seg = {}
    _pseg = sorted(list(segs.values()) + list(hidden.values()), key=lambda s: s["t_start"]) if hidden else seg_order
    for slot in range(N):
        ss = [s for s in _pseg if s["slot"] == slot]
        for a, b in zip(ss, ss[1:]): prev_seg[b["seg_id"]] = a
    def seam_fill(s, r):
        """SEAM FILL (post-pass): an UNASSIGNED run at a seam takes the label of the run on the other side of the
        seam — forward (seam successor's first run <- predecessor's last run) or backward (predecessor's last run
        <- successor's first run). This is what lets a label that was decided only AFTER the successor opened
        (post-pass modal / pigeonhole / another seam fill) still flow across the seam, and it chains through
        consecutive windows via the enclosing fixpoint loop. It only ever FILLS a '?', never overwrites, and
        abstains when the two sides disagree or when the identity is held by any other labelled run for > 1 min
        of this run's span (the bijection tolerance, so the post bijection pass has nothing to demote).
        A successful fill is propagated along the seam chain at once (the enclosing loop is capped at 4 passes,
        and a backward chain would otherwise advance only one window per pass)."""
        def _fill(sid, r):
            rl = runs[sid]; cand = {}
            p_sid = seam_prev.get(sid)
            if p_sid is not None and rl[0] is r and runs.get(p_sid) and runs[p_sid][-1]["label"] is not None:
                cand[p_sid] = runs[p_sid][-1]["label"]
            n_sid = seam_next.get(sid)
            if n_sid is not None and rl[-1] is r and runs.get(n_sid) and runs[n_sid][0]["label"] is not None:
                cand[n_sid] = runs[n_sid][0]["label"]
            if not cand or len(set(cand.values())) != 1: return False
            k = next(iter(cand.values())); a0, b0 = r["start"], (r["end"] or segs[sid]["t_end"])
            for sid2, rl2 in runs.items():
                if sid2 == sid or sid2 in cand: continue
                for r2 in rl2:
                    if r2["label"] != k: continue
                    if (min(b0, r2["end"] or end) - max(a0, r2["start"])).total_seconds() > 60.0: return False
            r["label"] = k; r["flags"].append("seam_inherit")
            r["seam_src"] = [runs[c_][-1] if c_ == p_sid else runs[c_][0] for c_ in cand]   # provenance (v2)
            flags.append(dict(kind="seam_inherit", sid=sid, slot=segs[sid]["slot"], t=r["start"].isoformat(), ident=k,
                              prev=p_sid, next=n_sid, src=sorted(cand), when="post"))
            return True
        if not _fill(s["seg_id"], r): return False
        todo = [s["seg_id"]]
        while todo:
            x = todo.pop()
            for y, first in ((seam_next.get(x), True), (seam_prev.get(x), False)):
                if y is None or not runs.get(y): continue
                ry = runs[y][0] if first else runs[y][-1]
                if ry["label"] is None and _fill(y, ry): todo.append(y)
        return True
    changed = True; passes = 0
    while changed and passes < 4:
        changed = False; passes += 1
        for s in seg_order:
            for r in runs.get(s["seg_id"], []):
                if r["label"] is not None: continue
                if (seam_prev or seam_next) and seam_fill(s, r): changed = True; continue
                # STICKY (v2): a seam run that LOST its chain label in a bijection conflict stays '?' — appearance
                # (modal / pigeonhole / soft inherit) may not give it a different colour against spatial continuity.
                if (seam_prev or seam_next) and "seam_inherit" in r["flags"] and "bijection_conflict" in r["flags"]: continue
                r_end = r["end"] or s["t_end"]
                # MODAL VOTE over the run's CONFIDENT windows, before pigeonhole. The aggregate MEAN can stay
                # muddy while the individual windows are decisive (04-23 dense: run 0_006 had 26/59 confident
                # windows yet identities 0 and 2 tied at 0.781 vs 0.783, so it abstained for 3 h). The matcher
                # study settled this: per-frame errors are random, so the MODE over a window set is right even
                # when the mean is not ("modal permutation over one hour is 100 % correct in EVERY hour").
                conf_w = [x for x in r["win"] if x["conf"] and x["status"] == "ok"]
                if len(conf_w) >= MODAL_MIN_WIN:
                    votes = Counter(x["pred"] for x in conf_w)
                    cand, cnt = votes.most_common(1)[0]
                    if cnt >= MODAL_FRAC * len(conf_w):
                        busy = False; tt2 = r["start"]
                        while tt2 < r_end and not busy:
                            for sl2 in range(N):
                                if sl2 == s["slot"]: continue
                                if label_at(sl2, tt2)[0] == cand: busy = True; break
                            tt2 += timedelta(minutes=1)
                        if not busy:
                            r["label"] = cand; r["flags"].append("modal"); changed = True
                            flags.append(dict(kind="modal", sid=s["seg_id"], slot=s["slot"], t=r["start"].isoformat(),
                                              ident=cand, vote=f"{cnt}/{len(conf_w)}"))
                            continue
                free_votes = Counter(); n_probe = 0; tt = r["start"]
                while tt < r_end:
                    taken = set()
                    for sl in range(N):
                        if sl == s["slot"]: continue
                        lab, _ = label_at(sl, tt)
                        if lab is not None: taken.add(lab)
                    if len(taken) == 3: free_votes[[k for k in range(N) if k not in taken][0]] += 1
                    n_probe += 1; tt += timedelta(minutes=1)
                if free_votes and n_probe >= 1 and free_votes.most_common(1)[0][1] >= 0.8 * n_probe:
                    cand = free_votes.most_common(1)[0][0]
                    scored = list(r["win"])
                    if scored:
                        agree = sum(1 for x in scored if x["argmax"] == cand)
                        if agree >= 0.5 * len(scored):
                            r["label"] = cand; r["flags"].append("pigeonhole"); changed = True
                            flags.append(dict(kind="pigeonhole", sid=s["seg_id"], slot=s["slot"], t=r["start"].isoformat(), ident=cand, argmax_agree=f"{agree}/{len(scored)}"))
                        elif "pigeonhole_conflict" not in r["flags"]:
                            # The pen holds exactly four animals in four slots: if the other three are held,
                            # the fourth is DETERMINED. That constraint is a physical fact; appearance is
                            # probabilistic. When they disagree, take the constraint and FLAG it loudly
                            # (PIGEONHOLE_MODE="assign") rather than abstaining — measured on 04-23 dense,
                            # every one of these had the pigeonhole RIGHT and the appearance argmax WRONG.
                            r["flags"].append("pigeonhole_conflict")
                            flags.append(dict(kind="pigeonhole_conflict", sid=s["seg_id"], slot=s["slot"], t=r["start"].isoformat(),
                                              ident=cand, argmax_agree=f"{agree}/{len(scored)}", assigned=(PIGEONHOLE_MODE == "assign")))
                            if PIGEONHOLE_MODE == "assign":
                                r["label"] = cand; changed = True
                    continue
                if r["no"] == 0 and s["open_reason"] == "self_recovery" and s["seg_id"] in prev_seg:
                    p = prev_seg[s["seg_id"]]
                    if (s["t_start"] - p["t_end"]).total_seconds() <= SOFT_MAX_ABSENCE and runs.get(p["seg_id"]):
                        pl = runs[p["seg_id"]][-1]["label"]
                        if pl is not None:
                            others = {label_at(sl, s["t_start"])[0] for sl in range(N) if sl != s["slot"]}
                            if pl not in others:
                                r["label"] = pl; r["flags"].append("soft_inherit"); changed = True
                                flags.append(dict(kind="soft_inherit", sid=s["seg_id"], slot=s["slot"], t=r["start"].isoformat(), ident=pl))
    enforce_bijection("post")
    for sid, rl in runs.items():
        for r in rl:
            if r["label"] is not None and not r["settled"] and "bootstrap" not in r["flags"] and not any(f in r["flags"] for f in ("pigeonhole", "soft_inherit", "hop_in", "contested", "reassigned")):
                r["flags"].append("provisional")
    # ---- SEAM PROVENANCE (v2, seam mode only; flags only — runs after every decision, changes no label) ----------
    def seam_provenance():
        # (1) a seam-inherited run carries its source's label-origin flags, along the whole chain (monotone fixpoint)
        ch_, it_ = True, 0
        while ch_ and it_ < 500:
            ch_, it_ = False, it_ + 1
            for sid_, rl_ in runs.items():
                for r_ in rl_:
                    if r_["label"] is None or "seam_inherit" not in r_["flags"]: continue
                    for y_ in r_.get("seam_src", ()):
                        if y_["label"] != r_["label"]: continue
                        for f_ in y_["flags"]:
                            if f_ in SEAM_CARRY_FLAGS and f_ not in r_["flags"]: r_["flags"].append(f_); ch_ = True
        # (2) the run's OWN confident windows mostly disagree with the label it inherited -> "seam_dissent"
        for sid_, rl_ in runs.items():
            for r_ in rl_:
                if r_["label"] is None or "seam_inherit" not in r_["flags"]: continue
                cw_ = [x for x in r_["win"] if x["conf"] and x["status"] == "ok"]
                n_ag = sum(1 for x in cw_ if x["pred"] == r_["label"])
                if len(cw_) >= MODAL_MIN_WIN and n_ag < 0.5 * len(cw_):
                    r_["flags"].append("seam_dissent")
                    flags.append(dict(kind="seam_dissent", sid=sid_, slot=segs[sid_]["slot"], t=r_["start"].isoformat(), ident=r_["label"],
                                      agree=f"{n_ag}/{len(cw_)}", votes={int(k_): v_ for k_, v_ in Counter(x["pred"] for x in cw_).items()}))
        # (3) seeds NO seam link reaches (a later dir's day/night seed): the label there is an appearance decision
        #     against the accumulated pool — after a recording gap ("gap_decision") or at a seam whose masks could
        #     not be linked ("seam_unlinked"). Every label downstream of it rests on it, so it is always review.
        d_lo, d_hi = {}, {}
        for s_ in (list(segs.values()) + list(hidden.values())) if hidden else segs.values():
            d_ = s_.get("day")
            d_lo[d_] = min(d_lo.get(d_, s_["t_start"]), s_["t_start"]); d_hi[d_] = max(d_hi.get(d_, s_["t_end"]), s_["t_end"])
        d_order = sorted(d_lo, key=lambda d_: d_lo[d_])
        for s_ in seg_order:
            if s_["open_reason"] not in ("seed_day", "seed_night") or s_["seg_id"] in seam_prev or not runs.get(s_["seg_id"]): continue
            i_ = d_order.index(s_.get("day"))
            if i_ == 0: continue                                   # the first dir's seeds are the bootstrap
            gap_s = (s_["t_start"] - d_hi[d_order[i_ - 1]]).total_seconds()
            kind_ = "gap_decision" if gap_s > 60.0 else "seam_unlinked"
            r_ = runs[s_["seg_id"]][0]
            agg_ = [round(float(r_["agg"][k] / r_["agg_n"][k]), 4) if r_["agg_n"][k] > 0 else None for k in range(N)]
            vals_ = sorted((v_ for v_ in agg_ if v_ is not None), reverse=True)
            if r_["label"] is not None and agg_[r_["label"]] is not None:
                oth_ = [v_ for k, v_ in enumerate(agg_) if k != r_["label"] and v_ is not None]
                marg_ = round(agg_[r_["label"]] - max(oth_), 4) if oth_ else None
            else:
                marg_ = round(vals_[0] - vals_[1], 4) if len(vals_) > 1 else None
            cw_ = [x for x in r_["win"] if x["conf"] and x["status"] == "ok"]
            if kind_ not in r_["flags"]: r_["flags"].append(kind_)
            flags.append(dict(kind=kind_, sid=s_["seg_id"], slot=s_["slot"], t=r_["start"].isoformat(), ident=r_["label"],
                              gap_s=round(gap_s, 1), agg=agg_, agg_margin=marg_, label_flags=[f_ for f_ in r_["flags"] if f_ != kind_],
                              decided_at=(r_["decided_at"].isoformat() if r_.get("decided_at") else None),
                              conf_votes={int(k_): v_ for k_, v_ in Counter(x["pred"] for x in cw_).items()}, n_windows=len(r_["win"])))
    if seam_prev or seam_next: seam_provenance()
    # An identity is STARVED if it was UNSCORABLE (fewer than TOPK visible exemplars) for a large share of
    # the day's windows — not merely if its pool is empty at the END. 04-26's identity 3 finally filled at
    # 17:29 but was unmatchable for the preceding 11.75 h, which is what cost it 819 min of slot-time.
    starved = {k for k in range(N) if n_win_tot and unscorable[k] > 0.25 * n_win_tot}
    if ANCHOR_S is not None:                     # extension: every identity is always scorable against its anchors
        if starved: print("  (anchors on) temporal-pool unscorable-window share: " + ", ".join(f"{k}:{100*unscorable[k]/max(n_win_tot,1):.0f}%" for k in range(N)))
        starved = set()
    if starved: print(f"  unscorable-window share: " + ", ".join(f"{k}:{100*unscorable[k]/max(n_win_tot,1):.0f}%" for k in range(N)))
    if warm:
        for sid, rl in runs.items():
            for r in rl:
                if r["label"] in warm and "warm_pool" not in r["flags"]: r["flags"].append("warm_pool")
    return runs, decisions, flags, day0, end, starved

# ----------------------------------------------------------------------------- scoring
def score(segs, crops, runs, gt_asg, end, T):
    W = timedelta(minutes=T)
    def our_label(c):
        for r in runs.get(c["sid"], []):
            if r["start"] <= c["w"] < (r["end"] or end): return r["label"], r
        return None, None
    def contested(c, r):
        return any(x["status"] != "ok" and x["t"] <= c["w"] < x["t"] + W for x in r["win"])
    iv = gt_asg.get("__intervals__") if isinstance(gt_asg, dict) else None
    rows = []
    for c in crops:
        ours, r = our_label(c)
        if iv is not None:
            # interval GT: our local identity renders as video label G[ours]; the truth is the calf
            # actually carrying that label at this instant. Correct <=> rendered label == true calf.
            shown = None if ours is None else iv["G"].get(int(ours))
            g = None if shown is None else iv["truth"](shown, c["w"])
            ours_cmp = shown
        else:
            g = gt_asg.get(c["sid"]); ours_cmp = ours
        fl = bool(r and (any(f in r["flags"] for f in REVIEW_FLAGS) or contested(c, r)))
        rows.append(dict(sid=c["sid"], slot=c["slot"], w=c["w"], phase=c["phase"], gt=None if g is None else int(g),
                         ours=ours_cmp, flagged=fl, gated=c["gated"]))
    both = [x for x in rows if x["gt"] is not None and x["ours"] is not None]
    acc = lambda xs: (sum(x["gt"] == x["ours"] for x in xs) / len(xs)) if xs else float("nan")
    out = {"n_crops": len(rows), "n_gt": sum(x["gt"] is not None for x in rows), "n_ours": sum(x["ours"] is not None for x in rows),
           "n_both": len(both), "acc_both": acc(both),
           "acc_by_phase": {ph: acc([x for x in both if x["phase"] == ph]) for ph in ("day", "dusk", "night")},
           "n_by_phase": {ph: sum(1 for x in both if x["phase"] == ph) for ph in ("day", "dusk", "night")},
           "silent_errors": sum(1 for x in both if x["gt"] != x["ours"] and not x["flagged"]),
           "flagged_errors": sum(1 for x in both if x["gt"] != x["ours"] and x["flagged"]),
           "flagged_crops": sum(1 for x in rows if x["flagged"]),
           "acc_unflagged": acc([x for x in both if not x["flagged"]]), "n_unflagged": sum(1 for x in both if not x["flagged"]),
           "gained_vs_gt": sum(1 for x in rows if x["gt"] is None and x["ours"] is not None),
           "lost_vs_gt": sum(1 for x in rows if x["gt"] is not None and x["ours"] is None),
           "gt_coverage": sum(x["gt"] is not None for x in rows) / max(len(rows), 1),
           "our_coverage": sum(x["ours"] is not None for x in rows) / max(len(rows), 1)}
    def intervals(pred):
        out_, cur = [], None
        for x in rows:
            if pred(x):
                if cur and cur["sid"] == x["sid"] and cur["ours"] == x["ours"] and (x["w"] - cur["to"]) <= timedelta(minutes=2): cur["to"] = x["w"]; cur["n"] += 1; cur["flagged"] = cur["flagged"] and x["flagged"]
                else:
                    cur = dict(sid=x["sid"], slot=x["slot"], frm=x["w"], to=x["w"], gt=x["gt"], ours=x["ours"], n=1, flagged=x["flagged"]); out_.append(cur)
        return [dict(sid=o["sid"], slot=o["slot"], **{"from": o["frm"].strftime("%H:%M"), "to": o["to"].strftime("%H:%M")}, gt=o["gt"], ours=o["ours"], n_crops=o["n"], all_flagged=o["flagged"]) for o in out_]
    out["disagreements"] = intervals(lambda x: x["gt"] is not None and x["ours"] is not None and x["gt"] != x["ours"])
    out["gained"] = intervals(lambda x: x["gt"] is None and x["ours"] is not None)
    out["lost"] = intervals(lambda x: x["gt"] is not None and x["ours"] is None)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--day", default=None)
    ap.add_argument("--days", default=None,
                    help="comma-separated days -> ONE merged timeline whose exemplar pool ACCUMULATES across "
                         "days (global identity space). Without this each day starts from an empty pool.")
    ap.add_argument("--week_map", default=None, help="week_ident_map.json (day -> {local identity: global})")
    ap.add_argument("--pool_h", type=float, default=0.0,
                    help="pool recency window in hours (0 = 3 h single-day default; week mode uses 36 h so "
                         "exemplars survive the ~6 h overnight recording gap)")
    ap.add_argument("--T", type=float, default=3.0); ap.add_argument("--gt", default="segment_assignments.json")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--links", default=None,
                    help="Tol_33 reopen_linkability.json: lets a REOPEN inherit its predecessor's identity "
                         "SPATIALLY (never by appearance) when its first mask still sits on the same slot's "
                         "last mask. Measured: fires on 29-65 %% of reopens by day, 0 %% at night.")
    ap.add_argument("--bands", default="crop_bands3.npz", help="band cache filename (e.g. crop_bands3_dense.npz)")
    ap.add_argument("--topk", type=int, default=0,
                    help="exemplars averaged per identity (0 = TOPK default 5). Should scale with crop density: "
                         "a fixed 5 drawn from a 4.7x larger pool compresses every margin toward the ceiling.")
    ap.add_argument("--pigeonhole", default="assign", choices=["assign", "abstain"],
                    help="when the pen constraint and appearance disagree: assign+flag (default) or abstain")
    ap.add_argument("--variant", default="parts3", choices=["parts3", "whole"],
                    help="crops mode only: parts3 = frozen 3-band matcher; whole = single-pass whole crop")
    ap.add_argument("--features", default="crops", choices=["crops", "obs"],
                    help="crops = saved canonical crops every 60 s (frozen 3-part matcher); "
                         "obs = the tracker's own 10-s masked-crop embeddings (whole-crop, 6x denser)")
    # ---- extension flags (anchor-pool re-ID, 2026-09-26); none passed => production, byte-identical ----
    ap.add_argument("--anchors", default=None, help="frozen anchor pool npz (Tol_50; sha256 checked against its manifest)")
    ap.add_argument("--anchor_pi", default=None, help="pi json (Tol_51): {\"pi\": {coat: global}, \"status\": ..., \"anchor_sha256\": ...}; "
                                                        "results are flagged provisional unless status == \"final\" (and no provisional=true)")
    ap.add_argument("--anchor_w", type=float, default=None, help="blend weight w in (0,1]: s = w*s_anchor + (1-w)*s_temporal; 1 = anchors only")
    ap.add_argument("--anchor_cache", default=None, help="per-band anchor score cache dir (default <anchors dir>/score_cache)")
    ap.add_argument("--temporal", default="recency", choices=["recency", "even"],
                    help="temporal pool: recency (production) or even (hour-stratified over the whole lagged history)")
    ap.add_argument("--even_cap", type=int, default=0, help="even pool: exemplars per identity (0 = POOL_MAX_N)")
    ap.add_argument("--boot", default="seeds", choices=["seeds", "none"],
                    help="seeds (production) = first day's seeds bootstrap; none = no seed bootstrap (requires --anchors)")
    ap.add_argument("--noncalf", default=None,
                    help="non-calf json: Tol_53 threshold json (noncalf_threshold_v1.json, anchor mode; requires --anchors) or "
                         "Tol_54 colour json (noncalf_colour_v1.json, colour mode; reads <root>/<day>/crop_colour_v1.npz): "
                         "NON-CALF segments get no identity, feed no pool, and are written under 'hidden'")
    a = ap.parse_args()
    DAYS = [x.strip() for x in a.days.split(",")] if a.days else None
    if DAYS and not a.day: a.day = DAYS[0]
    d = os.path.join(a.root, a.day); out_dir = a.out_dir or d
    global VARIANT, PIGEONHOLE_MODE, TOPK, POOL_H, POOL_CAP_SEC
    global ANCHOR_S, ANCHOR_W, TEMPORAL, EVEN_CAP, BOOT, _EXT, _CE3, NONCALF, NONCALF_MODE, COLOUR_CF
    if a.anchors:
        if not a.anchor_pi or a.anchor_w is None: ap.error("--anchors requires --anchor_pi and --anchor_w")
        if not (0.0 < a.anchor_w <= 1.0): ap.error("--anchor_w must be in (0, 1]")
        if a.features != "crops" or a.variant != "parts3": ap.error("--anchors needs --features crops --variant parts3 (the frozen matcher)")
    elif a.anchor_pi or a.anchor_w is not None or a.anchor_cache:
        ap.error("--anchor_pi / --anchor_w / --anchor_cache need --anchors")
    if a.boot == "none" and not a.anchors: ap.error("--boot none without --anchors leaves the label space undefined")
    if a.noncalf:
        NONCALF_MODE = noncalf_mode(json.load(open(a.noncalf, encoding="utf-8")), a.noncalf)
        if NONCALF_MODE == "anchor" and not a.anchors: ap.error("--noncalf requires --anchors (it thresholds the anchor score)")
        if NONCALF_MODE == "colour" and a.features != "crops":
            ap.error("--noncalf (colour) needs --features crops (its rows align with the crop bands npz)")
    if a.even_cap and a.temporal != "even": ap.error("--even_cap needs --temporal even")
    TEMPORAL = a.temporal; BOOT = a.boot; EVEN_CAP = a.even_cap or None; ANCHOR_W = a.anchor_w
    _EXT = bool(a.anchors) or a.temporal != "recency" or a.boot != "seeds"
    VARIANT = a.variant; PIGEONHOLE_MODE = a.pigeonhole
    if a.pool_h: POOL_H = a.pool_h; POOL_CAP_SEC = a.pool_h * 3600.0
    elif a.days: POOL_H = 36.0; POOL_CAP_SEC = 36 * 3600.0
    if a.days: print(f"  week mode: pool window {POOL_H:.0f} h (spans the overnight gap)")
    if a.topk: TOPK = a.topk
    t0 = time.time()
    if DAYS:
        WMAP = json.load(open(a.week_map))["global_ident_map"] if a.week_map else {}
        WMAP = {k: {int(a_): int(b_) for a_, b_ in v.items()} for k, v in WMAP.items()}
        segs, crops, CE, density = load_week(a.root, DAYS, a.features, a.bands, WMAP)
    else:
        segs, crops, CE, density = load_inputs(d, a.features, a.bands)
    EXT_PROV = None
    if _EXT:
        EXT_PROV = dict(temporal=TEMPORAL, even_cap=(EVEN_CAP or POOL_MAX_N) if TEMPORAL == "even" else None, boot=BOOT,
                        anchors=None, anchor_w=ANCHOR_W, spec="SPEC_anchor_reid.md + addendum 2026-09-26")
        if a.anchors:
            pool_ = load_anchor_pool(a.anchors, load=False)           # sha + manifest check; embeddings only on a cache miss
            P_, pj_ = load_pi(a.anchor_pi, pool_[3])
            inv_ = {g: c for c, g in P_.items()}
            cdir_ = a.anchor_cache or os.path.join(os.path.dirname(os.path.abspath(a.anchors)), "score_cache")
            parts_, bshas_ = [], {}
            for day_ in (DAYS or [a.day]):
                m5_, bsha_, _, cp_ = anchor_scores_for_band(os.path.join(a.root, day_, a.bands), a.anchors, cdir_, pool_)
                parts_.append(np.asarray(m5_, dtype=np.float64)[:, [inv_[g] for g in range(N)]]); bshas_[day_] = bsha_
            ANCHOR_S = np.concatenate(parts_, axis=0)
            assert len(ANCHOR_S) == len(CE), f"anchor scores {len(ANCHOR_S)} vs CE {len(CE)} rows"
            # pi is PROVISIONAL (B1 appearance check) until the specialist's blind coat key is reconciled (B3): only a pi
            # file whose "status" is "final" makes anchor results non-provisional (2026-09-26)
            EXT_PROV.update(anchors=os.path.abspath(a.anchors), anchor_sha256=pool_[3], pi=P_, pi_file=os.path.abspath(a.anchor_pi),
                            pi_sha256=sha256_file(a.anchor_pi), pi_status=pj_.get("status"),
                            pi_provisional=(bool(pj_.get("provisional", False)) or pj_.get("status") != "final"),
                            band_sha256=bshas_, topk=TOPK)
            if EXT_PROV["pi_provisional"]: print(f"  !! pi status {pj_.get('status')!r}: every anchor-arm result is PROVISIONAL")
            del pool_; _POOL_MEM.clear()
        if CE.ndim == 3 and VARIANT == "parts3": _CE3 = build_ce3(CE)
        print("  EXTENSION ON: " + json.dumps({k: v for k, v in EXT_PROV.items() if k not in ("band_sha256",)}, default=str))
    LINKS = {}
    if a.links:
        _lk = json.load(open(a.links))
        _rows = _lk.get(a.day) if isinstance(_lk, dict) and a.day in _lk else (_lk if isinstance(_lk, list) else [])
        for _r in _rows:
            LINKS[_r["seg"]] = (_r["prev"], float(_r.get("iou", 0.0)), float(_r.get("absence_s", 1e9)))
            if _r.get("seam"): LINKS[_r["seg"]] += (True,)      # seam rows only; reopen rows keep their 3-tuple
        n_ok = sum(1 for v in LINKS.values() if v[1] >= SOFT_MIN_LINK_IOU and v[2] <= SOFT_MAX_ABSENCE)
        print(f"  link table: {len(LINKS)} successions, {n_ok} clear IoU>={SOFT_MIN_LINK_IOU} within {SOFT_MAX_ABSENCE:.0f}s")
        n_seam = sum(1 for v in LINKS.values() if len(v) > 3)
        if n_seam:
            _ok = lambda v: v[1] >= SEAM_MIN_IOU and SEAM_MIN_ABSENCE <= v[2] <= SEAM_MAX_ABSENCE
            n_seam_ok = sum(1 for v in LINKS.values() if len(v) > 3 and _ok(v))
            print(f"  of which SEAM links (window seams): {n_seam}, {n_seam_ok} usable "
                  f"(IoU>={SEAM_MIN_IOU}, absence {SEAM_MIN_ABSENCE:.0f}..{SEAM_MAX_ABSENCE:.0f}s)")
            for k_, v in LINKS.items():
                if len(v) > 3 and not _ok(v): print(f"    REJECTED seam link {k_} <- {v[0]}: IoU {v[1]:.3f}, absence {v[2]:+.2f}s")
    n_g = sum(c["gated"] for c in crops); n_over = sum(c["over"] for c in crops)
    print(f"{a.day}: {len(segs)} segments, {len(crops)} {a.features} ({n_g} pass the gate, {n_over} over-extended), T={a.T} min")
    segs_all, crops_all, NC_SEGS, SPAN, NC_PROV = segs, crops, {}, None, None
    if a.noncalf and NONCALF_MODE == "anchor":
        NONCALF = load_noncalf(a.noncalf, EXT_PROV["anchor_sha256"])
        NC_SEGS = noncalf_segments(crops, ANCHOR_S, NONCALF)
        SPAN = (min(c["w"] for c in crops), max(c["w"] for c in crops) + timedelta(seconds=1))
        EXT_PROV["noncalf"] = dict(file=os.path.abspath(a.noncalf), sha256=sha256_file(a.noncalf), version=NONCALF.get("version"),
                                   crop_threshold=float(NONCALF["crop_threshold"]), min_gated_crops=int(NONCALF["min_gated_crops"]),
                                   frac_below=float(NONCALF["frac_below"]), n_hidden=len(NC_SEGS))
        NC_PROV = EXT_PROV["noncalf"]
        print(f"  NON-CALF filter {NONCALF.get('version')} (crop threshold {float(NONCALF['crop_threshold']):.4f}, >= "
              f"{int(NONCALF['min_gated_crops'])} gated crops, >= {float(NONCALF['frac_below']):.2f} below): {len(NC_SEGS)} segment(s) hidden")
        for sid in sorted(NC_SEGS, key=lambda x: (segs[x]["t_start"], segs[x]["slot"])):
            x = NC_SEGS[sid]
            print(f"    NONCALF {sid} slot {segs[sid]['slot']} {segs[sid]['t_start'].isoformat()} .. {segs[sid]['t_end'].isoformat()}: "
                  f"{x['n_below']}/{x['n_gated']} gated crops below (median score {x['score_median']:.4f}) -> hidden (no identity, no pool)")
        _nm = noncalf_segments(crops, ANCHOR_S, dict(NONCALF, min_gated_crops=1, frac_below=1e-9))
        _nm = sorted(((v["frac_below"], k, v["n_gated"]) for k, v in _nm.items() if k not in NC_SEGS), reverse=True)[:3]
        if _nm: print("    closest kept segments (frac below, gated crops): " + ", ".join(f"{k} {f:.2f} of {n_}" for f, k, n_ in _nm))
    elif a.noncalf:                                                          # colour mode (Tol_54 json)
        NONCALF = load_noncalf_colour(a.noncalf)
        _cnpz = NONCALF["statistic"]["npz"]
        COLOUR_CF, _csha = load_colour_frac(a.root, DAYS or [a.day], a.bands, _cnpz, len(CE))
        NC_SEGS = colour_noncalf_segments(crops, COLOUR_CF, NONCALF)
        SPAN = (min(c["w"] for c in crops), max(c["w"] for c in crops) + timedelta(seconds=1))
        _ng = [c for c in crops if c["gated"]]
        NC_PROV = dict(mode="colour", file=os.path.abspath(a.noncalf), sha256=sha256_file(a.noncalf), version=NONCALF.get("version"),
                       derived_from=NONCALF.get("derived_from"), statistic=NONCALF["statistic"]["name"],
                       crop_threshold=float(NONCALF["crop_threshold"]), min_gated_crops=int(NONCALF["min_gated_crops"]),
                       frac_above=float(NONCALF["frac_above"]), compare="coloured_frac > crop_threshold (strict)",
                       npz=_cnpz, npz_sha256=_csha, n_gated_crops=len(_ng),
                       n_gated_above=sum(1 for c in _ng if COLOUR_CF[c["i"]] > float(NONCALF["crop_threshold"])),
                       n_gated_nan=sum(1 for c in _ng if not np.isfinite(COLOUR_CF[c["i"]])), n_hidden=len(NC_SEGS))
        if EXT_PROV is not None: EXT_PROV["noncalf"] = NC_PROV
        print(f"  NON-CALF COLOUR filter {NONCALF.get('version')} (crop coloured_frac > {float(NONCALF['crop_threshold']):.4f}, >= "
              f"{int(NONCALF['min_gated_crops'])} gated crops, >= {float(NONCALF['frac_above']):.2f} above; {NC_PROV['n_gated_above']} of "
              f"{len(_ng)} gated crops above): {len(NC_SEGS)} segment(s) hidden")
        for sid in sorted(NC_SEGS, key=lambda x: (segs[x]["t_start"], segs[x]["slot"])):
            x = NC_SEGS[sid]
            print(f"    NONCALF {sid} slot {segs[sid]['slot']} {segs[sid]['t_start'].isoformat()} .. {segs[sid]['t_end'].isoformat()}: "
                  f"{x['n_above']}/{x['n_gated']} gated crops coloured (median coloured_frac {x['cf_median']}) -> hidden (no identity, no pool)")
        _nm = colour_noncalf_segments(crops, COLOUR_CF, dict(NONCALF, min_gated_crops=1, frac_above=1e-9))
        _nm = sorted(((v["frac_above"], k, v["n_gated"]) for k, v in _nm.items() if k not in NC_SEGS), reverse=True)[:3]
        if _nm: print("    closest kept segments (frac above, gated crops): " + ", ".join(f"{k} {f:.2f} of {n_}" for f, k, n_ in _nm))
    if a.noncalf:
        segs = {k: v for k, v in segs.items() if k not in NC_SEGS}
        crops = [c for c in crops if c["sid"] not in NC_SEGS]
    HID = {k: segs_all[k] for k in NC_SEGS} or None                          # --noncalf: continuity barriers only
    runs, decisions, flags, day0, end, starved = run_layer(segs, crops, CE, a.T, density, LINKS, span=SPAN, hidden=HID)
    if starved:
        # An identity with no day-start seed only ever gets labelled in the POST-pass (pigeonhole / modal /
        # inherit), which runs after all feeding — so its pool stays empty, it is unmatchable, and everything
        # that really is that animal matches someone else and is then demoted by the bijection pass. 04-26 is
        # the case: its day start produced only THREE seeds, identity 3 held 2.2 h of 18, and 819 min of its
        # slot-time went UNASSIGNED. Re-run once with those identities' pools warm-started from pass-1 labels.
        by_lab = defaultdict(list)
        for sid, rl in runs.items():
            for r in rl:
                if r["label"] in starved:
                    for c in crops:
                        if c["sid"] == sid and r["start"] <= c["w"] < (r["end"] or end) and c["gated"]:
                            by_lab[r["label"]].append(c["i"])
        seeds = {k: v for k, v in by_lab.items() if v}
        print(f"  starved identities in pass 1: {sorted(starved)}; warm-starting {{"
              + ", ".join(f"{k}: {len(v)} crops" for k, v in sorted(seeds.items())) + "} and re-running")
        if seeds:
            runs, decisions, flags, day0, end, starved2 = run_layer(segs, crops, CE, a.T, density, LINKS, seeds, span=SPAN, hidden=HID)
            if starved2: print(f"  still starved after pass 2: {sorted(starved2)}")
    def _hidden(day_=None):
        """the relabel "hidden" list of the NON-CALF segments [of one day], in (start, slot) order"""
        out_ = []
        for sid in sorted(NC_SEGS, key=lambda x: (segs_all[x]["t_start"], segs_all[x]["slot"])):
            s_ = segs_all[sid]
            if day_ is not None and s_.get("day") != day_: continue
            out_.append(dict(slot=int(s_["slot"]), **{"from": s_["t_start"].isoformat(), "to": s_["t_end"].isoformat()},
                             seg=s_.get("local_seg", sid), why="noncalf", **({"mode": "colour"} if NONCALF_MODE == "colour" else {}),
                             **NC_SEGS[sid]))
        return out_
    if NONCALF is not None and NONCALF_MODE == "anchor":
        for sid, h_ in zip(sorted(NC_SEGS, key=lambda x: (segs_all[x]["t_start"], segs_all[x]["slot"])), _hidden()):
            flags.append(dict(kind="noncalf", sid=sid, slot=h_["slot"], t=h_["from"], t_end=h_["to"], n_gated=h_["n_gated"],
                              n_below=h_["n_below"], frac_below=h_["frac_below"], score_median=h_["score_median"],
                              why="crops look like none of the four calves: no identity, no pool, hidden in the overlay"))
    elif NONCALF is not None:                                                # colour mode
        for sid, h_ in zip(sorted(NC_SEGS, key=lambda x: (segs_all[x]["t_start"], segs_all[x]["slot"])), _hidden()):
            flags.append(dict(kind="noncalf", mode="colour", sid=sid, slot=h_["slot"], t=h_["from"], t_end=h_["to"],
                              n_gated=h_["n_gated"], n_above=h_["n_above"], frac_above=h_["frac_above"], cf_median=h_["cf_median"],
                              why="crops are clearly coloured (not black / white / grey): not one of the four black-and-white "
                                  "calves: no identity, no pool, hidden in the overlay"))
    if DAYS:
        # score each day against its own GT, translating that day's LOCAL gt labels into GLOBAL identities
        print()
        for day in DAYS:
            gp = os.path.join(a.root, day, "segment_assignments_reviewed.json")
            if not os.path.exists(gp): gp = os.path.join(a.root, day, "segment_assignments.json")
            if not os.path.exists(gp): print(f"  {day}: no GT"); continue
            gmap = WMAP.get(day, {i: i for i in range(N)})
            gasg = {f"{day}/{k}": (None if v is None else gmap.get(int(v))) for k, v in json.load(open(gp))["assignments"].items()}
            dcrops = [c for c in crops_all if c.get("day") == day]
            sc_d = score(segs, dcrops, runs, gasg, end, a.T)
            print(f"  {day}: {100*sc_d['acc_both']:6.2f} % on {sc_d['n_both']:5d} crops "
                  f"(day {100*sc_d['acc_by_phase']['day']:.1f} / dusk {100*sc_d['acc_by_phase']['dusk']:.1f} / night {100*sc_d['acc_by_phase']['night']:.1f}); "
                  f"SILENT {sc_d['silent_errors']}; unflagged {100*sc_d['acc_unflagged']:.2f} %; coverage {100*sc_d['our_coverage']:.1f} %")
        # one relabel file per day, already in GLOBAL identities (ready for Tol_17)
        for day in DAYS:
            rules = defaultdict(list)
            for sid, rl in runs.items():
                if segs[sid].get("day") != day: continue
                for r in rl:
                    if r["label"] is None: continue
                    rules[str(segs[sid]["slot"])].append([r["start"].isoformat(), (r["end"] or segs[sid]["t_end"]).isoformat(), int(r["label"])])
            for k in rules: rules[k].sort()
            _rel = {"rules": rules, "exclusions": [], "day": day,
                    "source": f"Tol_30 WEEK mode (accumulating pool), T={a.T} min, global identities"}
            if EXT_PROV is not None: _rel["extension"] = EXT_PROV          # extension provenance (never in production)
            if NONCALF is not None: _rel["hidden"] = _hidden(day)          # --noncalf only: Tol_17 draws nothing there
            if NONCALF is not None and EXT_PROV is None: _rel["noncalf"] = NC_PROV   # colour mode without anchors only
            json.dump(_rel, open(os.path.join(out_dir, f"relabel_week_{day}.json"), "w"), indent=1)
        print("")
        print("  written %d per-day relabel files (global identities) to %s" % (len(DAYS), out_dir))
        _wk = {"days": DAYS, "T": a.T, "flags": flags,
               "runs": {sid: [dict(start=r["start"].isoformat(), end=(r["end"] or segs[sid]["t_end"]).isoformat(),
                                   label=r["label"], flags=r["flags"], n_windows=len(r["win"]),
                                   n_conf=sum(1 for x in r["win"] if x["conf"]), settled=r["settled"]) for r in rl]
                        for sid, rl in runs.items()}}
        if EXT_PROV is not None:
            # extension only: provenance + every scored run-window with the blend's parts (M7 diagnostics)
            _wk["extension"] = EXT_PROV
            _wk["decisions"] = [ext_decision(x) for x in decisions]
        if NONCALF is not None:                                            # --noncalf only
            _wk["noncalf"] = dict(NC_PROV, hidden=dict(zip(
                sorted(NC_SEGS, key=lambda x: (segs_all[x]["t_start"], segs_all[x]["slot"])), _hidden())))
        json.dump(_wk, open(os.path.join(out_dir, "periodic_identity_WEEK.json"), "w"), indent=1, default=str)
        return
    gtj = json.load(open(os.path.join(d, a.gt), encoding="utf-8"))
    if "intervals" in gtj:
        # interval ground truth from the reviewers' notes (label -> true calf per wall-clock interval):
        # score per CROP, not per segment, and drop crops near a change point / in an excluded span
        G = {int(k): int(v) for k, v in gtj["global_ident_map"].items()}
        def _w(x):
            h, m_, sec = (int(y) for y in x.split(":")); dd = datetime.strptime(a.day, "%Y%m%d")
            if h >= 24: h -= 24; dd += timedelta(days=1)
            return dd.replace(hour=h, minute=m_, second=sec)
        ivs = [(_w(f), _w(t_), {int(k): v for k, v in mp.items()}) for f, t_, mp in gtj["intervals"]]
        exc = [(_w(f), _w(t_)) for f, t_, *_ in gtj.get("excluded", [])]
        marg = float(gtj.get("exclude_margin_sec", 180))
        cps = sorted({x[0] for x in ivs} | {x[1] for x in ivs})
        def truth_of(shown_label, w):
            """the calf actually carrying `shown_label` at wall time w (None where GT is undefined)"""
            if any(f <= w < t_ for f, t_ in exc): return None
            if any(abs((w - cp).total_seconds()) <= marg for cp in cps): return None
            for f, t_, mp in ivs:
                if f <= w < t_: return mp.get(int(shown_label))
            return None
        gt = {"__intervals__": {"G": G, "truth": truth_of}}
    else:
        gt = gtj["assignments"]
    sc = score(segs_all, crops_all, runs, gt, end, a.T)
    n_runs = sum(len(v) for v in runs.values()); n_lab = sum(1 for v in runs.values() for r in v if r["label"] is not None)
    conf = [x for x in decisions if x["conf"]]
    kinds = Counter(f["kind"] for f in flags); st_ = Counter(x["status"] for x in decisions); n_chaos = sum(1 for x in decisions if x["chaos"])
    print(f"  windows scored: {len(decisions)} run-windows, {len(conf)} confident ({100*len(conf)/max(len(decisions),1):.0f} %), "
          f"mean margin {np.mean([x['margin'] for x in decisions]):.3f}; status {dict(st_)}; in chaos {n_chaos}")
    print(f"  runs: {n_runs} ({n_lab} labelled) from {len(segs)} segments; flags: {dict(kinds)}")
    print(f"  vs GT: {100*sc['acc_both']:.2f} % on {sc['n_both']} crops (day {100*sc['acc_by_phase']['day']:.1f} / dusk {100*sc['acc_by_phase']['dusk']:.1f} / night {100*sc['acc_by_phase']['night']:.1f}); "
          f"SILENT errors {sc['silent_errors']}, flagged errors {sc['flagged_errors']}; flagged crops {sc['flagged_crops']} ({100*sc['flagged_crops']/max(sc['n_crops'],1):.1f} %); "
          f"unflagged accuracy {100*sc['acc_unflagged']:.2f} % on {sc['n_unflagged']}; coverage ours {100*sc['our_coverage']:.1f} % vs GT {100*sc['gt_coverage']:.1f} % (gained {sc['gained_vs_gt']}, lost {sc['lost_vs_gt']} crops)")
    for x in sc["disagreements"]: print("   DISAGREE", x)
    for x in sc["gained"]: print("   GAINED  ", x)
    for x in sc["lost"]: print("   LOST    ", x)
    for x in flags:
        if x["kind"] in ("hop", "contested", "reassigned", "pigeonhole", "pigeonhole_conflict"): print("   FLAG    ", x)
    rules = defaultdict(list)
    for sid, rl in runs.items():
        for r in rl:
            if r["label"] is None: continue
            rules[str(segs[sid]["slot"])].append([r["start"].isoformat(), (r["end"] or segs[sid]["t_end"]).isoformat(), int(r["label"])])
    for k in rules: rules[k].sort()
    tag = f"T{int(a.T) if float(a.T).is_integer() else a.T}" + (f"_k{a.topk}" if a.topk else "") + ("" if a.features == "crops" else "_obs") + ("" if a.variant == "parts3" else "_whole")
    _rel = {"rules": rules, "exclusions": [], "source": f"Tol_30 periodic identity layer, T={a.T} min", "day": a.day}
    if EXT_PROV is not None: _rel["extension"] = EXT_PROV                  # extension provenance (never in production)
    if NONCALF is not None: _rel["hidden"] = _hidden()                     # --noncalf only
    if NONCALF is not None and EXT_PROV is None: _rel["noncalf"] = NC_PROV # colour mode without anchors only
    json.dump(_rel, open(os.path.join(out_dir, f"relabel_periodic_{a.day}_{tag}.json"), "w"), indent=1)
    detail = {"day": a.day, "T": a.T, "params": dict(tau=TAU, gap_min=GAP_MIN, pool_h=POOL_H, pool_cap_sec=POOL_CAP_SEC, min_q=MIN_Q, k_hop=K_HOP, settle_win=SETTLE_WIN, settle_min=SETTLE_MIN,
                                                     chaos_events=CHAOS_EVENTS, chaos_min=CHAOS_MIN, area_over=AREA_OVER, gate_area=GATE_AREA, gate_nb=GATE_NB, soft_max_absence=SOFT_MAX_ABSENCE),
              "score": sc, "flags": flags,
              "runs": {sid: [dict(start=r["start"].isoformat(), end=(r["end"] or segs[sid]["t_end"]).isoformat(), label=r["label"], flags=r["flags"],
                                  n_windows=len(r["win"]), n_conf=sum(1 for x in r["win"] if x["conf"]), settled=r["settled"],
                                  agg=[round(float(r["agg"][k] / r["agg_n"][k]), 4) if r["agg_n"][k] > 0 else None for k in range(N)]) for r in rl] for sid, rl in runs.items()},
              "decisions": [dict(sid=x["sid"], slot=x["slot"], t=x["t"].isoformat(), pred=x["pred"], margin=round(x["margin"], 4), argmax=x["argmax"], conf=x["conf"],
                                 status=x["status"], chaos=x["chaos"], sims=x["sims"], n_q=x["n_q"]) for x in decisions]}
    if EXT_PROV is not None:                                               # extension only (production file unchanged)
        detail["extension"] = EXT_PROV; detail["decisions"] = [ext_decision(x) for x in decisions]
    if NONCALF is not None: detail["noncalf"] = dict(NC_PROV, hidden=_hidden())
    json.dump(detail, open(os.path.join(out_dir, f"periodic_identity_{a.day}_{tag}.json"), "w"), indent=1, default=str)
    print(f"  written {out_dir}/periodic_identity_{a.day}_{tag}.json + relabel_periodic_{a.day}_{tag}.json  ({time.time()-t0:.1f} s)")

if __name__ == "__main__":
    main()
