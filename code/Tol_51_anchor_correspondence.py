"""Tol_51 — B1 SEALED APPEARANCE CHECK. Local CPU only.

Question: which 04-22 week-global identity g is which 04-18/19 training coat c (0 Black / 1 White / 2 half&half /
3 two dots on the back)? The frozen 04-18/19 anchor pool is matched against the VERIFIED 04-22 identities with the
frozen matcher (DINOv3-ViT-S bands, parts-3, orientation-max, /3, per-crop mean of the top-5 anchor similarities =
Tol_30.pair_sim with one query), and aggregated over the WHOLE day:

    S[c, g] = mean over the gated dense crops whose reviewed-GT identity is g of mean5(crop, anchors of coat c)

Hungarian over the 4x4 S -> pi_app : coat c -> global g. Reported with per-pair margins (query side = the LODO preview's
diag margin; anchor side), the best-vs-second-best permutation gap, a per-phase breakdown (phase4), an hourly-block
Hungarian (modal permutation + support), a 1,000 x 10-min-block bootstrap (share returning pi_app), per-crop argmax and
3-min-window consistency with pi_app (comparable to the LODO preview table), and a hub check.

STATUS OF THE OUTPUT: PROVISIONAL. pi_app is an appearance opinion. By design the specialist's blind coat key
(B2) is the authority; until it arrives, pi_provisional.json lets the anchor arms run, and every anchor-arm result is
flagged provisional (re-scored if the key differs). pi is never FINALLY set from appearance.

LEAKAGE GUARDS (hard):
  * anchors: 04-18/19 only (the frozen pool's manifest / day field is checked; the temporary pool asserts every band
    path is under /Tracking_train/2024041[89]_).
  * queries: 04-22 only (dir, GT file and every band path are asserted to be 04-22 Tracking_v4v2). Nothing from
    04-23..28 is opened.
  * the specialist's coat key is never read: any path containing "coat_key" or "SECRET" is refused.

POOL: --pool <anchor_pool_v1.npz> (Tol_50; sha256 checked against its manifest by Tol_30.load_anchor_pool). If it does
not exist yet and --allow_temp_pool is given, a TEMPORARY pool is built in memory with the SAME gate chain as
anchor_sizing/size_anchors.py + lodo_anchor_check.py (gates G-a..G-e, per-phase balance, seed 0) and the outputs carry
pool_kind = TEMPORARY and anchor_sha256 = "TEMPORARY-POOL:<content sha>", which Tol_30.load_pi refuses against any
real pool. Re-run once anchor_pool_v1.npz exists.

Outputs (out_dir, default $SP6/train_gt/anchor_pool/):
  anchor_correspondence_B1.json   full sealed record (all matrices, margins, bootstrap, hourly, phases, input sha256s)
  pi_provisional.json             {"pi": {coat: global}, "anchor_sha256", "status", ...}  (Tol_30 --anchor_pi)
  pi_provisional_table.md         the 4x4 table + margins + per-phase / hourly / bootstrap summary
  anchor_correspondence_B1.sha256 sha256 of the three files above (the B1 seal, logged before B2)

Usage:
  python Tol_51_anchor_correspondence.py [--pool .../anchor_pool_v1.npz] [--allow_temp_pool] [--out_dir DIR]
"""
import os, re, sys, json, glob, time, argparse, hashlib, itertools
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

NB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NB)
import Tol_30_periodic_identity as T  # noqa: E402  (load_inputs = production crop gate; anchor_mean5 = frozen matcher)

SP6 = r"<LOCAL_RUN_DIR>"
SPA = r"<LOCAL_RUN_DIR>/ablation"
DAY_Q = "20240422"
TRAIN_DAYS = ("20240418", "20240419")
FORBIDDEN_DAYS = ("20240423", "20240424", "20240425", "20240426", "20240427", "20240428")
PHASES = ("day", "dawn", "dusk", "night")
N = T.N
BANDS = "crop_bands3_dense.npz"
HOUR_MIN_CROPS = 20     # hourly block scored only if every identity has >= 20 gated crops (5 crop-min) in that hour
BLOCK_MIN = 10          # bootstrap block length (min)
WIN_MIN = 3             # 3-min window consistency (= production T, = LODO preview)
N_BOOT = 1000
SEED = 0

def phase4(w):
    """phase4 (same as Tol_42/44 and the anchor pool): night >= 21 or < 6, dawn 6-7, dusk 20, day 8-19"""
    h = w.hour
    if h >= 21 or h < 6: return "night"
    if h < 8: return "dawn"
    if h >= 20: return "dusk"
    return "day"

def guard(p):
    q = str(p).replace("\\", "/")
    if re.search(r"coat_key|SECRET", q, flags=re.I):
        raise SystemExit(f"refusing to open {q}: the specialist's coat key stays blind (B2)")
    return p

def guard_query_path(p):
    q = str(p).replace("\\", "/")
    for d in FORBIDDEN_DAYS:
        if d in q: raise SystemExit(f"LEAKAGE GUARD: {q} names test day {d}; B1 reads 04-22 only")
    return guard(p)

def sha(p): return T.sha256_file(guard(p))

def find_key(o, key):
    """first value of `key` anywhere in a nested json object (manifest layouts may nest provenance)"""
    if isinstance(o, dict):
        if key in o: return o[key]
        for v in o.values():
            r = find_key(v, key)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = find_key(v, key)
            if r is not None: return r
    return None

# ----------------------------------------------------------------------------- anchor pools
def load_real_pool(path):
    """the frozen Tol_50 pool. Tol_30.load_anchor_pool refuses a file whose sha256 differs from its manifest."""
    guard(path)
    mp = path[:-4] + ".manifest.json"
    if not os.path.exists(mp): raise SystemExit(f"anchor pool manifest missing: {mp}")
    man = json.load(open(mp, encoding="utf-8"))
    evidence = {}
    df = find_key(man, "derived_from")
    if df is not None:
        df = sorted(str(x) for x in df)
        evidence["manifest_derived_from"] = df
        if df != sorted(TRAIN_DAYS): raise SystemExit(f"LEAKAGE GUARD: pool derived_from {df} != {list(TRAIN_DAYS)}")
    nf = find_key(man, "not_from")
    if nf is not None: evidence["manifest_not_from"] = sorted(str(x) for x in nf)
    A3, A3R, ident, a_sha = T.load_anchor_pool(path)
    z = np.load(path, allow_pickle=False)
    extra = {}
    for k in ("day", "window", "band_row", "phase4"):
        if k in z.files:
            try: extra[k] = z[k]
            except Exception as e: extra[k] = None; evidence[f"{k}_unreadable"] = repr(e)
    if extra.get("day") is not None:
        days = sorted(set(str(x) for x in extra["day"]))
        evidence["npz_days"] = days
        if not set(days) <= set(TRAIN_DAYS): raise SystemExit(f"LEAKAGE GUARD: pool holds days {days}")
    if "manifest_derived_from" not in evidence and "npz_days" not in evidence:
        raise SystemExit("cannot verify that the pool is 04-18/19-only (no derived_from in the manifest, no day field)")
    info = dict(pool_kind="FROZEN", path=os.path.abspath(path), sha256=a_sha, manifest=os.path.abspath(mp),
                manifest_sha256=sha(mp), source_evidence=evidence,
                n_per_identity={int(k): int((ident == k).sum()) for k in range(N)})
    if extra.get("phase4") is not None:
        info["n_per_identity_phase"] = {f"{k}|{p}": int(((ident == k) & (extra["phase4"].astype(str) == p)).sum())
                                        for k in range(N) for p in PHASES}
    keys = None
    if extra.get("window") is not None and extra.get("band_row") is not None:
        keys = set(zip((str(x) for x in extra["window"]), (int(x) for x in extra["band_row"])))
    return A3, A3R, ident.astype(np.int64), a_sha, info, keys

def build_temp_pool():
    """TEMPORARY pool with the gate chain of anchor_sizing/size_anchors.py and the balance of
    lodo_anchor_check.py (per phase, every identity subsampled to the phase minimum, rng seed 0, no
    replacement). Constants read from the training-only tune_scout summary."""
    P = datetime.fromisoformat
    root = SP6 + "/train_ident/root"; rel = SP6 + "/train_gt/relabel_volume"
    gtp = SP6 + "/train_gt/train_identity_gt.json"; tsp = SP6 + "/tune_scout/train_summary_segv1d.json"
    GT = json.load(open(guard(gtp), encoding="utf-8")); TS = json.load(open(guard(tsp), encoding="utf-8"))
    if sorted(TS["source_days"]) != sorted(TRAIN_DAYS): raise SystemExit(f"tune_scout source_days {TS['source_days']}")
    AREF = {d: float(TS["constants"][f"A_ref[{d}]"]["value"]) for d in TRAIN_DAYS}
    FLO = {k: float(v) for k, v in TS["band"]["frag_lo"].items()}; OHI = float(TS["band"]["over_hi"])
    assert AREF == {"20240418": 86500.0, "20240419": 87774.0} and FLO == {"dayl": 0.44, "dark": 0.43} and OHI == 1.32, \
        "gate constants differ from the frozen G-e band constants"
    wins = sorted(os.path.basename(d) for d in glob.glob(root + "/*") if os.path.isdir(d))
    if len(wins) != 15 or not all(re.fullmatch(r"2024041[89]_w\d{4}-\d{4}", w) for w in wins):
        raise SystemExit(f"unexpected training windows {wins}")
    rel_sha = GT["relabel_source"]["sha256"]; in_sha = {}
    for w in wins:                                                   # P3: the files the specialist watched
        f = f"relabel_week_{w}.json"; s = sha(os.path.join(rel, f)); in_sha[f] = s
        if s != rel_sha[f]: raise SystemExit(f"P3: {f} sha {s} != train_identity_gt relabel_source {rel_sha[f]}")
    exc = defaultdict(list)                                          # G-c specialist faults
    for f in GT["faults"]:
        for k in f["identities"]: exc[k].append((P(f["exclude_start"]), P(f["exclude_end"])))
    strict = defaultdict(list)                                       # G-d unlisted mask evidence +-90 s (+ absorber)
    for it in GT["unlisted_mask_evidence"]["items"]:
        if it["type"] == "no_masks_at_all": continue
        ids = set(it.get("identities") or []); ab = it.get("absorber")
        if isinstance(ab, int): ids.add(ab)
        for k in ids: strict[k].append((P(it["wall_start"]) - timedelta(seconds=90), P(it["wall_end"]) + timedelta(seconds=90)))
    cnt = defaultdict(Counter); rows, embs = [], []
    for win in wins:
        d = os.path.join(root, win); day = win[:8]
        bp = os.path.join(d, BANDS); in_sha[f"{win}/{BANDS}"] = sha(bp)
        paths = [str(p).replace("\\", "/") for p in np.load(bp, allow_pickle=True)["paths"]]
        bad = [p for p in paths if f"/Tracking_train/{win}/" not in p]
        if bad: raise SystemExit(f"P2: {len(bad)} band paths of {win} are not under /Tracking_train/{win}/ e.g. {bad[0]}")
        segs, crops, CE, _ = T.load_inputs(d, "crops", BANDS)
        R = {int(s): [(P(a), P(b), int(k)) for a, b, k in v]
             for s, v in json.load(open(os.path.join(rel, f"relabel_week_{win}.json"), encoding="utf-8"))["rules"].items()}
        for c in crops:
            w = c["w"]; ph = phase4(w)
            if not c["gated"]: continue                                                  # G-a
            lab = next((k for a, b, k in R.get(c["slot"], []) if a <= w < b), None)
            cnt["gated"][(lab, ph)] += 1
            if lab is None: continue                                                     # G-b
            if any(a <= w < b for a, b in exc[lab]): continue                            # G-c
            cnt["spec_excl"][(lab, ph)] += 1
            if any(a <= w < b for a, b in strict[lab]): continue                         # G-d
            cnt["strict_excl"][(lab, ph)] += 1
            fl = FLO["dayl"] if ph == "day" else FLO["dark"]
            ar = c["area"] or 0.0
            if not (fl * AREF[day] <= ar <= OHI * AREF[day]): cnt["band_reject"][(lab, ph)] += 1; continue   # G-e
            cnt["band"][(lab, ph)] += 1
            rows.append((win, int(c["i"]), int(lab), ph)); embs.append(CE[c["i"]])
    E = np.stack(embs).astype(np.float32); del embs
    rng = np.random.default_rng(SEED); idx = defaultdict(list)
    for i, r in enumerate(rows): idx[(r[2], r[3])].append(i)
    sel, bal = [], {}
    for p in PHASES:
        n = min(len(idx[(k, p)]) for k in range(N)); bal[p] = int(n)
        for k in range(N):
            if n: sel += list(rng.choice(idx[(k, p)], n, replace=False))
    sel = np.array(sel, dtype=np.int64)
    E = np.ascontiguousarray(E[sel]); ident = np.array([rows[i][2] for i in sel], dtype=np.int64)
    h = hashlib.sha256(); h.update(E.tobytes()); h.update(ident.astype(np.int8).tobytes())
    a_sha = "TEMPORARY-POOL:" + h.hexdigest()
    A3, A3R = T.build_ce3(E)
    band_counts = {f"{k}|{p}": int(cnt["band"][(k, p)]) for k in range(N) for p in PHASES}
    sizing_match = None
    szp = SP6 + "/train_gt/anchor_sizing/anchor_sizing.json"
    if os.path.exists(szp):
        ref = json.load(open(szp))["counts"]["band"]
        sizing_match = all(int(ref.get(k, 0)) == v for k, v in band_counts.items())
    in_sha["train_identity_gt.json"] = sha(gtp); in_sha["train_summary_segv1d.json"] = sha(tsp)
    info = dict(pool_kind="TEMPORARY", sha256=a_sha, built_by="Tol_51 build_temp_pool (size_anchors.py gate chain + lodo balance, seed 0)",
                windows=wins, band_counts_per_identity_phase=band_counts, matches_anchor_sizing_json=sizing_match,
                balanced_per_identity_per_phase=bal, n_per_identity={int(k): int((ident == k).sum()) for k in range(N)},
                input_sha256=in_sha, derived_from=list(TRAIN_DAYS), not_from=list((DAY_Q,) + FORBIDDEN_DAYS))
    keys = set((rows[i][0], rows[i][1]) for i in sel)
    return A3, A3R, ident, a_sha, info, keys

# ----------------------------------------------------------------------------- analysis helpers
def smat(M5, g_arr, mask):
    """S[c, g] = mean over masked crops of GT identity g of M5[:, c]; n[g] = crops used"""
    S = np.full((N, N), np.nan); n = np.zeros(N, np.int64)
    for g in range(N):
        s = mask & (g_arr == g); n[g] = int(s.sum())
        if n[g]: S[:, g] = M5[s].mean(axis=0)
    return S, n

def hung(S):
    r, c = linear_sum_assignment(-S)
    return tuple(int(c[list(r).index(i)]) for i in range(N))          # pi[c] = g

def margins(S, pi):
    pinv = {g: c for c, g in enumerate(pi)}
    q = [float(S[pinv[g], g] - max(S[c, g] for c in range(N) if c != pinv[g])) for g in range(N)]
    a = [float(S[c, pi[c]] - max(S[c, g] for g in range(N) if g != pi[c])) for c in range(N)]
    return q, a

def perm_rank(S):
    sc = sorted(((float(np.mean([S[c, p[c]] for c in range(N)])), p) for p in itertools.permutations(range(N))), reverse=True)
    return sc

def rnd(x, k=4):
    if isinstance(x, (list, tuple)): return [rnd(v, k) for v in x]
    if isinstance(x, np.ndarray): return rnd(x.tolist(), k)
    if isinstance(x, float): return None if np.isnan(x) else round(x, k)
    return x

def pistr(pi, names):
    return ", ".join(f"{c} {names[c]} -> G{pi[c]}" for c in range(N))

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=SP6 + "/train_gt/anchor_pool/anchor_pool_v1.npz")
    ap.add_argument("--allow_temp_pool", action="store_true",
                    help="if --pool does not exist, build a TEMPORARY pool with the same gate code (outputs marked TEMPORARY)")
    ap.add_argument("--query_root", default=SPA)
    ap.add_argument("--out_dir", default=SP6 + "/train_gt/anchor_pool")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows cp1252 console cannot print the table
    except Exception: pass
    t0 = time.time()
    assert T.TOPK == 5 and T.VARIANT == "parts3" and T.P3 == [1, 2, 3] and T.P3R == [5, 6, 7], "frozen matcher changed"
    for p in (a.pool, a.query_root, a.out_dir): guard(p)
    os.makedirs(a.out_dir, exist_ok=True)
    names = json.load(open(SP6 + "/train_gt/train_identity_gt.json", encoding="utf-8"))["identity_key"]
    names = {int(k): v for k, v in names.items()}

    # ---- anchors
    if os.path.exists(a.pool):
        A3, A3R, ident, a_sha, pinfo, pkeys = load_real_pool(a.pool)
    elif a.allow_temp_pool:
        print(f"[{time.time()-t0:6.1f}s] {a.pool} absent -> building a TEMPORARY pool (same gate code)", flush=True)
        A3, A3R, ident, a_sha, pinfo, pkeys = build_temp_pool()
    else:
        raise SystemExit(f"{a.pool} does not exist (pass --allow_temp_pool for a TEMPORARY pool)")
    nper = [int((ident == k).sum()) for k in range(N)]
    print(f"[{time.time()-t0:6.1f}s] pool {pinfo['pool_kind']} {a_sha[:40]}: {len(ident)} anchors, per coat {nper}", flush=True)
    if len(set(nper)) != 1: print(f"  WARNING: pool not balanced across identities: {nper}")

    # ---- 04-22 queries (verified GT, gated dense crops)
    qdir = guard_query_path(os.path.join(a.query_root, DAY_Q))
    bp = guard_query_path(os.path.join(qdir, BANDS)); gp = guard_query_path(os.path.join(qdir, "segment_assignments_reviewed.json"))
    qpaths = [str(p).replace("\\", "/") for p in np.load(bp, allow_pickle=True)["paths"]]
    badq = [p for p in qpaths if f"/Tracking_v4v2/{DAY_Q}/" not in p]
    if badq: raise SystemExit(f"LEAKAGE GUARD: {len(badq)} query band paths not under /Tracking_v4v2/{DAY_Q}/, e.g. {badq[0]}")
    gt = json.load(open(gp, encoding="utf-8"))
    if gt.get("day") != DAY_Q: raise SystemExit(f"GT file day {gt.get('day')} != {DAY_Q}")
    asg = gt["assignments"]
    segs, crops, CE, _ = T.load_inputs(qdir, "crops", BANDS)
    assert CE.shape[1:] == (8, 384) and len(CE) == len(qpaths), f"band layout {CE.shape}"
    drop = Counter()
    Q = []
    for c in crops:
        if not c["gated"]: drop["ungated"] += 1; continue
        g = asg.get(c["sid"], "__missing__")
        if g == "__missing__": drop["segment_not_in_gt"] += 1; continue
        if g is None: drop["gt_unassigned"] += 1; continue
        Q.append((c, int(g)))
    print(f"[{time.time()-t0:6.1f}s] 04-22: {len(crops)} crops, {len(Q)} gated with GT; dropped {dict(drop)}", flush=True)
    qi = np.array([c["i"] for c, _ in Q]); g_arr = np.array([g for _, g in Q])
    walls = [c["w"] for c, _ in Q]; ph_arr = np.array([phase4(w) for w in walls])
    Q3 = np.ascontiguousarray(CE[qi][:, T.P3, :].reshape(len(qi), -1))
    M5 = T.anchor_mean5(Q3, A3, A3R, ident).astype(np.float64)        # [nq, 4] coat columns
    print(f"[{time.time()-t0:6.1f}s] mean5 computed {M5.shape}", flush=True)
    # independent brute-force check (full sort instead of partition) on 16 random query rows
    rng = np.random.default_rng(SEED); chk = rng.choice(len(qi), min(16, len(qi)), replace=False); maxdiff = 0.0
    for k in range(N):
        sel = np.flatnonzero(ident == k)
        s = np.maximum(Q3[chk] @ A3[sel].T, Q3[chk] @ A3R[sel].T) / 3.0            # [16, n_anchors(k)]
        ref = np.sort(s, axis=1)[:, -T.TOPK:].mean(axis=1)
        maxdiff = max(maxdiff, float(np.abs(ref - M5[chk, k]).max()))
    assert maxdiff < 1e-5, f"mean5 self-check failed: max diff {maxdiff}"

    allm = np.ones(len(qi), bool)
    # ---- day level
    S, n = smat(M5, g_arr, allm)
    pi = hung(S); qm, am = margins(S, pi); rank = perm_rank(S)
    gap = rank[0][0] - rank[1][0]
    second = rank[1][1]; diff_pairs = [c for c in range(N) if second[c] != pi[c]]
    argmax_col = [int(np.argmax(S[:, g])) for g in range(N)]     # best coat per 04-22 identity
    argmax_row = [int(np.argmax(S[c, :])) for c in range(N)]     # best 04-22 identity per coat
    hub_anchor_spread = float(np.ptp(np.nanmean(S, axis=1))); query_spread = float(np.ptp(np.nanmean(S, axis=0)))
    pinv = {g: c for c, g in enumerate(pi)}
    crop_arg = np.argmax(M5, axis=1); crop_ok = np.array([pi[crop_arg[j]] == g_arr[j] for j in range(len(qi))])

    # ---- per phase
    phases = {}
    for p in PHASES:
        m = ph_arr == p
        if not m.any(): continue
        Sp, np_ = smat(M5, g_arr, m)
        ent = dict(n_crops_per_global=np_.tolist(), matrix_c_by_g=rnd(Sp))
        if (np_ > 0).all():
            pp = hung(Sp); q_, a_ = margins(Sp, pp); rk = perm_rank(Sp)
            ent.update(hungarian_pi=list(pp), equals_pi_app=(pp == pi), query_margins_under_own_pi=rnd(q_),
                       query_margins_under_pi_app=rnd(margins(Sp, pi)[0]), anchor_margins_under_pi_app=rnd(margins(Sp, pi)[1]),
                       perm_gap=round(rk[0][0] - rk[1][0], 5),
                       crop_argmax_consistency_with_pi_app=round(float(crop_ok[m].mean()) * 100, 2))
        phases[p] = ent

    # ---- per GT segment (diagnostic only: is a phase-level disagreement spread over every segment of that phase — a
    #      domain effect — or concentrated in a few segments?)
    sids = np.array([c["sid"] for c, _ in Q]); per_seg = []
    for sid in sorted(set(sids.tolist()), key=lambda s: segs[s]["t_start"]):
        m = sids == sid; row = M5[m].mean(axis=0); g = int(g_arr[m][0]); c_true = pinv[g]; am_ = int(np.argmax(row))
        ws_ = [walls[j] for j in np.flatnonzero(m)]
        per_seg.append(dict(seg=sid, g=g, n=int(m.sum()), start=min(ws_).isoformat(), end=max(ws_).isoformat(),
                            phase=Counter(ph_arr[m].tolist()).most_common(1)[0][0], row_by_coat=rnd(row),
                            argmax_coat=am_, consistent_with_pi_app=(pi[am_] == g),
                            margin_under_pi_app=round(float(row[c_true] - max(row[k] for k in range(N) if k != c_true)), 4),
                            crop_argmax_consistency=round(float(crop_ok[m].mean()) * 100, 1)))

    # ---- hourly blocks
    hrs = np.array([w.hour for w in walls]); hourly = []; hp = Counter()
    for h in sorted(set(hrs.tolist())):
        m = hrs == h
        Sh, nh = smat(M5, g_arr, m)
        ent = dict(hour=int(h), n_crops_per_global=nh.tolist())
        if (nh >= HOUR_MIN_CROPS).all():
            ph_ = hung(Sh); q_ = margins(Sh, ph_)[0]
            ent.update(hungarian_pi=list(ph_), equals_pi_app=(ph_ == pi), min_query_margin=round(min(q_), 4),
                       query_margins_under_pi_app=rnd(margins(Sh, pi)[0]))
            hp[ph_] += 1
        else: ent["skipped"] = f"an identity has < {HOUR_MIN_CROPS} gated crops"
        hourly.append(ent)
    n_hours = sum(hp.values())
    modal, modal_n = hp.most_common(1)[0] if hp else (None, 0)

    # ---- 10-min block bootstrap
    t_first = min(walls)
    blk = np.array([int((w - t_first.replace(minute=(t_first.minute // BLOCK_MIN) * BLOCK_MIN, second=0, microsecond=0)).total_seconds() // (BLOCK_MIN * 60)) for w in walls])
    ub = np.unique(blk); bix = {b: j for j, b in enumerate(ub)}
    sums = np.zeros((len(ub), N, N)); cnts = np.zeros((len(ub), N))
    for j in range(len(qi)):
        b = bix[blk[j]]; sums[b, g_arr[j]] += M5[j]; cnts[b, g_arr[j]] += 1
    brng = np.random.default_rng(SEED); bp_ = Counter(); undefined = 0; boot_q = []
    for _ in range(a.n_boot):
        s = brng.integers(0, len(ub), len(ub))
        ss = sums[s].sum(axis=0); cc = cnts[s].sum(axis=0)
        if (cc == 0).any(): undefined += 1; continue
        Sb = (ss / cc[:, None]).T                                      # [c, g]
        pb = hung(Sb); bp_[pb] += 1; boot_q.append(margins(Sb, pi)[0])
    boot_support = bp_[pi] / a.n_boot
    boot_q = np.array(boot_q) if boot_q else np.zeros((0, N))

    # ---- 3-min window consistency (LODO-preview comparable; "consistency with pi_app", NOT accuracy)
    groups = defaultdict(list)
    for j, w in enumerate(walls):
        t = w.replace(second=0, microsecond=0); t -= timedelta(minutes=t.minute % WIN_MIN); groups[(t, g_arr[j])].append(j)
    by_t = defaultdict(list)
    for (t, g), jj in groups.items():
        if len(jj) >= T.MIN_Q: by_t[t].append((g, M5[jj].mean(axis=0)))
    wrec = []
    for t, lv in sorted(by_t.items()):
        Mw = np.array([r for _, r in lv])                               # rows = live 04-22 identities, cols = coats
        ri, cj = linear_sum_assignment(-Mw)
        for a_, (g, row) in enumerate(lv):
            c_true = pinv[g]; am_ = int(np.argmax(row))
            wrec.append(dict(ph=phase4(t), ok_arg=(am_ == c_true), ok_h=(int(cj[list(ri).index(a_)]) == c_true),
                             m=float(row[c_true] - max(row[k] for k in range(N) if k != c_true))))
    def wsum(rr):
        mm = np.array([r["m"] for r in rr])
        return dict(n=len(rr), argmax_pct=round(float(np.mean([r["ok_arg"] for r in rr])) * 100, 1),
                    hungarian_pct=round(float(np.mean([r["ok_h"] for r in rr])) * 100, 1),
                    median_margin=round(float(np.median(mm)), 4), p10_margin=round(float(np.percentile(mm, 10)), 4),
                    frac_margin_ge_tau=round(float(np.mean(mm >= T.TAU)) * 100, 1))
    win3 = {"ALL": wsum(wrec)}
    for p in PHASES:
        rr = [r for r in wrec if r["ph"] == p]
        if rr: win3[p] = wsum(rr)

    # ---- record
    T30 = os.path.join(NB, "Tol_30_periodic_identity.py")
    in_sha = {"Tol_51": sha(os.path.abspath(__file__)), "Tol_30": sha(T30),
              f"{DAY_Q}/{BANDS}": sha(bp), f"{DAY_Q}/segment_assignments_reviewed.json": sha(gp)}
    for f in ("segments.json", "events.jsonl", "observations_meta.json"):
        in_sha[f"{DAY_Q}/{f}"] = sha(os.path.join(qdir, f))
    pool_overlap = None
    if pkeys is not None and pinfo["pool_kind"] == "FROZEN":
        tmp_keys_p = os.path.join(a.out_dir, "_temp_pool", "temp_pool_keys.json")
        if os.path.exists(tmp_keys_p):
            tk = set((w, int(r)) for w, r in json.load(open(tmp_keys_p))["keys"])
            pool_overlap = dict(temp=len(tk), frozen=len(pkeys), common=len(tk & pkeys))
    status = "PROVISIONAL_B1_APPEARANCE" + ("_TEMPORARY_POOL" if pinfo["pool_kind"] == "TEMPORARY" else "")
    rec = dict(
        what="B1 sealed appearance check: frozen 04-18/19 anchors vs verified 04-22 global identities (SPEC_anchor_reid §3.2 B1)",
        status=status,
        note=("PROVISIONAL. Appearance opinion only; the specialist's blind coat key (B2) is the authority and anchor arms are "
              "re-scored if it differs. Flag every anchor-arm result provisional until B3."),
        created=datetime.now().isoformat(timespec="seconds"),
        coat_names=names, label_spaces=dict(c="training coat key (04-18/19)", g="week-global = 04-22 local (reviewed GT)"),
        pool=pinfo, matcher=dict(bands="DINOv3-ViT-S crop_bands3_dense", variant="parts3", orientation="max(P3, P3R)",
                                 scale="/3", topk=T.TOPK, per_crop="mean of top-k anchor similarities per coat",
                                 function="Tol_30.anchor_mean5", self_check_max_abs_diff=maxdiff),
        queries=dict(day=DAY_Q, dir=os.path.abspath(qdir), gt="segment_assignments_reviewed.json (UNASSIGNED dropped)",
                     gate="Tol_30.load_inputs production gate", n_crops_total=len(crops), n_used=len(qi), dropped=dict(drop),
                     n_per_global=n.tolist(), n_per_phase={p: int((ph_arr == p).sum()) for p in PHASES},
                     window=[min(walls).isoformat(), max(walls).isoformat()]),
        day_level=dict(matrix_c_by_g=rnd(S), hungarian_pi=list(pi),
                       pi_named={names[c]: f"G{pi[c]}" for c in range(N)},
                       query_margins_by_global=rnd(qm), anchor_margins_by_coat=rnd(am),
                       pair_margins=[dict(coat=c, coat_name=names[c], g=pi[c], query_margin=round(qm[pi[c]], 4),
                                          anchor_margin=round(am[c], 4)) for c in range(N)],
                       argmax_coat_per_global=argmax_col, argmax_global_per_coat=argmax_row,
                       argmax_agrees_with_hungarian=(all(pi[argmax_col[g]] == g for g in range(N)) and
                                                     all(argmax_row[c] == pi[c] for c in range(N))),
                       perm_best=[round(rank[0][0], 5), list(rank[0][1])], perm_second=[round(rank[1][0], 5), list(second)],
                       perm_gap_mean_per_pair=round(gap, 5), perm_second_differs_on_coats=diff_pairs,
                       perm_top5=[[round(s_, 5), list(p_)] for s_, p_ in rank[:5]],
                       hub_anchor_col_spread=round(hub_anchor_spread, 4), query_row_spread=round(query_spread, 4),
                       crop_argmax_consistency_with_pi_app=round(float(crop_ok.mean()) * 100, 2),
                       crop_argmax_consistency_by_global={f"G{g}": round(float(crop_ok[g_arr == g].mean()) * 100, 2) for g in range(N)}),
        per_phase=phases, per_segment=per_seg,
        hourly=dict(min_crops_per_identity=HOUR_MIN_CROPS, blocks=hourly, n_scored=n_hours,
                    modal_pi=(list(modal) if modal else None), modal_support=(round(modal_n / n_hours, 4) if n_hours else None),
                    modal_equals_pi_app=(modal == pi), n_equal_pi_app=int(hp[pi]),
                    all_perms={"".join(map(str, k)): v for k, v in hp.items()}),
        bootstrap=dict(n=a.n_boot, block_min=BLOCK_MIN, n_blocks=int(len(ub)), seed=SEED, support_pi_app=round(boot_support, 4),
                       undefined=undefined, perms={"".join(map(str, k)): v for k, v in bp_.most_common()},
                       query_margin_p05_by_global=rnd(np.percentile(boot_q, 5, axis=0)) if len(boot_q) else None,
                       query_margin_median_by_global=rnd(np.median(boot_q, axis=0)) if len(boot_q) else None),
        window3=dict(note="per (3-min window, 04-22 identity) with >= MIN_Q gated crops; consistency with pi_app, not accuracy",
                     **win3),
        temp_vs_frozen_pool_overlap=pool_overlap,
        b3_reference=dict(accept_rule="human key == pi_app and bootstrap support >= 0.90 -> ACCEPT; equal but < 0.90 -> ACCEPT-WEAK; differ -> STOP",
                          support_meets_0_90=bool(boot_support >= 0.90)),
        leakage=dict(anchors_from=list(TRAIN_DAYS), queries_from=[DAY_Q], not_read=list(FORBIDDEN_DAYS) + ["coat_key/SECRET_*"]),
        input_sha256=in_sha, runtime_s=round(time.time() - t0, 1))
    out_b1 = os.path.join(a.out_dir, "anchor_correspondence_B1.json")
    json.dump(rec, open(out_b1, "w", encoding="utf-8"), indent=1, default=str)
    b1_sha = sha(out_b1)
    if pinfo["pool_kind"] == "TEMPORARY":
        td = os.path.join(a.out_dir, "_temp_pool"); os.makedirs(td, exist_ok=True)
        json.dump(dict(pool_sha256=a_sha, keys=sorted([w, r] for w, r in pkeys)), open(os.path.join(td, "temp_pool_keys.json"), "w"))
        json.dump(rec, open(os.path.join(td, "anchor_correspondence_B1.TEMP.json"), "w", encoding="utf-8"), indent=1, default=str)
    pij = dict(pi={str(c): int(pi[c]) for c in range(N)},
               pi_named={f"{c} {names[c]}": f"G{pi[c]}" for c in range(N)},
               pi_inverse={f"G{g}": f"{pinv[g]} {names[pinv[g]]}" for g in range(N)},
               direction="pi[c] = g : training coat identity c (04-18/19 key) -> week-global identity g (04-22 local)",
               status=status, provisional=True,
               note=("Set by the B1 appearance check (Hungarian over the whole-day 4x4). NOT the final correspondence: the "
                     "specialist's blind coat key replaces it (B3). Anchor-arm results using this pi are PROVISIONAL."),
               anchor_sha256=a_sha, pool_kind=pinfo["pool_kind"],
               query_margins_by_global={f"G{g}": round(qm[g], 4) for g in range(N)},
               anchor_margins_by_coat={f"{c} {names[c]}": round(am[c], 4) for c in range(N)},
               perm_gap_mean_per_pair=round(gap, 5), bootstrap_support=round(boot_support, 4),
               hourly_modal_support=(round(modal_n / n_hours, 4) if n_hours else None), hourly_modal_equals_pi=(modal == pi),
               b1_record=os.path.abspath(out_b1), b1_sha256=b1_sha, created=rec["created"])
    out_pi = os.path.join(a.out_dir, "pi_provisional.json")
    json.dump(pij, open(out_pi, "w", encoding="utf-8"), indent=1)

    # ---- table
    L = []
    L.append(f"# B1 sealed appearance check — 04-18/19 anchors vs verified 04-22 identities ({status})\n")
    L.append(f"Pool: **{pinfo['pool_kind']}** `{a_sha}` ({len(ident)} anchors, {nper[0]} per coat). "
             f"Queries: {len(qi)} gated 04-22 dense crops with reviewed GT ({min(walls):%H:%M}–{max(walls):%H:%M}). "
             f"Matcher: parts-3, orientation-max, top-{T.TOPK} (Tol_30.anchor_mean5).\n")
    L.append("S[c, g] = mean over the crops of 04-22 identity g of the per-crop mean top-5 similarity to the anchors of coat c. "
             "**Bold** = Hungarian π_app.\n")
    L.append("| anchor coat c \\ 04-22 global g | " + " | ".join(f"G{g} (n={n[g]})" for g in range(N)) + " | anchor-side margin |")
    L.append("|---|" + "---|" * (N + 1))
    for c in range(N):
        cells = [(f"**{S[c, g]:.4f}**" if pi[c] == g else f"{S[c, g]:.4f}") for g in range(N)]
        L.append(f"| {c} {names[c]} | " + " | ".join(cells) + f" | {am[c]:+.4f} |")
    L.append("| query-side margin (own coat vs best other coat) | " + " | ".join(f"{qm[g]:+.4f}" for g in range(N)) + " | |\n")
    L.append("**π_app (provisional):** " + pistr(pi, names) + "\n")
    L.append(f"- best-vs-second permutation gap: {gap:+.5f} (mean per pair); second-best = {list(second)} (differs on coats {diff_pairs})")
    L.append(f"- 10-min-block bootstrap ({a.n_boot}×, {len(ub)} blocks): π_app returned in **{100*boot_support:.1f} %** "
             f"(B3 threshold 90 %: {'met' if boot_support >= 0.90 else 'NOT met'}); undefined resamples {undefined}")
    L.append(f"- hourly-block Hungarian ({n_hours} hours scored, ≥{HOUR_MIN_CROPS} crops/identity): modal {list(modal) if modal else None} "
             f"support {modal_n}/{n_hours} = {100*modal_n/max(n_hours,1):.0f} % (modal == π_app: {modal == pi})")
    L.append(f"- per-crop argmax consistent with π_app: {100*crop_ok.mean():.1f} %; 3-min windows: argmax {win3['ALL']['argmax_pct']} % / "
             f"Hungarian {win3['ALL']['hungarian_pct']} %, median margin {win3['ALL']['median_margin']:+.4f}, ≥TAU {win3['ALL']['frac_margin_ge_tau']} %")
    L.append(f"- hub check: anchor-column mean spread {hub_anchor_spread:.4f}, query-row spread {query_spread:.4f}\n")
    L.append("## Per phase (phase4)\n")
    L.append("| phase | n per G0..G3 | Hungarian π | == π_app | query margins under π_app (G0..G3) | perm gap | crop argmax % |")
    L.append("|---|---|---|---|---|---|---|")
    for p, e in phases.items():
        if "hungarian_pi" in e:
            L.append(f"| {p} | {e['n_crops_per_global']} | {e['hungarian_pi']} | {e['equals_pi_app']} | "
                     f"{', '.join(f'{x:+.4f}' for x in e['query_margins_under_pi_app'])} | {e['perm_gap']:+.5f} | {e['crop_argmax_consistency_with_pi_app']} |")
        else:
            L.append(f"| {p} | {e['n_crops_per_global']} | (an identity has no crops) | | | | |")
    L.append("\n## Hourly blocks\n")
    L.append("| hour | n per G0..G3 | Hungarian π | == π_app | min query margin (own π) |")
    L.append("|---|---|---|---|---|")
    for e in hourly:
        if "hungarian_pi" in e:
            L.append(f"| {e['hour']:02d} | {e['n_crops_per_global']} | {e['hungarian_pi']} | {e['equals_pi_app']} | {e['min_query_margin']:+.4f} |")
        else:
            L.append(f"| {e['hour']:02d} | {e['n_crops_per_global']} | skipped | | |")
    L.append("\n## Per GT segment (diagnostic)\n")
    L.append("| segment | G | n | phase | start–end | best coat | consistent with π_app | margin under π_app | crop argmax % |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for e in per_seg:
        L.append(f"| {e['seg']} | G{e['g']} | {e['n']} | {e['phase']} | {e['start'][11:16]}–{e['end'][11:16]} | {e['argmax_coat']} {names[e['argmax_coat']]} | "
                 f"{e['consistent_with_pi_app']} | {e['margin_under_pi_app']:+.4f} | {e['crop_argmax_consistency']} |")
    L.append(f"\nπ is written as `pi[c] = g` (training coat c → 04-22 global g). B1 record sha256 `{b1_sha}`.")
    out_md = os.path.join(a.out_dir, "pi_provisional_table.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    seal = os.path.join(a.out_dir, "anchor_correspondence_B1.sha256")
    shas = {os.path.basename(p): sha(p) for p in (out_b1, out_pi, out_md)}
    open(seal, "w", encoding="utf-8").write("".join(f"{v}  {k}\n" for k, v in shas.items()))
    print("\n".join(L))
    print(json.dumps(dict(pi=pij["pi"], status=status, anchor_sha256=a_sha, sha256=shas), indent=1))
    print(f"[{time.time()-t0:6.1f}s] DONE Tol_51", flush=True)

if __name__ == "__main__":
    main()
