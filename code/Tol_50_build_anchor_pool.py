# -*- coding: utf-8 -*-
"""Tol · N50 — build the FROZEN 04-18/19 anchor pool for anchor-pool re-ID.

Design (amended 2026-09-26): anchors from 04-18/19 ONLY, frozen.
Local CPU only; no Databricks job, no Volume write. Reads the Tracking_train windows of 04-18/19 (dense 15-s crop bands,
segments / events / observations meta), the specialist-verified RENDERED relabel files and fault list, and the
training-derived area band. From 04-22+ it reads ONLY the npz HEADERS (shape / dtype, never values) of the test-week dense
band files, for the structural half of precondition P1.

Gate chain (every dense crop, all must hold, applied in this order):
  G-a  production crop gate = Tol_30.load_inputs 'gated': area >= 20,000 px, neighbour IoS < 0.10, not over-extended
       (> 1.5x the slot's rolling 2-h median), not within +-5 s of merge_resolved, not after a strike start
  G-b  coloured: a rule of the RENDERED relabel file covers (slot, wall) — Tol_17 ident_at semantics (first matching
       [a, b) rule, open bounds allowed, the file's own 'exclusions' render grey)
  G-c  outside every specialist fault exclusion [exclude_start, exclude_end) of that identity (merge: the listed animals;
       merge_split / not_usable: all four, enforced even if the list is shorter)
  G-d  strict: outside every unlisted mask-evidence item widened +-90 s, for its identities AND its absorber
       (no_masks_at_all has no crops and is skipped)
  G-e  frag_lo(phase) * A_ref(day) <= area <= over_hi * A_ref(day); frag_lo[dayl] for phase4 'day', frag_lo[dark] for
       dawn / dusk / night; the constants are READ from tune_scout/train_summary_segv1d.json and asserted to be derived
       from 04-18/19 only (derived_from subset of the training days, not_from covering 04-22..28)
Balance: per phase4 (night >= 21 or < 6, dawn 6-7, dusk 20, day 8-19) every identity is subsampled to that
phase's minimum count across identities: numpy default_rng(seed), without replacement, candidates in canonical
(wall, window, band_row) order, draws in the fixed order phase day, dawn, dusk, night x identity 0..3. Deterministic.

Outputs (<out_dir>/anchor_pool_<tag>.npz and anchor_pool_<tag>.manifest.json — the manifest name that
Tol_30.load_anchor_pool reads and checks the pool's sha256 against):
  emb [n, 8, 384] fp16, copied BIT-EXACT from the band files (Tol_30 re-normalises it exactly as load_inputs does);
  ident_train int8 (training coat key 0 Black / 1 White / 2 half&half / 3 two dots); phase4, day, window, seg_id, slot,
  wall (ISO), area, band_row (row in the source band file), src_path (crop jpg on the Volume), tracker_backbone.
  Rows are sorted by (ident_train, wall, window, band_row). All arrays are non-object dtypes (allow_pickle=False loads).
The npz is written byte-deterministically (stored, fixed zip timestamps, fixed member order), so a rebuild reproduces the
sha256. FROZEN: an existing output whose sha256 differs from a rebuild is never overwritten (bump --tag instead).

Optional in the same run: Stage 0 S0a (training days only) — leave-one-day-out anchor transfer
04-18 <-> 04-19 on the FINAL gate chain + balance, frozen matcher via Tol_30.anchor_mean5; written to the manifest.
Always after writing: CONSUMER check — Tol_30.load_anchor_pool accepts the file (manifest sha), its parts-3 rows equal the
pool's own, and Tol_30.anchor_mean5 equals Tol_30.pair_sim for 200 (query crop, identity) pairs within 1e-6.

Usage:
  python Tol_50_build_anchor_pool.py --selftest
  python Tol_50_build_anchor_pool.py --days 20240418,20240419 --s0a [--sizing .../anchor_sizing.json]
"""
import os, sys, re, json, glob, time, argparse, hashlib, zipfile, shutil, ast, importlib.util, copy
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
SP6 = r"<LOCAL_RUN_DIR>"
SPA = r"<LOCAL_RUN_DIR>/ablation"
DEFAULTS = dict(
    root=SP6 + "/train_ident/root",
    relabel_dir=SP6 + "/train_gt/relabel_volume",
    gt=SP6 + "/train_gt/train_identity_gt.json",
    constants=SP6 + "/tune_scout/train_summary_segv1d.json",
    t30=HERE + "/Tol_30_periodic_identity.py",
    t30_ref=SP6 + "/train_gt/anchor_ablation/Tol_30_periodic_identity.BACKUP_329ccf51.py",
    out_dir=SP6 + "/train_gt/anchor_pool",
    p1_ref_root=SPA,
    bands="crop_bands3_dense.npz",
)

N = 4
TRAIN_DAYS = ("20240418", "20240419")
TEST_DAYS = tuple(f"202404{d}" for d in range(22, 29))
PHASES = ("day", "dawn", "dusk", "night")
COAT = {0: "Black", 1: "White", 2: "half&half", 3: "two dots on the back"}
STUDENT_WINDOWS = ("20240418_w2123-2206",)   # Tol_03_track_v4 run 435523887848301, backbone=student
STRICT_PAD_S = 90.0
ALL4_KINDS = ("merge_split", "not_usable")
STAGES = ("all", "gated", "labelled", "spec_excl", "strict_excl", "band", "band_reject")   # = anchor_sizing.json stage names
WIN_RX = re.compile(r"^(2024041[89])_w\d{4}-\d{4}$")
PATH_RX = re.compile(r"/Tracking_train/(2024041[89]_w\d{4}-\d{4})/crops_dense/([^/]+)/(\d{6})\.jpg$")
EMB_SHAPE = (8, 384)


# ----------------------------------------------------------------------------- small helpers
def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()

def sha256_text(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()

def phase4(w):
    """Tol_44/Tol_42 phase4 on the wall hour (= train_gt/gtlib.phase4)."""
    h = w.hour
    if h >= 21 or h < 6: return "night"
    if h < 8: return "dawn"
    if h == 20: return "dusk"
    return "day"

def kp(k, p): return f"{k}|{p}"

def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def fn_source(path, name):
    src = open(path, encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name: return ast.get_source_segment(src, node)
    return None

def npz_headers(path):
    """{array name: {shape, dtype}} from the .npy headers only — no array values are read."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for nm in zf.namelist():
            with zf.open(nm) as f:
                v = np.lib.format.read_magic(f)
                rd = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _fo, dt = rd(f)
            out[nm[:-4] if nm.endswith(".npy") else nm] = dict(shape=[int(x) for x in shape], dtype=dt.str)
    return out

def save_npz_deterministic(path, arrays):
    """np.savez-compatible archive, byte-reproducible: STORED members in the given order, fixed 1980-01-01 timestamps,
    fixed attributes. Written to a temp file and moved into place (never partially visible)."""
    tmp = path + f".part{os.getpid()}"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, arr in arrays:
            arr = np.ascontiguousarray(arr)
            assert arr.dtype != object, f"{name}: object arrays are not allowed (allow_pickle=False consumers)"
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED; zi.create_system = 0; zi.external_attr = 0
            with zf.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, arr, allow_pickle=False)
    os.replace(tmp, path)

def write_json(path, obj):
    tmp = path + f".part{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(obj, f, indent=1, sort_keys=False, default=str)
    os.replace(tmp, path)

def norm_rows(E16):
    """exactly Tol_30.load_inputs: float32, per-band L2 normalisation with +1e-8"""
    E = E16.astype(np.float32); E /= (np.linalg.norm(E, axis=2, keepdims=True) + 1e-8)
    return E


# ----------------------------------------------------------------------------- label sources
def load_relabel(path):
    rel = json.load(open(path, encoding="utf-8"))
    P = lambda s: datetime.fromisoformat(s) if s else None
    R = {int(s): [(P(a), P(b), int(k)) for a, b, k in v] for s, v in rel["rules"].items()}
    EX = [(P(e.get("from")), P(e.get("to")), set(int(x) for x in e["slots"])) for e in rel.get("exclusions", [])]
    return R, EX

def ident_at(R, EX, slot, w):
    """Tol_17 ident_at (= train_gt/gtlib.ident_array): the relabel file's exclusions render grey; else the FIRST rule
    [a, b) covering w (open bounds allowed); no rule = grey '?' = None."""
    for a, b, ss in EX:
        if slot in ss and (a is None or a <= w) and (b is None or w < b): return None
    for a, b, k in R.get(slot, []):
        if (a is None or a <= w) and (b is None or w < b): return k
    return None

def gt_exclusions(GT):
    """G-c (specialist faults) and G-d (strict: unlisted mask evidence +-90 s, identities + absorber) per identity."""
    P = datetime.fromisoformat
    spec, strict = defaultdict(list), defaultdict(list)
    used = dict(faults=[], unlisted=[], unlisted_skipped=[])
    for f in GT["faults"]:
        ids = set(int(k) for k in f["identities"])
        if f["kind"] in ALL4_KINDS: ids |= set(range(N))
        a, b = P(f["exclude_start"]), P(f["exclude_end"])
        if not a < b: raise SystemExit(f"fault {f['id']}: empty exclusion {a} .. {b}")
        for k in sorted(ids): spec[k].append((a, b))
        used["faults"].append(dict(id=f["id"], kind=f["kind"], identities=sorted(ids), exclude=[a.isoformat(), b.isoformat()]))
    for j, it in enumerate(GT["unlisted_mask_evidence"]["items"]):
        if it["type"] == "no_masks_at_all":
            used["unlisted_skipped"].append(dict(item=j, type=it["type"], wall=[it["wall_start"], it["wall_end"]])); continue
        ids = set(int(k) for k in (it.get("identities") or []))
        ab = it.get("absorber")
        if isinstance(ab, int) and not isinstance(ab, bool): ids.add(ab)
        a = P(it["wall_start"]) - timedelta(seconds=STRICT_PAD_S); b = P(it["wall_end"]) + timedelta(seconds=STRICT_PAD_S)
        for k in sorted(ids): strict[k].append((a, b))
        used["unlisted"].append(dict(item=j, type=it["type"], identities=sorted(ids), exclude=[a.isoformat(), b.isoformat()]))
    return spec, strict, used

def load_band_constants(path, days):
    """G-e constants READ from the training summary; each must be derived from 04-18/19 only and declare not_from 04-22..28."""
    C = json.load(open(path, encoding="utf-8"))["constants"]
    prov = {}
    def get(key):
        if key not in C: raise SystemExit(f"constant {key} missing from {path}")
        e = C[key]; df = list(e.get("derived_from") or []); nf = set(e.get("not_from") or [])
        if not df or any(d not in TRAIN_DAYS for d in df):
            raise SystemExit(f"LEAKAGE GUARD: constant {key} derived_from {df} is not a subset of the training days {TRAIN_DAYS}")
        if not set(TEST_DAYS) <= nf:
            raise SystemExit(f"LEAKAGE GUARD: constant {key} does not declare not_from all test days (missing {sorted(set(TEST_DAYS) - nf)})")
        prov[key] = dict(value=e["value"], provenance=e.get("provenance"), derived_from=df, not_from=sorted(nf))
        return float(e["value"])
    out = dict(A_ref={d: get(f"A_ref[{d}]") for d in days}, frag_lo={"dayl": get("frag_lo[dayl]"), "dark": get("frag_lo[dark]")},
               over_hi=get("over_hi"))
    return out, prov


# ----------------------------------------------------------------------------- preconditions
def check_t30(T, t30_path, t30_ref):
    """The gate is Tol_30.load_inputs; record its sha and assert it is the function that labelled the training days."""
    info = dict(path=os.path.abspath(t30_path), sha256=sha256_file(t30_path))
    src = fn_source(t30_path, "load_inputs"); info["load_inputs_sha256"] = sha256_text(src)
    info["gate_constants"] = dict(GATE_AREA=T.GATE_AREA, GATE_NB=T.GATE_NB, AREA_OVER=T.AREA_OVER, MERGE_GUARD=T.MERGE_GUARD,
                                  OBS_MATCH_SEC=T.OBS_MATCH_SEC, TOPK=T.TOPK, P3=list(T.P3), P3R=list(T.P3R))
    if t30_ref and os.path.exists(t30_ref):
        R = import_module(t30_ref, "_tol30_ref_for_tol50")
        rsrc = fn_source(t30_ref, "load_inputs")
        info["ref"] = dict(path=os.path.abspath(t30_ref), sha256=sha256_file(t30_ref), load_inputs_sha256=sha256_text(rsrc))
        same_c = all(getattr(R, k) == getattr(T, k) for k in ("GATE_AREA", "GATE_NB", "AREA_OVER", "MERGE_GUARD", "OBS_MATCH_SEC", "CHAOS_TYPES"))
        info["load_inputs_identical_to_ref"] = (src == rsrc) and same_c
        if not info["load_inputs_identical_to_ref"]:
            raise SystemExit("Tol_30.load_inputs (or its gate constants) differs from the version that labelled the training days — STOP")
    else:
        info["ref"] = None; info["load_inputs_identical_to_ref"] = None
    return info

def check_p1(band_files, ref_root):
    """P1 structural half: band shape [8, 384], fp16, same npz keys, same crop-path layout, for the training band files
    and (headers only) the test-week dense band files. The code-level half (Tol_32 canonical_crop / rle_decode / m_bbox
    and the Tol_28 band layout diffed between the workspace code each job ran) needs the workspace exports and is NOT
    done here; the local sources' sha256 / mtime are recorded as evidence."""
    out = dict(train={}, test_week={}, code_level="NOT PERFORMED here (needs the workspace exports of the code each job ran); local source evidence below")
    keys0 = None
    for win, p in sorted(band_files.items()):
        h = npz_headers(p)
        ok = (h.get("emb", {}).get("dtype") == "<f2" and tuple(h["emb"]["shape"][1:]) == EMB_SHAPE and "paths" in h
              and h["paths"]["dtype"].startswith("<U") and h["paths"]["shape"][0] == h["emb"]["shape"][0])
        if not ok: raise SystemExit(f"P1 STOP: training band file {p} has an unexpected layout {h}")
        keys0 = keys0 or sorted(h)
        if sorted(h) != keys0: raise SystemExit(f"P1 STOP: training band keys differ {sorted(h)} vs {keys0}")
        out["train"][win] = h
    if ref_root:
        for d in TEST_DAYS:
            p = os.path.join(ref_root, d, DEFAULTS["bands"])
            if not os.path.exists(p): out["test_week"][d] = "missing"; continue
            h = npz_headers(p)
            ok = (sorted(h) == keys0 and h["emb"]["dtype"] == "<f2" and tuple(h["emb"]["shape"][1:]) == EMB_SHAPE
                  and h["paths"]["dtype"].startswith("<U"))
            if not ok: raise SystemExit(f"P1 STOP: test-week band file {p} differs in layout: {h}")
            out["test_week"][d] = dict(emb_dtype=h["emb"]["dtype"], emb_shape_tail=h["emb"]["shape"][1:], keys=sorted(h))
    ev = {}
    for f in ("Tol_32_dense_crops.py", "Tol_28_build_bands.py"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            ev[f] = dict(sha256=sha256_file(p), mtime=datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"))
    out["local_source_evidence"] = ev
    out["structural_pass"] = True
    return out


# ----------------------------------------------------------------------------- build
def balance(cands, seed):
    """indices into cands: per phase, every identity subsampled (seed, no replacement) to the phase minimum."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for j, c in enumerate(cands): by[(c["lab"], c["ph"])].append(j)
    quota = {p: min(len(by[(k, p)]) for k in range(N)) for p in PHASES}
    keep = []
    for p in PHASES:
        for k in range(N):
            idx = sorted(by[(k, p)], key=lambda j: cands[j]["key"])
            n = quota[p]
            if n == 0: continue
            pick = rng.choice(len(idx), n, replace=False)
            keep.extend(idx[t] for t in sorted(int(x) for x in pick))
    return keep, quota

def build(cfg, T):
    t0 = time.time()
    days = list(cfg["days"])
    # ---- P2: training days only, explicit, asserted
    if not days: raise SystemExit("--days is required (F14; no default)")
    for d in days:
        if d not in TRAIN_DAYS: raise SystemExit(f"LEAKAGE GUARD (P2): --days may only name the training days {TRAIN_DAYS}, got {d}")
    root = cfg["root"]
    all_dirs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
    wins = [w for w in all_dirs if w[:8] in days]
    if not wins: raise SystemExit(f"no windows for {days} under {root}")
    for w in wins:
        if not WIN_RX.match(w): raise SystemExit(f"P2: unexpected window dir name {w}")
    GT = json.load(open(cfg["gt"], encoding="utf-8"))
    # ---- P3: relabel files = the ones the specialist watched
    rs = GT["relabel_source"]["sha256"]
    exp = sorted(f for f in rs if f.startswith("relabel_week_") and f[len("relabel_week_"):][:8] in days)
    got = sorted(f"relabel_week_{w}.json" for w in wins)
    if exp != got: raise SystemExit(f"P3: window set {got} != relabel_source set {exp}")
    relabel_sha = {}
    for f in got:
        s = sha256_file(os.path.join(cfg["relabel_dir"], f))
        if s != rs[f]: raise SystemExit(f"P3 STOP: {f} sha256 {s} != train_identity_gt.relabel_source {rs[f]}")
        relabel_sha[f] = s
    spec_ex, strict_ex, ex_used = gt_exclusions(GT)
    band_c, band_prov = load_band_constants(cfg["constants"], days)
    cnt = {s: Counter() for s in STAGES}; drops = {g: Counter() for g in ("G-a", "G-b", "G-c", "G-d", "G-e")}
    fates, cands, E16c = [], [], []
    inputs = {}; band_files = {}; unattached = {}; max_norm_dev = 0.0
    for win in wins:
        d = os.path.join(root, win); day = win[:8]
        bp = os.path.join(d, cfg["bands"]); band_files[win] = bp
        with np.load(bp, allow_pickle=False) as z:          # closed at once (Windows file locks)
            paths = [str(p).replace("\\", "/") for p in z["paths"]]; E16 = np.array(z["emb"])
        if E16.dtype != np.float16 or E16.shape[1:] != EMB_SHAPE or len(E16) != len(paths):
            raise SystemExit(f"P1 STOP: {bp} emb {E16.dtype} {E16.shape} / {len(paths)} paths")
        for p in paths:                                      # P2 on EVERY band path
            m = PATH_RX.search(p)
            if not m or m.group(1) != win: raise SystemExit(f"P2 STOP: band path not under /Tracking_train/{win}/crops_dense/: {p}")
        segs, crops, CE, _ = T.load_inputs(d, "crops", cfg["bands"])
        if CE.shape != E16.shape: raise SystemExit(f"{win}: load_inputs CE {CE.shape} vs band {E16.shape}")
        if not np.array_equal(norm_rows(E16), CE): raise SystemExit(f"{win}: normalised fp16 rows != load_inputs CE (row alignment)")
        nd = float(np.abs(np.linalg.norm(E16.astype(np.float32), axis=2) - 1.0).max()); max_norm_dev = max(max_norm_dev, nd)
        if nd > 5e-3: raise SystemExit(f"P1 STOP: {win} band rows are not L2-normalised (max |norm-1| {nd:.4g})")
        unattached[win] = len(paths) - len(crops)
        inputs[win] = dict(band=dict(path=os.path.abspath(bp), sha256=sha256_file(bp), n_rows=len(paths)),
                           **{f: sha256_file(os.path.join(d, f)) for f in ("segments.json", "events.jsonl", "observations_meta.json")
                              if os.path.exists(os.path.join(d, f))},
                           relabel=relabel_sha[f"relabel_week_{win}.json"])
        R, EX = load_relabel(os.path.join(cfg["relabel_dir"], f"relabel_week_{win}.json"))
        backbone = "student" if win in cfg.get("student_windows", STUDENT_WINDOWS) else "teacher"
        for c in crops:
            w = c["w"]; ph = phase4(w); i = int(c["i"])
            lab = ident_at(R, EX, int(c["slot"]), w)
            cnt["all"][kp(None, ph)] += 1
            base = (win, i)
            if not c["gated"]:
                drops["G-a"][kp(lab, ph)] += 1; fates.append((base, "G-a", lab)); continue
            cnt["gated"][kp(lab, ph)] += 1
            if lab is None:
                drops["G-b"][kp(lab, ph)] += 1; fates.append((base, "G-b", lab)); continue
            cnt["labelled"][kp(lab, ph)] += 1
            if any(a <= w < b for a, b in spec_ex[lab]):
                drops["G-c"][kp(lab, ph)] += 1; fates.append((base, "G-c", lab)); continue
            cnt["spec_excl"][kp(lab, ph)] += 1
            if any(a <= w < b for a, b in strict_ex[lab]):
                drops["G-d"][kp(lab, ph)] += 1; fates.append((base, "G-d", lab)); continue
            cnt["strict_excl"][kp(lab, ph)] += 1
            fl = band_c["frag_lo"]["dayl"] if ph == "day" else band_c["frag_lo"]["dark"]
            aref = band_c["A_ref"][day]; ar = c["area"] or 0.0
            if not (fl * aref <= ar <= band_c["over_hi"] * aref):
                cnt["band_reject"][kp(lab, ph)] += 1; drops["G-e"][kp(lab, ph)] += 1; fates.append((base, "G-e", lab)); continue
            cnt["band"][kp(lab, ph)] += 1; fates.append((base, "pass", lab))
            cands.append(dict(win=win, day=day, row=i, sid=c["sid"], slot=int(c["slot"]), w=w, wall=w.isoformat(), ph=ph, lab=int(lab),
                              area=float(ar), path=paths[i], backbone=backbone, key=(w.isoformat(), win, i)))
            E16c.append(E16[i])
        print(f"  {win}: {len(paths)} rows, {len(crops)} crops, running pass {len(cands)}  ({time.time()-t0:.1f} s)", flush=True)
    E16c = np.stack(E16c) if E16c else np.zeros((0,) + EMB_SHAPE, np.float16)
    keep, quota = balance(cands, cfg["seed"])
    keep.sort(key=lambda j: (cands[j]["lab"], cands[j]["key"]))
    P = [cands[j] for j in keep]
    arrays = [
        ("emb", E16c[keep] if keep else np.zeros((0,) + EMB_SHAPE, np.float16)),
        ("ident_train", np.array([c["lab"] for c in P], dtype=np.int8)),
        ("phase4", np.array([c["ph"] for c in P], dtype="<U5")),
        ("day", np.array([c["day"] for c in P], dtype="<U8")),
        ("window", np.array([c["win"] for c in P], dtype="<U19")),
        ("seg_id", np.array([c["sid"] for c in P], dtype=str)),
        ("slot", np.array([c["slot"] for c in P], dtype=np.int8)),
        ("wall", np.array([c["wall"] for c in P], dtype=str)),
        ("area", np.array([c["area"] for c in P], dtype=np.float64)),
        ("band_row", np.array([c["row"] for c in P], dtype=np.int32)),
        ("src_path", np.array([c["path"] for c in P], dtype=str)),
        ("tracker_backbone", np.array([c["backbone"] for c in P], dtype="<U7")),
    ]
    # diagnostics (no effect on the pool): same (day, wall, identity) present twice (overlapping window extents)
    dup_c = sum(v - 1 for v in Counter((c["wall"], c["lab"]) for c in cands).values() if v > 1)
    dup_p = sum(v - 1 for v in Counter((c["wall"], c["lab"]) for c in P).values() if v > 1)
    pool_counts = Counter(kp(c["lab"], c["ph"]) for c in P)
    by_day = Counter(f"{c['day']}|{c['lab']}|{c['ph']}" for c in P)
    by_bb = Counter(f"{c['backbone']}|{c['lab']}" for c in P)
    res = dict(days=days, windows=wins, skipped_dirs=[w for w in all_dirs if w not in wins], inputs=inputs, band_files=band_files,
               relabel_sha=relabel_sha, GT_sha=sha256_file(cfg["gt"]), constants_sha=sha256_file(cfg["constants"]),
               band_constants=band_c, band_constants_provenance=band_prov, exclusions_used=ex_used,
               funnel={s: dict(sorted(cnt[s].items())) for s in STAGES}, drops={g: dict(sorted(v.items())) for g, v in drops.items()},
               quota=quota, pool_counts=dict(sorted(pool_counts.items())), pool_by_day=dict(sorted(by_day.items())),
               pool_by_backbone=dict(sorted(by_bb.items())), unattached_band_rows=unattached, max_norm_dev=max_norm_dev,
               duplicates=dict(candidates_same_wall_identity=dup_c, pool_same_wall_identity=dup_p),
               arrays=arrays, fates=fates, cands=cands, E16c=E16c, keep=keep, seconds=round(time.time() - t0, 1))
    return res


# ----------------------------------------------------------------------------- Stage 0 S0a (training days only)
def parts3_flat(E16, chunk=4096):
    n = len(E16); Q3 = np.empty((n, 3 * EMB_SHAPE[1]), np.float32); R3 = np.empty_like(Q3)
    for s in range(0, n, chunk):
        E = norm_rows(E16[s:s + chunk])
        Q3[s:s + chunk] = E[:, [1, 2, 3], :].reshape(len(E), -1); R3[s:s + chunk] = E[:, [5, 6, 7], :].reshape(len(E), -1)
    return Q3, R3

def stage0_s0a(res, T, seed):
    """LODO anchor transfer on the FINAL gate chain (S0a; port of anchor_sizing/lodo_anchor_check.py): anchors =
    one day's band-passing crops balanced per identity x phase exactly as the pool; queries = the other day's band-passing
    crops, grouped per (window, slot, 3-min window) with >= 2 crops; frozen matcher via Tol_30.anchor_mean5."""
    assert list(T.P3) == [1, 2, 3] and list(T.P3R) == [5, 6, 7]
    cands = res["cands"]; Q3, R3 = parts3_flat(res["E16c"])
    out = {}
    for qday, aday in (("20240419", "20240418"), ("20240418", "20240419")):
        a_all = [j for j, c in enumerate(cands) if c["day"] == aday]
        qall = np.array([j for j, c in enumerate(cands) if c["day"] == qday], dtype=np.int64)
        if not a_all or not len(qall): out[f"{aday}->{qday}"] = dict(skipped="no crops"); continue
        keep, quota = balance([cands[j] for j in a_all], seed)
        A = np.array([a_all[t] for t in keep], dtype=np.int64); ident = np.array([cands[j]["lab"] for j in A], dtype=np.int64)
        if any((ident == k).sum() < T.TOPK for k in range(N)):
            out[f"{aday}->{qday}"] = dict(skipped=f"an identity has < {T.TOPK} anchors", sizes={k: int((ident == k).sum()) for k in range(N)}); continue
        PQ = T.anchor_mean5(Q3[qall], Q3[A], R3[A], ident)
        pos = {int(i): j for j, i in enumerate(qall)}
        groups = defaultdict(list)
        for j in qall:
            c = cands[j]; t0 = c["w"].replace(second=0, microsecond=0); t0 -= timedelta(minutes=t0.minute % 3)
            groups[(c["win"], t0)].append(int(j))
        rows = []
        for key in sorted(groups):
            by_slot = defaultdict(list)
            for i in groups[key]: by_slot[cands[i]["slot"]].append(i)
            live = [(s, v) for s, v in sorted(by_slot.items()) if len(v) >= 2]
            if not live: continue
            M = np.array([PQ[[pos[i] for i in v]].mean(axis=0) for s, v in live])
            truth = [cands[v[0]]["lab"] for s, v in live]
            ri, cj = linear_sum_assignment(-M)
            hung = {int(r): int(c_) for r, c_ in zip(ri, cj)}
            for a, (s, v) in enumerate(live):
                row = M[a]; g = truth[a]
                marg = row[g] - max(row[k] for k in range(N) if k != g)
                rows.append(dict(ph=cands[v[0]]["ph"], ok_argmax=int(np.argmax(row)) == g, ok_hung=hung.get(a) == g,
                                 marg=float(marg), M=row, g=g))
        by = defaultdict(list)
        for r in rows: by[r["ph"]].append(r); by["ALL"].append(r)
        o = {}
        for p in ["ALL"] + [p for p in PHASES if p in by]:
            rr = by[p]; m = np.array([r["marg"] for r in rr])
            o[p] = dict(n=len(rr), argmax_acc=round(float(np.mean([r["ok_argmax"] for r in rr])) * 100, 1),
                        hung_acc=round(float(np.mean([r["ok_hung"] for r in rr])) * 100, 1),
                        margin_true_median=round(float(np.median(m)), 4), margin_true_p10=round(float(np.percentile(m, 10)), 4),
                        frac_margin_ge_tau=round(float(np.mean(m >= T.TAU)) * 100, 1))
        D = np.zeros((N, N)); n = np.zeros(N)
        for r in rows: D[r["g"]] += r["M"]; n[r["g"]] += 1
        D /= np.maximum(n, 1)[:, None]
        ri, cj = linear_sum_assignment(-D)
        dm = [float(D[g, g] - max(D[g, k] for k in range(N) if k != g)) for g in range(N)]
        o["day_level"] = dict(matrix=np.round(D, 4).tolist(), hungarian=[int(x) for x in cj], hungarian_is_identity=[int(x) for x in cj] == list(range(N)),
                              diag_margins=[round(x, 4) for x in dm], hub_col_spread=round(float(np.ptp(D.mean(0))), 4),
                              row_spread=round(float(np.ptp(D.mean(1))), 4))
        o["anchor_sizes"] = {str(k): int((ident == k).sum()) for k in range(N)}; o["anchor_quota"] = quota
        out[f"{aday}->{qday}"] = o
        print(f"  S0a anchors {aday} -> queries {qday}: ALL {o['ALL']}  day-level hungarian {o['day_level']['hungarian']} "
              f"margins {o['day_level']['diag_margins']}", flush=True)
    return out


# ----------------------------------------------------------------------------- outputs
def compare_sizing(res, sizing_path):
    S = json.load(open(sizing_path, encoding="utf-8"))
    diffs = {}
    for st in STAGES:
        a = {k: v for k, v in S["counts"].get(st, {}).items()}; b = res["funnel"][st]
        dd = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        if dd: diffs[st] = dd
    bal_diff = {p: [S["balanced"].get(p), res["quota"][p]] for p in PHASES if S["balanced"].get(p) != res["quota"][p]}
    if bal_diff: diffs["balanced"] = bal_diff
    return dict(path=os.path.abspath(sizing_path), sha256=sha256_file(sizing_path), consistent=not diffs, diffs=diffs)

def consumer_check(T, out_npz, res, n_q=50, seed=1):
    """Tol_30 accepts the file (manifest sha), its A3/A3R equal the pool's own parts-3 rows, and anchor_mean5 == pair_sim."""
    A3, A3R, ident, a_sha = T.load_anchor_pool(out_npz)
    with np.load(out_npz, allow_pickle=False) as z: zemb = np.array(z["emb"]); zid = np.array(z["ident_train"])
    E = norm_rows(zemb); ok_rows = bool(np.array_equal(A3, np.ascontiguousarray(E[:, T.P3, :].reshape(len(E), -1)))
                                        and np.array_equal(A3R, np.ascontiguousarray(E[:, T.P3R, :].reshape(len(E), -1))))
    ok_rows = ok_rows and bool(np.array_equal(ident, zid.astype(np.int64)))
    ok_emb = bool(np.array_equal(zemb, res["E16c"][res["keep"]])); zemb = None
    in_pool = set(res["keep"])
    pool_q = [j for j in range(len(res["cands"])) if j not in in_pool] or list(range(len(res["cands"])))
    rng = np.random.default_rng(seed); q = np.sort(rng.choice(len(pool_q), min(n_q, len(pool_q)), replace=False))
    qi = [pool_q[t] for t in q]
    CEq = norm_rows(res["E16c"][qi]); CEcat = np.concatenate([CEq, E], axis=0); E = None
    Q3 = np.ascontiguousarray(CEq[:, T.P3, :].reshape(len(CEq), -1))
    m5 = T.anchor_mean5(Q3, A3, A3R, ident)
    maxd = 0.0
    for a in range(len(qi)):
        for c in range(N):
            ref = T.pair_sim(CEcat, [a], len(qi) + np.flatnonzero(ident == c))
            maxd = max(maxd, abs(float(m5[a, c]) - ref))
    return dict(load_anchor_pool_accepts=True, anchor_sha256=a_sha, a3_rows_equal=ok_rows, emb_bit_exact_vs_band_rows=ok_emb,
                cache_eq_pairs=len(qi) * N, cache_eq_max_abs_diff=maxd, cache_eq_pass=maxd <= 1e-6,
                queries="band-passing training crops NOT in the pool, numpy default_rng(1)")

def write_outputs(res, cfg, T, t30_info, p1, s0a=None, sizing=None):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    out_npz = os.path.join(cfg["out_dir"], f"anchor_pool_{cfg['tag']}.npz")
    man_path = out_npz[:-4] + ".manifest.json"
    cand = out_npz[:-4] + ".candidate.npz"
    save_npz_deterministic(cand, res["arrays"])
    new_sha = sha256_file(cand)
    if os.path.exists(out_npz):
        old = sha256_file(out_npz)
        os.remove(cand)
        if old != new_sha:
            raise SystemExit(f"FROZEN: {out_npz} exists with sha256 {old}; this build gives {new_sha}. Never overwritten — use a new --tag.")
        status = "identical_rebuild"
    else:
        os.replace(cand, out_npz); status = "new"
    print(f"  pool {status}: {out_npz} sha256 {new_sha}", flush=True)
    write_manifest = status == "new" or cfg.get("rewrite_manifest") or not os.path.exists(man_path)
    man = None
    if write_manifest:
        n = len(res["keep"])
        man = dict(
            what="FROZEN anchor pool for anchor-pool re-ID: verified 04-18/19 dense crops (SPEC_anchor_reid.md §2)",
            spec=["SPEC_anchor_reid.md §2", "SPEC_anchor_reid_addendum_2026-09-26.md (G5: 04-18/19 only, frozen)"],
            tool=dict(path=os.path.abspath(__file__), sha256=sha256_file(os.path.abspath(__file__))),
            built_at=datetime.now().isoformat(timespec="seconds"),
            derived_from=list(TRAIN_DAYS), not_from=list(TEST_DAYS),
            output=dict(path=os.path.abspath(out_npz), sha256=new_sha, bytes=os.path.getsize(out_npz), n=n,
                        arrays={nm: dict(dtype=np.asarray(a).dtype.str, shape=list(np.asarray(a).shape)) for nm, a in res["arrays"]},
                        row_order="(ident_train, wall, window, band_row)", npz_writer="deterministic (stored, 1980-01-01 timestamps)"),
            identity_key=dict(space="training coat key (C); Tol_30 maps it to global ids through pi (Tol_51)", names={str(k): v for k, v in COAT.items()}),
            inputs=dict(windows=res["inputs"], train_identity_gt=dict(path=os.path.abspath(cfg["gt"]), sha256=res["GT_sha"]),
                        relabel_files=res["relabel_sha"], relabel_sha_matches_train_identity_gt=True,
                        constants=dict(path=os.path.abspath(cfg["constants"]), sha256=res["constants_sha"]), tol30=t30_info),
            preconditions={"P1": p1, "P2": {"days": res["days"], "windows": res["windows"], "skipped_dirs": res["skipped_dirs"],
                                            "every_band_path_matches": PATH_RX.pattern, "pass": True},
                           "P3": {"pass": True, "n_files": len(res["relabel_sha"])}},
            gate_chain=dict(
                order=["G-a production crop gate (Tol_30.load_inputs.gated)", "G-b coloured (relabel rule, Tol_17 ident_at)",
                       "G-c specialist fault exclusions [exclude_start, exclude_end) of that identity",
                       f"G-d strict: unlisted mask evidence +-{STRICT_PAD_S:.0f} s, identities + absorber (no_masks_at_all skipped)",
                       "G-e frag_lo(phase)*A_ref(day) <= area <= over_hi*A_ref(day)"],
                interval_semantics="half-open [a, b) for relabel rules and both exclusion lists (= calf_hours / gtlib / size_anchors)",
                G_a_constants=t30_info["gate_constants"],
                G_c_kinds_all_four=list(ALL4_KINDS),
                G_e=dict(A_ref=res["band_constants"]["A_ref"], frag_lo=res["band_constants"]["frag_lo"], over_hi=res["band_constants"]["over_hi"],
                         frag_lo_by_phase={"day": "dayl", "dawn": "dark", "dusk": "dark", "night": "dark"},
                         provenance=res["band_constants_provenance"]),
                phase4="night >= 21 or < 6, dawn 6-7, dusk 20, day 8-19 (wall hour)",
                exclusions_used=res["exclusions_used"]),
            gate_funnel=dict(counts=res["funnel"], drops=res["drops"],
                             totals_by_identity={st: {str(k): sum(v for key, v in res["funnel"][st].items() if key.split("|")[0] == str(k)) for k in list(range(N)) + ["None"]}
                                                 for st in STAGES},
                             unattached_band_rows=res["unattached_band_rows"],
                             note="keys 'identity|phase'; 'None' = no identity rule (grey) — 'all' is keyed None because it precedes labelling"),
            balance=dict(method="per phase4, every identity subsampled to the phase minimum across identities; numpy default_rng(seed), "
                                "without replacement, candidates in (wall, window, band_row) order, draws phase day,dawn,dusk,night x identity 0..3",
                         seed=cfg["seed"], quota_per_identity=res["quota"], per_identity_total=sum(res["quota"].values()),
                         pool_total=n, counts=res["pool_counts"], by_day=res["pool_by_day"]),
            tracker_backbone=dict(student_windows=list(cfg.get("student_windows", STUDENT_WINDOWS)), counts=res["pool_by_backbone"]),
            diagnostics=dict(max_band_norm_dev=res["max_norm_dev"], duplicates=res["duplicates"], build_seconds=res["seconds"]),
            not_done_v1=["temporal thinning", "near-duplicate dedupe", "night up-weighting", "similarity calibration/centring"],
            stage0_s0a=s0a, sizing_consistency=sizing,
        )
        write_json(man_path, man)
    else:
        m0 = json.load(open(man_path, encoding="utf-8"))
        if m0.get("output", {}).get("sha256") != new_sha: raise SystemExit(f"existing manifest {man_path} does not record {new_sha}")
    cc = consumer_check(T, out_npz, res)
    print(f"  consumer check: {cc}", flush=True)
    if not (cc["a3_rows_equal"] and cc["emb_bit_exact_vs_band_rows"] and cc["cache_eq_pass"]):
        raise SystemExit(f"CONSUMER CHECK FAILED: {cc}")
    if write_manifest:
        man["consumer_check"] = cc; write_json(man_path, man)
    return dict(npz=out_npz, npz_sha256=new_sha, manifest=man_path, manifest_sha256=sha256_file(man_path), status=status,
                manifest_written=bool(write_manifest), consumer_check=cc)


# ----------------------------------------------------------------------------- SELFTEST (synthetic)
def _synthetic(base, seed=0):
    """Two synthetic Tracking_train windows with planted gate hits; returns cfg + expected per-crop fate / label."""
    rng = np.random.default_rng(seed)
    root = os.path.join(base, "root"); rdir = os.path.join(base, "relabel"); os.makedirs(root); os.makedirs(rdir)
    D = datetime.fromisoformat
    protos = rng.normal(size=(N,) + EMB_SHAPE).astype(np.float32)
    W1, W2 = "20240418_w1930-2130", "20240419_w0600-0830"
    wins = {W1: dict(start=D("2024-04-18T19:30:00"), end=D("2024-04-18T21:30:00"), aref=80000.0,
                     segs=[(f"{s}_001", s, D("2024-04-18T19:30:00"), D("2024-04-18T21:30:00"), "day_end") for s in range(N)]),
            W2: dict(start=D("2024-04-19T06:00:00"), end=D("2024-04-19T08:30:00"), aref=90000.0,
                     segs=[("0_001", 0, D("2024-04-19T06:00:00"), D("2024-04-19T08:30:00"), "day_end"),
                           ("1_001", 1, D("2024-04-19T06:00:00"), D("2024-04-19T08:30:00"), "day_end"),
                           ("2_001", 2, D("2024-04-19T06:00:00"), D("2024-04-19T08:00:00"), "detector_contradiction"),
                           ("2_002", 2, D("2024-04-19T08:00:05"), D("2024-04-19T08:30:00"), "day_end"),
                           ("3_001", 3, D("2024-04-19T06:00:00"), D("2024-04-19T08:30:00"), "day_end")])}
    T_ = lambda s: D(s)
    # relabel rules (true labels): W1 slot s -> (s+1)%4, slot 3 grey 20:00-20:10; W2 slot s -> s with a 0<->1 swap at 07:00
    rules = {W1: {str(s): [[wins[W1]["start"].isoformat(), (wins[W1]["end"] + timedelta(minutes=1)).isoformat(), (s + 1) % N]] for s in range(3)},
             W2: {"0": [["2024-04-19T06:00:00", "2024-04-19T07:00:00", 0], ["2024-04-19T07:00:00", "2024-04-19T08:31:00", 1]],
                  "1": [["2024-04-19T06:00:00", "2024-04-19T07:00:00", 1], ["2024-04-19T07:00:00", "2024-04-19T08:31:00", 0]],
                  "2": [["2024-04-19T06:00:00", "2024-04-19T08:31:00", 2]], "3": [["2024-04-19T06:00:00", "2024-04-19T08:31:00", 3]]}}
    rules[W1]["3"] = [["2024-04-18T19:30:00", "2024-04-18T20:00:00", 0], ["2024-04-18T20:10:00", "2024-04-18T21:31:00", 0]]
    def truth(win, slot, w):
        if win == W1: return (slot + 1) % N
        if slot in (0, 1) and w >= T_("2024-04-19T07:00:00"): return 1 - slot
        return slot
    within = lambda w, a, b: T_(a) <= w < T_(b)
    expect, rows_by_win = {}, {}
    for win, spec in wins.items():
        paths, embs, obs = [], [], []
        k = 0
        for sid, slot, a, b, _cr in spec["segs"]:
            w = a + timedelta(seconds=15)
            while w < b:
                area = spec["aref"] - (2000.0 if win == W2 else 0.0) + (k % 7) * 100.0; nb = 0.0; fate = "pass"
                g = truth(win, slot, w)
                lab = None if (win == W1 and slot == 3 and within(w, "2024-04-18T20:00:00", "2024-04-18T20:10:00")) else g
                # ---- planted gate hits (disjoint by construction)
                if win == W1 and slot == 0 and within(w, "2024-04-18T19:40:00", "2024-04-18T19:41:00"): area = 15000.0; fate = "G-a"
                elif win == W1 and slot == 1 and within(w, "2024-04-18T19:45:00", "2024-04-18T19:46:00"): nb = 0.2; fate = "G-a"
                elif win == W1 and w == T_("2024-04-18T19:50:00"): fate = "G-a"                           # merge_resolved +-5 s
                elif win == W2 and sid == "2_001" and w >= T_("2024-04-19T07:55:00"): fate = "G-a"       # after the strike start
                elif win == W2 and slot == 3 and w == T_("2024-04-19T07:30:00"): area = 1.7 * 88000.0; fate = "G-a"   # over-extended
                elif lab is None: fate = "G-b"
                elif win == W1 and lab in (0, 1) and within(w, "2024-04-18T19:35:00", "2024-04-18T19:36:00"): fate = "G-c"   # merge [0,1]
                elif win == W2 and within(w, "2024-04-19T08:20:00", "2024-04-19T08:21:00"): fate = "G-c"   # not_usable listed [2] -> all four
                elif win == W2 and lab in (3, 1) and within(w, "2024-04-19T06:28:30", "2024-04-19T06:31:40"): fate = "G-d"  # ids [3] + absorber 1
                elif win == W1 and lab in (0, 2) and within(w, "2024-04-18T21:08:30", "2024-04-18T21:11:35"): fate = "G-d"  # pair overlap [0,2]
                elif win == W1 and slot == 2 and w == T_("2024-04-18T19:55:00"): area = 34800.0; fate = "G-e"   # < 0.44*80000 (day)
                elif win == W1 and slot == 2 and w == T_("2024-04-18T21:20:00"): area = 34800.0; fate = "pass"  # >= 0.43*80000 (night)
                elif win == W2 and slot == 0 and w == T_("2024-04-19T08:10:00"): area = 119700.0; fate = "G-e"  # > 1.32*90000
                elif win == W2 and slot == 0 and w == T_("2024-04-19T08:10:15"): area = 118790.0; fate = "pass"
                expect[(win, len(paths))] = (fate, lab)
                paths.append(f"/Volumes/x/Tol/Tracking_train/{win}/crops_dense/{sid}/{w.strftime('%H%M%S')}.jpg")
                e = protos[g] + 0.9 * rng.normal(size=EMB_SHAPE).astype(np.float32)
                embs.append((e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float16))
                obs.append(dict(seg_id=sid, slot=slot, wall=w.isoformat(), area=area, max_neighbor_ios=nb))
                w += timedelta(seconds=15); k += 1
        d = os.path.join(root, win); os.makedirs(d)
        np.savez(os.path.join(d, DEFAULTS["bands"]), paths=np.array(paths), emb=np.stack(embs))
        json.dump(dict(day=win[:8], segments=[dict(seg_id=s, slot=sl, t_start_wall=a.isoformat(), t_end_wall=b.isoformat(), close_reason=cr)
                                              for s, sl, a, b, cr in spec["segs"]]), open(os.path.join(d, "segments.json"), "w"))
        json.dump(dict(obs=obs), open(os.path.join(d, "observations_meta.json"), "w"))
        with open(os.path.join(d, "events.jsonl"), "w") as f:
            if win == W1: f.write(json.dumps(dict(type="merge_resolved", wall="2024-04-18T19:50:00")) + "\n")
            if win == W2: f.write(json.dumps(dict(type="strike", seg_id="2_001", wall="2024-04-19T07:55:00", n=1)) + "\n")
        json.dump(dict(rules=rules[win], exclusions=[]), open(os.path.join(rdir, f"relabel_week_{win}.json"), "w"))
        rows_by_win[win] = len(paths)
    gt = dict(relabel_source=dict(sha256={f: sha256_file(os.path.join(rdir, f)) for f in sorted(os.listdir(rdir))}),
              faults=[dict(id="F1", kind="merge", identities=[0, 1], exclude_start="2024-04-18T19:35:00", exclude_end="2024-04-18T19:36:00"),
                      dict(id="F2", kind="not_usable", identities=[2], exclude_start="2024-04-19T08:20:00", exclude_end="2024-04-19T08:21:00")],
              unlisted_mask_evidence=dict(items=[
                  dict(type="identity_missing", identities=[3], absorber=1, wall_start="2024-04-19T06:30:00", wall_end="2024-04-19T06:30:10"),
                  dict(type="no_masks_at_all", identities=[0, 1, 2, 3], absorber=None, wall_start="2024-04-19T06:45:00", wall_end="2024-04-19T06:45:10"),
                  dict(type="pair_overlap_ios>=0.5", identities=[0, 2], absorber=None, wall_start="2024-04-18T21:10:00", wall_end="2024-04-18T21:10:05")]))
    json.dump(gt, open(os.path.join(base, "gt.json"), "w"))
    cons = {"A_ref[20240418]": 80000, "A_ref[20240419]": 90000, "frag_lo[dayl]": 0.44, "frag_lo[dark]": 0.43, "over_hi": 1.32}
    json.dump(dict(constants={k: dict(value=v, provenance="synthetic", derived_from=["20240418"] if "0418" in k else (["20240419"] if "0419" in k else list(TRAIN_DAYS)),
                                      not_from=list(TEST_DAYS)) for k, v in cons.items()}), open(os.path.join(base, "constants.json"), "w"))
    cfg = dict(days=list(TRAIN_DAYS), root=root, relabel_dir=rdir, gt=os.path.join(base, "gt.json"), constants=os.path.join(base, "constants.json"),
               bands=DEFAULTS["bands"], out_dir=os.path.join(base, "out"), tag="st", seed=0, student_windows=())
    return cfg, expect

def _expect_raise(fn, what, needle=None):
    try: fn()
    except SystemExit as e:
        if needle and needle not in str(e): raise AssertionError(f"{what}: raised, but not the expected guard: {e}")
        print(f"    ok  {what}: refused ({str(e)[:110]})"); return
    raise AssertionError(f"{what}: did NOT raise")

def selftest(T, t30_path, t30_ref):
    base = os.path.join(SP6, "train_gt", "_tol50_selftest")
    shutil.rmtree(base, ignore_errors=True); os.makedirs(base)
    cfg, expect = _synthetic(base)
    t30_info = check_t30(T, t30_path, t30_ref)
    print(f"  T30 {t30_info['sha256'][:12]}  load_inputs identical to the labelling version: {t30_info['load_inputs_identical_to_ref']}")
    res = build(cfg, T)
    # 1. every crop's fate and label is exactly the planted one
    got = {base_: (f, lab) for base_, f, lab in res["fates"]}
    assert set(got) == set(expect), f"crop sets differ: {len(got)} vs {len(expect)}"
    bad = [(k, expect[k], got[k]) for k in expect if expect[k] != got[k]]
    assert not bad, f"fate mismatches: {bad[:10]}"
    fc = Counter(f for f, _ in expect.values()); print(f"    ok  per-crop fates == planted for all {len(expect)} crops: {dict(sorted(fc.items()))}")
    # 2. balance: counts == phase minimum over identities of the planted passes
    pas = Counter()
    for c in res["cands"]: pas[(c["lab"], c["ph"])] += 1
    quota = {p: min(pas[(k, p)] for k in range(N)) for p in PHASES}
    assert quota == res["quota"], (quota, res["quota"])
    A = dict(res["arrays"]); cnt = Counter(zip(A["ident_train"].tolist(), A["phase4"].tolist()))
    assert all(cnt[(k, p)] == quota[p] for k in range(N) for p in PHASES), (cnt, quota)
    assert sum(cnt.values()) == N * sum(quota.values())
    print(f"    ok  balance: per identity {quota} (planted passes {dict(sorted((f'{k}|{p}', v) for (k, p), v in pas.items()))})")
    # 3. pool rows: only passing crops, correct label, emb bit-exact vs the source band row, metadata consistent
    srcs = {}
    for j in range(len(A["ident_train"])):
        win, row = str(A["window"][j]), int(A["band_row"][j])
        f, lab = expect[(win, row)]
        assert f == "pass" and lab == int(A["ident_train"][j]), (win, row, f, lab)
        if win not in srcs:
            with np.load(os.path.join(cfg["root"], win, cfg["bands"])) as f_: srcs[win] = {k_: f_[k_] for k_ in f_.files}
        src = srcs[win]
        assert np.array_equal(src["emb"][row], A["emb"][j]) and str(src["paths"][row]) == str(A["src_path"][j])
        assert str(A["src_path"][j]).endswith(f"/{A['seg_id'][j]}/{str(A['wall'][j])[11:].replace(':', '')}.jpg")
    print(f"    ok  {len(A['ident_train'])} pool rows: all planted-pass, labels right, emb bit-exact, path/seg/wall consistent")
    # 4. S0a wiring on identity-dependent synthetic embeddings
    s0a = stage0_s0a(res, T, cfg["seed"])
    for k_, v in s0a.items():
        assert v.get("day_level", {}).get("hungarian_is_identity"), f"S0a {k_}: {v}"
    print("    ok  S0a runs; day-level Hungarian = identity both directions (synthetic, identity-dependent embeddings)")
    # 5. write, consumer accepts; deterministic rebuild; frozen; tamper refused
    p1 = check_p1(res["band_files"], None)
    w1 = write_outputs(res, cfg, T, t30_info, p1, s0a=s0a)
    assert w1["status"] == "new" and w1["consumer_check"]["cache_eq_pass"]
    print(f"    ok  written + consumer check (CACHE-EQ max diff {w1['consumer_check']['cache_eq_max_abs_diff']:.2e})")
    res2 = build(cfg, T); w2 = write_outputs(res2, cfg, T, t30_info, p1)
    assert w2["status"] == "identical_rebuild" and w2["npz_sha256"] == w1["npz_sha256"]
    cfg_b = dict(cfg, out_dir=os.path.join(base, "out_b")); w3 = write_outputs(build(cfg_b, T), cfg_b, T, t30_info, p1)
    assert w3["npz_sha256"] == w1["npz_sha256"]
    print(f"    ok  deterministic: rebuild in place and into a fresh dir give sha {w1['npz_sha256'][:16]}")
    cfg_s = dict(cfg, seed=1)
    _expect_raise(lambda: write_outputs(build(cfg_s, T), cfg_s, T, t30_info, p1), "frozen pool vs a different build (seed 1)", "FROZEN")
    man = json.load(open(w1["manifest"], encoding="utf-8")); man0 = copy.deepcopy(man)
    man["output"]["sha256"] = "0" * 64; write_json(w1["manifest"], man)
    _expect_raise(lambda: T.load_anchor_pool(w1["npz"]), "Tol_30.load_anchor_pool with a manifest sha mismatch", "refusing")
    write_json(w1["manifest"], man0)
    # 6. guards: P2 days, P2 paths, P3 relabel sha, constants leakage
    _expect_raise(lambda: build(dict(cfg, days=["20240422"]), T), "P2 --days 20240422", "LEAKAGE GUARD")
    _expect_raise(lambda: build(dict(cfg, days=[]), T), "P2 no --days", "required")
    g2 = json.load(open(cfg["gt"])); f0 = sorted(g2["relabel_source"]["sha256"])[0]; g2["relabel_source"]["sha256"][f0] = "f" * 64
    json.dump(g2, open(os.path.join(base, "gt_bad.json"), "w"))
    _expect_raise(lambda: build(dict(cfg, gt=os.path.join(base, "gt_bad.json")), T), "P3 relabel sha mismatch", "P3 STOP")
    c2 = json.load(open(cfg["constants"])); c2["constants"]["A_ref[20240418]"]["derived_from"] = ["20240422"]
    json.dump(c2, open(os.path.join(base, "cons_bad.json"), "w"))
    _expect_raise(lambda: build(dict(cfg, constants=os.path.join(base, "cons_bad.json")), T), "constant derived from 04-22", "LEAKAGE GUARD")
    root_b = os.path.join(base, "root_badpath"); shutil.copytree(cfg["root"], root_b)
    W1 = "20240418_w1930-2130"; zp = os.path.join(root_b, W1, cfg["bands"])
    with np.load(zp) as zz: pth = zz["paths"].tolist(); emb_b = np.array(zz["emb"])
    pth[3] = pth[3].replace("/Tracking_train/", "/Tracking_v4v2/"); np.savez(zp, paths=np.array(pth), emb=emb_b)
    _expect_raise(lambda: build(dict(cfg, root=root_b), T), "P2 a band path outside /Tracking_train/", "P2 STOP")
    h = npz_headers(os.path.join(cfg["root"], W1, cfg["bands"]))
    assert h["emb"]["dtype"] == "<f2" and h["emb"]["shape"][1:] == [8, 384], h
    print("    ok  npz header reader (P1 structural)")
    import gc; res = res2 = None; gc.collect()
    shutil.rmtree(base, ignore_errors=True)
    if os.path.exists(base): print(f"    note: could not remove {base} (file lock); remove it by hand")
    print("SELFTEST PASSED")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", default=None, help="REQUIRED (no default, F14): 20240418,20240419")
    ap.add_argument("--root", default=DEFAULTS["root"]); ap.add_argument("--relabel_dir", default=DEFAULTS["relabel_dir"])
    ap.add_argument("--gt", default=DEFAULTS["gt"]); ap.add_argument("--constants", default=DEFAULTS["constants"])
    ap.add_argument("--bands", default=DEFAULTS["bands"]); ap.add_argument("--t30", default=DEFAULTS["t30"])
    ap.add_argument("--t30_ref", default=DEFAULTS["t30_ref"], help="the Tol_30 that labelled the training days (load_inputs must match)")
    ap.add_argument("--out_dir", default=DEFAULTS["out_dir"]); ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p1_ref_root", default=DEFAULTS["p1_ref_root"], help="test-week band dir (HEADERS only) for P1; '' to skip")
    ap.add_argument("--s0a", action="store_true", help="run Stage 0 S0a (LODO 04-18 <-> 04-19) and write it to the manifest")
    ap.add_argument("--sizing", default=None, help="anchor_sizing.json to check the funnel / balance counts against")
    ap.add_argument("--rewrite_manifest", action="store_true", help="rewrite the manifest of an IDENTICAL rebuild")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    T = import_module(a.t30, "Tol_30_periodic_identity")
    if a.selftest:
        selftest(T, a.t30, a.t30_ref); print(f"DONE selftest {time.time()-t0:.1f} s", flush=True); return
    if not a.days: ap.error("--days is required (F14; no default): --days 20240418,20240419")
    cfg = dict(days=[d.strip() for d in a.days.split(",") if d.strip()], root=a.root, relabel_dir=a.relabel_dir, gt=a.gt,
               constants=a.constants, bands=a.bands, out_dir=a.out_dir, tag=a.tag, seed=a.seed, rewrite_manifest=a.rewrite_manifest)
    t30_info = check_t30(T, a.t30, a.t30_ref)
    print(f"T30 {t30_info['sha256'][:12]} (load_inputs identical to the labelling version: {t30_info['load_inputs_identical_to_ref']})", flush=True)
    res = build(cfg, T)
    p1 = check_p1(res["band_files"], a.p1_ref_root or None)
    print(f"P1 structural pass; test-week headers: { {d: (v if isinstance(v, str) else v['emb_shape_tail']) for d, v in p1['test_week'].items()} }", flush=True)
    for st in STAGES:
        tot = {k: sum(v for key, v in res["funnel"][st].items() if key.split("|")[0] == str(k)) for k in list(range(N)) + ["None"]}
        print(f"  funnel {st:12s} {tot}")
    print(f"  balance quota per identity {res['quota']} -> {sum(res['quota'].values())} per identity, pool {len(res['keep'])}", flush=True)
    sizing = compare_sizing(res, a.sizing) if a.sizing else None
    if sizing: print(f"  sizing consistency: {sizing['consistent']}  diffs: {json.dumps(sizing['diffs'])[:800]}", flush=True)
    s0a = stage0_s0a(res, T, a.seed) if a.s0a else None
    w = write_outputs(res, cfg, T, t30_info, p1, s0a=s0a, sizing=sizing)
    print(json.dumps(dict(npz=w["npz"], npz_sha256=w["npz_sha256"], manifest=w["manifest"], manifest_sha256=w["manifest_sha256"],
                          status=w["status"], manifest_written=w["manifest_written"]), indent=1))
    print(f"DONE {time.time()-t0:.1f} s", flush=True)

if __name__ == "__main__":
    main()
