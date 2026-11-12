# -*- coding: utf-8 -*-
"""Tol · N54 — COLOUR statistic of every masked canonical crop, and the COLOUR non-calf rule derived from 04-18/19 ONLY.

Why (decision 2026-09-30): "exclude colour except for white/black should be a strong filter". All four calves are black
and white (Holstein-type), so a track whose masked crop pixels are clearly COLOURED is not a calf — e.g. the farm worker in blue
overalls that tracker v5 followed as seg 1_012 (slot 1) on 04-23 15:38:51-15:49:20. The anchor-similarity filter (Tol_53)
failed acceptance on that case (worker median anchor score 0.663, inside the calves' range); colour is a cue the calves do not
share. Night / IR crops are greyscale, so the statistic is naturally inert there.

INPUT: the dense 15-s canonical crops written by Tol_32 (= Tol_03 canonical_crop, verbatim): 224x224 JPEG (q88, 4:2:0), the
masked calf rotated spine-vertical, a JOINT 2-98 % percentile contrast stretch over all channels of the foreground, everything
outside the mask set to the flat fill FILL = (124, 116, 104) (JPEG reproduces it as ~(124, 116, 105) +- 2).

PER-CROP STATISTIC (pure function crop_colour, on FOREGROUND pixels only):
  background  = pixels within BG_TOL (max-abs per channel) of FILL that belong to a connected component (8-conn) touching the
                crop border or of >= HOLE_MIN px (mask holes are filled with FILL too); foreground = the rest.
  edge ring   = the foreground is eroded by EDGE_ERODE px before any colour is measured: the mask boundary is a blend of coat and
                fill (bilinear rotate + bilinear resize) and JPEG 4:2:0 chroma is 2x2-subsampled and rings across strong edges,
                so its colour is not the animal's. Measured on 3,000 training crops (colorfilter/pixdiag.py): the 1-3 px ring
                is a minor contributor (coloured rate 0.1-0.7 % vs ~1 % inside at the old 0.35/0.35 grid), so the erosion is a
                safety margin, not a load-bearing choice.
  HSV         = V = max(R,G,B)/255, S = (max-min)/max, H = standard hexcone hue (degrees).
  coloured    = S >= S_MIN and V >= V_MIN.
  coloured_frac = #coloured / n_fg ; sat_median = median S over the foreground pixels with V >= V_MIN (the pixels whose S is
                meaningful; NaN if none); hue_mode = centre of the most populated 10-degree hue bin of the COLOURED pixels (NaN if
                none); n_fg = foreground pixels after erosion; also bright_frac (= #{V >= V_MIN} / n_fg) and hue_hist
                (12 x 30-degree bins: fraction of the foreground coloured in each hue sector, sums to coloured_frac).
  n_fg < MIN_FG -> every statistic NaN (a crop without enough foreground says nothing).

WHY S_MIN = 0.35, V_MIN = 0.50 (first principles + the training crops only; colorfilter/pixdiag0.npz + grid0.json, 04-18/19):
  * V_MIN is the load-bearing constant. The canonical crop's JOINT contrast stretch maps the 2nd-percentile channel value to 0,
    so a near-black coat pixel with a few levels of tint is amplified into a high saturation: in the training crops 77 / 76 /
    66 / 56 / 33 / 14 / 5.7 / 2.5 / 0.75 / 0.06 % of the foreground pixels in the V bins [0,.1) ... [.9,1] have S >= 0.35.
    That dark-coat artefact band ends at V ~ 0.5, and its hue is BLUE (57 % of the coloured calf pixels at S,V >= .35 fall in
    210-240 degrees, 30 % in 240-270: black hair reflects sky light and the camera renders shadows cool). It therefore has the
    SAME hue as blue workwear and cannot be removed by a hue rule — only by brightness. V_MIN = 0.5 ("the brightest channel is
    at least half of full scale") is where the artefact rate falls below ~6 % per bin, and at V = 0.5 JPEG chroma noise
    (+- ~5 levels at q88) cannot reach S = 0.35 (needs a 45-level channel spread).
  * S_MIN = 0.35: HSV saturation 0.35 means the smallest channel is <= 65 % of the largest — outside every achromatic
    rendering of a black / white / grey coat under a warm or cool light cast (a white coat under dusk / tungsten light is
    S ~ 0.1-0.25) and a permissive "clearly coloured" level (vivid clothing / equipment is S 0.6-1.0). It is deliberately
    NOT raised to 0.5 to suppress pale pink skin or straw at the mask edge: those are handled by the crop-fraction threshold and
    the segment majority rule, and a lower S_MIN keeps faded / pale clothing detectable.
  Grid on 3,000 training crops (coloured_frac P99.9 / max): (0.35, 0.35) 0.102 / 0.117; (0.35, 0.50) 0.032 / 0.043;
  (0.50, 0.50) 0.018 / 0.022 — V_MIN 0.35 -> 0.50 removes 70 % of the calf tail, S_MIN 0.35 -> 0.50 only a further 45 %.

COLOUR NON-CALF RULE (--build, training days only; pre-declared constants below):
  populations (the Tol_50 gate chain, imported, applied in its order; funnel asserted equal to the frozen anchor pool's
  manifest so the crop set is provably the verified one):
    MAIN   G-a Tol_30 production crop gate + G-b verified identity rule + G-c specialist faults + G-d strict unlisted mask
           evidence = verified calf crops, all phases.                                                   -> crop_threshold
    TRACK  every G-a-gated crop of every training tracker segment (grey + fault crops included: a test day has no fault
           list and a real calf segment there carries its merge / partial-mask crops too).               -> min_gated_crops
  crop_threshold  = ceil to THR_DP decimals of THR_MULT x max(P{THR_PCT} of MAIN pooled, P{THR_PCT} of MAIN in each phase4)
                    (a crop is "coloured" iff coloured_frac > crop_threshold, strict). THR_MULT = 2 is the margin for the
                    3-10 days of appearance drift to the test week (brighter sun, fresh straw); the rule's resulting value
                    is also reported against the MAIN and TRACK maxima.
  frac_above      = FRAC_ABOVE (a MAJORITY of the segment's gated crops: robust to a person whose mask also covers a calf).
  min_gated_crops = max(MIN_FLOOR, ceil(MIN_SAFETY x n0)); n0 = 1 + the largest n for which SOME n consecutive gated crops of
                    ONE training tracker segment (TRACK) have >= frac_above of their crops above crop_threshold (0 if none).
  A segment is NON-CALF (hidden) iff n_gated >= min_gated_crops AND n_above >= frac_above x n_gated. Asserted: 0 training
  segments hidden. (colour_noncalf_segments below is the pure reference implementation for a Tol_30 consumer.)

Outputs: per root/day (or training window) <out>/<root>/<day|window>/crop_colour_v1.npz {paths (the bands npz 'paths' array,
same strings, same order), coloured_frac, sat_median, hue_mode, n_fg, bright_frac, hue_hist} (byte-deterministic);
<anchor_pool>/noncalf_colour_v1.json (FROZEN, byte-deterministic, never overwritten with different content); report JSON +
contact sheets under the colorfilter work dir.

Usage:  python Tol_54_crop_colour.py --selftest
        python Tol_54_crop_colour.py --score --bands_path <crop_bands3_dense.npz> --crops <dir with <seg>/<HHMMSS>.jpg> --npz_out <npz>
        python Tol_54_crop_colour.py --score_train            (all 15 windows; local crops under --train_crops)
        python Tol_54_crop_colour.py --build [--dry]          (training days only)
        python Tol_54_crop_colour.py --accept                 (04-22/23 v5a10: the worker + every other segment; reads the
                                                               frozen json, never changes it)
"""
import os, sys, re, json, time, math, argparse, hashlib, zipfile, importlib.util, ast
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
SP6 = r"<LOCAL_RUN_DIR>"
CFW = r"<LOCAL_RUN_DIR>/colorfilter"
DEFAULTS = dict(
    root=SP6 + "/train_ident/root",
    relabel_dir=SP6 + "/train_gt/relabel_volume",
    gt=SP6 + "/train_gt/train_identity_gt.json",
    anchors=SP6 + "/train_gt/anchor_pool/anchor_pool_v1.npz",
    t30=HERE + "/Tol_30_periodic_identity.py",
    t30_ref=SP6 + "/train_gt/anchor_ablation/Tol_30_periodic_identity.BACKUP_329ccf51.py",
    t50=HERE + "/Tol_50_build_anchor_pool.py",
    train_crops=r"<LOCAL_RUN_DIR>/cf/tr",
    v5_root=r"<LOCAL_RUN_DIR>/g3/trk/Tracking_v5a10",
    v5_crops=r"<LOCAL_RUN_DIR>/cf/v5",
    scores=CFW + "/scores",
    out=SP6 + "/train_gt/anchor_pool/noncalf_colour_v1.json",
    report=CFW + "/tol54_report.json",
    accept_report=CFW + "/tol54_accept.json",
    bands="crop_bands3_dense.npz",
)
N = 4
TRAIN_DAYS = ("20240418", "20240419")
TEST_DAYS = tuple(f"202404{d}" for d in range(22, 29))
PHASES = ("day", "dawn", "dusk", "night")
VERSION = "noncalf_colour_v1"
NPZ_NAME = "crop_colour_v1.npz"
# ---- per-crop statistic (see the module docstring for the derivation) ------------------------------------------------
FILL = (124, 116, 104)       # Tol_32 / Tol_03 canonical_crop FILL (verbatim)
BG_TOL = 10                  # max-abs channel distance to FILL of a background pixel (JPEG noise on the flat fill is +- 2)
HOLE_MIN = 30                # an enclosed FILL-coloured component this large is a mask hole, not coat
EDGE_ERODE = 3               # px of mask-boundary ring dropped before measuring colour
S_MIN = 0.35
V_MIN = 0.50
MIN_FG = 200                 # px after erosion; below -> NaN statistics
HUE_MODE_BIN = 10.0          # degrees
HUE_HIST_BINS = 12           # 30-degree sectors
# ---- pre-declared decision rules --------------------------------------------------------------------------------------
THR_PCT = 99.9
THR_MULT = 2.0
THR_DP = 2
FRAC_ABOVE = 0.5
MIN_FLOOR = 8                # 8 dense crops = 2 min at the 15-s cadence: never decide on less evidence
MIN_SAFETY = 2.0
PCTS = (50.0, 90.0, 99.0, 99.5, 99.9)
PATH_TAIL_RX = re.compile(r"/crops_dense/([^/]+)/(\d{6})\.jpg$")


# ----------------------------------------------------------------------------- helpers
def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()

def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def fn_sha(path, name):
    src = open(path, encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return hashlib.sha256(ast.get_source_segment(src, node).encode("utf-8")).hexdigest()
    return None

def r6(x):
    if x is None: return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, 6)

def ceil_dp(x, dp):
    f = 10 ** dp
    return math.ceil(round(float(x) * f, 6)) / f

def pct_table(x):
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    if len(x) == 0: return dict(n=0)
    out = dict(n=int(len(x)), mean=r6(x.mean()))
    for p in PCTS: out[f"P{p:g}"] = r6(np.percentile(x, p))
    out["max"] = r6(x.max())
    return out

def save_npz_deterministic(path, arrays):
    """np.load-compatible archive, byte-reproducible (STORED members in the given order, fixed timestamps), written to a
    temp file and moved into place (= Tol_50.save_npz_deterministic)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".part{os.getpid()}"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, arr in arrays:
            arr = np.ascontiguousarray(arr)
            assert arr.dtype != object, f"{name}: object arrays are not allowed"
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED; zi.create_system = 0; zi.external_attr = 0
            with zf.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, arr, allow_pickle=False)
    os.replace(tmp, path)

def write_frozen(path, obj, overwrite=False):
    txt = json.dumps(obj, indent=1, sort_keys=False, ensure_ascii=True) + "\n"
    if os.path.exists(path):
        old = open(path, encoding="utf-8").read()
        if old == txt: return "identical_rebuild"
        if not overwrite: raise SystemExit(f"FROZEN: {path} exists with different content — never overwritten (bump the version)")
    tmp = path + f".part{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f: f.write(txt)
    os.replace(tmp, path)
    return "new"


# ----------------------------------------------------------------------------- the per-crop statistic
_K3 = np.ones((3, 3), bool)

def foreground(rgb):
    """bool [H, W]: not background (FILL-coloured and border-connected or a >= HOLE_MIN component), eroded by EDGE_ERODE"""
    d = np.abs(rgb.astype(np.int16) - np.array(FILL, np.int16)).max(axis=2)
    bgl = d <= BG_TOL
    lab, n = ndi.label(bgl, structure=_K3)
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = sizes >= HOLE_MIN
        keep[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True
        keep[0] = False
        bg = keep[lab]
    else:
        bg = np.zeros(bgl.shape, bool)
    fg = ~bg
    if EDGE_ERODE: fg = ndi.binary_erosion(fg, structure=_K3, iterations=EDGE_ERODE, border_value=0)
    return fg

def hsv_arrays(x):
    """x float [n, 3] in [0, 1] -> (H degrees [0, 360), S, V)"""
    mx = x.max(axis=1); mn = x.min(axis=1); c = mx - mn
    S = np.where(mx > 0, c / np.maximum(mx, 1e-12), 0.0)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]; cc = np.maximum(c, 1e-12)
    H = np.where(mx == r, np.mod((g - b) / cc, 6.0), np.where(mx == g, (b - r) / cc + 2.0, (r - g) / cc + 4.0)) * 60.0
    H = np.where(c > 0, np.mod(H, 360.0), 0.0)
    return H, S, mx

def crop_colour(rgb):
    """rgb uint8 [H, W, 3] -> dict(coloured_frac, sat_median, hue_mode, n_fg, bright_frac, hue_hist[12])"""
    fg = foreground(rgb); n = int(fg.sum())
    nanh = np.full(HUE_HIST_BINS, np.nan, np.float32)
    if n < MIN_FG:
        return dict(coloured_frac=np.nan, sat_median=np.nan, hue_mode=np.nan, n_fg=n, bright_frac=np.nan, hue_hist=nanh)
    x = rgb[fg].astype(np.float64) / 255.0
    H, S, V = hsv_arrays(x)
    bright = V >= V_MIN
    col = bright & (S >= S_MIN)
    nc = int(col.sum())
    hh = np.histogram(H[col], bins=HUE_HIST_BINS, range=(0.0, 360.0))[0].astype(np.float64) / n
    if nc:
        nb = int(round(360.0 / HUE_MODE_BIN))
        hm = np.bincount(np.minimum((H[col] / HUE_MODE_BIN).astype(np.int64), nb - 1), minlength=nb)
        hue_mode = (int(np.argmax(hm)) + 0.5) * HUE_MODE_BIN
    else:
        hue_mode = np.nan
    return dict(coloured_frac=nc / n, sat_median=float(np.median(S[bright])) if bright.any() else np.nan, hue_mode=hue_mode,
                n_fg=n, bright_frac=float(bright.mean()), hue_hist=hh.astype(np.float32))

def read_rgb(p):
    with Image.open(p) as im: return np.asarray(im.convert("RGB"))

def local_crop_path(band_path, crops_dir):
    m = PATH_TAIL_RX.search(str(band_path).replace("\\", "/"))
    if not m: raise SystemExit(f"band path is not a crops_dense crop: {band_path}")
    return os.path.join(crops_dir, m.group(1), m.group(2) + ".jpg")

def score_bands(bands_path, crops_dir, out_path, log_every=2000):
    """one root/day (or training window): every row of the bands npz 'paths', in order -> crop_colour_v1.npz"""
    t0 = time.time()
    with np.load(bands_path, allow_pickle=False) as z: P = np.array(z["paths"])
    n = len(P)
    cf = np.full(n, np.nan, np.float32); sm = np.full(n, np.nan, np.float32); hm = np.full(n, np.nan, np.float32)
    nf = np.zeros(n, np.int32); bf = np.full(n, np.nan, np.float32); hh = np.full((n, HUE_HIST_BINS), np.nan, np.float32)
    missing = [str(P[i]) for i in range(n) if not os.path.exists(local_crop_path(P[i], crops_dir))]
    if missing: raise SystemExit(f"{len(missing)} of {n} crops missing under {crops_dir} (e.g. {missing[0]}) — download first")
    for i in range(n):
        r = crop_colour(read_rgb(local_crop_path(P[i], crops_dir)))
        cf[i] = r["coloured_frac"]; sm[i] = r["sat_median"]; hm[i] = r["hue_mode"]; nf[i] = r["n_fg"]
        bf[i] = r["bright_frac"]; hh[i] = r["hue_hist"]
        if log_every and (i + 1) % log_every == 0: print(f"    {i + 1}/{n}  {time.time() - t0:.0f} s", flush=True)
    save_npz_deterministic(out_path, [("paths", P), ("coloured_frac", cf), ("sat_median", sm), ("hue_mode", hm), ("n_fg", nf),
                                      ("bright_frac", bf), ("hue_hist", hh)])
    return dict(n=n, n_nan=int(np.sum(~np.isfinite(cf))), seconds=round(time.time() - t0, 1), sha256=sha256_file(out_path))

def load_colour(npz_path, bands_path):
    """colour arrays for one root/day, asserted row-aligned with the bands npz (identical 'paths' array)"""
    with np.load(npz_path, allow_pickle=False) as z: C = {k: np.array(z[k]) for k in z.files}
    with np.load(bands_path, allow_pickle=False) as z: P = np.array(z["paths"])
    if C["paths"].shape != P.shape or not np.array_equal(C["paths"], P):
        raise SystemExit(f"{npz_path}: paths differ from {bands_path} — rescore")
    return C


# ----------------------------------------------------------------------------- the rule (pure; reference for consumers)
def colour_noncalf_segments(crops, coloured_frac, cfg):
    """{sid: dict(n_gated, n_above, frac_above, cf_median)} for every NON-CALF segment: >= cfg["min_gated_crops"] gated crops
    of which >= cfg["frac_above"] have coloured_frac[c["i"]] > cfg["crop_threshold"] (strict >; NaN never counts as above
    but DOES count as a gated crop). crops = Tol_30.load_inputs crops (keys sid, i, gated)."""
    thr = float(cfg["crop_threshold"]); mg = int(cfg["min_gated_crops"]); fr = float(cfg["frac_above"])
    by = defaultdict(list)
    for c in crops:
        if c["gated"]: by[c["sid"]].append(float(coloured_frac[c["i"]]))
    out = {}
    for sid, v in by.items():
        n = len(v); na = sum(1 for x in v if x > thr)
        if n >= mg and na >= fr * n:
            fv = [x for x in v if math.isfinite(x)]
            out[sid] = dict(n_gated=n, n_above=na, frac_above=round(na / n, 4),
                            cf_median=round(float(np.median(fv)), 5) if fv else None)
    return out

def block_max_frac(b, nmax):
    """b: 0/1 sequence; m[n] (n = 1..min(len(b), nmax)) = max over contiguous blocks of length n of mean(b)"""
    b = np.asarray(b, dtype=np.int64); L = len(b)
    cs = np.concatenate([[0], np.cumsum(b)])
    return {n: float((cs[n:] - cs[:-n]).max()) / n for n in range(1, min(L, nmax) + 1)}

def segment_evidence(seqs, thr, frac):
    """n0 = 1 + the largest n such that some n consecutive gated crops of ONE segment have >= frac above thr (0 if none)"""
    nmax = max(len(v) for v in seqs.values())
    worst = defaultdict(float); arg = {}
    for key, v in seqs.items():
        for n, f in block_max_frac([x > thr for _, x, _ in v], nmax).items():
            if f > worst[n]: worst[n] = f; arg[n] = key
    trig = [n for n in range(1, nmax + 1) if worst.get(n, 0.0) >= frac]
    return ((max(trig) + 1) if trig else 0), {n: worst.get(n, 0.0) for n in range(1, nmax + 1)}, arg


# ----------------------------------------------------------------------------- training collection (Tol_50 gate chain)
def collect_train(a, T, M50):
    wins = sorted(w for w in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, w)))
    for w in wins:
        if not M50.WIN_RX.match(w) or w[:8] not in TRAIN_DAYS:
            raise SystemExit(f"LEAKAGE GUARD: {w} is not a 04-18/19 training window")
    GT = json.load(open(a.gt, encoding="utf-8")); rs = GT["relabel_source"]["sha256"]
    spec_ex, strict_ex, _ = M50.gt_exclusions(GT)
    rows = []; cnt = {s: Counter() for s in ("all", "gated", "labelled", "spec_excl", "strict_excl")}; inputs = {}
    for win in wins:
        d = os.path.join(a.root, win); bp = os.path.join(d, a.bands)
        rf = f"relabel_week_{win}.json"; rp = os.path.join(a.relabel_dir, rf)
        if sha256_file(rp) != rs.get(rf): raise SystemExit(f"{rf} is not the relabel file the specialist verified")
        cp = os.path.join(a.scores, "Tracking_train", win, NPZ_NAME)
        C = load_colour(cp, bp)
        for p in C["paths"]:
            m = M50.PATH_RX.search(str(p).replace("\\", "/"))
            if not m or m.group(1) != win: raise SystemExit(f"band path not under /Tracking_train/{win}/: {p}")
        segs, crops, _, _ = T.load_inputs(d, "crops", a.bands)
        R, EX = M50.load_relabel(rp)
        inputs[win] = dict(band_sha256=sha256_file(bp), colour_npz_sha256=sha256_file(cp), relabel_sha256=rs[rf],
                           **{f: sha256_file(os.path.join(d, f)) for f in ("segments.json", "events.jsonl", "observations_meta.json")})
        for c in crops:
            w = c["w"]; ph = M50.phase4(w); lab = M50.ident_at(R, EX, int(c["slot"]), w)
            cnt["all"][M50.kp(None, ph)] += 1
            if not c["gated"]: st = "G-a"
            else:
                cnt["gated"][M50.kp(lab, ph)] += 1
                if lab is None: st = "G-b"
                else:
                    cnt["labelled"][M50.kp(lab, ph)] += 1
                    if any(x <= w < y for x, y in spec_ex[lab]): st = "G-c"
                    else:
                        cnt["spec_excl"][M50.kp(lab, ph)] += 1
                        if any(x <= w < y for x, y in strict_ex[lab]): st = "G-d"
                        else: cnt["strict_excl"][M50.kp(lab, ph)] += 1; st = "pass"
            i = c["i"]
            rows.append(dict(win=win, day=win[:8], sid=c["sid"], slot=int(c["slot"]), w=w, ph=ph, lab=lab, gated=bool(c["gated"]),
                             stage=st, path=str(C["paths"][i]), local=local_crop_path(C["paths"][i], os.path.join(a.train_crops, win)),
                             cf=float(C["coloured_frac"][i]), sat=float(C["sat_median"][i]), hue=float(C["hue_mode"][i]),
                             n_fg=int(C["n_fg"][i]), bright=float(C["bright_frac"][i]), hh=C["hue_hist"][i]))
        print(f"  {win}: {len(crops)} crops ({sum(1 for c in crops if c['gated'])} gated)", flush=True)
    return rows, cnt, inputs, wins

def funnel_check(cnt, manifest):
    ref = manifest["gate_funnel"]["counts"]; diffs = {}
    for st, c in cnt.items():
        x = ref.get(st, {}); y = dict(c)
        dd = {k: [x.get(k), y.get(k)] for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)}
        if dd: diffs[st] = dd
    return diffs


# ----------------------------------------------------------------------------- contact sheet
def contact_sheet(items, path, title, cols=10, tile=160):
    """items: list of dict(local, caption); each tile = crop over its coloured-pixel overlay (magenta)"""
    from PIL import ImageDraw
    rows_ = int(math.ceil(len(items) / cols)); cap = 26; th = 2 * tile + cap
    W, H = cols * (tile + 4) + 4, rows_ * (th + 4) + 34
    sheet = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(sheet)
    dr.text((6, 8), title, fill=(0, 0, 0))
    for k, it in enumerate(items):
        rgb = read_rgb(it["local"])
        fg = foreground(rgb)
        x = rgb.astype(np.float64) / 255.0
        Hh, S, V = hsv_arrays(x.reshape(-1, 3)); col = ((S >= S_MIN) & (V >= V_MIN)).reshape(fg.shape) & fg
        ov = (rgb.astype(np.float64) * 0.35 + 0.65 * 128).astype(np.uint8); ov[~fg] = (255, 255, 255); ov[col] = (255, 0, 255)
        r_, c_ = divmod(k, cols); x0 = 4 + c_ * (tile + 4); y0 = 34 + r_ * (th + 4)
        sheet.paste(Image.fromarray(rgb).resize((tile, tile), Image.BILINEAR), (x0, y0))
        sheet.paste(Image.fromarray(ov).resize((tile, tile), Image.NEAREST), (x0, y0 + tile))
        for j, line in enumerate(it["caption"].split("\n")[:2]): dr.text((x0 + 2, y0 + 2 * tile + 1 + 12 * j), line, fill=(0, 0, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True); sheet.save(path)
    return path

def cap(r):
    return (f"{r['cf']:.3f} s{r['sat']:.2f} h{r['hue']:.0f}\n{r['win'][4:8]} {r['sid']} {r['w']:%H%M%S} {r['ph'][:2]}"
            f"{'' if r['lab'] is None else ' id' + str(r['lab'])} {r['stage']}")


# ----------------------------------------------------------------------------- build (training days only)
def build(a):
    t0 = time.time()
    T = import_module(a.t30, "Tol_30_periodic_identity_t54")
    M50 = import_module(a.t50, "Tol_50_build_anchor_pool_t54")
    t30_info = M50.check_t30(T, a.t30, a.t30_ref)
    man_p = a.anchors[:-4] + ".manifest.json"; man = json.load(open(man_p, encoding="utf-8"))
    rows, cnt, inputs, wins = collect_train(a, T, M50)
    diffs = funnel_check(cnt, man)
    if diffs: raise SystemExit(f"gate funnel differs from the frozen pool's manifest: {json.dumps(diffs)[:1500]}")
    print(f"  funnel == frozen pool manifest ({len(rows)} crops)  ({time.time() - t0:.0f} s)", flush=True)
    cf = np.array([r["cf"] for r in rows], np.float64)
    pops = {"MAIN (G-a..G-d, threshold)": lambda r: r["stage"] == "pass",
            "G-a..G-c": lambda r: r["stage"] in ("pass", "G-d"),
            "G-a..G-b (faults included)": lambda r: r["stage"] in ("pass", "G-d", "G-c"),
            "TRACK (all gated, grey + faults included)": lambda r: r["gated"],
            "grey (gated, no identity rule)": lambda r: r["stage"] == "G-b",
            "ungated (not used by the rule)": lambda r: not r["gated"]}
    dist, sat_dist, bright_dist = {}, {}, {}
    for name, f in pops.items():
        J = [j for j, r in enumerate(rows) if f(r)]; ph = np.array([rows[j]["ph"] for j in J])
        dist[name] = dict(pooled=pct_table(cf[J]), by_phase={p: pct_table(cf[J][ph == p]) for p in PHASES},
                          n_nan=int(np.sum(~np.isfinite(cf[J]))))
    J = [j for j, r in enumerate(rows) if r["stage"] == "pass"]; ph = np.array([rows[j]["ph"] for j in J])
    for nm, key in (("sat_median", "sat"), ("bright_frac", "bright")):
        v = np.array([rows[j][key] for j in J], np.float64)
        (sat_dist if key == "sat" else bright_dist).update(pooled=pct_table(v), by_phase={p: pct_table(v[ph == p]) for p in PHASES})
    if np.sum(~np.isfinite(cf[J])): raise SystemExit(f"{int(np.sum(~np.isfinite(cf[J])))} MAIN crops have no statistic (n_fg < MIN_FG)")
    by_ident = {str(k): pct_table(cf[[j for j in J if rows[j]["lab"] == k]]) for k in range(N)}
    by_win = {w: pct_table(cf[[j for j in J if rows[j]["win"] == w]]) for w in wins}
    by_hour = {f"{h:02d}": pct_table(cf[[j for j in J if rows[j]["w"].hour == h]]) for h in range(24)
               if any(rows[j]["w"].hour == h for j in J)}
    HH = np.stack([rows[j]["hh"] for j in J]).astype(np.float64)
    hue_share = (HH.sum(axis=0) / max(HH.sum(), 1e-12)).round(4).tolist()
    # ---- crop threshold
    p_pool = float(np.percentile(cf[J], THR_PCT))
    p_phase = {p: float(np.percentile(cf[J][ph == p], THR_PCT)) for p in PHASES if np.any(ph == p)}
    raw = max([p_pool] + list(p_phase.values()))
    thr = ceil_dp(THR_MULT * raw, THR_DP)
    binding = "pooled" if raw == p_pool else "phase:" + max(p_phase, key=p_phase.get)
    Jt = [j for j, r in enumerate(rows) if r["gated"]]
    thr_prov = dict(rule=f"ceil_{THR_DP}dp({THR_MULT:g} x max(P{THR_PCT:g} of MAIN pooled, P{THR_PCT:g} of MAIN in each phase4)); "
                         f"a crop is coloured iff coloured_frac > crop_threshold (strict)",
                    raw=r6(raw), pooled_value=r6(p_pool), phase_values={k: r6(v) for k, v in p_phase.items()}, binding=binding,
                    mult=THR_MULT, main_max=r6(np.nanmax(cf[J])), track_max=r6(np.nanmax(cf[Jt])),
                    main_frac_above={"pooled": r6(np.mean(cf[J] > thr)), **{p: r6(np.mean(cf[J][ph == p] > thr)) for p in p_phase}},
                    track_n_above=int(np.sum(cf[Jt] > thr)), track_n=len(Jt))
    # ---- segment rule (TRACK)
    seqs = defaultdict(list)
    for j in Jt: seqs[(rows[j]["win"], rows[j]["sid"])].append((rows[j]["w"], float(cf[j]), rows[j]["stage"]))
    for k in seqs: seqs[k].sort()
    n0, worst, warg = segment_evidence(seqs, thr, FRAC_ABOVE)
    min_g = max(MIN_FLOOR, int(math.ceil(MIN_SAFETY * n0)))
    # the pure rule, on the real training crops (keys as Tol_30 would pass them)
    crops_like = [dict(sid=f"{rows[j]['win']}/{rows[j]['sid']}", i=j, gated=rows[j]["gated"]) for j in range(len(rows))]
    cfg = dict(crop_threshold=thr, min_gated_crops=min_g, frac_above=FRAC_ABOVE)
    fp = colour_noncalf_segments(crops_like, cf, cfg)
    if fp: raise SystemExit(f"the chosen rule hides TRAINING segments: {fp}")
    seg_tab = sorted(((f"{w}/{s}", len(v), sum(1 for _, x, _ in v if x > thr) / len(v), float(np.nanmedian([x for _, x, _ in v])),
                       float(np.nanmax([x for _, x, _ in v]))) for (w, s), v in seqs.items()), key=lambda t: (-t[2], -t[4]))
    runs = Counter()
    for v in seqs.values():
        r_ = 0
        for _, x, _ in v + [(None, -1.0, None)]:
            if x > thr: r_ += 1
            elif r_: runs[r_] += 1; r_ = 0
    sens = {}
    for mult in (1.0, 1.5, 3.0):
        t_ = ceil_dp(mult * raw, THR_DP); n0s, _, _ = segment_evidence(seqs, t_, FRAC_ABOVE)
        sens[f"mult {mult:g}"] = dict(threshold=t_, n0=n0s, min_gated_crops_by_rule=max(MIN_FLOOR, int(math.ceil(MIN_SAFETY * n0s))),
                                      track_n_above=int(np.sum(cf[Jt] > t_)))
    # ---- contact sheets (LOOK at them)
    top = sorted(J, key=lambda j: -cf[j])[:40]
    sheet = contact_sheet([dict(local=rows[j]["local"], caption=cap(rows[j])) for j in top], os.path.join(CFW, "sheets", "top40_main.png"),
                          f"Tol_54: the 40 MOST-COLOURED verified calf crops (MAIN, 04-18/19)  S>={S_MIN} V>={V_MIN}  magenta = coloured px  thr {thr}")
    topt = sorted([j for j in Jt if rows[j]["stage"] != "pass"], key=lambda j: -cf[j])[:40]
    sheet2 = contact_sheet([dict(local=rows[j]["local"], caption=cap(rows[j])) for j in topt], os.path.join(CFW, "sheets", "top40_track_nonmain.png"),
                           "Tol_54: the 40 most-coloured GATED training crops OUTSIDE MAIN (grey / faults / strict)")
    frozen = {
        "what": "COLOUR non-calf filter for the identity layer (user decision 2026-09-30: 'exclude colour except for white/black "
                "should be a strong filter'): a track whose masked crops are clearly coloured is not one of the four black-and-"
                "white calves and is HIDDEN (e.g. the 04-23 15:38-15:49 farm worker in blue overalls, v5 seg 1_012)",
        "version": VERSION,
        "derived_from": list(TRAIN_DAYS), "not_from": list(TEST_DAYS),
        "rule": "segment is NON-CALF (hidden) iff n_gated >= min_gated_crops and n_above >= frac_above * n_gated; n_gated = crops "
                "of the segment passing Tol_30's production crop gate (load_inputs 'gated'); n_above = #{gated crop i : "
                "coloured_frac[i] > crop_threshold} (strict >; a NaN statistic is gated but never above). coloured_frac = the "
                "crop_colour_v1.npz row aligned with the bands npz row i (identical 'paths' array).",
        "crop_threshold": thr, "min_gated_crops": min_g, "frac_above": FRAC_ABOVE,
        "statistic": dict(name="coloured_frac", npz=NPZ_NAME,
                          definition="fraction of the crop's FOREGROUND pixels with HSV S >= S_MIN and V >= V_MIN; foreground = "
                                     "not background (within BG_TOL of FILL and border-connected or >= HOLE_MIN px), eroded by "
                                     "EDGE_ERODE px", S_MIN=S_MIN, V_MIN=V_MIN, FILL=list(FILL), BG_TOL=BG_TOL, HOLE_MIN=HOLE_MIN,
                          EDGE_ERODE=EDGE_ERODE, MIN_FG=MIN_FG, input="Tol_32 dense canonical crops (224 px JPEG q88, joint "
                                                                          "2-98 % stretch, FILL outside the mask)",
                          crop_colour_sha256=fn_sha(os.path.abspath(__file__), "crop_colour"),
                          foreground_sha256=fn_sha(os.path.abspath(__file__), "foreground"),
                          hsv_arrays_sha256=fn_sha(os.path.abspath(__file__), "hsv_arrays")),
        "provenance": {
            "S_MIN_V_MIN": "first principles + 04-18/19 training crops only (Tol_54 module docstring): V_MIN 0.5 removes the dark-"
                           "coat artefact of the joint contrast stretch (black coat pixels amplified to high S, hue BLUE 210-270 deg "
                           "= the hue of blue workwear, so it cannot be removed by hue); S_MIN 0.35 = min channel <= 65 % of max, "
                           "outside any achromatic coat rendering under a light cast",
            "crop_threshold": thr_prov,
            "frac_above": dict(rule="majority of the gated crops (pre-declared FRAC_ABOVE; not fitted)", value=FRAC_ABOVE),
            "min_gated_crops": dict(rule=f"max(MIN_FLOOR={MIN_FLOOR}, ceil(MIN_SAFETY={MIN_SAFETY:g} * n0)); n0 = 1 + the largest "
                                         "block size n such that some n consecutive gated crops of ONE training tracker segment "
                                         "(TRACK) have >= frac_above of their crops above crop_threshold (0 if none)",
                                    n0=n0, triggering_block_at_n0_minus_1=(dict(zip(("window", "seg"), warg[n0 - 1])) if n0 > 1 else None),
                                    value=min_g),
            "population_MAIN": "G-a Tol_30 production crop gate, G-b verified identity rule (train relabel files), G-c specialist "
                               "fault exclusions, G-d strict unlisted mask evidence +-90 s (Tol_50 gate chain; funnel asserted "
                               "equal to the frozen anchor pool's manifest); ALL phases",
            "population_TRACK": "every G-a-gated crop of every training tracker segment (grey + fault crops included)",
            "n_main": len(J), "n_track": len(Jt), "n_training_segments": len(seqs), "training_segments_hidden": 0,
            "sensitivity": sens,
        },
        "training_distribution_MAIN": dist["MAIN (G-a..G-d, threshold)"],
        "tools": dict(tol54=dict(file="Tol_54_crop_colour.py", sha256=sha256_file(os.path.abspath(__file__))),
                      tol30=dict(load_inputs_sha256=fn_sha(a.t30, "load_inputs"),
                                 load_inputs_identical_to_labelling_version=t30_info["load_inputs_identical_to_ref"]),
                      tol50=dict(sha256=sha256_file(a.t50), gt_exclusions_sha256=fn_sha(a.t50, "gt_exclusions"),
                                 ident_at_sha256=fn_sha(a.t50, "ident_at"))),
        "inputs": dict(train_identity_gt_sha256=sha256_file(a.gt), anchor_pool_manifest_sha256=sha256_file(man_p), windows=inputs),
    }
    report = dict(frozen_json=a.out, threshold=thr, min_gated_crops=min_g, frac_above=FRAC_ABOVE, threshold_provenance=thr_prov,
                  distributions=dist, main_sat_median=sat_dist, main_bright_frac=bright_dist, main_by_identity=by_ident,
                  main_by_window=by_win, main_by_hour=by_hour, main_hue_sector_share_of_coloured_px=hue_share,
                  top40_main=[dict(win=rows[j]["win"], seg=rows[j]["sid"], wall=rows[j]["w"].isoformat(), phase=rows[j]["ph"],
                                   ident=rows[j]["lab"], cf=r6(cf[j]), sat_median=r6(rows[j]["sat"]), hue_mode=r6(rows[j]["hue"]),
                                   bright_frac=r6(rows[j]["bright"])) for j in top],
                  segment_rule=dict(n0=n0, worst_block_frac_by_n={str(n): r6(f) for n, f in worst.items() if n <= 60},
                                    above_threshold_run_lengths=dict(sorted(runs.items())),
                                    training_segments_top=[dict(seg=k, n_gated=n, frac_above=r6(f), cf_median=r6(m), cf_max=r6(mx))
                                                           for k, n, f, m, mx in seg_tab[:12]]),
                  sensitivity=sens, sheets=[sheet, sheet2], seconds=round(time.time() - t0, 1))
    return frozen, report


# ----------------------------------------------------------------------------- acceptance (04-22/23; reads, never tunes)
WORKER = ("20240423", "1_012")

def accept(a):
    J = json.load(open(a.out, encoding="utf-8"))
    if J.get("version") != VERSION or set(J["derived_from"]) - set(TRAIN_DAYS): raise SystemExit("not a training-derived v1 json")
    T = import_module(a.t30, "Tol_30_periodic_identity_t54a")
    cfg = {k: J[k] for k in ("crop_threshold", "min_gated_crops", "frac_above")}
    out = dict(frozen_json=a.out, frozen_sha256=sha256_file(a.out), rule=cfg, days={})
    for day in ("20240422", "20240423"):
        d = os.path.join(a.v5_root, day); bp = os.path.join(d, a.bands)
        cp = os.path.join(a.scores, "Tracking_v5a10", day, NPZ_NAME)
        if not os.path.exists(cp): out["days"][day] = "not scored"; continue
        C = load_colour(cp, bp)
        segs, crops, _, _ = T.load_inputs(d, "crops", a.bands)
        cf = C["coloured_frac"].astype(np.float64)
        hid = colour_noncalf_segments(crops, cf, cfg)
        per = defaultdict(list)
        for c in crops:
            if c["gated"]: per[c["sid"]].append(float(cf[c["i"]]))
        seg_tab = sorted(((sid, len(v), sum(1 for x in v if x > cfg["crop_threshold"]) / len(v), float(np.nanmedian(v)),
                           float(np.nanmax(v))) for sid, v in per.items()), key=lambda t: (-t[2], -t[4]))
        oth = [float(cf[c["i"]]) for c in crops if c["gated"] and not (day, c["sid"]) == WORKER]
        rec = dict(bands_sha256=sha256_file(bp), colour_sha256=sha256_file(cp), n_crops=len(crops),
                   hidden={k: dict(v, slot=int(segs[k]["slot"]), t_start=segs[k]["t_start_wall"], t_end=segs[k]["t_end_wall"],
                                   open=segs[k].get("open_reason")) for k, v in hid.items()},
                   segments_top=[dict(seg=k, n_gated=n, frac_above=r6(f), cf_median=r6(m), cf_max=r6(mx)) for k, n, f, m, mx in seg_tab[:10]],
                   other_gated_crops=dict(pct_table(oth), n_above=int(np.sum(np.array(oth) > cfg["crop_threshold"]))))
        if day == WORKER[0]:
            wc = sorted([c for c in crops if c["sid"] == WORKER[1]], key=lambda c: c["w"])
            rec["worker"] = dict(
                seg=WORKER[1], n_crops=len(wc), n_gated=sum(1 for c in wc if c["gated"]),
                n_above=sum(1 for c in wc if c["gated"] and cf[c["i"]] > cfg["crop_threshold"]),
                hidden=WORKER[1] in hid,
                cf_gated=pct_table([cf[c["i"]] for c in wc if c["gated"]]), cf_all=pct_table([cf[c["i"]] for c in wc]),
                crops=[dict(wall=f"{c['w']:%H:%M:%S}", gated=bool(c["gated"]), cf=r6(cf[c["i"]]), sat_median=r6(C["sat_median"][c["i"]]),
                            hue_mode=r6(C["hue_mode"][c["i"]]), bright_frac=r6(C["bright_frac"][c["i"]]), n_fg=int(C["n_fg"][c["i"]]))
                       for c in wc])
            if wc:
                contact_sheet([dict(local=local_crop_path(C["paths"][c["i"]], os.path.join(a.v5_crops, day)),
                                    caption=f"{cf[c['i']]:.3f} s{C['sat_median'][c['i']]:.2f} h{C['hue_mode'][c['i']]:.0f}\n"
                                            f"{c['w']:%H%M%S} {'g' if c['gated'] else 'UNGATED'}") for c in wc],
                              os.path.join(CFW, "sheets", "worker_1_012.png"),
                              f"Tol_54 acceptance: v5 20240423 seg 1_012 (farm worker)  thr {cfg['crop_threshold']}")
        out["days"][day] = rec
    return out


# ----------------------------------------------------------------------------- SELFTEST (synthetic images)
def _synthetic_calf(rng, coloured=None, frac=0.6, size=224, tint=(0, 0, 0), dark_colour=False, ring=False, hole=False, jpeg=True):
    """FILL background + an ellipse 'calf' with black / white patches (+ optional coloured region), JPEG q88 round trip"""
    img = np.empty((size, size, 3), np.float64); img[:] = FILL
    yy, xx = np.mgrid[:size, :size]
    m = ((xx - size / 2) / (size * 0.33)) ** 2 + ((yy - size / 2) / (size * 0.44)) ** 2 <= 1.0
    patch = (np.sin(xx / 17.0) + np.cos(yy / 23.0)) > 0
    calf = np.where(patch[..., None], 235.0, 18.0) + np.array(tint, np.float64) + rng.normal(0, 3, (size, size, 3))
    img[m] = calf[m]
    if coloured is not None:
        ys = np.flatnonzero(m.any(axis=1)); y0, y1 = ys.min(), ys.max()
        cut = y0 + int(round(frac * (y1 - y0)))
        reg = m & (yy <= cut)
        img[reg] = np.array(coloured, np.float64) + rng.normal(0, 3, (int(reg.sum()), 3))
    if dark_colour:                                   # tinted near-black, as the joint stretch leaves a black coat
        reg = m & (xx < size / 2); img[reg] = np.array((12, 30, 70), np.float64)
    if ring:                                          # a 2-px vividly coloured ring exactly at the mask boundary
        er = ndi.binary_erosion(m, structure=_K3, iterations=2); img[m & ~er] = (0, 80, 255)
    if hole:                                          # a FILL-coloured hole inside the mask (mask hole)
        img[(xx - size / 2) ** 2 + (yy - size / 2) ** 2 < 15 ** 2] = FILL
    if not jpeg: return np.clip(img, 0, 255).astype(np.uint8), m
    import io
    buf = io.BytesIO(); Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB")), m

def selftest(a):
    rng = np.random.default_rng(0)
    # 1. HSV vs colorsys
    import colorsys
    x = rng.random((500, 3)); x[:20] = x[:20, :1]           # include greys
    H, S, V = hsv_arrays(x)
    for k in range(len(x)):
        h, s, v = colorsys.rgb_to_hsv(*x[k])
        assert abs(S[k] - s) < 1e-9 and abs(V[k] - v) < 1e-9, (k, S[k], s)
        if s > 1e-9: assert min(abs(H[k] - 360 * h), 360 - abs(H[k] - 360 * h)) < 1e-6, (k, H[k], 360 * h)
    # 2. pure background -> no foreground, NaN statistics
    bg = np.empty((224, 224, 3), np.uint8); bg[:] = (124, 116, 105)
    r = crop_colour(bg); assert r["n_fg"] == 0 and not math.isfinite(r["coloured_frac"]) and not math.isfinite(r["hue_mode"]), r
    # 3. black-and-white calf (with a warm and a cool light cast) -> essentially no coloured pixels
    for tint in ((0, 0, 0), (12, 4, -12), (-8, 0, 10)):
        rgb, m = _synthetic_calf(rng, tint=tint); r = crop_colour(rgb)
        assert r["coloured_frac"] < 0.01, (tint, r["coloured_frac"])
        assert 0.8 * (m.sum() * 0.93) < r["n_fg"] <= m.sum(), (r["n_fg"], m.sum())
    # 4. blue overalls over ~60 % of the mask -> coloured_frac ~0.6, hue ~220
    rgb, m = _synthetic_calf(rng, coloured=(20, 100, 235), frac=0.6); r = crop_colour(rgb)
    assert 0.5 < r["coloured_frac"] < 0.7 and 210 <= r["hue_mode"] <= 230 and r["sat_median"] > 0.35, r
    assert abs(float(np.nansum(r["hue_hist"])) - r["coloured_frac"]) < 1e-6
    # 5. hue modes: red wraps near 0, green ~120, orange hi-vis ~ 20-30
    for col, lo, hi in (((230, 25, 20), 0, 10), ((30, 200, 40), 115, 130), ((250, 110, 10), 20, 30), ((230, 20, 60), 340, 360)):
        rgb, _ = _synthetic_calf(rng, coloured=col, frac=0.9); r = crop_colour(rgb)
        assert lo <= r["hue_mode"] <= hi and r["coloured_frac"] > 0.75, (col, r["hue_mode"], r["coloured_frac"])
    # 6. dark tinted coat (V < V_MIN) and pale casts (S < S_MIN) are NOT coloured
    rgb, _ = _synthetic_calf(rng, dark_colour=True); r = crop_colour(rgb)
    assert r["coloured_frac"] < 0.01, r
    pale = _synthetic_calf(rng, coloured=(235, 200, 175), frac=0.9)[0]; r = crop_colour(pale)   # S ~ 0.26, V 0.92
    assert r["coloured_frac"] < 0.01, r
    # 7. a thin coloured ring at the mask boundary is eroded away; an enclosed FILL hole is background
    rgb, m = _synthetic_calf(rng, ring=True, jpeg=False); r = crop_colour(rgb)
    assert r["coloured_frac"] == 0.0, r                                       # exact pixels: the 2-px ring is fully eroded
    ring_frac = float((m & ~ndi.binary_erosion(m, structure=_K3, iterations=2)).sum()) / float(m.sum())
    rgb, m = _synthetic_calf(rng, ring=True); r = crop_colour(rgb)             # JPEG 4:2:0 bleeds some ring chroma inward
    assert r["coloured_frac"] < 0.35 * ring_frac, (r["coloured_frac"], ring_frac)
    rgb, m = _synthetic_calf(rng, hole=True); fg = foreground(rgb)
    assert not fg[112, 112] and fg[112, 112 + 40], "enclosed FILL hole must be background, coat next to it foreground"
    # 8. thresholds bite exactly at S_MIN / V_MIN (no JPEG: exact pixels)
    img = np.empty((64, 64, 3), np.uint8); img[:] = FILL
    v = int(math.ceil(V_MIN * 255)); lo_ = int(math.floor(v * (1 - S_MIN)))          # S >= S_MIN at V >= V_MIN
    img[8:56, 8:32] = (v, lo_, lo_); img[8:56, 32:56] = (v - 1, 0, 0)                    # left coloured, right too dark
    r = crop_colour(img); fgn = r["n_fg"]
    assert fgn > 0 and abs(r["coloured_frac"] - 0.5) < 0.1, r
    # 9. segment rule edges + block analysis vs brute force
    crops, cf = [], []
    def seg(sid, vals, gated=True):
        for x in vals: crops.append(dict(sid=sid, i=len(cf), gated=gated)); cf.append(x)
    seg("a", [0.9] * 8); seg("b", [0.9] * 7); seg("c", [0.9] * 4 + [0.0] * 4); seg("d", [0.9] * 3 + [0.0] * 5)
    seg("e", [0.2] * 8); seg("f", [0.9] * 20, gated=False); seg("g", [0.9] * 4 + [float("nan")] * 4); seg("h", [0.9] * 3 + [float("nan")] * 5)
    got = colour_noncalf_segments(crops, np.array(cf), dict(crop_threshold=0.2, min_gated_crops=8, frac_above=0.5))
    assert sorted(got) == ["a", "c", "g"] and got["c"]["n_above"] == 4 and got["g"]["n_gated"] == 8, got   # e: 0.2 is NOT > 0.2
    for L in (1, 2, 7, 40):
        for _ in range(20):
            b = (rng.random(L) < 0.3).astype(int); mm = block_max_frac(b, 50)
            for n in range(1, L + 1):
                assert abs(mm[n] - max(sum(b[i:i + n]) for i in range(L - n + 1)) / n) < 1e-12
    seqs = {("w", "s1"): [(k, x, "pass") for k, x in enumerate([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])]}
    n0, _, _ = segment_evidence(seqs, 0.5, 0.5); assert n0 == 7, n0          # 3 of 6 -> n=6 triggers, n=7 (3/7) does not
    assert ceil_dp(0.0301, 2) == 0.04 and ceil_dp(0.06, 2) == 0.06 and ceil_dp(0.0600000001, 2) == 0.06
    # 10. scorer end-to-end on a synthetic root: row order = bands order, byte-deterministic, missing crops refused
    import tempfile, shutil
    td = tempfile.mkdtemp(prefix="t54_")
    try:
        paths = []
        for k, (sid, col) in enumerate([("0_001", None), ("1_012", (20, 100, 235)), ("0_001", None), ("2_003", (230, 20, 25))]):
            rgb, _ = _synthetic_calf(rng, coloured=col, frac=0.7)
            hh = f"1200{k:02d}"; os.makedirs(os.path.join(td, "crops", sid), exist_ok=True)
            Image.fromarray(rgb).save(os.path.join(td, "crops", sid, hh + ".jpg"), quality=95)
            paths.append(f"/Volumes/x/Tol/Tracking_v5a10/20240423/crops_dense/{sid}/{hh}.jpg")
        P = np.array(paths[::-1]); bp = os.path.join(td, "bands.npz"); np.savez(bp, paths=P, emb=np.zeros((4, 8, 2), np.float16))
        o1, o2 = os.path.join(td, "a", NPZ_NAME), os.path.join(td, "b", NPZ_NAME)
        s1 = score_bands(bp, os.path.join(td, "crops"), o1, 0); s2 = score_bands(bp, os.path.join(td, "crops"), o2, 0)
        assert s1["sha256"] == s2["sha256"], "not byte-deterministic"
        C = load_colour(o1, bp); assert np.array_equal(C["paths"], P)
        cfv = C["coloured_frac"]
        assert cfv[0] > 0.5 and cfv[2] > 0.5 and cfv[1] < 0.01 and cfv[3] < 0.01, cfv        # reversed order preserved
        assert set(C) == {"paths", "coloured_frac", "sat_median", "hue_mode", "n_fg", "bright_frac", "hue_hist"}
        os.remove(os.path.join(td, "crops", "0_001", "120000.jpg"))
        try: score_bands(bp, os.path.join(td, "crops"), o1, 0); raise AssertionError("missing crop accepted")
        except SystemExit as e: assert "missing" in str(e)
        np.savez(os.path.join(td, "bands2.npz"), paths=P[::-1])
        try: load_colour(o2, os.path.join(td, "bands2.npz")); raise AssertionError("misaligned colour npz accepted")
        except SystemExit as e: assert "paths differ" in str(e)
        # 11. frozen writer
        tp = os.path.join(td, "x.json")
        assert write_frozen(tp, {"a": 1}) == "new" and write_frozen(tp, {"a": 1}) == "identical_rebuild"
        try: write_frozen(tp, {"a": 2}); raise AssertionError("frozen overwrite accepted")
        except SystemExit as e: assert "FROZEN" in str(e)
        # 12. leakage guard: a test-day window dir under the training root is refused
        os.makedirs(os.path.join(td, "root", "20240422_w0000-0100"))
        class A_: pass
        aa = A_(); aa.root = os.path.join(td, "root"); aa.gt = a.gt
        M50 = import_module(a.t50, "Tol_50_build_anchor_pool_t54st")
        try: collect_train(aa, None, M50); raise AssertionError("a test-day window was accepted")
        except SystemExit as e: assert "LEAKAGE GUARD" in str(e)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("Tol_54 SELFTEST PASSED (HSV == colorsys, empty crop, b/w calf under casts, blue/red/green/orange/magenta hue modes, "
          "V_MIN / S_MIN exclusions, edge ring erosion, FILL hole, exact threshold bite, segment rule edges incl. NaN / ungated, "
          "block analysis, scorer row order + determinism + missing / misaligned refusal, frozen writer, leakage guard)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in DEFAULTS: ap.add_argument(f"--{k}", default=DEFAULTS[k])
    ap.add_argument("--crops", default=None); ap.add_argument("--bands_path", default=None); ap.add_argument("--npz_out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--score", action="store_true"); ap.add_argument("--score_train", action="store_true")
    ap.add_argument("--score_v5", default=None, help="comma list of v5a10 days to score (local crops under --v5_crops)")
    ap.add_argument("--build", action="store_true"); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--accept", action="store_true"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.selftest: selftest(a); return
    if a.score:
        if not (a.crops and a.npz_out and a.bands_path): ap.error("--score needs --bands_path <npz> --crops <dir> --npz_out <npz>")
        print(json.dumps(score_bands(a.bands_path, a.crops, a.npz_out))); return
    if a.score_train or a.score_v5:
        jobs = []
        if a.score_train:
            for w in sorted(os.listdir(a.root)):
                jobs.append((os.path.join(a.root, w, a.bands), os.path.join(a.train_crops, w), os.path.join(a.scores, "Tracking_train", w, NPZ_NAME)))
        for d in (a.score_v5.split(",") if a.score_v5 else []):
            jobs.append((os.path.join(a.v5_root, d, a.bands), os.path.join(a.v5_crops, d), os.path.join(a.scores, "Tracking_v5a10", d, NPZ_NAME)))
        for bp, cd, op in jobs:
            if os.path.exists(op) and not a.force:
                load_colour(op, bp); print(f"  exists (aligned): {op}", flush=True); continue
            print(f"  scoring {bp}", flush=True); print("   ", json.dumps(score_bands(bp, cd, op)), flush=True)
        print("SCORE DONE"); return
    if a.build:
        frozen, report = build(a)
        os.makedirs(os.path.dirname(a.report), exist_ok=True)
        with open(a.report, "w", encoding="utf-8") as f: json.dump(report, f, indent=1, default=str)
        print(f"  report {a.report}")
        print(json.dumps({k: frozen[k] for k in ("crop_threshold", "min_gated_crops", "frac_above")}),
              json.dumps(frozen["provenance"]["crop_threshold"]), json.dumps(frozen["provenance"]["min_gated_crops"]), sep="\n")
        if a.dry: print("  --dry: frozen JSON NOT written"); return
        st = write_frozen(a.out, frozen); print(f"  {VERSION}: {st} {a.out} sha256 {sha256_file(a.out)}"); return
    if a.accept:
        rep = accept(a)
        with open(a.accept_report, "w", encoding="utf-8") as f: json.dump(rep, f, indent=1, default=str)
        print(f"  accept report {a.accept_report}"); return
    ap.error("one of --selftest / --score / --score_train / --score_v5 / --build / --accept")

if __name__ == "__main__":
    main()
