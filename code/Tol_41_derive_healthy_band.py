# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N41 — derive the healthy-area band from the TRAINING days (closes the test-day leak)
# MAGIC
# MAGIC The data-split rules forbid deriving any training constant from a test day, and this rule
# MAGIC was **violated**: `Tol_39_label_pool_survey.py:117-118` hardcodes a healthy band of
# MAGIC `[0.45, 1.60] × day-ref`, and both that band and the "real calf = 43,000–90,000 px" figure were
# MAGIC fitted on **04-22/04-23 — the test days**. Those constants would otherwise define which frames enter
# MAGIC the detector's TRAINING set, which makes them trained parameters fitted on the test set.
# MAGIC
# MAGIC `Tol_40` already computes `day_ref` per calendar day from the training days themselves, so the
# MAGIC *reference* is clean. Only the **multipliers** are inherited. This notebook re-derives them from
# MAGIC 04-18/04-19 alone.
# MAGIC
# MAGIC ## The circularity problem, and how the anchor set avoids it
# MAGIC
# MAGIC Fitting a band to the same masks it will filter learns the tracker's *error* distribution and then
# MAGIC blesses it. So the band is derived only from an **anchor set** selected by criteria that do not
# MAGIC presuppose any per-mask area band:
# MAGIC
# MAGIC 1. **Exactly 4 masks.** The pen holds exactly four calves — an exogenous, label-free prior.
# MAGIC 2. **Mutually disjoint** (all pairwise IoS < `max_ios`). Purely an overlap test; area-independent.
# MAGIC 3. **Total masked area ≈ 4 × day_ref** (within `sum_tol`). This is the key non-circular step: it
# MAGIC    constrains the AGGREGATE, which is insensitive to how the total is divided, so it selects frames
# MAGIC    where no animal is missing without asserting anything about individual masks.
# MAGIC 4. **Temporally stable** — each slot's area CV below `cv_max` over ±`cv_win` s. About consistency,
# MAGIC    not magnitude, so it is not circular with the band either.
# MAGIC
# MAGIC The per-mask area distribution **within** that anchor set is then what a healthy calf mask actually
# MAGIC looks like on these two days, and the band is read off its percentiles.
# MAGIC
# MAGIC **Honest limit, reported not hidden:** criteria 2–4 cannot exclude a *persistent* merge-plus-fragment
# MAGIC pair whose areas happen to sum correctly (two calves lying together for minutes would be stable). The
# MAGIC notebook therefore reports the anchor distribution's shape — a clean unimodal distribution is evidence
# MAGIC the anchor set is uncontaminated; visible bimodality means it is not, and the band must not be trusted.
# MAGIC
# MAGIC Two independent cross-checks: the band is derived **separately for 04-18 and 04-19** (they should
# MAGIC agree), and the result is compared against the leaked `[0.45, 1.60]` so the size of the leak's actual
# MAGIC effect is known rather than assumed.
# MAGIC
# MAGIC Pure RLE arithmetic. Measures only; writes one JSON that the export will read.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, json, gzip, glob, re, time
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

dbutils.widgets.text("track_root", "Tracking_train")
dbutils.widgets.text("stride", "5")          # 5 fps rows -> stride 5 = 1 Hz
dbutils.widgets.text("max_ios", "0.05")      # anchor: pairwise IoS must be below this
dbutils.widgets.text("sum_tol", "0.10")      # anchor: |sum(areas)/(4*day_ref) - 1| <= sum_tol
dbutils.widgets.text("cv_win", "30")         # anchor: temporal stability half-window, seconds
dbutils.widgets.text("cv_max", "0.10,0.25,0.50,9.99")  # SWEEP: last value = stability off
dbutils.widgets.text("cover", "0.98")        # band covers this fraction of the anchor distribution

VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
ROOT = dbutils.widgets.get("track_root").strip()
STRIDE = max(1, int(dbutils.widgets.get("stride") or 5))
MAX_IOS = float(dbutils.widgets.get("max_ios") or 0.05)
SUM_TOL = float(dbutils.widgets.get("sum_tol") or 0.10)
CV_WIN = float(dbutils.widgets.get("cv_win") or 30)
CV_SWEEP = [float(x) for x in dbutils.widgets.get("cv_max").split(",") if x.strip()]
CV_MAX = CV_SWEEP[0]
COVER = float(dbutils.widgets.get("cover") or 0.98)
N_CALVES = 4
LEAKED = (0.45, 1.60)                        # the test-day-derived band being replaced

TRAIN_DAYS = {"20240418", "20240419"}        # hard gate: this notebook must never read a test day
print(f"root {ROOT} | anchors: 4 masks, IoS<{MAX_IOS}, sum within +/-{SUM_TOL:.0%}, "
      f"CV sweep {CV_SWEEP} over +/-{CV_WIN:.0f}s | band covers {COVER:.0%}")

# COMMAND ----------

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

RXT = re.compile(r"-(\d{8})-(\d{6})-")
def clip_start(v):
    m = RXT.search(v.replace("_", "-"))
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

def phase_of(w):
    if w.hour >= 21 or w.hour < 6: return "night"
    if w.hour >= 20: return "dusk"
    return "day"

WIN = sorted(d for d in os.listdir(osp.join(TOL, ROOT))
             if re.match(r"^\d{8}(_w\d{4}-\d{4})?$", d))
BYDAY = defaultdict(list)
for w in WIN: BYDAY[w[:8]].append(w)

# INV: a training constant may not see a test day. Fail loudly rather than quietly filtering.
intruders = set(BYDAY) - TRAIN_DAYS
assert not intruders, (f"{ROOT} contains non-training days {sorted(intruders)} — deriving the band "
                       f"from these would repeat the very leak this notebook exists to close")
print(f"{len(WIN)} window dirs, {len(BYDAY)} calendar days: {sorted(BYDAY)}  (training days only ✓)")

# COMMAND ----------

ROWS = defaultdict(list)
t0 = time.time()
for day, wins in BYDAY.items():
    for w in wins:
        for gz in sorted(glob.glob(osp.join(TOL, ROOT, w, "masks_*.json.gz"))):
            with gzip.open(gz, "rt") as f: d = json.load(f)
            w0 = clip_start(d["clip"])
            for r in d["frames"][::STRIDE]:
                wall = w0 + timedelta(seconds=r["t_sec"])
                pres = []
                for s in range(N_CALVES):
                    rl = (r.get("masks") or {}).get(str(s))
                    if not rl: continue
                    pres.append((s, runs_of(rl)))
                worst = 0.0
                for a in range(len(pres)):
                    for b in range(a + 1, len(pres)):
                        it = inter(pres[a][1], pres[b][1])
                        mn = min(pres[a][1][2], pres[b][1][2])
                        if mn: worst = max(worst, it / mn)
                ROWS[day].append(dict(window=w, t=wall.timestamp(), phase=phase_of(wall),
                                      n=len(pres), worst_ios=round(worst, 4),
                                      area={str(s): ru[2] for s, ru in pres}))
            del d
    print(f"  {day}: {len(ROWS[day])} sampled frames ({(time.time()-t0)/60:.1f} min)")

# COMMAND ----------

# MAGIC %md ## Anchor selection, then the band

# COMMAND ----------

def pct(v, q): return float(np.percentile(v, q)) if len(v) else float("nan")

RESULT = {}
for day, rows in ROWS.items():
    rows.sort(key=lambda r: r["t"])
    # pass 1 -- provisional reference: median LARGEST daylight mask (robust to fragments,
    # which can only pull a mask DOWN, never make it the largest)
    dayl = [max(r["area"].values()) for r in rows if r["phase"] == "day" and r["area"]]
    ref0 = float(np.median(dayl))

    # temporal stability per (window, slot): area CV over +/- CV_WIN seconds
    stable = {}
    for w in {r["window"] for r in rows}:
        sub = [r for r in rows if r["window"] == w]
        ts = np.array([r["t"] for r in sub])
        for s in range(N_CALVES):
            a = np.array([r["area"].get(str(s), np.nan) for r in sub], float)
            for i in range(len(sub)):
                lo, hi = np.searchsorted(ts, (ts[i] - CV_WIN, ts[i] + CV_WIN))
                seg = a[lo:hi][~np.isnan(a[lo:hi])]
                cv = (seg.std() / seg.mean()) if len(seg) >= 3 and seg.mean() > 0 else np.inf
                stable[(w, s, i)] = cv
    idx_in_win = {}
    for w in {r["window"] for r in rows}:
        for i, r in enumerate([x for x in rows if x["window"] == w]):
            idx_in_win[id(r)] = i

    base = []
    for r in rows:
        if r["n"] != N_CALVES or r["worst_ios"] >= MAX_IOS: continue
        tot = sum(r["area"].values())
        if abs(tot / (N_CALVES * ref0) - 1.0) > SUM_TOL: continue
        base.append(r)

    SWEEP = {}
    for cvt in CV_SWEEP:
        sel = [r for r in base
               if all(stable.get((r["window"], s, idx_in_win[id(r)]), np.inf) < cvt
                      for s in range(N_CALVES))]
        rr = np.array([a / ref0 for r in sel for a in r["area"].values()])
        if len(rr) < 200:
            SWEEP[cvt] = dict(anchors=len(sel), band=None, note="too few anchors to fit a band")
            continue
        SWEEP[cvt] = dict(anchors=len(sel),
                          band=(round(float(np.percentile(rr, (1-COVER)/2*100)), 3),
                                round(float(np.percentile(rr, (1+COVER)/2*100)), 3)),
                          median=round(float(np.median(rr)), 3))
    print(f"  {day} band vs stability pressure (selecting on low CV then measuring spread biases NARROW):")
    for cvt in CV_SWEEP:
        v = SWEEP[cvt]
        lab = "off" if cvt > 9 else f"{cvt:.2f}"
        print(f"     CV<{lab:>4s}: {v['anchors']:6,d} anchors  band "
              + (f"[{v['band'][0]}, {v['band'][1]}]  median {v['median']}" if v["band"] else v["note"]))
    anchors = [r for r in base
               if all(stable.get((r["window"], s, idx_in_win[id(r)]), np.inf) < CV_SWEEP[-1]
                      for s in range(N_CALVES))]

    ratios = np.array([a / ref0 for r in anchors for a in r["area"].values()])
    ref1 = float(np.median([a for r in anchors for a in r["area"].values()])) if anchors else float("nan")
    lo_q, hi_q = (1 - COVER) / 2 * 100, (1 + COVER) / 2 * 100
    band = (round(pct(ratios, lo_q), 3), round(pct(ratios, hi_q), 3))

    # shape check -- a clean unimodal anchor distribution is the evidence that the anchor set is
    # uncontaminated; a second mode near ~0.5 or ~2.0 would be fragments or merges leaking in
    hist, edges = np.histogram(ratios, bins=np.arange(0.0, 2.55, 0.1))
    peak = int(np.argmax(hist))
    # a second MODE needs a trough between it and the peak; broad spread alone is not bimodality
    side = []
    for i in range(len(hist)):
        if abs(i - peak) < 3 or hist[i] <= 0.25 * hist[peak]: continue
        a, b = (i, peak) if i < peak else (peak, i)
        if min(hist[a + 1:b]) < 0.5 * min(hist[i], hist[peak]): side.append(i)

    RESULT[day] = dict(cv_sweep={str(k): v for k, v in SWEEP.items()},
        frames=len(rows), anchors=len(anchors),
        anchor_pct=round(100.0 * len(anchors) / max(len(rows), 1), 1),
        ref_pass1_largest_daylight=int(ref0), ref_pass2_anchor_median=int(ref1),
        ref_shift=round(ref1 / ref0, 3),
        band=band,
        pctiles={q: round(pct(ratios, q), 3) for q in (0.1, 0.5, 1, 2.5, 5, 25, 50, 75, 95, 97.5, 99, 99.5, 99.9)},
        hist=[int(x) for x in hist], hist_edges=[round(float(e), 2) for e in edges],
        secondary_modes=[round(float(edges[i]), 2) for i in side],
        unimodal=not side)
    print(f"{day}: {len(anchors):,}/{len(rows):,} anchor frames ({RESULT[day]['anchor_pct']} %)  "
          f"ref {int(ref0):,} -> {int(ref1):,} px (x{RESULT[day]['ref_shift']})")
    print(f"    derived band at {COVER:.0%} coverage: [{band[0]}, {band[1]}]   "
          f"unimodal: {RESULT[day]['unimodal']}" +
          ("" if RESULT[day]['unimodal'] else f"  ⚠ secondary mass at {RESULT[day]['secondary_modes']}"))

# COMMAND ----------

# MAGIC %md ## Agreement, the leak's size, and the yield trade

# COMMAND ----------

days = sorted(RESULT)
lo = float(np.mean([RESULT[d]["band"][0] for d in days]))
hi = float(np.mean([RESULT[d]["band"][1] for d in days]))
# round outward so the pooled band is never tighter than either day's
FINAL = (round(np.floor(lo * 100) / 100, 2), round(np.ceil(hi * 100) / 100, 2))

print("per-day derived bands")
for d in days: print(f"  {d}: {RESULT[d]['band']}")
if len(days) == 2:
    d0, d1 = days
    dlo = abs(RESULT[d0]["band"][0] - RESULT[d1]["band"][0])
    dhi = abs(RESULT[d0]["band"][1] - RESULT[d1]["band"][1])
    print(f"  cross-day disagreement: lo {dlo:.3f}, hi {dhi:.3f} "
          f"({'consistent' if max(dlo, dhi) < 0.15 else 'DIVERGENT — do not pool'})")
print(f"\nPOOLED TRAINING-DERIVED BAND: [{FINAL[0]}, {FINAL[1]}]")
print(f"LEAKED TEST-DERIVED BAND:     [{LEAKED[0]}, {LEAKED[1]}]")
print(f"  shift: lo {FINAL[0]-LEAKED[0]:+.2f}, hi {FINAL[1]-LEAKED[1]:+.2f}")

def yield_of(band):
    out = {}
    for day, rows in ROWS.items():
        ref = RESULT[day]["ref_pass1_largest_daylight"]
        for ph in ("day", "dusk", "night"):
            sel = [r for r in rows if r["phase"] == ph]
            ok = [r for r in sel if r["n"] == N_CALVES and r["worst_ios"] < 0.10
                  and all(band[0] * ref <= a <= band[1] * ref for a in r["area"].values())]
            out[f"{day}/{ph}"] = (len(ok), round(100.0 * len(ok) / max(len(sel), 1), 1))
    return out

yd, yl = yield_of(FINAL), yield_of(LEAKED)
print(f"\n{'stratum':18s} {'derived':>16s} {'leaked':>16s} {'delta pp':>9s}")
print("-" * 62)
for k in sorted(yd):
    print(f"{k:18s} {yd[k][0]:8,d} {yd[k][1]:6.1f}% {yl[k][0]:8,d} {yl[k][1]:6.1f}% {yd[k][1]-yl[k][1]:+8.1f}")

out = {"derived_from": sorted(BYDAY), "track_root": ROOT,
       "anchor_rule": {"n_masks": N_CALVES, "max_ios": MAX_IOS, "sum_tol": SUM_TOL,
                       "cv_win_s": CV_WIN, "cv_max": CV_MAX, "coverage": COVER},
       "per_day": RESULT, "band": {"frag_lo": FINAL[0], "over_hi": FINAL[1]},
       "superseded_band": {"frag_lo": LEAKED[0], "over_hi": LEAKED[1],
                           "why": "fitted on 04-22/04-23, which are TEST days (DATA_SPLITS.md F2)"},
       "yield_derived": yd, "yield_leaked": yl}
p = osp.join(TOL, "Metrics_healthy_band_train.json")
with open(p, "w") as f: json.dump(out, f, indent=1)
print(f"\nwritten {p}")
print("The export must read frag_lo/over_hi from THIS file, not from a literal in a notebook.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading it
# MAGIC
# MAGIC **`unimodal` must be true on both days.** It is the evidence that the anchor set is uncontaminated.
# MAGIC A secondary mode near ~0.5 means fragments survived the anchor rule; near ~2.0 means merges did.
# MAGIC Either way the band is then fitted to the tracker's errors and must not be used.
# MAGIC
# MAGIC **The two days must agree.** They are independent derivations. Disagreement above ~0.15 in either
# MAGIC edge means the anchor rule is picking up something day-specific and pooling them would hide it.
# MAGIC
# MAGIC **`ref_shift` should be near 1.0.** Pass 1 anchors on the median largest daylight mask; pass 2 on the
# MAGIC median of all four masks in anchor frames. If these disagree materially, the "largest mask" proxy is
# MAGIC biased on these days and the reference itself needs revisiting.
# MAGIC
# MAGIC **The yield table quantifies what the leak actually cost.** If the derived band admits nearly the same
# MAGIC frames as the leaked one, the violation was real but harmless in effect — worth stating plainly rather
# MAGIC than either hiding it or overstating it. If it differs materially, every yield number computed with
# MAGIC the old band (including `Tol_40`'s) has to be recomputed.
