"""GT-free PRODUCTION identity relabel for the v5 watch renders.

Runs Tol_30 in WEEK mode (accumulating exemplar pool, global identities anchored on 04-22's day-start seeds) on a LOCAL
copy of a v5 production root that holds NO ground truth, validates the output, and (only with --upload) writes

    Tol/Metadata/relabel_<slug>/relabel_week_<day>.json          <- the ONLY file the L5/L6/L7 render payloads read
    Tol/Metadata/relabel_<slug>/runs/<tag>/{provenance.json, periodic_identity_WEEK.json, tol30_week.log,
                                            relabel_week_<every day of the run>.json}

    python prod_relabel_v5.py run --root Tracking_v5a10 --days 20240422,20240423 --render-days 20240422,20240423 \\
        [--tol30 <Tol_30 path>] [--tol30-sha <hex or prefix>] [--work <dir>] \\
        --require-gate <G3_verdict.json> [--require-gate <G5_verdict_<day>.json> ...] [--upload] [--allow-overwrite]
    python prod_relabel_v5.py selftest [--data <dir holding 20240422/ 20240423/ Tol_30 inputs>] [--full]
        (FakeRemote, no network; one REAL week-mode run on 04-22 ~10 min, --full = 04-22+04-23 ~50 min)

WHY THIS IS GT-FREE (every guard is checked, not assumed):
  * the local copy is built from a WHITELIST: segments.json, events.jsonl, observations_meta.json,
    crop_bands3_dense.npz (+ ROOT_CONFIG.json); anything else found in the copy aborts the run;
  * the remote day dir is refused if it holds any GT-like or identity file (segment_assignments*, *ground_truth*,
    gt_*, week_ident_map*, relabel_*) -- that would also mean Tol_15_v2 ran on a v5 root;
  * Tol_30 is called WITHOUT --gt, --week_map or --links; week mode reads GT only for scoring, from
    <root>/<day>/segment_assignments*.json, which the whitelist excludes -- and the run must print "<day>: no GT" for
    every day, else it aborts;
  * Tol_30 runs from a sha-pinned SNAPSHOT (Tol_30 is patched concurrently by another workflow) and the snapshot must be
    unchanged afterwards.
  * --identity B60 (since 2026-09-30) adds Tol_30 --noncalf with the frozen COLOUR json (noncalf_colour_v1.json, Tol_54,
    derived from 04-18/19 only, sha-pinned). Its per-crop statistic crop_colour_v1.npz is NOT GT: it is a pure function of
    the root's own crops_dense JPEGs (fraction of clearly coloured foreground pixels). It is read from the Volume day dir if
    present (row alignment with the bands + COLOUR_SPOT rows recomputed bit-equal), else COMPUTED here: every crop the bands
    npz references is downloaded from <root>/<day>/crops_dense (read-only; a local cache is reused only where every file's
    size equals the Volume listing) and scored by a sha-pinned SNAPSHOT of Tol_54 (statistic functions == the frozen json's,
    constants == its statistic block). The npz joins the whitelisted local copy; sha256, source, crop manifest and tool
    pins go to provenance.colour.
GATED: --upload needs --require-gate. Every verdict must be accepted by
gate_v5.verdict_accepted (PASS, or PASS_WITH_REVIEW acknowledged with `gate_v5.py ack`) for THIS root; a G3 verdict must
be among them and its `tol30_sha` must equal this run's Tol_30 snapshot (ONE Tol_30 revision for the scoring, night-gap
and production chains); every --render-days day must be covered: G3 covers 20240422/23, a G5 verdict its `day` (04-24:
g5 includes Tol_48 --g5a; 04-25..28: the operational gate).
Days must be a PREFIX of the test week (20240422, 20240423, ...): the week-mode identity space is anchored on the first
day, so a run that started later would re-anchor the colours. An already-uploaded relabel is never overwritten
without --allow-overwrite (a render that may already have been reviewed); relabels of earlier days that this run would change are
reported in provenance.watched_days_changed.
"""
import os, sys, re, json, time, shutil, hashlib, argparse, subprocess, tempfile, importlib.util, ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v5_remote  # noqa: E402

NB = "<LOCAL_CODE_DIR>"
LAUNCHER = NB + "/Tol_47b_launcher.py"
TOOL = ("prod_relabel_v5 6.2 (2026-09-30, --identity B60 = B60 + --noncalf COLOUR v1 | B60nc = B60 + --noncalf anchor v1, "
        "dry-run only)")
BANDS = "crop_bands3_dense.npz"                     # = the L5/L6/L7 downstream Tol_28 out_name
NEED = ("segments.json", "events.jsonl", "observations_meta.json", BANDS)
COLOUR_NPZ = "crop_colour_v1.npz"                   # Tol_54 per-crop colour statistic, row-aligned with BANDS (not GT)
WHITELIST = frozenset(NEED) | {"ROOT_CONFIG.json", COLOUR_NPZ}
GT_RX = re.compile(r"segment_assignments|ground_truth|(^|_)gt[_.]|week_ident_map|relabel_", re.I)
T_MIN = 3                                           # Tol_30 week mode --T 3
N_ID = 4
# Identity-layer configuration (decision 2026-09-28: "keep B60"): s = 0.6 * anchor + 0.4 * even temporal pool, no seed
# bootstrap; anchors = the frozen 04-18/19 pool, correspondence = pi_final (specialist blind key == B1 appearance, B3 ACCEPT).
# "B60" = Tol_52's B60 arm (Tol_52 ARMS["B60"]) + Tol_30 --noncalf with the frozen COLOUR json (decision 2026-09-30:
# "exclude colour except for white/black should be a strong filter"; Tol_54, derived from 04-18/19 only, sha-pinned below;
# its 04-22/23 acceptance hid exactly the 04-23 farm worker, v5 seg 1_012, and nothing else — colorfilter/tol54_accept.json).
# "production" = plain Tol_30 week mode (no extra flags).
# "B60nc" = B60's base arm + Tol_30 --noncalf with the frozen Tol_53 ANCHOR threshold json (derived from 04-18/19 only,
# sha-pinned below; decision 2026-09-29). REVIEW 2026-09-29: v1 FAILED its 04-22/23 acceptance (it hides nothing; the
# worker's anchor scores overlap real calves'), so it may not be uploaded (IDENTITY_UNACCEPTED): dry runs only.
ANCHOR_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "train_gt", "anchor_pool")
ANCHOR_POOL = os.path.join(ANCHOR_DIR, "anchor_pool_v1.npz")
ANCHOR_PI = os.path.join(ANCHOR_DIR, "pi_final.json")
ANCHOR_POOL_SHA = "303be22cdd9f09f162d2bff5eb6a7027bd9114e389ce11d2592f27e9895c70a9"
NONCALF_JSON = os.path.join(ANCHOR_DIR, "noncalf_threshold_v1.json")           # Tol_53_noncalf_threshold.py --build
NONCALF_SHA = "c480bcdd1ff09299c77ea7d4ce4bab9cbe643627d5028e36d9f59ccb37f02a3a"
COLOUR_JSON = os.path.join(ANCHOR_DIR, "noncalf_colour_v1.json")               # Tol_54_crop_colour.py --build
COLOUR_SHA = "9a84b864ea20a3cc225079af93b93b3d68fde8bed1b253956566b7564e84fefd"
TOL54 = NB + "/Tol_54_crop_colour.py"
# Tol_54's compute path beyond the three statistic functions the frozen json pins itself (crop_colour, foreground,
# hsv_arrays): function-level sha256 (Tol_54.fn_sha) of the file that built the json (7a7d23d8...); unrelated edits to
# Tol_54 (build / accept / selftest) stay allowed, a change anywhere on the scoring path is refused.
TOL54_FN_SHA = {"read_rgb": "612b3d07e78122628ffce1dab0000c54e403809d599983632daa80fec358846f",
                "local_crop_path": "0dfdced603b741155ad12e73a39ef73b1e7fc0622c4d80f32e9b935663670c62",
                "score_bands": "f93655c3a3326d7bc8dee011b21e9c87a59febfbf245b75abff04bca9f19a23f",
                "load_colour": "8e7b2568562dcab48d78e173eb653e3a4812e39eae32cf6ddd40cc0702f020ec",
                "save_npz_deterministic": "efbb7f76efb16fc7274be50cba8eae20a322915d7de3c711fd8930132c1b0b7e"}
TOL54_CONSTS = ("S_MIN", "V_MIN", "BG_TOL", "HOLE_MIN", "EDGE_ERODE", "MIN_FG")   # == the frozen json's statistic block
TOL54_PATH_TAIL_RX = r"/crops_dense/([^/]+)/(\d{6})\.jpg$"                         # Tol_54.PATH_TAIL_RX (7a7d23d8...)
COLOUR_SPOT = 16                                    # rows of a Volume-provided colour npz recomputed here, bit-equal
COLOUR_DL_WORKERS = 6                               # parallel per-segment crops_dense copies (read-only)
TEST_WEEK = tuple(f"202404{d}" for d in range(22, 29))
IDENTITY_CONFIGS = {"production": None,
                    "B60": {"anchor_w": "0.6", "temporal": "even", "boot": "none", "noncalf": "colour"},
                    "B60nc": {"anchor_w": "0.6", "temporal": "even", "boot": "none", "noncalf": "anchor"}}
IDENTITY_UNACCEPTED = {"B60nc": "Tol_30 --noncalf v1 (noncalf_threshold_v1) FAILED its 04-22/23 acceptance on 2026-09-29: it hid "
                                "nothing, the 04-23 worker track 1_012 included (noncalf/acceptance_report.json)"}

def identity_flags(name):
    """Tol_30 extension flags + provenance for an identity config; refuses unless the pool/pi are the frozen, final ones."""
    cfg = IDENTITY_CONFIGS[name]
    if cfg is None: return [], {"name": name, "flags": []}
    for p in (ANCHOR_POOL, ANCHOR_PI):
        if not os.path.exists(p): raise Refused(f"identity {name}: {p} missing")
    ps = sha256_file(ANCHOR_POOL)
    if ps != ANCHOR_POOL_SHA: raise Refused(f"identity {name}: anchor pool sha {ps[:16]} is not the frozen pool {ANCHOR_POOL_SHA[:16]}")
    pj = json.load(open(ANCHOR_PI, encoding="utf-8"))
    if pj.get("status") != "final" or pj.get("anchor_sha256") != ANCHOR_POOL_SHA:
        raise Refused(f"identity {name}: {ANCHOR_PI} is not a final pi for the frozen pool (status {pj.get('status')})")
    flags = ["--anchors", ANCHOR_POOL, "--anchor_pi", ANCHOR_PI, "--anchor_w", cfg["anchor_w"],
             "--temporal", cfg["temporal"], "--boot", cfg["boot"]]
    prov = {"name": name, "flags": flags, "anchor_pool_sha256": ps, "pi_sha256": sha256_file(ANCHOR_PI),
            "pi": pj.get("pi"), "note": "anchors from 04-18/19 only (frozen); pi = specialist blind coat key (2026-09-27), "
            "equal to the sealed B1 appearance check; no test-day GT enters the run"}
    if cfg.get("noncalf") == "colour":
        if not os.path.exists(COLOUR_JSON): raise Refused(f"identity {name}: {COLOUR_JSON} missing (Tol_54 --build)")
        cs = sha256_file(COLOUR_JSON)
        if cs != COLOUR_SHA:
            raise Refused(f"identity {name}: colour non-calf json sha {cs[:16]} is not the pinned {COLOUR_SHA[:16]} (frozen Tol_54 output)")
        cj = json.load(open(COLOUR_JSON, encoding="utf-8"))
        st = cj.get("statistic") or {}
        if sorted(cj.get("derived_from") or ["?"]) != ["20240418", "20240419"] or not set(TEST_WEEK) <= set(cj.get("not_from") or []):
            raise Refused(f"identity {name}: {COLOUR_JSON} derived_from {cj.get('derived_from')} / not_from {cj.get('not_from')} "
                          f"(training days only)")
        if st.get("name") != "coloured_frac" or st.get("npz") != COLOUR_NPZ:
            raise Refused(f"identity {name}: {COLOUR_JSON} statistic {st.get('name')!r} / npz {st.get('npz')!r}")
        flags += ["--noncalf", COLOUR_JSON]
        prov["noncalf"] = {"mode": "colour", "file": COLOUR_JSON, "sha256": cs, "version": cj.get("version"),
                           "derived_from": cj.get("derived_from"), "not_from": cj.get("not_from"),
                           **{k: cj[k] for k in ("crop_threshold", "min_gated_crops", "frac_above")},
                           "statistic": {k: st.get(k) for k in ("name", "npz", "S_MIN", "V_MIN", "FILL", "BG_TOL", "HOLE_MIN",
                                                                  "EDGE_ERODE", "MIN_FG", "crop_colour_sha256", "foreground_sha256",
                                                                  "hsv_arrays_sha256")},
                           "built_by_tol54_sha256": (cj.get("tools") or {}).get("tol54", {}).get("sha256"),
                           "semantics": "segments whose gated crops are clearly COLOURED (coloured_frac > crop_threshold for >= "
                                        "frac_above of >= min_gated_crops gated crops) are not one of the four black-and-white "
                                        "calves: no identity, listed under the relabel's 'hidden' (Tol_17 draws nothing there)"}
    elif cfg.get("noncalf") == "anchor":
        if not os.path.exists(NONCALF_JSON): raise Refused(f"identity {name}: {NONCALF_JSON} missing (Tol_53 --build)")
        ns = sha256_file(NONCALF_JSON)
        if ns != NONCALF_SHA:
            raise Refused(f"identity {name}: noncalf threshold sha {ns[:16]} is not the pinned {NONCALF_SHA[:16]} (frozen Tol_53 output)")
        nj = json.load(open(NONCALF_JSON, encoding="utf-8"))
        if nj.get("anchor_pool", {}).get("sha256") != ANCHOR_POOL_SHA:
            raise Refused(f"identity {name}: {NONCALF_JSON} was derived for another anchor pool")
        if sorted(nj.get("derived_from") or ["?"]) != ["20240418", "20240419"]:
            raise Refused(f"identity {name}: {NONCALF_JSON} derived_from {nj.get('derived_from')} (training days only)")
        flags += ["--noncalf", NONCALF_JSON]
        prov["noncalf"] = {"mode": "anchor", "file": NONCALF_JSON, "sha256": ns, "version": nj.get("version"),
                           "derived_from": nj.get("derived_from"),
                           **{k: nj[k] for k in ("crop_threshold", "min_gated_crops", "frac_below")},
                           "semantics": "segments whose gated crops look like none of the four calves get no identity and are "
                                        "listed under the relabel's 'hidden' (Tol_17 draws nothing for that slot there)"}
    return flags, prov

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

class Refused(RuntimeError): pass

FAKE_T30 = r"""import sys, os, json, shutil
a = sys.argv; days = a[a.index("--days") + 1].split(","); out = a[a.index("--out_dir") + 1]
assert "--gt" not in a and "--week_map" not in a and "--links" not in a
for d in days:
    shutil.copyfile(os.path.join("@SRC@", f"relabel_week_{d}.json"), os.path.join(out, f"relabel_week_{d}.json"))
    print(f"  {d}: no GT")
json.dump({"days": days, "runs": {}}, open(os.path.join(out, "periodic_identity_WEEK.json"), "w"))
"""

# ---------------------------------------------------------------------------------------------------------------------
def check_days(days, render_days, expected_clips):
    week = sorted(expected_clips)
    if not days or days != week[:len(days)]:
        raise Refused(f"--days {days} must be a prefix of the test week {week} (identities are anchored on {week[0]})")
    bad = [d for d in render_days if d not in days]
    if not render_days or bad: raise Refused(f"--render-days {render_days} must be a non-empty subset of --days")

def local_guard(root_dir, days):
    """Whitelist: the local Tol_30 input copy may hold ONLY the Tol_30 inputs (+ ROOT_CONFIG.json)."""
    extra = []
    for dp, dn, fn in os.walk(root_dir):
        rel = os.path.relpath(dp, root_dir).replace("\\", "/")
        if rel == ".":
            extra += [f for f in fn]; extra += [d + "/" for d in dn if d not in days]; continue
        if rel.count("/") > 0 or rel not in days:
            extra += [f"{rel}/{x}" for x in fn + dn]; continue
        extra += [f"{rel}/{f}" for f in fn if f not in WHITELIST] + [f"{rel}/{d}/" for d in dn]
    if extra: raise Refused(f"local copy holds files outside the Tol_30-input whitelist (GT must never be visible): {extra}")

def remote_guard(listing, day):
    names = [e["name"] for e in (listing or [])]
    bad = [n for n in names if GT_RX.search(n)]
    if bad:
        raise Refused(f"remote day dir {day} holds GT-like / identity files {bad} (Tol_15_v2 must never run on a v5 "
                      f"root) — refusing")
    return names

HIDDEN_KEYS = {"anchor": ("score_median",), "colour": ("n_gated", "n_above", "frac_above", "cf_median")}

def validate_hidden(J, day, rules_spans, mode=None):
    """the --noncalf "hidden" list: [{slot, from, to, seg, why: "noncalf", ...}] — anchor entries carry score_median (no
    "mode" key), colour entries "mode": "colour" + n_gated / n_above / frac_above / cf_median. mode (the filter this identity
    config runs, None = either) must match every entry. A hidden track is not an animal, so no identity rule of the SAME
    slot may overlap it (> 1 s). Returns (stats, problems, warnings)."""
    probs, warns, hid = [], [], J.get("hidden")
    if not isinstance(hid, list): return {}, [f"'hidden' is {type(hid).__name__}, not a list"], warns
    d0 = datetime.strptime(day, "%Y%m%d"); tot, out = 0.0, []
    for j, h in enumerate(hid):
        hm = h.get("mode", "anchor") if isinstance(h, dict) else None
        if not isinstance(h, dict) or hm not in HIDDEN_KEYS or not {"slot", "from", "to", "seg", "why", *HIDDEN_KEYS[hm]} <= set(h):
            probs.append(f"hidden[{j}]: needs slot/from/to/seg/why + {'/'.join(HIDDEN_KEYS.get(hm) or ('mode anchor|colour',))}, "
                         f"got {h!r}"); continue
        if mode is not None and hm != mode:
            probs.append(f"hidden[{j}]: a {hm}-filter entry, but this identity config runs the {mode} filter"); continue
        try:
            a, b, sl = datetime.fromisoformat(h["from"]), datetime.fromisoformat(h["to"]), int(h["slot"])
            float(h["cf_median" if hm == "colour" else "score_median"])
        except Exception as e:
            probs.append(f"hidden[{j}]: bad entry {h!r} ({e})"); continue
        if sl not in range(N_ID): probs.append(f"hidden[{j}]: slot {sl} outside 0..{N_ID - 1}")
        if h["why"] != "noncalf": probs.append(f"hidden[{j}]: why {h['why']!r} (only 'noncalf' is defined)")
        if a > b: probs.append(f"hidden[{j}]: inverted interval {h['from']}..{h['to']}")
        if not (d0 <= a < d0 + timedelta(days=1, hours=1)): probs.append(f"hidden[{j}]: {h['from']} is not on {day}")
        for ra, rb, lab, rs in rules_spans:
            if int(rs) == sl and (min(b, rb) - max(a, ra)).total_seconds() > 1.0:
                probs.append(f"hidden[{j}] (slot {sl} {h['from']}..{h['to']}) overlaps identity rule {ra.isoformat()}..{rb.isoformat()} "
                             f"-> {lab}: a hidden track must not carry an identity")
        tot += max(0.0, (b - a).total_seconds())
        out.append({"slot": sl, "from": h["from"], "to": h["to"], "seg": h["seg"],
                    **({"mode": "colour"} if hm == "colour" else {}), **{k: h[k] for k in HIDDEN_KEYS[hm]}})
    return {"n_hidden": len(hid), "hidden_slot_h": round(tot / 3600, 4), "hidden": out}, probs, warns

def validate_relabel(path, day, require_hidden=False, mode=None):
    """Structure + sanity of one Tol_30 week-mode relabel file. Returns (stats, problems, warnings). require_hidden (identity
    configs with --noncalf): the file must carry the "hidden" list (proof that the filter ran), valid per validate_hidden
    (mode: the filter the config runs, "anchor" | "colour"; None = either)."""
    J = json.load(open(path, encoding="utf-8"))
    probs, warns = [], []
    if J.get("day") != day: probs.append(f"day field {J.get('day')!r} != {day}")
    rules = J.get("rules")
    if not isinstance(rules, dict): return {}, probs + ["no rules dict"], warns
    d0 = datetime.strptime(day, "%Y%m%d")
    per_label, spans, n = {}, [], 0
    for slot, rl in rules.items():
        if slot not in {str(i) for i in range(N_ID)}: probs.append(f"slot key {slot!r}"); continue
        prev_end = None
        for r in rl:
            n += 1
            try:
                a, b, lab = datetime.fromisoformat(r[0]), datetime.fromisoformat(r[1]), int(r[2])
            except Exception as e:
                probs.append(f"slot {slot}: bad rule {r!r} ({e})"); continue
            if lab not in range(N_ID): probs.append(f"slot {slot}: label {lab} outside 0..{N_ID - 1}")
            if a > b: probs.append(f"slot {slot}: inverted interval {r[0]}..{r[1]}")
            elif a == b: warns.append(f"slot {slot}: zero-length rule at {r[0]} (Tol_17 never matches it: harmless)")
            if not (d0 <= a < d0 + timedelta(days=1, hours=1)): probs.append(f"slot {slot}: {r[0]} is not on {day}")
            if prev_end is not None and a < prev_end - timedelta(seconds=1):
                warns.append(f"slot {slot}: rule starting {r[0]} overlaps the previous one")
            prev_end = b if prev_end is None else max(prev_end, b)
            per_label[lab] = per_label.get(lab, 0.0) + max(0.0, (b - a).total_seconds())
            spans.append((a, b, lab, slot))
    # one identity on two slots at once (Tol_30's pen constraint should prevent it; reported, not fatal)
    dup_s, ev = 0.0, []
    for a, b, lab, slot in spans: ev += [(a, 1, lab, slot), (b, -1, lab, slot)]
    ev.sort(key=lambda x: (x[0], x[1]))
    live, last_t = {}, None
    for t, k, lab, slot in ev:
        if last_t is not None: dup_s += sum((t - last_t).total_seconds() for l_, c in live.items() if c > 1)
        live[lab] = live.get(lab, 0) + k; last_t = t
    if dup_s > 0: warns.append(f"an identity is on two slots at once for {dup_s:.0f} s in total")
    stats = {"n_rules": n, "labelled_h_by_identity": {str(k): round(v / 3600, 3) for k, v in sorted(per_label.items())},
             "labelled_slot_h": round(sum(per_label.values()) / 3600, 3), "duplicate_identity_s": round(dup_s, 1)}
    if "hidden" in J:
        hs, hp, hw = validate_hidden(J, day, spans, mode=mode)
        stats.update(hs); probs += hp; warns += hw
    elif require_hidden:
        probs.append("no 'hidden' list: this identity config runs Tol_30 --noncalf, whose relabel always carries one")
    return stats, probs, warns

def check_gates(paths, root, render_days, upload):
    """(tol30 sha the G3 verdict pins, records). Refuses unless every verdict is accepted for `root`, a G3 verdict is
    present, and every render day is covered."""
    paths = [p for p in (paths or []) if p]
    if upload and not paths:
        raise Refused("--upload needs --require-gate (a G3 verdict, plus a G5 verdict per render day after 04-23): an "
                      "ungated relabel must never reach a watch render")
    if not paths: return None, []
    import gate_v5
    cover, t30, recs = {}, None, []
    for p in paths:
        J = json.load(open(p, encoding="utf-8"))
        if J.get("gate") not in ("G3", "G5"): raise Refused(f"--require-gate {p}: gate {J.get('gate')!r} (need G3 or G5)")
        ok, why = gate_v5.verdict_accepted(J, root=root)
        if not ok: raise Refused(f"--require-gate {p}: {why} (need an accepted {J.get('gate')} for {root})")
        days = J.get("days") or (["20240422", "20240423"] if J["gate"] == "G3" else [J.get("day")])
        for d in days: cover.setdefault(d, p)
        if J["gate"] == "G3":
            if not J.get("tol30_sha"):
                raise Refused(f"--require-gate {p}: the G3 verdict records no tol30_sha (gate_v5.py g3 --tol30-sha)")
            t30 = J["tol30_sha"]
        recs.append({"path": os.path.abspath(p), "gate": J["gate"], "status": J["status"], "days": days,
                     "sha256": sha256_file(p), "review_ack": J.get("review_ack")})
    if t30 is None:
        raise Refused("no G3 verdict among --require-gate: the production relabel must use the Tol_30 revision that G3 "
                      "scored with")
    miss = [d for d in render_days if d not in cover]
    if miss: raise Refused(f"render days {miss} are not covered by an accepted G3 / G5 verdict for {root}")
    return t30, recs

def production_base(root, reg):
    """The production root `root` is, or a change-control revision of (<root>_r<N>, N >= 2; = Tol_47b_launcher.root_base);
    None otherwise."""
    if root in reg["PRODUCTION_ROOTS"]: return root
    m = re.match(reg["ROOT_REV_PATTERN"], root or "")
    return m.group(1) if m and m.group(1) in reg["PRODUCTION_ROOTS"] else None

# --------------------------------------------------------------------------------------------------------------------- colour
BAND_CROP_RX = re.compile(r"/Tol/(?P<root>[^/]+)/(?P<day>[^/]+)/crops_dense/(?P<seg>[^/]+)/(?P<hh>\d{6})\.jpg$")

def load_tol54(snap, cj):
    """import the Tol_54 SNAPSHOT and pin it: the three statistic functions == the frozen json's own function sha256s, the
    rest of the scoring path == TOL54_FN_SHA, the constants == the json's statistic block. Returns (module, pins)."""
    spec = importlib.util.spec_from_file_location("tol54_snapshot_prod_relabel", snap)
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    st = cj["statistic"]; bad = []
    want = {**{f: st.get(f + "_sha256") for f in ("crop_colour", "foreground", "hsv_arrays")}, **TOL54_FN_SHA}
    got = {f: M.fn_sha(snap, f) for f in want}
    bad += [f"{f}() sha {str(got[f])[:12]} != {str(w)[:12]}" for f, w in want.items() if got[f] != w or not w]
    bad += [f"{k} {getattr(M, k, None)!r} != {st.get(k)!r}" for k in TOL54_CONSTS if getattr(M, k, None) != st.get(k)]
    if list(getattr(M, "FILL", ())) != list(st.get("FILL") or []): bad.append(f"FILL {getattr(M, 'FILL', None)} != {st.get('FILL')}")
    if getattr(M, "NPZ_NAME", None) != COLOUR_NPZ: bad.append(f"NPZ_NAME {getattr(M, 'NPZ_NAME', None)!r} != {COLOUR_NPZ!r}")
    # review 2026-09-30: the module-level objects the pinned functions read are pinned too — _K3 (foreground()'s 8-connectivity
    # AND its erosion element) and PATH_TAIL_RX (local_crop_path()) — and every pinned name must be BOUND at run time to the
    # (first) top-level def whose source was hashed: a later re-assignment or second def of the same name would run code the
    # function sha never saw.
    import numpy as np
    k3 = getattr(M, "_K3", None)
    if not (isinstance(k3, np.ndarray) and k3.dtype == bool and k3.shape == (3, 3) and bool(k3.all())):
        bad.append(f"_K3 {k3!r} != np.ones((3, 3), bool)")
    rx = getattr(getattr(M, "PATH_TAIL_RX", None), "pattern", None)
    if rx != TOL54_PATH_TAIL_RX: bad.append(f"PATH_TAIL_RX {rx!r} != {TOL54_PATH_TAIL_RX!r}")
    first_def = {}
    for node in ast.parse(open(snap, encoding="utf-8").read()).body:
        if isinstance(node, ast.FunctionDef): first_def.setdefault(node.name, node.lineno)
    for f in want:
        co = getattr(getattr(M, f, None), "__code__", None)
        if co is None or os.path.normcase(os.path.abspath(co.co_filename)) != os.path.normcase(os.path.abspath(snap)) \
                or co.co_firstlineno != first_def.get(f):
            bad.append(f"{f}() is not bound to its hashed def (line {first_def.get(f)}; bound: "
                       f"{getattr(co, 'co_filename', None)}:{getattr(co, 'co_firstlineno', None)})")
    if bad: raise Refused("Tol_54 is not the statistic the frozen colour json was built with: " + "; ".join(bad))
    import scipy, PIL                                                   # provenance only (JPEG decode / labelling libraries)
    return M, {"fn_sha256": got, "constants": {k: getattr(M, k) for k in TOL54_CONSTS + ("FILL", "NPZ_NAME")},
               "module_objects": {"_K3": "np.ones((3, 3), bool)", "PATH_TAIL_RX": rx}, "bindings_checked": sorted(want),
               "libs": {"numpy": np.__version__, "scipy": scipy.__version__, "PIL": PIL.__version__}}

def spot_rows(n, k):
    """k evenly spaced row indices of 0..n-1 (deterministic; all rows when n <= k)"""
    if n <= k: return list(range(n))
    return sorted({int(round(j * (n - 1) / (k - 1))) for j in range(k)})

def band_crop_index(band_paths, root, day):
    """{seg: {"<HHMMSS>.jpg", ...}} of the crops the bands npz references; every path must be a <root>/<day>/crops_dense crop"""
    need = defaultdict(set)
    for p in band_paths:
        m = BAND_CROP_RX.search(str(p).replace("\\", "/"))
        if not m or m["root"] != root or m["day"] != day:
            raise Refused(f"band path {p} is not a {root}/{day}/crops_dense crop — the bands npz belongs to another root / day")
        need[m["seg"]].add(m["hh"] + ".jpg")
    return need

RETRY_TRIES, RETRY_WAIT, RETRY_CAP = 6, 5.0, 60.0    # review 2026-09-30: the API returned 'EOF' for EVERY call (serial too) for
                                                   # tens of seconds at a time; 3 tries over 9 s failed a real dry run twice

def backoff(k):
    """seconds to wait after failed try k (0-based): 5, 10, 20, 40, 60 -> ~2.2 min over RETRY_TRIES tries"""
    return min(RETRY_WAIT * 2 ** k, RETRY_CAP)

def retry(fn, tries=None, wait=None):
    """a read-only CLI call, retried on RuntimeError (transient API errors, e.g. 'EOF' seen 2026-09-30) with exponential
    backoff (backoff(); `wait` overrides the base for tests); last error re-raised"""
    tries = tries or RETRY_TRIES
    for k in range(tries):
        try:
            return fn()
        except RuntimeError:
            if k == tries - 1: raise
            time.sleep(backoff(k) if wait is None else wait * (k + 1))

def fetch_crops(remote, rv_day, need, crops_dir, log=print):
    """every referenced crop -> <crops_dir>/<seg>/<HHMMSS>.jpg (read-only): a segment dir is REUSED only if every referenced
    file exists locally with the size the Volume lists, else copied recursively (<= COLOUR_DL_WORKERS in parallel, RETRY_TRIES
    tries with backoff()).
    Returns stats incl. a manifest sha256 over (seg/file, sha256) of the referenced crops."""
    def one(seg):
        L = retry(lambda: remote.ls(f"{rv_day}/crops_dense/{seg}"))
        if L is None: raise Refused(f"{rv_day}/crops_dense/{seg} does not exist (the bands reference {len(need[seg])} crops there)")
        sizes = {e["name"]: e["size"] for e in L if not e["is_directory"]}
        miss = sorted(need[seg] - set(sizes))
        if miss: raise Refused(f"{rv_day}/crops_dense/{seg} lacks {len(miss)} crops the bands reference (e.g. {miss[0]})")
        ld = os.path.join(crops_dir, seg)
        def ok():
            return all(sizes[f] is not None and os.path.isfile(os.path.join(ld, f)) and os.path.getsize(os.path.join(ld, f)) == int(sizes[f])
                       for f in need[seg])
        state = "reused" if ok() else None
        err = None
        for _try in range(0 if state else RETRY_TRIES):
            if _try: time.sleep(backoff(_try - 1))
            try:
                remote.get(f"{rv_day}/crops_dense/{seg}", ld, recursive=True)
            except RuntimeError as e:
                err = e; continue
            if ok(): state = "downloaded"; break
            err = RuntimeError(f"crops_dense/{seg}: local files do not match the Volume sizes after the copy")
        if state is None: raise RuntimeError(f"crops_dense/{seg}: download failed {RETRY_TRIES} times: {err}")
        # content hashes here, in the worker (file opens dominate on Windows: ~8 ms each, serially ~2 min per day)
        return seg, (state, {f: sha256_file(os.path.join(ld, f)) for f in sorted(need[seg])})
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=COLOUR_DL_WORKERS) as ex: res = dict(ex.map(one, sorted(need)))
    h = hashlib.sha256()
    for seg in sorted(need):
        for f in sorted(need[seg]): h.update(f"{seg}/{f} {res[seg][1][f]}\n".encode())
    st = {"crops_dir": os.path.abspath(crops_dir), "n_crops": sum(len(v) for v in need.values()), "n_segs": len(need),
          "downloaded_segs": sorted(k for k, v in res.items() if v[0] == "downloaded"),
          "reused_segs": sorted(k for k, v in res.items() if v[0] == "reused"), "crops_manifest_sha256": h.hexdigest(),
          "seconds": round(time.time() - t0, 1)}
    log(f"    crops_dense: {st['n_crops']} crops in {st['n_segs']} segments ({len(st['downloaded_segs'])} downloaded, "
        f"{len(st['reused_segs'])} reused) {st['seconds']:.0f} s")
    return st

def colour_for_day(remote, rv, root, day, names, dd, crops_root, M, log=print):
    """<dd>/crop_colour_v1.npz for one day, row-aligned with <dd>/<BANDS>: read from the Volume day dir if present (alignment +
    COLOUR_SPOT rows recomputed bit-equal from their crops), else computed by the pinned Tol_54 from the day's crops_dense.
    Returns the provenance record."""
    import numpy as np
    rv_day, npz, bp = f"{rv}/{day}", os.path.join(dd, COLOUR_NPZ), os.path.join(dd, BANDS)
    with np.load(bp, allow_pickle=False) as z: P = [str(p) for p in z["paths"]]
    need = band_crop_index(P, root, day)
    t0 = time.time()
    if COLOUR_NPZ in names:
        retry(lambda: remote.get(f"{rv_day}/{COLOUR_NPZ}", npz))
        try:
            C = M.load_colour(npz, bp)
        except SystemExit as e:
            raise Refused(f"{rv_day}/{COLOUR_NPZ} is not row-aligned with the day's bands: {e}")
        idx = spot_rows(len(P), COLOUR_SPOT); spot, tmp = [], os.path.join(crops_root, "_spot", day)
        os.makedirs(tmp, exist_ok=True)
        try:
            for i in idx:
                m = BAND_CROP_RX.search(P[i].replace("\\", "/")); lp = os.path.join(tmp, f"{i}.jpg")
                retry(lambda: remote.get(f"{rv_day}/crops_dense/{m['seg']}/{m['hh']}.jpg", lp))
                r = M.crop_colour(M.read_rgb(lp))
                a_, b_ = np.float32(r["coloured_frac"]), np.float32(C["coloured_frac"][i])
                same = (np.isnan(a_) and np.isnan(b_)) or a_ == b_
                spot.append({"row": i, "crop": f"{m['seg']}/{m['hh']}.jpg", "stored": None if np.isnan(b_) else float(b_),
                             "recomputed": None if np.isnan(a_) else float(a_), "equal": bool(same)})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        badr = [s for s in spot if not s["equal"]]
        if badr: raise Refused(f"{rv_day}/{COLOUR_NPZ}: {len(badr)} of {len(spot)} spot rows differ from the pinned Tol_54 "
                               f"statistic (e.g. {badr[0]}) — not computed with the frozen statistic")
        rec = {"source": "volume", "spot_check": {"n": len(spot), "all_equal": True, "rows": spot}}
    else:
        cst = fetch_crops(remote, rv_day, need, os.path.join(crops_root, day), log=log)
        r = M.score_bands(bp, os.path.join(crops_root, day), npz, log_every=0)
        C = M.load_colour(npz, bp)
        rec = {"source": "computed", "crops": cst, "score_seconds": r["seconds"]}
    cf = np.asarray(C["coloured_frac"], dtype=np.float64)
    rec.update(sha256=sha256_file(npz), n_rows=len(P), n_nan=int(np.sum(~np.isfinite(cf))), seconds=round(time.time() - t0, 1))
    log(f"  {day}: {COLOUR_NPZ} {rec['source']} ({len(P)} rows, {rec['n_nan']} NaN, sha {rec['sha256'][:16]}) {rec['seconds']:.0f} s")
    return rec

# ---------------------------------------------------------------------------------------------------------------------
def run(a, remote, reg, log=print):
    t_all = time.time()
    TOL, fmt, EXP = reg["TOL"], reg["RELABEL_DIR_FMT"], reg["EXPECTED_CLIPS"]
    days = [d.strip() for d in a.days.split(",") if d.strip()]
    rdays = [d.strip() for d in a.render_days.split(",") if d.strip()]
    check_days(days, rdays, EXP)
    idn = getattr(a, "identity", "production") or "production"
    if a.upload and idn in IDENTITY_UNACCEPTED:                          # checked before any work is done
        raise Refused(f"--identity {idn} may not be uploaded: {IDENTITY_UNACCEPTED[idn]} — dry runs only")
    id_flags, id_prov = identity_flags(idn)                              # pinned files checked before any work is done
    nc_mode = (id_prov.get("noncalf") or {}).get("mode")                 # None | "anchor" | "colour"
    if production_base(a.root, reg) is None and not getattr(a, "allow_any_root", False):
        raise Refused(f"--root {a.root} is not a production root {reg['PRODUCTION_ROOTS']} (or a change-control revision "
                      f"<root>_r<N> of one)")
    slug = a.root[len("Tracking_"):]
    rg = getattr(a, "require_gate", None)
    gate_t30, gate_recs = check_gates([rg] if isinstance(rg, str) else rg, a.root, rdays, a.upload)
    dest = TOL + "/" + fmt.format(slug=slug)
    existing = {e["name"] for e in (remote.ls(dest) or [])}
    clash = [d for d in rdays if f"relabel_week_{d}.json" in existing]
    if a.upload and clash and not a.allow_overwrite:                     # checked before any work is done
        raise Refused(f"{dest} already holds relabels for {clash} (a render that may already have been reviewed); pass "
                      f"--allow-overwrite only if a re-render is intended")
    tag = f"week_{days[0]}_{days[-1]}_{datetime.now():%Y%m%d_%H%M%S}"
    k = 1
    while os.path.exists(os.path.join(a.work, slug, tag)): tag = tag.split("~")[0] + f"~{k}"; k += 1
    work = os.path.join(a.work, slug, tag)
    root_dir, out_dir = os.path.join(work, "root"), os.path.join(work, "out")
    os.makedirs(root_dir); os.makedirs(out_dir)
    prov = {"tool": TOOL, "created": datetime.now().isoformat(timespec="seconds"), "root": a.root, "slug": slug,
            "gates": gate_recs,
            "days": days, "render_days": rdays, "tag": tag, "work": work, "gt_used": False,
            "gt_guards": {"local_whitelist": sorted(WHITELIST), "remote_refuse_rx": GT_RX.pattern,
                          "tol30_flags_never_passed": ["--gt", "--week_map", "--links"],
                          "stdout_must_say": "<day>: no GT for every day"},
            "user": remote.whoami(), "inputs": {}, "root_config": {}}
    rv = TOL + "/" + a.root
    # ---- 1. fetch (whitelist) + completeness + remote GT guard ----
    root_cfg_local = os.path.join(work, "ROOT_CONFIG.root.json")
    remote.get(rv + "/ROOT_CONFIG.json", root_cfg_local)
    RC = json.load(open(root_cfg_local, encoding="utf-8"))
    listings = {}
    for d in days:
        L = remote.ls(f"{rv}/{d}")
        if L is None: raise Refused(f"{rv}/{d} does not exist")
        names = listings[d] = remote_guard(L, d)
        nd = sum(1 for n in names if n.startswith(".done_"))
        if nd != EXP[d]: raise Refused(f"{a.root}/{d}: {nd} .done_ markers, expected {EXP[d]} — the day is incomplete")
        miss = [f for f in NEED + ("ROOT_CONFIG.json",) if f not in names]
        if miss: raise Refused(f"{a.root}/{d} lacks {miss} (L5/L6/L7 downstream finished?)")
        dd = os.path.join(root_dir, d); os.makedirs(dd)
        for f in NEED + ("ROOT_CONFIG.json",):
            remote.get(f"{rv}/{d}/{f}", os.path.join(dd, f))
        prov["inputs"][d] = {f: {"sha256": sha256_file(os.path.join(dd, f)), "bytes": os.path.getsize(os.path.join(dd, f))}
                             for f in NEED}
        dc = json.load(open(os.path.join(dd, "ROOT_CONFIG.json"), encoding="utf-8"))
        diffs = [k for k in ("widgets", "shas", "band", "gpu_class") if dc.get(k) != RC.get(k)]
        if diffs: raise Refused(f"{a.root}/{d}/ROOT_CONFIG.json differs from the root's in {diffs}: one config per root")
        prov["root_config"][d] = sha256_file(os.path.join(dd, "ROOT_CONFIG.json"))
        log(f"  fetched {a.root}/{d}: {nd} clips done, inputs {sum(v['bytes'] for v in prov['inputs'][d].values()) / 2**20:.0f} MiB")
    prov["root_config"]["gpu_class"] = RC.get("gpu_class"); prov["root_config"]["shas"] = RC.get("shas")
    local_guard(root_dir, days)
    # ---- 2. Tol_30 snapshot ----
    snap = os.path.join(work, "tol30_snapshot.py")
    shutil.copyfile(a.tol30, snap)
    s30 = sha256_file(snap)
    if a.tol30_sha and not s30.startswith(a.tol30_sha.lower()):
        raise Refused(f"Tol_30 sha256 {s30} does not match the pin {a.tol30_sha}")
    if gate_t30 and s30 != gate_t30:
        raise Refused(f"Tol_30 sha256 {s30} is not the revision G3 scored with ({gate_t30}): one Tol_30 for the scoring, "
                      f"night-gap and production chains")
    prov["tol30"] = {"source": os.path.abspath(a.tol30), "sha256": s30, "pinned": a.tol30_sha or None}
    # ---- 2b. colour statistic (--noncalf colour only): <day>/crop_colour_v1.npz next to the bands, NOT GT ----
    if nc_mode == "colour":
        snap54 = os.path.join(work, "tol54_snapshot.py")
        shutil.copyfile(TOL54, snap54)
        M54, pins54 = load_tol54(snap54, json.load(open(COLOUR_JSON, encoding="utf-8")))
        crops_root = getattr(a, "colour_crops", None) or os.path.join(tempfile.gettempdir(), "prc", a.root)
        prov["colour"] = {"json": COLOUR_JSON, "json_sha256": sha256_file(COLOUR_JSON), "npz": COLOUR_NPZ,
                          "tol54": {"source": os.path.abspath(TOL54), "sha256": sha256_file(snap54), **pins54},
                          "crops_cache": os.path.abspath(crops_root), "days": {}}
        if prov["colour"]["json_sha256"] != COLOUR_SHA: raise Refused(f"{COLOUR_JSON} changed since identity_flags pinned it")
        log(f"  colour statistic ({COLOUR_NPZ}) with Tol_54 {prov['colour']['tol54']['sha256'][:16]}… (crops cache {crops_root})")
        for d in days:
            prov["colour"]["days"][d] = colour_for_day(remote, rv, a.root, d, listings[d], os.path.join(root_dir, d), crops_root,
                                                       M54, log=log)
        if sha256_file(snap54) != prov["colour"]["tol54"]["sha256"]: raise RuntimeError("Tol_54 snapshot changed during the run")
        local_guard(root_dir, days)                                      # the npz is whitelisted; nothing else
    # ---- 3. week mode, GT-free ----
    prov["identity_config"] = id_prov
    if idn in IDENTITY_UNACCEPTED: prov["identity_config"]["unaccepted"] = IDENTITY_UNACCEPTED[idn]
    need_hidden = "--noncalf" in id_flags
    cmd = [sys.executable, snap, "--root", root_dir, "--days", ",".join(days), "--T", str(T_MIN), "--bands", BANDS,
           "--out_dir", out_dir] + id_flags
    ac = getattr(a, "anchor_cache", None)                                # optional: where Tol_30 caches per-band anchor
    if ac:                                                               # scores (keyed + validated by pool / band sha;
        if "--anchors" not in id_flags: raise Refused(f"--anchor-cache needs an anchor identity config, not {idn}")
        cmd += ["--anchor_cache", ac]                                    # never changes an output byte)
        prov["anchor_cache"] = os.path.abspath(ac)
    assert not any(x in cmd for x in ("--gt", "--week_map", "--links"))
    log(f"  Tol_30 week mode ({s30[:16]}…) on {days} …")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    with open(os.path.join(out_dir, "tol30_week.log"), "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n" + p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0: raise RuntimeError(f"Tol_30 week mode failed (rc {p.returncode}):\n{p.stderr[-2000:]}")
    if sha256_file(snap) != s30: raise RuntimeError("Tol_30 snapshot changed during the run")
    no_gt = {d: bool(re.search(rf"^\s*{d}: no GT\s*$", p.stdout, re.M)) for d in days}
    if not all(no_gt.values()):
        raise Refused(f"Tol_30 did not report 'no GT' for {[d for d, v in no_gt.items() if not v]} — a GT file was visible")
    local_guard(root_dir, days)                                          # Tol_30 wrote only to out_dir
    prov.update(cmd=" ".join(cmd), tol30_runtime_s=round(time.time() - t0, 1), stdout_no_gt=no_gt)
    # ---- 4. validate outputs ----
    prov["outputs"], problems = {}, []
    for d in days:
        op = os.path.join(out_dir, f"relabel_week_{d}.json")
        if not os.path.exists(op): problems.append(f"{d}: Tol_30 wrote no relabel_week_{d}.json"); continue
        st, pr, wr = validate_relabel(op, d, require_hidden=need_hidden, mode=nc_mode)
        prov["outputs"][d] = {"sha256": sha256_file(op), **st, "warnings": wr}
        if need_hidden and not pr:
            log(f"  {d}: {st.get('n_hidden', 0)} non-calf track(s) hidden ({nc_mode}) "
                + ", ".join(f"slot {h['slot']} {h['from'][11:19]}-{h['to'][11:19]} ({h['seg']}, median "
                            f"{h.get('cf_median', h.get('score_median'))})" for h in st.get("hidden", [])))
        problems += [f"{d}: {x}" for x in pr]
        if not st.get("n_rules"): problems.append(f"{d}: no identity rule at all")
    if not os.path.exists(os.path.join(out_dir, "periodic_identity_WEEK.json")):
        problems.append("periodic_identity_WEEK.json missing")
    if problems: raise RuntimeError("relabel validation failed: " + "; ".join(problems))
    # ---- 5. upload (explicit only) ----
    prov["upload"] = {"requested": bool(a.upload), "dest": dest, "files": [], "existing_before": sorted(existing)}
    changed = {}
    for d in days:
        nm = f"relabel_week_{d}.json"
        if d in rdays or nm not in existing: continue
        old = os.path.join(work, "previous_" + nm)
        remote.get(f"{dest}/{nm}", old)
        a_, b_ = json.load(open(old, encoding="utf-8")).get("rules"), \
            json.load(open(os.path.join(out_dir, nm), encoding="utf-8")).get("rules")
        changed[d] = {"identical_rules": a_ == b_}
    prov["watched_days_changed"] = changed
    if any(not v["identical_rules"] for v in changed.values()):
        log(f"  ⚠ this run would relabel already-uploaded (watched?) days: "
            f"{[d for d, v in changed.items() if not v['identical_rules']]} — NOT uploaded (render days only); reported")
    if a.upload:
        run_dir = f"{dest}/runs/{tag}"
        for d in rdays:
            remote.put(os.path.join(out_dir, f"relabel_week_{d}.json"), f"{dest}/relabel_week_{d}.json")
            prov["upload"]["files"].append(f"{dest}/relabel_week_{d}.json")
        for d in days:
            remote.put(os.path.join(out_dir, f"relabel_week_{d}.json"), f"{run_dir}/relabel_week_{d}.json")
        for f in ("periodic_identity_WEEK.json", "tol30_week.log"):
            remote.put(os.path.join(out_dir, f), f"{run_dir}/{f}")
        # read back what the renders will read
        for d in rdays:
            chk = os.path.join(work, f"readback_{d}.json")
            remote.get(f"{dest}/relabel_week_{d}.json", chk)
            if sha256_file(chk) != prov["outputs"][d]["sha256"]: raise RuntimeError(f"read-back mismatch for {d}")
        prov["upload"]["run_dir"] = run_dir; prov["upload"]["verified_readback"] = True
    prov["runtime_s"] = round(time.time() - t_all, 1)
    pp_ = os.path.join(work, "provenance.json")
    json.dump(prov, open(pp_, "w", encoding="utf-8"), indent=1)
    if a.upload: remote.put(pp_, f"{prov['upload']['run_dir']}/provenance.json")
    log(f"  relabels {', '.join(f'{d}: {prov['outputs'][d]['n_rules']} rules / {prov['outputs'][d]['labelled_slot_h']} slot-h' for d in days)}")
    log(f"  {'UPLOADED ' + str(prov['upload']['files']) if a.upload else 'dry run (no --upload): nothing written to the Volume'}")
    log(f"  provenance {pp_}")
    return prov

# ---------------------------------------------------------------------------------------------------------------------
def make_colour_day(M, vol, TOL, root, day, segs, rng, coloured=()):
    """selftest helper: synthetic masked crops on the fake Volume <TOL>/<root>/<day>/crops_dense/<seg>/<HHMMSS>.jpg (segments in
    `coloured` wear blue) + the list of their band paths (bands order = reversed file order, to prove row alignment)"""
    from PIL import Image
    paths = []
    for s, n in segs:
        dd = os.path.join(vol, root, day, "crops_dense", s); os.makedirs(dd, exist_ok=True)
        for k in range(n):
            rgb, _ = M._synthetic_calf(rng, coloured=(20, 100, 235) if s in coloured else None, frac=0.7)
            hh = f"12{len(paths) // 60:02d}{len(paths) % 60:02d}"
            Image.fromarray(rgb).save(os.path.join(dd, hh + ".jpg"), quality=95)
            paths.append(f"{TOL}/{root}/{day}/crops_dense/{s}/{hh}.jpg")
    return paths[::-1]

def selftest_colour(tmp, TOL):
    """--identity B60's colour stage without network or real data: Tol_54 pins, Tol_30's rule == Tol_54's reference,
    band-path root/day guard, compute (download -> score) == Tol_54 on the Volume files, cache reuse / size-mismatch
    re-download / missing-crop refusal, Volume-provided npz (alignment + bit-equal spot rows; tampered / misaligned refused)."""
    import numpy as np
    cj = json.load(open(COLOUR_JSON, encoding="utf-8"))
    snap = os.path.join(tmp, "t54_snap.py"); shutil.copyfile(TOL54, snap)
    M, pins = load_tol54(snap, cj)
    assert pins["constants"]["S_MIN"] == 0.35 and pins["constants"]["V_MIN"] == 0.5, pins
    src = open(TOL54, encoding="utf-8").read()
    for old, new, frag in (("\nS_MIN = 0.35\n", "\nS_MIN = 0.36\n", "S_MIN"), ("im.convert(\"RGB\")", "im.convert(\"RGB\").copy()", "read_rgb()"),
                           ("\nEDGE_ERODE = 3 ", "\nEDGE_ERODE = 2 ", "EDGE_ERODE"), ("    fg = ~bg\n", "    fg = ~bg  # x\n", "foreground()"),
                           # review 2026-09-30: module objects + run-time bindings (function source unchanged in all three)
                           ("\n_K3 = np.ones((3, 3), bool)\n", "\n_K3 = np.ones((3, 3), bool); _K3[0, 0] = _K3[0, 2] = _K3[2, 0] = _K3[2, 2] = False\n", "_K3"),
                           ('PATH_TAIL_RX = re.compile(r"/crops_dense/([^/]+)/(\\d{6})\\.jpg$")',
                            'PATH_TAIL_RX = re.compile(r"/crops_dense/([^/]+)/(\\d{4,6})\\.jpg$")', "PATH_TAIL_RX"),
                           ("\ndef local_crop_path(band_path, crops_dir):", "\nhsv_arrays = lambda x: (x[:, 0], x[:, 1], x.max(axis=1))\n"
                                                                          "def local_crop_path(band_path, crops_dir):", "hsv_arrays() is not bound")):
        assert src.count(old) == 1, old
        bad = os.path.join(tmp, f"t54_bad_{frag.strip('()')}.py")
        with open(bad, "w", encoding="utf-8") as f: f.write(src.replace(old, new, 1))
        try:
            load_tol54(bad, cj); raise AssertionError(f"a Tol_54 with a changed {frag} was accepted")
        except Refused as e:
            assert frag in str(e), (frag, e)
    # Tol_30's colour rule (what the Tol_30 run applies) == Tol_54's reference implementation
    spec = importlib.util.spec_from_file_location("tol30_for_prod_relabel_st", NB + "/Tol_30_periodic_identity.py")
    T30 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T30)
    rng = np.random.default_rng(3); n_hid = 0
    for _ in range(200):
        crops, cf = [], []
        for s in range(int(rng.integers(1, 10))):
            ph = rng.random()
            for _k in range(int(rng.integers(0, 40))):
                u = rng.random()
                x = float("nan") if u < 0.05 else (0.07 if u < 0.1 else (0.1 + 0.9 * rng.random() if rng.random() < ph else 0.06 * rng.random()))
                crops.append(dict(sid=f"s{s}", i=len(cf), gated=bool(rng.random() < 0.85))); cf.append(x)
        cf = np.array(cf); cfg = dict(crop_threshold=float(cj["crop_threshold"]), min_gated_crops=int(rng.integers(1, 25)),
                                      frac_above=float(rng.choice([0.5, 0.25, 1.0])))
        a_, b_ = T30.colour_noncalf_segments(crops, cf, cfg), M.colour_noncalf_segments(crops, cf, cfg)
        assert a_ == b_, (a_, b_, cfg); n_hid += bool(a_)
    assert n_hid > 20, n_hid
    # compute path on a synthetic day through a FakeRemote
    vol = os.path.join(tmp, "cvol"); R = v5_remote.FakeRemote(TOL, vol); root, day = "Tracking_x", "20240422"
    P = make_colour_day(M, vol, TOL, root, day, [("0_001", 5), ("1_012", 6), ("2_003", 4)], rng, coloured=("1_012",))
    dd = os.path.join(tmp, "croot", day); os.makedirs(dd)
    np.savez(os.path.join(dd, BANDS), paths=np.array(P), emb=np.zeros((len(P), 8, 4), np.float16))
    ref = os.path.join(tmp, "ref_colour.npz")
    M.score_bands(os.path.join(dd, BANDS), os.path.join(vol, root, day, "crops_dense"), ref, log_every=0)
    cc = os.path.join(tmp, "cc"); rv = f"{TOL}/{root}"; q = lambda m: None
    r1 = colour_for_day(R, rv, root, day, ["crops_dense"], dd, cc, M, log=q)
    assert r1["source"] == "computed" and r1["sha256"] == sha256_file(ref) and r1["n_rows"] == 15, r1
    assert sorted(r1["crops"]["downloaded_segs"]) == ["0_001", "1_012", "2_003"] and not r1["crops"]["reused_segs"], r1
    C = M.load_colour(os.path.join(dd, COLOUR_NPZ), os.path.join(dd, BANDS)); cfv = C["coloured_frac"]
    blue = [i for i, p in enumerate(P) if "/1_012/" in p]
    assert all(cfv[i] > 0.5 for i in blue) and all(cfv[i] < 0.01 for i in range(len(P)) if i not in blue), cfv
    r2 = colour_for_day(R, rv, root, day, ["crops_dense"], dd, cc, M, log=q)
    assert r2["crops"]["reused_segs"] == ["0_001", "1_012", "2_003"] and r2["sha256"] == r1["sha256"], r2
    assert r2["crops"]["crops_manifest_sha256"] == r1["crops"]["crops_manifest_sha256"]
    f0 = os.path.join(cc, day, "1_012", sorted(os.listdir(os.path.join(cc, day, "1_012")))[0])
    open(f0, "wb").write(open(f0, "rb").read()[:100])                  # a truncated cached crop -> that segment re-downloaded
    r3 = colour_for_day(R, rv, root, day, ["crops_dense"], dd, cc, M, log=q)
    assert r3["crops"]["downloaded_segs"] == ["1_012"] and r3["sha256"] == r1["sha256"], r3
    np.savez(os.path.join(dd, BANDS), paths=np.array([p.replace("/Tracking_x/", "/Tracking_y/") for p in P]), emb=np.zeros((len(P), 8, 4)))
    for bad_names, frag in ((["crops_dense"], "another root"),):
        try:
            colour_for_day(R, rv, root, day, bad_names, dd, cc, M, log=q); raise AssertionError(frag)
        except Refused as e:
            assert frag in str(e), e
    np.savez(os.path.join(dd, BANDS), paths=np.array(P), emb=np.zeros((len(P), 8, 4), np.float16))
    gone = P[0].split(f"{root}/{day}/")[1]; keep_b = open(os.path.join(vol, root, day, *gone.split("/")), "rb").read()
    os.remove(os.path.join(vol, root, day, *gone.split("/")))
    try:
        colour_for_day(R, rv, root, day, ["crops_dense"], dd, os.path.join(tmp, "cc2"), M, log=q); raise AssertionError("missing crop")
    except Refused as e:
        assert "lacks 1 crops" in str(e), e
    open(os.path.join(vol, root, day, *gone.split("/")), "wb").write(keep_b)
    assert [backoff(k) for k in range(RETRY_TRIES - 1)] == [5.0, 10.0, 20.0, 40.0, 60.0], [backoff(k) for k in range(RETRY_TRIES)]
    import threading
    class Flaky(v5_remote.FakeRemote):                                  # transient CLI errors (seen: "EOF") are retried
        n = 0; g = 0; lock = threading.Lock()                           # (fetch_crops runs segments in parallel threads)
        def ls(self, path):
            with Flaky.lock:
                fail = "/crops_dense/" in path and Flaky.n < 4; Flaky.n += fail
            if fail: raise RuntimeError(f"fs ls {path} failed: EOF")
            return super().ls(path)
        def get(self, path, local, recursive=False):
            with Flaky.lock:
                fail = recursive and Flaky.g < 3; Flaky.g += fail
            if fail: raise RuntimeError(f"fs cp {path} failed: EOF")
            return super().get(path, local, recursive=recursive)
    global RETRY_WAIT
    keep_w = RETRY_WAIT; RETRY_WAIT = 0.01                             # same schedule shape, no 2-min sleeps in the selftest
    try:
        r5 = colour_for_day(Flaky(TOL, vol), rv, root, day, ["crops_dense"], dd, os.path.join(tmp, "cc3"), M, log=q)
        assert Flaky.n == 4 and Flaky.g == 3 and r5["sha256"] == r1["sha256"] and len(r5["crops"]["downloaded_segs"]) == 3, (Flaky.n, r5)
        class Dead(v5_remote.FakeRemote):                               # a PERSISTENT failure still aborts (no silent skip)
            def ls(self, path):
                if "/crops_dense/" in path: raise RuntimeError(f"fs ls {path} failed: EOF")
                return super().ls(path)
        try:
            colour_for_day(Dead(TOL, vol), rv, root, day, ["crops_dense"], dd, os.path.join(tmp, "cc4"), M, log=q)
            raise AssertionError("a persistently failing listing was accepted")
        except RuntimeError as e:
            assert "EOF" in str(e) and not isinstance(e, AssertionError), e
    finally:
        RETRY_WAIT = keep_w
    # Volume-provided npz: aligned + bit-equal spot rows accepted; tampered spot row / misaligned rows refused
    shutil.copyfile(ref, os.path.join(vol, root, day, COLOUR_NPZ))
    r4 = colour_for_day(R, rv, root, day, ["crops_dense", COLOUR_NPZ], dd, cc, M, log=q)
    assert r4["source"] == "volume" and r4["spot_check"]["all_equal"] and r4["spot_check"]["n"] == 15 and r4["sha256"] == r1["sha256"], r4
    assert spot_rows(21290, COLOUR_SPOT)[:2] == [0, 1419] and len(spot_rows(21290, COLOUR_SPOT)) == COLOUR_SPOT
    with np.load(ref) as z: Z = {k: np.array(z[k]) for k in z.files}
    for mut, frag in ((lambda Z: Z["coloured_frac"].__setitem__(3, Z["coloured_frac"][3] + np.float32(1e-3)), "spot rows differ"),
                      (lambda Z: Z.__setitem__("paths", Z["paths"][::-1].copy()), "not row-aligned")):
        Zb = {k: v.copy() for k, v in Z.items()}; mut(Zb)
        np.savez(os.path.join(vol, root, day, COLOUR_NPZ), **Zb)
        try:
            colour_for_day(R, rv, root, day, ["crops_dense", COLOUR_NPZ], dd, cc, M, log=q); raise AssertionError(frag)
        except Refused as e:
            assert frag in str(e), (frag, e)
    os.remove(os.path.join(vol, root, day, COLOUR_NPZ))
    print("prod_relabel SELFTEST colour stage (Tol_54 pins: S_MIN / EDGE_ERODE / read_rgb / foreground / _K3 / PATH_TAIL_RX / re-bound hsv_arrays refused; Tol_30 "
          f"rule == Tol_54 reference on 200 random cases ({n_hid} hiding); compute == Tol_54 on the Volume crops; cache reuse, "
          "truncated-file re-download, missing crop, foreign-root bands refused; Volume npz: 15 spot rows bit-equal accepted, "
          "tampered / misaligned refused; transient EOF on ls / cp retried with backoff, a persistent failure aborts): OK")

def selftest(data=None, full=False):
    """full=False: ONE real Tol_30 week-mode run on 04-22 (~10 min measured); full=True: 04-22 + 04-23 = the watch-#1
    configuration (~50 min measured 2026-09-24). Every other case uses a stand-in Tol_30 that replays run 1."""
    reg = v5_remote.launcher_registry(LAUNCHER)
    TOL = reg["TOL"]; EXP = reg["EXPECTED_CLIPS"]
    # unit: days, guards, validation
    for bad in (["20240423"], ["20240422", "20240424"], []):
        try:
            check_days(bad, bad[:1] or ["x"], EXP); raise AssertionError(f"days {bad} accepted")
        except Refused:
            pass
    check_days(["20240422", "20240423"], ["20240423"], EXP)
    for n in ("segment_assignments.json", "segment_assignments_reviewed.json", "reid_ground_truth_0423.json",
              "gt_0424_from_user_notes.json", "week_ident_map.json", "relabel_week_20240422.json"):
        try:
            remote_guard([{"name": n}], "20240422"); raise AssertionError(n)
        except Refused:
            pass
    assert remote_guard([{"name": x} for x in ("masks_a.json.gz", "detections_a.json.gz", "segments.json",
                                                ".done_a", "crops_dense", "observations.npz")], "d")
    tmp = tempfile.mkdtemp(prefix="prod_relabel_st_")
    try:
        r0 = os.path.join(tmp, "g"); os.makedirs(os.path.join(r0, "20240422"))
        for f in WHITELIST: open(os.path.join(r0, "20240422", f), "w").close()
        local_guard(r0, ["20240422"])
        for extra in ("20240422/segment_assignments_transferred.json", "week_ident_map.json", "20240423/x.json"):
            q = os.path.join(r0, extra); os.makedirs(os.path.dirname(q), exist_ok=True); open(q, "w").close()
            try:
                local_guard(r0, ["20240422"]); raise AssertionError(extra)
            except Refused:
                os.remove(q)
        rp = os.path.join(tmp, "r.json")
        json.dump({"day": "20240422", "rules": {"0": [["2024-04-22T12:45:16", "2024-04-22T13:00:00", 0]],
                                                "1": [["2024-04-22T12:45:16", "2024-04-22T12:50:00", 0]]}}, open(rp, "w"))
        st, pr, wr = validate_relabel(rp, "20240422")
        assert not pr and st["n_rules"] == 2 and st["duplicate_identity_s"] == 284.0 and wr, (st, pr, wr)
        json.dump({"day": "20240422", "rules": {"5": [["2024-04-22T12:45:16", "2024-04-22T12:40:00", 7]]}}, open(rp, "w"))
        assert validate_relabel(rp, "20240422")[1]
        json.dump({"day": "20240422", "rules": {"3": [["2024-04-22T15:03:29.893351", "2024-04-22T15:03:29.893351", 1]]}},
                  open(rp, "w"))                                          # seen in a real Tol_30 week-mode output
        st, pr, wr = validate_relabel(rp, "20240422")
        assert not pr and any("zero-length" in w for w in wr), (pr, wr)
        # the --noncalf "hidden" list (validate / pass-through), 2026-09-29
        base = {"day": "20240423", "rules": {"1": [["2024-04-23T13:05:02", "2024-04-23T15:38:51.259625", 2],
                                                    ["2024-04-23T15:49:37", "2024-04-23T15:50:42", 2]],
                                              "2": [["2024-04-23T15:00:00", "2024-04-23T16:00:00", 1]]}}
        hid = {"slot": 1, "from": "2024-04-23T15:38:51.259625", "to": "2024-04-23T15:49:20.557709", "seg": "1_012",
               "why": "noncalf", "score_median": 0.5, "n_gated": 30, "n_below": 29, "frac_below": 0.9667}
        def hv(hidden, require=False, rules=None):
            J = dict(base, **({"hidden": hidden} if hidden is not None else {}))
            if rules is not None: J["rules"] = rules
            json.dump(J, open(rp, "w")); return validate_relabel(rp, "20240423", require_hidden=require)
        st, pr, wr = hv([hid], require=True)
        assert not pr and st["n_hidden"] == 1 and st["hidden"][0]["seg"] == "1_012" and st["hidden_slot_h"] > 0.17, (st, pr)
        assert not hv([], require=True)[1] and hv(None, require=True)[1] and not hv(None)[1]
        for bad, frag in (({"slot": 4}, "slot 4"), ({"why": "person"}, "why"), ({"from": "2024-04-23T15:50:00"}, "inverted"),
                          ({"seg": None, "score_median": "x"}, "bad entry"), ({"from": "2024-04-25T15:38:51"}, "not on")):
            pr = hv([dict(hid, **bad)])[1]
            assert any(frag in x for x in pr), (bad, pr)
        pr = hv([hid], rules={"1": [["2024-04-23T13:05:02", "2024-04-23T15:45:00", 2]]})[1]       # same slot carries identity
        assert any("must not carry an identity" in x for x in pr), pr
        assert not hv([hid], rules={"2": [["2024-04-23T15:00:00", "2024-04-23T16:00:00", 1]]})[1]   # other slot: fine
        assert hv({"slot": 1}, require=True)[1]
        print("prod_relabel SELFTEST unit (days prefix, remote + local GT guards, relabel validation, 'hidden' list): OK")
        # colour-mode "hidden" entries (2026-09-30): mode "colour" + n_gated / n_above / frac_above / cf_median
        hidc = {"slot": 1, "from": "2024-04-23T15:38:51.259625", "to": "2024-04-23T15:49:20.557709", "seg": "1_012",
                "why": "noncalf", "mode": "colour", "n_gated": 48, "n_above": 48, "frac_above": 1.0, "cf_median": 0.87635}
        st, pr, wr = hv([hidc], require=True)
        assert not pr and st["hidden"][0]["mode"] == "colour" and st["hidden"][0]["cf_median"] == 0.87635, (st, pr)
        J_ = dict(base, hidden=[hidc]); json.dump(J_, open(rp, "w"))
        assert not validate_relabel(rp, "20240423", True, mode="colour")[1]
        assert any("anchor filter" in x for x in validate_relabel(rp, "20240423", True, mode="anchor")[1])
        json.dump(dict(base, hidden=[hid]), open(rp, "w"))
        assert any("colour filter" in x for x in validate_relabel(rp, "20240423", True, mode="colour")[1])
        assert not validate_relabel(rp, "20240423", True, mode="anchor")[1]
        for drop in ("cf_median", "n_above"):
            pr = hv([{k: v for k, v in hidc.items() if k != drop}])[1]; assert any("cf_median" in x for x in pr), (drop, pr)
        assert any("bad entry" in x for x in hv([dict(hidc, cf_median="x")])[1])
        assert any("mode anchor|colour" in x for x in hv([dict(hidc, mode="hue")])[1])
        print("prod_relabel SELFTEST colour 'hidden' entries (keys, mode must match the config's filter): OK")
        # --identity B60 = Tol_52's B60 arm + --noncalf COLOUR json (pinned); B60nc = the arm + --noncalf ANCHOR json (pinned,
        # unaccepted): missing / changed files refused, and neither config reads the other's json
        assert identity_flags("production") == ([], {"name": "production", "flags": []})   # (json paths swapped via globals())
        b60 = ["--anchors", ANCHOR_POOL, "--anchor_pi", ANCHOR_PI, "--anchor_w", "0.6", "--temporal", "even", "--boot", "none"]
        fl, pv = identity_flags("B60")
        assert fl == b60 + ["--noncalf", COLOUR_JSON] and pv["noncalf"]["mode"] == "colour" and pv["noncalf"]["sha256"] == COLOUR_SHA, (fl, pv)
        assert pv["noncalf"]["crop_threshold"] == 0.07 and pv["noncalf"]["min_gated_crops"] == 18 and "B60" not in IDENTITY_UNACCEPTED
        fl, pv = identity_flags("B60nc")
        assert fl[-2:] == ["--noncalf", NONCALF_JSON] and pv["noncalf"]["sha256"] == NONCALF_SHA and fl[:-2] == b60, fl
        assert pv["noncalf"]["mode"] == "anchor" and "B60nc" in IDENTITY_UNACCEPTED
        for var, frag_missing in (("NONCALF_JSON", "Tol_53"), ("COLOUR_JSON", "Tol_54")):
            keep = globals()[var]; other = "B60" if var == "NONCALF_JSON" else "B60nc"; mine = "B60nc" if other == "B60" else "B60"
            try:
                globals()[var] = os.path.join(tmp, "missing.json")
                try:
                    identity_flags(mine); raise AssertionError(f"missing {var} accepted")
                except Refused as e:
                    assert "missing" in str(e) and frag_missing in str(e), e
                identity_flags(other)                                        # the other config never reads this json
                globals()[var] = os.path.join(tmp, "changed.json")
                J = json.load(open(keep, encoding="utf-8")); J["crop_threshold"] = float(J["crop_threshold"]) + 0.01
                json.dump(J, open(globals()[var], "w"))
                try:
                    identity_flags(mine); raise AssertionError(f"changed {var} accepted")
                except Refused as e:
                    assert "not the pinned" in str(e)
            finally:
                globals()[var] = keep
        print("prod_relabel SELFTEST identity (B60 = Tol_52 arm + --noncalf colour pinned; B60nc = arm + --noncalf anchor pinned: "
              "flags, provenance, missing / changed file refused, neither reads the other's json): OK")
        # the colour stage: Tol_54 pins, Tol_30's rule == Tol_54's reference, fetch / compute / reuse / Volume-read paths
        selftest_colour(tmp, TOL)
        # gating
        def vfile(name, **kw):
            J = {"gate": "G3", "status": "PASS", "root": "Tracking_v5a10", "days": ["20240422", "20240423"],
                 "tol30_sha": "ab" * 32, "checks": [{"id": "x", "status": "PASS"}], **kw}
            q = os.path.join(tmp, name); json.dump(J, open(q, "w")); return q
        g3, g5 = vfile("g3.json"), vfile("g5.json", gate="G5", day="20240424", days=None, tol30_sha=None)
        assert check_gates([g3], "Tracking_v5a10", ["20240422", "20240423"], True)[0] == "ab" * 32
        assert check_gates([g3, g5], "Tracking_v5a10", ["20240424"], True)[0] == "ab" * 32
        assert check_gates(None, "Tracking_v5a10", ["20240422"], False) == (None, [])        # a dry run may be ungated
        for paths, rd, frag in (
                (None, ["20240422"], "--upload needs --require-gate"),
                ([g5], ["20240424"], "no G3 verdict"),
                ([g3], ["20240424"], "not covered"),
                ([vfile("f.json", status="FAIL")], ["20240422"], "status FAIL"),
                ([vfile("r.json", status="PASS_WITH_REVIEW", checks=[{"id": "b", "status": "REVIEW"}])], ["20240422"],
                 "unacknowledged"),
                ([vfile("o.json", root="Tracking_v5a100cap21")], ["20240422"], "root"),
                ([vfile("n.json", tol30_sha=None)], ["20240422"], "no tol30_sha"),
                ([vfile("g1.json", gate="G1")], ["20240422"], "need G3 or G5")):
            try:
                check_gates(paths, "Tracking_v5a10", rd, True); raise AssertionError(frag)
            except Refused as e:
                assert frag in str(e), (frag, e)
        assert production_base("Tracking_v5a10_r2", reg) == "Tracking_v5a10" and production_base("Tracking_v5a10", reg)
        assert production_base("Tracking_v5a100cap21_r3", reg) == "Tracking_v5a100cap21"
        assert all(production_base(r, reg) is None for r in ("Tracking_v5a10_pilot", "Tracking_v5a10_r1", "Tracking_v4v2",
                                                             "Tracking_v5eq_ref_r2", "Tracking_v5a10_pilot_r2"))
        acked = vfile("a.json", status="PASS_WITH_REVIEW", checks=[{"id": "b", "status": "REVIEW"}],
                      review_ack={"by": "o", "note": "resumes checked", "rows": ["b"]})
        assert check_gates([acked], "Tracking_v5a10", ["20240423"], True)[0] == "ab" * 32
        print("prod_relabel SELFTEST gating (upload needs an accepted G3 for the root + G5 per later render day; FAIL / "
              "unacknowledged review / wrong root / no Tol_30 pin refused; acknowledged review accepted): OK")
        # end-to-end on real Tol_30 inputs through a FakeRemote (no network)
        real_days = ["20240422", "20240423"] if full else ["20240422"]
        if not data or not all(os.path.exists(os.path.join(data, d, f)) for d in ("20240422", "20240423") for f in NEED):
            print(f"prod_relabel SELFTEST end-to-end: SKIPPED (no Tol_30 inputs under {data})"); return
        vol = os.path.join(tmp, "vol"); R = v5_remote.FakeRemote(TOL, vol)
        root = "Tracking_v5a10"
        rc = {"widgets": {"detector": "rfdetr"}, "shas": {"core": "c" * 64}, "gpu_class": "A10", "band": {"x": 1}}
        os.makedirs(os.path.join(vol, root))
        json.dump(rc, open(os.path.join(vol, root, "ROOT_CONFIG.json"), "w"))
        for d in ("20240422", "20240423"):
            dd = os.path.join(vol, root, d); os.makedirs(dd)
            for f in NEED: shutil.copyfile(os.path.join(data, d, f), os.path.join(dd, f))
            json.dump({**rc, "written_by": d}, open(os.path.join(dd, "ROOT_CONFIG.json"), "w"))
            for k in range(EXP[d]): open(os.path.join(dd, f".done_clip{k:02d}"), "w").close()
            open(os.path.join(dd, "masks_x.json.gz"), "w").close()      # present remotely, never copied
        t30_real = NB + "/Tol_30_periodic_identity.py"
        g3_real = vfile("g3_real.json", tol30_sha=sha256_file(t30_real))   # the Tol_30 G3 "scored with"
        ns = lambda **kw: argparse.Namespace(**{**dict(root=root, days=",".join(real_days),
                                                       render_days=",".join(real_days), tol30=t30_real,
                                                       tol30_sha=None, work=os.path.join(tmp, "work"), upload=True,
                                                       allow_overwrite=False, require_gate=[g3_real]), **kw})
        t0 = time.time()
        try:
            run(ns(require_gate=None), R, reg, log=lambda m: None); raise AssertionError("ungated upload accepted")
        except Refused as e:
            assert "--upload needs --require-gate" in str(e)
        prov = run(ns(), R, reg, log=lambda m: None)
        assert prov["gates"] and prov["gates"][0]["gate"] == "G3"
        dest = os.path.join(vol, "Metadata", "relabel_v5a10")
        assert sorted(os.listdir(dest)) == [f"relabel_week_{d}.json" for d in real_days] + ["runs"], os.listdir(dest)
        assert prov["gt_used"] is False and all(prov["stdout_no_gt"].values()) and prov["upload"]["verified_readback"]
        assert all(prov["outputs"][d]["n_rules"] > 0 for d in real_days), prov["outputs"]
        n_zero = sum(1 for d in real_days for w in prov["outputs"][d]["warnings"] if "zero-length" in w)
        run_files = os.listdir(os.path.join(dest, "runs", prov["tag"]))
        assert {"provenance.json", "periodic_identity_WEEK.json", "tol30_week.log"} <= set(run_files), run_files
        # a second upload of the same render days is refused (renders that may already have been reviewed) ...
        try:
            run(ns(), R, reg, log=lambda m: None); raise AssertionError("overwrite accepted")
        except Refused as e:
            assert "already holds relabels" in str(e)
        # ... a dry run is fine and reports whether earlier (already uploaded) days would change. A stand-in Tol_30 that
        # replays run 1's outputs keeps the selftest to ONE real week-mode run.
        src = os.path.join(tmp, "t30_src"); os.makedirs(src)
        for d in real_days:
            shutil.copyfile(os.path.join(dest, f"relabel_week_{d}.json"), os.path.join(src, f"relabel_week_{d}.json"))
        if not full:                                                     # a valid stand-in 04-23 relabel
            json.dump({"day": "20240423", "rules": {"0": [["2024-04-23T05:44:28", "2024-04-23T06:14:26", 0]]}},
                      open(os.path.join(src, "relabel_week_20240423.json"), "w"))
        ns2 = lambda **kw: ns(**{"days": "20240422,20240423", "require_gate": None, **kw})
        fake30 = os.path.join(tmp, "fake_tol30.py")
        with open(fake30, "w", encoding="utf-8") as f:
            f.write(FAKE_T30.replace("@SRC@", src.replace("\\", "/")))
        try:        # a Tol_30 other than the one G3 scored with is refused
            run(ns2(render_days="20240423", upload=False, tol30=fake30, require_gate=[g3_real]), R, reg, log=lambda m: None)
            raise AssertionError("a Tol_30 other than G3's was accepted")
        except Refused as e:
            assert "not the revision G3 scored with" in str(e), e
        p2 = run(ns2(render_days="20240423", upload=False, tol30=fake30), R, reg, log=lambda m: None)
        assert p2["watched_days_changed"] == {"20240422": {"identical_rules": True}}, p2["watched_days_changed"]
        J2 = json.load(open(os.path.join(src, "relabel_week_20240422.json")))
        k0 = sorted(J2["rules"])[0]; J2["rules"][k0][0][2] = (int(J2["rules"][k0][0][2]) + 1) % N_ID
        json.dump(J2, open(os.path.join(src, "relabel_week_20240422.json"), "w"))
        p3 = run(ns2(render_days="20240423", upload=False, tol30=fake30), R, reg, log=lambda m: None)
        assert p3["watched_days_changed"] == {"20240422": {"identical_rules": False}}, p3["watched_days_changed"]
        # --identity B60nc end-to-end with a stand-in Tol_30: --noncalf (anchor json) reaches it, the hidden list is required
        # + passed through; B60nc may not be uploaded (unaccepted)
        fakeb = os.path.join(tmp, "fake_tol30_b60.py")
        with open(fakeb, "w", encoding="utf-8") as f:
            f.write(FAKE_T30.replace("@SRC@", src.replace("\\", "/")).replace(
                "assert \"--gt\" not in a", "assert a[a.index('--noncalf') + 1].endswith('noncalf_threshold_v1.json'); assert \"--gt\" not in a"))
        # --identity B60 (COLOUR filter) end-to-end on a change-control revision root holding a small SYNTHETIC day pair: the
        # 04-22 colour npz is COMPUTED from its crops_dense, the 04-23 one is read from the Volume (spot-checked); the stand-in
        # Tol_30 asserts --noncalf is the colour json and that every day's npz sits next to its bands, row-aligned
        import numpy as np
        snap54 = os.path.join(tmp, "t54_e2e.py"); shutil.copyfile(TOL54, snap54)
        M54, _ = load_tol54(snap54, json.load(open(COLOUR_JSON, encoding="utf-8")))
        r2 = "Tracking_v5a10_r2"; rng = np.random.default_rng(11)
        os.makedirs(os.path.join(vol, r2)); json.dump(rc, open(os.path.join(vol, r2, "ROOT_CONFIG.json"), "w"))
        for d in ("20240422", "20240423"):
            dd = os.path.join(vol, r2, d); os.makedirs(dd, exist_ok=True)
            Pd = make_colour_day(M54, vol, TOL, r2, d, [("0_001", 4), ("1_012", 5)], rng, coloured=("1_012",))
            np.savez(os.path.join(dd, BANDS), paths=np.array(Pd), emb=np.zeros((len(Pd), 8, 4), np.float16))
            for f in NEED[:3]:
                with open(os.path.join(dd, f), "w") as fh: fh.write("{}")      # never read by the stand-in
            json.dump({**rc, "written_by": d}, open(os.path.join(dd, "ROOT_CONFIG.json"), "w"))
            for k in range(EXP[d]): open(os.path.join(dd, f".done_clip{k:02d}"), "w").close()
        d2 = os.path.join(vol, r2, "20240423")
        M54.score_bands(os.path.join(d2, BANDS), os.path.join(d2, "crops_dense"), os.path.join(d2, COLOUR_NPZ), log_every=0)
        fakec = os.path.join(tmp, "fake_tol30_b60colour.py")
        chk = ("import numpy as np; root_ = a[a.index('--root') + 1]; "
               "assert a[a.index('--noncalf') + 1].endswith('noncalf_colour_v1.json'); "
               "assert all([str(x) for x in np.load(os.path.join(root_, d_, 'crop_colour_v1.npz'))['paths']] == "
               "[str(x) for x in np.load(os.path.join(root_, d_, 'crop_bands3_dense.npz'))['paths']] for d_ in days); "
               "assert \"--gt\" not in a")
        with open(fakec, "w", encoding="utf-8") as f:
            f.write(FAKE_T30.replace("@SRC@", src.replace("\\", "/")).replace("assert \"--gt\" not in a", chk))
        def set_hidden(h23):
            for d in ("20240422", "20240423"):
                J = json.load(open(os.path.join(src, f"relabel_week_{d}.json")))
                if h23 is None: J.pop("hidden", None)
                else: J["hidden"] = h23 if d == "20240423" else []
                json.dump(J, open(os.path.join(src, f"relabel_week_{d}.json"), "w"))
        ncc = lambda **kw: ns2(root=r2, render_days="20240423", upload=False, tol30=fakec, identity="B60",
                               colour_crops=os.path.join(tmp, "cc_e2e"), anchor_cache=os.path.join(tmp, "ac"), **kw)
        set_hidden([dict(hidc, **{"from": "2024-04-23T15:40:00", "to": "2024-04-23T15:40:00"})])
        p5 = run(ncc(), R, reg, log=lambda m: None)
        assert p5["identity_config"]["noncalf"]["mode"] == "colour" and f"--noncalf {COLOUR_JSON}" in p5["cmd"], p5["identity_config"]
        assert "unaccepted" not in p5["identity_config"], p5["identity_config"]
        assert f"--anchor_cache {os.path.join(tmp, 'ac')}" in p5["cmd"] and p5["anchor_cache"] == os.path.abspath(os.path.join(tmp, "ac"))
        try:
            run(ns2(render_days="20240423", upload=False, tol30=fake30, anchor_cache=os.path.join(tmp, "ac")), R, reg, log=lambda m: None)
            raise AssertionError("--anchor-cache accepted for the production identity")
        except Refused as e:
            assert "--anchor-cache needs" in str(e), e
        cd = p5["colour"]["days"]
        assert cd["20240422"]["source"] == "computed" and cd["20240422"]["crops"]["n_crops"] == 9, cd
        assert cd["20240423"]["source"] == "volume" and cd["20240423"]["spot_check"]["all_equal"], cd
        assert p5["colour"]["json_sha256"] == COLOUR_SHA and p5["colour"]["tol54"]["fn_sha256"]["read_rgb"] == TOL54_FN_SHA["read_rgb"]
        assert all(os.path.exists(os.path.join(p5["work"], "root", d, COLOUR_NPZ)) for d in ("20240422", "20240423"))
        assert p5["outputs"]["20240423"]["n_hidden"] == 1 and p5["outputs"]["20240423"]["hidden"][0]["mode"] == "colour", p5["outputs"]
        assert p5["outputs"]["20240422"]["n_hidden"] == 0
        for h23, frag in (([hid], "colour filter"), (None, "no 'hidden' list")):   # anchor-style entry / no list: refused
            set_hidden(h23)
            try:
                run(ncc(), R, reg, log=lambda m: None); raise AssertionError(f"B60 relabel accepted ({frag})")
            except RuntimeError as e:
                assert frag in str(e), (frag, e)
        set_hidden(None)
        try:
            run(ns2(render_days="20240423", upload=True, tol30=fakeb, identity="B60nc"), R, reg, log=lambda m: None)
            raise AssertionError("an upload with the unaccepted B60nc was accepted")
        except Refused as e:
            assert "may not be uploaded" in str(e), e
        for d in ("20240422", "20240423"):
            J = json.load(open(os.path.join(src, f"relabel_week_{d}.json"))); J["hidden"] = []
            json.dump(J, open(os.path.join(src, f"relabel_week_{d}.json"), "w"))
        p4 = run(ns2(render_days="20240423", upload=False, tol30=fakeb, identity="B60nc"), R, reg, log=lambda m: None)
        assert p4["identity_config"]["noncalf"]["sha256"] == NONCALF_SHA and "--noncalf" in p4["cmd"], p4["identity_config"]
        assert p4["identity_config"].get("unaccepted"), p4["identity_config"]
        assert all(p4["outputs"][d]["n_hidden"] == 0 for d in ("20240422", "20240423")), p4["outputs"]
        for d in ("20240422", "20240423"):                               # a B60nc relabel WITHOUT the hidden list is refused
            J = json.load(open(os.path.join(src, f"relabel_week_{d}.json"))); J.pop("hidden")
            json.dump(J, open(os.path.join(src, f"relabel_week_{d}.json"), "w"))
        try:
            run(ns2(render_days="20240423", upload=False, tol30=fakeb, identity="B60nc"), R, reg, log=lambda m: None)
            raise AssertionError("a B60nc relabel without 'hidden' was accepted")
        except RuntimeError as e:
            assert "no 'hidden' list" in str(e), e
        # a stand-in that does NOT print "<day>: no GT" (i.e. GT was visible to it) is refused
        with open(fake30, "w", encoding="utf-8") as f:
            f.write(FAKE_T30.replace("@SRC@", src.replace("\\", "/")).replace(": no GT", ": 99.00 % vs GT"))
        try:
            run(ns2(render_days="20240423", upload=False, tol30=fake30), R, reg, log=lambda m: None)
            raise AssertionError("a Tol_30 run that scored against GT was accepted")
        except Refused as e:
            assert "no GT" in str(e)
        # a GT file in the remote day dir (e.g. Tol_15_v2 ran on the root) refuses before anything runs
        open(os.path.join(vol, root, "20240423", "segment_assignments.json"), "w").close()
        try:
            run(ns2(render_days="20240423", upload=False), R, reg, log=lambda m: None); raise AssertionError("GT in root accepted")
        except Refused as e:
            assert "Tol_15_v2" in str(e)
        os.remove(os.path.join(vol, root, "20240423", "segment_assignments.json"))
        # an incomplete day refuses
        os.remove(os.path.join(vol, root, "20240423", ".done_clip00"))
        try:
            run(ns2(render_days="20240423", upload=False), R, reg, log=lambda m: None); raise AssertionError("incomplete day accepted")
        except Refused as e:
            assert "incomplete" in str(e)
        print(f"prod_relabel SELFTEST end-to-end (REAL Tol_30 week mode on {real_days} from {data}, FakeRemote upload + "
              f"read-back, overwrite refusal, watched-day diff (same / changed), GT-scoring Tol_30 refused, GT-in-root "
              f"refusal, incomplete-day refusal, B60 --noncalf colour (npz computed / Volume-read + spot-checked, hidden "
              f"list required, anchor-style entry refused), B60nc --noncalf anchor, B60nc upload + missing-hidden refusal): "
              f"OK ({time.time() - t0:.0f} s; "
              + ", ".join(f"{d}: {prov['outputs'][d]['n_rules']} rules / {prov['outputs'][d]['labelled_slot_h']} slot-h"
                          for d in real_days) + f"; {n_zero} zero-length rules tolerated)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.argv[1:2] == ["selftest"]:
        dflt = os.path.join(HERE, "data", "Tracking_v4v2")
        data = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else dflt
        selftest(data, full="--full" in sys.argv); print("prod_relabel SELFTEST PASSED"); sys.exit(0)
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["run"])
    ap.add_argument("--root", required=True); ap.add_argument("--days", required=True)
    ap.add_argument("--render-days", dest="render_days", required=True)
    ap.add_argument("--tol30", default=NB + "/Tol_30_periodic_identity.py")
    ap.add_argument("--tol30-sha", dest="tol30_sha", default=None)
    ap.add_argument("--work", default=os.path.join(os.path.dirname(HERE), "prod_relabel"))
    ap.add_argument("--require-gate", dest="require_gate", action="append", default=None,
                    help="an accepted G3 verdict (mandatory with --upload) + a G5 verdict per render day after 04-23")
    ap.add_argument("--upload", action="store_true"); ap.add_argument("--allow-overwrite", dest="allow_overwrite", action="store_true")
    ap.add_argument("--identity", choices=sorted(IDENTITY_CONFIGS), default="production",
                    help="identity-layer config: production (plain Tol_30), B60 (chosen 2026-09-28, + the COLOUR non-calf "
                         "filter since 2026-09-30) or B60nc (B60's base arm + the ANCHOR non-calf filter v1; FAILED acceptance "
                         "2026-09-29: dry runs only)")
    ap.add_argument("--anchor-cache", dest="anchor_cache", default=None,
                    help="B60 / B60nc: Tol_30 --anchor_cache (default <anchor pool dir>/score_cache); cache files are keyed and "
                         "validated by anchor-pool + band sha, so this changes where scores are cached, never an output")
    ap.add_argument("--colour-crops", dest="colour_crops", default=None,
                    help="--identity B60: local crops cache of THIS root, <dir>/<day>/<seg>/<HHMMSS>.jpg (default "
                         "<tempdir>/prc/<root>); a segment is reused only if every file matches the Volume listing's size")
    a = ap.parse_args()
    reg = v5_remote.launcher_registry(LAUNCHER)
    try:
        run(a, v5_remote.CliRemote(), reg)
    except Refused as e:
        print(f"REFUSED: {e}"); sys.exit(2)
