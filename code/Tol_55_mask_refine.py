# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N55 — offline mask refinement of `Tracking_v5a10` (geometry only, no re-tracking)
# MAGIC
# MAGIC Implements the mask-refinement design v1 (2026-10-02). It keeps v5's tracking and identities as they are and
# MAGIC improves the masks where v5 holds only a FRAGMENT of a calf. Every decision lives in
# MAGIC `tol_refine_core.py` (imported from this notebook's workspace directory, sha256 echoed); this notebook is the
# MAGIC Databricks driver.
# MAGIC
# MAGIC **Invariants (asserted on every published clip).** Frames, `frame_idx`, `t_sec`, slot keys, segments,
# MAGIC events and the identity relabels are untouched; a mask that is not refined stays byte-identical; a refined mask
# MAGIC contains ≥ `TAU_CONTAIN` of the original fragment and overlaps no other slot. Refinement never reads identity
# MAGIC (I7), so it does not wait for the identity layer. `Tracking_v5a10` is read-only (I8).
# MAGIC
# MAGIC **Modes** (`mode` widget)
# MAGIC | mode | stage | GPU | what |
# MAGIC |---|---|---|---|
# MAGIC | `census` | S0 | no | exact trigger-row / new-call counts per clip (the GPU budget) |
# MAGIC | `derive` | S1 | yes | RF-DETR at 0.5 Hz on the 28 detector-UNSEEN ∪ VALID training clips + colour sampling of all 46 → frozen `Tol/Metadata/<constants_tag>/` (TAU_CONTAIN, TAU_CONFLICT, CARRY_MAX_S, CHROMA_MIN, LUT, LLR_MIN, STATIC, BL_P99, SOL_P1, B ship gate). Opens no test-day path. |
# MAGIC | `detect` | S2 | yes | E0 parity on every stored production call of the clip, then RF-DETR (exact production contract) on every trigger frame without a stored call → `detections_refine_<clip>.json.gz`. Needs only READ constants: can run on all 7 days now. |
# MAGIC | `apply` | S3 | no | gates (A) + carry → `<out_root>/<day>/` (+ review crops) |
# MAGIC | `redetect` | S2+S3 | yes | both, per clip, one video staging |
# MAGIC | `grow` | S4 | no | colour grow (B) on the `nodet` daylight rows → `<grow_root>/<day>/` (experimental; refuses unless the B ship gate passed) |
# MAGIC | `both` | S2+S3+S4 | yes | everything per clip |
# MAGIC
# MAGIC **Frames** are decoded from the source video with production's `seek_thread` calls (`tol_v5_core.iter_frames`), so
# MAGIC the detector sees the pixels production saw (E0 checks it bit-for-bit on the same GPU class). The JPEG
# MAGIC `Frames_5fps` tars are not used (re-encoded pixels, 7 GB per clip, no byte offsets). Videos are staged with
# MAGIC `dbutils.fs.cp` to `/local_disk0` (timed separately); every output is built on local disk and copied to the Volume
# MAGIC (UC Volumes: no append, no seek-on-write).
# MAGIC
# MAGIC Local check: `TOL_SELFTEST=1 python Tol_55_mask_refine.py` (no GPU / dbutils / cv2 / torch: the core selftest plus
# MAGIC the detect → apply → grow drivers of this notebook on a synthetic day with a stub detector).

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm" "scipy" "rfdetr==1.10.1" "pycocotools" "peft"
# MAGIC %pip uninstall -y opencv-python opencv-contrib-python
# MAGIC %pip install --no-deps --force-reinstall "opencv-python-headless==4.11.0.86"

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

import os, os.path as osp, sys, io, re, json, time, math, shutil, tarfile, tempfile, bisect
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before anything touches CUDA
SELFTEST = os.environ.get("TOL_SELFTEST") == "1"
dbutils = globals().get("dbutils")           # the Databricks notebook global (None when run as a script)
IS_NOTEBOOK = dbutils is not None
if hasattr(sys.stdout, "reconfigure") and not IS_NOTEBOOK:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def _core_dir():
    """tol_v5_core + tol_refine_core from EXACTLY one directory (Tol_47 pattern): this notebook's workspace directory,
    or the directory of this file when run as a script (SELFTEST)."""
    if not IS_NOTEBOOK:
        return osp.dirname(osp.abspath(__file__))
    try:
        return "/Workspace" + osp.dirname(
            dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
    except Exception as e:
        print(f"⚠ notebook path unreadable ({e!r}): cores from the default Workspace directory")
        return "/Workspace/Users/<DATABRICKS_USER>/Model Distillation/Testing on Tol2"
CORE_DIR = _core_dir()
for _m in ("tol_v5_core.py", "tol_refine_core.py"):
    if not osp.isfile(osp.join(CORE_DIR, _m)):
        raise RuntimeError(f"{_m} is not in {CORE_DIR} (deploy it beside Tol_55 as a workspace FILE)")
while CORE_DIR in sys.path: sys.path.remove(CORE_DIR)
sys.path.insert(0, CORE_DIR)
for _m in ("tol_v5_core", "tol_refine_core"): sys.modules.pop(_m, None)
import tol_v5_core as V5
import tol_refine_core as RC
for _mod in (V5, RC):
    if osp.normcase(osp.realpath(osp.dirname(osp.abspath(_mod.__file__)))) != osp.normcase(osp.realpath(CORE_DIR)):
        raise RuntimeError(f"{_mod.__name__} resolved to {_mod.__file__}, not {CORE_DIR}")
V5_SHA, RC_SHA = V5.sha256_file(V5.__file__), V5.sha256_file(RC.__file__)
print(f"tol_v5_core {V5.CORE_VERSION} sha256 {V5_SHA[:16]} | tol_refine_core {RC.REFINE_CORE_VERSION} sha256 {RC_SHA[:16]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drivers (pure Python; all I/O through an `Env`, so the SELFTEST runs them unchanged)

# COMMAND ----------

class Env:
    """Injected I/O. Volume paths are POSIX paths readable through the FUSE mount (os.path works); bulk copies go
    through stage_small / publish (Databricks: dbutils.fs.cp with retry). open_video(local) returns an object with
    .frames(fidxs) -> iterator of (fidx, ok, rgb uint8 HxWx3) and .close(); detect(rgb) -> (rows [(bool mask, score)],
    kept_idx, det_ms); jpeg(rgb) -> bytes."""
    def __init__(self, stage_small, publish, stage_video, open_video, detect, jpeg, log=print):
        self.stage_small, self.publish, self.stage_video = stage_small, publish, stage_video
        self.open_video, self.detect, self.jpeg, self.log = open_video, detect, jpeg, log

def _local(ctx, name):
    return osp.join(ctx["LOCAL"], name)

def write_json_local(path, obj, gz=False):
    if gz:
        RC.dump_json_gz(path, obj)
    else:
        with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=1, default=str)
    return path

def publish_json(env, ctx, obj, vol_path, gz=False):
    loc = write_json_local(_local(ctx, osp.basename(vol_path)), obj, gz=gz)
    env.publish(loc, vol_path)
    n = osp.getsize(loc); os.remove(loc)
    return n

def touch_marker(env, ctx, vol_path):
    loc = _local(ctx, osp.basename(vol_path))
    open(loc, "w").close(); env.publish(loc, vol_path); os.remove(loc)

def source_clips(ctx):
    """Clips of the source day with a source .done_ marker, ascending by start (masks_<stem>.json.gz names)."""
    d = ctx["SRC_DIR"]
    stems = [RC.clip_stem(n) for n in os.listdir(d) if n.startswith("masks_") and n.endswith(".json.gz")]
    done = {n[len(".done_"):] for n in os.listdir(d) if n.startswith(".done_")}
    ok = sorted((s for s in stems if s in done), key=RC.clip_start)
    skipped = sorted(set(stems) - done)
    return ok, skipped

def load_clip_inputs(env, ctx, stem):
    """stage + load the source masks and the production detections of one clip."""
    lm = _local(ctx, f"src_masks_{stem}.json.gz")
    env.stage_small(osp.join(ctx["SRC_DIR"], f"masks_{stem}.json.gz"), lm)
    src = RC.load_json_gz(lm)
    ctx.setdefault("src_sha", {})[stem] = RC.sha256_file(lm)   # of the staged copy: no second ~37 MB read over FUSE
    os.remove(lm)
    dv = osp.join(ctx["SRC_DIR"], f"detections_{stem}.json.gz")
    stored = {}
    if osp.exists(dv):
        ld = _local(ctx, f"src_det_{stem}.json.gz"); env.stage_small(dv, ld)
        stored = RC.stored_by_fidx(RC.load_json_gz(ld)); os.remove(ld)
    elif not ctx.get("allow_no_stored"):
        raise RuntimeError(f"{stem}: production detections_{stem}.json.gz missing in {ctx['SRC_DIR']}")
    return src, stored

class Encoder:
    """raw RF-DETR masks -> production RLE (tol_v5_core.rle_fast == rle_encode, S1) on worker threads."""
    def __init__(self, n=4):
        self.ex = ThreadPoolExecutor(max_workers=n); self.n = n
    @staticmethod
    def _enc(rows):
        cache = V5.MaskCache()
        out = [(V5.rle_fast(m, cache), float(s), int(V5.area_fast(m, cache))) for m, s in rows]
        return out
    def submit(self, rows): return self.ex.submit(self._enc, rows)
    def close(self): self.ex.shutdown(wait=True)

def detect_clip(env, ctx, stem, src, stored, video, enc):
    """S2 on one clip: E0 parity on every stored production call (gate: strict -> raise on the first failing call),
    then one RF-DETR call per planned trigger frame. Returns (records {fidx: rec}, parity list, stats)."""
    C = ctx["C"]; clip = src["clip"]; frames = src["frames"]
    t_by_f = {int(fr["frame_idx"]): fr["t_sec"] for fr in frames}
    w0 = RC.clip_start(clip)
    sel = RC.sample_selection(frames, ctx["sample"])
    plan = RC.plan_new_calls(frames, clip, C, stored.keys(), sel)
    st = {"plan": len(plan), "stored": len(stored), "det_ms": [], "decode_wait_s": 0.0, "encode_wait_s": 0.0,
          "decode_fail": 0, "parity_n": 0, "parity_fail": 0}
    parity = []
    pf = sorted(stored) if ctx["parity"] != "off" else []
    if ctx.get("parity_max_per_clip"): pf = pf[:int(ctx["parity_max_per_clip"])]
    def run(fidxs, kind):
        out = {}; pend = []
        t_last = time.perf_counter()
        for f, ok, rgb in video.frames(fidxs):
            st["decode_wait_s"] += time.perf_counter() - t_last
            t_sec = t_by_f.get(f, None)
            pool = RC.pool_of(RC.wall_of(w0, t_sec)) if t_sec is not None else "dayl"
            if not ok:
                st["decode_fail"] += 1
                out[f] = RC.det_record(clip, f, t_sec, [], [], 0.0, pool, C, kind=kind, decode_ok=False)
            else:
                rows, kidx, ms = env.detect(rgb)
                st["det_ms"].append(ms)
                pend.append((f, t_sec, pool, kidx, ms, enc.submit(rows)))
                if len(pend) > 32:
                    t0 = time.perf_counter(); f0, t0s, p0, k0, m0, fut = pend.pop(0)
                    out[f0] = RC.det_record(clip, f0, t0s, fut.result(), k0, m0, p0, C, kind=kind)
                    st["encode_wait_s"] += time.perf_counter() - t0
            t_last = time.perf_counter()
        for f0, t0s, p0, k0, m0, fut in pend:
            t0 = time.perf_counter()
            out[f0] = RC.det_record(clip, f0, t0s, fut.result(), k0, m0, p0, C, kind=kind)
            st["encode_wait_s"] += time.perf_counter() - t0
        return out
    if pf:
        got = run(pf, "parity")
        for f in pf:
            r = RC.parity_compare(stored[f], got[f], exact=ctx["parity_exact"]) if got[f].get("decode_ok", True) else \
                {"ok": False, "why": ["decode failed"], "frame_idx": f}
            parity.append(r); st["parity_n"] += 1
            if not r["ok"]: st["parity_fail"] += 1
        st["parity_order_differs"] = sum(1 for r in parity if r.get("order_differs"))
        fails = [r for r in parity if not r["ok"]]
        if fails and ctx["parity"] == "strict":
            # every parity call of the clip is compared first; the evidence (all comparisons + the new raw records) is
            # published before the loud stop, so the failure can be inspected without re-running the GPU
            cb = ctx.get("on_parity_fail")
            if cb is not None: cb(stem, parity, got, st)
            r = fails[0]
            raise RuntimeError(f"E0 PARITY FAIL {stem}: {len(fails)}/{len(parity)} stored calls, first frame "
                               f"{r['frame_idx']}: {r['why']} -- the detector path does not reproduce production "
                               f"(exact={ctx['parity_exact']}); nothing else runs")
    recs = run(plan, "refine") if plan else {}
    return recs, parity, st

def det_file_obj(ctx, stem, recs, extra):
    C = ctx["C"]
    return {"clip": stem, "detector": "rfdetr", "det_run_tag": ctx["det_run_tag"], "op_threshold": C["op_threshold"],
            "band_px": {p: [RC.px(C, "FRAG_LO", p), C["OVER_HI"] * C["A_REF"]] for p in ("dayl", "dark")},
            "kind": "refine", "sample": ctx["sample"], "gpu_class": ctx.get("gpu_class"), "refine": extra,
            "records": [recs[f] for f in sorted(recs)]}

def load_new_records(env, ctx, stem):
    vp = osp.join(ctx["OUT_DIR"], f"detections_refine_{stem}.json.gz")
    if not osp.exists(vp) or not osp.exists(osp.join(ctx["OUT_DIR"], f".det_done_{stem}")):
        raise RuntimeError(f"{stem}: no finished detections_refine in {ctx['OUT_DIR']} -- run mode=detect (S2) first")
    lp = _local(ctx, f"dref_{stem}.json.gz"); env.stage_small(vp, lp)
    d = RC.load_json_gz(lp); os.remove(lp)
    if d.get("sample") != ctx["sample"]:
        raise RuntimeError(f"{stem}: detections_refine sample {d.get('sample')!r} != {ctx['sample']!r}")
    return {int(r["frame_idx"]): r for r in d["records"]}

def pick_sheet_rows(side, extra, cases, stem):
    """E8 review crops: <= SHEET_PER_CAT_PER_CLIP per (code x phase-group) and per refusal reason (middle row
    of its list: deterministic), the listed measurement cases, and the double-claim rows (one per 60 s)."""
    cat = defaultdict(list)
    for r in side:
        grp = "day" if r["phase"] in ("day", "dawn") else "dark"
        if r["code"] in ("d", "c", "g"): cat[f"{r['code']}_{grp}"].append(r)
        else: cat[f"r_{r['reason']}"].append(r)
    picks = []
    k = RC.DESIGN["SHEET_PER_CAT_PER_CLIP"]
    for name in sorted(cat):
        L = cat[name]
        for j in range(min(k, len(L))):
            r = L[(len(L) * (2 * j + 1)) // (2 * k)] if k > 1 else L[len(L) // 2]
            picks.append(("cat", name, r["frame_idx"], r["slot"]))
    for c in cases or []:
        if c[0] == stem: picks.append(("case", f"case{c[3] if len(c) > 3 else ''}", int(c[1]), int(c[2])))
    last = None
    for z in sorted(extra.get("doubleclaim_rows") or [], key=lambda q: q["ts"]):
        if last is None or z["ts"] - last >= 60.0:
            picks.append(("dclaim", "doubleclaim", int(z["fidx"]), -1)); last = z["ts"]
    return picks

def write_sheets(env, ctx, stem, src, out, picks, video, stored, new):
    """crop JPEGs (raw pixels only; Tol_56 draws the overlays) + an index -> <out>/sheets/sheet_<stem>.tar"""
    if not picks or video is None: return 0
    by_f = defaultdict(list)
    for p in picks: by_f[p[2]].append(p)
    fr_by = {int(fs["frame_idx"]): (fs, fo) for fs, fo in zip(src["frames"], out["frames"])}
    idx = []
    tp = _local(ctx, f"sheet_{stem}.tar")
    m = RC.DESIGN["SHEET_MARGIN_PX"]
    with tarfile.open(tp, "w") as tf:
        for f, ok, rgb in video.frames(sorted(f for f in by_f if f in fr_by)):   # a listed case outside this clip's
            if not ok: continue                                                 # frames is skipped, never a crash
            fs, fo = fr_by[f]
            rec = stored.get(f) or new.get(f)
            dets = RC.kept_runs(rec) if rec is not None and rec.get("decode_ok", True) else []
            for kind, label, _f, slot in by_f[f]:
                if slot < 0:
                    box = (0, RC.H, 0, RC.W)
                else:
                    Ms = RC.runs_of(fs["masks"][str(slot)]) if fs["masks"].get(str(slot)) else None
                    Mo = RC.runs_of(fo["masks"][str(slot)]) if fo["masks"].get(str(slot)) else None
                    ext = [d for d in dets if Ms is not None and RC.inter(Ms, d)]
                    if Ms is None: continue
                    box = RC.crop_box([Ms, Mo] if Mo is not None else [Ms], m, extra=ext)
                y0, y1, x0, x1 = box
                b = env.jpeg(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
                name = f"{stem}/{f:07d}_s{slot}_{label}.jpg"
                ti = tarfile.TarInfo(name); ti.size = len(b); ti.mtime = 0
                tf.addfile(ti, io.BytesIO(b))
                idx.append({"name": name, "kind": kind, "label": label, "frame_idx": f, "slot": slot, "box": list(box)})
        jb = json.dumps({"clip": stem, "items": idx}).encode()
        ti = tarfile.TarInfo(f"{stem}/index.json"); ti.size = len(jb); ti.mtime = 0; tf.addfile(ti, io.BytesIO(jb))
    os.makedirs(osp.join(ctx["OUT_DIR"], "sheets"), exist_ok=True)
    env.publish(tp, osp.join(ctx["OUT_DIR"], "sheets", f"sheet_{stem}.tar"))
    n = osp.getsize(tp); os.remove(tp)
    return n

def refine_block(ctx, src_sha, counts, extra=None):
    return {"spec": RC.REFINE_SPEC, "core_version": RC.REFINE_CORE_VERSION, "source_root": ctx["src_root"],
            "no_shrink": ctx["C"].get("NO_SHRINK", True), "no_shrink_note": RC.NO_SHRINK_NOTE,
            "source_masks_sha256": src_sha, "constants_sha256": ctx["C"].get("constants_sha256"),
            "detector_sha256": ctx.get("det_sha"), "refine_core_sha256": RC_SHA, "v5_core_sha256": V5_SHA,
            "notebook_source_norm_sha256": ctx.get("src_norm_sha"), "gpu_class": ctx.get("gpu_class"),
            "sample": ctx["sample"], "codes": dict(counts), "created": datetime.now().isoformat(timespec="seconds"),
            **(extra or {})}

def apply_clip_driver(env, ctx, stem, src, stored, new, video=None):
    """S3 on one clip: gates + carry -> refined masks + sidecar (+ review crops), asserts, publish, .done_ marker."""
    C = ctx["C"]; t0 = time.perf_counter()
    sel = RC.sample_selection(src["frames"], ctx["sample"])
    out, side, extra = RC.apply_clip(src, stored, new, ctx["segs"], C, sample_sel=sel)
    t_apply = time.perf_counter() - t0
    RC.verify_clip_output(src, out, side, C)
    t_ver = time.perf_counter() - t0 - t_apply
    src_sha = ctx["src_sha"].get(stem)
    out["refine"] = refine_block(ctx, src_sha, extra["counts"])
    nb = publish_json(env, ctx, out, osp.join(ctx["OUT_DIR"], f"masks_{stem}.json.gz"), gz=True)
    sc = {"clip": stem, "spec": RC.REFINE_SPEC, "constants_sha256": C.get("constants_sha256"), "sample": ctx["sample"],
          "n_trigger_rows": len(side), "records": side, "doubleclaim_rows": extra["doubleclaim_rows"],
          "orphans": extra["orphans"], "n_calls": extra["n_calls"], "counts": extra["counts"]}
    nb += publish_json(env, ctx, sc, osp.join(ctx["OUT_DIR"], f"refine_{stem}.json.gz"), gz=True)
    t_pub = time.perf_counter() - t0 - t_apply - t_ver
    ns = 0
    if ctx["sheets"]:
        picks = pick_sheet_rows(side, extra, ctx.get("sheet_cases"), stem)
        ns = write_sheets(env, ctx, stem, src, out, picks, video, stored, new)
    touch_marker(env, ctx, osp.join(ctx["OUT_DIR"], f".done_{stem}"))
    reasons = Counter(r["reason"] for r in side if r["code"] == "r")
    return {"apply_s": round(t_apply, 2), "verify_s": round(t_ver, 2), "publish_s": round(t_pub, 2), "bytes": nb,
            "sheet_bytes": ns, "codes": extra["counts"], "refused_by_reason": dict(reasons), "n_trigger_rows": len(side),
            "n_doubleclaim_rows": len(extra["doubleclaim_rows"]), "n_orphans": len(extra["orphans"]), "n_calls": extra["n_calls"]}

def grow_clip_driver(env, ctx, stem, src, stored, new, video):
    """S4 on one clip: (B) colour grow on r/nodet day rows of the refined root -> <grow_root>/<day>/."""
    C = ctx["C"]; G = ctx["grow"]; t0 = time.perf_counter()
    lm = _local(ctx, f"ref_masks_{stem}.json.gz"); env.stage_small(osp.join(ctx["OUT_DIR"], f"masks_{stem}.json.gz"), lm)
    ref = RC.load_json_gz(lm); os.remove(lm)
    ls = _local(ctx, f"ref_side_{stem}.json.gz"); env.stage_small(osp.join(ctx["OUT_DIR"], f"refine_{stem}.json.gz"), ls)
    sc = RC.load_json_gz(ls); os.remove(ls)
    w0 = RC.clip_start(src["clip"])
    frames_s, frames_r = src["frames"], ref["frames"]
    pos = {int(fr["frame_idx"]): i for i, fr in enumerate(frames_s)}
    side = sc["records"]
    elig = defaultdict(list)
    for k, r in enumerate(side):
        if not RC.grow_eligible(r): continue
        if RC.phase_of(RC.wall_of(w0, r["t_sec"])) != "day":
            r["reason"] = "grow_ineligible_phase"; continue
        elig[r["frame_idx"]].append(k)
    # DYN(clip): N_BG_FRAMES evenly spaced day-phase colour frames, excluding every slot mask and every detection
    day_i = [i for i, fr in enumerate(frames_s) if RC.phase_of(RC.wall_of(w0, fr["t_sec"])) == "day"]
    nbg = RC.DESIGN["N_BG_FRAMES"]
    bg_i = [day_i[int(j)] for j in np.linspace(0, len(day_i) - 1, nbg).round().astype(int)] if len(day_i) >= nbg else day_i
    stats = {"eligible_rows": sum(len(v) for v in elig.values()), "bg_frames": len(bg_i), "grown": 0,
             "refused": Counter(), "shift": None}
    def excl_of(i):
        f = int(frames_s[i]["frame_idx"]); m = np.zeros((RC.H, RC.W), bool)
        for fr in (frames_s[i], frames_r[i]):
            for v in (fr.get("masks") or {}).values():
                if v: m |= RC.runs_to_dense(RC.runs_of(v))
        rec = stored.get(f) or new.get(f)
        for d in (RC.kept_runs(rec) or []) if rec is not None and rec.get("decode_ok", True) else []:
            m |= RC.runs_to_dense(d)
        return m
    dyn = None; static = G["static_full"]
    if elig and bg_i:
        bgf, bge = [], []
        for f, ok, rgb in video.frames([int(frames_s[i]["frame_idx"]) for i in bg_i]):
            if not ok: continue
            i = pos[f]; ex = excl_of(i)
            ch = RC.chroma_of(rgb, ex)
            if ch is None or ch < C["CHROMA_MIN"]: continue
            bgf.append(rgb); bge.append(ex)
        stats["bg_colour_frames"] = len(bgf)
        if len(bgf) >= RC.DESIGN["N_BG_MIN"]:
            dsub, med, cnt = RC.dyn_structure(bgf, bge, G["lut"], C["LLR_MIN"])
            dy, dx, act = RC.bg_shift_check(G["train_bg_sub"], med)
            stats["shift"] = {"dy": dy, "dx": dx, "action": act}
            if act == "shift": static = RC.shift_mask(G["static_full"], dy, dx)
            dyn = RC.upsample_mask(dsub)
            for nm, arr in (("bg", med), ("dyn", dsub)):
                lp = _local(ctx, f"{nm}_{stem}.png")
                with open(lp, "wb") as fh: fh.write(RC.png_bytes(arr))
                env.publish(lp, osp.join(ctx["GROW_DIR"], f"{nm}_{stem}.png")); os.remove(lp)
        else:
            stats["no_dyn"] = True
    out = {k: v for k, v in ref.items()}
    out_frames = [dict(fr, masks=dict(fr["masks"]), prov=dict(fr["prov"])) for fr in frames_r]
    if elig and dyn is not None:
        for f, ok, rgb in video.frames(sorted(elig)):
            i = pos[f]
            for k in elig[f]:
                r = side[k]; s = r["slot"]
                if not ok: r["reason"] = "grow_ineligible_chroma"; r["grow"] = {"decode_ok": False}; continue
                ex = excl_of(i)
                ch = RC.chroma_of(rgb, ex)
                if ch is None or ch < C["CHROMA_MIN"]:
                    r["reason"] = "grow_ineligible_chroma"; r["grow"] = {"chroma": ch}; stats["refused"]["chroma"] += 1; continue
                Ms = RC.runs_of(frames_s[i]["masks"][str(s)])
                oth = []
                for t in range(RC.N_SLOTS):
                    if t == s: continue
                    for fr in (frames_s[i], out_frames[i]):
                        v = fr["masks"].get(str(t))
                        if v: oth.append(RC.runs_of(v))
                rec = stored.get(f) or new.get(f)
                dets = RC.kept_runs(rec) or []
                Gr, info = RC.grow_row(rgb, Ms, oth, dets, static, dyn, G["lut"], C["LLR_MIN"], C, pool=r["pool"])
                info["chroma"] = ch; r["grow"] = info
                if Gr is None:
                    r["reason"] = info["why"]; stats["refused"][info["why"]] += 1
                else:
                    r["code"] = "g"; r["reason"] = None; r["area_after"] = Gr.area
                    out_frames[i]["masks"][str(s)] = RC.rle_from_runs(Gr); out_frames[i]["prov"][str(s)] = "g"
                    stats["grown"] += 1
    elif elig:
        for ks in elig.values():
            for k in ks: side[k]["reason"] = "grow_ineligible_chroma"; side[k]["grow"] = {"no_dyn": True}
    out["frames"] = out_frames
    RC.verify_clip_output(src, out, side, C, allowed_codes=("k", "r", "d", "c", "g"), grow_floor_trig=True)
    out["refine"] = dict(ref.get("refine") or {}, grow={"core_version": RC.REFINE_CORE_VERSION, "grown": stats["grown"],
                                                       "constants_sha256": C.get("constants_sha256")})
    nb = publish_json(env, ctx, out, osp.join(ctx["GROW_DIR"], f"masks_{stem}.json.gz"), gz=True)
    sc2 = dict(sc, records=side, grow_stats={k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()})
    nb += publish_json(env, ctx, sc2, osp.join(ctx["GROW_DIR"], f"refine_{stem}.json.gz"), gz=True)
    touch_marker(env, ctx, osp.join(ctx["GROW_DIR"], f".done_{stem}"))
    stats["refused"] = dict(stats["refused"]); stats["grow_s"] = round(time.perf_counter() - t0, 2); stats["bytes"] = nb
    return stats

I4_FILES = ("segments.json", "events.jsonl", "ROOT_CONFIG.json", "observations.npz", "observations_meta.json",
            "crop_bands3_dense.npz")

def finalize_day(env, ctx, dest_dir, stems_done, sidecar_dir=None):
    """day-level files once every source clip of the day is done: I4 verbatim copies (sha-asserted), doubleclaim_<day>,
    orphan summary."""
    rep = {"i4": {}}
    for fn in I4_FILES:
        sp, dp = osp.join(ctx["SRC_DIR"], fn), osp.join(dest_dir, fn)
        if not osp.exists(sp): rep["i4"][fn] = "absent_in_source"; continue
        s_sha = RC.sha256_file(sp)
        if not (osp.exists(dp) and RC.sha256_file(dp) == s_sha):
            lp = _local(ctx, f"i4_{fn}"); env.stage_small(sp, lp); env.publish(lp, dp); os.remove(lp)
        d_sha = RC.sha256_file(dp)
        if d_sha != s_sha: raise RuntimeError(f"I4: {fn} copy sha {d_sha[:12]} != source {s_sha[:12]}")
        rep["i4"][fn] = s_sha
    rows, orph = [], 0
    for stem in stems_done:
        lp = _local(ctx, f"sc_{stem}.json.gz"); env.stage_small(osp.join(sidecar_dir or dest_dir, f"refine_{stem}.json.gz"), lp)
        sc = RC.load_json_gz(lp); os.remove(lp)
        rows += sc.get("doubleclaim_rows") or []; orph += len(sc.get("orphans") or [])
    iv = RC.doubleclaim_intervals(rows)
    publish_json(env, ctx, {"day": ctx["day"], "spec": RC.REFINE_SPEC, "rule": "one in-band kept detection contains >= "
                            "TAU_CONTAIN of two slots' masks; rows <= DOUBLECLAIM_GAP_S apart; intervals >= DOUBLECLAIM_MIN_S",
                            "design": {k: RC.DESIGN[k] for k in ("DOUBLECLAIM_MIN_S", "DOUBLECLAIM_GAP_S")},
                            "n_rows": len(rows), "intervals": iv, "orphan_detections": orph,
                            "note": "identity-defect candidates for review (§9.5); never acted on"},
                 osp.join(dest_dir, f"doubleclaim_{ctx['day']}.json"))
    rep.update(doubleclaim_intervals=len(iv), doubleclaim_rows=len(rows), orphans=orph)
    return rep

REFINE_CONFIG_GUARD = ("spec", "src_root", "sample", "constants_sha256", "detector_sha256", "refine_core_sha256", "det_run_tag",
                       "no_shrink")

def check_refine_config(env, ctx, dest_dir, rec):
    """one config per refined day dir (Tol_47 ROOT_CONFIG pattern): write-if-absent, then compare the guarded keys."""
    p = osp.join(dest_dir, "REFINE_CONFIG.json")
    if not osp.exists(p):
        publish_json(env, ctx, rec, p)
        return "written"
    old = RC.load_json(p)
    diff = [(k, old.get(k), rec.get(k)) for k in REFINE_CONFIG_GUARD if old.get(k) != rec.get(k)]
    if diff:
        raise RuntimeError(f"REFINE_CONFIG mismatch in {dest_dir} (a day dir holds ONE configuration): {diff}")
    return "matches"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local SELFTEST (exits here when `TOL_SELFTEST=1`)

# COMMAND ----------

def _selftest_pipeline():
    """The drivers above on a synthetic two-clip day: stub detector (exactly reproduces the stored calls -> E0 passes),
    fake video (synthetic RGB scene), local dirs as the 'Volume'. Checks codes, invariants, resume, parity failure, grow."""
    from PIL import Image
    ck = RC._Check(); C = RC.selftest_constants(); calves = RC._scene()
    rng = np.random.default_rng(1)
    img = np.zeros((RC.H, RC.W, 3), np.uint8); img[...] = (190, 160, 90)
    img += rng.integers(0, 12, (RC.H, RC.W, 3), dtype=np.uint8)
    patt = (np.add.outer(np.arange(RC.H) // 40, np.arange(RC.W) // 40) % 2).astype(bool)
    for m in calves: img[m & patt] = (20, 20, 20); img[m & ~patt] = (235, 235, 235)
    ys, xs = np.nonzero(calves[0]); frag = np.zeros((RC.H, RC.W), bool); s_ = xs < np.quantile(xs, 0.3); frag[ys[s_], xs[s_]] = True
    ys3, xs3 = np.nonzero(calves[3]); frag3 = np.zeros((RC.H, RC.W), bool); s3 = ys3 < np.quantile(ys3, 0.35); frag3[ys3[s3], xs3[s3]] = True
    td = tempfile.mkdtemp(prefix="t55_")
    vol, loc = osp.join(td, "vol"), osp.join(td, "loc")
    SRC, OUT, GRW = osp.join(vol, "Tracking_v5a10", "20240422"), osp.join(vol, "Tracking_ref", "20240422"), osp.join(vol, "Tracking_grow", "20240422")
    for d in (SRC, OUT, GRW, loc): os.makedirs(d, exist_ok=True)
    clips = ["Tol_ch01-20240422-140000-140010.mp4", "Tol_ch01-20240422-140010-140020.mp4"]
    det_counter = Counter()
    def scene_dets(f):            # the 'truth' of the stub detector at native frame f
        if 60 <= f < 70: return calves[1:]                    # calf 0 unseen for 2 rows -> nodet (carried)
        if f == 95: return [calves[0] | calves[1]] + calves[2:]   # merged blob -> big
        return calves
    for ci, clip in enumerate(clips):
        seq = []
        for i in range(50):
            m0 = frag if 5 <= i < 40 else calves[0]
            m3 = frag3 if (ci == 1 and 10 <= i < 20) else calves[3]
            seq.append([m0, calves[1], calves[2], m3])
        src = RC._mk_src(clip, seq)
        RC.dump_json_gz(osp.join(SRC, f"masks_{RC.clip_stem(clip)}.json.gz"), src)
        recs = [RC._det_rec(calves, f, f / 25.0, "dayl", C) for f in (0, 50, 100, 150, 200)]
        for r in recs: r["clip"] = clip
        RC.dump_json_gz(osp.join(SRC, f"detections_{RC.clip_stem(clip)}.json.gz"), {"clip": RC.clip_stem(clip), "records": recs})
        open(osp.join(SRC, f".done_{RC.clip_stem(clip)}"), "w").close()
    segs = {"day": "20240422", "segments": [{"seg_id": f"{s}_001", "slot": s, "t_start_wall": "2024-04-22T14:00:00",
                                             "t_end_wall": "2024-04-22T14:00:20"} for s in range(4)]}
    with open(osp.join(SRC, "segments.json"), "w") as f: json.dump(segs, f)
    with open(osp.join(SRC, "events.jsonl"), "w") as f: f.write('{"type": "x"}\n')
    with open(osp.join(SRC, "ROOT_CONFIG.json"), "w") as f: json.dump({"gpu_class": "A10"}, f)
    class FakeVideo:
        def __init__(self, path): self.path = path
        def frames(self, fidxs):
            for f in fidxs: yield f, True, img
        def close(self): pass
    perturb = {"on": False}
    def stub_detect(rgb):
        f = stub_detect.cur
        ms = scene_dets(f)
        if perturb["on"]: ms = [ms[1], ms[0]] + ms[2:]
        det_counter["calls"] += 1
        return [(m, 0.9) for m in ms], V5.post_chain_idx([(m, 0.9) for m in ms], topk=RC.DET_TOPK)[0], 5.0
    stub_detect.cur = None
    class TrackingVideo(FakeVideo):
        def frames(self, fidxs):
            for f in fidxs:
                stub_detect.cur = f; yield f, True, img
    def jpeg(rgb):
        b = io.BytesIO(); Image.fromarray(rgb).save(b, format="JPEG", quality=85); return b.getvalue()
    env = Env(stage_small=lambda s, d: shutil.copyfile(s, d), publish=lambda s, d: (os.makedirs(osp.dirname(d), exist_ok=True), shutil.copyfile(s, d)),
              stage_video=lambda v, d: 0.0, open_video=lambda p: TrackingVideo(p), detect=stub_detect, jpeg=jpeg)
    ctx = {"LOCAL": loc, "SRC_DIR": SRC, "OUT_DIR": OUT, "GROW_DIR": GRW, "C": C, "sample": "1", "parity": "strict",
           "parity_exact": True, "sheets": True, "sheet_cases": [[RC.clip_stem(clips[0]), 100, 0, 1]], "day": "20240422",
           "src_root": "Tracking_v5a10", "det_run_tag": RC.DET_RUN_TAG, "segs": RC.SegIndex(segs), "src_sha": {},
           "gpu_class": "A10", "det_sha": "stub"}
    stems, skipped = source_clips(ctx)
    ck(stems == [RC.clip_stem(c) for c in clips] and not skipped, f"source clips {stems}")
    enc = Encoder(2)
    for stem in stems:
        src, stored = load_clip_inputs(env, ctx, stem)
        ck(ctx["src_sha"][stem] == RC.sha256_file(osp.join(SRC, f"masks_{stem}.json.gz")), "source sha of the staged copy")
        vid = env.open_video(stem)
        recs, par, st = detect_clip(env, ctx, stem, src, stored, vid, enc)
        ck(st["parity_n"] == 5 and st["parity_fail"] == 0, f"E0 parity on the stub: {st}")
        publish_json(env, ctx, det_file_obj(ctx, stem, recs, st), osp.join(OUT, f"detections_refine_{stem}.json.gz"), gz=True)
        touch_marker(env, ctx, osp.join(OUT, f".det_done_{stem}"))
        new = load_new_records(env, ctx, stem)
        ck(set(new) == set(RC.plan_new_calls(src["frames"], src["clip"], C, stored.keys(), None)), "every planned call recorded")
        rep = apply_clip_driver(env, ctx, stem, src, stored, new, vid)
        ck(rep["codes"].get("d", 0) > 0 and rep["refused_by_reason"].get("big", 0) == 1, f"apply codes {rep}")
    enc.close()
    rm = RC.load_json_gz(osp.join(OUT, f"masks_{stems[0]}.json.gz"))
    side = RC.load_json_gz(osp.join(OUT, f"refine_{stems[0]}.json.gz"))["records"]
    codes = {r["frame_idx"]: (r["code"], r["reason"]) for r in side if r["slot"] == 0}
    ck(codes[60][0] == "c" and codes[65][0] == "c" and codes[95] == ("r", "big") and codes[100][0] == "d",
       f"codes: nodet carried, big refused, stored reused {codes[60]} {codes[95]} {codes[100]}")
    ck(rm["refine"]["spec"] == RC.REFINE_SPEC and rm["frames"][10]["prov"]["0"] in ("d", "c"), "refine block + prov")
    with tarfile.open(osp.join(OUT, "sheets", f"sheet_{stems[0]}.tar")) as tf:
        names = tf.getnames()
    ck(any(n.endswith("index.json") for n in names) and any("case1" in n for n in names), f"sheet tar {names[:4]}")
    rep = finalize_day(env, ctx, OUT, stems)
    ck(rep["i4"]["segments.json"] == RC.sha256_file(osp.join(SRC, "segments.json")) and
       rep["i4"]["crop_bands3_dense.npz"] == "absent_in_source", f"I4 copies {rep['i4']}")
    cfg = {"spec": RC.REFINE_SPEC, "src_root": "Tracking_v5a10", "sample": "1", "constants_sha256": "x",
           "detector_sha256": "stub", "refine_core_sha256": RC_SHA, "det_run_tag": RC.DET_RUN_TAG}
    ck(check_refine_config(env, ctx, OUT, cfg) == "written" and check_refine_config(env, ctx, OUT, cfg) == "matches", "REFINE_CONFIG")
    try:
        check_refine_config(env, ctx, OUT, dict(cfg, sample="5")); ck(False, "REFINE_CONFIG mismatch accepted")
    except RuntimeError:
        pass
    # E0 must stop the run when the detector path deviates
    perturb["on"] = True
    src, stored = load_clip_inputs(env, ctx, stems[0])
    called = {}
    ctx_pf = dict(ctx, on_parity_fail=lambda stem, par, got, st: called.update(n=len(par), nf=sum(not p["ok"] for p in par),
                                                                             recs=len(got)))
    try:
        detect_clip(env, ctx_pf, stems[0], src, stored, env.open_video("x"), Encoder(1)); ck(False, "parity failure not raised")
    except RuntimeError as e:
        ck("E0 PARITY FAIL" in str(e), f"parity failure message {e}")
    ck(called.get("n") == 5 and called.get("nf", 0) >= 1 and called.get("recs") == 5,
       f"every parity call compared and the evidence handed over before the stop: {called}")
    # the same swapped detector output is EQUIVALENT under the tolerance rule (another GPU class): the kept set is the
    # same, only the score-ranked raw order differs
    recs_t, par_t, st_t = detect_clip(env, dict(ctx, parity_exact=False), stems[0], src, stored, env.open_video("x"), Encoder(1))
    ck(st_t["parity_fail"] == 0 and st_t["parity_order_differs"] == 5, f"tolerance parity on a raw reorder: {st_t}")
    perturb["on"] = False
    # grow on the refined root: calf 3's fragment on clip 2 is r/nodet? (the stub always detects calf 3 -> d); force a
    # nodet day row by removing calf 3 from the detections of clip 2 rows 10-19 and re-applying
    ctx2 = dict(ctx, OUT_DIR=osp.join(vol, "Tracking_ref2", "20240422"), GROW_DIR=osp.join(vol, "Tracking_grow2", "20240422"),
                sheets=False)
    os.makedirs(ctx2["OUT_DIR"], exist_ok=True); os.makedirs(ctx2["GROW_DIR"], exist_ok=True)
    stem = stems[1]; src, stored = load_clip_inputs(env, ctx2, stem)
    new = load_new_records(env, ctx, stem)
    for f in range(50, 100, 5):
        if f in new:
            new[f] = RC._det_rec(calves[:3], f, f / 25.0, "dayl", C); new[f]["clip"] = src["clip"]
    rep = apply_clip_driver(env, ctx2, stem, src, stored, new, None)
    sd = RC.load_json_gz(osp.join(ctx2["OUT_DIR"], f"refine_{stem}.json.gz"))["records"]
    n_nodet = sum(1 for r in sd if r["slot"] == 3 and r["reason"] == "nodet")
    ck(n_nodet >= 4, f"nodet rows for the grow test: {n_nodet}")
    pos = np.zeros((RC.H, RC.W), bool)
    for m in calves: pos |= RC.erode(m, 3)
    lut = RC.build_lut(RC.lut_hist(img, pos), RC.lut_hist(img, ~RC.dilate(np.any(calves, axis=0), 15)))
    Cg = dict(C, CHROMA_MIN=30.0, LLR_MIN=0.5, BL_P99=700.0, SOL_P1=0.7)
    ctx2["C"] = Cg
    ctx2["grow"] = {"lut": lut, "static_full": np.zeros((RC.H, RC.W), bool), "train_bg_sub": RC.subsample(img)}
    gs = grow_clip_driver(env, ctx2, stem, src, stored, new, FakeVideo("x"))
    ck(gs["grown"] >= 4 and gs["shift"]["action"] == "none", f"grow stats {gs}")
    gm = RC.load_json_gz(osp.join(ctx2["GROW_DIR"], f"masks_{stem}.json.gz"))
    g_rows = [(i, fr) for i, fr in enumerate(gm["frames"]) if fr["prov"]["3"] == "g"]
    if g_rows:
        Gd = RC.runs_to_dense(RC.runs_of(g_rows[0][1]["masks"]["3"]))
        iou = float((Gd & calves[3]).sum()) / float((Gd | calves[3]).sum())
        ck(iou > 0.9, f"grown mask IoU with the calf {iou:.3f}")
    shutil.rmtree(td, ignore_errors=True)
    print(f"Tol_55 pipeline selftest: {ck.n} checks, {len(ck.fail)} failed")
    if ck.fail: raise SystemExit("PIPELINE SELFTEST FAILED:\n  " + "\n  ".join(ck.fail))

def _audit_against_tol47():
    """the RF-DETR call must be Tol_47's verbatim (_rf_raw body, post_chain_idx arguments)."""
    p47 = osp.join(CORE_DIR, "Tol_47_track_v5.py")
    if not osp.exists(p47): print("  (Tol_47 not beside this file: verbatim audit SKIPPED)"); return
    me = open(osp.abspath(__file__), encoding="utf-8").read(); t47 = open(p47, encoding="utf-8").read()
    def body(txt, name):
        L = txt.splitlines()
        i0 = next((i for i, l in enumerate(L) if re.match(rf"^\s*def {name}\(", l)), None)
        if i0 is None: raise AssertionError(f"{name} not found")
        ind = len(L[i0]) - len(L[i0].lstrip())
        out = [L[i0].strip()]
        for l in L[i0 + 1:]:
            if l.strip() and (len(l) - len(l.lstrip())) <= ind: break
            if l.strip() and not l.strip().startswith("#"): out.append(l[ind:].rstrip())
        return "\n".join(out)
    a, b = body(me, "_rf_raw"), body(t47, "_rf_raw")
    a = a.replace("V5.run_with_precision", "core.run_with_precision")
    if a != b: raise AssertionError("_rf_raw differs from Tol_47's")
    if "V5.post_chain_idx(rows, topk=RC.DET_TOPK, iou=V5.m_iou_fast, ios=V5.m_ios_fast, area=_area_px)" not in me:
        raise AssertionError("post-chain call differs from Tol_47's detect()")
    # Tol_47 binds m_iou / m_ios / rle_encode to the fast twins under exact_fast (the v5a10 root ran exact_fast=true)
    if "rle_encode, m_iou, m_ios = core.rle_fast, core.m_iou_fast, core.m_ios_fast" not in t47:
        raise AssertionError("Tol_47 no longer binds the exact_fast twins: re-check the post-chain functions")
    if "kidx, _nf = core.post_chain_idx(rows, topk=(topk or 0), iou=m_iou, ios=m_ios, area=_area_px)" not in t47:
        raise AssertionError("Tol_47 detect() post-chain call changed")
    print("  Tol_47 verbatim audit: _rf_raw identical, post-chain call identical")

if SELFTEST:
    _ts = os.environ.get("TOL_SELFTEST_TS") or osp.join(
        osp.expanduser("~"), "<LOCAL_RUN_DIR>",
        "scratchpad", "tune_scout", "train_summary_segv1d.json")
    RC.selftest(_ts if osp.exists(_ts) else None)
    _selftest_pipeline()
    _audit_against_tol47()
    print("Tol_55 SELFTEST PASSED")
    raise SystemExit(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets, paths, environment

# COMMAND ----------

def P(name, default):
    if IS_NOTEBOOK:
        try: dbutils.widgets.text(name, default)
        except Exception: pass
        return (dbutils.widgets.get(name) or default)
    return os.environ.get("TOL55_" + name.upper(), default)

MODES = ("census", "derive", "detect", "apply", "redetect", "grow", "both")
W = {k: (P(k, d) or "").strip() for k, d in (
    ("mode", ""), ("day", ""), ("src_root", "Tracking_v5a10"), ("out_root", "Tracking_v5a10_refined"),
    ("grow_root", "Tracking_v5a10_refined_grow"), ("sample", "1"), ("max_clips", "0"),
    ("det_run_tag", RC.DET_RUN_TAG), ("constants_tag", "refine_v1"), ("parity", "strict"), ("parity_max_per_clip", "0"),
    ("sheets", "true"), ("sheet_cases", ""), ("prefetch", "true"), ("workers", "8"), ("no_shrink", "true"),
    ("faults_json", "Detector_COCO/segv1d/train_label_faults.json"), ("train_root", "Tracking_train"),
    ("train_relabel_dir", "Metadata/relabel_train_week"))}
MODE = W["mode"].lower()
if MODE not in MODES: raise RuntimeError(f"mode must be one of {MODES}, got {MODE!r}")
DAY = W["day"]
if MODE != "derive":
    if DAY not in RC.TEST_DAYS: raise RuntimeError(f"day must be one of {RC.TEST_DAYS}, got {DAY!r}")
MAX_CLIPS = int(W["max_clips"] or 0); SMOKE = MAX_CLIPS > 0
SAMPLE = W["sample"].lower()
RC.sample_selection([], SAMPLE)                                   # validates the value
if W["parity"] not in ("strict", "report"): raise RuntimeError("parity must be strict|report (E0 is never skipped)")
CANON = ("Tracking_v5a10_refined", "Tracking_v5a10_refined_grow")
if (SMOKE or SAMPLE != "1") and (W["out_root"] in CANON or W["grow_root"] in CANON) and MODE not in ("census", "derive"):
    raise RuntimeError(f"smoke (max_clips>0) or sample!='1' must not write the canonical roots {CANON}: set out_root / "
                       "grow_root to a separate root (e.g. Tracking_v5a10_refined_smoke)")
if W["src_root"] in (W["out_root"], W["grow_root"]): raise RuntimeError("I8: the source root is read-only")
SHEET_CASES = json.loads(W["sheet_cases"]) if W["sheet_cases"] else []
if W["no_shrink"].lower() not in ("true", "false"): raise RuntimeError("no_shrink must be true|false")
NO_SHRINK = W["no_shrink"].lower() == "true"        # addition to the v1 design (tol_refine_core.NEVER_CARRY note); false = v1 rule
NEED_GPU = MODE in ("derive", "detect", "redetect", "both")
DO_DETECT = MODE in ("detect", "redetect", "both"); DO_APPLY = MODE in ("apply", "redetect", "both")
DO_GROW = MODE in ("grow", "both")
print(f"Tol_55 mode={MODE} day={DAY or '-'} src={W['src_root']} out={W['out_root']} grow={W['grow_root']} sample={SAMPLE} "
      f"max_clips={MAX_CLIPS} parity={W['parity']}")

VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
VIDEO_DIR = "/dbfs/mnt/playbehavior/Tol/Videos"
DBFS_VIDEO = "dbfs:/mnt/playbehavior/Tol/Videos"
DET_CKPT_DIR = osp.join(TOL, "Detector_Checkpoints", W["det_run_tag"])
CONST_DIR = osp.join(TOL, "Metadata", W["constants_tag"])
if MODE == "derive" and SMOKE:                       # a smoke derivation never touches the frozen directory
    CONST_DIR = osp.join(TOL, "Metadata", W["constants_tag"] + "_smoke", datetime.now().strftime("%Y%m%d_%H%M%S"))
SRC_DIR = osp.join(TOL, W["src_root"], DAY) if DAY else None
OUT_DIR = osp.join(TOL, W["out_root"], DAY) if DAY else None
GROW_DIR = osp.join(TOL, W["grow_root"], DAY) if DAY else None
LOCAL = f"/local_disk0/tmp/tol55_{DAY or 'derive'}_{MODE}"
os.makedirs(LOCAL, exist_ok=True)
for _d in ([OUT_DIR] if (DO_DETECT or DO_APPLY) else []) + ([GROW_DIR] if DO_GROW else []) + ([CONST_DIR] if MODE == "derive" else []):
    os.makedirs(_d, exist_ok=True)

def _retry(fn, what, tries=5, base=5.0):
    last = None
    for k in range(tries):
        try: return fn()
        except Exception as e:
            last = e; print(f"  ⚠ {what}: attempt {k + 1}/{tries} failed: {e}", flush=True)
            time.sleep(base * (2 ** k))
    raise RuntimeError(f"{what}: failed {tries}x: {last}")

def stage_small(src, dst):
    if osp.exists(dst): os.remove(dst)
    if MODE == "derive": RC.assert_not_test_path(src)
    _retry(lambda: dbutils.fs.cp("dbfs:" + src, "file:" + dst), f"stage {osp.basename(src)}")
    return dst

def publish(src, dst):
    """local file -> Volume (dbutils.fs.cp with retry); an existing destination is replaced (re-runs after a crash
    between a publish and its .done_ marker, telemetry / config updates)."""
    if osp.exists(dst):
        try: os.remove(dst)
        except OSError: pass
    _retry(lambda: dbutils.fs.cp("file:" + src, "dbfs:" + dst), f"publish {osp.basename(dst)}")

def stage_video(vname, dst):
    if MODE == "derive": RC.assert_not_test_path(vname)
    if osp.exists(dst): os.remove(dst)
    t = time.time()
    _retry(lambda: dbutils.fs.cp(f"{DBFS_VIDEO}/{vname}", "file:" + dst), f"stage video {vname}")
    el = time.time() - t; mb = osp.getsize(dst) / 1e6
    print(f"    staged {vname} {mb:.0f} MB in {el / 60:.1f} min ({mb / max(el, 1e-9):.1f} MB/s)", flush=True)
    return el

TS = RC.load_json(osp.join(DET_CKPT_DIR, "train_summary.json"))
READ_C = RC.read_constants(TS, W["det_run_tag"])
print(f"READ: A_ref {READ_C['A_REF']:.0f}, FRAG_LO {READ_C['FRAG_LO']}, OVER_HI {READ_C['OVER_HI']}, TRIG_LO {READ_C['TRIG_LO']} "
      f"-> [{RC.trig_lo_px('dayl', READ_C):,.0f} / {RC.trig_lo_px('dark', READ_C):,.0f}] px, LINK_IOU_MIN {READ_C['LINK_IOU_MIN']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPU + RF-DETR-Seg (verbatim Tol_47 contract) and the video reader

# COMMAND ----------

GPU_NAME = GPU_CLASS = None; RF = RF_NC = None; DET_SHA = None
if NEED_GPU or DO_GROW or DO_APPLY or MODE == "census":
    import cv2
    if not cv2.__version__.startswith("4.11.0"): raise RuntimeError(f"cv2 {cv2.__version__}: headless 4.11.0 expected")
if NEED_GPU:
    import torch
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError("PYTORCH_CUDA_ALLOC_CONF lacks expandable_segments:True")
    if not torch.cuda.is_available(): raise RuntimeError("mode needs a GPU and torch sees none")
    torch.set_float32_matmul_precision("highest")             # Tol_47:264 (production ran fp32 at 'highest')
    GPU_NAME = torch.cuda.get_device_name(0); GPU_CLASS = V5.gpu_class(GPU_NAME)
    print(f"GPU {GPU_NAME} ({GPU_CLASS})")
    OP_THRESHOLD = READ_C["op_threshold"]

    def _stage_small(src, dst):
        return stage_small(src, dst)

    def _rf_raw(arr):
        """RF.predict(threshold=op) keeps score > op (strict); drop class_id >= num_classes (Tol_44:1096-1108). TF32
        only inside predict; 'highest' restored on every path and asserted (core.run_with_precision, S9)."""
        def _pred():
            with torch.inference_mode():
                return RF.predict(arr, threshold=OP_THRESHOLD, include_source_image=False)
        d = V5.run_with_precision(torch.set_float32_matmul_precision, torch.get_float32_matmul_precision, _pred)
        rows = []
        if d.mask is not None and len(d):
            k = d.class_id < RF_NC
            rows = [(np.asarray(m, dtype=bool), float(s)) for m, s in zip(d.mask[k], d.confidence[k])]
        for m, s in rows:
            if m.shape != (1296, 2304) or not np.isfinite(s): raise RuntimeError("RF-DETR output contract")
        return rows

    def load_rfdetr():
        """Tol_47:583-607 verbatim except the staging path (LOCAL) -- checkpoint byte size + sha, resolution pinned,
        one class 'calf', positional-encoding assert, black-frame assert, precision 'highest' after the load."""
        global RF, RF_NC, DET_SHA
        os.environ["RF_HOME"] = "/local_disk0/rf_home"
        from rfdetr import from_checkpoint          # side effect: set_float32_matmul_precision('high') (detr.py:52-55)
        torch.set_float32_matmul_precision("highest")                                # immediately restored
        loc = _stage_small(osp.join(DET_CKPT_DIR, "checkpoint_best_total.pth"), osp.join(LOCAL, "rf_best.pth"))
        if osp.getsize(loc) != V5.RF_CKPT_BYTES: raise RuntimeError(f"RF checkpoint is {osp.getsize(loc)} B, not {V5.RF_CKPT_BYTES}")
        sha = V5.sha256_file(loc)
        if sha != TS["checkpoint_best_total_sha256"] or not sha.startswith(V5.RF_CKPT_SHA_PREFIX):
            raise RuntimeError(f"RF checkpoint sha {sha[:16]} is not the train_summary checkpoint")
        RF = from_checkpoint(loc, resolution=V5.RF_RESOLUTION)   # pinned: strip_checkpoint drops model_config
        RF_NC = int(RF.model_config.num_classes)
        if RF_NC != 1: raise RuntimeError(f"num_classes {RF_NC} != 1")
        if list(RF.class_names) != ["calf"]: raise RuntimeError(f"class_names {RF.class_names} != ['calf']")
        if int(RF.model_config.resolution) != V5.RF_RESOLUTION: raise RuntimeError("resolution != 624")
        if int(RF.model_config.positional_encoding_size) != V5.RF_RESOLUTION // int(RF.model_config.patch_size):
            raise RuntimeError("positional_encoding_size != resolution // patch_size")
        kb, _nf = V5.post_chain_idx(_rf_raw(np.zeros((1296, 2304, 3), np.uint8)), topk=0)
        if kb: raise RuntimeError(f"RF-DETR keeps {len(kb)} masks on a black frame")
        if torch.get_float32_matmul_precision() != "highest": raise RuntimeError("precision not 'highest' after the RF load")
        DET_SHA = sha
        print(f"RF-DETR-Seg ready: sha256 {sha[:16]}, resolution {V5.RF_RESOLUTION}, op {OP_THRESHOLD}, "
              f"max_memory_allocated {round(torch.cuda.max_memory_allocated() / 2**30, 3)} GiB")

    def _area_px(m):
        return V5.area_fast(m)                        # Tol_47 _area_px with exact_fast on (the v5a10 root config)

    def detect_rgb(arr):
        """Tol_47 detect() for detector=rfdetr: _rf_raw on the RGB uint8 frame, the production post-chain (top-6) with
        the functions production bound under exact_fast=true (Tracking_v5a10 ROOT_CONFIG widgets; Tol_47:500
        `rle_encode, m_iou, m_ios = core.rle_fast, core.m_iou_fast, core.m_ios_fast`; bit-identical to the dense m_iou /
        m_ios -- core S1/S2 -- but ~30x faster: the dense pair cost ~33 ms per call, about the detector's own 37 ms),
        exact areas, synchronised timing, precision asserted; the mask cache is cleared per frame as the tracker does."""
        V5.CACHE.clear()
        arr = np.asarray(Image.fromarray(arr))        # the production path: PIL image of the BGR->RGB decode
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rows = _rf_raw(arr)
        kidx, _nf = V5.post_chain_idx(rows, topk=RC.DET_TOPK, iou=V5.m_iou_fast, ios=V5.m_ios_fast, area=_area_px)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
        V5.CACHE.clear()
        if torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("float32 matmul precision is not 'highest' after detect()")
        return rows, list(kidx), ms

    from PIL import Image
    load_rfdetr()
else:
    def detect_rgb(arr): raise RuntimeError(f"mode {MODE} runs no detector")
    from PIL import Image

class VideoFrames:
    """production decode: cv2.VideoCapture + tol_v5_core.iter_frames(mode='seek_thread') + BGR->RGB."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise RuntimeError(f"cv2.VideoCapture cannot open {path}")
        self.native = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    def frames(self, fidxs):
        it = V5.iter_frames(self.cap, list(fidxs), mode="seek_thread", pos_prop=cv2.CAP_PROP_POS_FRAMES)
        try:
            for f, ok, fr in it:
                yield f, ok, (cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None)
        finally:
            it.close()
    def close(self):
        self.cap.release()

def jpeg_bytes(rgb):
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok: raise RuntimeError("jpeg encode failed")
    return b.tobytes()

ENV = Env(stage_small=stage_small, publish=publish, stage_video=stage_video,
          open_video=lambda p: VideoFrames(p), detect=detect_rgb, jpeg=jpeg_bytes) if MODE != "census" else None

def _source_norm_sha():
    try:
        nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        for cand in ("/Workspace" + nbp, "/Workspace" + nbp + ".py"):
            if osp.isfile(cand):
                b = open(cand, "rb").read()
                if b.lstrip(b"\xef\xbb\xbf").startswith(b"# Databricks notebook source"): return V5.source_norm_sha256(b)
        import base64
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.workspace import ExportFormat
        return V5.source_norm_sha256(base64.b64decode(WorkspaceClient().workspace.export(nbp, format=ExportFormat.SOURCE).content))
    except Exception as e:
        print(f"⚠ notebook source sha unavailable: {e!r}"); return None
SRC_NORM_SHA = _source_norm_sha()
print(f"Tol_55 source_norm sha256 {SRC_NORM_SHA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## mode = derive (S1): training-day constants (opens no test-day path)

# COMMAND ----------

def derive():
    t_all = time.time()
    C = dict(READ_C)
    rng_seed = 20261002
    faults_p = osp.join(TOL, W["faults_json"]); RC.assert_not_test_path(faults_p)
    if RC.sha256_file(faults_p) != READ_C["faults_json_sha256"]:
        raise RuntimeError(f"faults_json sha != train_summary provenance faults_json_sha256 ({READ_C['faults_json_sha256'][:12]})")
    FAULTS = RC.load_json(faults_p)
    fault_iv = defaultdict(list)
    for f in FAULTS["faults"]:
        fault_iv[f["day"]].append((RC.ts_of(datetime.fromisoformat(f["exclude_start"])), RC.ts_of(datetime.fromisoformat(f["exclude_end"]))))
    train_set, valid_set = set(READ_C["train_clips"]), set(READ_C["valid_clips"])
    TR = osp.join(TOL, W["train_root"])
    windows = sorted(d for d in os.listdir(TR) if re.match(r"^2024041[89]_w\d{4}-\d{4}$", d))
    clips = []
    for w in windows:
        RC.assert_not_test_path(w)
        for n in sorted(os.listdir(osp.join(TR, w))):
            if n.startswith("masks_") and n.endswith(".json.gz"):
                stem = RC.clip_stem(n)
                role = "train" if stem in train_set else ("valid" if stem in valid_set else "unseen")
                clips.append((w, stem, role))
    clips.sort(key=lambda z: RC.clip_start(z[1]))
    roles = Counter(r for _w, _s, r in clips)
    print(f"derive: {len(clips)} training clips {dict(roles)} in {len(windows)} windows")
    if not SMOKE and (len(clips) != 46 or roles["train"] != 18 or roles["valid"] != 4 or roles["unseen"] != 24):
        raise RuntimeError(f"expected 46 clips = 18 train + 4 valid + 24 unseen, got {dict(roles)}")
    if SMOKE:                                    # stratified: a time-ordered head is all TRAIN clips (no detector frames)
        clips = RC.smoke_clip_subset(clips, MAX_CLIPS, lambda st: RC.pool_of(RC.clip_start(st) + timedelta(minutes=10)))
        print(f"SMOKE derive on {[(s, r) for _w, s, r in clips]}")
    rel_cache, seg_cache = {}, {}
    def rules_of(w):
        if w not in rel_cache:
            p = osp.join(TOL, W["train_relabel_dir"], f"relabel_week_{w}.json"); RC.assert_not_test_path(p)
            R, hid, exc = RC.parse_train_relabel(RC.load_json(p))       # rules + hidden + exclusions (merge episodes)
            rel_cache[w] = (R, hid, exc, RC.sha256_file(p))
        return rel_cache[w]
    def segs_of(w):
        if w not in seg_cache:
            p = osp.join(TR, w, "segments.json"); RC.assert_not_test_path(p); seg_cache[w] = RC.SegIndex(RC.load_json(p))
        return seg_cache[w]
    sub = RC.DESIGN["SUB"]
    # ---- accumulators ----
    acc = {"carry": defaultdict(list), "bl": [], "sol": [], "chroma": [], "conflict": {"dayl": [], "dark": []},
           "offsets": [], "det_frames": [], "colour": [], "bgate": [], "bgclip": defaultdict(list), "ts_rows": Counter(),
           "per_clip": [], "det_ms": []}
    for ci, (w, stem, role) in enumerate(clips):
        t_clip = time.time()
        day = RC.clip_start(stem).strftime("%Y%m%d")
        lm = osp.join(LOCAL, "m.json.gz"); stage_small(osp.join(TR, w, f"masks_{stem}.json.gz"), lm)
        src = RC.load_json_gz(lm); os.remove(lm)
        frames = src["frames"]; w0 = RC.clip_start(src["clip"]); SI = segs_of(w); R_, HID, EXC, _rsha = rules_of(w)
        n = len(frames)
        pos = {int(fr["frame_idx"]): q for q, fr in enumerate(frames)}     # production skips unreadable frames
        tlist = [float(fr["t_sec"]) for fr in frames]
        runs_c = {}
        def M_at(i):
            if i not in runs_c:
                ms = frames[i].get("masks") or {}
                runs_c[i] = [RC.runs_of(ms[str(s)]) if ms.get(str(s)) else None for s in range(RC.N_SLOTS)]
                if len(runs_c) > 64: runs_c.pop(next(iter(runs_c)))
            return runs_c[i]
        tcache = {}
        def trusted(i):
            if i in tcache: return tcache[i]
            fr = frames[i]; wl = RC.wall_of(w0, fr["t_sec"]); ts = RC.ts_of(wl); pool = RC.pool_of(wl)
            ok = RC.trusted_frame(M_at(i), pool, C)
            if ok and any(a <= ts < b for a, b in fault_iv.get(day, [])): ok = False
            if ok and not RC.relabel_trusted(R_, HID, EXC, ts): ok = False     # rules, hidden, exclusions
            tcache[i] = ok
            return ok
        # ---- mask-only statistics: activity + carry window samples, shape statistics ----
        one_hz = list(range(0, n, 5))
        cents = [(frames[i]["t_sec"], [RC.centroid_runs(m) for m in M_at(i)]) for i in one_hz]
        acts = RC.activity_scores(cents)
        thr = READ_C["ACTIVE_THR"][day]
        grid = RC.DESIGN["CARRY_GRID"]
        for k, i in enumerate(one_hz):
            if acts[k] is None or acts[k] < thr or not trusted(i): continue
            Mi = M_at(i); tsi = RC.ts_of(RC.wall_of(w0, frames[i]["t_sec"]))
            for dt in grid:
                j = i + int(round(dt * 5))
                if j >= n or abs(frames[j]["t_sec"] - frames[i]["t_sec"] - dt) > 0.05 or not trusted(j): continue
                Mj = M_at(j); tsj = RC.ts_of(RC.wall_of(w0, frames[j]["t_sec"]))
                for s in range(RC.N_SLOTS):
                    if SI.at(s, tsi) >= 0 and SI.at(s, tsi) == SI.at(s, tsj):
                        acc["carry"][dt].append(RC.iou_r(Mi[s], Mj[s]))
        for i in range(0, n, 150):                                        # every 30 s
            wl = RC.wall_of(w0, frames[i]["t_sec"])
            if RC.phase_of(wl) != "day" or not trusted(i): continue
            for m in M_at(i):
                bb = RC.runs_bbox(m); cr = RC.runs_to_crop(m, *bb)
                acc["bl"].append(RC.body_len(cr)); acc["sol"].append(RC.solidity(cr))
        # ---- frame plan: colour 1/min (all clips), detector 0.5 Hz trusted (UNSEEN ∪ VALID), DYN bg + B gate ----
        col_i = set()
        for t in np.arange(0.0, tlist[-1] + 1e-9, RC.DESIGN["COLOUR_EVERY_S"]):
            q = bisect.bisect_left(tlist, t)
            cand = [x for x in (q - 1, q) if 0 <= x < n]
            col_i.add(min(cand, key=lambda x: (abs(tlist[x] - t), x)))
        col_i = sorted(col_i)
        det_i = [i for i in range(0, n, int(RC.DESIGN["DERIVE_EVERY_S"] * 5)) if role != "train" and trusted(i)]
        day_rows = [i for i in range(n) if RC.phase_of(RC.wall_of(w0, frames[i]["t_sec"])) == "day"]
        nbg = RC.DESIGN["N_BG_FRAMES"]
        bg_i = ([day_rows[int(j)] for j in np.linspace(0, len(day_rows) - 1, nbg).round().astype(int)]
                if role != "train" and len(day_rows) >= nbg else [])
        bg_every = int(RC.DESIGN["B_GATE_EVERY_S"] * 5)
        bgate_i = {i for i in det_i if i % bg_every == 0 and RC.phase_of(RC.wall_of(w0, frames[i]["t_sec"])) == "day"}
        col_s, det_s, bg_s = set(col_i), set(det_i), set(bg_i)
        want = sorted(col_s | det_s | bg_s)
        vloc = osp.join(LOCAL, "v.mp4"); t_stage = stage_video(src["clip"], vloc)
        vid = VideoFrames(vloc)
        t_dec = 0.0; t_last = time.perf_counter(); n_det = 0
        for f, ok, rgb in vid.frames([int(frames[i]["frame_idx"]) for i in want]):
            t_dec += time.perf_counter() - t_last
            if not ok: t_last = time.perf_counter(); continue
            i = pos[int(f)]
            wl = RC.wall_of(w0, frames[i]["t_sec"]); pool = RC.pool_of(wl); ph = RC.phase_of(wl)
            Mi = M_at(i)
            if i in col_s or i in bg_s:
                allm = np.zeros((RC.H, RC.W), bool)
                for m in Mi:
                    if m is not None: allm |= RC.runs_to_dense(m)
                ch = RC.chroma_of(rgb, allm)
                crec = {"clip": stem, "role": role, "day": day, "hour": wl.hour, "phase": ph, "chroma": ch,
                        "trusted": trusted(i), "rgb": RC.subsample(rgb).copy(),
                        "excl": RC.dilate(RC.subsample(allm), math.ceil(RC.DESIGN["BG_EXCL_DIL"] / sub)),
                        "pos": RC.erode(RC.subsample(allm), math.ceil(3 / sub)) if trusted(i) else None,
                        "T_sub": RC.subsample(allm) if trusted(i) else None}
                if i in col_s: acc["colour"].append(crec); acc["chroma"].append(ch)
                if i in bg_s: acc["bgclip"][stem].append(crec)
            if i in det_s:
                rows, kidx, ms = detect_rgb(rgb); acc["det_ms"].append(ms); n_det += 1
                _mc = V5.MaskCache()
                D = [RC.runs_of(V5.rle_fast(rows[k][0], _mc)) for k in kidx]
                T = list(Mi); mt = RC.hungarian(T, D)
                for c, dj in mt.items():
                    for c2 in range(RC.N_SLOTS):
                        if c2 != c: acc["conflict"][pool].append(RC.ios_r(D[dj], T[c2]))
                if (i // int(RC.DESIGN["DERIVE_EVERY_S"] * 5)) % 5 == 0:     # boundary offsets at 0.1 Hz
                    for c, dj in mt.items():
                        y0, y1, x0, x1 = RC.crop_box([T[c], D[dj]], 4)
                        o = RC.signed_offset(RC.runs_to_crop(T[c], y0, y1, x0, x1), RC.runs_to_crop(D[dj], y0, y1, x0, x1))
                        if o is not None: acc["offsets"].append(o)
                pack = {"clip": stem, "ci": ci, "fidx": int(f), "pool": pool, "phase": ph,
                        "T": [np.asarray(RC.rle_from_runs(t)["counts"], np.int32) for t in T],
                        "D": [np.asarray(RC.rle_from_runs(d)["counts"], np.int32) for d in D], "match": mt}
                acc["det_frames"].append(pack)
                if i in bgate_i: acc["bgate"].append(dict(pack, rgb=rgb.copy(), ci=ci))
            t_last = time.perf_counter()
        vid.close(); os.remove(vloc)
        acc["per_clip"].append({"clip": stem, "window": w, "role": role, "rows": n, "colour_frames": len(col_i),
                                "det_frames": n_det, "bg_frames": len(bg_i), "stage_min": round(t_stage / 60, 2),
                                "decode_s": round(t_dec, 1), "clip_min": round((time.time() - t_clip) / 60, 2)})
        print(f"[{ci + 1}/{len(clips)}] {stem} ({role}): {n_det} det frames, {len(col_i)} colour, {len(bg_i)} bg, "
              f"{(time.time() - t_clip) / 60:.1f} min", flush=True)
    # ---------------- constants ----------------
    rep = {"per_clip": acc["per_clip"], "notes": []}
    viol = []                                   # sanity-band violations: collected, reported, then a loud stop
    out = {}
    def rec(name, value, rule, days, n=None, dist=None, clips_used=None):
        out[name] = {"value": value, "rule": rule, "derived_from_days": sorted(days), "derived_from_clips": clips_used,
                     "not_from": list(RC.TEST_DAYS), "n": n, "percentiles": dist}
    uv = sorted({p["clip"] for p in acc["det_frames"]})
    # CHROMA_MIN
    chv = [c for c in acc["chroma"] if c is not None]
    k, eta = RC.otsu(chv)
    if eta < RC.DESIGN["CHROMA_ETA_MIN"]:
        viol.append(f"CHROMA_MIN: per-frame saturation is not bimodal (eta {eta:.3f} < {RC.DESIGN['CHROMA_ETA_MIN']})")
    C["CHROMA_MIN"] = float(k + 1)
    rec("CHROMA_MIN", C["CHROMA_MIN"], "Otsu valley (class 0 = v <= k; CHROMA_MIN = k + 1) of the per-frame median HSV "
        "saturation outside the teacher masks (dilated BG_EXCL_DIL), 1 frame/min, all 46 clips", RC.TRAIN_DAYS, len(chv),
        dict(RC.pcts(chv), eta=round(eta, 4)))
    colour_day = [r for r in acc["colour"] if r["phase"] == "day" and r["chroma"] is not None and r["chroma"] >= C["CHROMA_MIN"]]
    # LUT from TRAIN-clip day colour trusted frames
    hp = np.zeros(1 << (3 * RC.DESIGN["LUT_BITS"]), np.int64); hn = hp.copy(); n_lut = 0
    for r in colour_day:
        if r["role"] != "train" or not r["trusted"]: continue
        hp += RC.lut_hist(r["rgb"], r["pos"]); hn += RC.lut_hist(r["rgb"], ~r["excl"]); n_lut += 1
    if n_lut == 0: raise RuntimeError("LUT: no TRAIN-clip daylight colour trusted frame")
    lut = RC.build_lut(hp, hn)
    # STATIC (two-step, see tol_refine_core.ring_threshold): LLR_MIN0 without STATIC -> STATIC -> LLR_MIN with STATIC
    held = [r for r in colour_day if r["role"] != "train" and r["trusted"]]
    if not held: raise RuntimeError("LLR_MIN: no detector-unseen daylight colour trusted frame")
    ring_out = math.ceil(round(READ_C["BODY_LEN_PX"] / 10) / sub); ring_in = math.ceil(RC.DESIGN["STRUCT_DIL"] / sub)
    rings = [(r, RC.dilate(r["T_sub"], ring_out) & ~RC.dilate(r["T_sub"], ring_in)) for r in held]
    n_ring = int(sum(int(g.sum()) for _r, g in rings))
    llr_all = np.concatenate([lut[RC.lut_index(r["rgb"])][g] for r, g in rings])
    llr0, share0 = RC.ring_threshold(llr_all, n_ring, lut)
    hours = defaultdict(list)
    for r in colour_day: hours[(r["day"], r["hour"])].append(r)
    hour_meds = []
    for key in sorted(hours):
        L = hours[key]
        med, cnt = RC.masked_median(np.stack([r["rgb"] for r in L]), np.stack([r["excl"] for r in L]))
        hour_meds.append((med, cnt))
    static_sub = RC.static_from_hours(hour_meds, lut, llr0)
    st_ring = RC.dilate(static_sub, 0)
    llr_ns = np.concatenate([lut[RC.lut_index(r["rgb"])][g & ~st_ring] for r, g in rings])
    llr_min, share = RC.ring_threshold(llr_ns, n_ring, lut)
    C["LLR_MIN"] = llr_min
    # STATIC is REBUILT at the final LLR_MIN (review 2026-10-02): painting uses calf_like = LUT >= LLR_MIN, so a STATIC
    # built at the stricter LLR_MIN0 would leave structure pixels with LLR_MIN <= LUT < LLR_MIN0 paintable. LLR_MIN <=
    # LLR_MIN0 and STATIC is monotone, so the final STATIC is a superset of STATIC0 and the ring share (numerator excludes
    # STATIC) can only fall: the LLR_MIN criterion still holds with the final STATIC.
    static0_px = int(static_sub.sum())
    static_sub = RC.static_from_hours(hour_meds, lut, llr_min)
    llr_fin = np.concatenate([lut[RC.lut_index(r["rgb"])][g & ~static_sub] for r, g in rings])
    share_final = float((llr_fin >= llr_min).sum()) / max(n_ring, 1)
    if share_final > RC.DESIGN["RING_FP_MAX"] + 1e-12:
        viol.append(f"LLR_MIN: ring share {share_final:.5f} with the final STATIC > RING_FP_MAX")
    rec("LLR_MIN", llr_min, "smallest LUT value at which (ring pixels calf-like AND not STATIC0) / (ring pixels) <= "
        f"RING_FP_MAX {RC.DESIGN['RING_FP_MAX']}; ring = dil(T, RING_PX) \\ dil(T, STRUCT_DIL) at 1/SUB; held-out "
        "(UNSEEN ∪ VALID) daylight colour trusted frames; STATIC0 built at LLR_MIN0 (no STATIC exclusion); the frozen "
        "STATIC is rebuilt at LLR_MIN (superset of STATIC0)", RC.TRAIN_DAYS, n_ring,
        {"share": round(share, 5), "share_final_static": round(share_final, 5), "llr_min0": llr0, "share0": round(share0, 5),
         "frames": len(held), "static0_px_sub": static0_px, "static_px_sub": int(static_sub.sum())})
    # train background for the shift check: per-pixel median over the hour medians (pixels with >= N_BG_MIN samples)
    hm = np.stack([m for m, c in hour_meds]); hc = np.stack([c for m, c in hour_meds])
    train_bg, _cnt = RC.masked_median(hm, hc < RC.DESIGN["N_BG_MIN"])
    # TAU_CONFLICT
    for p in ("dayl", "dark"):
        v = acc["conflict"][p]
        if not v: raise RuntimeError(f"TAU_CONFLICT[{p}]: no samples")
    C["TAU_CONFLICT"] = {p: RC.ceil_2dp(RC.pct(acc["conflict"][p], 99.5)) for p in ("dayl", "dark")}
    for p in ("dayl", "dark"):
        if not READ_C["DISJ_IOS"] <= C["TAU_CONFLICT"][p] <= READ_C["VETO_IOS"]:
            viol.append(f"TAU_CONFLICT[{p}] {C['TAU_CONFLICT'][p]} outside [DISJ_IOS {READ_C['DISJ_IOS']}, VETO_IOS {READ_C['VETO_IOS']}]")
    rec("TAU_CONFLICT", C["TAU_CONFLICT"], "ceil_2dp(P99.5 of ios(d*_c, T_c')) over ordered pairs of trusted teacher masks "
        "c != c' of one frame (touching included); d* = Hungarian IoU >= 0.5 match; 0.5 Hz; UNSEEN ∪ VALID clips",
        RC.TRAIN_DAYS, {p: len(acc["conflict"][p]) for p in ("dayl", "dark")},
        {p: RC.pcts(acc["conflict"][p]) for p in ("dayl", "dark")}, uv)
    # TAU_CONTAIN from synthetic fragments (bled by the sampled boundary offset)
    offs = [float(o) for o in acc["offsets"]]
    if not offs: raise RuntimeError("boundary offsets: no samples")
    cs = {"dayl": [], "dark": []}; co = {"dayl": [], "dark": []}
    def _syn(pk):
        rng = np.random.default_rng([rng_seed, pk["ci"], pk["fidx"]])
        T = [RC.runs_of({"size": [RC.H, RC.W], "counts": t.tolist()}) for t in pk["T"]]
        D = [RC.runs_of({"size": [RC.H, RC.W], "counts": d.tolist()}) for d in pk["D"]]
        return pk["pool"], RC.syn_cover_samples(T, D, pk["match"], offs, rng, RC.trig_lo_px(pk["pool"], C))
    with ThreadPoolExecutor(max_workers=int(W["workers"] or 8)) as ex:
        for p, (a, b) in ex.map(_syn, acc["det_frames"]):
            cs[p] += a; co[p] += b
    C["TAU_CONTAIN"] = {}
    for p in ("dayl", "dark"):
        if not cs[p]: raise RuntimeError(f"TAU_CONTAIN[{p}]: no synthetic fragments")
        v = RC.floor_2dp(RC.pct(cs[p], 1)); C["TAU_CONTAIN"][p] = v
        oth = RC.pct(co[p], 99.9) if co[p] else 0.0
        lo, hi = RC.SANITY_TAU_CONTAIN
        bad = []
        if not lo <= v <= hi: bad.append(f"TAU_CONTAIN[{p}] {v} outside {RC.SANITY_TAU_CONTAIN}")
        if oth >= v: bad.append(f"P99.9 of cover(F_syn, d_other)[{p}] {oth:.4f} >= TAU_CONTAIN {v}")
        viol += bad
    rec("TAU_CONTAIN", C["TAU_CONTAIN"], "floor_2dp(P1 of cover(F_syn, d*)) over the 10 synthetic body-part fragments of "
        "every matched trusted teacher mask (bled by k ~ the signed T-vs-d* boundary offset distribution), 0.5 Hz, "
        "UNSEEN ∪ VALID clips; asserted P99.9 of cover(F_syn, d_other) < value", RC.TRAIN_DAYS,
        {p: len(cs[p]) for p in ("dayl", "dark")},
        {p: dict(RC.pcts(cs[p]), other_p99_9=round(RC.pct(co[p], 99.9), 5) if co[p] else None) for p in ("dayl", "dark")}, uv)
    rep["boundary_offset_px"] = RC.pcts(offs)
    # CARRY_MAX_S
    cw, table = RC.carry_window({dt: acc["carry"][dt] for dt in RC.DESIGN["CARRY_GRID"]})
    C["CARRY_MAX_S"] = cw
    rec("CARRY_MAX_S", cw, f"largest Δ on {RC.DESIGN['CARRY_GRID']} s (contiguous from 0) with P5 of IoU(T_t, T_t+Δ) >= "
        f"CARRY_Q {RC.DESIGN['CARRY_Q']} over trusted teacher masks of one segment in ACTIVE rows (activity >= ACTIVE_THR[day]); "
        "all 46 clips; 0 = carry off", RC.TRAIN_DAYS, sum(len(v) for v in acc["carry"].values()), table)
    # BL_P99, SOL_P1
    C["BL_P99"] = round(RC.pct(acc["bl"], 99), 1); C["SOL_P1"] = round(RC.pct(acc["sol"], 1), 4)
    rec("BL_P99", C["BL_P99"], "P99 of 4 sqrt(lambda_max) of trusted daylight teacher masks (every 30 s, all 46 clips)",
        RC.TRAIN_DAYS, len(acc["bl"]), RC.pcts(acc["bl"]))
    rec("SOL_P1", C["SOL_P1"], "P1 of solidity (area / convex hull of the boundary pixel corners) of the same masks",
        RC.TRAIN_DAYS, len(acc["sol"]), RC.pcts(acc["sol"]))
    # STATIC full-res check of the B ship gate + DYN per B-gate clip
    static_full = RC.upsample_mask(static_sub)
    dyn_by = {}
    for stem, L in acc["bgclip"].items():
        L = [r for r in L if r["chroma"] is not None and r["chroma"] >= C["CHROMA_MIN"]]
        if len(L) < RC.DESIGN["N_BG_MIN"]: continue
        med, cnt = RC.masked_median(np.stack([r["rgb"] for r in L]), np.stack([r["excl"] for r in L]))
        dyn_by[stem] = RC.upsample_mask(RC.structure_from_background(med, cnt, lut, llr_min))
    gate_rows = []
    def _bgate(pk):
        rgb = pk["rgb"]
        allm = np.zeros((RC.H, RC.W), bool)
        T = [RC.runs_of({"size": [RC.H, RC.W], "counts": t.tolist()}) for t in pk["T"]]
        for t in T: allm |= RC.runs_to_dense(t)
        ch = RC.chroma_of(rgb, allm)
        if ch is None or ch < C["CHROMA_MIN"] or pk["clip"] not in dyn_by: return []
        D = [RC.runs_of({"size": [RC.H, RC.W], "counts": d.tolist()}) for d in pk["D"]]
        rng = np.random.default_rng([rng_seed, 7, pk["ci"], pk["fidx"]])
        rows = []
        for c, dj in sorted(pk["match"].items()):
            y0, y1, x0, x1 = RC.crop_box([T[c]], RC.DESIGN["SYN_K_ABS_MAX"] + 2)
            Tc = RC.runs_to_crop(T[c], y0, y1, x0, x1)
            Td3 = RC.dilate(RC.runs_to_dense(T[c]), RC.DESIGN["STRUCT_DIL"])
            for _nm, Fc in RC.synth_fragments(Tc):
                Fb = RC.bleed(Fc, offs[int(rng.integers(len(offs)))])
                F = RC.runs_from_crop(Fb, y0, x0)
                if F.area == 0 or F.area >= RC.trig_lo_px("dayl", C): continue
                dets = [D[j] for j in range(len(D)) if j != dj]
                if any(RC.cover(F, d) >= C["TAU_CONFLICT"]["dayl"] for d in dets):
                    rows.append({"accepted": False, "skip": "not_nodet"}); continue
                G, info = RC.grow_row(rgb, F, [T[k] for k in range(RC.N_SLOTS) if k != c], dets, static_full, dyn_by[pk["clip"]],
                                      lut, llr_min, C)
                r = {"accepted": G is not None, "why": info.get("why"), "iou_before": RC.iou_r(F, T[c]), "a_ref": C["A_REF"]}
                if G is not None:
                    Gd = RC.runs_to_dense(G)
                    r["iou_after"] = RC.iou_r(G, T[c]); r["false_fill_px"] = int((Gd & ~Td3).sum())
                    r["other_fill_px"] = max(RC.inter(G, T[k]) for k in range(RC.N_SLOTS) if k != c)
                rows.append(r)
        return rows
    with ThreadPoolExecutor(max_workers=int(W["workers"] or 8)) as ex:
        for rr in ex.map(_bgate, acc["bgate"]): gate_rows += [r for r in rr if "skip" not in r]
    ok, vals, crit = RC.b_gate_eval(gate_rows)
    vals["refused_by_reason"] = dict(Counter(r["why"] for r in gate_rows if not r["accepted"]))
    vals["frames"] = len(acc["bgate"]); vals["clips_with_dyn"] = len(dyn_by)
    print(f"B ship gate: {'PASS' if ok else 'FAIL'} {vals}")
    rep.update(constants={k: C.get(k) for k in RC.DERIVED_KEYS}, derived=out, b_ship_gate={"pass": ok, **vals},
               det_ms_median=RC.pct(acc["det_ms"], 50), minutes=round((time.time() - t_all) / 60, 1), gpu=GPU_NAME,
               violations=viol)
    if viol:
        fp = osp.join(CONST_DIR, f"derive_report_FAILED_{datetime.now():%Y%m%d_%H%M%S}.json")
        publish_json(ENV, {"LOCAL": LOCAL}, rep, fp)
        msg = f"derive sanity check(s) failed -- nothing frozen; every distribution is in {fp}: " + "; ".join(viol)
        if not SMOKE: raise RuntimeError(msg)
        print("⚠ SMOKE (continuing): " + msg)
    # ---------------- write the frozen artefacts ----------------
    lp = osp.join(LOCAL, "colour_lut_v1.npz"); RC.save_npz_det(lp, lut=lut, hist_pos=hp, hist_neg=hn)
    arts = {}
    for name, data in (("colour_lut_v1.npz", open(lp, "rb").read()), ("static_structure_v1.png", RC.png_bytes(static_sub)),
                       ("train_bg_v1.png", RC.png_bytes(train_bg))):
        arts[name] = RC.sha256_bytes(data)
        loc = osp.join(LOCAL, name)
        with open(loc, "wb") as fh: fh.write(data)
        dst = osp.join(CONST_DIR, name)
        if osp.exists(dst):
            if RC.sha256_file(dst) != arts[name]: raise RuntimeError(f"frozen artefact {dst} exists with different content")
        else:
            publish(loc, dst)
    consts = {"version": "refine_constants_v1", "spec": RC.REFINE_SPEC, "core_version": RC.REFINE_CORE_VERSION,
              "smoke": SMOKE, "READ": {k: READ_C[k] for k in ("A_REF", "FRAG_LO", "OVER_HI", "TRIG_LO", "LINK_IOU_MIN",
                                                             "BODY_LEN_PX", "ACTIVE_THR", "DISJ_IOS", "VETO_IOS")},
              "READ_provenance": READ_C["provenance"], "DERIVED": out, "DESIGN": RC.DESIGN,
              "SANITY": {"TAU_CONTAIN": RC.SANITY_TAU_CONTAIN, "TAU_CONFLICT": [READ_C["DISJ_IOS"], READ_C["VETO_IOS"]]},
              "b_ship_gate": {"pass": ok, "values": vals, "criteria": crit, "frames_every_s": RC.DESIGN["B_GATE_EVERY_S"]},
              "artefacts": arts,
              "inputs": {"train_summary_sha256": RC.sha256_file(osp.join(DET_CKPT_DIR, "train_summary.json")),
                         "faults_json": W["faults_json"], "faults_json_sha256": READ_C["faults_json_sha256"],
                         "relabel_sha256": {w: rel_cache[w][3] for w in sorted(rel_cache)},
                         "clips": [[w, s, r] for w, s, r in clips], "detector_sha256": DET_SHA,
                         "gpu_class": GPU_CLASS, "refine_core_sha256": RC_SHA, "v5_core_sha256": V5_SHA,
                         "notebook_source_norm_sha256": SRC_NORM_SHA}}
    cb = json.dumps(consts, indent=1, default=str).encode("utf-8")
    cp = osp.join(CONST_DIR, "refine_constants_v1.json")
    if osp.exists(cp):
        if RC.sha256_file(cp) != RC.sha256_bytes(cb):
            raise RuntimeError(f"{cp} exists with DIFFERENT content: frozen, never overwritten (delete by hand to re-derive)")
    else:
        loc = osp.join(LOCAL, "refine_constants_v1.json")
        with open(loc, "wb") as fh: fh.write(cb)
        publish(loc, cp)
    rep.update(constants_sha256=RC.sha256_bytes(cb), minutes=round((time.time() - t_all) / 60, 1))
    rp = osp.join(CONST_DIR, "derive_report_v1.json")
    if not osp.exists(rp): publish_json(ENV, {"LOCAL": LOCAL}, rep, rp)
    print(json.dumps({k: C[k] for k in RC.DERIVED_KEYS}, indent=1))
    print(f"derive done in {(time.time() - t_all) / 60:.1f} min -> {CONST_DIR}")
    return rep

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main

# COMMAND ----------

def load_grow_artefacts(C):
    if not C.get("B_SHIP", {}).get("pass"):
        raise RuntimeError("(B) colour grow refused: the B ship gate did not pass on the training days -- reported only")
    lut = np.load(osp.join(CONST_DIR, "colour_lut_v1.npz"))["lut"]
    st = RC.png_read(osp.join(CONST_DIR, "static_structure_v1.png")) > 0
    bg = RC.png_read(osp.join(CONST_DIR, "train_bg_v1.png"))
    for name in ("colour_lut_v1.npz", "static_structure_v1.png", "train_bg_v1.png"):
        if RC.sha256_file(osp.join(CONST_DIR, name)) != RC.load_json(osp.join(CONST_DIR, "refine_constants_v1.json"))["artefacts"][name]:
            raise RuntimeError(f"{name} sha != the constants file's artefact sha")
    return {"lut": lut, "static_full": RC.upsample_mask(st), "train_bg_sub": bg}

def main_day():
    t_run = time.time()
    need_derived = DO_APPLY or DO_GROW
    if need_derived:
        cpath = osp.join(CONST_DIR, "refine_constants_v1.json")
        if not osp.exists(cpath): raise RuntimeError(f"{cpath} missing: run mode=derive first (S1); mode=detect needs no constants")
        C = RC.load_refine_constants(cpath, READ_C); RC.check_apply_constants(C)
        if C["smoke"]: raise RuntimeError("the constants file is a SMOKE derivation: never applied")
        C["NO_SHRINK"] = NO_SHRINK
        print(f"(A) gates: TAU_CONTAIN {C['TAU_CONTAIN']}, TAU_CONFLICT {C['TAU_CONFLICT']}, CARRY_MAX_S {C['CARRY_MAX_S']}, "
              f"no_shrink {NO_SHRINK}{'' if NO_SHRINK else ' (v1 rule)'}")
    else:
        C = dict(READ_C)
    src_cfg = RC.load_json(osp.join(SRC_DIR, "ROOT_CONFIG.json"))
    ctx = {"LOCAL": LOCAL, "SRC_DIR": SRC_DIR, "OUT_DIR": OUT_DIR, "GROW_DIR": GROW_DIR, "C": C, "sample": SAMPLE,
           "parity": W["parity"], "parity_max_per_clip": int(W["parity_max_per_clip"] or 0),
           "parity_exact": (GPU_CLASS == src_cfg.get("gpu_class")) if NEED_GPU else None,
           "sheets": W["sheets"].lower() == "true", "sheet_cases": SHEET_CASES, "day": DAY, "src_root": W["src_root"],
           "det_run_tag": W["det_run_tag"],
           # segments only for apply / grow (carry); detect never reads them -- on a day the tracker is still writing,
           # segments.json is rewritten in place (Tol_47 write_segments, plain open('w')) and a read can see half a file
           "segs": RC.SegIndex(V5.read_json_retry(osp.join(SRC_DIR, "segments.json"))) if (DO_APPLY or DO_GROW) else None,
           "src_sha": {}, "gpu_class": GPU_CLASS, "det_sha": DET_SHA, "src_norm_sha": SRC_NORM_SHA}
    if NEED_GPU:
        print(f"E0 parity: {'EXACT (same GPU class ' + str(GPU_CLASS) + ')' if ctx['parity_exact'] else 'TOLERANCE (production ' + str(src_cfg.get('gpu_class')) + ', here ' + str(GPU_CLASS) + ')'}; mode {W['parity']}")
    stems, not_done = source_clips(ctx)
    if not_done: print(f"⚠ {len(not_done)} source clips without .done_ (still tracking?) are skipped: {not_done[:3]}")
    day_complete = not not_done
    if (DO_APPLY or DO_GROW) and not day_complete and not SMOKE:
        raise RuntimeError(f"{DAY}: the source day is incomplete ({len(not_done)} clips not done): apply / grow need the "
                           "final segments.json -- run mode=detect now, apply when the tracker has finished")
    if MAX_CLIPS: stems = stems[:MAX_CLIPS]
    if ctx["sheets"] and DO_APPLY: print(f"review crops on ({len(SHEET_CASES)} listed cases)")
    dref = osp.join(OUT_DIR, "REFINE_DETECT.json")
    if DO_DETECT:
        drec = {"detector_sha256": DET_SHA, "det_run_tag": W["det_run_tag"], "sample": SAMPLE, "src_root": W["src_root"]}
        if osp.exists(dref):
            old = RC.load_json(dref)
            diff = [(k, old.get(k), v) for k, v in drec.items() if old.get(k) != v]
            if diff: raise RuntimeError(f"REFINE_DETECT mismatch (one detector config per day dir): {diff}")
        else:
            publish_json(ENV, ctx, dict(drec, gpu_class=GPU_CLASS, refine_core_sha256=RC_SHA,
                                        created=datetime.now().isoformat(timespec="seconds")), dref)
    det_sha_cfg = DET_SHA if DO_DETECT else (RC.load_json(dref)["detector_sha256"] if osp.exists(dref) else None)
    if (DO_APPLY or DO_GROW) and det_sha_cfg is None:
        raise RuntimeError(f"{dref} missing: the detections of this day dir were not made by mode=detect / redetect")
    cfg_rec = {"spec": RC.REFINE_SPEC, "src_root": W["src_root"], "sample": SAMPLE, "det_run_tag": W["det_run_tag"],
               "no_shrink": NO_SHRINK, "no_shrink_note": RC.NO_SHRINK_NOTE,
               "constants_sha256": C.get("constants_sha256"), "detector_sha256": det_sha_cfg,
               "refine_core_sha256": RC_SHA, "v5_core_sha256": V5_SHA, "notebook_source_norm_sha256": SRC_NORM_SHA,
               "widgets": W, "gpu_class": GPU_CLASS, "constants": {k: C.get(k) for k in RC.DERIVED_KEYS + ("A_REF", "FRAG_LO",
                                                                                                        "OVER_HI", "TRIG_LO", "LINK_IOU_MIN")},
               "created": datetime.now().isoformat(timespec="seconds")}
    if DO_APPLY:
        print(f"REFINE_CONFIG {OUT_DIR}: {check_refine_config(ENV, ctx, OUT_DIR, cfg_rec)}")
    if DO_GROW:
        ctx["grow"] = load_grow_artefacts(C)
        print(f"REFINE_CONFIG {GROW_DIR}: {check_refine_config(ENV, ctx, GROW_DIR, cfg_rec)}")
    telem_p = osp.join(TOL, f"Metrics_refine_{W['out_root']}_{DAY}.json")
    TEL = RC.load_json(telem_p) if osp.exists(telem_p) else {"day": DAY, "runs": [], "clips": {}}
    run_rec = {"mode": MODE, "started": datetime.now().isoformat(timespec="seconds"), "gpu": GPU_NAME,
               "gpu_class": GPU_CLASS, "sample": SAMPLE, "widgets": W, "smoke": SMOKE}
    TEL["runs"].append(run_rec)
    def save_tel():
        TEL["updated"] = datetime.now().isoformat(timespec="seconds")
        publish_json(ENV, ctx, TEL, telem_p)
    def on_parity_fail(stem, parity, got, st):
        """E0 strict failure: telemetry + the full comparison evidence on the Volume, then detect_clip raises."""
        fails = [p for p in parity if not p["ok"]]
        TEL["clips"].setdefault(stem, {})["detect_parity_fail"] = {
            "n": len(parity), "n_fail": len(fails), "fail": fails[:50], "gpu_class": GPU_CLASS, "exact": ctx["parity_exact"],
            "decode_wait_s": round(st.get("decode_wait_s", 0.0), 1), "stage_note": "no detections_refine written for this clip"}
        run_rec["failed"] = f"E0 parity {stem}: {len(fails)}/{len(parity)}"
        try:
            publish_json(ENV, ctx, {"clip": stem, "gpu_class": GPU_CLASS, "exact": ctx["parity_exact"], "parity": parity,
                                    "records": [got[f] for f in sorted(got)]},
                         osp.join(OUT_DIR, f"parity_fail_{stem}.json.gz"), gz=True)
        finally:
            save_tel()
    ctx["on_parity_fail"] = on_parity_fail
    enc = Encoder(4) if DO_DETECT else None
    prefetch = ThreadPoolExecutor(max_workers=1) if (W["prefetch"].lower() == "true" and (DO_DETECT or DO_GROW or ctx["sheets"])) else None
    fut = None
    def video_needed(stem):
        if DO_DETECT and not osp.exists(osp.join(OUT_DIR, f".det_done_{stem}")): return True
        if DO_APPLY and not osp.exists(osp.join(OUT_DIR, f".done_{stem}")) and ctx["sheets"]: return True
        if DO_GROW and not osp.exists(osp.join(GROW_DIR, f".done_{stem}")): return True
        return False
    todo = [s for s in stems if video_needed(s) or (DO_APPLY and not osp.exists(osp.join(OUT_DIR, f".done_{s}")))]
    print(f"{len(stems)} clips, {len(todo)} to do")
    vslot = [osp.join(LOCAL, "v0.mp4"), osp.join(LOCAL, "v1.mp4")]
    def _stage(k):
        stem = todo[k]
        if not video_needed(stem): return None, 0.0
        return vslot[k % 2], stage_video(stem + ".mp4", vslot[k % 2])     # stem = video name (no duplicates here)
    for k, stem in enumerate(todo):
        t_clip = time.time(); ent = TEL["clips"].get(stem, {})
        if fut is not None:
            vpath, t_stage = fut.result()
        else:
            vpath, t_stage = _stage(k)
        fut = prefetch.submit(_stage, k + 1) if (prefetch is not None and k + 1 < len(todo)) else None
        src, stored = load_clip_inputs(ENV, ctx, stem)            # also records ctx["src_sha"][stem]
        if src["clip"].replace(" ", "_")[:-4] != stem: raise RuntimeError(f"{stem}: masks header clip {src['clip']!r}")
        vid = ENV.open_video(vpath) if vpath else None
        new = None
        if DO_DETECT and not osp.exists(osp.join(OUT_DIR, f".det_done_{stem}")):
            t0 = time.time()
            recs, par, st = detect_clip(ENV, ctx, stem, src, stored, vid, enc)
            ms = sorted(st.pop("det_ms"))
            st.update(det_ms_median=ms[len(ms) // 2] if ms else None, det_ms_p95=ms[int(0.95 * (len(ms) - 1))] if ms else None,
                      det_gpu_min=round(sum(ms) / 60000, 2), detect_wall_min=round((time.time() - t0) / 60, 2),
                      stage_min=round(t_stage / 60, 2))
            nb = publish_json(ENV, ctx, det_file_obj(ctx, stem, recs, dict(st, parity=par)),
                              osp.join(OUT_DIR, f"detections_refine_{stem}.json.gz"), gz=True)
            touch_marker(ENV, ctx, osp.join(OUT_DIR, f".det_done_{stem}"))
            st["bytes"] = nb; ent["detect"] = dict(st, parity_failures=[p for p in par if not p["ok"]][:20]); new = recs
            print(f"  [{k + 1}/{len(todo)}] {stem}: detect {st['plan']} calls ({st['det_gpu_min']} GPU-min, "
                  f"{st['detect_wall_min']} min wall), E0 {st['parity_n'] - st['parity_fail']}/{st['parity_n']} OK", flush=True)
        if DO_APPLY and not osp.exists(osp.join(OUT_DIR, f".done_{stem}")):
            if new is None: new = load_new_records(ENV, ctx, stem)
            rep = apply_clip_driver(ENV, ctx, stem, src, stored, new, vid)
            ent["apply"] = rep
            print(f"  [{k + 1}/{len(todo)}] {stem}: apply {rep['codes']} refused {rep['refused_by_reason']} "
                  f"({rep['apply_s']} s)", flush=True)
        if DO_GROW and not osp.exists(osp.join(GROW_DIR, f".done_{stem}")):
            if new is None: new = load_new_records(ENV, ctx, stem)
            gs = grow_clip_driver(ENV, ctx, stem, src, stored, new, vid)
            ent["grow"] = gs
            print(f"  [{k + 1}/{len(todo)}] {stem}: grow {gs['grown']} of {gs['eligible_rows']} eligible rows "
                  f"(refused {gs['refused']}, shift {gs['shift']})", flush=True)
        if vid is not None: vid.close()
        if vpath and osp.exists(vpath): os.remove(vpath)
        ent["clip_min"] = round((time.time() - t_clip) / 60, 2); TEL["clips"][stem] = ent
        save_tel()
    if prefetch is not None: prefetch.shutdown(wait=True)
    if enc is not None: enc.close()
    if DO_APPLY and day_complete and not SMOKE:
        done = sorted((s for s in source_clips(ctx)[0] if osp.exists(osp.join(OUT_DIR, f".done_{s}"))), key=RC.clip_start)
        if len(done) == len(source_clips(ctx)[0]):
            TEL["finalize"] = finalize_day(ENV, ctx, OUT_DIR, done)
            print(f"day finalized: {TEL['finalize']}")
    if DO_GROW and day_complete and not SMOKE:
        done = sorted((s for s in source_clips(ctx)[0] if osp.exists(osp.join(GROW_DIR, f".done_{s}"))), key=RC.clip_start)
        if len(done) == len(source_clips(ctx)[0]):
            TEL["finalize_grow"] = finalize_day(ENV, ctx, GROW_DIR, done)
    run_rec["minutes"] = round((time.time() - t_run) / 60, 1)
    save_tel()
    print(f"Tol_55 {MODE} {DAY}: done in {run_rec['minutes']} min; telemetry {telem_p}")

def main_census():
    C = dict(READ_C)
    ctx = {"SRC_DIR": SRC_DIR, "LOCAL": LOCAL}
    stems, not_done = source_clips(ctx)
    res = {}
    for stem in (stems[:MAX_CLIPS] if MAX_CLIPS else stems):
        src = RC.load_json_gz(osp.join(SRC_DIR, f"masks_{stem}.json.gz"))
        dp = osp.join(SRC_DIR, f"detections_{stem}.json.gz")
        st = RC.stored_by_fidx(RC.load_json_gz(dp)) if osp.exists(dp) else {}
        res[stem] = {s: RC.census_clip(src["frames"], src["clip"], C, st.keys(), sample=s) for s in ("1", "5", "render")}
        print(stem, {s: res[stem][s]["new_calls"] for s in res[stem]}, flush=True)
    tot = {s: {k: sum(v[s][k] for v in res.values()) for k in ("frames", "trigger_rows", "trigger_frames", "new_calls",
                                                                   "trigger_frames_stored_call")} for s in ("1", "5", "render")}
    out = {"day": DAY, "src_root": W["src_root"], "trig_lo": READ_C["TRIG_LO"], "clips": res, "totals": tot,
           "source_clips_not_done": not_done, "created": datetime.now().isoformat(timespec="seconds")}
    p = osp.join(TOL, f"Metrics_refine_census_{W['src_root']}_{DAY}.json")
    with open(osp.join(LOCAL, "census.json"), "w") as f: json.dump(out, f, indent=1)
    publish(osp.join(LOCAL, "census.json"), p)
    print(json.dumps(tot, indent=1)); print(f"-> {p}")

if MODE == "derive":
    derive()
elif MODE == "census":
    main_census()
else:
    main_day()
