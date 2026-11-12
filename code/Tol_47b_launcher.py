# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N47b — v5 launcher: (script, env) jobs as subprocesses on ONE task-owned GPU node
# MAGIC
# MAGIC **Lineage.** A copy of `Tol_03b_multiplex.py` (N3b) adapted for tracker v5 (with the
# MAGIC binding decisions of 2026-09-24). Why a launcher at all: a SHARED job cluster exposes **no GPU** to its task REPLs on
# MAGIC this workspace (measured 2026-08-17), so every GPU task owns its node, and processes that must share
# MAGIC one node (the E6 ref ‖ reg arms; equivalence then drift floor) run as subprocesses of this one task.
# MAGIC
# MAGIC **What it does.**
# MAGIC 1. Installs the **v5 environment**: the three pip lines verbatim from `Tol_44_detector_eval.py:70-72`, so
# MAGIC    the E6 reference arm (`Tol_03_track_v4` as a subprocess) runs in the same environment as v5 and E6 compares code only.
# MAGIC 2. Stages the worker scripts from the Volume (`Tol/Code/v5/`) and **checks each against a sha256 pin** from the payload,
# MAGIC    plus a static contract check (env prefix, staging-service protocol, every widget name present).
# MAGIC 3. **CUDA probe in a subprocess**: `torch.zeros(1, device='cuda')` + device name, version asserts,
# MAGIC    `nvidia-smi -q` driver/licence lines only; retries, then raises. The launcher itself never initialises CUDA, so the
# MAGIC    whole card belongs to the workers and device-memory numbers are not confounded by a launcher context.
# MAGIC 4. Runs the jobs in **phases**: phases run in order, the jobs of one phase run concurrently. Each job's widgets are a
# MAGIC    named preset (`v5` = the v5 default values; `regression` = the regression values;
# MAGIC    `t03` = Tol_03 with the v2 student) plus per-job overrides, validated against the target script's widget list and
# MAGIC    mapped to `TOLV5_<NAME>` / `TOLV4_<NAME>` env vars. Unknown widget names raise (an unforwarded or misspelt widget
# MAGIC    silently takes the worker default — lesson of 2026-09-02).
# MAGIC 5. Acts as the workers' **staging service** (workers have no `dbutils`; `shutil` over the /dbfs FUSE runs at ~0.5 MB/s):
# MAGIC    one request directory **per job**, so two concurrent jobs of the same `stage` can request the same clip without
# MAGIC    colliding (N3b keyed requests by stage only).
# MAGIC 6. Logs to `Tol/Launcher_logs/<launch_tag>/`: per-job worker logs (synced during the run), `summary.json`, and
# MAGIC    `attempts.json` (cluster id per attempt: a task retry must land on a new cluster; with
# MAGIC    `retry_policy=probe_only` a retry after a job failure behind a healthy probe is refused).
# MAGIC 7. (tol47b-2) **Verifies what the workers actually imported**: every Tol_47 worker prints
# MAGIC    `tol_v5_core … from <path> sha256 <hex>` right after its import; the launcher parses it within ~30 s and kills
# MAGIC    the job if that sha is not the staged, pinned core (a stale `/Workspace` core could shadow the
# MAGIC    staged copy). **Attributes every `mode=equiv` verdict file** (`Metrics_v5_equiv_<GPU>_<timestamp>.json`, which
# MAGIC    carries no launch tag) to its job by the worker's own log line and `echo.stage_svc_key`, and copies it to
# MAGIC    `Tol/Metrics_v5_equiv_<GPU>_<launch_tag>__<job>.json` (`summary.equiv_outputs`).
# MAGIC    **E6 phase order**: an E6 verdict must compare arms written by earlier phases of the same
# MAGIC    launch, and no other job (e.g. the report-only drift floor) may run in an earlier phase, where its failure would
# MAGIC    skip the verdict under `stop_on_fail`.
# MAGIC 8. (tol47b-3, launcher review r1) **At most one `mode=equiv` job per phase** (their verdict files are named by a
# MAGIC    one-second timestamp, so two concurrent verdicts could collide — finding 7); verdicts therefore run one per
# MAGIC    phase, and a failed E6 verdict — a read-only comparison — no longer skips the later phases (only a core-sha
# MAGIC    mismatch does). **Launcher-only roots are never resumed** (finding 3): a `mode=track` job whose
# MAGIC    `Tol/<root>/<daydir>` or telemetry file already exists raises `config_error` before the probe, unless
# MAGIC    `reset_launcher_roots=true`, which MOVES them to `Tol/Launcher_attic/<launch_tag>_<UTC stamp>/` — never
# MAGIC    deletes — and records the moves in `summary.reset`. **Direct roots are never reset**: a change after one was
# MAGIC    tracked goes into a new REVISION `<root>_r<N>` (`make_v5_payloads.py --root-rev N`), which inherits the base
# MAGIC    root's policy (`root_base`); the checker refuses a payload set that mixes revisions of one base root.
# MAGIC 9. (tol47b-4, launcher review r2) **A launcher-only root belongs to ONE payload** (the checker refuses two payloads
# MAGIC    writing one; the two conditional sequential payloads now have their own roots `…_seq_smoke` / `…_seq_window`),
# MAGIC    and a reset moves **everything** in each launcher-only root the launch writes: its day dirs + telemetry, the
# MAGIC    root's `ROOT_CONFIG.json` (always — a stale one raises after any Tol_47 change) and any stale day dir of an older
# MAGIC    payload layout (+ its telemetry), so a reset root can never keep a ROOT_CONFIG it was not written under
# MAGIC    (review r2 finding 1). **Deployed = reviewed** (review r2 finding 3): the payload's `expect_launcher` must equal
# MAGIC    this notebook's `LAUNCHER_VERSION` (else `config_error` before anything runs), and the summary records
# MAGIC    `launcher` plus `launcher_source` (the transport-normalised sha256 of the RUNNING notebook, read like Tol_47
# MAGIC    reads its own source) for `gate_v5` to compare with the reviewed local file.
# MAGIC
# MAGIC **Never** mix launcher runs and direct `Tol_47_track_v5` notebook runs in one tracking root (the source sha that
# MAGIC `ROOT_CONFIG.json` pins is computed from different files in the two modes).
# MAGIC
# MAGIC Local use (no dbutils / torch / cv2 needed): `TOL_SELFTEST=1 python Tol_47b_launcher.py` runs the SELFTEST (pure logic +
# MAGIC fake workers that execute Tol_03's real `stage_clip`); `... --emit-registry` prints the widget lists and presets as JSON;
# MAGIC `... --check-payloads DIR [--strict] [--tol47 FILE]` validates job payloads.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm" "scipy" "rfdetr==1.10.1" "pycocotools" "peft"
# MAGIC %pip uninstall -y opencv-python opencv-contrib-python
# MAGIC %pip install --no-deps --force-reinstall "opencv-python-headless==4.11.0.86"

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass   # local SELFTEST (TOL_SELFTEST=1): plain python

# COMMAND ----------

import os, os.path as osp, posixpath as pp, sys, re, json, time, glob, shutil, hashlib, subprocess, tempfile
from datetime import datetime, timezone

SELFTEST_ONLY = os.environ.get("TOL_SELFTEST") == "1"
IS_NOTEBOOK = True
try:
    dbutils  # noqa: F821
except NameError:
    IS_NOTEBOOK = False
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows cp1252 console cannot print '‖' / '⚠'

# COMMAND ----------

# MAGIC %md ## 1 · Pure logic (no dbutils, no GPU) — everything the SELFTEST and the payload checker exercise

# COMMAND ----------

LAUNCHER_VERSION = "tol47b-4 (2026-09-25)"
VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = pp.join(VOL, "Tol")
DBFS_VIDEO = "dbfs:/mnt/playbehavior/Tol/Videos"
NB_DIR = "/Users/<DATABRICKS_USER>/Model Distillation/Testing on Tol2"
LOCAL_ROOT = "/local_disk0/tmp/tol_v5_launcher"
ALLOC_CONF = "expandable_segments:True"                         # job spark_env_vars + asserted here
T03_SHA256 = "d7fe1cf4f87634f4951deb2e868f7b74dd8012311ad3a35065d7d27f2d372181"   # Tol_03 must stay byte-identical
CLIP_RX = re.compile(r'^(Tol_ch\d+)[-_](\d{8})[-_](\d{6})[-_](\d{6})(?: \((\d+)\))?\.mp4$')   # = Tol_03:241
HEX64 = re.compile(r"^[0-9a-f]{64}$")
NAME_RX = re.compile(r"^(?!.*__)[A-Za-z0-9_.-]{1,64}$")          # no "__": it separates fields in req__KEY__clip
ROOT_RX = re.compile(r"^Tracking_[A-Za-z0-9_]+$")

class ConfigError(RuntimeError): pass
class CudaProbeError(RuntimeError): pass
class JobsFailed(RuntimeError): pass

# ---- widget registries ----------------------------------------------------------------------------------------------
# Tol_03_track_v4.py: every P("...") call (T03:78-100); the SELFTEST re-extracts them from the file and compares.
TOL03_WIDGETS = ("stage", "max_clips", "max_frames", "from_hhmm", "to_hhmm", "track_root", "track_fps",
                 "allow_nonstandard_fps", "detect_day_sec", "detect_night_sec", "refresh_min_sec", "embed_every_sec",
                 "crop_every_sec", "backbone", "student_ckpt", "sam3_model", "partial_repair")
# Tol_47_track_v5.py: the widgets unchanged from T03, then the new or changed ones.
TOL47_UNCHANGED = ("stage", "max_clips", "max_frames", "from_hhmm", "to_hhmm", "track_fps", "allow_nonstandard_fps",
                   "detect_day_sec", "refresh_min_sec", "embed_every_sec", "crop_every_sec", "backbone", "sam3_model",
                   "partial_repair")
TOL47_NEW = ("mode", "track_root", "student_ckpt", "detector", "det_run_tag", "dark_fire", "dark_link", "dark_open",
             "dark_repeat_sec", "detect_night_sec", "v5_fixes", "pop_frames", "exact_fast", "decode", "gpu_tf",
             "allow_numeric_change", "mem_cap_gib", "mem_assert_alloc_gib", "mem_assert_reserved_gib",
             "mem_assert_session_frames", "cuda_probe_retries", "cuda_probe_sleep_s", "allow_mixed_gpu")
TOL47_WIDGETS = TOL47_UNCHANGED + TOL47_NEW
# Not in the main widget list: added by tol_v5_core (WIDGETS, "mode=equiv only (C15)") so Tol_47 can run the E6 comparison.
# Allowed in a job's overrides, never required, never in a preset, excluded from ROOT_CONFIG by the core.
TOL47_OPTIONAL = ("equiv_tests", "e6_ref_root", "e6_reg_root")

# ---- value domains (a typo in a VALUE is as silent as a typo in a name) --------------------------------------------
_ONOFF, _BOOL = ("on", "off"), ("true", "false")
DOMAIN = {"mode": ("track", "equiv"), "detector": ("rfdetr", "sam3"), "decode": ("seek", "seek_thread", "sequential"),
          "backbone": ("student", "teacher")}
DOMAIN.update({k: _ONOFF for k in ("dark_fire", "dark_link", "dark_open", "v5_fixes", "pop_frames", "exact_fast",
                                   "gpu_tf")})
DOMAIN.update({k: _BOOL for k in ("allow_nonstandard_fps", "allow_numeric_change", "allow_mixed_gpu",
                                  "partial_repair")})
FLOATS = ("track_fps", "detect_day_sec", "detect_night_sec", "refresh_min_sec", "embed_every_sec", "crop_every_sec",
          "dark_repeat_sec", "mem_cap_gib", "mem_assert_alloc_gib", "mem_assert_reserved_gib", "cuda_probe_sleep_s")
INTS = ("max_clips", "max_frames", "mem_assert_session_frames", "cuda_probe_retries")
HHMM_KEYS = ("from_hhmm", "to_hhmm")

def validate_value(name, v):
    if not isinstance(v, str):
        raise ConfigError(f"widget {name}: value must be a string (payload base_parameters are strings), got {v!r}")
    if name in DOMAIN and v not in DOMAIN[name]:
        raise ConfigError(f"widget {name}={v!r} not in {DOMAIN[name]}")
    if name in FLOATS:
        try:
            f = float(v)
        except ValueError:
            raise ConfigError(f"widget {name}={v!r} is not a number")
        if f != f or f < 0: raise ConfigError(f"widget {name}={v!r} must be a finite number >= 0")
    if name in INTS and not re.fullmatch(r"\d+", v):
        raise ConfigError(f"widget {name}={v!r} is not a non-negative integer")
    if name in HHMM_KEYS and v and not (re.fullmatch(r"\d{4}", v) and int(v[:2]) <= 24 and int(v[2:]) <= 59
                                        and int(v) <= 2400):
        raise ConfigError(f"widget {name}={v!r} must be empty or HHMM (Tol_03:258-266 parses int(v[:2]), int(v[2:]))")
    if name == "stage":
        try:
            datetime.strptime(v, "%Y%m%d")
        except ValueError:
            raise ConfigError(f"widget stage={v!r} is not YYYYMMDD")
    if name in ("track_root", "e6_ref_root", "e6_reg_root") and not ROOT_RX.match(v):
        raise ConfigError(f"widget {name}={v!r} must match {ROOT_RX.pattern}")
    if name == "equiv_tests" and not re.fullmatch(r"E[1-6](,E[1-6])*", v):
        raise ConfigError(f"widget equiv_tests={v!r} must be a comma list of E1..E6")
    if name in ("student_ckpt", "sam3_model", "det_run_tag") and not v.strip():
        raise ConfigError(f"widget {name} must not be empty")

def policy_problems(script, env):
    """Binding decisions that no payload may override (2026-09-24; the production student is v2)."""
    p = []
    if float(env.get("track_fps", "5.0")) != 5.0 or env.get("allow_nonstandard_fps", "false") != "false":
        p.append("track_fps must be 5.0 with allow_nonstandard_fps=false (5 fps is fixed; INV-6)")
    if env.get("student_ckpt") != "best_backbone_student_v2.pt":
        p.append("student_ckpt must be best_backbone_student_v2.pt (the production student; v5 default)")
    if script == "Tol_47_track_v5":
        if env.get("allow_numeric_change") != "false" or env.get("gpu_tf") != "off" or \
                env.get("decode") not in ("seek", "seek_thread"):
            p.append("no pixel/feature-changing speed-up this round: decode in {seek, seek_thread}, "
                     "gpu_tf=off, allow_numeric_change=false")
    return p

# ---- presets: the ONLY place the launcher holds configuration ------------------------------------------------------
_COMMON = {"max_clips": "0", "max_frames": "0", "from_hhmm": "", "to_hhmm": "", "track_fps": "5.0",
           "allow_nonstandard_fps": "false", "detect_day_sec": "120", "refresh_min_sec": "600",
           "embed_every_sec": "10", "crop_every_sec": "60", "backbone": "student", "sam3_model": "facebook/sam3",
           "student_ckpt": "best_backbone_student_v2.pt",        # T03:95 default is v1 -> ALWAYS explicit
           "partial_repair": "true"}                             # partial repair on
PRESETS = {
    # v5 defaults: partial_repair on, night detector cadence 120 s, dark refire 10 s, no numeric change
    "v5": {**_COMMON, "mode": "track", "detector": "rfdetr", "det_run_tag": "segv1d_tracked_seg-small_r624",
           "dark_fire": "on", "dark_link": "on", "dark_open": "on", "dark_repeat_sec": "10",
           "detect_night_sec": "120", "v5_fixes": "on", "pop_frames": "on", "exact_fast": "on",
           "decode": "seek_thread", "gpu_tf": "off", "allow_numeric_change": "false", "mem_cap_gib": "0",
           "mem_assert_alloc_gib": "6", "mem_assert_reserved_gib": "10", "mem_assert_session_frames": "2",
           "cuda_probe_retries": "3", "cuda_probe_sleep_s": "90", "allow_mixed_gpu": "false"},
}
# regression values (dark_link/dark_open are "off via master"; set off explicitly)
PRESETS["regression"] = {**PRESETS["v5"], "detector": "sam3", "dark_fire": "off", "dark_link": "off",
                         "dark_open": "off", "detect_night_sec": "300", "v5_fixes": "off"}
# Tol_03_track_v4 as the E6 reference arm: T03 defaults except the student (v2) — same values as the regression arm
PRESETS["t03"] = {**_COMMON, "detect_night_sec": "300"}
REQUIRED_RUN_KEYS = ("stage", "track_root")                   # per job, never in a preset
ROOT_EXEMPT = ("stage", "from_hhmm", "to_hhmm", "max_clips", "max_frames")   # ROOT_CONFIG exemption list

SCRIPTS = {
    "Tol_47_track_v5": {"file": "Tol_47_track_v5.py", "prefix": "TOLV5_", "widgets": TOL47_WIDGETS,
                        "optional": TOL47_OPTIONAL,
                        "presets": ("v5", "regression"), "needs": ("tol_v5_core.py",), "local_prefix": "tol_v5_",
                        "telemetry": "Metrics_track_v5_{root}_{daydir}.json"},
    "Tol_03_track_v4": {"file": "Tol_03_track_v4.py", "prefix": "TOLV4_", "widgets": TOL03_WIDGETS,
                        "optional": (),
                        "presets": ("t03",), "needs": (), "local_prefix": "tol_v4_",
                        "telemetry": "Metrics_track_v4_{root}_{daydir}.json"},
}
SUPPORT_FILES = ("tol_v5_core.py",)

LAUNCHER_WIDGETS = (   # (name, default)
    ("launch_tag", ""), ("jobs_json", ""), ("sha_pins", ""), ("code_dir", "Code/v5"), ("expect_gpu", ""),
    ("stop_on_fail", "true"), ("retry_policy", "probe_only"), ("allow_unpinned", "false"),
    ("cuda_probe_retries", "3"), ("cuda_probe_sleep_s", "90"), ("stagger_sec", "20"),
    ("status_every_sec", "300"), ("log_sync_every_sec", "900"), ("reset_launcher_roots", "false"),
    ("expect_launcher", ""))      # review r2 finding 3: = LAUNCHER_VERSION of the reviewed launcher (make_v5_payloads)

# ---- jobs ------------------------------------------------------------------------------------------------------------
JOB_KEYS = ("name", "script", "phase", "preset", "set")

def expand_job(script, preset, over):
    env = dict(PRESETS[preset]); env.update(over)
    missing = [k for k in SCRIPTS[script]["widgets"] if k not in env]
    if missing: raise ConfigError(f"missing widgets {missing} (required per job: {REQUIRED_RUN_KEYS})")
    extra = [k for k in env if k not in SCRIPTS[script]["widgets"] + SCRIPTS[script]["optional"]]
    if extra: raise ConfigError(f"widgets {extra} do not exist in {script}")
    for k, v in env.items(): validate_value(k, v)
    pol = policy_problems(script, env)
    if pol: raise ConfigError("; ".join(pol))
    return env

def parse_jobs(text):
    try:
        raw = json.loads(text)
    except Exception as e:
        raise ConfigError(f"jobs_json is not valid JSON: {e}")
    if not isinstance(raw, list) or not raw: raise ConfigError("jobs_json must be a non-empty JSON list")
    jobs, names = [], set()
    for i, j in enumerate(raw):
        if not isinstance(j, dict): raise ConfigError(f"job #{i} is not an object")
        bad = sorted(set(j) - set(JOB_KEYS))
        if bad: raise ConfigError(f"job #{i}: unknown keys {bad} (allowed {JOB_KEYS})")
        name = j.get("name")
        if not isinstance(name, str) or not NAME_RX.match(name): raise ConfigError(f"job #{i}: bad name {name!r}")
        if name in names: raise ConfigError(f"duplicate job name {name!r}")
        script = j.get("script")
        if script not in SCRIPTS: raise ConfigError(f"job {name}: script {script!r} not in {sorted(SCRIPTS)}")
        phase = j.get("phase")
        if not isinstance(phase, int) or isinstance(phase, bool) or phase < 1:
            raise ConfigError(f"job {name}: phase must be an integer >= 1, got {phase!r}")
        preset = j.get("preset")
        if preset not in SCRIPTS[script]["presets"]:
            raise ConfigError(f"job {name}: preset {preset!r} not allowed for {script} {SCRIPTS[script]['presets']}")
        over = j.get("set", {})
        if not isinstance(over, dict): raise ConfigError(f"job {name}: 'set' must be an object")
        for k, v in over.items():
            if k not in SCRIPTS[script]["widgets"] + SCRIPTS[script]["optional"]:
                raise ConfigError(f"job {name}: widget {k!r} does not exist in {script} — refusing (it would be "
                                  f"silently ignored and the worker default used)")
            if not isinstance(v, str): raise ConfigError(f"job {name}: widget {k} must be a string, got {v!r}")
        try:
            env = expand_job(script, preset, over)
        except ConfigError as e:
            raise ConfigError(f"job {name}: {e}")
        jobs.append({"name": name, "script": script, "phase": phase, "preset": preset, "set": dict(over),
                     "env": env, "idx": i})
        names.add(name)
    return jobs

def wsuf(frm, to):
    """Mirror of Tol_03:127 (the SELFTEST checks the line is unchanged in Tol_03)."""
    return f"_w{frm or '0000'}-{to or '2400'}" if (frm or to) else ""

def day_dir(env):
    return env["stage"] + wsuf(env.get("from_hhmm", ""), env.get("to_hhmm", ""))

def local_key(job):
    """Per-stage local scratch dir of the worker: Tol_03:132 'tol_v4_{STAGE}{WSUF}'; Tol_47 'tol_v5_…'."""
    return SCRIPTS[job["script"]]["local_prefix"] + day_dir(job["env"])

def plan_phases(jobs):
    phases = {}
    for j in jobs: phases.setdefault(j["phase"], []).append(j)
    seen_out = {}
    for p in sorted(phases):
        lk = {}
        for j in phases[p]:
            k = local_key(j)
            if k in lk:
                raise ConfigError(f"phase {p}: jobs {lk[k]} and {j['name']} would share the local scratch dir "
                                  f"/local_disk0/tmp/{k} (staged clip, events, obs) — put them in different phases")
            lk[k] = j["name"]
        for j in phases[p]:
            if j["env"].get("mode", "track") != "track": continue     # mode=equiv never writes a tracking root
            ok = (j["env"]["track_root"], day_dir(j["env"]))
            if ok in seen_out:
                raise ConfigError(f"jobs {seen_out[ok]} and {j['name']} write the same output dir "
                                  f"Tol/{ok[0]}/{ok[1]} — one launch must not resume/overwrite its own output")
            seen_out[ok] = j["name"]
    return [(p, phases[p]) for p in sorted(phases)]

def is_e6_verdict(job):
    return job["script"] == "Tol_47_track_v5" and job["env"].get("mode") == "equiv" and \
        "E6" in job["env"].get("equiv_tests", "").split(",")

def e6_order_problems(jobs):
    """Critic finding 10 + launcher review r1 finding 7. An E6 verdict (Tol_47 mode=equiv, equiv_tests has E6) compares
    Tol/<e6_ref_root>/<daydir> with Tol/<e6_reg_root>/<daydir>. (a) Both dirs must be written by jobs of THIS launch in
    an EARLIER phase (a same-phase or later arm would be compared half-written, a missing arm = a stale dir). (b) Every
    job in a phase before the verdict must be an arm of some E6 verdict of the launch, or itself an E6 verdict: with
    stop_on_fail, any other job (e.g. the report-only drift floor) failing first would skip the verdict; a failed
    verdict does not skip later phases (blocking_failures). (c) At most ONE mode=equiv job per phase: Tol_47 names its
    verdict file Metrics_v5_equiv_<GPU>_<YYYYmmdd_HHMMSS>.json (one-second resolution, Tol_47:1856), so two concurrent
    verdict jobs can write the same name and one verdict is lost (equiv_output_unattributed)."""
    p, arms, ver = [], set(), [j for j in jobs if is_e6_verdict(j)]
    by_phase = {}
    for j in jobs:
        if j["env"].get("mode") == "equiv": by_phase.setdefault(j["phase"], []).append(j["name"])
    for ph, names in sorted(by_phase.items()):
        if len(names) > 1:
            p.append(f"phase {ph} runs {len(names)} mode=equiv jobs concurrently {names}: their one-second verdict "
                     f"file names can collide — one per phase (launcher review r1 finding 7)")
    for v in ver:
        e = v["env"]
        if not e.get("e6_ref_root") or not e.get("e6_reg_root"):
            p.append(f"{v['name']}: an E6 verdict needs e6_ref_root and e6_reg_root"); continue
        dd = day_dir(e)
        for side in ("e6_ref_root", "e6_reg_root"):
            w = [j for j in jobs if j["env"].get("mode", "track") == "track" and j["env"]["track_root"] == e[side]
                 and day_dir(j["env"]) == dd]
            if not w:
                p.append(f"{v['name']}: no job of this launch writes Tol/{e[side]}/{dd} ({side})"); continue
            late = [j["name"] for j in w if j["phase"] >= v["phase"]]
            if late:
                p.append(f"{v['name']} (phase {v['phase']}) compares Tol/{e[side]}/{dd}, written by {late} in the "
                         f"same or a later phase")
            arms |= {j["name"] for j in w}
    vnames = {v["name"] for v in ver}
    for v in ver:
        for j in jobs:
            if j["phase"] < v["phase"] and j["name"] not in arms and j["name"] not in vnames:
                p.append(f"{j['name']} (phase {j['phase']}) is not an E6 arm but runs before the E6 verdict "
                         f"{v['name']} (phase {v['phase']}): with stop_on_fail its failure would skip the verdict — "
                         f"put it in a later phase")
    return p

def blocking_failures(failed, jobs):
    """The failures that skip later phases under stop_on_fail. A failed E6 verdict is a read-only comparison of arms
    that are already written: the later verdicts and the floor still run (their evidence is needed to diagnose it), and
    the launch still ends job_failed. A core-sha mismatch always blocks (the next workers import the same core)."""
    by = {j["name"]: j for j in jobs}
    return {n: v for n, v in failed.items()
            if not (n in by and is_e6_verdict(by[n]) and v != "core_sha_mismatch")}

# ---- launcher-only roots are never resumed (launcher review r1 finding 3) --------------------------------------------
ATTIC_DIR = "Launcher_attic"

def launcher_only_roots():
    return tuple(r for r, (_g, m) in ROOT_POLICY.items() if m == "launcher")

def job_outputs(job, tol=None):
    """What a mode=track job writes under Tol: its day dir and its telemetry file."""
    tol = tol or TOL
    e = job["env"]; root, dd = e["track_root"], day_dir(e)
    return {"day": pp.join(tol, root, dd),
            "telemetry": pp.join(tol, SCRIPTS[job["script"]]["telemetry"].format(root=root, daydir=dd))}

DAYDIR_RX = re.compile(r"^\d{8}(_w\d{4}-\d{4})?$")          # <stage><wsuf> (wsuf, Tol_03:127)

def reset_plan(jobs, exists, listdir, stamp, tol=None):
    """(found, moves). Launcher-only roots (E6 arms, sequential arms, drift floors) are never resumed and never mixed.
    Resuming is wrong in every case: Tol_03 / Tol_47 re-seed events.jsonl from the Volume copy and append a second run
    (Tol_03:487-490), .done_ markers skip clips of the old run, and after any Tol_47 change the root's ROOT_CONFIG.json
    raises at start (tol_v5_core.root_config_check).
    `found`, for every launcher-only root a mode=track job of this launch writes (review r2 finding 1):
      - this launch's day dirs and telemetry files that exist;
      - EVERY other child of the root -- its ROOT_CONFIG.json whenever it exists (a stale one blocks the re-run; the
        old rule kept it while another day dir remained, which stranded the two L2s payloads that shared a root), a
        stale day dir of an older payload layout (+ its telemetry, either script's name), a crashed write's temp file.
        The payload checker makes every launcher-only root belong to ONE payload, so nothing of another payload is
        touched, and a reset root never keeps outputs written under a ROOT_CONFIG it no longer holds.
    `moves` = [(src, dst)] into Tol/Launcher_attic/<stamp>/<same relative path> (moved, never deleted)."""
    tol = tol or TOL
    lo = set(launcher_only_roots())
    found, moves, by_root = [], [], {}
    add = lambda p: found.append(p) if p not in found else None
    for j in jobs:
        if j["env"].get("mode", "track") != "track" or j["env"]["track_root"] not in lo: continue
        by_root.setdefault(j["env"]["track_root"], []).append(j)
    for root, js in sorted(by_root.items()):
        mine = {day_dir(j["env"]) for j in js}
        for j in js:
            o = job_outputs(j, tol)
            for k in ("day", "telemetry"):
                if exists(o[k]): add(o[k])
        for n in sorted(listdir(pp.join(tol, root)) or []):
            if n in mine: continue
            add(pp.join(tol, root, n))
            if DAYDIR_RX.match(n):
                for s in SCRIPTS.values():
                    t = pp.join(tol, s["telemetry"].format(root=root, daydir=n))
                    if exists(t): add(t)
    attic = pp.join(tol, ATTIC_DIR, stamp)
    for src in found:
        moves.append((src, pp.join(attic, src[len(tol) + 1:])))
    return found, moves

# ---- what a worker actually ran --------------------------------------------------------
CORE_LINE_RX = re.compile(r"^tol_v5_core (.+?) from (.+) sha256 ([0-9a-f]{64})\s*$")          # Tol_47:117
EQUIV_LINE_RX = re.compile(r"EQUIV (PASS|FAIL) on .* -> (\S.*?Metrics_v5_equiv_[A-Za-z0-9]+_\d{8}_\d{6}\.json)\s*$")

def worker_core_from_log(text):
    """(core path, sha256) of the tol_v5_core the Tol_47 worker imported (its first stdout line after the import)."""
    for line in (text or "").splitlines():
        m = CORE_LINE_RX.match(line.strip())
        if m: return m.group(2), m.group(3)
    return None

def worker_core_problem(text, staged_core_sha):
    """None while unknown/OK, else the reason the job must be stopped (it imported another tol_v5_core)."""
    got = worker_core_from_log(text)
    if got is None or not staged_core_sha: return None
    if got[1] != staged_core_sha:
        return (f"worker imported tol_v5_core from {got[0]} with sha256 {got[1]}, not the staged/pinned "
                f"{staged_core_sha} (a stale /Workspace core shadowed the staged copy)")
    return None

def equiv_source(job_name, log_text, candidates, read_json):
    """Which Metrics_v5_equiv_<GPU>_<ts>.json did this mode=equiv job write? `candidates` = the files that appeared
    during its phase. The worker's final 'EQUIV PASS|FAIL … -> <path>' line names it; `echo.stage_svc_key` (= the job
    name, TOLV5_SVC_KEY) must confirm it, so a same-second name collision with a concurrent job (the name has only
    second resolution) is detected instead of silently attributing the other job's verdict.
    Returns (path or None, how, problems)."""
    norm = lambda p: osp.normpath(p).replace("\\", "/")
    named = None
    for line in reversed((log_text or "").splitlines()):
        m = EQUIV_LINE_RX.search(line.strip())
        if m: named = m.group(2); break
    owned = []
    for c in sorted(candidates):
        try:
            J = read_json(c)
        except Exception:
            continue
        if isinstance(J, dict) and job_name in ((J.get("echo") or {}).get("stage_svc_key"), J.get("job")):
            owned.append(c)                                   # Tol_47 writes both echo.stage_svc_key and job (= SVC_KEY)
    if named is not None:
        hit = [c for c in owned if norm(c) == norm(named)]
        if hit: return hit[0], "log+echo", []
        return None, "log", [f"the log names {named} but no candidate with echo.stage_svc_key == {job_name!r} is that "
                             f"file (overwritten by a concurrent job with the same second-resolution name?)"]
    if len(owned) == 1: return owned[0], "echo", []
    if not owned: return None, None, [f"no new Metrics_v5_equiv file carries echo.stage_svc_key == {job_name!r}"]
    return None, None, [f"{len(owned)} new files carry echo.stage_svc_key == {job_name!r}: {owned}"]

def equiv_copy_name(gpu_cls, tag, job_name):
    """Tagged copy of a verdict file: Tol/Metrics_v5_equiv_<GPU>_<launch_tag>__<job>.json."""
    return f"Metrics_v5_equiv_{gpu_cls}_{tag}__{job_name}.json"

def build_env(base_env, job, svc_dir, code_dir, hf_token):
    """Subprocess env: inherit the notebook env minus any TOL* leftovers, then the job's widgets."""
    env = {k: v for k, v in base_env.items() if not k.startswith(("TOLV4_", "TOLV5_", "TOL_"))}
    pre = SCRIPTS[job["script"]]["prefix"]
    for k, v in job["env"].items(): env[pre + k.upper()] = v
    # Tol_03 reads TOLV4_STAGE_SVC (T03:91); Tol_47 (a copy) may keep that name or rename it -> set both.
    env["TOLV4_STAGE_SVC"] = svc_dir; env["TOLV5_STAGE_SVC"] = svc_dir
    if job["script"] == "Tol_47_track_v5":
        env["TOLV5_SVC_KEY"] = job["name"]                      # Tol_47:148 request key (per-job dirs already isolate)
    env["HF_TOKEN"] = hf_token                                  # T03:118 subprocess login; never printed
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF
    env["PYTHONPATH"] = code_dir + (os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else "")
    return env

def parse_pins(text):
    if not text.strip(): return {}
    try:
        d = json.loads(text)
    except Exception as e:
        raise ConfigError(f"sha_pins is not valid JSON: {e}")
    if not isinstance(d, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in d.items()):
        raise ConfigError("sha_pins must be a JSON object {filename: sha256hex}")
    return d

def needed_files(jobs):
    out = []
    for j in jobs:
        for f in (SCRIPTS[j["script"]]["file"],) + SCRIPTS[j["script"]]["needs"]:
            if f not in out: out.append(f)
    return out

def check_pins(files, pins, allow_unpinned):
    probs = []
    for f in files:
        v = pins.get(f, "")
        if not HEX64.match(v):
            probs.append(f"{f}: pin {v!r} is not a 64-hex sha256")
    if probs and not allow_unpinned:
        raise ConfigError("sha256 pins missing/placeholder (fill with build/make_v5_payloads.py --fill-pins): "
                          + "; ".join(probs))
    return probs

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def source_norm_sha256(data):
    """= tol_v5_core.source_norm_sha256 (the SELFTEST asserts equality on the real core): sha256 of a notebook source
    normalised for transport -- BOM dropped, CRLF/CR -> LF, trailing whitespace stripped, blank lines dropped -- so the
    local .py and the imported workspace notebook (FUSE read or SOURCE export) give ONE value (review r2 finding 3)."""
    t = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
    t = t.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256("\n".join(l.rstrip() for l in t.split("\n") if l.strip()).encode("utf-8")).hexdigest()

def contract_problems(script_or_file, text, support_text=""):
    """Static contract between the launcher and a worker/support file (cheap; run on the staged copy).
    support_text = the staged support module(s) the worker imports (tol_v5_core.py): the widget table and the env
    prefix may legitimately live there (core.WIDGETS / core.ENV_PREFIX); the staging protocol and the HF login must be
    in the worker itself."""
    p = []
    if script_or_file == "tol_v5_core.py":
        if "def selftest" not in text: p.append("tol_v5_core.py has no selftest()")
        return p
    s = SCRIPTS[script_or_file]
    both = text + "\n" + support_text
    if not text.startswith("# Databricks notebook source"): p.append("not a Databricks source-format notebook")
    if f'"{s["prefix"]}"' not in both and f"'{s['prefix']}'" not in both:
        p.append(f"never reads the {s['prefix']}<NAME> env prefix — every widget would take its default")
    if "STAGE_SVC" not in text or not all(t in text for t in ("req__", "ready__", "error__")):
        p.append("no staging-service protocol (req__/ready__/error__ + STAGE_SVC) — the worker would fall back to "
                 "shutil over /dbfs FUSE (~0.5 MB/s, 43 min per clip)")
    if not ("TOLV4_STAGE_SVC" in text or "TOLV5_STAGE_SVC" in text):
        p.append("reads neither TOLV4_STAGE_SVC nor TOLV5_STAGE_SVC")
    if 'os.environ["HF_TOKEN"]' not in text and "os.environ['HF_TOKEN']" not in text and \
            'os.environ.get("HF_TOKEN"' not in text:
        p.append("does not read HF_TOKEN in subprocess mode")
    missing = [w for w in s["widgets"] if f'"{w}"' not in both and f"'{w}'" not in both]
    if missing: p.append(f"widget names never appear as string literals (worker or its core): {missing}")
    if script_or_file == "Tol_47_track_v5":
        if "tol_v5_core" not in text: p.append("does not import tol_v5_core")
        # the launcher's run-time checks parse these three outputs (findings 3 / 12): a format change must fail
        # here, at staging, not silently disable the checks
        for what, rx in (("core import line", r'print\(f"tol_v5_core \{[^}]+\} from \{core\.__file__\} sha256 \{CORE_SHA\}"\)'),
                         ("EQUIV verdict line", r'EQUIV \{[^}]*PASS[^}]*FAIL[^}]*\} on \{GPU_NAME\}: .*-> \{EQ_PATH\}'),
                         ("echo.stage_svc_key", r'"stage_svc_key":\s*SVC_KEY')):
            if not re.search(rx, text, re.S):
                p.append(f"{what} not found in the expected format — the launcher's worker-core / verdict "
                         f"attribution checks rely on it")
    return p

# ---- staging service ---------------------------------------------------------------------------------------------------
def _answer(svc_dir, kind, stage, vname, text):
    tmp = osp.join(svc_dir, f"ans_tmp__{kind}__{stage}__{vname}")     # never matches the req__* glob
    with open(tmp, "w") as f: f.write(text)
    os.replace(tmp, osp.join(svc_dir, f"{kind}__{stage}__{vname}"))   # atomic: a worker never sees half an answer

def serve_requests(svc_dirs, alive, copy_fn, dst_prefix, log, records):
    """Answer every pending request. Protocol = Tol_03:925-957 (req__{STAGE}__{vname} holds the local dst path;
    answer ready__/error__ with the same suffix). One directory per job, so requests never collide."""
    n = 0
    for job, d in svc_dirs.items():
        for req in sorted(glob.glob(osp.join(d, "req__*"))):
            name = osp.basename(req)[len("req__"):]
            stage, sep, vname = name.partition("__")
            try:
                with open(req) as f: dst = f.read().strip()
                os.remove(req)
            except OSError:
                continue                                       # being replaced right now; next scan
            if not alive(job):
                log(f"[svc] dropping request from finished job {job}: {vname}"); continue
            err, t = None, time.time()
            parts = dst.replace("\\", "/").split("/")
            if not sep or not CLIP_RX.match(vname):
                err = f"refused: {name!r} is not a Tol clip request"
            elif not dst.startswith(dst_prefix) or ".." in parts:
                err = f"refused: destination {dst!r} is outside {dst_prefix}"
            else:
                try:
                    copy_fn(vname, dst)
                    if not osp.exists(dst) or osp.getsize(dst) == 0: raise RuntimeError("copy produced no/empty file")
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"[:500]
            if err is None:
                mb, s = osp.getsize(dst) / 1e6, time.time() - t
                _answer(d, "ready", stage, vname, dst)
                records.append({"job": job, "clip": vname, "mb": round(mb, 1), "s": round(s, 1)})
                log(f"[svc] staged {vname} for {job}: {mb:.0f} MB in {s:.0f}s ({mb / max(s, 1e-9):.1f} MB/s)")
            else:
                _answer(d, "error", stage, vname, err)
                records.append({"job": job, "clip": vname, "error": err})
                log(f"[svc] FAILED staging {vname} for {job}: {err}")
            n += 1
    return n

# ---- running a phase ---------------------------------------------------------------------------------------------------
def run_phase(jobs, spawn, serve, sample=None, status=None, sync=None, *, stagger_s=20.0, poll_s=5.0,
              status_every=300.0, sync_every=900.0, clock=time.time, sleep=time.sleep):
    """Spawn the phase's jobs (staggered, serving staging requests meanwhile) and wait for all of them.
    Returns {name: {"proc", "t0", "t1", "rc", ...}}. Any launcher-side exception terminates the children."""
    pending, running = list(jobs), {}
    last_spawn, last_status, last_sync = -1e18, clock(), clock()
    def alive(name):
        r = running.get(name)
        return r is not None and r["proc"].poll() is None
    try:
        while pending or any(r["proc"].poll() is None for r in running.values()):
            if pending and clock() - last_spawn >= stagger_s:
                j = pending.pop(0); running[j["name"]] = spawn(j); last_spawn = clock()
            serve(alive)
            if sample: sample(running)
            if status and clock() - last_status >= status_every: status(running); last_status = clock()
            if sync and clock() - last_sync >= sync_every: sync(running); last_sync = clock()
            sleep(poll_s)
        serve(alive)                                            # drop anything a finished worker left behind
    except BaseException:
        for r in running.values():
            if r["proc"].poll() is None: r["proc"].terminate()
        t_end = time.time() + 60
        for r in running.values():
            try:
                r["proc"].wait(timeout=max(1.0, t_end - time.time()))
            except Exception:
                r["proc"].kill()
        raise
    for r in running.values():
        r["rc"] = r["proc"].wait(); r["t1"] = clock()
        if r.get("logf"): r["logf"].close()
    return running

# ---- CUDA / environment probe (runs in a SUBPROCESS: the launcher never creates a CUDA context) ----------------------
PROBE_SRC = r'''
import os, json, time
from importlib import metadata as md
out = {"ok": False, "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
def _v(dist):
    try: return md.version(dist)
    except Exception: return None
out["dist"] = {d: _v(d) for d in ("torch", "transformers", "rfdetr", "supervision", "opencv-python-headless",
               "opencv-python", "opencv-contrib-python", "timm", "numpy", "Pillow", "accelerate", "scipy", "peft",
               "pycocotools", "huggingface_hub")}
rt = {}
for mod in ("numpy", "PIL", "cv2", "transformers", "timm", "scipy"):
    try:
        m = __import__(mod); rt[mod] = getattr(m, "__version__", None)
        if mod == "cv2": rt["cv2_file"] = getattr(m, "__file__", None)
    except Exception as e:
        rt[mod] = "IMPORT-ERROR " + repr(e)[:200]
out["runtime"] = rt
t = time.time()
try:
    import torch
    rt["torch"] = torch.__version__; out["torch_cuda"] = torch.version.cuda
    x = torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    p = torch.cuda.get_device_properties(0); free, total = torch.cuda.mem_get_info()
    out.update(ok=True, name=torch.cuda.get_device_name(0), cc=[p.major, p.minor],
               total_gib=round(total / 2**30, 2), free_gib=round(free / 2**30, 2), init_s=round(time.time() - t, 2))
except Exception as e:
    out["cuda_error"] = (type(e).__name__ + ": " + str(e))[:600]
print("PROBE_JSON " + json.dumps(out), flush=True)
'''

def parse_probe(stdout):
    for line in reversed((stdout or "").splitlines()):
        if line.startswith("PROBE_JSON "):
            try:
                return json.loads(line[len("PROBE_JSON "):])
            except Exception:
                return None
    return None

def _vtuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", (v or "").split("+")[0])[:3])

def env_problems(p):
    """Deterministic environment mismatches (never retried). Only judged when the probe got far enough to report."""
    if not p or "runtime" not in p: return []
    rt, dist, out = p["runtime"], p.get("dist", {}), []
    if p.get("alloc_conf") != ALLOC_CONF:
        out.append(f"PYTORCH_CUDA_ALLOC_CONF={p.get('alloc_conf')!r}, expected {ALLOC_CONF!r}.")
    if rt.get("transformers") != "5.2.0": out.append(f"transformers {rt.get('transformers')} != 5.2.0")
    if dist.get("rfdetr") != "1.10.1": out.append(f"rfdetr {dist.get('rfdetr')} != 1.10.1")
    if not str(rt.get("cv2") or "").startswith("4.11.0"):
        out.append(f"cv2 {rt.get('cv2')} is not 4.11.0 (GUI OpenCV from supervision would shadow the headless build)")
    if not _vtuple(rt.get("numpy")) or _vtuple(rt.get("numpy"))[0] >= 2: out.append(f"numpy {rt.get('numpy')} is not <2")
    if rt.get("PIL") != "11.1.0": out.append(f"Pillow {rt.get('PIL')} != 11.1.0")
    tv = _vtuple(rt.get("torch"))
    if tv and tv < (2, 5): out.append(f"torch {rt.get('torch')} < 2.5 (rfdetr needs torch.accelerator)")
    return out

def gpu_class(name):
    n = (name or "").upper()
    if "A100" in n: return "A100"
    if re.search(r"A10(?!0)", n): return "A10"
    return "other"

def smi_lines(text):
    """Log only the driver and vGPU-licence lines of nvidia-smi -q."""
    return [l.strip() for l in (text or "").splitlines() if re.search(r"Driver Version|CUDA Version|Licen", l)]

def run_cuda_probe(run_cmd, retries, sleep_s, log, sleeper=time.sleep):
    """run_cmd() -> (rc, stdout, stderr). Retries CUDA failures `retries` times; environment mismatches raise at once."""
    tries = []
    for k in range(retries + 1):
        try:
            rc, out, err = run_cmd()
        except Exception as e:
            rc, out, err = -1, "", f"{type(e).__name__}: {e}"
        p = parse_probe(out) or {"ok": False, "cuda_error": f"no PROBE_JSON (rc={rc}): {(err or '')[-400:]}"}
        tries.append({"attempt": k + 1, "ok": bool(p.get("ok")), "error": p.get("cuda_error")})
        probs = env_problems(p)
        if probs:
            raise ConfigError("environment does not match the v5 pins (deterministic, not retried): " + "; ".join(probs))
        if p.get("ok"):
            return p, tries
        log(f"⚠ CUDA probe attempt {k + 1}/{retries + 1} failed: {p.get('cuda_error')}")
        if k < retries: sleeper(sleep_s)
    raise CudaProbeError(f"CUDA unusable after {retries + 1} probe attempts: {tries[-1]['error']}")

def attempt_gate(prev, cluster_id, parent_run, retry_policy):
    """None if this attempt may proceed, else the reason. prev = attempts.json entries of this launch_tag."""
    same = [a for a in prev if parent_run and a.get("parent_run") == parent_run]
    for a in same:
        if cluster_id and a.get("cluster_id") == cluster_id:
            return (f"retry of run {parent_run} landed on the SAME cluster {cluster_id} as attempt {a.get('attempt')} "
                    f"(outcome {a.get('outcome')}) — a retry must land on a new cluster")
        if retry_policy == "probe_only" and a.get("outcome") in ("job_failed", "config_error"):
            return (f"attempt {a.get('attempt')} of run {parent_run} ended {a.get('outcome')!r} behind a healthy "
                    f"CUDA probe — presumed deterministic, not retried (retry_policy=probe_only). "
                    f"Fix the cause and resubmit.")
    return None

def parse_launcher_widgets(W):
    tag = W.get("launch_tag", "")
    if not NAME_RX.match(tag): raise ConfigError(f"launch_tag {tag!r} must match {NAME_RX.pattern}")
    el = W.get("expect_launcher", "")
    if el and el != LAUNCHER_VERSION:        # review r2 finding 3: the workspace notebook is not the reviewed launcher
        raise ConfigError(f"expect_launcher {el!r} but this notebook is {LAUNCHER_VERSION!r}: the deployed launcher is "
                          f"not the one the payloads were generated for — re-import the reviewed Tol_47b_launcher.py "
                          f"or regenerate the payloads")
    jobs = parse_jobs(W.get("jobs_json", ""))
    phases = plan_phases(jobs)
    op = e6_order_problems(jobs)
    if op: raise ConfigError("E6 phase order: " + "; ".join(op))
    pins = parse_pins(W.get("sha_pins", ""))
    code_dir = W.get("code_dir", "Code/v5").strip().strip("/")
    if not code_dir or ".." in code_dir.split("/"): raise ConfigError(f"code_dir {code_dir!r} invalid")
    eg = W.get("expect_gpu", "")
    if eg not in ("", "A100", "A10"): raise ConfigError(f"expect_gpu {eg!r} not in ('', 'A100', 'A10')")
    cfg = {"launch_tag": tag, "jobs": jobs, "phases": phases, "pins": pins, "code_dir": code_dir, "expect_gpu": eg,
           "expect_launcher": el}
    for k in ("stop_on_fail", "allow_unpinned", "reset_launcher_roots"):
        if W.get(k, "") not in _BOOL: raise ConfigError(f"{k} must be true|false")
        cfg[k] = W[k] == "true"
    if W.get("retry_policy", "") not in ("probe_only", "always"):
        raise ConfigError("retry_policy must be probe_only|always")
    cfg["retry_policy"] = W["retry_policy"]
    for k, cast in (("cuda_probe_retries", int), ("cuda_probe_sleep_s", float), ("stagger_sec", float),
                    ("status_every_sec", float), ("log_sync_every_sec", float)):
        try:
            cfg[k] = cast(W.get(k, ""))
        except ValueError:
            raise ConfigError(f"{k}={W.get(k)!r} is not a number")
        if cfg[k] < 0: raise ConfigError(f"{k} must be >= 0")
    cfg["files"] = needed_files(jobs)
    cfg["pin_problems"] = check_pins(cfg["files"], pins, cfg["allow_unpinned"])
    return cfg

# ---- payload checker (local, used by the reviewer / orchestrator before any submit) ---------------------------------
NODE_GPU = {"Standard_NC24ads_A100_v4": "A100", "Standard_NV36ads_A10_v5": "A10",
            "Standard_D16ds_v5": None, "Standard_D8ds_v5": None}     # verified with `databricks clusters list-node-types`
DOWNSTREAM_WIDGETS = {    # verified against the notebooks 2026-09-24; re-parsed from the files when --notebooks is given
    "Tol_32_dense_crops": ("day", "track_root", "crop_every_sec", "out_subdir", "max_clips", "selfcheck",
                           "tracker_crop_sec", "anchors_only"),
    "Tol_28_build_bands": ("day", "track_root", "crops_subdir", "out_name"),
    "Tol_17_overlay_video": ("day", "track_root", "out_w", "sample_sec", "out_fps", "max_clips", "ident_map",
                             "from_hhmm", "to_hhmm", "relabel_json", "out_name"),
}
NOTEBOOK_GPU = {"Tol_47_track_v5": True, "Tol_47b_launcher": True, "Tol_28_build_bands": True,
                "Tol_32_dense_crops": False, "Tol_17_overlay_video": False}
# tracking roots (+ the launcher-only equivalence/sequential roots): (GPU class, run mode)
ROOT_POLICY = {"Tracking_v5a10": ("A10", "direct"), "Tracking_v5a10_pilot": ("A10", "direct"),
               "Tracking_v5a10_nodark": ("A10", "direct"), "Tracking_v5a100cap21": ("A100", "direct"),
               "Tracking_v5eq_ref": ("A100", "launcher"), "Tracking_v5eq_reg": ("A100", "launcher"),
               # the conditional sequential re-runs (L2s): one root pair PER PAYLOAD (review r2 finding 1 -- a shared
               # launcher-only root made a reset of one payload strand the other payload's ROOT_CONFIG)
               "Tracking_v5eq_ref_seq_smoke": ("A100", "launcher"), "Tracking_v5eq_reg_seq_smoke": ("A100", "launcher"),
               "Tracking_v5eq_ref_seq_window": ("A100", "launcher"), "Tracking_v5eq_reg_seq_window": ("A100", "launcher"),
               "Tracking_v5floor_a100": ("A100", "launcher"), "Tracking_v5floor_a10": ("A10", "launcher"),
               "Tracking_v5equiv_a100": ("A100", "launcher"), "Tracking_v5equiv_a10": ("A10", "launcher")}
PRODUCTION_ROOTS = ("Tracking_v5a10", "Tracking_v5a100cap21")
PARAM_MAX_CHARS = 2000        # conservative per-value ceiling for notebook base_parameters
# Change control for DIRECT roots (launcher review r1 finding 3). Tol_47 publishes ROOT_CONFIG.json write-if-absent and
# raises on any sha/flag difference, so a direct root can never be re-tracked with changed code, and moving or deleting
# production outputs is not an option (the CLI has no mv; the outputs may already have been reviewed). A change after a direct root
# was tracked therefore goes into a NEW REVISION of that root, `<root>_r<N>` (N >= 2, `make_v5_payloads.py --root-rev N`),
# which inherits the base root's policy (GPU class, run mode, production rules, relabel dir `relabel_<slug>_r<N>`).
# Launcher-only roots take no revisions: they are reset in place (reset_launcher_roots).
ROOT_REV_RX = re.compile(r"^(Tracking_[A-Za-z0-9_]+?)_r([2-9]|[1-9][0-9]+)$")

def root_base(root):
    """Tracking_v5a10_r2 -> Tracking_v5a10 (a revision of a DIRECT root); any other name is returned unchanged."""
    m = ROOT_REV_RX.match(root or "")
    if m and m.group(1) in ROOT_POLICY and ROOT_POLICY[m.group(1)][1] == "direct": return m.group(1)
    return root
# Production identity relabels: written ONLY by the GT-free Tol_30 week-mode step
# (build/prod_relabel_v5.py, uploads) and read ONLY by the Tol_17 watch renders. One definition for both sides.
RELABEL_DIR_FMT = "Metadata/relabel_{slug}"

def root_slug(root):
    """Tracking_v5a10 -> v5a10, Tracking_v5a100cap21 -> v5a100cap21 (render names and relabel dirs)."""
    if not ROOT_RX.match(root or ""): raise ConfigError(f"root {root!r} must match {ROOT_RX.pattern}")
    return root[len("Tracking_"):]

def relabel_path(root, day):
    return pp.join(TOL, RELABEL_DIR_FMT.format(slug=root_slug(root)), f"relabel_week_{day}.json")

# clips per recorded day (Tol_ch04 strays and the "(1)" duplicate excluded)
EXPECTED_CLIPS = {"20240422": 20, "20240423": 33, "20240424": 33, "20240425": 32, "20240426": 32, "20240427": 33,
                  "20240428": 34}

def core_problems(core_path):
    """Cross-check the launcher registry/presets against tol_v5_core.py's WIDGETS / REGRESSION_FLAGS / ENV_PREFIX
    (static AST read: the core is not imported). Returns a list of disagreements."""
    import ast
    with open(core_path, encoding="utf-8") as f: tree = ast.parse(f.read())
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ("WIDGETS", "REGRESSION_FLAGS", "ENV_PREFIX"):
            try:
                vals[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    p = []
    W_ = vals.get("WIDGETS")
    if not isinstance(W_, dict): return ["tol_v5_core.py: no literal WIDGETS dict found"]
    if set(W_) != set(TOL47_WIDGETS) | set(TOL47_OPTIONAL):
        p.append(f"core WIDGETS names differ: core-only {sorted(set(W_) - set(TOL47_WIDGETS) - set(TOL47_OPTIONAL))}, "
                 f"launcher-only {sorted((set(TOL47_WIDGETS) | set(TOL47_OPTIONAL)) - set(W_))}")
    for k in TOL47_WIDGETS:
        if k in W_ and k not in REQUIRED_RUN_KEYS and str(W_[k][0]) != PRESETS["v5"][k]:
            p.append(f"v5 default of {k}: core {W_[k][0]!r} != launcher preset {PRESETS['v5'][k]!r}")
    for k, v in (vals.get("REGRESSION_FLAGS") or {}).items():
        if PRESETS["regression"].get(k) != v:
            p.append(f"regression flag {k}: core {v!r} != launcher preset {PRESETS['regression'].get(k)!r}")
    if vals.get("ENV_PREFIX", "TOLV5_") != "TOLV5_": p.append(f"core ENV_PREFIX {vals.get('ENV_PREFIX')!r}")
    return p

def _widgets_in_file(path):
    with open(path, encoding="utf-8") as f: txt = f.read()
    return tuple(dict.fromkeys(re.findall(r'dbutils\.widgets\.text\(\s*"([^"]+)"', txt)))

def check_payload_dir(dirpath, strict=False, notebooks=None, tol47=None, core=None):
    """Returns (errors, warnings, report_lines)."""
    errs, warns, rep = [], [], []
    if core and osp.exists(core):
        cp0 = core_problems(core)
        errs += [f"tol_v5_core cross-check: {x}" for x in cp0]
        rep.append(f"tol_v5_core cross-check ({osp.basename(core)}, sha256 {sha256_file(core)[:16]}…): "
                   f"{'OK — WIDGETS names/defaults and REGRESSION_FLAGS match the launcher presets' if not cp0 else str(len(cp0)) + ' problem(s)'}")
    elif core:
        warns.append(f"tol_v5_core file {core} not found — presets not cross-checked")
    dw = dict(DOWNSTREAM_WIDGETS)
    if notebooks:
        for nb in dw:
            fp = osp.join(notebooks, nb + ".py")
            if osp.exists(fp):
                got = _widgets_in_file(fp)
                if set(got) != set(dw[nb]):
                    errs.append(f"{nb}.py widgets {got} differ from the registry {dw[nb]} — update DOWNSTREAM_WIDGETS")
                dw[nb] = got
    tracks, run_names, tags, produced, equiv_names = [], {}, {}, set(), {}
    files = sorted(f for f in os.listdir(dirpath) if f.endswith(".json"))
    if not files: errs.append(f"no *.json payloads in {dirpath}")
    for fn in files:
        fp = osp.join(dirpath, fn)
        raw = open(fp, "rb").read()
        if raw.startswith(b"\xef\xbb\xbf"): errs.append(f"{fn}: UTF-8 BOM"); continue
        try:
            P_ = json.loads(raw.decode("utf-8"))
        except Exception as e:
            errs.append(f"{fn}: invalid JSON/UTF-8: {e}"); continue
        rn = P_.get("run_name")
        if not isinstance(rn, str) or not rn: errs.append(f"{fn}: run_name missing")
        elif rn in run_names: errs.append(f"{fn}: run_name {rn!r} duplicates {run_names[rn]}")
        else: run_names[rn] = fn
        unknown_top = set(P_) - {"run_name", "tasks", "timeout_seconds"}
        if unknown_top: errs.append(f"{fn}: unknown top-level keys {sorted(unknown_top)}")
        tasks = P_.get("tasks") or []
        if not tasks: errs.append(f"{fn}: no tasks")
        keys = [t.get("task_key") for t in tasks]
        if len(set(keys)) != len(keys): errs.append(f"{fn}: duplicate task_key")
        for t in tasks:
            tk = f"{fn}:{t.get('task_key')}"
            if "existing_cluster_id" in t or "job_cluster_key" in t or "new_cluster" not in t:
                errs.append(f"{tk}: must own its node (inline new_cluster only; shared clusters expose no GPU)"); continue
            c = t["new_cluster"]
            node = c.get("node_type_id")
            if node not in NODE_GPU: errs.append(f"{tk}: node_type_id {node!r} not in the verified list {sorted(NODE_GPU)}")
            if c.get("num_workers") != 0: errs.append(f"{tk}: num_workers must be 0")
            sc = c.get("spark_conf") or {}
            if sc.get("spark.databricks.cluster.profile") != "singleNode" or sc.get("spark.master") != "local[*]":
                errs.append(f"{tk}: not a single-node spark_conf")
            if (c.get("custom_tags") or {}).get("ResourceClass") != "SingleNode": errs.append(f"{tk}: ResourceClass tag")
            if c.get("data_security_mode") != "SINGLE_USER" or c.get("single_user_name") != "<DATABRICKS_USER>":
                errs.append(f"{tk}: data_security_mode/single_user_name")
            if (c.get("azure_attributes") or {}).get("availability") != "ON_DEMAND_AZURE":
                errs.append(f"{tk}: availability must be ON_DEMAND_AZURE (no spot eviction on multi-hour runs)")
            if (c.get("spark_env_vars") or {}).get("PYTORCH_CUDA_ALLOC_CONF") != ALLOC_CONF:
                errs.append(f"{tk}: spark_env_vars.PYTORCH_CUDA_ALLOC_CONF must be {ALLOC_CONF}")
            to = t.get("timeout_seconds")
            if not isinstance(to, int) or to <= 0: errs.append(f"{tk}: timeout_seconds must be a positive int")
            mr = t.get("max_retries", 0)
            if mr not in (0, 1): errs.append(f"{tk}: max_retries {mr} not in (0, 1)")
            if t.get("retry_on_timeout") not in (None, False): errs.append(f"{tk}: retry_on_timeout must be false")
            for dep in t.get("depends_on") or []:
                if dep.get("task_key") not in keys: errs.append(f"{tk}: depends_on unknown {dep}")
            nt = t.get("notebook_task") or {}
            path = nt.get("notebook_path", "")
            if nt.get("source") != "WORKSPACE" or not path.startswith(NB_DIR + "/"):
                errs.append(f"{tk}: notebook_path must be under {NB_DIR}/ with source WORKSPACE"); continue
            nb = path[len(NB_DIR) + 1:]
            bp = nt.get("base_parameters") or {}
            for k, v in bp.items():
                if not isinstance(v, str): errs.append(f"{tk}: base_parameters.{k} is not a string")
                elif len(v) > PARAM_MAX_CHARS: errs.append(f"{tk}: base_parameters.{k} is {len(v)} chars (> {PARAM_MAX_CHARS})")
            gpu = NODE_GPU.get(node)
            if nb not in NOTEBOOK_GPU: errs.append(f"{tk}: notebook {nb!r} is not one the v5 plan runs"); continue
            if NOTEBOOK_GPU[nb] != bool(gpu): errs.append(f"{tk}: {nb} on {'GPU' if gpu else 'CPU'} node {node}")
            sv = c.get("spark_version", "")
            if gpu and sv != "17.3.x-gpu-ml-scala2.13": errs.append(f"{tk}: GPU tasks use 17.3.x-gpu-ml-scala2.13")
            if not gpu and not sv.startswith("17.3.x-") or (not gpu and "gpu" in sv):
                errs.append(f"{tk}: CPU task runtime {sv!r}")
            if nb == "Tol_47_track_v5":
                unknown = [k for k in bp if k not in TOL47_WIDGETS + TOL47_OPTIONAL]
                if unknown: errs.append(f"{tk}: widgets not in the Tol_47 widget list: {unknown}")
                try:
                    env = expand_job("Tol_47_track_v5", "v5",
                                     {k: v for k, v in bp.items() if k in TOL47_WIDGETS + TOL47_OPTIONAL})
                except ConfigError as e:
                    errs.append(f"{tk}: {e}"); continue
                if set(TOL47_WIDGETS) - set(bp):
                    errs.append(f"{tk}: direct Tol_47 tasks must set EVERY Tol_47 widget explicitly; missing "
                                f"{sorted(set(TOL47_WIDGETS) - set(bp))}")
                tracks.append({"file": fn, "task": tk, "script": nb, "mode": "direct", "gpu": gpu, "env": env,
                               "timeout": to, "retries": mr, "preset_diff": sorted(
                                   k for k in env if k not in REQUIRED_RUN_KEYS + ROOT_EXEMPT + ("mem_cap_gib",)
                                   and env[k] != PRESETS["v5"].get(k))})
            elif nb == "Tol_47b_launcher":
                unknown = [k for k in bp if k not in dict(LAUNCHER_WIDGETS)]
                if unknown: errs.append(f"{tk}: launcher widgets unknown: {unknown}")
                Wd = dict(LAUNCHER_WIDGETS); Wd.update(bp)
                try:   # structure is always checked; --strict turns placeholder pins into errors (pre-submit)
                    cfg = parse_launcher_widgets({**Wd, "allow_unpinned": "true"})
                except ConfigError as e:
                    errs.append(f"{tk}: {e}"); continue
                if cfg["pin_problems"]:
                    (errs if strict else warns).append(f"{tk}: unpinned (run make_v5_payloads.py --fill-pins "
                                                       f"before submit): {cfg['pin_problems']}")
                if bp.get("allow_unpinned", "false") != "false": errs.append(f"{tk}: allow_unpinned must be false")
                if cfg["expect_launcher"] != LAUNCHER_VERSION:      # non-empty mismatch already raised in parse
                    errs.append(f"{tk}: expect_launcher {cfg['expect_launcher']!r} must be {LAUNCHER_VERSION!r} — "
                                f"regenerate with make_v5_payloads.py (the launcher refuses a payload generated for "
                                f"another revision; review r2 finding 3)")
                if cfg["reset_launcher_roots"]:
                    warns.append(f"{tk}: reset_launcher_roots=true — existing outputs of its launcher-only roots are MOVED "
                                 f"to Tol/{ATTIC_DIR}/ before phase 1 (a re-run: root reset)")
                t03pin = cfg["pins"].get("Tol_03_track_v4.py")
                if t03pin is not None and t03pin != T03_SHA256:
                    errs.append(f"{tk}: Tol_03 pin {t03pin} != the byte-identical T03 sha {T03_SHA256}")
                if cfg["launch_tag"] in tags: errs.append(f"{tk}: launch_tag duplicates {tags[cfg['launch_tag']]}")
                tags[cfg["launch_tag"]] = tk
                if cfg["expect_gpu"] != (gpu or ""):
                    errs.append(f"{tk}: expect_gpu {cfg['expect_gpu']!r} but node {node} is {gpu}")
                for j in cfg["jobs"]:
                    if j["env"].get("mode") == "equiv":     # verdict files are attributed by echo.stage_svc_key = name
                        if j["name"] in equiv_names:
                            errs.append(f"{tk}: equiv job name {j['name']!r} also used by {equiv_names[j['name']]} — "
                                        f"names must be unique across payloads (verdict attribution)")
                        equiv_names[j["name"]] = tk
                    tracks.append({"file": fn, "task": f"{tk}/{j['name']}", "script": j["script"], "mode": "launcher",
                                   "gpu": gpu, "env": j["env"], "timeout": to, "retries": mr, "preset": j["preset"],
                                   "preset_diff": []})
                rep.append(f"{tk}: launcher '{cfg['launch_tag']}' phases " + " -> ".join(
                    "[" + " ‖ ".join(f"{j['name']}({j['script'].split('_')[1]},{j['preset']},{day_dir(j['env'])}"
                                     f"@{j['env']['track_root']})" for j in js) + "]" for _, js in cfg["phases"]))
            else:
                exp = dw[nb]
                unknown = [k for k in bp if k not in exp]
                if unknown: errs.append(f"{tk}: widgets {unknown} not in {nb}")
                if not ROOT_RX.match(bp.get("track_root", "")) or not bp.get("day"):
                    errs.append(f"{tk}: {nb} needs day + track_root")
                else:
                    produced.add((nb, bp["track_root"], bp["day"], tk))
                    rj = bp.get("relabel_json", "")
                    if nb == "Tol_17_overlay_video" and root_base(bp["track_root"]) in PRODUCTION_ROOTS:
                        want = relabel_path(bp["track_root"], bp["day"])
                        if rj != want:     # the GT-free prod_relabel_v5.py step uploads exactly there (finding 7)
                            errs.append(f"{tk}: relabel_json {rj!r} != {want!r} (the only path the GT-free production "
                                        f"Tol_30 week-mode step writes)")
    # ---- cross-payload rules ----
    by_root, revs = {}, {}
    for tr in tracks:
        e, tk = tr["env"], tr["task"]
        root = e["track_root"]
        base = root_base(root)                   # a change-control revision <root>_r<N> has its base root's policy
        if base not in ROOT_POLICY:
            errs.append(f"{tk}: track_root {root} not in the spec's root list {sorted(ROOT_POLICY)} (revisions "
                        f"<direct root>_r<N>, N >= 2, only)"); continue
        revs.setdefault(base, {}).setdefault(root, []).append(tk)
        g, m = ROOT_POLICY[base]
        if tr["gpu"] != g: errs.append(f"{tk}: root {root} is {g}-only but runs on {tr['gpu']}")
        if tr["mode"] != m: errs.append(f"{tk}: root {root} is {m}-only but runs {tr['mode']}")
        cap = e.get("mem_cap_gib")
        if tr["script"] == "Tol_47_track_v5":
            if (base == "Tracking_v5a100cap21") != (cap == "21.0"):
                errs.append(f"{tk}: mem_cap_gib={cap} (21.0 only and always in Tracking_v5a100cap21)")
        if base in PRODUCTION_ROOTS:
            if tr["preset_diff"]: errs.append(f"{tk}: production root differs from the v5 preset in {tr['preset_diff']}")
            if any(e[k] not in ("", "0") for k in ROOT_EXEMPT[1:]): errs.append(f"{tk}: production runs are full days")
            if tr["timeout"] != 259200 or tr["retries"] != 1:
                errs.append(f"{tk}: production day tasks need timeout 259200 (72 h) and max_retries 1")
        if base == "Tracking_v5a10_pilot" and tr["preset_diff"]:
            errs.append(f"{tk}: pilot windows must run the v5 preset, differs in {tr['preset_diff']}")
        if base == "Tracking_v5a10_nodark" and tr["preset_diff"] != ["dark_fire", "detect_night_sec"]:
            errs.append(f"{tk}: the nodark arm differs from v5 only in dark_fire/detect_night_sec, got {tr['preset_diff']}")
        by_root.setdefault(root, []).append(tr)
    for base, rr in sorted(revs.items()):        # one revision per base root across the whole payload set
        if len(rr) > 1:
            errs.append(f"payloads mix revisions {sorted(rr)} of {base}: every payload of a chain must use ONE root "
                        f"(regenerate all payloads with the same make_v5_payloads.py --root-rev)")
    seen_dirs = {}
    for root, trs in by_root.items():
        for tr in trs:
            if tr["env"].get("mode", "track") != "track": continue
            d = (root, day_dir(tr["env"]))
            if d in seen_dirs: errs.append(f"{tr['task']}: output dir Tol/{d[0]}/{d[1]} also written by {seen_dirs[d]}")
            seen_dirs[d] = tr["task"]
        lfiles = sorted({tr["file"] for tr in trs if tr["mode"] == "launcher" and tr["env"].get("mode", "track") == "track"})
        if len(lfiles) > 1:        # review r2 finding 1: a reset of one payload would strand the other's ROOT_CONFIG
            errs.append(f"launcher-only root {root} is written by {len(lfiles)} payloads {lfiles}: one payload per "
                        f"launcher-only root (a reset moves the whole root)")
        cfgs = {}
        for tr in trs:
            key = json.dumps({k: v for k, v in sorted(tr["env"].items()) if k not in ROOT_EXEMPT + TOL47_OPTIONAL},
                             sort_keys=True)
            cfgs.setdefault((tr["script"], key), []).append(tr["task"])
        if len({s for s, _ in cfgs}) > 1:
            errs.append(f"root {root}: mixes scripts {sorted({s for s, _ in cfgs})}")
        if len(cfgs) > 1:
            errs.append(f"root {root}: {len(cfgs)} different configurations (ROOT_CONFIG would raise): "
                        + " | ".join(", ".join(v) for v in cfgs.values()))
    track_days = {(tr["env"]["track_root"], day_dir(tr["env"])) for tr in tracks if tr["env"].get("mode") == "track"}
    for nb, root, day, tk in sorted(produced):
        if (root, day) not in track_days:
            errs.append(f"{tk}: {nb} reads Tol/{root}/{day} but no tracking payload produces it")
    if tol47 and osp.exists(tol47):
        with open(tol47, encoding="utf-8") as f: txt = f.read()
        ctxt = open(core, encoding="utf-8").read() if (core and osp.exists(core)) else ""
        cp = contract_problems("Tol_47_track_v5", txt, ctxt)
        for p in cp: errs.append(f"Tol_47 file {tol47}: {p}")
        rep.append(f"Tol_47 cross-check ({osp.basename(tol47)}, sha256 {sha256_file(tol47)[:16]}…): "
                   f"{'OK — every Tol_47 widget name present, contract satisfied' if not cp else str(len(cp)) + ' problem(s)'}")
    elif tol47:
        warns.append(f"Tol_47 file {tol47} not found — widget names checked against the registry only")
    rep.append(f"{len(files)} payload files, {len(tracks)} tracking jobs over {len(by_root)} roots, "
               f"{len(produced)} downstream tasks")
    return errs, warns, rep

# COMMAND ----------

# MAGIC %md ## SELFTEST (local: `TOL_SELFTEST=1 python Tol_47b_launcher.py`) — preflight subset also runs on Databricks

# COMMAND ----------

def _expect_raise(fn, frag):
    try:
        fn()
    except (ConfigError, CudaProbeError) as e:
        assert frag in str(e), f"raised, but message lacks {frag!r}: {e}"
        return
    raise AssertionError(f"expected an error containing {frag!r}")

def preflight():
    """Pure checks: registries, presets, parsing, planning. Cheap; also run on Databricks before anything else."""
    assert len(TOL47_WIDGETS) == 37 and len(set(TOL47_WIDGETS)) == 37, "Tol_47 has 37 widgets"
    assert len(TOL03_WIDGETS) == 17 and len(set(TOL03_WIDGETS)) == 17
    assert set(TOL03_WIDGETS) - {"detect_night_sec", "track_root", "student_ckpt"} == set(TOL47_UNCHANGED)
    for name, s in SCRIPTS.items():
        for pr in s["presets"]:
            assert set(PRESETS[pr]) | set(REQUIRED_RUN_KEYS) == set(s["widgets"]), (name, pr)
            for k, v in PRESETS[pr].items(): validate_value(k, v)
            assert not policy_problems(name, {**PRESETS[pr], "stage": "20240422", "track_root": "Tracking_x"})
    diff = {k for k in PRESETS["v5"] if PRESETS["v5"][k] != PRESETS["regression"][k]}
    assert diff == {"detector", "dark_fire", "dark_link", "dark_open", "detect_night_sec", "v5_fixes"}, diff
    t03_vs_reg = {k: (PRESETS["t03"][k], PRESETS["regression"][k]) for k in PRESETS["t03"]
                  if PRESETS["t03"][k] != PRESETS["regression"].get(k)}
    assert not t03_vs_reg, f"E6 ref and reg arms must share every common widget value: {t03_vs_reg}"
    assert PRESETS["v5"]["detect_night_sec"] == "120" and PRESETS["v5"]["dark_repeat_sec"] == "10"   # night cadence 120 s, dark refire 10 s
    assert PRESETS["v5"]["partial_repair"] == "true"                                                 # partial repair on
    j = parse_jobs(json.dumps([{"name": "a", "script": "Tol_47_track_v5", "phase": 1, "preset": "v5",
                                "set": {"stage": "20240423", "track_root": "Tracking_v5a10", "from_hhmm": "1152",
                                        "to_hhmm": "1259"}}]))
    assert j[0]["env"]["dark_fire"] == "on" and day_dir(j[0]["env"]) == "20240423_w1152-1259"
    assert wsuf("", "") == "" and wsuf("1946", "2200") == "_w1946-2200" and wsuf("", "1319") == "_w0000-1319"
    assert wsuf("2018", "") == "_w2018-2400"
    assert gpu_class("NVIDIA A100 80GB PCIe") == "A100" and gpu_class("NVIDIA A10-24Q") == "A10"
    assert gpu_class("NVIDIA A10") == "A10" and gpu_class("Tesla T4") == "other"
    return True

def _selftest_parsing():
    base = {"stage": "20240423", "track_root": "Tracking_v5eq_reg"}
    ok = lambda **kw: json.dumps([{"name": "a", "script": "Tol_47_track_v5", "phase": 1, "preset": "regression",
                                   "set": {**base, **kw}}])
    parse_jobs(ok())
    je = parse_jobs(ok(mode="equiv", equiv_tests="E6", e6_ref_root="Tracking_v5eq_ref", e6_reg_root="Tracking_v5eq_reg"))[0]
    assert build_env({}, je, "/s", "/c", "t")["TOLV5_EQUIV_TESTS"] == "E6"
    _expect_raise(lambda: parse_jobs(ok(equiv_tests="E7")), "E1..E6")
    _expect_raise(lambda: parse_jobs(ok(dark_fyre="off")), "does not exist in Tol_47_track_v5")
    _expect_raise(lambda: parse_jobs(json.dumps([{"name": "a", "script": "Tol_47_track_v5", "phase": 1,
                                                  "preset": "v5", "set": {**base, "max_clips": 1}}])), "must be a string")
    _expect_raise(lambda: parse_jobs(ok(decode="sequential")), "speed-up this round")
    _expect_raise(lambda: parse_jobs(ok(gpu_tf="on")), "speed-up this round")
    _expect_raise(lambda: parse_jobs(ok(allow_numeric_change="true")), "speed-up this round")
    _expect_raise(lambda: parse_jobs(ok(track_fps="2.5")), "5 fps is fixed")
    _expect_raise(lambda: parse_jobs(ok(student_ckpt="best_backbone_student.pt")), "the production student")
    _expect_raise(lambda: parse_jobs(ok(dark_fire="yes")), "not in")
    _expect_raise(lambda: parse_jobs(ok(from_hhmm="12:00")), "HHMM")
    _expect_raise(lambda: parse_jobs(ok(stage="2024-04-23")), "YYYYMMDD")
    _expect_raise(lambda: parse_jobs(ok(track_root="Tracking v5")), "must match")
    _expect_raise(lambda: parse_jobs(ok(mem_cap_gib="x")), "not a number")
    _expect_raise(lambda: parse_jobs(json.dumps([{"name": "a", "script": "Tol_47_track_v5", "phase": 1,
                                                  "preset": "v5", "set": {"stage": "20240423"}}])), "missing widgets")
    _expect_raise(lambda: parse_jobs(json.dumps([{"name": "a", "script": "Tol_03_track_v4", "phase": 1,
                                                  "preset": "v5", "set": base}])), "not allowed for Tol_03")
    _expect_raise(lambda: parse_jobs(json.dumps([{"name": "a", "script": "Tol_03_track_v4", "phase": 1,
                                                  "preset": "t03", "set": {**base, "dark_fire": "off"}}])),
                  "does not exist in Tol_03")
    _expect_raise(lambda: parse_jobs(json.dumps([{"name": "a", "script": "X", "phase": 1, "preset": "v5"}])), "script")
    _expect_raise(lambda: parse_jobs(json.dumps([{"name": "a b", "script": "Tol_47_track_v5", "phase": 1,
                                                  "preset": "v5", "set": base}])), "bad name")
    _expect_raise(lambda: parse_jobs(json.dumps([{"name": "a", "script": "Tol_47_track_v5", "phase": 0,
                                                  "preset": "v5", "set": base}])), "phase")
    _expect_raise(lambda: parse_jobs(json.dumps([{"name": "a", "script": "Tol_47_track_v5", "phase": 1,
                                                  "preset": "v5", "set": base, "env": {}}])), "unknown keys")
    _expect_raise(lambda: parse_jobs("[]"), "non-empty")
    _expect_raise(lambda: parse_jobs("{"), "not valid JSON")
    two = lambda s1, s2: [{"name": "r", "script": s1[0], "phase": s1[1], "preset": s1[2], "set": s1[3]},
                          {"name": "g", "script": s2[0], "phase": s2[1], "preset": s2[2], "set": s2[3]}]
    w = {"stage": "20240423", "from_hhmm": "2018", "to_hhmm": "2227"}
    # E6 window: ref (Tol_03) ‖ reg (Tol_47) share stage+window but not the local dir prefix -> allowed concurrently
    ph = plan_phases(parse_jobs(json.dumps(two(("Tol_03_track_v4", 1, "t03", {**w, "track_root": "Tracking_v5eq_ref"}),
                                               ("Tol_47_track_v5", 1, "regression", {**w, "track_root": "Tracking_v5eq_reg"})))))
    assert len(ph) == 1 and len(ph[0][1]) == 2
    _expect_raise(lambda: plan_phases(parse_jobs(json.dumps(two(
        ("Tol_47_track_v5", 1, "regression", {**w, "track_root": "Tracking_v5eq_reg"}),
        ("Tol_47_track_v5", 1, "v5", {**w, "track_root": "Tracking_v5a10_pilot"}))))), "local scratch")
    _expect_raise(lambda: plan_phases(parse_jobs(json.dumps(two(
        ("Tol_47_track_v5", 1, "regression", {**w, "track_root": "Tracking_v5eq_reg"}),
        ("Tol_47_track_v5", 2, "v5", {**w, "track_root": "Tracking_v5eq_reg"}))))), "same output dir")
    ph = plan_phases(parse_jobs(json.dumps(two(("Tol_47_track_v5", 2, "v5", {**w, "track_root": "Tracking_a"}),
                                               ("Tol_47_track_v5", 1, "v5", {**w, "track_root": "Tracking_b"})))))
    assert [p for p, _ in ph] == [1, 2] and ph[0][1][0]["name"] == "g"
    # env mapping
    jb = parse_jobs(json.dumps([{"name": "a", "script": "Tol_47_track_v5", "phase": 1, "preset": "v5",
                                 "set": {"stage": "20240423", "track_root": "Tracking_v5a10"}}]))[0]
    env = build_env({"PATH": "x", "TOLV5_DARK_FIRE": "off", "TOLV4_STAGE": "1", "TOL_SELFTEST": "1",
                     "PYTHONPATH": "/p"}, jb, "/svc/a", "/code", "hf_secret")
    assert env["TOLV5_DARK_FIRE"] == "on" and "TOLV4_STAGE" not in env and "TOL_SELFTEST" not in env
    assert env["TOLV5_STAGE"] == "20240423" and env["TOLV5_MODE"] == "track" and env["TOLV5_DECODE"] == "seek_thread"
    assert env["TOLV4_STAGE_SVC"] == env["TOLV5_STAGE_SVC"] == "/svc/a" and env["HF_TOKEN"] == "hf_secret"
    assert env["PYTHONPATH"] == "/code" + os.pathsep + "/p" and env["PYTORCH_CUDA_ALLOC_CONF"] == ALLOC_CONF
    assert sum(1 for k in env if k.startswith("TOLV5_")) == 37 + 2          # 37 widgets + STAGE_SVC + SVC_KEY
    assert env["TOLV5_SVC_KEY"] == "a"
    jr = parse_jobs(json.dumps([{"name": "r", "script": "Tol_03_track_v4", "phase": 1, "preset": "t03",
                                 "set": {"stage": "20240424", "track_root": "Tracking_v5eq_ref"}}]))[0]
    env = build_env({}, jr, "/svc/r", "/code", "t")
    assert env["TOLV4_STUDENT_CKPT"] == "best_backbone_student_v2.pt" and env["TOLV4_DETECT_NIGHT_SEC"] == "300"
    assert sum(1 for k in env if k.startswith("TOLV4_")) == 17 + 1
    # pins / launcher widgets
    _expect_raise(lambda: check_pins(["a.py"], {"a.py": "__FILL__"}, False), "not a 64-hex")
    assert check_pins(["a.py"], {"a.py": "__FILL__"}, True)
    assert not check_pins(["a.py"], {"a.py": "0" * 64}, False)
    Wd = dict(LAUNCHER_WIDGETS); Wd.update(launch_tag="t", jobs_json=json.dumps(two(
        ("Tol_03_track_v4", 1, "t03", {**w, "track_root": "Tracking_v5eq_ref"}),
        ("Tol_47_track_v5", 1, "regression", {**w, "track_root": "Tracking_v5eq_reg"}))),
        sha_pins=json.dumps({"Tol_03_track_v4.py": T03_SHA256, "Tol_47_track_v5.py": "1" * 64,
                             "tol_v5_core.py": "2" * 64}))
    cfg = parse_launcher_widgets(Wd)
    assert cfg["files"] == ["Tol_03_track_v4.py", "Tol_47_track_v5.py", "tol_v5_core.py"] and not cfg["pin_problems"]
    _expect_raise(lambda: parse_launcher_widgets({**Wd, "expect_gpu": "H100"}), "expect_gpu")
    _expect_raise(lambda: parse_launcher_widgets({**Wd, "launch_tag": ""}), "launch_tag")
    _expect_raise(lambda: parse_launcher_widgets({**Wd, "code_dir": "../x"}), "code_dir")
    _expect_raise(lambda: parse_launcher_widgets({**Wd, "sha_pins": "{}"}), "pins missing")
    # contract: widget table / env prefix may live in the core; staging protocol + HF login must be in the worker
    fake47 = ('# Databricks notebook source\nSTAGE_SVC = os.environ.get("TOLV5_STAGE_SVC", "")\n'
              'x = "req__", "ready__", "error__"\nlogin(token=os.environ["HF_TOKEN"])\nimport tol_v5_core\n'
              'print(f"tol_v5_core {core.CORE_VERSION} from {core.__file__} sha256 {CORE_SHA}")\n'
              'ECHO = {"x": 1,\n        "stage_svc_key": SVC_KEY}\n'
              'print(f"\\nEQUIV {\'PASS\' if EQ[\'pass\'] else \'FAIL\'} on {GPU_NAME}: " +\n'
              '      ", ".join(x) + f" -> {EQ_PATH}", flush=True)\n')
    core_txt = 'ENV_PREFIX = "TOLV5_"\nWIDGETS = {' + ", ".join(f'"{w}": 1' for w in TOL47_WIDGETS) + "}\n"
    assert contract_problems("Tol_47_track_v5", fake47, core_txt) == [], contract_problems("Tol_47_track_v5", fake47, core_txt)
    assert any("never appear" in x for x in contract_problems("Tol_47_track_v5", fake47))
    assert any("staging-service" in x for x in contract_problems("Tol_47_track_v5", core_txt, core_txt))
    assert any("selftest" in x for x in contract_problems("tol_v5_core.py", core_txt))
    print("SELFTEST parsing/planning/env/pins/contract: OK")

def _selftest_probe():
    good = {"ok": True, "name": "NVIDIA A10-24Q", "alloc_conf": ALLOC_CONF, "dist": {"rfdetr": "1.10.1"},
            "runtime": {"transformers": "5.2.0", "cv2": "4.11.0", "numpy": "1.26.4", "PIL": "11.1.0",
                        "torch": "2.7.0+cu126"}}
    fail = {**good, "ok": False, "cuda_error": "RuntimeError: Found no NVIDIA driver"}
    seq = [fail, fail, good]; calls = []
    def cmd():
        calls.append(1); return 0, "noise\nPROBE_JSON " + json.dumps(seq[len(calls) - 1]) + "\n", ""
    p, tries = run_cuda_probe(cmd, 3, 0, lambda m: None, sleeper=lambda s: None)
    assert p["ok"] and len(tries) == 3 and len(calls) == 3
    _expect_raise(lambda: run_cuda_probe(lambda: (1, "", "boom"), 2, 0, lambda m: None, sleeper=lambda s: None),
                  "after 3 probe attempts")
    calls.clear()
    bad = {**fail, "runtime": {**good["runtime"], "cv2": "4.12.0"}}
    def cmd2():
        calls.append(1); return 0, "PROBE_JSON " + json.dumps(bad), ""
    _expect_raise(lambda: run_cuda_probe(cmd2, 3, 0, lambda m: None, sleeper=lambda s: None), "cv2 4.12.0")
    assert len(calls) == 1, "an environment mismatch must not be retried"
    assert env_problems({**good, "alloc_conf": None}) and env_problems({**good, "dist": {"rfdetr": "1.9"}})
    assert env_problems({**good, "runtime": {**good["runtime"], "numpy": "2.1.0"}})
    assert env_problems({**good, "runtime": {**good["runtime"], "torch": "2.4.1"}})
    assert not env_problems(good) and not env_problems({"ok": False, "cuda_error": "x"})
    smi = ("==NVSMI LOG==\nDriver Version : 550.90\nCUDA Version : 12.4\nProduct Name : NVIDIA A10-24Q\n"
           "    vGPU Software Licensed Product\n        License Status : Licensed (Expiry: 2026-10-1)\nSerial : 123\n")
    assert smi_lines(smi) == ["Driver Version : 550.90", "CUDA Version : 12.4", "vGPU Software Licensed Product",
                              "License Status : Licensed (Expiry: 2026-10-1)"]
    prev = [{"attempt": 1, "parent_run": "R1", "cluster_id": "C1", "outcome": "probe_failed"}]
    assert "SAME cluster" in attempt_gate(prev, "C1", "R1", "probe_only")
    assert attempt_gate(prev, "C2", "R1", "probe_only") is None
    assert attempt_gate(prev, "C1", "R2", "probe_only") is None          # a fresh submission, not a retry
    prev2 = [{"attempt": 1, "parent_run": "R1", "cluster_id": "C1", "outcome": "job_failed"}]
    assert "presumed deterministic" in attempt_gate(prev2, "C2", "R1", "probe_only")
    assert attempt_gate(prev2, "C2", "R1", "always") is None
    assert attempt_gate(prev2, "C2", None, "probe_only") is None          # run ids unknown -> cannot judge -> allow
    print("SELFTEST probe/attempt gate: OK")

def _t03_path():
    here = osp.dirname(osp.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    return osp.join(here, "Tol_03_track_v4.py")

def _selftest_t03_contract():
    fp = _t03_path()
    if not osp.exists(fp):
        print(f"SELFTEST Tol_03 contract: SKIPPED (no {fp})"); return None
    txt = open(fp, encoding="utf-8").read()
    assert sha256_file(fp) == T03_SHA256, "Tol_03_track_v4.py is no longer byte-identical to the v4 base"
    for line in ('WSUF = f"_w{FROM_HHMM or \'0000\'}-{TO_HHMM or \'2400\'}" if (FROM_HHMM or TO_HHMM) else ""',
                 'LOCAL = f"/local_disk0/tmp/tol_v4_{STAGE}{WSUF}"',
                 'STAGE_SVC  = os.environ.get("TOLV4_STAGE_SVC", "")',
                 'return os.environ.get("TOLV4_" + name.upper(), default)',
                 'login(token=os.environ["HF_TOKEN"].strip())',
                 'req = osp.join(STAGE_SVC, f"req__{STAGE}__{vname}")',
                 'rdy = osp.join(STAGE_SVC, f"ready__{STAGE}__{vname}")',
                 'err = osp.join(STAGE_SVC, f"error__{STAGE}__{vname}")'):
        assert line in txt, f"Tol_03 contract line changed: {line}"
    got = set(re.findall(r'\bP\("([a-z_0-9]+)"', txt))
    assert got == set(TOL03_WIDGETS), f"Tol_03 widgets {sorted(got)} != registry"
    assert not contract_problems("Tol_03_track_v4", txt), contract_problems("Tol_03_track_v4", txt)
    print("SELFTEST Tol_03 contract (sha, WSUF, LOCAL, env prefix, staging protocol, 17 widgets): OK")
    return txt

def _extract_stage_clip(txt):
    m = re.search(r"^def stage_clip\(vname, dst\):\n(?:(?:    .*|)\n)+", txt, re.M)
    assert m, "stage_clip not found"
    return m.group(0)

FAKE_WORKER = r'''
import os, os.path as osp, sys, time, shutil
STAGE = os.environ.get("TOLV4_STAGE") or os.environ["TOLV5_STAGE"]
STAGE_SVC = os.environ.get("TOLV4_STAGE_SVC") or os.environ.get("TOLV5_STAGE_SVC", "")
SVC_KEY = os.environ.get("TOLV5_SVC_KEY", "") or STAGE          # Tol_47:148 (Tol_03 has no SVC_KEY)
VIDEO_DIR = DBFS_VIDEO = "/nonexistent"
exec(open(os.environ["FAKE_STAGE_CLIP_SRC"]).read())
for vname in os.environ["FAKE_CLIPS"].split(","):
    dst = osp.join(os.environ["FAKE_DST_DIR"], "cur.mp4")
    stage_clip(vname, dst)
    assert open(dst, "rb").read() == ("CLIP:" + vname).encode(), "wrong bytes staged"
print("fake worker done", flush=True)
sys.exit(int(os.environ.get("FAKE_RC", "0")))
'''

def _selftest_staging(t03_txt):
    """Fake workers execute the REAL stage_clip of Tol_03 (and of Tol_47 when present) against serve_requests +
    run_phase: two same-stage arms concurrently, as in E6."""
    if t03_txt is None:
        print("SELFTEST staging with real stage_clip: SKIPPED (no Tol_03)"); return
    t47p = osp.join(osp.dirname(_t03_path()), "Tol_47_track_v5.py")
    t47_txt = open(t47p, encoding="utf-8").read() if osp.exists(t47p) else None
    tmp = tempfile.mkdtemp(prefix="tol47b_st_")
    try:
        src = osp.join(tmp, "videos"); os.makedirs(src)
        clips = ["Tol_ch01-20240423-201805-205213.mp4", "Tol_ch01-20240423-205213-212622.mp4"]
        for c in clips:
            with open(osp.join(src, c), "wb") as f: f.write(("CLIP:" + c).encode())
        sc = osp.join(tmp, "stage_clip_src.py")
        with open(sc, "w", encoding="utf-8") as f: f.write(_extract_stage_clip(t03_txt))
        sc47 = sc
        if t47_txt is not None:
            sc47 = osp.join(tmp, "stage_clip_src_47.py")
            with open(sc47, "w", encoding="utf-8") as f: f.write(_extract_stage_clip(t47_txt))
        wk = osp.join(tmp, "fake_worker.py")
        with open(wk, "w", encoding="utf-8") as f: f.write(FAKE_WORKER)
        svc_root, logs = osp.join(tmp, "svc"), osp.join(tmp, "logs"); os.makedirs(logs)
        def copy_fn(vname, dst):
            shutil.copyfile(osp.join(src, vname), dst)          # stands in for dbutils.fs.cp
        base = {"stage": "20240423", "from_hhmm": "2018", "to_hhmm": "2227"}
        spec = [  # same STAGE, same clips, concurrently: the N3b collision case
            {"name": "ref", "script": "Tol_03_track_v4", "phase": 1, "preset": "t03",
             "set": {**base, "track_root": "Tracking_v5eq_ref"}},
            {"name": "reg", "script": "Tol_47_track_v5", "phase": 1, "preset": "regression",
             "set": {**base, "track_root": "Tracking_v5eq_reg"}},
            {"name": "bad", "script": "Tol_03_track_v4", "phase": 1, "preset": "t03",
             "set": {"stage": "20240423", "from_hhmm": "0544", "to_hhmm": "0614", "track_root": "Tracking_v5eq_ref"}},
            {"name": "later", "script": "Tol_03_track_v4", "phase": 2, "preset": "t03",
             "set": {"stage": "20240422", "to_hhmm": "1319", "track_root": "Tracking_v5eq_ref"}},
        ]
        phases = plan_phases(parse_jobs(json.dumps(spec)))
        records, logl = [], []
        svc_dirs = {}
        def spawn(j):
            d = osp.join(svc_root, j["name"]); os.makedirs(d, exist_ok=True); svc_dirs[j["name"]] = d
            dst_dir = osp.join(tmp, "local", j["name"]); os.makedirs(dst_dir, exist_ok=True)
            env = build_env(dict(os.environ), j, d, tmp, "hf_x")
            env.update(FAKE_STAGE_CLIP_SRC=sc47 if j["script"] == "Tol_47_track_v5" else sc, FAKE_DST_DIR=dst_dir,
                       FAKE_CLIPS=",".join(clips) if j["name"] != "bad" else "Tol_ch01-20240423-054428-061426.mp4")
            lf = open(osp.join(logs, j["name"] + ".log"), "w")
            return {"proc": subprocess.Popen([sys.executable, wk], env=env, stdout=lf, stderr=subprocess.STDOUT),
                    "logf": lf, "t0": time.time(), "job": j}
        serve = lambda alive: serve_requests(svc_dirs, alive, copy_fn, tmp, logl.append, records)
        results, skipped = {}, []
        for p, js in phases:
            if any(r["rc"] != 0 for r in results.values()):
                skipped += [j["name"] for j in js]; continue
            results.update(run_phase(js, spawn, serve, stagger_s=0.0, poll_s=0.05))
        rcs = {k: v["rc"] for k, v in results.items()}
        assert rcs["ref"] == 0 and rcs["reg"] == 0, (rcs, open(osp.join(logs, "ref.log")).read()[-800:])
        assert rcs["bad"] != 0 and "staging service failed" in open(osp.join(logs, "bad.log")).read()
        assert skipped == ["later"], skipped
        ok = [r for r in records if "error" not in r]
        assert sorted((r["job"], r["clip"]) for r in ok) == sorted((j, c) for j in ("ref", "reg") for c in clips)
        assert [r["job"] for r in records if "error" in r] == ["bad"]
        left = [f for d in svc_dirs.values() for f in os.listdir(d)]
        assert not left, f"protocol files left behind: {left}"
        # refusal paths + dead-worker drop (unit level)
        d = osp.join(svc_root, "unit"); os.makedirs(d)
        with open(osp.join(d, "req__20240423__notaclip.txt"), "w") as f: f.write(osp.join(tmp, "x.mp4"))
        with open(osp.join(d, "req__20240423__" + clips[0]), "w") as f: f.write("/elsewhere/cur.mp4")
        with open(osp.join(d, "req__20240423__" + clips[1]), "w") as f: f.write(tmp + "/../escape.mp4")
        recs = []
        serve_requests({"unit": d}, lambda n: True, copy_fn, tmp, lambda m: None, recs)
        assert len(recs) == 3 and all("refused" in r["error"] for r in recs), recs
        assert sorted(f.split("__")[0] for f in os.listdir(d)) == ["error"] * 3
        assert not osp.exists(osp.join(osp.dirname(tmp), "escape.mp4"))
        for f in os.listdir(d): os.remove(osp.join(d, f))
        with open(osp.join(d, "req__20240423__" + clips[0]), "w") as f: f.write(osp.join(tmp, "y.mp4"))
        recs = []
        serve_requests({"unit": d}, lambda n: False, copy_fn, tmp, lambda m: None, recs)
        assert not recs and not os.listdir(d), "a finished job's request must be dropped without an answer"
        print(f"SELFTEST staging service (real stage_clip of Tol_03 + {'Tol_47' if t47_txt else 'Tol_03 (no Tol_47 yet)'}, "
              f"2 concurrent same-stage jobs, error path, stop_on_fail skip, refusals): OK — {len(ok)} clips staged")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def _selftest_checker():
    tmp = tempfile.mkdtemp(prefix="tol47b_ck_")
    try:
        nb = NB_DIR + "/"
        cl = lambda node, gpu: {"spark_version": "17.3.x-gpu-ml-scala2.13" if gpu else "17.3.x-scala2.13",
                                "node_type_id": node, "num_workers": 0, "data_security_mode": "SINGLE_USER",
                                "single_user_name": "<DATABRICKS_USER>", "custom_tags": {"ResourceClass": "SingleNode"},
                                "spark_conf": {"spark.databricks.cluster.profile": "singleNode", "spark.master": "local[*]"},
                                "azure_attributes": {"availability": "ON_DEMAND_AZURE"},
                                "spark_env_vars": {"PYTORCH_CUDA_ALLOC_CONF": ALLOC_CONF}}
        bp = {**PRESETS["v5"], "stage": "20240422", "track_root": "Tracking_v5a10"}
        good = {"run_name": "x", "tasks": [
            {"task_key": "t", "timeout_seconds": 259200, "max_retries": 1, "retry_on_timeout": False,
             "notebook_task": {"notebook_path": nb + "Tol_47_track_v5", "source": "WORKSPACE", "base_parameters": bp},
             "new_cluster": cl("Standard_NV36ads_A10_v5", True)},
            {"task_key": "c", "timeout_seconds": 100,
             "notebook_task": {"notebook_path": nb + "Tol_32_dense_crops", "source": "WORKSPACE",
                               "base_parameters": {"day": "20240422", "track_root": "Tracking_v5a10"}},
             "new_cluster": cl("Standard_D16ds_v5", False)},
            {"task_key": "r", "timeout_seconds": 100,
             "notebook_task": {"notebook_path": nb + "Tol_17_overlay_video", "source": "WORKSPACE",
                               "base_parameters": {"day": "20240422", "track_root": "Tracking_v5a10",
                                                   "relabel_json": relabel_path("Tracking_v5a10", "20240422"),
                                                   "out_name": "20240422_v5a10_ident"}},
             "new_cluster": {**cl("Standard_D8ds_v5", False), "spark_version": "17.3.x-cpu-ml-scala2.13"}}]}
        def run(obj, bom=False):
            for f in os.listdir(tmp): os.remove(osp.join(tmp, f))
            raw = json.dumps(obj).encode("utf-8")
            with open(osp.join(tmp, "p.json"), "wb") as f: f.write((b"\xef\xbb\xbf" if bom else b"") + raw)
            return check_payload_dir(tmp)[0]
        assert run(good) == [], run(good)
        assert any("BOM" in e for e in run(good, bom=True))
        import copy
        def mut(fn):
            o = copy.deepcopy(good); fn(o); return run(o)
        assert any("own its node" in e for e in mut(lambda o: o["tasks"][0].update(existing_cluster_id="x")))
        assert any("PYTORCH_CUDA_ALLOC_CONF" in e for e in mut(lambda o: o["tasks"][0]["new_cluster"].pop("spark_env_vars")))
        assert any("not in the Tol_47 widget list" in e for e in mut(lambda o: o["tasks"][0]["notebook_task"]["base_parameters"].update(darkfire="on")))
        assert any("EVERY" in e for e in mut(lambda o: o["tasks"][0]["notebook_task"]["base_parameters"].pop("gpu_tf")))
        assert any("A10-only" in e for e in mut(lambda o: o["tasks"][0]["new_cluster"].update(node_type_id="Standard_NC24ads_A100_v4")))
        assert any("72 h" in e for e in mut(lambda o: o["tasks"][0].update(timeout_seconds=3600)))
        assert any("differs from the v5 preset" in e for e in mut(lambda o: o["tasks"][0]["notebook_task"]["base_parameters"].update(dark_open="off")))
        assert any("no tracking payload" in e for e in mut(lambda o: o["tasks"][1]["notebook_task"]["base_parameters"].update(day="20240423")))
        assert any("not a number" in e or "must be empty or HHMM" in e
                   for e in mut(lambda o: o["tasks"][0]["notebook_task"]["base_parameters"].update(to_hhmm="25:00")))
        assert any("mem_cap_gib" in e for e in mut(lambda o: o["tasks"][0]["notebook_task"]["base_parameters"].update(mem_cap_gib="21.0")))
        assert any("GT-free production" in e for e in mut(lambda o: o["tasks"][2]["notebook_task"]["base_parameters"].update(
            relabel_json=TOL + "/Tracking_v4v2_reid/20240422/relabel_0422_reviewed.json")))
        # equiv job names must be unique across launcher payloads (verdict files are attributed by name)
        lt = lambda tag: {"task_key": tag, "timeout_seconds": 100, "max_retries": 0, "notebook_task": {
            "notebook_path": nb + "Tol_47b_launcher", "source": "WORKSPACE", "base_parameters": {
                "launch_tag": tag, "expect_gpu": "A100", "jobs_json": json.dumps([{
                    "name": "equiv_x", "script": "Tol_47_track_v5", "phase": 1, "preset": "v5",
                    "set": {"mode": "equiv", "stage": "20240422", "track_root": "Tracking_v5equiv_a100"}}])}},
            "new_cluster": cl("Standard_NC24ads_A100_v4", True)}
        dup = mut(lambda o: o["tasks"].extend([lt("L1a"), lt("L1b")]))
        assert any("equiv job name 'equiv_x'" in e for e in dup), dup
        # review r2 finding 3: a launcher payload must name the launcher revision it was generated for
        one = mut(lambda o: o["tasks"].append(lt("L1a")))
        assert any("expect_launcher ''" in e for e in one), one
        def with_el(t): t["notebook_task"]["base_parameters"]["expect_launcher"] = LAUNCHER_VERSION; return t
        assert not any("expect_launcher" in e for e in mut(lambda o: o["tasks"].append(with_el(lt("L1a"))))), \
            mut(lambda o: o["tasks"].append(with_el(lt("L1a"))))
        # review r2 finding 1: one payload per launcher-only root (checked across FILES)
        def lfl(tag, stage):
            t = with_el(lt(tag))
            t["notebook_task"]["base_parameters"]["jobs_json"] = json.dumps([{
                "name": f"floor_{tag}", "script": "Tol_47_track_v5", "phase": 1, "preset": "regression",
                "set": {"stage": stage, "to_hhmm": "1319", "partial_repair": "false", "track_root": "Tracking_v5floor_a100"}}])
            return {"run_name": tag, "tasks": [t]}
        for f in os.listdir(tmp): os.remove(osp.join(tmp, f))
        for i, (tag, st) in enumerate((("LA", "20240422"), ("LB", "20240423"))):
            with open(osp.join(tmp, f"p{i}.json"), "w", encoding="utf-8") as f: json.dump(lfl(tag, st), f)
        e2 = check_payload_dir(tmp)[0]
        assert any("launcher-only root Tracking_v5floor_a100 is written by 2 payloads" in e for e in e2), e2
        os.remove(osp.join(tmp, "p1.json"))
        assert check_payload_dir(tmp)[0] == [], check_payload_dir(tmp)[0]
        # change-control revisions of direct roots (review r1 finding 3): the base root's policy, one revision per chain
        def rev(o, r):
            for t in o["tasks"]: t["notebook_task"]["base_parameters"]["track_root"] = r
            o["tasks"][2]["notebook_task"]["base_parameters"]["relabel_json"] = relabel_path(r, "20240422")
        assert mut(lambda o: rev(o, "Tracking_v5a10_r2")) == [], mut(lambda o: rev(o, "Tracking_v5a10_r2"))
        assert any("A10-only" in e for e in mut(lambda o: (rev(o, "Tracking_v5a10_r3"), o["tasks"][0]["new_cluster"].update(
            node_type_id="Standard_NC24ads_A100_v4"))))
        assert any("GT-free production" in e for e in mut(lambda o: (rev(o, "Tracking_v5a10_r2"), o["tasks"][2][
            "notebook_task"]["base_parameters"].update(relabel_json=relabel_path("Tracking_v5a10", "20240422")))))
        for bad in ("Tracking_v5a10_r1", "Tracking_v5eq_ref_r2", "Tracking_v5a10_rx"):
            assert any("not in the spec's root list" in e for e in mut(lambda o: rev(o, bad))), bad
        def mix(o):
            o2 = copy.deepcopy(o["tasks"][0]); o2["task_key"] = "t2"
            o2["notebook_task"]["base_parameters"].update(track_root="Tracking_v5a10_r2", stage="20240423")
            o["tasks"].append(o2)
        assert any("mix revisions" in e for e in mut(mix)), mut(mix)
        print("SELFTEST payload checker (BOM, node ownership, alloc conf, unknown/missing widget, GPU-root, timeout, "
              "preset drift, orphan downstream, cap, relabel path, equiv name uniqueness, root revisions, "
              "expect_launcher, one payload per launcher-only root): OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def _selftest_core():
    tmp = tempfile.mkdtemp(prefix="tol47b_core_")
    try:
        fake = {k: (PRESETS["v5"].get(k, "x"), "str", None) for k in TOL47_WIDGETS + TOL47_OPTIONAL}
        fp = osp.join(tmp, "core.py")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(f"WIDGETS = {fake!r}\nREGRESSION_FLAGS = {{'detector': 'sam3'}}\nENV_PREFIX = 'TOLV5_'\n")
        assert core_problems(fp) == [], core_problems(fp)
        fake["dark_repeat_sec"] = ("30", "float", None); fake.pop("allow_mixed_gpu")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(f"WIDGETS = {fake!r}\nREGRESSION_FLAGS = {{'detector': 'rfdetr'}}\n")
        pr = core_problems(fp)
        assert any("dark_repeat_sec" in x for x in pr) and any("launcher-only ['allow_mixed_gpu']" in x for x in pr) \
            and any("regression flag detector" in x for x in pr), pr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    real = osp.join(osp.dirname(_t03_path()), "tol_v5_core.py")
    if osp.exists(real):
        pr = core_problems(real)
        assert not pr, f"launcher presets disagree with tol_v5_core.py: {pr}"
        print(f"SELFTEST tol_v5_core cross-check (real file, sha256 {sha256_file(real)[:16]}…): OK")
        import ast                                  # review r2 finding 3: ONE source_norm definition (core's)
        with open(real, encoding="utf-8") as f: tree = ast.parse(f.read())
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "source_norm_sha256"]
        assert fn, "tol_v5_core.py has no source_norm_sha256"
        ns_ = {"hashlib": hashlib}
        exec(compile(ast.Module(body=fn, type_ignores=[]), real, "exec"), ns_)
        me_ = open(osp.join(osp.dirname(_t03_path()), "Tol_47b_launcher.py"), "rb").read()
        for sample in (me_, me_.replace(b"\n", b"\r\n"), b"\xef\xbb\xbf# x \r\n\r\n\ty\r", "# ‖ z\n"):
            assert ns_["source_norm_sha256"](sample) == source_norm_sha256(sample), "source_norm differs from the core's"
        print("SELFTEST source_norm_sha256 == tol_v5_core.source_norm_sha256 (the gate's comparison of the running "
              "notebook with the reviewed file): OK")
    else:
        print("SELFTEST tol_v5_core cross-check: synthetic OK; real file SKIPPED (absent)")

def _selftest_v2():
    """tol47b-2 additions: E6 phase order (finding 10), worker core sha (finding 3), verdict attribution + tag
    (finding 12), relabel path shared with the GT-free production step (finding 7)."""
    J = lambda name, script, phase, preset, **kw: {"name": name, "script": script, "phase": phase, "preset": preset,
                                                   "set": kw}
    SM = dict(stage="20240424", from_hhmm="1946", to_hhmm="2200", max_clips="1", max_frames="1500")
    WN = dict(stage="20240423", from_hhmm="2018", to_hhmm="2227")
    FL = dict(stage="20240422", to_hhmm="1319", partial_repair="false")
    V = lambda name, phase, w, ref="Tracking_v5eq_ref", reg="Tracking_v5eq_reg": J(
        name, "Tol_47_track_v5", phase, "v5", mode="equiv", equiv_tests="E6", e6_ref_root=ref, e6_reg_root=reg,
        track_root="Tracking_v5equiv_a100", **{k: v for k, v in w.items() if k in ("stage", "from_hhmm", "to_hhmm")})
    arms = [J("e6s_ref", "Tol_03_track_v4", 1, "t03", track_root="Tracking_v5eq_ref", **SM),
            J("e6s_reg", "Tol_47_track_v5", 1, "regression", track_root="Tracking_v5eq_reg", **SM),
            J("e6w_ref", "Tol_03_track_v4", 2, "t03", track_root="Tracking_v5eq_ref", **WN),
            J("e6w_reg", "Tol_47_track_v5", 2, "regression", track_root="Tracking_v5eq_reg", **WN)]
    good = arms + [V("e6cmp_smoke", 3, SM), V("e6cmp_window", 4, WN),
                   J("floor_a100", "Tol_47_track_v5", 5, "regression", track_root="Tracking_v5floor_a100", **FL)]
    jobs = parse_jobs(json.dumps(good)); plan_phases(jobs)
    assert e6_order_problems(jobs) == [], e6_order_problems(jobs)
    # the tol47b-1 L2 order: floor (report-only) in phase 3, verdicts in phase 4 -> refused (finding 10 + one per phase)
    old = arms + [J("floor_a100", "Tol_47_track_v5", 3, "regression", track_root="Tracking_v5floor_a100", **FL),
                  V("e6cmp_smoke", 4, SM), V("e6cmp_window", 4, WN)]
    pr = e6_order_problems(parse_jobs(json.dumps(old)))
    assert any("floor_a100" in x and "later phase" in x for x in pr) and any("one per phase" in x for x in pr), pr
    Wd = dict(LAUNCHER_WIDGETS); Wd.update(launch_tag="t", jobs_json=json.dumps(old), allow_unpinned="true")
    _expect_raise(lambda: parse_launcher_widgets(Wd), "E6 phase order")
    # verdict in the same phase as its arm / arm missing
    pr = e6_order_problems(parse_jobs(json.dumps(arms[:2] + [V("v", 1, SM)])))
    assert any("same or a later phase" in x for x in pr), pr
    pr = e6_order_problems(parse_jobs(json.dumps(arms[:1] + [V("v", 2, SM)])))
    assert any("no job of this launch writes Tol/Tracking_v5eq_reg/20240424_w1946-2200" in x for x in pr), pr
    # an E1-E5 equivalence job is not an E6 verdict: L1 / L3A structures are untouched
    l3a = [J("equiv_a10", "Tol_47_track_v5", 1, "v5", mode="equiv", stage="20240422", track_root="Tracking_v5equiv_a10"),
           J("floor_a10", "Tol_47_track_v5", 2, "regression", track_root="Tracking_v5floor_a10", **FL)]
    assert e6_order_problems(parse_jobs(json.dumps(l3a))) == []
    # worker core line (Tol_47:117 format, path with spaces) -> pin check
    h1, h2 = "a" * 64, "b" * 64
    log = ("noise\ntol_v5_core v5.0 2026-09-24 from /local_disk0/tmp/tol_v5_launcher/L1/code/tol_v5_core.py "
           f"sha256 {h1}\nmore\n")
    assert worker_core_from_log(log) == ("/local_disk0/tmp/tol_v5_launcher/L1/code/tol_v5_core.py", h1)
    assert worker_core_problem(log, h1) is None and worker_core_problem("no line yet", h1) is None
    ws = log.replace("/local_disk0/tmp/tol_v5_launcher/L1/code", "/Workspace/Users/x/Model Distillation/Testing on Tol2")
    assert "shadowed the staged copy" in worker_core_problem(ws.replace(h1, h2), h1)
    # verdict attribution: log line + echo.stage_svc_key; a same-second overwrite by a concurrent job is detected
    P1 = "/Volumes/c/s/b/SAM3 Distillation and re-id/Tol/Metrics_v5_equiv_A100_20260925_101500.json"
    P2 = "/Volumes/c/s/b/SAM3 Distillation and re-id/Tol/Metrics_v5_equiv_A100_20260925_101522.json"
    files = {P1: {"echo": {"stage_svc_key": "e6cmp_smoke"}}, P2: {"echo": {"stage_svc_key": "e6cmp_window"}}}
    rd = lambda p: files[p]
    ln = lambda p, v="PASS": f"x\n\nEQUIV {v} on NVIDIA A100 80GB PCIe: E6 True -> {p}\n"
    assert equiv_source("e6cmp_smoke", ln(P1), {P1, P2}, rd) == (P1, "log+echo", [])
    assert equiv_source("e6cmp_window", ln(P2, "FAIL"), {P1, P2}, rd)[0] == P2
    assert equiv_source("e6cmp_window", "died before the verdict line", {P1, P2}, rd) == (P2, "echo", [])
    files[P1] = {"echo": {"stage_svc_key": "e6cmp_window"}}          # window overwrote smoke's file (same second)
    s_, _h, pr = equiv_source("e6cmp_smoke", ln(P1), {P1, P2}, rd)
    assert s_ is None and pr and "overwritten" in pr[0], pr
    assert equiv_source("e6cmp_smoke", "", set(), rd)[0] is None
    assert equiv_copy_name("A100", "L2_regression_floor_a100", "e6cmp_smoke") == \
        "Metrics_v5_equiv_A100_L2_regression_floor_a100__e6cmp_smoke.json"
    # relabel path (read by Tol_17, written by build/prod_relabel_v5.py)
    assert root_slug("Tracking_v5a10") == "v5a10" and root_slug("Tracking_v5a100cap21") == "v5a100cap21"
    assert relabel_path("Tracking_v5a10", "20240423").endswith("/Tol/Metadata/relabel_v5a10/relabel_week_20240423.json")
    _expect_raise(lambda: root_slug("Tracking v5"), "must match")
    assert sum(EXPECTED_CLIPS.values()) == 217
    print("SELFTEST tol47b-2 (E6 phase order, worker core sha, verdict attribution + tag, relabel path): OK")
    # ---- tol47b-3 (launcher review r1 findings 3 and 7) ----
    # the tol47b-2 L2 order (both verdicts concurrently in phase 3) -> refused: one-second verdict names can collide
    t2 = arms + [V("e6cmp_smoke", 3, SM), V("e6cmp_window", 3, WN),
                 J("floor_a100", "Tol_47_track_v5", 4, "regression", track_root="Tracking_v5floor_a100", **FL)]
    pr = e6_order_problems(parse_jobs(json.dumps(t2)))
    assert len(pr) == 1 and "phase 3 runs 2 mode=equiv jobs" in pr[0] and "finding 7" in pr[0], pr
    pr = e6_order_problems(parse_jobs(json.dumps(l3a + [J("equiv_b", "Tol_47_track_v5", 1, "v5", mode="equiv",
                                                          stage="20240423", track_root="Tracking_v5equiv_a10")])))
    assert pr and "one per phase" in pr[0], pr                        # also for E1-E5 jobs (L1 || L2 share Tol/)
    # a failed verdict never skips the later phases; a core-sha mismatch always does; an arm failure does
    jb = parse_jobs(json.dumps(good))
    assert blocking_failures({"e6cmp_smoke": 1}, jb) == {}
    assert blocking_failures({"e6cmp_smoke": "equiv_output_unattributed"}, jb) == {}
    assert blocking_failures({"e6cmp_smoke": "core_sha_mismatch"}, jb) == {"e6cmp_smoke": "core_sha_mismatch"}
    assert blocking_failures({"e6s_ref": 1, "e6cmp_window": 1}, jb) == {"e6s_ref": 1}
    # launcher-only roots are never resumed: existing outputs found, moved to the attic (never deleted)
    T = "/V/Tol"
    ex = {f"{T}/Tracking_v5eq_reg/20240423_w2018-2227", f"{T}/Metrics_track_v5_Tracking_v5eq_reg_20240423_w2018-2227.json",
          f"{T}/Tracking_v5eq_reg/ROOT_CONFIG.json", f"{T}/Tracking_v5eq_ref/20240424_w1946-2200",
          f"{T}/Metrics_track_v4_Tracking_v5eq_ref_20240424_w1946-2200.json", f"{T}/Tracking_v5a10/20240422"}
    ls = {f"{T}/Tracking_v5eq_reg": ["ROOT_CONFIG.json", "20240423_w2018-2227"], f"{T}/Tracking_v5eq_ref": ["20240424_w1946-2200"]}
    found, moves = reset_plan(jb, ex.__contains__, lambda d: ls.get(d, []), "L2_x_20260925T000000Z", tol=T)
    assert set(found) == ex - {f"{T}/Tracking_v5a10/20240422"}, found
    assert all(d == T + "/Launcher_attic/L2_x_20260925T000000Z/" + s[len(T) + 1:] for s, d in moves) and len(moves) == 5
    assert reset_plan(jb, lambda p: False, lambda d: [], "s", tol=T) == ([], [])
    # tol47b-4 (review r2 finding 1): a stale day dir of an older payload layout no longer keeps the ROOT_CONFIG in
    # place -- the root is reset whole (ROOT_CONFIG + the stale day dir + its telemetry), so it can never hold outputs
    # written under a ROOT_CONFIG it no longer has
    ls[f"{T}/Tracking_v5eq_reg"].append("20240425")
    ex2 = ex | {f"{T}/Tracking_v5eq_reg/20240425", f"{T}/Metrics_track_v5_Tracking_v5eq_reg_20240425.json"}
    found, _m = reset_plan(jb, ex2.__contains__, lambda d: ls.get(d, []), "s", tol=T)
    assert {f"{T}/Tracking_v5eq_reg/ROOT_CONFIG.json", f"{T}/Tracking_v5eq_reg/20240425",
            f"{T}/Metrics_track_v5_Tracking_v5eq_reg_20240425.json"} <= set(found) and len(found) == 7, found
    # the reviewer's reproduction: only the ROOT_CONFIG of a launcher-only root is left (its day dirs were moved by an
    # earlier reset / a partial run) -> it is found (refused without reset) and moved with reset
    only_rc = {f"{T}/Tracking_v5eq_reg/ROOT_CONFIG.json"}
    found, _m = reset_plan(jb, only_rc.__contains__,
                           lambda d: ["ROOT_CONFIG.json"] if d == f"{T}/Tracking_v5eq_reg" else [], "s", tol=T)
    assert found == [f"{T}/Tracking_v5eq_reg/ROOT_CONFIG.json"], found
    # the two conditional sequential payloads: own roots each, so resetting one never touches the other
    SEQ = {"smoke": SM, "window": WN}
    seqjobs = {k: parse_jobs(json.dumps([
        J(f"{k}_ref", "Tol_03_track_v4", 1, "t03", track_root=f"Tracking_v5eq_ref_seq_{k}", **w),
        J(f"{k}_reg", "Tol_47_track_v5", 2, "regression", track_root=f"Tracking_v5eq_reg_seq_{k}", **w),
        V(f"{k}_cmp", 3, w, ref=f"Tracking_v5eq_ref_seq_{k}", reg=f"Tracking_v5eq_reg_seq_{k}")])) for k, w in SEQ.items()}
    both = {}
    for k, w in SEQ.items():
        dd_ = day_dir(seqjobs[k][0]["env"])
        both[f"{T}/Tracking_v5eq_ref_seq_{k}"] = [dd_]
        both[f"{T}/Tracking_v5eq_reg_seq_{k}"] = ["ROOT_CONFIG.json", dd_]
    exb = {f"{r}/{n}" for r, ns_ in both.items() for n in ns_}
    for k in SEQ:
        assert e6_order_problems(seqjobs[k]) == [], e6_order_problems(seqjobs[k])
        found, _m = reset_plan(seqjobs[k], exb.__contains__, lambda d: both.get(d, []), "s", tol=T)
        other = [x for x in SEQ if x != k][0]
        assert f"{T}/Tracking_v5eq_reg_seq_{k}/ROOT_CONFIG.json" in found and len(found) == 3 and \
            not any(f"_seq_{other}" in f for f in found), (k, found)
    prod = parse_jobs(json.dumps([J("p", "Tol_47_track_v5", 1, "v5", stage="20240422", track_root="Tracking_v5a10")]))
    assert reset_plan(prod, ex.__contains__, lambda d: [], "s", tol=T) == ([], [])     # production roots: never touched
    # change-control revisions of DIRECT roots only (launcher roots are reset in place)
    assert root_base("Tracking_v5a10_r2") == "Tracking_v5a10" and root_base("Tracking_v5a10_pilot_r2") == "Tracking_v5a10_pilot"
    assert root_base("Tracking_v5a100cap21_r12") == "Tracking_v5a100cap21" and root_base("Tracking_v5a10") == "Tracking_v5a10"
    assert root_base("Tracking_v5a10_r1") == "Tracking_v5a10_r1" and root_base("Tracking_v5eq_ref_r2") == "Tracking_v5eq_ref_r2"
    assert relabel_path("Tracking_v5a10_r2", "20240422").endswith("/Metadata/relabel_v5a10_r2/relabel_week_20240422.json")
    Wd = dict(LAUNCHER_WIDGETS); Wd.update(launch_tag="t", jobs_json=json.dumps(good), allow_unpinned="true")
    assert parse_launcher_widgets(Wd)["reset_launcher_roots"] is False
    _expect_raise(lambda: parse_launcher_widgets({**Wd, "reset_launcher_roots": "yes"}), "reset_launcher_roots")
    # tol47b-4 (review r2 finding 3): a payload generated for another launcher revision is refused before anything runs
    assert parse_launcher_widgets({**Wd, "expect_launcher": LAUNCHER_VERSION})["expect_launcher"] == LAUNCHER_VERSION
    _expect_raise(lambda: parse_launcher_widgets({**Wd, "expect_launcher": "tol47b-3 (2026-09-25)"}), "expect_launcher")
    assert source_norm_sha256(b"\xef\xbb\xbf# a\r\n\r\nb  \r\n") == source_norm_sha256("# a\nb\n") != \
        source_norm_sha256("# a\nc\n")
    print("SELFTEST tol47b-3/4 (one mode=equiv job per phase, a failed verdict never skips later phases, launcher-only "
          "roots refused / reset WHOLE (ROOT_CONFIG always, stale day dirs + telemetry), the two sequential payloads on "
          "their own roots, production roots untouched, direct-root revisions <root>_r<N>, expect_launcher, "
          "source_norm): OK")

def run_selftest():
    preflight(); print("SELFTEST preflight (registries 37/17, presets, wsuf, gpu_class): OK")
    _selftest_v2()
    _selftest_parsing()
    _selftest_probe()
    t03 = _selftest_t03_contract()
    _selftest_staging(t03)
    _selftest_checker()
    _selftest_core()
    print("SELFTEST PASSED")

def _cli(argv):
    if "--emit-registry" in argv:
        print(json.dumps({"TOL47_WIDGETS": TOL47_WIDGETS, "TOL03_WIDGETS": TOL03_WIDGETS,
                          "LAUNCHER_WIDGETS": LAUNCHER_WIDGETS, "PRESETS": PRESETS, "ROOT_EXEMPT": ROOT_EXEMPT,
                          "REQUIRED_RUN_KEYS": REQUIRED_RUN_KEYS, "T03_SHA256": T03_SHA256,
                          "DOWNSTREAM_WIDGETS": DOWNSTREAM_WIDGETS, "NB_DIR": NB_DIR, "ALLOC_CONF": ALLOC_CONF,
                          "LAUNCHER_VERSION": LAUNCHER_VERSION, "TOL": TOL, "RELABEL_DIR_FMT": RELABEL_DIR_FMT,
                          "EXPECTED_CLIPS": EXPECTED_CLIPS, "PRODUCTION_ROOTS": PRODUCTION_ROOTS,
                          "ROOT_POLICY": ROOT_POLICY, "ROOT_REV_PATTERN": ROOT_REV_RX.pattern,
                          "SCRIPTS": {k: {kk: vv for kk, vv in v.items() if kk != "widgets"} for k, v in SCRIPTS.items()}},
                         indent=1))
        return 0
    if "--check-payloads" in argv:
        d = argv[argv.index("--check-payloads") + 1]
        nbs = argv[argv.index("--notebooks") + 1] if "--notebooks" in argv else osp.dirname(_t03_path())
        t47 = argv[argv.index("--tol47") + 1] if "--tol47" in argv else osp.join(nbs, "Tol_47_track_v5.py")
        core = argv[argv.index("--core") + 1] if "--core" in argv else osp.join(nbs, "tol_v5_core.py")
        errs, warns, rep = check_payload_dir(d, strict="--strict" in argv, notebooks=nbs, tol47=t47, core=core)
        for r in rep: print("  " + r)
        for w_ in warns: print("WARN  " + w_)
        for e in errs: print("ERROR " + e)
        print(f"payload check: {'PASS' if not errs else 'FAIL'} ({len(errs)} errors, {len(warns)} warnings)")
        return 1 if errs else 0
    run_selftest()
    return 0

if SELFTEST_ONLY:
    sys.exit(_cli(sys.argv[1:]))
if not IS_NOTEBOOK:
    raise SystemExit("Tol_47b_launcher runs on Databricks; locally use TOL_SELFTEST=1 (see the header)")
preflight()
print(f"{LAUNCHER_VERSION}: preflight OK")

# COMMAND ----------

# MAGIC %md ## 2 · Widgets

# COMMAND ----------

for _n, _d in LAUNCHER_WIDGETS:
    dbutils.widgets.text(_n, _d)
W = {_n: (dbutils.widgets.get(_n) or "").strip() for _n, _ in LAUNCHER_WIDGETS}
for _n in ("launch_tag", "expect_gpu", "stop_on_fail", "retry_policy", "code_dir", "expect_launcher"):
    print(f"  widget {_n:14s} = {W[_n]!r}")

# COMMAND ----------

# MAGIC %md ## 3 · Launch (one cell: every outcome is recorded in `attempts.json`)

# COMMAND ----------

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

WORKSPACE_FUSE = "/Workspace"

def _self_source():
    """What notebook is RUNNING (review r2 finding 3): its transport-normalised sha256, read the way Tol_47 reads its
    own source (Tol_47 _source_bytes: the workspace FUSE in the SOURCE format, else a SOURCE export with the Databricks
    SDK). Never raises: {source_norm, basis} or {source_norm: None, basis: <why not>}. gate_v5 compares it with
    source_norm_sha256 of the reviewed local Tol_47b_launcher.py."""
    try:
        nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    except Exception as e:
        return {"source_norm": None, "basis": f"notebook path unreadable: {e!r}"[:300]}
    errs = []
    for cand in (WORKSPACE_FUSE + nbp, WORKSPACE_FUSE + nbp + ".py"):
        try:
            if osp.isfile(cand):
                with open(cand, "rb") as f: b = f.read()
                if b.lstrip(b"\xef\xbb\xbf").startswith(b"# Databricks notebook source"):
                    return {"source_norm": source_norm_sha256(b), "basis": "fuse:" + cand}
                errs.append(f"{cand}: not the SOURCE format")
        except Exception as e:
            errs.append(f"{cand}: {e!r}")
    try:
        import base64
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.workspace import ExportFormat
        b = base64.b64decode(WorkspaceClient().workspace.export(nbp, format=ExportFormat.SOURCE).content)
        return {"source_norm": source_norm_sha256(b), "basis": "sdk_export:" + nbp}
    except Exception as e:
        errs.append(f"sdk export: {e!r}")
    return {"source_norm": None, "basis": "; ".join(errs)[:600]}

def _run_context():
    """Only the id tags — the context JSON may carry credentials, so it is never printed or stored whole."""
    out = {}
    try:
        tags = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()).get("tags", {})
        for k in ("jobId", "runId", "multitaskParentRunId", "taskKey", "clusterId", "jobRunId", "idInJob"):
            if tags.get(k): out[k] = str(tags[k])
    except Exception:
        pass
    cid = out.get("clusterId") or os.environ.get("DB_CLUSTER_ID")
    if not cid:
        try:
            cid = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")   # noqa: F821
        except Exception:
            cid = None
    out["cluster_id"] = cid
    out["parent_run"] = out.get("multitaskParentRunId") or out.get("jobRunId")
    return out

def _write_json_vol(obj, vol_path, local_path):
    with open(local_path, "w") as f: json.dump(obj, f, indent=1, default=str)
    os.makedirs(osp.dirname(vol_path), exist_ok=True)
    shutil.copyfile(local_path, vol_path)            # sequential write: allowed on UC Volumes (no append/seek)

def _rss_gib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"): return int(line.split()[1]) / 2**20
    except Exception:
        return None
    return None

def _smi_query():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        used, total, util = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return used, total, util
    except Exception:
        return None

def _tail(path, n=1):
    try:
        with open(path, errors="replace") as f:
            lines = [l.rstrip() for l in f.readlines() if l.strip()]
        return lines[-n:] if lines else ["(no output yet)"]
    except Exception as e:
        return [f"(log unreadable: {e})"]

def launch(W):
    tag = W.get("launch_tag", "")
    if not NAME_RX.match(tag): raise ConfigError(f"launch_tag {tag!r} must match {NAME_RX.pattern}")
    LOGV = pp.join(TOL, "Launcher_logs", tag)
    LOC = pp.join(LOCAL_ROOT, tag)
    SVC, WLOG, CODE = pp.join(LOC, "svc"), pp.join(LOC, "logs"), pp.join(LOC, "code")
    shutil.rmtree(LOC, ignore_errors=True)
    for d in (SVC, WLOG, CODE): os.makedirs(d, exist_ok=True)
    os.makedirs(LOGV, exist_ok=True)
    ctx = _run_context()
    att_path = pp.join(LOGV, "attempts.json")
    prev = []
    if osp.exists(att_path):
        try:
            prev = json.load(open(att_path))
        except Exception as e:
            print(f"⚠ attempts.json unreadable ({e}) — treating as empty")
    me = {"attempt": len(prev) + 1, "t_start": _now(), "cluster_id": ctx.get("cluster_id"),
          "parent_run": ctx.get("parent_run"), "task_run": ctx.get("runId"), "outcome": "started",
          "launcher": LAUNCHER_VERSION}
    attempts = prev + [me]
    save_attempts = lambda: _write_json_vol(attempts, att_path, pp.join(LOC, "attempts.json"))
    save_attempts()
    print(f"attempt {me['attempt']} | cluster {me['cluster_id']} | parent run {me['parent_run']} | task run {me['task_run']}")
    summary = {"launch_tag": tag, "launcher": LAUNCHER_VERSION, "launcher_source": _self_source(),
               "context": {k: v for k, v in ctx.items()},
               "attempt": me["attempt"], "t_start": me["t_start"], "phases": [], "staging": [],
               "gpu_samples": {"peak_used_mib": None, "total_mib": None}}
    me["launcher_source_norm"] = summary["launcher_source"]["source_norm"]
    print(f"{LAUNCHER_VERSION}: running notebook source_norm {summary['launcher_source']['source_norm']} "
          f"({summary['launcher_source']['basis'][:120]})")
    summ_path = pp.join(LOGV, "summary.json")
    save_summary = lambda: _write_json_vol(summary, summ_path, pp.join(LOC, "summary.json"))
    try:
        gate = attempt_gate(prev, me["cluster_id"], me["parent_run"], W.get("retry_policy", "probe_only"))
        if gate:
            me["outcome"] = "refused"; raise ConfigError(gate)
        # ---- config ----------------------------------------------------------------------------------------------
        try:
            cfg = parse_launcher_widgets(W)
        except ConfigError:
            me["outcome"] = "config_error"; raise
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)
        if os.environ["PYTORCH_CUDA_ALLOC_CONF"] != ALLOC_CONF:
            me["outcome"] = "config_error"
            raise ConfigError(f"PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']!r} — the job's "
                              f"spark_env_vars must set {ALLOC_CONF}.")
        if cfg["pin_problems"]: print("⚠⚠ UNPINNED worker files (allow_unpinned=true): " + "; ".join(cfg["pin_problems"]))
        summary["jobs"] = [{k: j[k] for k in ("name", "script", "phase", "preset", "set", "env")} for j in cfg["jobs"]]
        # launcher-only roots are never resumed (review r1 finding 3): refuse BEFORE the probe; move after it succeeds
        _stamp = f"{tag}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        found, moves = reset_plan(cfg["jobs"], osp.exists,
                                  lambda d: os.listdir(d) if osp.isdir(d) else [], _stamp)
        summary["reset"] = {"requested": cfg["reset_launcher_roots"], "found": found, "moves": []}
        if found and not cfg["reset_launcher_roots"]:
            me["outcome"] = "config_error"
            raise ConfigError(f"launcher-only root output(s) already exist: {found} — resuming them would append a second "
                              f"run to events.jsonl / keep the old .done_ markers (E6 arms contaminated) and an old "
                              f"ROOT_CONFIG.json raises after any Tol_47 change. Resubmit with reset_launcher_roots=true "
                              f"(make_v5_payloads.py --reset-launcher-roots <payload>): they are MOVED to "
                              f"Tol/{ATTIC_DIR}/<tag>_<stamp>/, never deleted (root reset)")
        for p, js in cfg["phases"]:
            print(f"phase {p}: " + " ‖ ".join(f"{j['name']} = {j['script']}[{j['preset']}] "
                                              f"{j['env']['track_root']}/{day_dir(j['env'])} mode={j['env'].get('mode', 'track')}"
                                              for j in js))
        # ---- stage + verify code ---------------------------------------------------------------------------------
        src_dir = pp.join(TOL, cfg["code_dir"])
        shas = {}
        for f in cfg["files"]:
            s = pp.join(src_dir, f)
            if not osp.exists(s):
                me["outcome"] = "config_error"
                raise ConfigError(f"{s} missing — upload it: databricks fs cp \"Testing on Tol2/{f}\" "
                                  f"\"dbfs:{s}\" --overwrite")
            shutil.copyfile(s, pp.join(CODE, f))
            shas[f] = sha256_file(pp.join(CODE, f))
            pin = cfg["pins"].get(f, "")
            if HEX64.match(pin) and shas[f] != pin:
                me["outcome"] = "config_error"
                raise ConfigError(f"{f}: staged sha256 {shas[f]} != pin {pin} — the Volume copy is not the reviewed "
                                  f"file (re-upload, or re-pin after review)")
            print(f"  staged {f}: sha256 {shas[f]}  ({'pinned' if HEX64.match(pin) else 'UNPINNED'})")
        _txt = lambda f: open(pp.join(CODE, f), encoding="utf-8").read()
        for f in cfg["files"]:
            if f in SUPPORT_FILES:
                cp_ = contract_problems(f, _txt(f))
            else:
                key = next(k for k, v in SCRIPTS.items() if v["file"] == f)
                cp_ = contract_problems(key, _txt(f), "\n".join(_txt(x) for x in SCRIPTS[key]["needs"]))
            if cp_:
                me["outcome"] = "config_error"; raise ConfigError(f"{f}: contract: " + "; ".join(cp_))
        summary["code"] = {"dir": src_dir, "sha256": shas, "pins": cfg["pins"]}
        # ---- CUDA + environment probe (subprocess) ---------------------------------------------------------------
        penv = {k: v for k, v in os.environ.items() if not k.startswith(("TOLV4_", "TOLV5_", "TOL_"))}
        penv["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF
        def _probe_cmd():
            r = subprocess.run([sys.executable, "-c", PROBE_SRC], env=penv, capture_output=True, text=True,
                               timeout=900)
            return r.returncode, r.stdout, r.stderr
        try:
            probe, tries = run_cuda_probe(_probe_cmd, cfg["cuda_probe_retries"], cfg["cuda_probe_sleep_s"], print)
        except CudaProbeError:
            me["outcome"] = "probe_failed"; raise
        except ConfigError:
            me["outcome"] = "config_error"; raise
        finally:
            try:
                r = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=60)
                summary["nvidia_smi"] = smi_lines(r.stdout)
                for l in summary["nvidia_smi"]: print("  nvidia-smi | " + l)
            except Exception as e:
                summary["nvidia_smi"] = [f"nvidia-smi failed: {e}"]
        gcls = gpu_class(probe.get("name"))
        summary["probe"] = {**probe, "tries": tries, "gpu_class": gcls}
        me["probe_ok"] = True; me["gpu"] = probe.get("name")
        print(f"GPU {probe.get('name')} ({gcls}, cc {probe.get('cc')}, {probe.get('total_gib')} GiB, free "
              f"{probe.get('free_gib')} GiB) | torch {probe['runtime'].get('torch')} | transformers "
              f"{probe['runtime'].get('transformers')} | rfdetr {probe['dist'].get('rfdetr')} | cv2 "
              f"{probe['runtime'].get('cv2')} | timm {probe['runtime'].get('timm')} | numpy {probe['runtime'].get('numpy')}")
        if cfg["expect_gpu"] and gcls != cfg["expect_gpu"]:
            me["outcome"] = "config_error"
            raise ConfigError(f"payload expects a {cfg['expect_gpu']} but the node has {probe.get('name')}")
        save_attempts(); save_summary()
        # ---- reset (reset_launcher_roots=true only; the probe passed, so a probe-failed attempt changes nothing) ----
        for src, dst in moves:
            os.makedirs(osp.dirname(dst), exist_ok=True)
            dbutils.fs.mv(src, dst, True)                        # a move within the Volume: never a delete
            for _ in range(30):                                  # the /Volumes FUSE view may lag the Files API briefly
                if not osp.exists(src) and osp.exists(dst): break
                time.sleep(2)
            else:
                raise RuntimeError(f"reset: after the move {src} -> {dst} the source still exists or the destination "
                                   f"is missing (60 s)")
            summary["reset"]["moves"].append({"src": src, "dst": dst})
            print(f"  reset: moved {src} -> {dst}")
        if moves:
            me["reset_moves"] = len(moves); save_attempts(); save_summary()
        # ---- run -------------------------------------------------------------------------------------------------
        HF = dbutils.secrets.get(scope="hf", key="token").strip().strip('"').strip("'").strip()
        records, svc_dirs, gpu_peak = summary["staging"], {}, {"used": 0.0, "total": None, "t": 0.0}
        def copy_fn(vname, dst):
            dbutils.fs.cp(f"{DBFS_VIDEO}/{vname}", "file:" + dst)
        def spawn(j):
            lk = pp.join("/local_disk0/tmp", local_key(j))
            if osp.exists(lk):
                shutil.rmtree(lk, ignore_errors=True)           # stale scratch of an earlier phase on this node
                print(f"  cleared stale local scratch {lk}")
            d = pp.join(SVC, j["name"]); os.makedirs(d, exist_ok=True); svc_dirs[j["name"]] = d
            env = build_env(dict(os.environ), j, d, CODE, HF)
            lf = open(pp.join(WLOG, j["name"] + ".log"), "w")
            proc = subprocess.Popen([sys.executable, pp.join(CODE, SCRIPTS[j["script"]]["file"])], env=env,
                                    stdout=lf, stderr=subprocess.STDOUT, cwd=CODE)
            print(f"spawned {j['name']} ({j['script']}[{j['preset']}]) pid={proc.pid}")
            return {"proc": proc, "logf": lf, "t0": time.time(), "job": j, "peak_rss_gib": 0.0}
        def serve(alive):
            return serve_requests(svc_dirs, alive, copy_fn, "/local_disk0/", print, records)
        core_pin = shas.get("tol_v5_core.py")
        def check_core(name, r):
            """Stop a Tol_47 worker that imported a tol_v5_core other than the staged, pinned one."""
            if r["job"]["script"] != "Tol_47_track_v5" or r.get("core_seen"): return
            try:
                with open(pp.join(WLOG, name + ".log"), errors="replace") as f_:
                    txt = f_.read(1 << 20)                       # the line comes right after the import (log head)
            except Exception:
                return
            got = worker_core_from_log(txt)
            if got is None: return
            r["core_seen"] = {"path": got[0], "sha256": got[1]}
            prob = worker_core_problem(txt, core_pin)
            if prob:
                r["sha_problem"] = prob
                print(f"⚠⚠ {name}: {prob} — terminating it")
                if r["proc"].poll() is None: r["proc"].terminate()
            else:
                print(f"  {name}: imports the staged tol_v5_core ({got[1][:16]}…) from {got[0]}")
        def sample(running):
            if time.time() - gpu_peak["t"] < 30: return
            gpu_peak["t"] = time.time()
            q = _smi_query()
            if q:
                gpu_peak["used"] = max(gpu_peak["used"], q[0]); gpu_peak["total"] = q[1]; gpu_peak["util"] = q[2]
            for name, r in running.items():
                check_core(name, r)
                if r["proc"].poll() is None:
                    g = _rss_gib(r["proc"].pid)
                    if g:
                        r["peak_rss_gib"] = max(r["peak_rss_gib"], g)
                        r.setdefault("rss_first", (time.time(), g)); r["rss_last"] = (time.time(), g)
        def track_dir(j):
            return pp.join(TOL, j["env"]["track_root"], day_dir(j["env"]))
        def status(running):
            el = (time.time() - T0) / 3600
            print(f"[{el:5.2f}h] GPU used {gpu_peak.get('used', 0):.0f}/{gpu_peak.get('total') or 0:.0f} MiB (peak), "
                  f"util {gpu_peak.get('util', '?')}%")
            for name, r in running.items():
                st = "alive" if r["proc"].poll() is None else f"EXITED rc={r['proc'].returncode}"
                nd = len(glob.glob(pp.join(track_dir(r["job"]), ".done_*")))
                print(f"   {name:12s} {st:14s} done={nd:3d} rss<= {r['peak_rss_gib']:.1f} GiB | "
                      f"{_tail(pp.join(WLOG, name + '.log'))[0][-150:]}")
        def sync(running):
            for name in running:
                try:
                    shutil.copyfile(pp.join(WLOG, name + ".log"), pp.join(LOGV, name + ".log"))
                except Exception as e:
                    print(f"  log sync failed for {name}: {e}")
            summary["gpu_samples"] = {"peak_used_mib": gpu_peak["used"], "total_mib": gpu_peak["total"]}
            save_summary()
        T0 = time.time()
        failed, skipped = {}, []
        summary["equiv_outputs"], summary["worker_core"] = {}, {}
        EQ_GLOB = pp.join(TOL, "Metrics_v5_equiv_*.json")
        def _read_json(p_):
            with open(p_) as f_: return json.load(f_)
        for p, js in cfg["phases"]:
            ph = {"phase": p, "jobs": [j["name"] for j in js], "t_start": _now()}
            summary["phases"].append(ph)
            blocking = blocking_failures(failed, cfg["jobs"])
            if blocking and cfg["stop_on_fail"]:
                ph["skipped"] = True; skipped += [j["name"] for j in js]
                print(f"phase {p} SKIPPED (stop_on_fail; failed: {sorted(blocking)})"); continue
            if failed:
                print(f"phase {p} runs although {sorted(failed)} failed: a failed E6 verdict is a read-only comparison "
                      f"(review r1 finding 7); the launch still ends job_failed")
            eq_before = set(glob.glob(EQ_GLOB))
            res = run_phase(js, spawn, serve, sample, status, sync, stagger_s=cfg["stagger_sec"], poll_s=5.0,
                            status_every=cfg["status_every_sec"], sync_every=cfg["log_sync_every_sec"])
            ph["t_end"] = _now(); ph["results"] = {}
            eq_new = set(glob.glob(EQ_GLOB)) - eq_before
            for name, r in res.items():
                check_core(name, r)                              # a short job may finish between two samples
                rs = None
                if r.get("rss_first") and r.get("rss_last") and r["rss_last"][0] - r["rss_first"][0] > 600:
                    rs = round((r["rss_last"][1] - r["rss_first"][1]) / ((r["rss_last"][0] - r["rss_first"][0]) / 3600), 3)
                ph["results"][name] = {"rc": r["rc"], "hours": round((r["t1"] - r["t0"]) / 3600, 3),
                                       "peak_rss_gib": round(r["peak_rss_gib"], 2), "rss_gib_per_h": rs,
                                       "track_dir": track_dir(r["job"]),
                                       "done_markers": len(glob.glob(pp.join(track_dir(r["job"]), ".done_*")))}
                if r["job"]["script"] == "Tol_47_track_v5":
                    summary["worker_core"][name] = r.get("core_seen") or "not seen (worker died before the import?)"
                if r["rc"] != 0: failed[name] = r["rc"]
                if r.get("sha_problem"):
                    failed[name] = "core_sha_mismatch"; ph["results"][name]["sha_problem"] = r["sha_problem"]
                if r["job"]["env"].get("mode") == "equiv":       # attribute + tag the verdict file
                    try:
                        ltxt = open(pp.join(WLOG, name + ".log"), errors="replace").read()
                    except Exception:
                        ltxt = ""
                    src, how, probs = equiv_source(name, ltxt, eq_new, _read_json)
                    rec = {"source": src, "matched_by": how, "problems": probs}
                    if src:
                        dst = pp.join(TOL, equiv_copy_name(gcls, tag, name))
                        try:
                            shutil.copyfile(src, dst)
                            J = _read_json(dst)
                            rec.update({"copy": dst, "sha256": sha256_file(dst), "tests": J.get("tests"),
                                        "pass": J.get("pass"), "gpu_class": J.get("gpu_class"),
                                        "echo_shas": (J.get("echo") or {}).get("shas")})
                            es = rec["echo_shas"] or {}
                            for f_, k_ in (("tol_v5_core.py", "core"), ("Tol_47_track_v5.py", "source")):
                                if shas.get(f_) and es.get(k_) and es[k_] != shas[f_]:
                                    probs.append(f"verdict echo sha {k_} {es[k_]} != staged {f_} {shas[f_]}")
                        except Exception as e:
                            probs.append(f"copy/read failed: {type(e).__name__}: {e}")
                    summary["equiv_outputs"][name] = rec
                    print(f"  {name}: verdict file {src} -> {rec.get('copy')} (pass={rec.get('pass')}, by {how})"
                          + (f"  ⚠ {probs}" if probs else ""))
                    if probs and name not in failed: failed[name] = "equiv_output_unattributed"
            summary["failed"] = dict(failed)                     # visible to mon_v5 while later phases run (finding 9)
            sync(res)
            print(f"phase {p} finished: " + ", ".join(f"{n} rc={v['rc']} ({v['hours']:.2f} h)"
                                                      for n, v in ph["results"].items()))
        # ---- logs next to outputs --------------------------------------------------------------------------------
        for j in cfg["jobs"]:
            lp = pp.join(WLOG, j["name"] + ".log")
            if not osp.exists(lp): continue
            try:
                shutil.copyfile(lp, pp.join(LOGV, j["name"] + ".log"))
                if j["env"].get("mode", "track") == "track" and osp.isdir(track_dir(j)):
                    shutil.copyfile(lp, pp.join(track_dir(j), f"worker_log_{j['name']}.txt"))
            except Exception as e:
                print(f"log copy failed for {j['name']}: {e}")
            print(f"\n===== tail {j['name']} =====")
            for l in _tail(lp, 25): print("  " + l)
        summary.update(failed=failed, skipped=skipped, t_end=_now(),
                       gpu_samples={"peak_used_mib": gpu_peak["used"], "total_mib": gpu_peak["total"]})
        if failed or skipped:
            me["outcome"] = "job_failed"
            raise JobsFailed(f"job(s) failed: {failed}; skipped: {skipped} — read {LOGV}/<job>.log. Tracking jobs "
                             f"resume at clip granularity (.done_ markers) when resubmitted.")
        me["outcome"] = "ok"
        print(f"\n✓ all {len(cfg['jobs'])} jobs finished clean — summary {summ_path}")
    except BaseException as e:
        if me["outcome"] in ("started",):
            me["outcome"] = "interrupted" if isinstance(e, KeyboardInterrupt) else "launcher_error"
        me["error"] = f"{type(e).__name__}: {e}"[:800]
        raise
    finally:
        me["t_end"] = _now()
        summary["outcome"] = me["outcome"]
        try:
            save_attempts(); save_summary()
        except Exception as e2:
            print(f"⚠ could not write attempts/summary: {e2}")

# COMMAND ----------

launch(W)
