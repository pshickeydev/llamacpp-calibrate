#!/usr/bin/env python3
"""Per-model llama.cpp batch/ubatch calibration for a lemonade server.

Runs on the machine that HOSTS the lemonade server. For every downloaded
`llamacpp` model it sweeps
(-b, -ub) candidate pairs by driving `lemonade bench --llamacpp-args`
once per pair, scores each pair on prefill throughput at two context
points (short + long, the cockpit C1 two-point protocol), and stores the
winner with full provenance — mirroring the identity rules of
ai-toolbox-cockpit/docs/LLAMA_CPP_UBATCH_CALIBRATIONS.md:

  * results are keyed to the EXACT model id as registered in lemonade;
  * a baseline calibration is never reused as an mtp calibration
    (serving_config is detected from the model's effective args and
    recorded in every entry);
  * every stored number carries provenance: engine version, server
    version, timestamp, pair, scenario points, reps.

Measurement runs through lemonade itself (`lemonade bench`), so the
server's own backend/device selection is what gets calibrated, and no
file access to the model store is needed.

After the sweep, `apply` writes each model's winning batch sizes into
lemonade's per-model options (POST /v1/models/<id>/options), so every
future `lemonade load/run` of that model — from any client on the LAN —
automatically uses the calibrated values. `-b`/`-ub` are merged into the
model's existing effective args (sampling/serving flags preserved).

Usage on the server host (key: same one the `lemonade` CLI uses; the
bench step shells out to `lemonade`, so a working CLI implies a working
script):

    export LEMONADE_API_KEY=<key>          # or pass --api-key
    python3 calctl.py list                 # inventory + identity preview
    python3 calctl.py sweep                # calibrate all llamacpp models
    python3 calctl.py sweep --only <model-substring> --pairs 2048:512,2048:2048
    python3 calctl.py apply                # write winners into lemonade
    python3 calctl.py apply --dry-run
    python3 calctl.py status               # show stored winners

State lives in ./calibrations/<model-id>.json (resumable: finished pairs
are skipped unless --force). A combined ./calibrations/summary.json is
rewritten after every model.

Sizing the sweep: each pair = one bench invocation = one model reload +
(--reps) runs of each point. Cost scales with model size and memory
bandwidth; run one pair first to gauge your hardware, then let the full
grid run unattended.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

# (batch, ubatch) candidates, llama.cpp defaults first so the baseline is
# always measured and the gain over it is reportable.
DEFAULT_PAIRS: list[tuple[int, int]] = [
    (2048, 512),
    (2048, 1024),
    (2048, 2048),
    (4096, 1024),
    (4096, 2048),
    (8192, 1024),
    (8192, 2048),
    (1024, 512),
]

# Two context points (cockpit C1): short = zero-history proxy, long =
# deep-context attention cost. Values are llama-bench-style target_tokens
# for the bench scenario filler.
SHORT_TOKENS = 512
LONG_TOKENS = 16384
GEN_TOKENS = 32

# ---------------------------------------------------------------- server i/o


class Server:
    """Minimal Lemonade Server HTTP client (stdlib only)."""

    def __init__(self, base_url: str, api_key: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, method: str, path: str,
                 body: JsonDict | None = None) -> JsonDict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            sys.exit(f"error: {method} {path} -> HTTP {e.code}: {detail}")
        except urllib.error.URLError as e:
            sys.exit(f"error: {method} {path} -> {e.reason} "
                     f"(is the lemonade server up at {self.base_url}?)")

    def models(self) -> list[JsonDict]:
        return self._request("GET", "/api/v1/models").get("data", [])

    def health(self) -> JsonDict:
        return self._request("GET", "/api/v1/health")

    def get_options(self, model_id: str) -> JsonDict:
        return self._request("GET", f"/api/v1/models/{model_id}/options")

    def save_options(self, model_id: str, options: JsonDict,
                     dry_run: bool = False) -> JsonDict:
        body = dict(options)
        if dry_run:
            body["dry_run"] = True
        return self._request(
            "POST", f"/api/v1/models/{model_id}/options", body)


# ---------------------------------------------------------------- inventory


def calib_models(server: Server) -> list[JsonDict]:
    """Downloaded llamacpp models with their resolved launch identity."""
    out = []
    for m in server.models():
        if m.get("recipe") != "llamacpp" or not m.get("downloaded"):
            continue
        opts = server.get_options(m["id"])
        eff = opts.get("effective", {})
        args = eff.get("llamacpp_args") or ""
        out.append({
            "id": m["id"],
            "checkpoint": m.get("checkpoint", ""),
            "size_gb": m.get("size"),
            "max_ctx": m.get("max_context_window"),
            "resolved_ctx_size": opts.get("resolved_ctx_size"),
            "llamacpp_backend": eff.get("llamacpp_backend"),
            "llamacpp_device": eff.get("llamacpp_device") or "",
            "llamacpp_args": args,
            # cockpit rule: baseline and mtp calibrations never mix
            "serving_config": "mtp" if "spec-type" in args else "baseline",
        })
    return out


# ---------------------------------------------------------------- scenarios


def make_scenarios(long_tokens: int) -> JsonDict:
    """Two-point bench scenario file: short prefill+decode, long prefill+decode."""
    filler = ("The history of the Roman Empire is long and complex, and "
              "no two historians agree on where to place its beginning. ")
    def pt(name: str, target: int) -> JsonDict:
        return {
            "name": name,
            "category": "general",
            "messages": [{"role": "user",
                          "content": "Summarize the provided passage in a "
                                     "few sentences."}],
            "max_tokens": GEN_TOKENS,
            "context": {"filler": filler,
                        "target_tokens": target,
                        "position": "end"},
        }
    scenarios = [pt(f"calib-short-{SHORT_TOKENS}", SHORT_TOKENS)]
    if long_tokens >= 2048:
        scenarios.append(pt(f"calib-long-{long_tokens}", long_tokens))
    return {"scenarios": scenarios}


# ---------------------------------------------------------------- bench run


def run_bench(args: argparse.Namespace, model_id: str,
              extra_args: str, scenarios: JsonDict,
              ctx_size: int) -> tuple[JsonDict | None, str | None]:
    """One `lemonade bench` invocation for one (-b,-ub) pair.

    Returns (parsed_json, error|None). Bench reloads the model per args
    set, so each pair is measured on a clean load.
    """
    with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False) as fh:
        json.dump(scenarios, fh)
        scen_path = fh.name
    cmd = [args.bench_bin,
           "--host", args.host, "--port", str(args.port), "--no-discovery"]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    ctx_flag = ["--ctx-size", str(ctx_size)] if ctx_size > 0 else []
    cmd += ["bench", model_id,
            "--llamacpp-args", extra_args,
            *ctx_flag,
            "--scenario-file", scen_path,
            "--runs", str(args.reps), "--warmup", str(args.warmup),
            "--no-memory", "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return None, f"bench timeout after {args.timeout}s"
    except OSError as e:
        return None, f"exec failed: {e}"
    finally:
        os.unlink(scen_path)

    out = proc.stdout or ""
    start = out.find("\n{")
    start = start + 1 if start >= 0 else (0 if out.startswith("{") else -1)
    if start < 0:
        err = (proc.stderr or out or "").strip().splitlines()
        return None, (err[-1] if err else "no JSON in bench output")
    try:
        return json.loads(out[start:]), None
    except json.JSONDecodeError as e:
        return None, f"bad JSON from bench: {e}"


def extract_pair_result(bench: JsonDict, extra_args: str) -> tuple[
        dict[str, JsonDict], str | None]:
    """Pull per-scenario {ttft_ms, tps, input_tokens, output_tokens} rows."""
    try:
        results = bench["models"][0]["results"]
    except (KeyError, IndexError):
        return {}, "bench JSON has no model results"
    run = next((r for r in results
                if (r.get("backend_args") or "") == extra_args), None)
    if run is None:
        run = results[0] if len(results) == 1 else None
    if run is None:
        return {}, f"no bench run for args '{extra_args}'"
    rows: dict[str, JsonDict] = {}
    for sc in run.get("scenarios", []):
        if sc.get("failed_runs"):
            return {}, (f"scenario {sc.get('name')}: "
                        f"{sc['failed_runs']} failed run(s)")
        ttft = sc.get("ttft_ms", {}).get("p50")
        tps = sc.get("tps", {}).get("p50")
        if not ttft or not tps:
            return {}, f"scenario {sc.get('name')}: missing timing"
        rows[sc["name"]] = {
            "ttft_ms": round(ttft, 3),
            "tps": round(tps, 3),
            "input_tokens": sc.get("input_tokens", 0),
            "output_tokens": sc.get("output_tokens", 0),
        }
    if not rows:
        return {}, "bench reported no scenarios"
    return rows, None


def score_rows(rows: dict[str, JsonDict], mode: str) -> float:
    """Rank score: geometric mean of per-point prefill t/s (default).

    Prefill rate = input_tokens / ttft is what -b/-ub actually move;
    decode tps is recorded but excluded from the default score because
    speculative decoding (mtp serving) makes it noisy. `combined` folds
    decode tps in with equal weight per point.
    """
    pp = []
    for r in rows.values():
        if r["input_tokens"] > 0 and r["ttft_ms"] > 0:
            pp.append(r["input_tokens"] / (r["ttft_ms"] / 1000.0))
    if not pp:
        return 0.0
    gm_pp = statistics.geometric_mean(pp)
    if mode == "prefill":
        return gm_pp
    tg = [r["tps"] for r in rows.values() if r["tps"] > 0]
    if mode == "combined" and tg:
        return statistics.geometric_mean(
            [gm_pp, statistics.geometric_mean(tg)])
    return gm_pp


# ---------------------------------------------------------------- state i/o


def state_path(state_dir: Path, model_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
    return state_dir / f"{safe}.json"


def load_state(path: Path) -> JsonDict:
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(path: Path, state: JsonDict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(path)


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- sweep


def sweep_model(server: Server, args: argparse.Namespace, model: JsonDict,
                pairs: list[tuple[int, int]]) -> JsonDict | None:
    path = state_path(Path(args.state_dir), model["id"])
    state = load_state(path)

    long_tokens = min(LONG_TOKENS, (model["resolved_ctx_size"] or LONG_TOKENS)
                      - GEN_TOKENS - 1024)
    scenarios = make_scenarios(long_tokens)
    ctx_size = 0 if args.ctx_auto else min(
        LONG_TOKENS + GEN_TOKENS + 1024,
        model["resolved_ctx_size"] or LONG_TOKENS + GEN_TOKENS + 1024)

    identity = {
        "model_id": model["id"],
        "checkpoint": model["checkpoint"],
        "backend": model["llamacpp_backend"],
        "device": model["llamacpp_device"],
        "serving_config": model["serving_config"],
        "base_args": strip_batch(model["llamacpp_args"]),
        "ctx_size": "auto" if ctx_size <= 0 else ctx_size,
        "long_tokens": long_tokens,
        "short_tokens": SHORT_TOKENS,
        "gen_tokens": GEN_TOKENS,
        "reps": args.reps,
        "score_mode": args.score,
    }

    # protocol fingerprint: stale entries from a different protocol must
    # never satisfy a resume check.
    fp = json.dumps(identity, sort_keys=True)
    if state.get("fingerprint") not in (None, fp):
        print(f"[warn] {model['id']}: protocol changed; previous results "
              f"kept in {path.name} but not reused. Bump --state-dir or "
              f"delete the file to start clean.")
        alt = path.with_suffix(f".{hash(fp) & 0xffff:04x}.prev.json")
        if not alt.exists():
            path.rename(alt)
            print(f"[warn] previous state moved to {alt.name}")
        state = {}
    state.setdefault("fingerprint", fp)
    state["identity"] = identity

    results: JsonDict = state.setdefault("results", {})
    winner: JsonDict | None = None

    base = strip_batch(model["llamacpp_args"])
    for (b, ub) in pairs:
        # /v1/load semantics: a concrete *_args REPLACES the model/
        # architecture scope, so re-send the model's own serving args
        # (sampling, --spec-type draft-mtp) with only -b/-ub swapped —
        # an mtp model must be measured under mtp.
        extra = f"{base} -b {b} -ub {ub}".strip() + (
            f" {args.extra}" if args.extra else "")
        key = extra
        prev = results.get(key)
        if prev and prev.get("status") == "calibrated" and not args.force:
            print(f"  [skip] {extra} already calibrated "
                  f"(score {prev['score']:.0f}); --force to redo")
            continue
        if args.dry_run:
            print(f"  [dry ] lemonade bench {model['id']} "
                  f"--llamacpp-args \"{extra}\""
                  + ("" if ctx_size > 0 else " (ctx auto)")
                  + f" --runs {args.reps}")
            continue

        print(f"  [ run] {extra} ...", flush=True)
        bench, err = run_bench(args, model["id"], extra, scenarios, ctx_size)
        entry: JsonDict = {
            "batch_size": b, "ubatch_size": ub,
            "backend_args": extra,
            "timestamp": now(),
        }
        if err:
            entry.update(status="failed", error=err)
            print(f"  [fail] {extra}: {err}")
        else:
            rows, err = extract_pair_result(bench or {}, extra)
            if err:
                entry.update(status="failed", error=err,
                             raw=bench)
                print(f"  [fail] {extra}: {err}")
            else:
                assert bench is not None
                hw = bench.get("hardware", {})
                entry.update(
                    status="calibrated",
                    scenarios=rows,
                    prefill_tps={
                        name: round(r["input_tokens"] / (r["ttft_ms"] / 1000), 2)
                        for name, r in rows.items()},
                    score=round(score_rows(rows, args.score), 2),
                    engine=hw.get("backends", {}),
                    engine_version=str(hw.get("backends", {}).get(
                        f"llamacpp/{model['llamacpp_backend']}", "unknown")),
                    lemonade_version=hw.get("lemonade_version", "unknown"),
                )
                pp = "  ".join(
                    f"{n.split('-',1)[-1]}={v:.0f}pp"
                    for n, v in entry["prefill_tps"].items())
                print(f"  [ ok ] {extra}: {pp}  score={entry['score']}")
        results[key] = entry
        state["results"] = results
        save_state(path, state)

    calibrated = [e for e in results.values()
                  if e.get("status") == "calibrated"]
    if not calibrated:
        print(f"  !! no pair calibrated for {model['id']}")
        return None

    winner = max(calibrated, key=lambda e: e["score"])
    ranked = sorted(calibrated, key=lambda e: -e["score"])
    baseline = next((e for e in calibrated
                     if e["batch_size"] == 2048 and e["ubatch_size"] == 512),
                    None)
    gain = None
    if baseline and baseline["score"] > 0:
        gain = round(100 * (winner["score"] / baseline["score"] - 1), 1)

    state["summary"] = {
        "best_batch_size": winner["batch_size"],
        "best_ubatch_size": winner["ubatch_size"],
        "score": winner["score"],
        "gain_vs_baseline_pct": gain,
        "llamacpp_args_patch": f"-b {winner['batch_size']} "
                               f"-ub {winner['ubatch_size']}",
        "ranking": [
            {"batch_size": e["batch_size"], "ubatch_size": e["ubatch_size"],
             "score": e["score"]} for e in ranked],
        "updated": now(),
    }
    save_state(path, state)
    write_combined(Path(args.state_dir))

    print(f"  WINNER {model['id']}: -b {winner['batch_size']} -ub "
          f"{winner['ubatch_size']}  score={winner['score']}"
          + (f"  (+{gain}% vs 2048/512)" if gain is not None else ""))
    return state["summary"]


def write_combined(state_dir: Path) -> JsonDict:
    """Merge every per-model summary into summary.json (the deliverable)."""
    combined: JsonDict = {"updated": now(), "models": {}}
    for p in sorted(state_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        st = load_state(p)
        s = st.get("summary")
        ident = st.get("identity", {})
        if not s:
            continue
        combined["models"][ident.get("model_id", p.stem)] = {
            "batch_size": s["best_batch_size"],
            "ubatch_size": s["best_ubatch_size"],
            "llamacpp_args": s["llamacpp_args_patch"],
            "score": s["score"],
            "gain_vs_baseline_pct": s.get("gain_vs_baseline_pct"),
            "serving_config": ident.get("serving_config"),
            "backend": ident.get("backend"),
            "protocol": ident,
            "provenance_file": p.name,
            "updated": s.get("updated"),
        }
    tmp = state_dir / "summary.tmp"
    with open(tmp, "w") as f:
        json.dump(combined, f, indent=2, sort_keys=True)
    tmp.replace(state_dir / "summary.json")
    return combined


# ---------------------------------------------------------------- apply


ARG_B_RE = re.compile(r"\s*(?:-{1,2}(?:b|batch-size)|--batch-size)(?:\s+|\s*=\s*)\d+")
ARG_UB_RE = re.compile(r"\s*(?:-{1,2}ub|-{1,2}ubatch-size)(?:\s+|\s*=\s*)\d+")


def patch_args(base_args: str, b: int, ub: int) -> str:
    """Replace any existing -b/-ub in base args, append calibrated pair."""
    s = ARG_UB_RE.sub("", ARG_B_RE.sub("", base_args or ""))
    s = re.sub(r"\s{2,}", " ", s).strip()
    return f"{s} -b {b} -ub {ub}".strip()


def strip_batch(base_args: str) -> str:
    """Args with any -b/-ub removed: the calibration identity must not
    include the very values we sweep (apply writes them back into the
    effective args, which would poison resume fingerprints)."""
    s = re.sub(r"\s{2,}", " ",
               ARG_UB_RE.sub("", ARG_B_RE.sub("", base_args or ""))).strip()
    return s


def apply_model(server: Server, model_id: str, b: int, ub: int,
                dry_run: bool) -> bool:
    """Write the calibrated pair into saved per-model options.

    A saved llamacpp_args REPLACES the model/architecture args scope at
    load time (per the /v1/load priority chain), so we must persist the
    full effective args with -b/-ub swapped, or sampling/MTP flags from
    the registry would be dropped. patch_args is idempotent, so re-
    applies land on the saved string unchanged.
    """
    opts = server.get_options(model_id)
    eff_args = opts.get("effective", {}).get("llamacpp_args") or ""
    new_args = patch_args(eff_args, b, ub)
    if new_args == eff_args and (opts.get("saved") or {}).get(
            "llamacpp_args") is not None:
        print(f"  [=] {model_id}: saved args already carry "
              f"`-b {b} -ub {ub}`")
        return True
    if new_args == eff_args:
        print(f"  [=] {model_id}: effective args already carry "
              f"`-b {b} -ub {ub}` (registry default); nothing to save")
        return True
    verb = "would save" if dry_run else "saved"
    resp = server.save_options(model_id, {"llamacpp_args": new_args},
                               dry_run=dry_run)
    print(f"  [{'dry' if dry_run else ' ok'}] {model_id}: {verb} "
          f"llamacpp_args = \"{new_args}\"")
    if not dry_run:
        got = (resp.get("saved") or {}).get("llamacpp_args")
        if got != new_args:
            print(f"  !! verify failed: saved={got!r}")
            return False
    return True


# ---------------------------------------------------------------- commands


def cmd_list(server: Server, args: argparse.Namespace,
             models: list[JsonDict]) -> None:
    print(f"{'model id':<42} {'sz GB':>6}  {'serving':<9} {'backend':<7} ctx")
    for m in models:
        if args.only and not _only(m["id"], args.only):
            continue
        print(f"{m['id']:<42} {m['size_gb'] or 0:>6.1f}  "
              f"{m['serving_config']:<9} {m['llamacpp_backend']:<7} "
              f"{m['resolved_ctx_size']}")


def _only(model_id: str, patterns: list[str]) -> bool:
    return any(p.lower() in model_id.lower() for p in patterns)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "sweep", "apply", "status"])
    ap.add_argument("--host", default=os.environ.get("LEMONADE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("LEMONADE_PORT", "13305")))
    ap.add_argument("--api-key",
                    default=os.environ.get("LEMONADE_API_KEY")
                    or os.environ.get("LEMONADE_ADMIN_API_KEY"))
    ap.add_argument("--bench-bin", default="lemonade",
                    help="lemonade CLI used for benching (default: PATH)")
    ap.add_argument("--only", action="append", default=[],
                    help="substring filter on model id (repeatable)")
    ap.add_argument("--state-dir", default="calibrations")
    ap.add_argument("--pairs",
                    help="comma list B:UB (default: %s)"
                    % ",".join(f"{b}:{u}" for b, u in DEFAULT_PAIRS))
    ap.add_argument("--reps", type=int, default=2,
                    help="measurement runs per point per pair (bench --runs)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="warmup runs per point (bench --warmup)")
    ap.add_argument("--score", default="prefill",
                    choices=["prefill", "combined"],
                    help="prefill = geometric mean of pp t/s (default); "
                         "combined folds in decode tps")
    ap.add_argument("--ctx-auto", action="store_true",
                    help="let bench pick ctx (default: explicit "
                         f"{LONG_TOKENS}+{GEN_TOKENS}+1024 capped by model)")
    ap.add_argument("--extra", default="",
                    help="extra llama-server args appended to every pair "
                         "(e.g. kv cache types)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-pair bench timeout seconds (default 3600)")
    ap.add_argument("--force", action="store_true",
                    help="re-measure pairs already in state")
    ap.add_argument("--dry-run", action="store_true",
                    help="sweep: print bench commands; apply: server-side "
                         "dry_run validation, persists nothing")
    args = ap.parse_args()

    if not args.api_key and (args.host, args.port) not in (
            ("127.0.0.1", 13305), ("localhost", 13305)):
        print("note: no --api-key set; unauthenticated requests will be "
              "rejected by a server with auth enabled", file=sys.stderr)

    server = Server(f"http://{args.host}:{args.port}", args.api_key)
    models = calib_models(server)

    if args.command == "list":
        cmd_list(server, args, models)
        return

    selected = [m for m in models if not args.only
                or _only(m["id"], args.only)]
    if not selected:
        sys.exit("error: no downloaded llamacpp models match")

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "sweep":
        pairs = parse_pairs(args.pairs) if args.pairs else list(DEFAULT_PAIRS)
        print(f"engine check: ", end="")
        try:
            proc = subprocess.run([args.bench_bin, "--version"],
                                  capture_output=True, text=True, timeout=15)
            print(((proc.stdout or "") + (proc.stderr or "")).strip()
                  .splitlines()[0])
        except Exception as e:
            if args.dry_run:
                print("lemonade CLI not runnable (dry-run, ok)")
            else:
                sys.exit(f"error: `{args.bench_bin} --version` failed: {e}")
        for i, m in enumerate(selected, 1):
            print(f"\n=== [{i}/{len(selected)}] {m['id']} "
                  f"({m['serving_config']}, {m['llamacpp_backend']}, "
                  f"{m['size_gb']} GB) ===")
            sweep_model(server, args, m, pairs)
        combined = write_combined(state_dir)
        print(f"\ncombined record: {state_dir / 'summary.json'} "
              f"({len(combined['models'])} models with winners)")
        return

    if args.command == "apply":
        ok = True
        for m in selected:
            st = load_state(state_path(state_dir, m["id"]))
            s = st.get("summary")
            if not s:
                print(f"  [ -] {m['id']}: no calibration yet "
                      f"(run sweep first)")
                continue
            ident = st.get("identity", {})
            if ident.get("serving_config") != m["serving_config"]:
                print(f"  [ !!] {m['id']}: calibration was taken under "
                      f"serving_config={ident.get('serving_config')} but "
                      f"model now runs {m['serving_config']}; skipping "
                      f"(re-sweep)")
                continue
            ok &= apply_model(server, m["id"], s["best_batch_size"],
                              s["best_ubatch_size"], args.dry_run)
        sys.exit(0 if ok else 1)

    # status
    for m in selected:
        st = load_state(state_path(state_dir, m["id"]))
        s = st.get("summary")
        if not s:
            print(f"{m['id']:<42} —")
            continue
        eff = server.get_options(m["id"]).get("effective", {})
        applied = (f"-b {s['best_batch_size']} -ub {s['best_ubatch_size']}"
                   in (eff.get("llamacpp_args") or ""))
        print(f"{m['id']:<42} best -b {s['best_batch_size']} "
              f"-ub {s['best_ubatch_size']}  score={s['score']:<9} "
              f"gain={s.get('gain_vs_baseline_pct')}%  "
              f"applied={'yes' if applied else 'NO'}")


def parse_pairs(spec: str) -> list[tuple[int, int]]:
    pairs = []
    for chunk in spec.replace(" ", "").split(","):
        if not chunk:
            continue
        m = re.match(r"^(\d+)[:x](\d+)$", chunk)
        if not m:
            sys.exit(f"error: bad pair '{chunk}' (expected B:UB e.g. 2048:2048)")
        b, ub = int(m.group(1)), int(m.group(2))
        if b < ub:
            sys.exit(f"error: pair {b}:{ub}: -b must be >= -ub")
        pairs.append((b, ub))
    return pairs


if __name__ == "__main__":
    main()
