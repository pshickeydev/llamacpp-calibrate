#!/usr/bin/env python3
"""Per-artifact llama.cpp batch/ubatch calibration, cockpit methodology.

Sweeps (-b, -ub) candidate pairs for ONE exact GGUF with ONE exact engine and
records the winner together with the complete launch identity, mirroring the
rules in ai-toolbox-cockpit/docs/LLAMA_CPP_UBATCH_CALIBRATIONS.md:

  * measure only the exact artifact (filename), never borrow across quants;
  * a baseline calibration is never reused as an mtp/dspark calibration;
  * every stored number carries provenance (engine version, timestamp, pair,
    KV-cache type, flash-attention state, backend, context points).

Runs `llama-bench -o jsonl` once per candidate pair at two context points
(short prefill+decode and long prefill+decode, the C1 two-point protocol),
aggregates medians, ranks pairs, and writes a calibration JSON file that is
resumable (finished pairs are skipped unless --force).

Typical use on a Vulkan host. llama-bench is auto-discovered under
~/.cache/lemonade, /var/cache/lemonade, /opt/var/lib/lemonade; find
the model file with:
    find /var/cache/lemonade /opt/var/lib/lemonade -name '*.gguf'
then:
    python3 calibrate.py \
        --model <path from find, exact filename> \
        --backend-label vulkan --device Vulkan0 \
        --ngl 99 --fa on --kv f16 \
        --output <model>-vulkan.json

Then pin the winner in lemonade:

    lemonade run <model> --llamacpp-device Vulkan0 \
        --llamacpp-args "-b <B> -ub <UB> -fa on"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

# JSON-derived shapes. JsonDict is any parsed JSON object; State is the
# on-disk calibration file ({results: {key: PairResult}, summary: ...}).
JsonDict = dict[str, Any]
State = dict[str, Any]


class ParsedRow(TypedDict):
    """One normalized llama-bench measurement row."""

    test: str          # e.g. "pp512", "tg128", "pg2048,32"
    kind: str          # "pp" | "tg" | "pg"
    n: int | None
    tps: float


class PairResult(TypedDict, total=False):
    """One (batch, ubatch) entry in the calibration record.

    total=False: failed runs omit the metric fields; only the fields the
    writer always sets are required if you pass require_all.
    """

    batch_size: int
    ubatch_size: int
    identity: JsonDict
    timestamp: str
    command: str
    status: str                  # "calibrated" | "failed"
    error: str
    median_tps: dict[str, float]
    median_pp: float
    median_tg: float
    score: float
    engine_echo: JsonDict
    echo_mismatch: str


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

# Two context points: short (zero-history proxy) and long (deep-context
# attention cost). Format for llama-bench -pg: "PP,TG;PP,TG;..."
DEFAULT_POINTS: str = "512,32;16384,32"

# ---------------------------------------------------------------- discovery


def discover_llama_bench(explicit: str | None) -> Path:
    """Find a llama-bench binary: explicit path, PATH, or lemonade cache."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            sys.exit(f"error: --llama-bench not found: {p}")
        return p.resolve()

    which = shutil.which("llama-bench")
    if which:
        return Path(which).resolve()

    roots: list[Path] = [
        Path.home() / ".cache/lemonade",
        Path.home() / ".lemonade",
        # system-wide lemonade-server installs, e.g. /var/cache/
        # lemonade/bin/llamacpp/<backend>/llama-bench with models under
        # /opt/var/lib/lemonade/.cache
        Path("/var/cache/lemonade"),
        Path("/opt/var/lib/lemonade"),
        Path("/opt/lemonade"),
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.is_dir():
            candidates.extend(sorted(root.rglob("llama-bench")))
    # Prefer arch-specific (lemonade nightly) builds over generic ones.
    candidates.sort(key=lambda p: (0 if "gfx" in str(p).lower() else 1, str(p)))
    if not candidates:
        sys.exit(
            "error: could not find llama-bench. Install it "
            "(lemonade backends install llamacpp:<backend>) or pass "
            "--llama-bench /path/from: find ~/.cache/lemonade "
            "/var/cache/lemonade -name llama-bench"
        )
    if len(candidates) > 1:
        print("note: multiple llama-bench builds found:", file=sys.stderr)
        for c in candidates:
            print(f"  {c}", file=sys.stderr)
        print("using the first; override with --llama-bench", file=sys.stderr)
    return candidates[0].resolve()


def engine_version(llama_bench: Path) -> str:
    try:
        out = subprocess.run(
            [str(llama_bench), "--version"],
            capture_output=True, text=True, timeout=15,
        )
        text = (out.stdout or "") + (out.stderr or "")
        m = re.search(r"version:\s*(\S+)\s*(\([^)]*\))?", text)
        if m:
            return f"{m.group(1)} {m.group(2) or ''}".strip()
        for line in text.strip().splitlines():
            if not line.startswith(("load_backend:", "ggml_vulkan:")):
                return line
        return "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- launching

def build_command(
    llama_bench: Path, model: Path, batch: int,
    ubatch: int,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        str(llama_bench),
        "-m", str(model),
        "-ngl", str(args.ngl),
        "-b", str(batch),
        "-ub", str(ubatch),
        # this build takes one pp,tg per -pg flag; repeat it per point pair
        *[x for pt in args.points.split(";") if pt for x in ("-pg", pt)],
        # this build adds -pg on top of the default -p 512/-n 128 tests;
        # suppress the default pp test but keep one shallow pure-decode row
        # (tg32) so median_tg is real and --score combined is meaningful
        "-p", "0", "-n", "32",
        "-r", str(args.reps),
        "-o", "jsonl",
    ]
    if args.device:
        cmd += ["-dev", args.device]
    if args.fa and args.fa != "auto":
        cmd += ["-fa", args.fa]
    kv = args.kv
    if kv != "default":
        cmd += ["-ctk", kv, "-ctv", kv]
    if args.flash_attn_type:
        cmd += ["--flash-attn", args.flash_attn_type]
    if args.extra:
        cmd += args.extra.split()
    return [c for c in cmd if c is not None]


def run_pair(
    cmd: list[str], timeout: int
) -> tuple[list[JsonDict], str | None]:
    """Run one llama-bench invocation; return (json records, error|None)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [], "timeout"
    except OSError as e:
        return [], f"exec failed: {e}"

    records: list[JsonDict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(rec)

    if proc.returncode != 0 and not records:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return [], (err[-1] if err else f"exit {proc.returncode}")
    if not records:
        return [], "no jsonl records in output"
    return records, None


def parse_record(rec: JsonDict) -> ParsedRow | None:
    """Normalize one llama-bench jsonl row to {test, kind, n, tps}.

    Two schemas exist across llama.cpp builds:
      * classic: {"test": "pp512", "t/s": 123.4}
      * lemonade build (11393): raw metrics, no test/t-s keys — the test is
        identified by n_prompt/n_gen, rate is avg_ts (t/s). Rows with
        n_prompt>0 and n_gen>0 are the combined pg rows; pure rows are
        pp{n} or tg{n}. Rows lacking a usable rate are dropped (None).
    """
    tps: float | str | None = rec.get("t/s", rec.get("tps"))
    test: str | None = rec.get("test")
    n: int | None = rec.get("n")
    if tps is None and "avg_ts" in rec:
        n_prompt, n_gen = rec.get("n_prompt", 0), rec.get("n_gen", 0)
        if n_prompt and n_gen:
            test = f"pg{n_prompt},{n_gen}"
        elif n_prompt:
            test = f"pp{n_prompt}"
        elif n_gen:
            test = f"tg{n_gen}"
        else:
            return None
        tps = rec.get("avg_ts")
    if tps is None or test is None:
        return None
    m = re.match(r"([a-z]+)(\d+)", test)
    kind = m.group(1) if m else test
    return {"test": test, "kind": kind, "n": n, "tps": float(tps)}


# ---------------------------------------------------------------- state i/o


def load_state(path: Path) -> State:
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(path: Path, state: State) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(path)


def pair_key(batch: int, ubatch: int, points: str, reps: int,
             identity: JsonDict) -> str:
    # Resume/skip is only valid for an identical launch identity AND
    # measurement protocol; changing model, backend, serving config, KV
    # type, --points or --reps must not reuse stale entries.
    ident = (f"{identity['model_file']}|{identity['backend']}|"
             f"{identity['serving_config']}|{identity['kv_cache_type']}|"
             f"{identity['flash_attention']}")
    return f"{ident}-b{batch}-ub{ubatch}-pg{points}-r{reps}"


# ---------------------------------------------------------------- ranking


def score(results: dict[str, float], mode: str) -> float:
    """Rank score from per-test median t/s. prefill = prompt-side tests only.

    pg rows (n_prompt+n_gen blended, prefill-dominated) count as pp-family.
    """
    pp = [v for k, v in results.items()
          if k.startswith(("pp", "pg")) and v > 0]
    tg = [v for k, v in results.items() if k.startswith("tg") and v > 0]
    if not pp:
        return 0.0
    gm_pp = statistics.geometric_mean(pp)
    if mode == "prefill":
        return gm_pp
    if mode == "combined" and tg:
        return statistics.geometric_mean([gm_pp, statistics.geometric_mean(tg)])
    return gm_pp


# ---------------------------------------------------------------- main


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="exact GGUF to calibrate")
    ap.add_argument("--llama-bench", help="path (default: auto-discover)")
    ap.add_argument(
        "--pairs",
        help="comma list B:UB (default: %s)"
        % ",".join(f"{b}:{u}" for b, u in DEFAULT_PAIRS),
    )
    ap.add_argument("--points", default=DEFAULT_POINTS,
                    help="llama-bench -pg spec (default: %s)" % DEFAULT_POINTS)
    ap.add_argument("--reps", type=int, default=3, help="repeats per test")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--device", default=None, help="llama-bench -d, e.g. Vulkan0")
    ap.add_argument("--fa", default="auto", choices=["auto", "on", "off"],
                    help="flash attention (default auto = don't pass -fa)")
    ap.add_argument("--flash-attn-type", default=None,
                    help="newer llama-bench --flash-attn <type> (overrides --fa)")
    ap.add_argument("--kv", default="default",
                    choices=["default", "f16", "q8_0", "q4_0", "q4_1", "iq4_nl"],
                    help="KV-cache type for -ctk/-ctv (identity-relevant!)")
    ap.add_argument("--extra", default="",
                    help="extra raw llama-bench args (quoted), e.g. spec flags")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-pair timeout seconds (default 1800)")
    ap.add_argument("--score", default="combined", choices=["prefill", "combined"],
                    help="ranking: geometric mean of pp tests, or pp+tg")
    ap.add_argument("--backend-label", required=True,
                    help="engine identity, e.g. vulkan-radv or rocm-nightly-b1260")
    ap.add_argument("--serving-config", default="baseline",
                    choices=["baseline", "mtp", "dspark"],
                    help="never mix calibrations across these")
    ap.add_argument("--output", required=True, help="calibration JSON path")
    ap.add_argument("--force", action="store_true",
                    help="re-run pairs already in --output")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands, run nothing")
    args = ap.parse_args()

    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        sys.exit(f"error: model not found: {model}")

    pairs = parse_pairs(args.pairs) if args.pairs else list(DEFAULT_PAIRS)
    llama_bench = discover_llama_bench(args.llama_bench)
    out_path = Path(args.output).expanduser()
    state = load_state(out_path)

    identity: JsonDict = {
        "model_file": model.name,          # cockpit matches on filename pattern
        "model_path": str(model),
        "backend": args.backend_label,     # cockpit: toolbox_id equivalent
        "serving_config": args.serving_config,
        "kv_cache_type": args.kv,
        "flash_attention": args.flash_attn_type or args.fa,
        "ngl": args.ngl,
        "points": args.points,
        "reps": args.reps,
        "engine_version": "pending",
    }

    if args.dry_run:
        for b, ub in pairs:
            print(" ".join(
                build_command(llama_bench, model, b, ub, args)
            ))
        return

    identity["engine_version"] = engine_version(llama_bench)
    print(f"llama-bench : {llama_bench}")
    print(f"engine      : {identity['engine_version']}")
    print(f"model       : {model.name}")
    print(f"identity    : backend={args.backend_label} "
          f"serving={args.serving_config} kv={args.kv} "
          f"fa={identity['flash_attention']}")
    print(f"candidates  : {len(pairs)} pairs, {args.reps} reps, points={args.points}")
    print("-" * 64)

    for b, ub in pairs:
        key = pair_key(b, ub, args.points, args.reps, identity)
        prev = state.get("results", {}).get(key)
        if prev and not args.force:
            if prev.get("status") == "calibrated":
                print(f"[skip] b={b} ub={ub} already calibrated "
                      f"(median pp {prev['median_pp']:.1f} t/s); --force to redo")
                continue

        cmd = build_command(llama_bench, model, b, ub, args)
        print(f"[ run] b={b} ub={ub} ...", flush=True)
        records, err = run_pair(cmd, args.timeout)

        entry: PairResult = {
            "batch_size": b,
            "ubatch_size": ub,
            "identity": identity,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "command": " ".join(cmd),
        }
        if err:
            entry["status"] = "failed"
            entry["error"] = err
            print(f"[fail] b={b} ub={ub}: {err}")
        else:
            parsed: list[ParsedRow] = [
                p for p in (parse_record(r) for r in records) if p]
            # cross-check the engine's echoed config against our identity;
            # a silently ignored flag would poison the calibration
            if records:
                echo = records[0]
                entry["engine_echo"] = {
                    k: echo.get(k) for k in
                    ("flash_attn", "type_k", "type_v", "n_gpu_layers",
                     "devices", "build_commit")
                }
                want_fa = {"on": 1, "off": 0}.get(args.fa)
                got_fa = echo.get("flash_attn")
                if want_fa is not None and got_fa not in (None, want_fa):
                    entry["echo_mismatch"] = (
                        f"requested -fa {args.fa} but engine ran {got_fa}")
                if args.kv != "default" and (
                        echo.get("type_k") != args.kv
                        or echo.get("type_v") != args.kv):
                    entry["echo_mismatch"] = (
                        f"requested -ctk/-ctv {args.kv} but engine ran "
                        f"{echo.get('type_k')}/{echo.get('type_v')}")
                if entry.get("echo_mismatch"):
                    entry["status"] = "failed"
                    entry["error"] = entry["echo_mismatch"]
                    print(f"[fail] b={b} ub={ub}: {entry['error']}")
                    state.setdefault("results", {})[key] = entry
                    save_state(out_path, state)
                    continue
            per_test: dict[str, list[float]] = {}
            for p in parsed:
                per_test.setdefault(p["test"], []).append(p["tps"])
            medians = {t: statistics.median(v) for t, v in per_test.items()}
            entry["status"] = "calibrated"
            entry["median_tps"] = {t: round(v, 2) for t, v in sorted(medians.items())}
            def _gm(*prefixes: str) -> float:
                vals = [v for t, v in medians.items()
                        if t.startswith(prefixes) and v > 0]
                return statistics.geometric_mean(vals) if vals else 0.0
            entry["median_pp"] = _gm("pp", "pg")
            entry["median_tg"] = _gm("tg")
            entry["score"] = round(score(medians, args.score), 2)
            breakdown = "  ".join(f"{t}={v:.1f}" for t, v in sorted(medians.items()))
            print(f"[ ok ] b={b} ub={ub}: {breakdown}  score={entry['score']}")

        state.setdefault("results", {})[key] = entry
        save_state(out_path, state)

    # ---- summary: winner + machine-readable record -------------------------
    done: list[PairResult] = [e for e in state["results"].values()
            if e.get("status") == "calibrated"
            and e.get("identity", {}).get("backend") == args.backend_label
            and e["identity"].get("serving_config") == args.serving_config
            and e["identity"].get("kv_cache_type") == args.kv
            and e["identity"].get("model_file") == model.name
            and e["identity"].get("flash_attention")
            == identity["flash_attention"]
            and e["identity"].get("points") == args.points
            and e["identity"].get("reps") == args.reps]
    if not done:
        sys.exit("no pair calibrated; see failures above")

    winner: PairResult = max(done, key=lambda e: e["score"])
    ranked: list[PairResult] = sorted(done, key=lambda e: -e["score"])
    baseline: PairResult | None = next(
        (e for e in done
         if e["batch_size"] == 2048 and e["ubatch_size"] == 512), None
    )
    gain = ""
    if baseline and baseline["score"] > 0:
        gain = (f"  (+{100 * (winner['score'] / baseline['score'] - 1):.1f}% "
                f"vs llama.cpp default 2048/512)")

    state["summary"] = {
        "best_batch_size": winner["batch_size"],
        "best_ubatch_size": winner["ubatch_size"],
        "identity": identity,
        "llamacpp_args": (
            f"-b {winner['batch_size']} -ub {winner['ubatch_size']}"
            + ("" if args.fa in ("auto",) else f" -fa {args.fa}")
            + (f" --cache-type-k {args.kv} --cache-type-v {args.kv}"
               if args.kv != "default" else "")
        ),
        "ranking": [
            {"batch_size": e["batch_size"], "ubatch_size": e["ubatch_size"],
             "score": e["score"], "median_tps": e["median_tps"]}
            for e in ranked
        ],
    }
    save_state(out_path, state)

    print("=" * 64)
    print(f"WINNER      : -b {winner['batch_size']} -ub "
          f"{winner['ubatch_size']}{gain}")
    print(f"lemonade    : lemonade run <model>"
          f"{' --llamacpp-device ' + args.device if args.device else ''}"
          f" --llamacpp-args \"{state['summary']['llamacpp_args']}\"")
    print(f"record      : {out_path}")
    print("ranked:")
    for i, e in enumerate(ranked[:5], 1):
        print(f"  {i}. b={e['batch_size']:>5} ub={e['ubatch_size']:>5}"
              f"  score={e['score']:>9.1f}  {e['median_tps']}")


if __name__ == "__main__":
    main()
