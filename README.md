# llamacpp-calibrate

Per-model llama.cpp `-b`/`-ub` calibration for a Lemonade server, using the
measurement discipline of
[ai-toolbox-cockpit](https://github.com/kyuz0/ai-toolbox-cockpit)
(`docs/LLAMA_CPP_UBATCH_CALIBRATIONS.md`), narrowed to the single engine a
lemonade install runs.

**Copy `calctl.py` (and optionally this README) to the machine that hosts the
lemonade server and run it there.** `calctl.py` is stdlib-only Python 3.10+;
the only external requirement is a working `lemonade` CLI on PATH (bench runs
go through it).

## How it works

For every downloaded `llamacpp` model:

1. **Identity** — read the model's effective options
   (`GET /v1/models/<id>/options`): backend, device, serving args. A model
   whose args contain `--spec-type` is tagged `mtp`; baseline and mtp
   calibrations are never mixed (cockpit rule), and `apply` refuses to write
   an mtp calibration onto a model that has become baseline (or vice versa).
2. **Sweep** — one `lemonade bench --llamacpp-args "-b B -ub UB"` invocation
   per candidate pair. Bench reloads the model per args set, so every pair is
   measured on a clean load with the server's own backend selection.
   Measurement points (cockpit C1 two-point protocol): short prefill
   (512-token context) + long prefill (16K tokens, capped by the model's
   resolved context), 32 generated tokens each.
3. **Score** — geometric mean of per-point prefill throughput
   (`input_tokens / ttft`), which is what `-b`/`-ub` actually move. Decode
   tps is stored for reference; `--score combined` folds it in.
4. **Apply** — the winner is written into lemonade's per-model options
   (`POST /v1/models/<id>/options`), so every subsequent load of that model —
   from any client, including `lemonade run`, agents, the webapp — uses the
   calibrated batch sizes automatically. The full effective args string is
   persisted with only `-b`/`-ub` swapped, so registry sampling/MTP flags are
   preserved.

State is resumable: each pair is stored under a protocol fingerprint; re-running
`calctl.py sweep` skips finished pairs (`--force` re-measures). Provenance
(engine version, lemonade version, timestamp, raw per-scenario timings) is kept
per model in `calibrations/<model-id>.json`; `calibrations/summary.json` is the
combined per-model winner record.

## Usage (on the server host)

```sh
export LEMONADE_API_KEY=<key>     # same key the lemonade CLI uses

python3 calctl.py list            # inventory + serving-config detection
python3 calctl.py sweep           # all downloaded llamacpp models
python3 calctl.py sweep --only <substring> --pairs 2048:512,2048:1024,2048:2048
python3 calctl.py apply --dry-run # preview the options writes
python3 calctl.py apply           # persist winners (auto-applies forever after)
python3 calctl.py status          # winner + applied? per model
```

`--only` is a case-insensitive substring match on the model id (repeatable),
so sweeps can be grouped and re-run in pieces.

Useful knobs: `--reps` (measurement runs per point, default 2), `--warmup`
(default 1), `--score prefill|combined`, `--extra "…"` (args appended to every
pair, e.g. `--cache-type-k q8_0 --cache-type-v q8_0`), `--ctx-auto`,
`--timeout`, `--state-dir`.

Default pair grid: `2048:512` (llama.cpp default, the baseline for the
gain figure), `2048:1024`, `2048:2048`, `4096:1024`, `4096:2048`,
`8192:1024`, `8192:2048`, `1024:512`.

## Timing notes

Each pair costs one model reload plus `(warmup+reps)` runs of both points.
Expect roughly minutes per pair on small models and tens of minutes per pair
on models that saturate memory bandwidth — start with a trimmed `--pairs`
grid to gauge your hardware, then let a full sweep run unattended. The largest
models dominate total sweep time; consider sweeping them separately
(overnight) or with a reduced grid.

**Server contention:** the server loads one model at a time, and every bench
pair evicts whatever was loaded. If an interactive client (e.g. an agent
session running on one of the calibrated models) is active during the sweep,
it will fight the bench for the slot in both directions: the sweep records
failed or evicted-mid-run pairs, and the client pays a full model reload after
each eviction. Run the sweep only when no interactive client needs the server;
failed pairs are cheap to retry because resume skips everything already
calibrated.

Running against a *remote* server is also supported (`--host <addr>
--api-key …`), but bench traffic then crosses the network; running on the
server host keeps timings clean.

## Files

- `calctl.py` — the orchestrator.
- `calibrate.py` — earlier host-local `llama-bench` driver; superseded by
  `calctl.py` for the copy-to-server workflow (it needs direct GGUF paths,
  which may live under a service user's store). Kept for reference.
- `calibrations/` — created by sweep; per-model JSON + `summary.json`.
