"""Paths and helpers shared by the training scripts."""

import datetime
import hashlib
import json
import os
import sys

TEAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the submission folder
if TEAM_DIR not in sys.path:
    sys.path.insert(0, TEAM_DIR)
from possession.loaders import CACHE_DIR, OUTPUTS_DIR  # noqa: E402  (first: puts the upstream modules on sys.path)

# Params
BENCHMARK_DIR = os.path.join(OUTPUTS_DIR, "benchmark")   # the benchmark games as built (noise seed 0)
NOISE_SEEDS = (1, 2)       # both BiGRUs and ballnet also train on the games degraded with these seeds (noise_dir())
DFL_DIR = os.environ.get("DFL_DIR", os.path.join(OUTPUTS_DIR, "ext", "dfl"))  # the DFL open-data XML files (README)
DFL_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
LOO_DIR = os.path.join(CACHE_DIR, "ballnet_loo")         # leave-one-out ballnet nets and the columns they give
MODELS_DIR = os.path.join(TEAM_DIR, "models")


BOXDEAD_DIR = os.path.join(OUTPUTS_DIR, "benchmark_boxdead")   # the dead-ball BiGRU's copies (build_benchmark.py --boxdead)


def noise_dir(seed):
    return os.path.join(OUTPUTS_DIR, "benchmark_s{}".format(seed))


def boxdead_dir(seed):
    """The dead-ball BiGRU's copy for a noise seed: keeper holds hidden and slowed (live), box stoppages hidden (dead)."""
    return BOXDEAD_DIR if seed == 0 else "{}_s{}".format(BOXDEAD_DIR, seed)


def copies():
    """Every build of the benchmark: as built, then each noise seed."""
    return [BENCHMARK_DIR] + [noise_dir(s) for s in NOISE_SEEDS]


def _game_info(name, bench_dir=BENCHMARK_DIR):
    """(manifest entry, build fingerprint) of one benchmark game; bench_dir: another build of the
    benchmark (e.g. outputs/benchmark_s1, the same games degraded with another noise seed)."""
    with open(os.path.join(bench_dir, "manifest.json")) as handle:
        info = json.load(handle)["games"][name]
    return info, hashlib.sha1(json.dumps(info, sort_keys=True).encode()).hexdigest()[:8]


def benchmark_games(prefix=""):
    with open(os.path.join(BENCHMARK_DIR, "manifest.json")) as handle:
        return sorted(g for g in json.load(handle)["games"] if g.startswith(prefix))


def file_hash(path):
    """First 12 hex digits of the sha256 of a file's bytes."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:12]


def write_provenance(model_path, record):
    """models/<model>.json beside a saved model: its hash, the benchmark builds it was trained on and `record`. Keeps
    decoder_tuned_on, the model hash the decoder constants were tuned on, and says when the new model differs."""
    path = os.path.splitext(model_path)[0] + ".json"
    old = json.load(open(path)) if os.path.isfile(path) else {}
    model = file_hash(model_path)
    out = {"model_file": os.path.basename(model_path), "sha256_12": model, **record,
           "benchmark_build": {g: _game_info(g)[1] for g in benchmark_games("DFL")}, "date": datetime.date.today().isoformat()}
    if "decoder_tuned_on" in old:
        out["decoder_tuned_on"] = old["decoder_tuned_on"]
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    same = model == old.get("sha256_12")
    print("{}: sha256 {} ({})".format(os.path.basename(model_path), model, "identical to the shipped model" if same else
                                      "differs from the shipped {}".format(old.get("sha256_12"))))
    return same
