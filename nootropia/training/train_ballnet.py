"""Train ballnet (possession/ballnet.py): python nootropia/training/train_ballnet.py [--only GAME|ALL ...] [--dfl-dir DIR]

1. Leave one DFL game out (7 nets): each net is trained on the other six games x 3 noise copies and writes ballnet's
   columns for the held-out game's copies to common.LOO_DIR: those three and the ownership BiGRU's low-passed ones
   (common.lowpass_dirs(), build_benchmark.py --lowpass, built first). The ownership BiGRU trains on those
   (benchmark.ownership_inputs), so no game's columns come from a net that saw it.
2. Fit on all seven: models/ballnet.pt and its provenance record models/ballnet.json.

Needs the benchmark (build_benchmark.py) and the DFL raw positions XML (the clean ball, the training target) in DIR.
Resumes: nets already in LOO_DIR are reused. Each net takes ~25-35 min on CPU; --only runs a subset, so several
processes can share the work. Never trains on BAR-ATH.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import ballnet_fit, benchmark, common  # noqa: E402  (common first: puts possession and upstream on sys.path)
from possession import ballnet  # noqa: E402


def log(message):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), message), flush=True)


def train(held, games):
    """Net without `held` (None: all seven), trained on the other games' copies, as built and frozen."""
    seqs = [ballnet_fit.training_sequence(g, d) for d in common.copies() for g in games if g != held]
    log("training without {} on {} copies".format(held or "nothing", len(seqs)))
    return ballnet_fit.fit(seqs, log=log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", help="games to hold out (DFL-...) and/or ALL (default: all of them)")
    parser.add_argument("--dfl-dir", default=common.DFL_DIR, help="DFL raw positions XML directory")
    args = parser.parse_args()
    common.DFL_DIR = args.dfl_dir
    games = common.benchmark_games("DFL")
    os.makedirs(common.LOO_DIR, exist_ok=True)
    for held in args.only or games + ["ALL"]:
        if held == "ALL":
            ballnet_fit.save(train(None, games), ballnet.MODEL_PATH)
            common.write_provenance(ballnet.MODEL_PATH, {
                "train_command": "python nootropia/training/train_ballnet.py", "config": ballnet.CFG,
                "threads": ballnet.THREADS, "noise_seeds": list(common.NOISE_SEEDS), "frozen_copies": True})
            continue
        path = os.path.join(common.LOO_DIR, "net_{}.pt".format(held))
        net = ballnet.load(path) if os.path.isfile(path) else train(held, games)
        ballnet_fit.save(net, path)
        for d in common.copies() + common.lowpass_dirs():
            x, y, teams, ball, pitch = benchmark.copy_arrays(held, d)
            columns, _ = ballnet.frame_features(net, x, y, teams, ball, pitch)
            columns.to_pickle(os.path.join(common.LOO_DIR, "feat_{}_{}.pkl".format(os.path.basename(os.path.normpath(d)), held)))
        log("held out {}: net and columns written".format(held))
