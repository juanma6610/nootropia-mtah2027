"""Fit the dead-ball BiGRU: python nootropia/training/train_dead.py

On the 7 DFL benchmark games' dead-ball copies and their noise copies (build_benchmark.py --boxdead: keeper holds and
box stoppages hidden, benchmark.train_dirs("dead")); writes models/dead_sequence.pt and its provenance record
models/dead_sequence.json. Seeded and deterministic on CPU.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import benchmark, bigru, common  # noqa: E402  (common first: puts possession and upstream on sys.path)
from possession import dead  # noqa: E402

if __name__ == '__main__':
    games = common.benchmark_games("DFL")
    net, scaler, cfg = bigru.fit(benchmark.training_pairs(games, target="dead"), dead.CONFIG)
    bigru.save(net, scaler, cfg, dead.MODEL_PATH)
    common.write_provenance(dead.MODEL_PATH, {"train_command": "python nootropia/training/train_dead.py",
                                              "seed": cfg["seed"], "noise_seeds": list(common.NOISE_SEEDS), "config": cfg,
                                              "train_dirs": [os.path.basename(d) for d in benchmark.train_dirs("dead")],
                                              "train_build": {os.path.basename(d): {g: common._game_info(g, d)[1] for g in games}
                                                              for d in benchmark.train_dirs("dead")}})
