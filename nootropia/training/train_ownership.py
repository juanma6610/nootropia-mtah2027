"""Fit the ownership BiGRU: python nootropia/training/train_ownership.py

On the 7 DFL benchmark games and their noise copies, with ballnet's leave-one-out columns (run train_ballnet.py first);
writes models/sequence.pt and its provenance record models/sequence.json. Seeded and deterministic on CPU.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import benchmark, bigru, common  # noqa: E402  (common first: puts possession and upstream on sys.path)
from possession import sequence_model  # noqa: E402

if __name__ == '__main__':
    games = common.benchmark_games("DFL")
    data = {g: benchmark.ownership_inputs(g) for g in games}
    net, scaler, cfg = bigru.fit(benchmark.training_pairs(games, data=data))
    bigru.save(net, scaler, cfg, sequence_model.MODEL_PATH)
    common.write_provenance(sequence_model.MODEL_PATH, {"train_command": "python nootropia/training/train_ownership.py",
                                                         "seed": cfg["seed"], "noise_seeds": list(common.NOISE_SEEDS),
                                                         "config": cfg})
