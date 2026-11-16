"""Fit the players-only dead-ball model (possession/players_dead.py): python nootropia/training/train_players_dead.py

On the 7 DFL benchmark games and their noise copies (build_benchmark.py), players seen through freeze.py's virtual
camera; labels dead vs live from the benchmark's own flags. Seeded and deterministic on CPU (~20 min); writes
models/players_dead.pt and reports whether it is byte-identical to the shipped one.
"""

import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import benchmark, common, freeze  # noqa: E402  (common first: puts possession and upstream on sys.path)
from possession import players_dead as pdm  # noqa: E402
from possession.loaders import _cached  # noqa: E402
import numpy as np  # noqa: E402

WINDOW, STRIDE, BATCH, LR, SEED, EPOCHS = 50, 25, 64, 1e-3, 0, 6     # as the dead-ball BiGRU


def copy_inputs(game, bench_dir):
    """{"P", "M", "y" (dead), "m" (graded), "n"} of one benchmark copy at players_dead.STEP. Cached per build."""
    _, build = common._game_info(game, bench_dir)

    def make():
        x, y, teams, ball, pitch = benchmark.copy_arrays(game, bench_dir)
        x, y = x.astype(float), y.astype(float)
        idx = np.arange(0, len(x), pdm.STEP)
        H = freeze.copy_camera(ball[:, 0] / pitch[0])[idx]
        P, M = pdm.set_arrays(x, y, teams, pdm.onscreen(H, x[idx], y[idx], pitch), pitch)
        lab = np.asarray(benchmark.ownership_table(game, bench_dir)[1]).astype(str)[idx]
        return {"P": P, "M": M, "y": (lab == "D").astype(np.float32), "m": (lab != "X").astype(np.float32), "n": len(x)}
    return _cached(os.path.join(common.CACHE_DIR, "players_dead_{}_{}.pkl".format(game, build)), make)


def fit(seqs):
    import torch
    from torch import nn
    pdm.setup(SEED, pdm.TRAIN_THREADS)
    t0 = time.time()
    wins = [(i, s, sign) for i, q in enumerate(seqs) for s in range(0, len(q["y"]) - WINDOW + 1, STRIDE)
            if q["m"][s:s + WINDOW].any() for sign in (1.0, -1.0)]
    net = pdm._net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        order = torch.randperm(len(wins), generator=gen).tolist()
        for b in range(0, len(order), BATCH):
            batch = [wins[j] for j in order[b:b + BATCH]]
            P = np.stack([seqs[i]["P"][s:s + WINDOW] for i, s, _ in batch])
            P[..., 5] *= np.array([sg for _, _, sg in batch], np.float32)[:, None, None]     # team swap
            M = np.stack([seqs[i]["M"][s:s + WINDOW] for i, s, _ in batch]).astype(np.float32)
            Y = np.stack([seqs[i]["y"][s:s + WINDOW] for i, s, _ in batch])
            W = torch.from_numpy(np.stack([seqs[i]["m"][s:s + WINDOW] for i, s, _ in batch]))
            loss = (bce(net(torch.from_numpy(P), torch.from_numpy(M)), torch.from_numpy(Y)) * W).sum() / W.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    print("fit on {} sequences ({} windows): {:.0f} s".format(len(seqs), len(wins), time.time() - t0), flush=True)
    return net


def sha12(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:12]


if __name__ == "__main__":
    import torch
    games = common.benchmark_games("DFL")
    old = sha12(pdm.MODEL_PATH) if os.path.isfile(pdm.MODEL_PATH) else None
    path = os.environ.get("OUT", pdm.MODEL_PATH)
    torch.save(fit([copy_inputs(g, d) for d in common.copies() for g in games]).state_dict(), path)
    new = sha12(path)
    print("{}: sha256 {} ({})".format(os.path.basename(path), new,
                                       "identical to the shipped model" if new == old else "differs from the shipped {}".format(old)))
