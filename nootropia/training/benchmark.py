"""The benchmark copies as model inputs: their feature tables, labels and ballnet columns, and the (inputs, labels)
pairs the BiGRUs train on. Cached per build of each copy (common._game_info)."""

import os

import numpy as np
import pandas as pd

from .common import BENCHMARK_DIR, CACHE_DIR, LOO_DIR, NOISE_SEEDS, _game_info, boxdead_dir, lowpass_dir
from possession import dead, ownership
from possession.loaders import _cached
from possession.features import FEATURES_VERSION, tracking_features
import baseline  # noqa: E402  (upstream starter/baseline.py)


def _benchmark_game(name, info, bench_dir=BENCHMARK_DIR):
    from kloppy import metrica
    import possession_metrics as pm  # noqa: E402  (upstream evaluation/possession_metrics.py)

    files = info["files"]
    dataset = metrica.load_tracking_epts(meta_data=os.path.join(bench_dir, files["atd_meta"]),
                                         raw_data=os.path.join(bench_dir, files["atd_raw"]))
    home, away = dataset.metadata.teams
    table = pd.DataFrame(tracking_features(dataset.to_df(), [p.player_id for p in home.players],
                                           [p.player_id for p in away.players], tuple(info["pitch"]),
                                           info["fps"]))
    labels = pm.read_label_csv(os.path.join(bench_dir, files["gt"]))["label"].str.strip().str.upper()
    table["label"] = labels.to_numpy()
    table = table[table["label"] != "X"]
    floats = table.select_dtypes("float64").columns
    table[floats] = table[floats].astype(np.float32)
    return table


def benchmark_table(name, bench_dir=BENCHMARK_DIR):
    """Feature table of one benchmark game, graded frames only, with its `label` column. Cached under
    the feature version and a fingerprint of the game's build (noise profile and what it applied),
    so rebuilding the benchmark never serves a stale table."""
    info, build = _game_info(name, bench_dir)
    path = os.path.join(CACHE_DIR, "bench_{}_v{}_{}.pkl".format(name, FEATURES_VERSION, build))
    return _cached(path, lambda: _benchmark_game(name, info, bench_dir))


def _ownership_game(name, bench_dir):
    from kloppy import metrica

    info, _ = _game_info(name, bench_dir)
    files = info["files"]
    dataset = metrica.load_tracking_epts(meta_data=os.path.join(bench_dir, files["atd_meta"]),
                                         raw_data=os.path.join(bench_dir, files["atd_raw"]))
    home, away = dataset.metadata.teams
    df = dataset.to_df()
    ids = [p.player_id for p in home.players] + [p.player_id for p in away.players]
    length, width = info["pitch"]
    x = df[["{}_x".format(p) for p in ids]].to_numpy(dtype=float) * length
    y = df[["{}_y".format(p) for p in ids]].to_numpy(dtype=float) * width
    teams = np.tile(np.array([baseline.HOME] * len(home.players) + [baseline.AWAY] * len(away.players),
                             np.int8), (len(df), 1))
    ball = df[["ball_x", "ball_y"]].to_numpy(dtype=float) * (length, width)
    half = np.zeros(len(df), dtype=np.int8)
    for number, first, last in info["graded_blocks"]:
        half[first:last + 1] = number
    runs = [(first, last + 1) for _, first, last in info["graded_blocks"]]
    table = ownership.features(x, y, teams, ball, half, info["fps"], runs)
    floats = table.select_dtypes("float64").columns
    table[floats] = table[floats].astype(np.float32)
    import possession_metrics as pm  # noqa: E402  (upstream evaluation/possession_metrics.py)
    labels = pm.read_label_csv(os.path.join(bench_dir, files["gt"]))["label"].str.strip().str.upper().to_numpy()
    return table, labels


def ownership_table(name, bench_dir=BENCHMARK_DIR):
    """(features, labels) of one benchmark game, every frame. Cached per build (bench_dir: another
    build, common._game_info)."""
    _, build = _game_info(name, bench_dir)
    path = os.path.join(CACHE_DIR, "bench_{}_own_v{}_{}.pkl".format(name, ownership.INPUTS_VERSION, build))
    return _cached(path, lambda: _ownership_game(name, bench_dir))


def benchmark_sequence(name, bench_dir=BENCHMARK_DIR):
    """(inputs, labels) of one benchmark game, every frame (sequence_model.fit's pairs); the
    tracking cues are NaN on ungraded frames, as benchmark_table() keeps graded frames only.
    bench_dir: another build of the benchmark (_game_info)."""
    own, labels = ownership_table(name, bench_dir)
    cues = dead.design(benchmark_table(name, bench_dir)).reindex(range(len(labels)))
    return dead.sequence_inputs(cues, own), labels


def copy_arrays(game, bench_dir):
    """(x, y, teams, ball, pitch) of a benchmark copy (metres; teams from the metadata). Cached per build."""
    from kloppy import metrica
    info, build = _game_info(game, bench_dir)

    def build_():
        ds = metrica.load_tracking_epts(meta_data=os.path.join(bench_dir, info["files"]["atd_meta"]),
                                        raw_data=os.path.join(bench_dir, info["files"]["atd_raw"]))
        home, away = ds.metadata.teams
        df = ds.to_df()
        ids = [p.player_id for p in home.players] + [p.player_id for p in away.players]
        length, width = info["pitch"]
        x = df[["{}_x".format(p) for p in ids]].to_numpy(dtype=np.float32) * length
        y = df[["{}_y".format(p) for p in ids]].to_numpy(dtype=np.float32) * width
        teams = np.tile(np.array([baseline.HOME] * len(home.players) + [baseline.AWAY] * len(away.players),
                                 np.int8), (len(df), 1))
        ball = df[["ball_x", "ball_y"]].to_numpy(dtype=float) * (length, width)
        return x, y, teams, ball, (length, width)
    return _cached(os.path.join(CACHE_DIR, "ballnet_copy_{}_{}.pkl".format(game, build)), build_)


def copy_positions64(game, bench_dir):
    """(x, y) float64[N, S]: copy_arrays()'s slot positions in metres, computed in float64. ballnet's frozen copies are
    built from these, as the shipped nets were (copy_arrays() rounds to float32 first). Not cached (~100 MB a copy)."""
    from kloppy import metrica
    info, _ = _game_info(game, bench_dir)
    ds = metrica.load_tracking_epts(meta_data=os.path.join(bench_dir, info["files"]["atd_meta"]),
                                    raw_data=os.path.join(bench_dir, info["files"]["atd_raw"]))
    home, away = ds.metadata.teams
    df = ds.to_df()
    ids = [p.player_id for p in home.players] + [p.player_id for p in away.players]
    length, width = info["pitch"]
    return (df[["{}_x".format(p) for p in ids]].to_numpy(float) * length,
            df[["{}_y".format(p) for p in ids]].to_numpy(float) * width)


def ballnet_columns(game, bench_dir):
    """DataFrame of ballnet.COLUMNS for a benchmark copy, from the net trained without that game
    (train_ballnet.py writes them)."""
    path = os.path.join(LOO_DIR, "feat_{}_{}.pkl".format(os.path.basename(os.path.normpath(bench_dir)), game))
    if not os.path.isfile(path):
        raise FileNotFoundError("{} missing: run python nootropia/training/train_ballnet.py".format(path))
    return pd.read_pickle(path)


def ownership_inputs(name, bench_dir=None):
    """(inputs, labels) of the ownership BiGRU for one benchmark copy: ownership_table() plus ballnet's columns
    (ballnet_columns(), from the net trained without that game). The dead-ball BiGRU reads the tables without them."""
    bench_dir = bench_dir or BENCHMARK_DIR
    table, labels = ownership_table(name, bench_dir)
    return pd.concat([table.reset_index(drop=True), ballnet_columns(name, bench_dir)], axis=1), labels


def _benchmark_pairs(game, target, bench_dir=BENCHMARK_DIR):
    """(inputs, labels) of one benchmark game for a BiGRU target: ownership_inputs() (with ballnet's columns) for
    "home", benchmark_sequence() for "dead"."""
    return ownership_inputs(game, bench_dir) if target == "home" else benchmark_sequence(game, bench_dir)


def train_dirs(target, noise_seeds=NOISE_SEEDS):
    """[base, copy per noise seed]: the benchmark builds a BiGRU trains on. The dead-ball one trains on its own copies
    (boxdead_dir(): keeper holds and box stoppages hidden, build_benchmark.py --boxdead), the ownership one on its own
    too (lowpass_dir(): players and ball low-passed like BAR-ATH's tracker output, build_benchmark.py --lowpass)."""
    which = boxdead_dir if target == "dead" else lowpass_dir
    return [which(0)] + [which(s) for s in noise_seeds]


def training_pairs(games, held=None, data=None, target="home", noise_seeds=NOISE_SEEDS):
    """(inputs, labels) pairs to train a BiGRU without `held` (None: on all games): the other games
    (data: {game: pairs} of the plain build, used when that is the base), then their copies degraded with each of
    noise_seeds, from train_dirs(target). A held-out game's copies are never trained on."""
    others = [g for g in games if g != held]
    base, *copies = train_dirs(target, noise_seeds)
    reuse = data and base == BENCHMARK_DIR
    pairs = [data[g] if reuse else _benchmark_pairs(g, target, base) for g in others]
    for d in copies:
        pairs += [_benchmark_pairs(g, target, d) for g in others]
    return pairs
