"""Build the benchmark the models train on: python nootropia/training/build_benchmark.py [--dfl-dir DIR]

The 7 DFL open-data matches (Bassek et al. 2025, CC-BY 4.0) degraded to look like automatic tracking
(make_benchmark.py) with noise_profile.json, the profile the shipped models were trained with, once per noise seed:
outputs/benchmark (seed 0), outputs/benchmark_s1 and outputs/benchmark_s2. --boxdead builds the dead-ball BiGRU's
copies instead, with noise_profile_boxdead.json (noise_profile.json plus faster re-finding in play, the ball hidden in
the keeper's hands with each hold slowed to 3-20 s, and through 80% of the stoppages that start in a box):
outputs/benchmark_boxdead, _s1, _s2 (~1.5 h). The plain build is needed either way (benchmark_games() reads its
manifest). DIR (default common.DFL_DIR) holds, per
match, DFL_02_01_matchinformation_*_DFL-MAT-<id>.xml and DFL_04_03_positions_raw_observed_*_DFL-MAT-<id>.xml.
Seeded: the same files give the same copies. ~10 GB of memory while a match loads.
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from training import common  # noqa: E402
import make_benchmark  # noqa: E402  (this folder's builder)


def dfl_sources(dfl_dir):
    out = []
    for mid in common.DFL_MATCHES:
        meta = glob.glob(os.path.join(dfl_dir, "*matchinformation*_DFL-MAT-{}.xml".format(mid)))
        raw = glob.glob(os.path.join(dfl_dir, "*positions_raw_observed*_DFL-MAT-{}.xml".format(mid)))
        if len(meta) != 1 or len(raw) != 1:
            sys.exit("DFL match {}: need one matchinformation and one positions_raw_observed XML in {}".format(mid, dfl_dir))
        out.append("dfl-files:{},{}".format(meta[0], raw[0]))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", default=common.DFL_DIR, help="the DFL open-data XML files")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0] + list(common.NOISE_SEEDS), help="noise seeds to build")
    parser.add_argument("--boxdead", action="store_true", help="the dead-ball BiGRU's copies (noise_profile_boxdead.json)")
    args = parser.parse_args()
    sources = dfl_sources(args.dfl_dir)
    for seed in args.seeds:
        out = common.boxdead_dir(seed) if args.boxdead else (common.BENCHMARK_DIR if seed == 0 else common.noise_dir(seed))
        profile = "noise_profile_boxdead.json" if args.boxdead else "noise_profile.json"
        make_benchmark.build(argparse.Namespace(sources=sources, profile=os.path.join(HERE, profile),
                                                out=out, seed=seed, dfl_min_spell_frames=0,
                                                metrica_dir=str(make_benchmark.OUTPUTS / "ext" / "sample-data")))
