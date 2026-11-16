"""Regenerate the submission: python make_submission.py (from any directory) writes outputs/nootropia.csv next to it."""

import os
import sys

TEAM_DIR = os.path.dirname(os.path.abspath(__file__))  # this folder: possession/ and models/ sit next to this file
sys.path.insert(0, TEAM_DIR)
from possession.loaders import OUTPUTS_DIR  # noqa: E402
from possession.pipeline import predict  # noqa: E402
import baseline  # noqa: E402  (upstream starter/baseline.py)

if __name__ == '__main__':
    path = baseline.write_submission(os.path.join(OUTPUTS_DIR, "nootropia.csv"), predict())
    print("wrote {}".format(path))
