"""Our method: predict() gives the submission's labels (make_submission.py), inputs() the decoder's inputs."""

from . import dead, decode, ownership, tagging, whistle
from .features import load_features


def inputs():
    """dict of the decoder's inputs (and what they were built from), as predict() uses them."""
    tag_dead = tagging.tag_dead()
    p_home, p_dead = ownership.p_home(), dead.p_dead()
    set_pieces = tagging.restarts()
    tag_team, hints, whistles = tagging.tag_team(), dead.edges(load_features(), set_pieces.restart), whistle.cues()
    return {"p_home": p_home, "p_dead": p_dead,
            "restart_team": dead.restart_team(p_dead, set_pieces),
            "tag_team": tag_team, "hints": hints, "tag_dead": tag_dead, "whistles": whistles}


def predict():
    """H/A/D/X labels, one per tactical video frame (199319)."""
    x = inputs()
    return decode.decode(x["p_home"], x["p_dead"], x["restart_team"], x["tag_team"], hints=x["hints"],
                         whistles=x["whistles"])
