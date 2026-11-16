"""Rebuild models/ from public data (README, "Reproduce the models"):

    build_benchmark.py   the 7 DFL open-data matches degraded, 3 noise seeds
    train_ballnet.py     ballnet: 7 leave-one-out nets (their columns train the ownership BiGRU) and the full net
    train_ownership.py   the ownership BiGRU
    train_dead.py        the dead-ball BiGRU

Never trains on BAR-ATH.
"""
