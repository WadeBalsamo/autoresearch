"""Hands-off hyperparameter search over a track's editable constants.

Complements the autonomous agent loop (where Claude edits ``train.py`` by hand). The
optimizer patches the top-of-file constants in a track's ``train.py`` (without touching the
fixed ``prepare.py``), runs each config under its fixed time budget, greps
``primary_metric``, and logs a leaderboard. Same metric, same splits, fully comparable.
"""
