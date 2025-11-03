"""Command-line helper to train the UNO Bayesian neural network locally.

This is a thin wrapper around `notebooks.uno_bnn_curriculum_converted.run_cli`
so that users can simply run `python train_local_bnn.py` from the project root.
"""

from notebooks.uno_bnn_curriculum_converted import run_cli


if __name__ == "__main__":
    run_cli()
