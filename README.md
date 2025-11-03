# UNO BNN Local Training Guide

This project contains a Bayesian neural network (BNN) trainer for UNO gameplay scenarios.
The training code was ported from the original Jupyter notebook into a reusable command-line
script so you can reproduce experiments or fine-tune new models entirely on your own machine.

---

## 1. Prerequisites

- **Python**: 3.10 or newer (3.12 recommended).
- **Platform**: Linux, macOS, or Windows (WSL works well).
- **Hardware**: the reference PC (Intel i5-4570, 8?GB RAM, NVIDIA GT 1030) is sufficient.
  - Training defaults to CPU; CUDA is used automatically when available.
  - Expect the `quick` preset to finish within minutes on the reference CPU/GPU combo.
- **Disk space**: ~2?GB free (datasets, logs, and exported weights).

> ?? **Tip**: Close other heavy applications while training; the scenario generator
> can consume ~5?GB RAM on larger runs.

---

## 2. Environment Setup

```bash
# 1) (Optional) create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies (CPU wheels by default)
pip install --upgrade pip
pip install numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pyro-ppl tqdm

# 3) Verify everything imports correctly
python - <<'PY'
import torch, pyro, numpy
print("PyTorch", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("Pyro", pyro.__version__)
print("NumPy", numpy.__version__)
PY
```

### Optional: NVIDIA CUDA wheels

If you have a compatible GPU and wish to use CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Quick Start

All training commands are exposed through `train_local_bnn.py` in the repo root.

```bash
# Preview configuration without doing any work
python train_local_bnn.py --preset quick --dry-run

# Launch a short CPU-friendly training session
python train_local_bnn.py --preset quick

# Monitor detailed logs during training (see training_logs/)
tail -f training_logs/*.log
```

Available presets:

| Preset   | Scenarios | Epochs | Batch | Learning Rate | Progress Bar |
|----------|-----------|--------|-------|---------------|--------------|
| `quick`  | 1200      | 12     | 96    | 4e-3          | off          |
| `balanced` *(default)* | 2000 | 20 | 128 | 3e-3 | on |
| `max`    | 3200      | 30     | 192   | 2.5e-3        | on           |

Use the `--preset` flag to pick one, e.g. `--preset max`.

---

## 4. Command Reference

`python train_local_bnn.py --help` prints the full CLI once dependencies are installed.
Key options are summarised below:

- `--preset {quick,balanced,max}`: load a sensible bundle of hyperparameters.
- `--dry-run`: print the resolved configuration (after presets) and exit.
- `--num-scenarios N`: generate a custom number of synthetic game states.
- `--epochs E`: override training epochs.
- `--device {auto,cpu,cuda}`: force CPU/GPU selection.
- `--log-batches`: emit per-batch loss (default interval = 10 batches).
- `--batch-log-interval K`: change the per-batch logging stride.
- `--log-dir PATH`: choose where CSV & text logs are written (default `training_logs/`).
- `--no-text-log`: disable detailed text logs (console output only).
- `--export`: save the trained param store & metadata to `models/`.
- `--scenario-mix` "FINISHER=0.4,DEFENDER=0.6": customise curriculum sampling.
- `--output-dir PATH` / `--model-tag NAME`: control export destinations when `--export` is used.

### Logging & Artifacts

- **CSV metrics**: `training_logs/<run_name>.csv` (epoch, loss, accuracy, timings).
- **Text logs**: `training_logs/<run_name>.log` (more verbose, includes batch logs).
- **Exports**: `models/<model_tag>_param_store.pt` + metadata JSON (when `--export`).

---

## 5. Monitoring Progress

1. Watch the console or text log for epoch summaries, including train/val loss and accuracy.
2. Open the CSV in your favourite tool (Sheets, Excel, Pandas) to chart convergence.
3. Enable batch logging for more granular insight when tuning learning rates.

> ?? **Suggestion**: keep a notes file with the CLI command used, hardware, and
> duration. It speeds up future comparisons between presets.

---

## 6. Tips for Modest Hardware

- **Memory**: reduce `--num-scenarios` or `--batch-size` if you hit memory pressure.
- **Speed**: disable the progress bar (`--no-progress`) on remote shells to reduce output overhead.
- **Stability**: keep the default `clip_norm` of 10; lowering it may slow convergence.
- **GPU (GT 1030)**: supports CUDA but has limited VRAM. Start with the `quick` preset,
  consider `--batch-size 64` if you see CUDA OOM errors.

---

## 7. Troubleshooting

| Issue | Resolution |
|-------|------------|
| `Missing dependency 'numpy'` | Re-run the pip install block from Section 2. |
| `RuntimeError: CUDA error: out of memory` | Reduce `--batch-size`, try `--device cpu`, or close other GPU apps. |
| Training seems slow | Use `--preset quick` to validate pipeline, then scale up. |
| CSV/logs not written | Ensure `training_logs/` is writable; adjust with `--log-dir`. |
| Interrupted run (`Ctrl+C`) | The script exits cleanly with status 130; rerun the same command to start over. |

Still stuck? Re-run with `--log-level DEBUG` and inspect the text log for stack traces.

---

## 8. Next Steps

- Experiment with custom scenario mixes to focus the curriculum on specific tactical situations.
- Export artifacts (`--export`) and integrate them into downstream UNO simulations or agents.
- Revisit hyperparameters after reviewing validation accuracy; adjust learning rate or add more scenarios.

Happy training! ??

