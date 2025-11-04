# UNO Bayesian Neural Network (BNN) Training & Analysis Platform

A complete machine learning platform for training and analyzing Bayesian neural networks to play UNO 2v2. This project combines a deterministic game engine, synthetic scenario generation, Bayesian deep learning, and interactive interfaces for both training and gameplay analysis.

---

## 🎯 Overview

This platform enables you to:

- **Train Bayesian Neural Networks** to learn UNO 2v2 strategy from synthetic gameplay scenarios
- **Play UNO interactively** in your terminal with real-time BNN analysis and uncertainty metrics
- **Monitor training progress** through a web dashboard with live metrics and logging
- **Analyze model decisions** with predictive entropy, mutual information, and confidence intervals
- **Export trained models** for integration into downstream applications or agents

The implementation uses PyTorch and Pyro for Bayesian inference, a custom deterministic UNO engine for 2v2 gameplay (Classic and Go Wild modes), and includes comprehensive scenario generation with multiple bot personalities (Oracle, Aggressor, Supporter, Conservative, and Random bots).

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training Models](#-training-models)
- [Playing UNO with BNN Analysis](#-playing-uno-with-bnn-analysis)
- [Web Dashboard](#-web-dashboard)
- [Architecture Overview](#-architecture-overview)
- [Testing](#-testing)
- [Known Issues](#-known-issues)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)
- [Contributing](#-contributing)

---

## ✨ Features

### 🎓 Training Pipeline
- **Synthetic scenario generation** with 5 bot personalities exhibiting different strategies
- **Curriculum learning** across 6 scenario types (Setup, Finisher, Defender, Aggressor, Critical, Random)
- **Bayesian neural networks** with variational inference for uncertainty quantification
- **Multiple training presets** (quick, balanced, max, stable) optimized for different hardware
- **Comprehensive logging** with CSV metrics, text logs, and real-time progress tracking
- **Model export** with metadata versioning and compatibility checking

### 🎮 Interactive Console Play
- **Play UNO 2v2** against heuristic bots with live BNN recommendations
- **Real-time analysis** showing entropy, mutual information, and top predictions
- **Uncertainty visualization** with probability distributions and confidence intervals
- **Multiple game modes**: Classic 2v2 and Go Wild 2v2
- **Configurable personas** to adjust BNN feature encoding

### 📊 Web Dashboard
- **Browser-based training interface** for configuring and launching training runs
- **Live metrics streaming** via Server-Sent Events (SSE)
- **Progress visualization** with epoch tracking, ETA estimation, and loss curves
- **Training control** with start/stop capabilities
- **Log history viewer** with filterable output

### 🔬 Technical Highlights
- **Deterministic game engine** implementing full UNO ruleset for 2v2 play
- **State/Action encoders** with 103-dimensional state features and 66+ action classes
- **Monte Carlo sampling** for posterior predictive inference (default: 40 samples)
- **Color permutation augmentation** for data efficiency
- **KL divergence regularization** for calibrated uncertainty
- **CPU and CUDA support** with automatic device detection

---

## 📁 Project Structure

```
/workspace/
├── uno_engine/              # Core UNO game engine
│   ├── engine.py            # State transitions and rule validation
│   ├── deck.py              # Deck construction for game modes
│   └── models.py            # Data classes (GameState, Card, Player, etc.)
│
├── notebooks/
│   └── uno_bnn_curriculum_converted.py  # Main BNN training logic
│       # Includes: ScenarioForge, BotPolicies, StateEncoder, 
│       # ActionEncoder, BNN model/guide, training loop
│
├── train_local_bnn.py       # CLI entry point for training
├── play_bnn_console.py      # Interactive UNO console with BNN analysis
├── dashboard_server.py      # Flask backend for web dashboard
├── uno_bnn_training_dashboard.html  # Web UI for training
│
├── tests/
│   ├── test_engine.py       # Game engine validation tests
│   └── test_bnn_encoders.py # Encoder and BNN component tests
│
├── models/                  # Exported trained models (*.pt, *.json)
├── training_logs/           # Training metrics (CSV) and text logs
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🔧 Prerequisites

- **Python**: 3.10 or newer (3.12 recommended)
- **Platform**: Linux, macOS, or Windows (WSL works well)
- **Hardware**: 
  - **CPU**: Intel i5-4570 or equivalent (training possible on modest hardware)
  - **RAM**: 8GB minimum (16GB recommended for larger presets)
  - **GPU** (optional): NVIDIA CUDA-capable GPU with 2GB+ VRAM
  - **Disk**: ~2-5GB free for datasets, logs, and models
- **Python packages**: See `requirements.txt`

> 💡 **Note**: Training defaults to CPU. CUDA is automatically used when available. The `quick` preset completes in minutes on modest hardware.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd workspace
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

#### CPU-only Installation (Default)

```bash
pip install --upgrade pip
pip install numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pyro-ppl tqdm flask pytest
```

#### GPU Installation (CUDA 12.1)

```bash
pip install --upgrade pip
pip install numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyro-ppl tqdm flask pytest
```

#### Using `requirements.txt`

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python - <<'PY'
import torch, pyro, numpy
print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
print(f"Pyro {pyro.__version__}")
print(f"NumPy {numpy.__version__}")
print("✓ All dependencies installed successfully!")
PY
```

---

## 🚀 Quick Start

### Train Your First Model (5 minutes)

```bash
# Preview configuration without training
python train_local_bnn.py --preset quick --dry-run

# Train a quick model (12 epochs, 1200 scenarios)
python train_local_bnn.py --preset quick --export

# The model will be saved to models/ when complete
```

### Play UNO with the Trained Model

```bash
# Uses the latest model from models/ automatically
python play_bnn_console.py

# Or specify a model explicitly
python play_bnn_console.py --meta models/local_CLASSIC_2V2_20251103-221802_meta.json \
                           --param-store models/local_CLASSIC_2V2_20251103-221802_param_store.pt
```

### Launch the Web Dashboard

```bash
# Start the Flask server
python dashboard_server.py

# Open browser to http://127.0.0.1:8000
# Configure training parameters, start runs, and monitor progress
```

**Windows Users**: Run `run_dashboard.bat` to automatically set up a virtual environment and launch the dashboard.

---

## 🎓 Training Models

### Training Command Reference

```bash
python train_local_bnn.py [OPTIONS]
```

### Training Presets

Choose a preset to match your hardware and time budget:

| Preset | Scenarios | Epochs | Batch Size | Learning Rate | Recommended For | Duration |
|--------|-----------|--------|------------|---------------|-----------------|----------|
| `quick` | 1,200 | 12 | 96 | 4e-3 | Testing, debugging | 2-5 min |
| `balanced` (default) | 2,000 | 20 | 128 | 3e-3 | General use | 10-20 min |
| `max` | 3,200 | 30 | 192 | 2.5e-3 | High-quality models | 30-60 min |
| `stable` | 32,000 | 100 | 192 | 1e-4 | Research, production | 4-8 hours |

### Common Training Options

```bash
# Use a preset
python train_local_bnn.py --preset balanced --export

# Custom configuration
python train_local_bnn.py --num-scenarios 5000 \
                          --epochs 25 \
                          --batch-size 128 \
                          --learning-rate 0.002 \
                          --export

# Force CPU or GPU
python train_local_bnn.py --preset quick --device cpu
python train_local_bnn.py --preset max --device cuda

# Custom scenario mix (adjust curriculum balance)
python train_local_bnn.py --preset balanced \
                          --scenario-mix "FINISHER=0.4,DEFENDER=0.3,AGGRESSOR=0.3" \
                          --export

# Enable batch-level logging
python train_local_bnn.py --preset quick --log-batches --batch-log-interval 5

# Reproducible training with seed
python train_local_bnn.py --preset balanced --seed 42 --export

# Custom log directory
python train_local_bnn.py --preset max --log-dir ./my_logs --export
```

### Understanding Training Output

During training, you'll see output like:

```
[local_CLASSIC_2V2_20251104-120530] Preparing synthetic dataset...
[local_CLASSIC_2V2_20251104-120530] Dataset ready | samples=2000 | features=103 | actions=66
[local_CLASSIC_2V2_20251104-120530] Initializing BNN (hidden=256, device=cuda)

Epoch  1/20 | train 4.123 | val 4.087 | acc 0.015 | 12.3s
Epoch  2/20 | train 3.856 | val 3.921 | acc 0.023 | 11.8s
Epoch  3/20 | train 3.654 | val 3.702 | acc 0.034 | 11.9s
...
Epoch 20/20 | train 3.124 | val 3.198 | acc 0.089 | 12.1s

[local_CLASSIC_2V2_20251104-120530] Training complete!
[local_CLASSIC_2V2_20251104-120530] Exporting artifacts to models/
```

**Metrics**:
- **train**: Training loss (lower is better)
- **val**: Validation loss (should track train loss)
- **acc**: Validation accuracy (random baseline ≈ 1.5% for 66 classes)

### Training Artifacts

After training with `--export`, you'll find:

```
models/
├── local_CLASSIC_2V2_20251104-120530_meta.json        # Model metadata
└── local_CLASSIC_2V2_20251104-120530_param_store.pt   # Trained weights

training_logs/
├── local_CLASSIC_2V2_20251104-120530.log              # Detailed text log
└── local_CLASSIC_2V2_20251104-120530_<timestamp>.csv  # Epoch metrics
```

**Metadata includes**:
- Feature and action dimensions
- Schema version for compatibility checking
- Action encoder token mappings
- Training history (loss, accuracy curves)
- Hyperparameters and configuration

---

## 🎮 Playing UNO with BNN Analysis

### Starting a Game

```bash
python play_bnn_console.py
```

The console interface will:
1. Auto-detect the latest trained model in `models/`
2. Prompt you to select a game mode (Classic 2v2 or Go Wild 2v2)
3. Start an interactive UNO round with 4 players

### Gameplay Interface

During your turn, you'll see:

```
--- Your Turn --------------------------------------------------
Top card: Green 7 | Current color: Green
Draw stack active: 0 cards owed by P0.
Hand:
  [1] Green 0
  [2] Green 6
  [3] Draw Two (Green)
  [4] Yellow 4
  [5] Wild Draw Four

AllyBot's hand:
    - Blue 5
    - Red 2
    - Green Skip

Other players:
    NorthBot (Opponent) - 8 cards
    EastBot (Opponent) - 6 cards

BNN perspective:
  Scenario: SETUP | Entropy: 2.341 | MI: 0.123
  Confidence: 42.31% +/- 8.12% | MC=40
  Suggested action: Play Green 6
  Distribution summary: mu=15.15% +/- 12.34% | Selection trigger: probability
  Top predictions (mean +/- std):
    Play Green 6                             42.31% +/-  8.12% (valid)
    Play Green 0                             28.45% +/-  6.54% (valid)
    Play Draw Two (Green)                    18.23% +/-  5.32% (valid)

Action ([p]lay #, [d]raw, [q]uit): p 2
```

### Commands

- **`p <number>`** or **`play <number>`**: Play the card at the specified index
- **`d`** or **`draw`**: Draw a card from the deck
- **`q`** or **`quit`**: Exit the game

### BNN Analysis Explained

**Metrics shown**:
- **Scenario**: Inferred scenario type (SETUP, FINISHER, DEFENDER, etc.)
- **Entropy**: Predictive entropy (higher = more uncertainty)
- **MI** (Mutual Information): Epistemic uncertainty from model parameters
- **Confidence**: Probability assigned to the suggested action ± standard deviation
- **MC**: Number of Monte Carlo samples used for inference
- **Distribution summary**: Mean and std dev across valid actions
- **Selection trigger**: Reason for the suggestion (probability, entropy, fallback)

### Command-Line Options

```bash
# Use a specific model
python play_bnn_console.py --meta models/my_model_meta.json \
                           --param-store models/my_model_param_store.pt

# Change BNN persona (affects feature encoding)
python play_bnn_console.py --persona AggressorBot

# Adjust Monte Carlo samples (more = better estimates, slower)
python play_bnn_console.py --mc-samples 100

# Start directly in a specific mode
python play_bnn_console.py --mode classic  # or --mode wild

# Force CPU inference
python play_bnn_console.py --cpu
```

### Personas

Available personas (used for state encoding):
- **OracleBot** (default): Optimal play knowledge
- **AggressorBot**: Offensive strategy focus
- **SupporterBot**: Defensive/team play focus
- **ConservativeBot**: Risk-averse strategy
- **RandomBot**: Uniform random baseline

---

## 📊 Web Dashboard

### Starting the Dashboard

```bash
python dashboard_server.py
```

Then open your browser to **http://127.0.0.1:8000**

**Windows users**: Double-click `run_dashboard.bat` for automatic setup.

### Dashboard Features

**Configuration Panel**:
- Select training preset or customize hyperparameters
- Choose game mode (CLASSIC_2V2, GO_WILD_2V2)
- Set learning rate, batch size, epochs, scenarios
- Enable/disable progress bars
- Configure logging verbosity
- Toggle model export

**Progress Panel**:
- Real-time epoch progress with ETA estimation
- Current epoch metrics (train loss, val loss, accuracy)
- Status indicator (Idle, Preparing, Training, Exporting, Complete, Failed)
- Training duration tracking

**Log Viewer**:
- Live streaming logs via Server-Sent Events
- Color-coded log levels (DEBUG, INFO, WARNING, ERROR)
- Auto-scrolling with manual scroll lock
- Search and filter capabilities

**Controls**:
- **Start Training**: Launch a new training run
- **Stop Training**: Gracefully interrupt the current run
- **Reset**: Clear logs and reset dashboard state

### Dashboard Endpoints

The Flask server exposes these API endpoints:

- **GET** `/api/status`: Current training state
- **POST** `/api/train`: Start training with JSON payload
- **POST** `/api/stop`: Stop the active training run
- **GET** `/api/events`: Server-Sent Events stream for live updates
- **GET** `/api/logs`: Historical log entries
- **POST** `/api/reset`: Clear state and logs

### Custom Dashboard Configuration

Set environment variables to customize the server:

```bash
export UNO_DASH_HOST=0.0.0.0      # Listen on all interfaces
export UNO_DASH_PORT=9000         # Change port
export UNO_DASH_DEBUG=1           # Enable Flask debug mode

python dashboard_server.py
```

---

## 🏗️ Architecture Overview

### Components

**1. UNO Engine** (`uno_engine/`)
- Deterministic state machine for UNO 2v2
- Supports Classic and Go Wild modes
- Full ruleset: Wild cards, Draw stacking, Reverse, Skip, Discard-All
- Validation for all moves with `InvalidMoveError` exceptions

**2. Scenario Generation** (`notebooks/uno_bnn_curriculum_converted.py`)
- **ScenarioForge**: Generates synthetic game states across 6 scenario types
- **ScenarioLabeler**: Labels scenarios with expert bot actions
- **Bot Policies**: Oracle, Aggressor, Supporter, Conservative, Random

**3. Feature Engineering**
- **StateEncoder**: 103-dimensional state vector
  - Hand encoding (color/type histograms, playability)
  - Discard pile and current color
  - Opponent hand counts
  - Pending actions (draw stacks, color choices)
  - Team context and turn order
- **ActionEncoder**: Maps 66+ discrete actions to/from tokens
  - Play actions: `PLAY_NUMBER_<COLOR>_<VALUE>`, `PLAY_SKIP_<COLOR>`, etc.
  - Wild actions: `PLAY_WILD_<CHOSEN_COLOR>`, `PLAY_WILD_DRAW_FOUR_<COLOR>`
  - Draw action: `DRAW`

**4. Bayesian Neural Network**
- **Architecture**: 103 → 256 (hidden) → 66+ (output)
- **Prior**: Diagonal Gaussian with learnable mean/std
- **Posterior**: Variational guide (mean-field approximation)
- **Loss**: ELBO (Evidence Lower Bound) with KL divergence regularization
- **Inference**: Monte Carlo sampling (40 samples) for predictive distribution

**5. Training Pipeline**
- Curriculum learning with scenario type balancing
- Color permutation augmentation (4x data efficiency)
- Validation split (20% hold-out)
- CSV and text log outputs
- Model checkpointing with metadata

**6. Interactive Console** (`play_bnn_console.py`)
- Integrates UnoEngine + trained BNN artifacts
- Real-time inference with uncertainty quantification
- Human-in-the-loop gameplay against bots

**7. Web Dashboard** (`dashboard_server.py`)
- Flask backend with SSE for live updates
- Subprocess management for training isolation
- Graceful shutdown with SIGINT/SIGTERM handling

### Data Flow

```
┌─────────────────┐
│ ScenarioForge   │  Generate synthetic game states
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ScenarioLabeler │  Label with expert bot actions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ StateEncoder    │  Convert to 103-D feature vectors
│ ActionEncoder   │  Convert actions to class indices
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bayesian NN     │  Train with variational inference
│ (PyTorch/Pyro)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Export Artifacts│  Save param_store.pt + meta.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Interactive Play│  Load model → MC inference → gameplay
└─────────────────┘
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_engine.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=uno_engine --cov=notebooks
```

### Test Coverage

- **`tests/test_engine.py`**: Comprehensive game engine validation
  - Card play validation
  - Draw mechanics
  - Wild card color selection
  - Draw stacking
  - Team scoring
  - Win conditions
  
- **`tests/test_bnn_encoders.py`**: Encoder and BNN component tests (if exists)
  - Action encoding/decoding round-trips
  - State feature dimension consistency
  - Token parsing edge cases

### Adding Tests

To add new tests, create files in `tests/` following the pattern:

```python
# tests/test_my_feature.py
import pytest
from uno_engine import UnoEngine, GameMode
from notebooks.uno_bnn_curriculum_converted import StateEncoder

def test_my_feature():
    engine = UnoEngine()
    # ... test implementation
    assert expected_value == actual_value
```

---

## ⚠️ Known Issues

### Critical: Class Imbalance in Training Data

**Status**: 🔴 **CRITICAL - Affects Model Quality**

The trained BNN has a tendency to over-predict "Draw a card" actions due to class imbalance in the synthetic training data. This is thoroughly documented in `CRITIQUE.md`.

**Summary**:
- Training data includes `DRAW` as a valid action even when playable cards exist
- RandomBot (when enabled) selects uniformly from all actions, amplifying DRAW frequency
- Model learns to predict DRAW with 30-70% probability even with valid plays
- Validation accuracy remains near random guessing (1-2% vs. 1.5% baseline)

**Recommended Fixes** (see `CRITIQUE.md` for details):
1. Remove `DRAW` from enumerated actions when playable cards exist
2. Disable or reduce `RandomBot` contribution (`--include-random-bot=False`)
3. Add class balancing or weighted loss functions
4. Retrain with corrected data distribution

**Workaround for Current Models**:
- Use models for analysis and uncertainty visualization, not optimal play
- Compare BNN suggestions with heuristic bot decisions
- Focus on uncertainty metrics (entropy, MI) rather than exact action recommendations

### Other Issues

See `CRITIQUE.md` for a comprehensive analysis of:
- Encoder silent failures (🔴 Critical)
- DISCARD_ALL token parsing bug (🔴 Critical)
- Deterministic seeding ineffectiveness (🟡 High)
- Device selection limitations (🟡 High)
- Missing BNN pipeline tests (🟡 High)
- Artifact versioning gaps (🟢 Medium)

---

## 🔧 Troubleshooting

### Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'torch'`

```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Reinstall PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Problem**: `ImportError: cannot import name 'UnoEngine'`

```bash
# Ensure you're in the workspace directory
cd /workspace

# Verify uno_engine is present
ls -la uno_engine/
```

### Training Issues

**Problem**: `RuntimeError: CUDA out of memory`

```bash
# Reduce batch size
python train_local_bnn.py --preset quick --batch-size 64

# Or force CPU
python train_local_bnn.py --preset quick --device cpu
```

**Problem**: Training seems extremely slow

```bash
# Use quick preset for validation
python train_local_bnn.py --preset quick

# Disable progress bar on remote/headless systems
python train_local_bnn.py --preset balanced --no-progress
```

**Problem**: Validation accuracy stuck near 1-2%

This is expected due to the class imbalance issue. See **Known Issues** above. The model is learning, but converging to predict the majority class (DRAW). Consider applying the fixes in `CRITIQUE.md`.

### Console Play Issues

**Problem**: `No exported models found`

```bash
# Train a model first with --export
python train_local_bnn.py --preset quick --export

# Or specify model paths explicitly
python play_bnn_console.py --meta models/your_model_meta.json \
                           --param-store models/your_model_param_store.pt
```

**Problem**: `KeyError: 'DISCARD_ALL'` or similar token errors

This is a known bug (see `CRITIQUE.md`, Issue #4). Workaround: Avoid Go Wild mode until the encoder is patched, or apply the fix from the critique document.

### Dashboard Issues

**Problem**: Dashboard won't start - `ModuleNotFoundError: No module named 'flask'`

```bash
pip install flask
```

**Problem**: Training doesn't start from dashboard

Check the browser console and server logs:
```bash
python dashboard_server.py
# Look for error messages in the terminal
```

Ensure `train_local_bnn.py` is executable:
```bash
python train_local_bnn.py --help
```

**Problem**: Dashboard shows "Trainer already running" but nothing is active

```bash
# Reset the dashboard state
curl -X POST http://127.0.0.1:8000/api/reset

# Or restart the server
# Ctrl+C to stop, then:
python dashboard_server.py
```

### General Debug Tips

**Enable detailed logging**:
```bash
python train_local_bnn.py --preset quick --log-level DEBUG
```

**Check dependency versions**:
```bash
python -c "import torch, pyro; print(f'PyTorch {torch.__version__}, Pyro {pyro.__version__}')"
```

**Verify CUDA availability**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 Advanced Usage

### Custom Scenario Mixes

Adjust the curriculum to focus on specific tactical situations:

```bash
# Focus on end-game situations
python train_local_bnn.py --preset balanced \
                          --scenario-mix "FINISHER=0.6,DEFENDER=0.2,AGGRESSOR=0.2" \
                          --export

# Balanced tactical training
python train_local_bnn.py --preset max \
                          --scenario-mix "FINISHER=0.25,DEFENDER=0.25,AGGRESSOR=0.25,CRITICAL=0.25" \
                          --export
```

### Reproducible Training

```bash
# Set all random seeds for reproducibility
python train_local_bnn.py --preset balanced \
                          --seed 42 \
                          --rng-seed 42 \
                          --export

# Note: CUDA operations may still be non-deterministic
# See Known Issues for details
```

### Multi-GPU Training

```bash
# Select specific GPU
python train_local_bnn.py --preset max --device cuda:0

# Note: Multi-GPU parallelism is not currently implemented
# Training runs on a single device
```

### Batch Inference

To evaluate a trained model on a batch of game states programmatically:

```python
from play_bnn_console import load_bnn_artifacts, discover_latest_model, DEFAULT_MODELS_DIR
from notebooks.uno_bnn_curriculum_converted import evaluate_state_with_bnn, ScenarioType
import torch

# Load model
meta_path, param_path = discover_latest_model(DEFAULT_MODELS_DIR)
artifacts = load_bnn_artifacts(meta_path, param_path, device=torch.device('cpu'))

# Evaluate state
result = evaluate_state_with_bnn(
    artifacts,
    state=game_state,           # GameState object
    target_player="P0",
    persona_name="OracleBot",
    scenario_type=ScenarioType.SETUP,
    metadata={},
    num_samples=40
)

# Extract predictions
mean_probs = result["mc"]["mean_probs"]
std_probs = result["mc"]["std_probs"]
entropy = result["mc"]["predictive_entropy"]
mutual_info = result["mc"]["mutual_information"]
```

### Exporting Models for Deployment

Models are automatically exported with `--export`:

```bash
python train_local_bnn.py --preset max --export \
                          --model-tag my_production_model \
                          --output-dir ./deployed_models/
```

This creates:
- `my_production_model_param_store.pt`: PyTorch parameter store
- `my_production_model_meta.json`: Metadata for compatibility checking

Load in your application:
```python
from play_bnn_console import load_bnn_artifacts
artifacts = load_bnn_artifacts(meta_path, param_path)
```

---

## 🤝 Contributing

### Areas for Improvement

**High Priority**:
1. **Fix class imbalance** in training data (see `CRITIQUE.md`)
2. **Add BNN pipeline tests** (encoders, inference, dataset generation)
3. **Implement class balancing** in loss function
4. **Fix DISCARD_ALL token parsing** bug

**Medium Priority**:
5. Improve deterministic seeding (move from module-level to CLI)
6. Add artifact schema versioning
7. Support Apple Silicon MPS backend
8. Implement multi-GPU training

**Low Priority**:
9. Add data augmentation strategies (beyond color permutation)
10. Experiment with alternative BNN architectures
11. Implement online learning / fine-tuning
12. Add Tensorboard integration for training visualization

### Development Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/my-improvement

# 2. Make changes and add tests
# Edit code...
pytest  # Ensure tests pass

# 3. Run linting (if configured)
# black . --check
# mypy uno_engine/

# 4. Commit and push
git add .
git commit -m "Add improvement: description"
git push origin feature/my-improvement

# 5. Create pull request
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to public functions and classes
- Keep functions focused and modular
- Write tests for new features

---

## 📚 Additional Resources

### Documentation Files

- **`CRITIQUE.md`**: Comprehensive production-readiness analysis with known issues and fixes
- **`requirements.txt`**: Python package dependencies
- **Training logs**: See `training_logs/` for historical run data

### Related Concepts

- **Bayesian Neural Networks**: [Pyro documentation](http://pyro.ai/)
- **Variational Inference**: [ELBO Tutorial](http://pyro.ai/examples/svi_part_i.html)
- **UNO Rules**: [Official UNO Rules](https://www.unorules.com/)

---

## 📝 License

This project is provided as-is for educational and research purposes. Please review the license file (if present) for usage terms.

---

## 🙏 Acknowledgments

- **PyTorch** and **Pyro** teams for the deep learning framework
- **UNO** by Mattel for the game concept
- Community contributors and testers

---

## 📧 Support

For questions, issues, or suggestions:

1. Check **`CRITIQUE.md`** for known issues and solutions
2. Review the **Troubleshooting** section above
3. Search existing issues in the repository
4. Open a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - System information (Python version, OS, GPU/CPU)
   - Relevant logs or error messages

---

**Happy training! 🎉🎲🤖**

*Remember*: The current models have known class imbalance issues (see `CRITIQUE.md`). Use them for learning and experimentation, but apply the recommended fixes before production deployment.
