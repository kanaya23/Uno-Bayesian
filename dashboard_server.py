"""Dashboard backend wiring UNO training to the HTML console.

This module exposes a small Flask application that serves the
`uno_bnn_training_dashboard.html` interface and provides APIs to
launch `train_local_bnn.py`, stream structured logs, and surface
progress updates for the front-end widgets.

The server runs entirely locally and keeps a single training run in
flight at a time. Training output is captured from the trainer's
stdout/stderr streams and rebroadcast to connected EventSource clients
for a near-real-time experience.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from itertools import count
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)


BASE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = BASE_DIR / "train_local_bnn.py"

app = Flask(__name__)


STATUS_LABELS = {
    "idle": "Idle",
    "starting": "Starting",
    "preparing": "Preparing dataset",
    "running": "Training",
    "stopping": "Stopping",
    "stopped": "Stopped",
    "exporting": "Exporting artifacts",
    "completed": "Completed",
    "failed": "Failed",
}


PRESET_DEFAULTS = {
    "quick": {"epochs": 15, "num_scenarios": 1200, "batch_size": 96, "learning_rate": 0.002},
    "balanced": {"epochs": 30, "num_scenarios": 4000, "batch_size": 128, "learning_rate": 0.001},
    "max": {"epochs": 50, "num_scenarios": 6400, "batch_size": 192, "learning_rate": 0.0005},
    "stable": {"epochs": 100, "num_scenarios": 32000, "batch_size": 192, "learning_rate": 0.0001},
}


def _default_state() -> Dict[str, Any]:
    return {
        "status": "idle",
        "status_text": STATUS_LABELS["idle"],
        "percent": 0.0,
        "epoch_current": 0,
        "epoch_total": 0,
        "eta_seconds": None,
        "started_at": None,
        "finished_at": None,
        "preset": None,
        "preset_label": None,
        "scenarios": None,
        "batch_size": None,
        "learning_rate": None,
        "run_name": None,
        "progress_detail": "Awaiting run...",
        "log_path": None,
        "csv_path": None,
        "error": None,
        "command": None,
    }


state_lock = threading.Lock()
training_state: Dict[str, Any] = _default_state()


def _make_snapshot() -> Dict[str, Any]:
    with state_lock:
        snapshot = dict(training_state)
    return snapshot


def _reset_state() -> Dict[str, Any]:
    with state_lock:
        training_state.clear()
        training_state.update(_default_state())
        snapshot = dict(training_state)
    return snapshot


def _update_state(**updates: Any) -> Dict[str, Any]:
    with state_lock:
        training_state.update(updates)
        status = training_state.get("status", "idle")
        training_state["status_text"] = STATUS_LABELS.get(status, status.title())
        if status == "idle" and not updates.get("progress_detail"):
            training_state["progress_detail"] = "Awaiting run..."
        snapshot = dict(training_state)
    _broadcast("status", snapshot)
    return snapshot


def _serialise_state(state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resolved = dict(state or _make_snapshot())
    # Ensure everything is JSON serialisable.
    if isinstance(resolved.get("command"), list):
        resolved["command"] = " ".join(resolved["command"])
    return resolved


LOG_HISTORY = deque(maxlen=2000)


def _clear_history() -> None:
    LOG_HISTORY.clear()


SubscriberQueue = Queue
_subscribers: Dict[int, SubscriberQueue] = {}
_subscriber_id = count()
_subscribers_lock = threading.Lock()


def _subscribe() -> Tuple[int, SubscriberQueue]:
    queue: SubscriberQueue = Queue(maxsize=1000)
    ident = next(_subscriber_id)
    with _subscribers_lock:
        _subscribers[ident] = queue
    return ident, queue


def _unsubscribe(ident: int) -> None:
    with _subscribers_lock:
        _subscribers.pop(ident, None)


def _broadcast(event_type: str, payload: Dict[str, Any]) -> None:
    message = json.dumps(payload, ensure_ascii=False)
    with _subscribers_lock:
        subscribers = list(_subscribers.items())
    for ident, queue in subscribers:
        try:
            queue.put_nowait((event_type, message))
        except Full:
            # Drop stale events for slow consumers to avoid back-pressure.
            try:
                while True:
                    queue.get_nowait()
            except Empty:
                pass
            try:
                queue.put_nowait((event_type, message))
            except Full:
                # Give up on hopelessly slow consumer.
                _unsubscribe(ident)


def _broadcast_log(entry: Dict[str, Any]) -> None:
    LOG_HISTORY.append(entry)
    _broadcast("log", entry)


def _format_event(event_type: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


training_process: Optional[subprocess.Popen[str]] = None
training_thread: Optional[threading.Thread] = None
epoch_durations: List[float] = []

EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(?P<current>\d+)\s*/\s*(?P<total>\d+)\s*\|\s*train\s+(?P<train>[0-9.]+)\s*\|\s*val\s+(?P<val>[0-9.]+)\s*\|\s*acc\s+(?P<acc>[0-9.]+)\s*\|\s*(?P<seconds>[0-9.]+)s",
    re.IGNORECASE,
)
LEVEL_RE = re.compile(r"\|\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|")
RUN_NAME_RE = re.compile(r"\[(?P<run>[^\]]+)\]")
CSV_PATH_RE = re.compile(r"Metrics logged to (?P<path>.+\.csv)")
TEXT_LOG_RE = re.compile(r"text_log:\s*(?P<path>.+)$", re.IGNORECASE)


def _parse_level(line: str) -> str:
    match = LEVEL_RE.search(line)
    if match:
        return match.group(1)
    return "INFO"


def _ensure_trainer_exists() -> None:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"train_local_bnn.py not found at {TRAIN_SCRIPT}")


def _estimate_eta() -> Optional[float]:
    if not epoch_durations:
        return None
    state = _make_snapshot()
    total = state.get("epoch_total") or 0
    current = state.get("epoch_current") or 0
    if not total or current >= total:
        return None
    avg = sum(epoch_durations) / max(len(epoch_durations), 1)
    remaining = total - current
    return max(0.0, avg * remaining)


def _handle_epoch_line(line: str) -> None:
    match = EPOCH_LINE_RE.search(line)
    if not match:
        return
    current = int(match.group("current"))
    total = int(match.group("total"))
    train_loss = float(match.group("train"))
    val_loss = float(match.group("val"))
    accuracy = float(match.group("acc"))
    seconds = float(match.group("seconds"))
    epoch_durations.append(seconds)
    percent = min(100.0, round((current / total) * 100, 2)) if total > 0 else 0.0

    run_name_match = RUN_NAME_RE.search(line)
    run_name = run_name_match.group("run") if run_name_match else None

    eta = _estimate_eta()

    detail = (
        f"Epoch {current}/{total}"
        f" | train {train_loss:.4f} | val {val_loss:.4f} | acc {accuracy:.3f}"
    )

    updates: Dict[str, Any] = {
        "status": "running",
        "epoch_current": current,
        "epoch_total": total,
        "percent": percent,
        "progress_detail": detail,
        "eta_seconds": eta,
    }
    if run_name:
        updates.setdefault("run_name", run_name)

    _update_state(**updates)


def _process_line(line: str) -> None:
    stripped = line.rstrip()
    if not stripped:
        return

    level = _parse_level(stripped)
    entry = {
        "timestamp": time.time(),
        "message": stripped,
        "level": level,
    }
    _broadcast_log(entry)

    lower = stripped.lower()
    if "preparing synthetic dataset" in lower:
        _update_state(status="preparing", progress_detail="Generating synthetic scenarios...")
    elif "dataset ready" in lower:
        _update_state(progress_detail="Dataset ready - initializing trainer...")
    elif "exporting artifacts" in lower:
        _update_state(status="exporting", progress_detail="Exporting trained artifacts...")
    elif "training complete" in lower:
        _update_state(status="completed", percent=100.0, progress_detail="Training complete", eta_seconds=0.0)
    elif "training interrupted" in lower:
        _update_state(status="stopped", progress_detail="Training interrupted by user", eta_seconds=None)

    csv_match = CSV_PATH_RE.search(stripped)
    if csv_match:
        _update_state(csv_path=csv_match.group("path").strip())

    text_log_match = TEXT_LOG_RE.search(stripped)
    if text_log_match:
        _update_state(log_path=text_log_match.group("path").strip())

    _handle_epoch_line(stripped)


def _reader_loop(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for raw_line in iter(process.stdout.readline, ""):
        _process_line(raw_line)
    process.stdout.close()
    returncode = process.wait()

    snapshot = _make_snapshot()
    status = snapshot.get("status")
    if returncode == 0 and status not in {"completed", "stopped"}:
        _update_state(status="completed", percent=100.0, progress_detail="Training finished successfully", eta_seconds=0.0)
    elif returncode != 0 and status not in {"stopped"}:
        _update_state(status="failed", error=f"Trainer exited with code {returncode}", eta_seconds=None)

    _update_state(finished_at=time.time())


def _active_status() -> bool:
    snapshot = _make_snapshot()
    return snapshot.get("status") in {"starting", "preparing", "running", "exporting", "stopping"}


def _build_command(payload: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    preset = payload.get("preset", "balanced") or "balanced"
    preset_settings = PRESET_DEFAULTS.get(preset.lower(), PRESET_DEFAULTS["balanced"])

    epochs = int(payload.get("epochs") or preset_settings["epochs"])
    num_scenarios = int(payload.get("num_scenarios") or preset_settings["num_scenarios"])
    batch_size = int(payload.get("batch_size") or preset_settings["batch_size"])
    learning_rate = float(payload.get("learning_rate") or preset_settings["learning_rate"])
    mode = str(payload.get("mode") or "CLASSIC_2V2")
    log_level = str(payload.get("log_level") or "INFO")
    progress_bar = bool(payload.get("progress", True))
    export = bool(payload.get("export", False))
    include_random_bot = bool(payload.get("include_random_bot", False))
    random_bot_fraction = float(payload.get("random_bot_fraction", 0.05))
    if not 0.0 <= random_bot_fraction < 1.0:
        random_bot_fraction = max(0.0, min(random_bot_fraction, 0.99))
    run_name = payload.get("run_name")
    device = payload.get("device", "auto")

    command = [sys.executable, "-u", str(TRAIN_SCRIPT)]

    if preset:
        command.extend(["--preset", preset])

    command.extend([
        "--epochs",
        str(epochs),
        "--num-scenarios",
        str(num_scenarios),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        f"{learning_rate}",
        "--mode",
        mode,
        "--log-level",
        log_level,
        "--random-bot-fraction",
        str(random_bot_fraction),
        "--device",
        str(device),
    ])

    if not progress_bar:
        command.append("--no-progress")
    if export:
        command.append("--export")
    if include_random_bot:
        command.append("--include-random-bot")
    if run_name:
        command.extend(["--run-name", str(run_name)])

    # Optional parameters pass-through.
    for field in ("log_dir", "output_dir", "model_tag", "scenario_mix", "seed", "rng_seed"):
        value = payload.get(field)
        if value not in (None, ""):
            command.extend([f"--{field.replace('_', '-')}", str(value)])

    return command, {
        "preset": preset,
        "preset_label": preset.capitalize(),
        "epochs": epochs,
        "num_scenarios": num_scenarios,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "mode": mode,
    }


def _launch_training(command: List[str], config: Dict[str, Any]) -> None:
    global training_process, training_thread, epoch_durations

    _ensure_trainer_exists()

    if training_process is not None and training_process.poll() is None:
        raise RuntimeError("Training already in progress")

    epoch_durations = []
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    training_process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=creationflags,
    )

    _update_state(
        status="starting",
        started_at=time.time(),
        finished_at=None,
        error=None,
        percent=0.0,
        epoch_current=0,
        epoch_total=config["epochs"],
        eta_seconds=None,
        preset=config["preset"],
        preset_label=config["preset_label"],
        scenarios=config["num_scenarios"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        command=command,
        progress_detail="Launching trainer...",
    )

    _broadcast_log(
        {
            "timestamp": time.time(),
            "message": f"Launching training: {' '.join(command)}",
            "level": "INFO",
        }
    )

    training_thread = threading.Thread(target=_reader_loop, args=(training_process,), daemon=True)
    training_thread.start()


@app.route("/")
def index() -> Any:
    return send_from_directory(str(BASE_DIR), "uno_bnn_training_dashboard.html")


@app.route("/api/status", methods=["GET"])
def api_status() -> Response:
    return jsonify(_serialise_state())


@app.route("/api/train", methods=["POST"])
def api_train() -> Response:
    if _active_status():
        return jsonify({"error": "Trainer already running."}), 409

    payload = request.get_json(silent=True) or {}

    try:
        command, config = _build_command(payload)
    except Exception as exc:  # pragma: no cover - defensive parsing
        return jsonify({"error": f"Invalid training parameters: {exc}"}), 400

    _clear_history()

    try:
        _launch_training(command, config)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:  # pragma: no cover - startup failure guard
        return jsonify({"error": f"Failed to start training: {exc}"}), 500

    return jsonify({"ok": True, "state": _serialise_state()})


def _graceful_stop(timeout: float = 5.0) -> bool:
    global training_process
    proc = training_process
    if proc is None or proc.poll() is not None:
        return True

    _update_state(status="stopping", progress_detail="Stopping trainer...", eta_seconds=None)

    try:
        if os.name == "nt":
            proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
        else:
            proc.send_signal(signal.SIGINT)
    except Exception:
        proc.terminate()

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            training_process = None
            return True
        time.sleep(0.2)

    proc.terminate()
    time.sleep(0.5)
    if proc.poll() is None:
        proc.kill()

    training_process = None
    return False


@app.route("/api/stop", methods=["POST"])
def api_stop() -> Response:
    if not _active_status():
        return jsonify({"error": "No active training run."}), 409

    graceful = _graceful_stop()
    if graceful:
        _update_state(status="stopped", progress_detail="Training stopped by user", finished_at=time.time(), eta_seconds=None)
    else:
        _update_state(status="failed", progress_detail="Trainer force-killed", finished_at=time.time(), eta_seconds=None)

    return jsonify({"ok": graceful, "state": _serialise_state()})


@app.route("/api/events", methods=["GET"])
def api_events() -> Response:
    ident, queue = _subscribe()

    def stream() -> Iterable[str]:
        try:
            yield _format_event("status", _serialise_state())
            for entry in LOG_HISTORY:
                yield _format_event("log", entry)
            while True:
                try:
                    event_type, message = queue.get(timeout=15.0)
                except Empty:
                    yield ":keepalive\n\n"
                    continue
                payload = json.loads(message)
                yield _format_event(event_type, payload)
        finally:
            _unsubscribe(ident)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(stream()), mimetype="text/event-stream", headers=headers)


@app.route("/api/logs", methods=["GET"])
def api_logs() -> Response:
    return jsonify(list(LOG_HISTORY))


@app.route("/api/reset", methods=["POST"])
def api_reset() -> Response:
    if _active_status():
        return jsonify({"error": "Cannot reset while training is running."}), 409
    _clear_history()
    snapshot = _reset_state()
    _broadcast("status", snapshot)
    return jsonify({"ok": True, "state": snapshot})


if __name__ == "__main__":
    # Provide a friendly CLI for manual launches.
    host = os.environ.get("UNO_DASH_HOST", "127.0.0.1")
    port = int(os.environ.get("UNO_DASH_PORT", "8000"))
    debug = os.environ.get("UNO_DASH_DEBUG", "0") == "1"
    print(f"[dashboard] Serving UNO BNN dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)

