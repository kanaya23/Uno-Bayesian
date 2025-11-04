"""Console-based UNO 2v2 interface with Bayesian NN analysis.

This script wires together the deterministic `uno_engine` package with the
Bayesian neural network artifacts exported by the training pipeline. It allows
human players to face off against heuristic teammates/opponents in a classic or
Go Wild 2v2 UNO round while inspecting the BNN's live recommendations and
uncertainty metrics directly from the console.

The implementation intentionally keeps runtime dependencies identical to the
training stack: PyTorch, Pyro and NumPy. It detects exported artifacts in the
`models/` directory automatically but also exposes CLI overrides for custom
paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pyro
import torch

from uno_engine import UnoEngine, GameMode, InvalidMoveError
from uno_engine.models import Card, CardType, Color, GameState, PendingActionType, Player

try:
    from notebooks.uno_bnn_curriculum_converted import (
        ActionEncoder,
        ActionType,
        AggressorBot,
        ARTIFACT_SCHEMA_VERSION,
        BNNBot,
        BOT_ORDER,
        BotAction,
        BotPolicy,
        ConservativeBot,
        ScenarioType,
        StateBNNGuide,
        StateBayesianNN,
        StateEncoder,
        action_encoder_signature,
        SupporterBot,
        TrainingArtifacts,
        state_encoder_signature,
        evaluate_state_with_bnn,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    missing = exc.name or "required dependency"
    raise SystemExit(
        "Uno BNN console interface requires the converted notebook module.\n"
        "Please ensure project dependencies are installed and that the"
        " `notebooks.uno_bnn_curriculum_converted` module is importable.\n"
        f"Original error: {exc} (missing: {missing})"
    ) from exc


DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"

logger = logging.getLogger(__name__)


@dataclass
class PredictionEntry:
    token: str
    probability: float
    std_dev: float
    is_valid: bool
    action: Optional[BotAction]


@dataclass
class BNNAnalysis:
    scenario_type: ScenarioType
    entropy: float
    mutual_information: float
    entries: Sequence[PredictionEntry]
    suggested_action: BotAction
    suggested_token: str
    suggested_probability: float
    suggested_std: float
    mc_samples: int
    mean_probability: float
    probability_std: float
    selection_reason: str


def discover_latest_model(models_dir: Path) -> Optional[Tuple[Path, Path]]:
    """Return the newest (meta, param_store) pair found in *models_dir*."""

    if not models_dir.exists():
        return None

    candidates: List[Tuple[float, Path, Path]] = []
    for meta_path in models_dir.glob("*_meta.json"):
        if not meta_path.is_file():
            continue
        stem = meta_path.stem  # e.g. local_CLASSIC_2V2_20251103-160520_meta
        if not stem.endswith("_meta"):
            continue
        param_name = stem[:-5] + "_param_store.pt"
        param_path = meta_path.with_name(param_name)
        if not param_path.exists():
            continue
        candidates.append((meta_path.stat().st_mtime, meta_path, param_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, meta_path, param_path = candidates[0]
    return meta_path, param_path


def load_bnn_artifacts(
    meta_path: Path,
    param_store_path: Path,
    *,
    device: Optional[torch.device] = None,
) -> TrainingArtifacts:
    """Instantiate the BNN model + guide and hydrate weights from disk."""

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
    if not param_store_path.exists():
        raise FileNotFoundError(f"Param store file not found: {param_store_path}")

    with meta_path.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)

    feature_size = int(metadata["feature_size"])
    num_actions = int(metadata["num_actions"])
    action_tokens: Dict[int, str] = {
        int(index): token for index, token in metadata["action_tokens"].items()
    }

    schema_version = metadata.get("schema_version")
    if schema_version is None:
        raise RuntimeError("Artifact metadata missing 'schema_version'. Retrain with the latest tooling.")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Artifact schema version '{schema_version}' is incompatible with runtime version '{ARTIFACT_SCHEMA_VERSION}'."
        )

    pyro.clear_param_store()

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    player_order_meta = metadata.get("player_order")
    state_encoder = StateEncoder(player_order=player_order_meta)
    if state_encoder.feature_size != feature_size:
        raise RuntimeError(
            "Loaded state encoder feature dimension does not match metadata "
            f"(expected {feature_size}, got {state_encoder.feature_size})."
        )

    expected_state_sig = metadata.get("state_encoder_signature")
    team_map_meta = metadata.get("team_map")
    computed_state_sig = state_encoder_signature(state_encoder, team_map_meta)
    if expected_state_sig:
        if computed_state_sig != expected_state_sig:
            raise RuntimeError(
                "State encoder signature mismatch between runtime and artifact metadata."
            )
    else:
        logger.warning("Artifact metadata missing state_encoder_signature; skipping compatibility check.")

    action_encoder = ActionEncoder()
    action_encoder.lookup = {}
    action_encoder.reverse = {}
    for index in sorted(action_tokens):
        token = action_tokens[index]
        action_encoder.lookup[token] = index
        action_encoder.reverse[index] = token

    expected_action_sig = metadata.get("action_encoder_signature")
    computed_action_sig = action_encoder_signature(action_encoder)
    if expected_action_sig:
        if computed_action_sig != expected_action_sig:
            raise RuntimeError("Action encoder signature mismatch between runtime and artifact metadata.")
    else:
        logger.warning("Artifact metadata missing action_encoder_signature; skipping compatibility check.")

    model = StateBayesianNN(feature_size, num_actions, device=device).to(device)
    guide = StateBNNGuide(model, device=device).to(device)

    pyro.get_param_store().load(str(param_store_path), map_location=device)

    model.set_kl_scale(1.0)
    guide.set_kl_scale(1.0)

    history = metadata.get("history", {})
    artifacts = TrainingArtifacts(
        model=model,
        guide=guide,
        history=history,
        action_encoder=action_encoder,
        state_encoder=state_encoder,
        metadata=metadata,
        log_path=None,
    )
    return artifacts


def format_card(card: Card) -> str:
    """Return a human-friendly string representing *card*."""

    color = card.color.value.title() if card.color else "?"
    if card.type == CardType.NUMBER:
        return f"{color} {card.number}"
    if card.type == CardType.WILD:
        return "Wild"
    if card.type == CardType.WILD_DRAW_FOUR:
        return "Wild Draw Four"
    if card.type == CardType.DRAW_TWO:
        return f"Draw Two ({color})"
    if card.type == CardType.REVERSE:
        return f"Reverse ({color})"
    if card.type == CardType.SKIP:
        return f"Skip ({color})"
    if card.type == CardType.DISCARD_ALL:
        return f"Discard All ({color})"
    return f"{card.type.name.title()} ({color})"


def describe_token(token: str) -> str:
    token = token.upper()
    if token == "DRAW":
        return "Draw a card"
    if token == "PASS":
        return "Pass"
    parts = token.split("_")
    if len(parts) < 2:
        return token
    if parts[1] == "NUMBER" and len(parts) == 4:
        color = parts[2].title()
        number = parts[3]
        return f"Play {color} {number}"
    if parts[1] in {"SKIP", "REVERSE"} and len(parts) == 3:
        return f"Play {parts[1].title()} ({parts[2].title()})"
    if parts[1] == "DRAW" and len(parts) == 4:
        return f"Play Draw Two ({parts[3].title()})"
    if parts[1] == "DISCARD" and len(parts) == 4:
        return f"Play Discard All ({parts[3].title()})"
    if parts[1] == "WILD" and len(parts) == 3:
        return f"Play Wild choosing {parts[2].title()}"
    if parts[1] == "WILD" and len(parts) == 5 and parts[2] == "DRAW":
        return f"Play Wild Draw Four choosing {parts[4].title()}"
    return token


class ConsoleUnoBNNInterface:
    """Stateful controller for running interactive UNO rounds."""

    _CANONICAL_PLAYERS = ("P0", "P1", "P2", "P3")

    def __init__(
        self,
        artifacts: TrainingArtifacts,
        *,
        persona: str = "OracleBot",
        mc_samples: int = 40,
        user_name: str = "You",
        teammate_name: str = "AllyBot",
        opponent_names: Sequence[str] = ("NorthBot", "EastBot"),
    ) -> None:
        if persona not in BOT_ORDER:
            raise ValueError(
                f"Persona '{persona}' is unknown. Choose one of: {', '.join(BOT_ORDER)}"
            )

        if len(opponent_names) != 2:
            raise ValueError("Exactly two opponent names are required for 2v2 play.")

        self.artifacts = artifacts
        self.persona = persona
        self.mc_samples = mc_samples
        self.engine = UnoEngine()

        # Canonical identifiers expected by the BNN stack (P0..P3)
        self.user_id = "P0"
        self.first_opponent_id = "P1"
        self.teammate_id = "P2"
        self.second_opponent_id = "P3"

        self.display_names: Dict[str, str] = {
            self.user_id: user_name,
            self.teammate_id: teammate_name,
            self.first_opponent_id: opponent_names[0],
            self.second_opponent_id: opponent_names[1],
        }

        self._rng = random.Random(42)
        self._bnn_helper = BNNBot(
            artifacts,
            persona_hint=persona,
            rng=random.Random(101),
            mc_samples=mc_samples,
        )

        self.bot_map: Dict[str, BotPolicy] = {
            self.teammate_id: SupporterBot(rng=random.Random(7)),
            self.first_opponent_id: AggressorBot(rng=random.Random(11)),
            self.second_opponent_id: ConservativeBot(rng=random.Random(19)),
        }

        self._player_roles: Dict[str, str] = {
            self.user_id: "You",
            self.teammate_id: "Teammate",
            self.first_opponent_id: "Opponent",
            self.second_opponent_id: "Opponent",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, initial_mode: Optional[GameMode] = None) -> None:
        """Run one or more rounds until the user opts out."""

        mode = initial_mode or self._prompt_for_mode()
        keep_playing = True

        while keep_playing:
            state = self._start_round(mode)
            state = self._play_round(state)
            self._display_round_results(state)

            answer = input("Play another round? [y/N]: ").strip().lower()
            if answer in {"y", "yes"}:
                mode = self._prompt_for_mode()
            else:
                keep_playing = False

    # ------------------------------------------------------------------
    # Core loop helpers
    # ------------------------------------------------------------------
    def _start_round(self, mode: GameMode) -> GameState:
        players = [
            Player(player_id=self.user_id),
            Player(player_id=self.first_opponent_id),
            Player(player_id=self.teammate_id),
            Player(player_id=self.second_opponent_id),
        ]
        state = self.engine.init_game(players, mode=mode)
        print("\n============================================================")
        print(f"Starting new round ({mode.value.replace('_', ' ')})")
        print("Teams: {0} & {1} vs {2} & {3}".format(
            self.display_names[self.user_id],
            self.display_names[self.teammate_id],
            self.display_names[self.first_opponent_id],
            self.display_names[self.second_opponent_id],
        ))
        print("============================================================\n")
        return state

    def _play_round(self, state: GameState) -> GameState:
        while not state.round_over:
            state = self._resolve_pending_color_choice(state)
            if state.round_over:
                break

            current_player = state.current_player().player_id
            if current_player == self.user_id:
                state = self._handle_user_turn(state)
            else:
                state = self._handle_bot_turn(state, current_player)

        return state

    def _resolve_pending_color_choice(self, state: GameState) -> GameState:
        pending = state.pending_action
        if not pending or pending.type != PendingActionType.COLOR_CHOICE:
            return state

        chooser = pending.player_id
        if chooser == self.user_id:
            color = self._prompt_color_selection()
        else:
            bot = self.bot_map.get(chooser, self._bnn_helper)
            color = bot.choose_color(state, chooser)
            chooser_name = self.display_names.get(chooser, chooser)
            print(f"{chooser_name} chooses {color.value.title()} as the wild color.")

        try:
            state = self.engine.choose_color(state, chooser, color)
        except InvalidMoveError as exc:
            print(f"Color choice failed ({exc}); defaulting to RED.")
            state = self.engine.choose_color(state, chooser, Color.RED)
        return state

    # ------------------------------------------------------------------
    # Turn handlers
    # ------------------------------------------------------------------
    def _handle_user_turn(self, state: GameState) -> GameState:
        print("\n--- Your Turn --------------------------------------------------")
        self._display_common_state(state)
        analysis = self._analyze_state_with_bnn(state)
        self._display_bnn_analysis(state, analysis)

        while True:
            command = input("Action ([p]lay #, [d]raw, [pa]ss, [q]uit): ").strip().lower()
            if command in {"q", "quit"}:
                raise SystemExit("Session terminated by user.")

            if command.startswith("p"):
                parts = command.split()
                if len(parts) == 1:
                    try:
                        index = int(input("Enter card index to play: ").strip())
                    except ValueError:
                        print("Please supply a numeric card index.")
                        continue
                else:
                    try:
                        index = int(parts[1])
                    except ValueError:
                        print("Usage: p <index>")
                        continue
                state, success = self._attempt_play(state, index)
                if success:
                    return state
                continue

            if command in {"d", "draw"}:
                had_valid_moves = bool(self.engine.get_valid_moves(state, self.user_id))
                try:
                    state = self.engine.draw_card(state, self.user_id)
                    print("You draw a card.")
                    if had_valid_moves:
                        print("You forfeited your turn because playable cards were available.")
                    return state
                except InvalidMoveError as exc:
                    print(f"Draw not allowed: {exc}")
                    continue

            if command in {"pa", "pass"}:
                try:
                    state = self.engine.pass_turn(state, self.user_id)
                    print("You pass the turn.")
                    return state
                except InvalidMoveError as exc:
                    print(f"Pass not allowed: {exc}")
                    continue

            print("Unrecognized command. Try again.")

    def _handle_bot_turn(self, state: GameState, player_id: str) -> GameState:
        bot = self.bot_map.get(player_id, self._bnn_helper)
        display_name = self.display_names.get(player_id, player_id)
        print(f"\n{display_name} ({self._player_roles.get(player_id, 'Bot')}) is thinking...")

        action = bot.decide(self.engine, state, player_id)

        if player_id == self.teammate_id:
            current_color = state.current_color or state.discard_pile[-1].color
            token = self.artifacts.action_encoder._to_token(action)
            print(
                f"[AllyBot] move diagnostic: {describe_token(token)} | Current color: {current_color.value.title()}"
            )
            logger.info(
                "AllyBot move | token=%s | color=%s | reason=%s | prob=%.3f",
                describe_token(token),
                current_color.value,
                action.info.get("selection_reason", "n/a"),
                action.info.get("bnn_probability", float("nan")),
            )

        if action.action_type == ActionType.PLAY and action.card is not None:
            chosen_color = action.chosen_color
            if action.card.type in {CardType.WILD, CardType.WILD_DRAW_FOUR} and chosen_color is None:
                chosen_color = bot.choose_color(state, player_id)
                action.chosen_color = chosen_color

        try:
            if action.action_type == ActionType.PLAY and action.card is not None:
                state = self.engine.play_card(state, player_id, action.card, action.chosen_color)
                description = format_card(action.card)
                if action.card.type in {CardType.WILD, CardType.WILD_DRAW_FOUR} and action.chosen_color:
                    description += f" (chooses {action.chosen_color.value.title()})"
                print(f"{display_name} plays {description}.")
            elif action.action_type == ActionType.DRAW:
                state = self.engine.draw_card(state, player_id)
                print(f"{display_name} draws a card.")
            elif action.action_type == ActionType.PASS:
                state = self.engine.pass_turn(state, player_id)
                print(f"{display_name} passes.")
        except InvalidMoveError as exc:
            print(f"Bot action invalid ({exc}); forcing draw.")
            try:
                state = self.engine.draw_card(state, player_id)
            except InvalidMoveError:
                pass
        return state

    # ------------------------------------------------------------------
    # User helpers
    # ------------------------------------------------------------------
    def _attempt_play(self, state: GameState, index: int) -> Tuple[GameState, bool]:
        hand = list(state.current_player().hand)
        if index < 1 or index > len(hand):
            print("Card index out of range.")
            return state, False

        card = hand[index - 1]
        chosen_color: Optional[Color] = None
        if card.type in {CardType.WILD, CardType.WILD_DRAW_FOUR}:
            chosen_color = self._prompt_color_selection()

        try:
            next_state = self.engine.play_card(state, self.user_id, card, chosen_color)
        except InvalidMoveError as exc:
            print(f"Cannot play that card: {exc}")
            return state, False

        print(f"You play {format_card(card)}" + (
            f" choosing {chosen_color.value.title()}" if chosen_color else ""
        ) + ".")
        return next_state, True

    def _prompt_color_selection(self) -> Color:
        while True:
            choice = input("Choose color [r/y/g/b]: ").strip().lower()
            mapping = {
                "r": Color.RED,
                "y": Color.YELLOW,
                "g": Color.GREEN,
                "b": Color.BLUE,
            }
            if choice in mapping:
                return mapping[choice]
            print("Please choose r, y, g, or b.")

    def _prompt_for_mode(self) -> GameMode:
        while True:
            choice = input("Select mode: [1] Classic 2v2, [2] Go Wild 2v2: ").strip()
            if choice in {"1", "classic", "c"}:
                return GameMode.CLASSIC_2V2
            if choice in {"2", "wild", "w"}:
                return GameMode.GO_WILD_2V2
            print("Please enter 1 (Classic) or 2 (Go Wild).")

    def _display_common_state(self, state: GameState) -> None:
        top_card = state.discard_pile[-1]
        current_color = state.current_color or top_card.color
        print(f"Top card: {format_card(top_card)} | Current color: {current_color.value.title()}")
        if state.pending_action:
            pending = state.pending_action
            if pending.type == PendingActionType.DRAW_STACK:
                print(
                    f"Draw stack active: {pending.draw_penalty} cards owed by {pending.player_id}."
                )
            elif pending.type == PendingActionType.DRAWN_CARD_PLAY_WINDOW:
                print("Drawn card window: you may play the drawn card or pass.")

        print("Hand:")
        hand = state.current_player().hand
        for idx, card in enumerate(hand, start=1):
            print(f"  [{idx}] {format_card(card)}")

        teammate_hand = next(
            player.hand for player in state.players if player.player_id == self.teammate_id
        )
        print(f"\n{self.display_names[self.teammate_id]}'s hand:")
        for card in teammate_hand:
            print(f"    - {format_card(card)}")

        print("Other players:")
        for player in state.players:
            if player.player_id == self.user_id:
                continue
            if player.player_id == self.teammate_id:
                continue
            role = self._player_roles.get(player.player_id, "Bot")
            name = self.display_names.get(player.player_id, player.player_id)
            print(f"  {name:>10} ({role}) - {len(player.hand)} cards")

    def _display_bnn_analysis(self, state: GameState, analysis: BNNAnalysis) -> None:
        print("\nBNN perspective:")
        print(
            f"  Scenario: {analysis.scenario_type.value} | "
            f"Entropy: {analysis.entropy:.3f} | MI: {analysis.mutual_information:.3f}"
        )
        print(
            "  Confidence: "
            f"{analysis.suggested_probability:.2%} +/- {analysis.suggested_std:.2%}"
            f" | MC={analysis.mc_samples}"
        )
        suggestion_desc = describe_token(analysis.suggested_token)
        print(f"  Suggested action: {suggestion_desc}")
        print(
            "  Distribution summary: "
            f"mu={analysis.mean_probability:.2%} +/- {analysis.probability_std:.2%}"
            f" | Selection trigger: {analysis.selection_reason}"
        )

        print("  Top predictions (mean +/- std):")
        for entry in analysis.entries[:3]:
            status = "valid" if entry.is_valid else "not in hand"
            print(
                "    "
                f"{describe_token(entry.token):<40} "
                f"{entry.probability:>8.2%} +/- {entry.std_dev:>6.2%} ({status})"
            )

    def _analyze_state_with_bnn(self, state: GameState) -> BNNAnalysis:
        scenario_type = self._bnn_helper._infer_scenario_type(state, self.user_id)
        metadata = self._bnn_helper._infer_metadata(state, self.user_id)

        evaluation = evaluate_state_with_bnn(
            self.artifacts,
            state=state,
            target_player=self.user_id,
            persona_name=self.persona,
            scenario_type=scenario_type,
            metadata=metadata,
            num_samples=self.mc_samples,
        )

        mc = evaluation["mc"]
        mean_probs = mc["mean_probs"].reshape(-1).clone()

        std_probs = mc.get("std_probs")
        if std_probs is not None:
            std_probs = std_probs.reshape(-1).clone()

        entropy_tensor = mc["predictive_entropy"].reshape(-1)
        entropy = float(entropy_tensor[0].item())

        mi_tensor = mc["mutual_information"].reshape(-1)
        mutual_information = float(mi_tensor[0].item())

        valid_actions = self._bnn_helper.enumerate_actions(self.engine, state, self.user_id)
        valid_token_map = {
            self.artifacts.action_encoder._to_token(action): action for action in valid_actions
        }

        # Ensure mean_probs has the expected size for action encoder
        expected_size = len(self.artifacts.action_encoder.lookup)
        if mean_probs.numel() != expected_size:
            # Model output size mismatch - create properly sized tensor
            print(
                f"Warning: Model output size ({mean_probs.numel()}) doesn't match action encoder size ({expected_size})"
            )
            device = mean_probs.device
            dtype = mean_probs.dtype
            corrected_probs = torch.full((expected_size,), 1.0 / expected_size, dtype=dtype, device=device)
            copy_size = min(mean_probs.numel(), expected_size)
            if copy_size > 0:
                corrected_probs[:copy_size].copy_(mean_probs[:copy_size])

            total = corrected_probs.sum()
            if total.item() > 0:
                corrected_probs = corrected_probs / total
            mean_probs = corrected_probs

            if std_probs is not None:
                corrected_std = torch.zeros(expected_size, dtype=std_probs.dtype, device=std_probs.device)
                if copy_size > 0:
                    corrected_std[:copy_size].copy_(std_probs[:copy_size])
                std_probs = corrected_std

        mask = torch.zeros_like(mean_probs)
        valid_indices: List[int] = []
        for token in valid_token_map:
            idx = self.artifacts.action_encoder.lookup.get(token)
            if idx is None:
                continue
            # Ensure index is within bounds
            if int(idx) < mean_probs.size(0):
                valid_indices.append(int(idx))
        if valid_indices:
            mask[valid_indices] = 1.0
        mean_probs_masked = mean_probs * mask if valid_indices else mean_probs
        if valid_indices:
            total = mean_probs_masked.sum()
            if total.item() > 0:
                mean_probs_masked = mean_probs_masked / total
        std_probs_masked = None
        if std_probs is not None:
            std_probs_masked = std_probs.clone()
            if valid_indices:
                std_probs_masked = std_probs_masked * mask
        else:
            std_probs_masked = torch.zeros_like(mean_probs_masked)

        if valid_indices:
            prob_slice = mean_probs_masked[valid_indices]
            prob_mean = float(prob_slice.mean().item())
            prob_std = float(prob_slice.std(unbiased=False).item())
        else:
            prob_mean = float(mean_probs.mean().item())
            prob_std = float(mean_probs.std(unbiased=False).item())

        top_candidates = torch.argsort(mean_probs_masked, descending=True)
        entries: List[PredictionEntry] = []
        for index in top_candidates:
            if len(entries) >= 5:
                break
            idx = int(index.item())
            token = self.artifacts.action_encoder.decode(idx)
            action = valid_token_map.get(token)
            if action is None:
                continue
            probability = float(mean_probs_masked[idx].item())
            if valid_indices and probability <= 0:
                continue
            std_val = 0.0
            if std_probs_masked is not None:
                std_val = float(std_probs_masked[idx].item())
            entries.append(
                PredictionEntry(
                    token=token,
                    probability=probability,
                    std_dev=std_val,
                    is_valid=True,
                    action=action,
                )
            )

        if not entries:
            top_k = min(5, mean_probs.shape[0])
            top_values, top_indices = torch.topk(mean_probs, k=top_k)
            for value, index in zip(top_values, top_indices):
                idx = int(index.item())
                token = self.artifacts.action_encoder.decode(idx)
                action = valid_token_map.get(token)
                std_val = 0.0
                if std_probs is not None and 0 <= idx < std_probs.shape[0]:
                    std_val = float(std_probs[idx].item())
                entries.append(
                    PredictionEntry(
                        token=token,
                        probability=float(value.item()),
                        std_dev=std_val,
                        is_valid=action is not None,
                        action=action,
                    )
                )
                if len(entries) >= 5:
                    break

        suggestion = self._bnn_helper.decide(self.engine, state, self.user_id)
        suggested_token = self.artifacts.action_encoder._to_token(suggestion)
        suggested_index = self.artifacts.action_encoder.lookup.get(suggested_token)
        suggested_probability = 0.0
        suggested_std = 0.0
        if suggested_index is not None:
            idx = int(suggested_index)
            target_probs = mean_probs_masked if valid_indices else mean_probs
            if 0 <= idx < target_probs.shape[0]:
                suggested_probability = float(target_probs[idx].item())
            if std_probs_masked is not None and 0 <= idx < std_probs_masked.shape[0]:
                suggested_std = float(std_probs_masked[idx].item())

        suggested_probability = float(
            suggestion.info.get("bnn_probability", suggested_probability)
        )
        info_std = suggestion.info.get("bnn_probability_std")
        if info_std is not None:
            suggested_std = float(info_std)
        selection_reason = suggestion.info.get("selection_reason", "probability")

        if logger.isEnabledFor(logging.DEBUG):
            top_debug = ", ".join(
                f"{describe_token(entry.token)}={entry.probability:.2%}"
                for entry in entries[:3]
            )
            logger.debug(
                "BNN analysis | entropy=%.3f | MI=%.3f | reason=%s | top=%s",
                entropy,
                mutual_information,
                selection_reason,
                top_debug,
            )

        return BNNAnalysis(
            scenario_type=scenario_type,
            entropy=entropy,
            mutual_information=mutual_information,
            entries=entries,
            suggested_action=suggestion,
            suggested_token=suggested_token,
            suggested_probability=suggested_probability,
            suggested_std=suggested_std,
            mc_samples=self.mc_samples,
            mean_probability=prob_mean,
            probability_std=prob_std,
            selection_reason=selection_reason,
        )

    def _display_round_results(self, state: GameState) -> None:
        print("\n=== Round complete =========================================")
        if state.round_winner_id:
            indicator = (
                "You and your teammate won!"
                if self.user_id in state.round_winning_team_ids
                else "Opponents took the round."
            )
            winner_name = self.display_names.get(state.round_winner_id, state.round_winner_id)
            print(f"Winner: {winner_name} -> {indicator}")
        else:
            print("Round ended without a winner (likely due to max turns).")

        for team_index, score in state.team_scores.items():
            teammates = [
                player.player_id
                for player in state.players
                if state.team_map.get(player.player_id) == team_index
            ]
            label = " / ".join(self.display_names.get(pid, pid) for pid in teammates)
            print(f"  Team {team_index}: {label} | Score: {score}")

        print("============================================================\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play UNO 2v2 with BNN analysis (console UI).")
    parser.add_argument(
        "--meta",
        type=Path,
        help="Path to the exported *_meta.json file (defaults to latest in models/).",
    )
    parser.add_argument(
        "--param-store",
        type=Path,
        help="Path to the exported *_param_store.pt file (defaults to latest in models/).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory searched for exported models when --meta/--param-store are omitted.",
    )
    parser.add_argument(
        "--persona",
        choices=BOT_ORDER,
        default="OracleBot",
        help="Bot persona embedding fed to the BNN (affects feature encoding).",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=40,
        help="Number of Monte Carlo samples for BNN predictive inference.",
    )
    parser.add_argument(
        "--mode",
        choices=["classic", "wild"],
        help="Start directly in the specified mode (otherwise prompt).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return parser.parse_args(argv)


def determine_artifact_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.meta and args.param_store:
        return args.meta, args.param_store

    if args.meta or args.param_store:
        raise SystemExit("Both --meta and --param-store must be provided together.")

    discovered = discover_latest_model(args.models_dir)
    if discovered is None:
        raise SystemExit(
            "No exported models found. Please train a model with --export first "
            "or supply --meta/--param-store explicitly."
        )
    meta_path, param_path = discovered
    print(f"Using model artifacts: {meta_path.name}, {param_path.name}")
    return meta_path, param_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    meta_path, param_store_path = determine_artifact_paths(args)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        artifacts = load_bnn_artifacts(meta_path, param_store_path, device=device)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    interface = ConsoleUnoBNNInterface(artifacts, persona=args.persona)

    initial_mode: Optional[GameMode] = None
    if args.mode == "classic":
        initial_mode = GameMode.CLASSIC_2V2
    elif args.mode == "wild":
        initial_mode = GameMode.GO_WILD_2V2

    interface.run(initial_mode=initial_mode)


if __name__ == "__main__":  # pragma: no cover - manual execution
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")

