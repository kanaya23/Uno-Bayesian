# %% [code cell 0]
from __future__ import annotations

# If running in Colab, install dependencies.
import os
import subprocess
import sys

IN_COLAB = "COLAB_GPU" in os.environ or "google.colab" in sys.modules
if IN_COLAB:
    try:
        import pyro  # type: ignore
    except Exception:  # pragma: no cover - installation path
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "pyro-ppl",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )

# %% [code cell 1]

import abc
import argparse
import csv
import dataclasses
import enum
import functools
import itertools
import json
import logging
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError("Missing dependency 'numpy'. Install with `pip install numpy`.") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "Missing dependency 'torch'. Install with `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`."
    ) from exc

try:
    import pyro
    import pyro.distributions as dist
    from pyro import poutine
    from pyro.infer import SVI, Trace_ELBO
    from pyro.nn import PyroModule, PyroSample, PyroParam
    from pyro.optim import ClippedAdam
    from pyro.distributions import constraints
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError("Missing dependency 'pyro-ppl'. Install with `pip install pyro-ppl`.") from exc

logger = logging.getLogger(__name__)

from uno_engine.deck import build_deck_for_mode, shuffle_in_place
from uno_engine.engine import UnoEngine, InvalidMoveError
from uno_engine.models import (
    Card,
    CardType,
    Color,
    GameMode,
    GameState,
    PendingAction,
    PendingActionType,
    Player,
    PlayDirection,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

RNG = random.Random(13)
torch.manual_seed(13)
np.random.seed(13)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(13)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("medium")  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - older torch
        pass

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_NUM_WORKERS = max(1, min(4, (os.cpu_count() or 1)))

# %% [code cell 2]
class ScenarioType(enum.Enum):
    FINISHER = "FINISHER"
    DEFENDER = "DEFENDER"
    SETUP = "SETUP"
    COLOR_TRAP = "COLOR_TRAP"
    WILD_DILEMMA = "WILD_DILEMMA"


PLAYER_SEQUENCE = ("P0", "P1", "P2", "P3")
TEAM_MAP = {"P0": 0, "P2": 0, "P1": 1, "P3": 1}
TEAM_SCORES = {0: 0, 1: 0}
STANDARD_COLORS = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]


@dataclass
class ScenarioParameters:
    scenario_type: ScenarioType
    mode: GameMode = GameMode.CLASSIC_2V2
    target_player: str = "P0"
    distance_to_victory: int = 1
    draw_stack_size: int = 2
    hand_diversity: int = 3
    color_bias: Optional[Color] = None
    randomize: bool = True


@dataclass
class ScenarioExample:
    state: GameState
    target_player: str
    scenario_type: ScenarioType
    parameters: ScenarioParameters
    metadata: Dict[str, Any]


class ScenarioForge:
    """Procedurally generate goal-directed UNO states for downstream training."""

    def __init__(self, *, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, params: ScenarioParameters) -> ScenarioExample:
        handler = {
            ScenarioType.FINISHER: self._finisher,
            ScenarioType.DEFENDER: self._defender,
            ScenarioType.SETUP: self._setup,
            ScenarioType.COLOR_TRAP: self._color_trap,
            ScenarioType.WILD_DILEMMA: self._wild_dilemma,
        }[params.scenario_type]
        return handler(params)

    def generate_batch(
        self,
        scenario_mix: Dict[ScenarioType, float],
        *,
        mode: GameMode,
        batch_size: int,
        base_params: Optional[ScenarioParameters] = None,
    ) -> List[ScenarioExample]:
        weights = np.array(list(scenario_mix.values()), dtype=np.float64)
        if not math.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()
        scenarios = list(scenario_mix.keys())

        examples: List[ScenarioExample] = []
        for _ in range(batch_size):
            scenario_type = self.rng.choices(scenarios, weights=weights)[0]
            params = dataclasses.replace(
                base_params or ScenarioParameters(scenario_type=scenario_type, mode=mode),
                scenario_type=scenario_type,
                mode=mode,
            )
            examples.append(self.generate(params))
        return examples

    # ------------------------------------------------------------------
    # Scenario builders
    # ------------------------------------------------------------------
    def _finisher(self, params: ScenarioParameters) -> ScenarioExample:
        distance = max(1, params.distance_to_victory)
        top_color = params.color_bias or self.rng.choice(STANDARD_COLORS)
        top_number = self.rng.randint(0, 9)
        top_card = self._make_number_card(top_color, top_number)

        target_hand = self._make_random_hand(params.mode, hand_size=distance)
        playable_card = self._ensure_playable(target_hand, top_card, top_color)

        other_hands = {
            pid: self._make_random_hand(params.mode, hand_size=self.rng.randint(4, 7))
            for pid in PLAYER_SEQUENCE
            if pid != params.target_player
        }

        current_player_idx = PLAYER_SEQUENCE.index(params.target_player)
        state = self._assemble_state(
            mode=params.mode,
            hands={params.target_player: target_hand, **other_hands},
            discard=[top_card],
            current_player_index=current_player_idx,
            current_color=top_color,
        )

        metadata = {
            "playable_card": playable_card,
            "distance_to_victory": distance,
        }
        return ScenarioExample(state, params.target_player, ScenarioType.FINISHER, params, metadata)

    def _defender(self, params: ScenarioParameters) -> ScenarioExample:
        stack_size = max(2, params.draw_stack_size)
        stack_card_type = CardType.DRAW_TWO if stack_size % 4 != 0 else CardType.WILD_DRAW_FOUR
        top_card = (
            self._make_action_card(self.rng.choice(STANDARD_COLORS), CardType.DRAW_TWO)
            if stack_card_type == CardType.DRAW_TWO
            else self._make_wild_draw_four()
        )

        target_hand = self._make_random_hand(params.mode, hand_size=self.rng.randint(3, 6))
        response_card = self._ensure_stack_response(target_hand, stack_card_type)

        other_hands = {
            pid: self._make_random_hand(params.mode, hand_size=self.rng.randint(4, 7))
            for pid in PLAYER_SEQUENCE
            if pid != params.target_player
        }

        pending = PendingAction(
            player_id=params.target_player,
            type=PendingActionType.DRAW_STACK,
            allowed_cards=tuple(
                card
                for card in target_hand
                if card.type in {CardType.DRAW_TWO, CardType.WILD_DRAW_FOUR}
            ),
            draw_penalty=stack_size,
        )

        state = self._assemble_state(
            mode=params.mode,
            hands={params.target_player: target_hand, **other_hands},
            discard=[top_card],
            current_player_index=PLAYER_SEQUENCE.index(params.target_player),
            current_color=top_card.color if top_card.color != Color.WILD else None,
            pending_action=pending,
            draw_stack_total=stack_size,
        )

        metadata = {
            "draw_stack_penalty": stack_size,
            "response_card": response_card,
        }
        return ScenarioExample(state, params.target_player, ScenarioType.DEFENDER, params, metadata)

    def _setup(self, params: ScenarioParameters) -> ScenarioExample:
        diversity = max(2, min(4, params.hand_diversity))
        preferred_colors = self.rng.sample(STANDARD_COLORS, diversity)
        teammate = self._teammate_of(params.target_player)
        teammate_color = preferred_colors[0]

        top_color = params.color_bias or preferred_colors[-1]
        top_card = self._make_number_card(top_color, self.rng.randint(0, 9))

        target_hand = []
        for color in preferred_colors:
            target_hand.append(self._make_number_card(color, self.rng.randint(0, 9)))
        target_hand.append(self._make_action_card(preferred_colors[0], CardType.REVERSE))
        target_hand.append(self._make_wild())

        teammate_hand = [self._make_number_card(teammate_color, self.rng.randint(1, 9)) for _ in range(5)]
        opponent_hands = {
            pid: self._make_random_hand(params.mode, hand_size=self.rng.randint(5, 8))
            for pid in PLAYER_SEQUENCE
            if pid not in {params.target_player, teammate}
        }

        state = self._assemble_state(
            mode=params.mode,
            hands={
                params.target_player: target_hand,
                teammate: teammate_hand,
                **opponent_hands,
            },
            discard=[top_card],
            current_player_index=PLAYER_SEQUENCE.index(params.target_player),
            current_color=top_color,
        )

        metadata = {
            "teammate_preferred_color": teammate_color,
            "available_wild": True,
        }
        return ScenarioExample(state, params.target_player, ScenarioType.SETUP, params, metadata)

    def _color_trap(self, params: ScenarioParameters) -> ScenarioExample:
        weak_color = params.color_bias or self.rng.choice(STANDARD_COLORS)
        strong_color = self.rng.choice([c for c in STANDARD_COLORS if c != weak_color])
        top_card = self._make_number_card(strong_color, self.rng.randint(0, 9))

        target_hand = self._make_random_hand(params.mode, hand_size=self.rng.randint(5, 7))
        target_hand.append(self._make_action_card(strong_color, CardType.SKIP))

        teammate = self._teammate_of(params.target_player)
        teammate_hand = self._make_random_hand(params.mode, hand_size=self.rng.randint(4, 6))

        opponents = [pid for pid in PLAYER_SEQUENCE if pid not in {params.target_player, teammate}]
        opponent_hands = {
            opponents[0]: self._make_hand_without_color(params.mode, color=weak_color, hand_size=self.rng.randint(5, 7)),
            opponents[1]: self._make_random_hand(params.mode, hand_size=self.rng.randint(5, 7)),
        }

        state = self._assemble_state(
            mode=params.mode,
            hands={
                params.target_player: target_hand,
                teammate: teammate_hand,
                **opponent_hands,
            },
            discard=[top_card],
            current_player_index=PLAYER_SEQUENCE.index(params.target_player),
            current_color=top_card.color,
        )

        metadata = {
            "weak_opponent_color": weak_color.value,
            "target_color": strong_color.value,
        }
        return ScenarioExample(state, params.target_player, ScenarioType.COLOR_TRAP, params, metadata)

    def _wild_dilemma(self, params: ScenarioParameters) -> ScenarioExample:
        base_color = params.color_bias or self.rng.choice(STANDARD_COLORS)
        alt_color = self.rng.choice([c for c in STANDARD_COLORS if c != base_color])
        top_card = self._make_number_card(base_color, self.rng.randint(1, 9))

        target_hand = [
            self._make_number_card(base_color, top_card.number),
            self._make_number_card(alt_color, self.rng.randint(0, 9)),
            self._make_action_card(base_color, CardType.SKIP),
            self._make_wild(),
            self._make_wild_draw_four() if params.mode == GameMode.GO_WILD_2V2 else self._make_number_card(base_color, self.rng.randint(0, 9)),
        ]
        other_hands = {
            pid: self._make_random_hand(params.mode, hand_size=self.rng.randint(4, 7))
            for pid in PLAYER_SEQUENCE
            if pid != params.target_player
        }

        state = self._assemble_state(
            mode=params.mode,
            hands={params.target_player: target_hand, **other_hands},
            discard=[top_card],
            current_player_index=PLAYER_SEQUENCE.index(params.target_player),
            current_color=top_card.color,
        )

        metadata = {
            "normal_play_option": target_hand[0],
            "wild_play_option": target_hand[-2],
            "high_stakes": params.mode == GameMode.GO_WILD_2V2,
        }
        return ScenarioExample(state, params.target_player, ScenarioType.WILD_DILEMMA, params, metadata)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_random_hand(self, mode: GameMode, hand_size: int) -> List[Card]:
        hand: List[Card] = []
        for _ in range(hand_size):
            card_type = self.rng.choices(
                [CardType.NUMBER, CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO, CardType.WILD, CardType.WILD_DRAW_FOUR, CardType.DISCARD_ALL],
                weights=[0.55, 0.08, 0.08, 0.08, 0.12, 0.05, 0.04 if mode == GameMode.GO_WILD_2V2 else 0.0],
            )[0]
            if card_type == CardType.NUMBER:
                color = self.rng.choice(STANDARD_COLORS)
                hand.append(self._make_number_card(color, self.rng.randint(0, 9)))
            elif card_type in {CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO}:
                color = self.rng.choice(STANDARD_COLORS)
                hand.append(self._make_action_card(color, card_type))
            elif card_type == CardType.WILD:
                hand.append(self._make_wild())
            elif card_type == CardType.WILD_DRAW_FOUR:
                hand.append(self._make_wild_draw_four())
            else:  # DISCARD_ALL
                color = self.rng.choice(STANDARD_COLORS)
                hand.append(self._make_action_card(color, CardType.DISCARD_ALL))
        return hand

    def _make_hand_without_color(self, mode: GameMode, color: Color, hand_size: int) -> List[Card]:
        hand = []
        while len(hand) < hand_size:
            card = self._make_random_hand(mode, 1)[0]
            if card.color != color:
                hand.append(card)
        return hand

    def _assemble_state(
        self,
        *,
        mode: GameMode,
        hands: Dict[str, Sequence[Card]],
        discard: List[Card],
        current_player_index: int,
        current_color: Optional[Color],
        pending_action: Optional[PendingAction] = None,
        draw_stack_total: int = 0,
    ) -> GameState:
        players = [
            Player(player_id=pid, hand=list(hands[pid]), score=0)
            for pid in PLAYER_SEQUENCE
        ]

        deck = build_deck_for_mode(mode)
        for card in itertools.chain.from_iterable(hands.values()):
            self._try_remove_card(deck, card)
        for card in discard:
            self._try_remove_card(deck, card)
        shuffle_in_place(deck, self.rng)

        state = GameState(
            players=players,
            draw_pile=deck,
            discard_pile=list(discard),
            current_player_index=current_player_index,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=current_color,
            pending_action=pending_action,
            mode=mode,
            team_map=dict(TEAM_MAP),
            team_scores=dict(TEAM_SCORES),
            draw_stack_total=draw_stack_total,
        )
        return state

    def _try_remove_card(self, deck: List[Card], card: Card) -> None:
        for idx, candidate in enumerate(deck):
            if candidate == card:
                deck.pop(idx)
                return

    def _ensure_playable(self, hand: List[Card], top_card: Card, current_color: Color) -> Card:
        for card in hand:
            if self._is_playable(card, top_card, current_color):
                return card
        replacement = self._make_number_card(current_color, top_card.number)
        hand[0] = replacement
        return replacement

    def _ensure_stack_response(self, hand: List[Card], stack_type: CardType) -> Card:
        for card in hand:
            if card.type == stack_type or (card.type == CardType.WILD_DRAW_FOUR and stack_type == CardType.DRAW_TWO):
                return card
        if stack_type == CardType.DRAW_TWO:
            response = self._make_action_card(self.rng.choice(STANDARD_COLORS), CardType.DRAW_TWO)
        else:
            response = self._make_wild_draw_four()
        hand[0] = response
        return response

    def _is_playable(self, card: Card, top_card: Card, current_color: Color) -> bool:
        if card.type == CardType.WILD:
            return True
        if card.type == CardType.WILD_DRAW_FOUR:
            return True
        if card.color == current_color:
            return True
        if card.type == top_card.type:
            return True
        if card.type == CardType.NUMBER and top_card.type == CardType.NUMBER:
            return card.number == top_card.number
        return False

    def _make_number_card(self, color: Color, number: int) -> Card:
        return Card(color=color, type=CardType.NUMBER, number=number, value=number)

    def _make_action_card(self, color: Color, card_type: CardType) -> Card:
        value_map = {
            CardType.SKIP: 20,
            CardType.REVERSE: 20,
            CardType.DRAW_TWO: 20,
            CardType.DISCARD_ALL: 40,
        }
        return Card(color=color, type=card_type, value=value_map[card_type])

    def _make_wild(self) -> Card:
        return Card(color=Color.WILD, type=CardType.WILD, value=50)

    def _make_wild_draw_four(self) -> Card:
        return Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)

    def _teammate_of(self, player_id: str) -> str:
        team_index = TEAM_MAP[player_id]
        for pid, t_index in TEAM_MAP.items():
            if pid != player_id and t_index == team_index:
                return pid
        raise ValueError(f"No teammate registered for player '{player_id}'")

# %% [code cell 3]
class ActionType(enum.Enum):
    PLAY = "PLAY"
    DRAW = "DRAW"
    PASS = "PASS"


@dataclass
class BotAction:
    action_type: ActionType
    card: Optional[Card] = None
    chosen_color: Optional[Color] = None
    info: Dict[str, Any] = field(default_factory=dict)


class BotPolicy(abc.ABC):
    def __init__(self, name: str, rng: Optional[random.Random] = None) -> None:
        self.name = name
        self.rng = rng or random.Random()

    def enumerate_actions(self, engine: UnoEngine, state: GameState, player_id: str) -> List[BotAction]:
        valid_cards = engine.get_valid_moves(state, player_id)
        actions: List[BotAction] = []
        for card in valid_cards:
            if card.type in {CardType.WILD, CardType.WILD_DRAW_FOUR}:
                for color in STANDARD_COLORS:
                    actions.append(BotAction(ActionType.PLAY, card=card, chosen_color=color))
            else:
                actions.append(BotAction(ActionType.PLAY, card=card))
        if not actions:
            pending = state.pending_action
            if pending and pending.type == PendingActionType.DRAW_STACK:
                actions.append(BotAction(ActionType.DRAW))
            else:
                actions.append(BotAction(ActionType.DRAW))
        return actions

    @abc.abstractmethod
    def decide(self, engine: UnoEngine, state: GameState, player_id: str) -> BotAction:
        ...

    # Utility scoring helpers -------------------------------------------------
    def choose_color(self, state: GameState, player_id: str) -> Color:
        hand = next(
            (tuple(player.hand) for player in state.players if player.player_id == player_id),
            tuple(),
        )
        counts = self.color_histogram(hand)
        max_count = max(counts.values()) if counts else 0
        if max_count == 0:
            return self.rng.choice(STANDARD_COLORS)
        best_colors = [color for color, value in counts.items() if value == max_count]
        return self.rng.choice(best_colors)

    def teammate_id(self, state: GameState, player_id: str) -> str:
        return state.teammate_id(player_id)

    def team_indices(self, state: GameState, player_id: str) -> Tuple[int, int]:
        my_team = state.team_index_for(player_id)
        return my_team, 1 - my_team

    def color_histogram(self, cards: Sequence[Card]) -> Dict[Color, int]:
        counts: Dict[Color, int] = {color: 0 for color in STANDARD_COLORS}
        for card in cards:
            if card.color in counts:
                counts[card.color] += 1
        return counts

    def choose_color_for_teammate(self, teammate_hand: Sequence[Card]) -> Color:
        counts = self.color_histogram(teammate_hand)
        return max(counts.items(), key=lambda kv: kv[1])[0]


class OracleBot(BotPolicy):
    def __init__(self, *, rollout_count: int = 16, rollout_depth: int = 6, rng: Optional[random.Random] = None) -> None:
        super().__init__("OracleBot", rng=rng)
        self.rollout_count = rollout_count
        self.rollout_depth = rollout_depth
        self.engine = UnoEngine()

    def decide(self, engine: UnoEngine, state: GameState, player_id: str) -> BotAction:
        actions = self.enumerate_actions(engine, state, player_id)
        if len(actions) == 1:
            return actions[0]

        scores: List[float] = []
        for action in actions:
            estimate = self._estimate_value(state, player_id, action)
            scores.append(estimate)
        best_idx = int(np.argmax(scores))
        best_action = actions[best_idx]
        best_action.info["value_estimate"] = scores[best_idx]
        return best_action

    def _estimate_value(self, state: GameState, player_id: str, action: BotAction) -> float:
        total = 0.0
        for _ in range(self.rollout_count):
            simulated = self._apply_action(state, player_id, action)
            total += self._rollout_value(simulated, player_id)
        return total / self.rollout_count

    def _apply_action(self, state: GameState, player_id: str, action: BotAction) -> GameState:
        if action.action_type == ActionType.PLAY and action.card is not None:
            return self.engine.play_card(state, player_id, action.card, action.chosen_color)
        if action.action_type == ActionType.DRAW:
            try:
                return self.engine.draw_card(state, player_id)
            except InvalidMoveError:
                return state
        if action.action_type == ActionType.PASS:
            try:
                return self.engine.pass_turn(state, player_id)
            except InvalidMoveError:
                return state
        return state

    def _rollout_value(self, state: GameState, origin_player: str) -> float:
        working_state = state
        depth = self.rollout_depth
        current_engine = self.engine
        while depth > 0 and not working_state.round_over:
            current_player = working_state.current_player().player_id
            actions = self.enumerate_actions(current_engine, working_state, current_player)
            choice = self.rng.choice(actions)
            working_state = self._apply_action(working_state, current_player, choice)
            depth -= 1
        return self._heuristic_value(working_state, origin_player)

    def _heuristic_value(self, state: GameState, player_id: str) -> float:
        my_team, opp_team = self.team_indices(state, player_id)
        my_cards = sum(len(p.hand) for p in state.players if state.team_map[p.player_id] == my_team)
        opp_cards = sum(len(p.hand) for p in state.players if state.team_map[p.player_id] == opp_team)
        score = opp_cards - my_cards
        if state.round_over:
            if state.team_index_for(state.round_winner_id or player_id) == my_team:
                score += 50
            else:
                score -= 50
        return score


class AggressorBot(BotPolicy):
    def __init__(self, rng: Optional[random.Random] = None) -> None:
        super().__init__("AggressorBot", rng=rng)

    def decide(self, engine: UnoEngine, state: GameState, player_id: str) -> BotAction:
        actions = self.enumerate_actions(engine, state, player_id)
        priority = {
            CardType.WILD_DRAW_FOUR: 5,
            CardType.DRAW_TWO: 4,
            CardType.SKIP: 3,
            CardType.REVERSE: 2,
        }
        best = actions[0]
        best_score = -float("inf")
        for action in actions:
            score = 0.0
            if action.action_type != ActionType.PLAY or action.card is None:
                continue
            score += priority.get(action.card.type, 0)
            if action.card.type in {CardType.WILD, CardType.WILD_DRAW_FOUR} and action.chosen_color is not None:
                score += self._penalty_color_bonus(state, player_id, action.chosen_color)
            if score > best_score:
                best_score = score
                best = action
        return best

    def _penalty_color_bonus(self, state: GameState, player_id: str, color: Color) -> float:
        my_team, opp_team = self.team_indices(state, player_id)
        opp_color_counts = 0
        for player in state.players:
            if state.team_map[player.player_id] == opp_team:
                opp_color_counts += sum(1 for c in player.hand if c.color == color)
        return -0.1 * opp_color_counts


class SupporterBot(BotPolicy):
    def __init__(self, rng: Optional[random.Random] = None) -> None:
        super().__init__("SupporterBot", rng=rng)

    def decide(self, engine: UnoEngine, state: GameState, player_id: str) -> BotAction:
        actions = self.enumerate_actions(engine, state, player_id)
        teammate = self.teammate_id(state, player_id)
        teammate_hand = next(p.hand for p in state.players if p.player_id == teammate)
        preferred_color = self.choose_color_for_teammate(teammate_hand)

        scored: List[Tuple[float, BotAction]] = []
        for action in actions:
            score = 0.0
            if action.action_type == ActionType.PLAY and action.card is not None:
                if action.card.color == preferred_color:
                    score += 2.0
                if action.card.type in {CardType.SKIP, CardType.REVERSE}:
                    score += 1.0
                if action.card.type == CardType.WILD and action.chosen_color == preferred_color:
                    score += 3.0
                if action.card.type == CardType.WILD_DRAW_FOUR and action.chosen_color == preferred_color:
                    score += 2.5
                if action.card.type == CardType.DISCARD_ALL and action.card.color == preferred_color:
                    score += 2.0
            scored.append((score, action))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1]


class ConservativeBot(BotPolicy):
    def __init__(self, rng: Optional[random.Random] = None) -> None:
        super().__init__("ConservativeBot", rng=rng)

    def decide(self, engine: UnoEngine, state: GameState, player_id: str) -> BotAction:
        actions = self.enumerate_actions(engine, state, player_id)
        ranked: List[Tuple[float, BotAction]] = []
        for action in actions:
            if action.action_type != ActionType.PLAY or action.card is None:
                ranked.append((float("inf"), action))
                continue
            card_value = action.card.value
            penalty = 0.0
            if action.card.type in {CardType.WILD_DRAW_FOUR, CardType.DRAW_TWO}:
                penalty += 3.0
            if action.card.type == CardType.WILD:
                penalty += 1.5
            ranked.append((card_value + penalty, action))
        ranked.sort(key=lambda pair: pair[0])
        return ranked[0][1]


class RandomBot(BotPolicy):
    def __init__(self, rng: Optional[random.Random] = None) -> None:
        super().__init__("RandomBot", rng=rng)

    def decide(self, engine: UnoEngine, state: GameState, player_id: str) -> BotAction:
        actions = self.enumerate_actions(engine, state, player_id)
        return self.rng.choice(actions)

# %% [code cell 4]
@dataclass
class LabeledScenario:
    scenario: ScenarioExample
    bot_name: str
    action: BotAction
    resulting_state: GameState
    action_success: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ScenarioLabeler:
    def __init__(self, *, rng: Optional[random.Random] = None) -> None:
        self.engine = UnoEngine()
        self.rng = rng or random.Random()

    def label(
        self,
        scenarios: Sequence[ScenarioExample],
        bots: Sequence[BotPolicy],
    ) -> List[LabeledScenario]:
        labeled: List[LabeledScenario] = []
        for scenario in scenarios:
            for bot in bots:
                action = bot.decide(self.engine, scenario.state, scenario.target_player)
                next_state, success = self._apply_action(scenario.state, scenario.target_player, action)
                info = {"bot_decision_metadata": action.info}
                labeled.append(
                    LabeledScenario(
                        scenario=scenario,
                        bot_name=bot.name,
                        action=action,
                        resulting_state=next_state,
                        action_success=success,
                        info=info,
                    )
                )
        return labeled

    def logs_to_labeled(self, logs: Sequence[ActiveLog]) -> List[LabeledScenario]:
        converted: List[LabeledScenario] = []
        for log in logs:
            parameters = ScenarioParameters(
                scenario_type=log.scenario_type,
                mode=log.state.mode,
                target_player=log.target_player,
            )
            scenario = ScenarioExample(
                state=log.state,
                target_player=log.target_player,
                scenario_type=log.scenario_type,
                parameters=parameters,
                metadata=log.metadata,
            )
            converted.append(
                LabeledScenario(
                    scenario=scenario,
                    bot_name=log.persona_name,
                    action=log.action,
                    resulting_state=log.resulting_state,
                    action_success=True,
                    info={
                        "bnn_entropy": log.bnn_entropy,
                        "bnn_mutual_information": log.bnn_mutual_information,
                    },
                )
            )
        return converted

    def _apply_action(
        self,
        state: GameState,
        player_id: str,
        action: BotAction,
    ) -> Tuple[GameState, bool]:
        try:
            if action.action_type == ActionType.PLAY and action.card is not None:
                next_state = self.engine.play_card(state, player_id, action.card, action.chosen_color)
            elif action.action_type == ActionType.DRAW:
                next_state = self.engine.draw_card(state, player_id)
            elif action.action_type == ActionType.PASS:
                next_state = self.engine.pass_turn(state, player_id)
            else:
                next_state = state
            return next_state, True
        except InvalidMoveError:
            return state, False

# %% [code cell 5]
# Quick sanity check: generate one sample per archetype and preview bot actions.
forge = ScenarioForge(rng=random.Random(21))
labeler = ScenarioLabeler(rng=random.Random(22))
bots = [OracleBot(rng=random.Random(23)), AggressorBot(rng=random.Random(24))]

samples: Dict[str, Dict[str, Any]] = {}
for scenario_type in ScenarioType:
    scenario = forge.generate(ScenarioParameters(scenario_type=scenario_type))
    labeled = labeler.label([scenario], bots)
    samples[scenario_type.value] = {
        "target_player": scenario.target_player,
        "top_discard": scenario.state.discard_pile[-1].type.name,
        "bot_actions": {entry.bot_name: entry.action.info for entry in labeled},
    }
samples
# %% [code cell 6]
CARD_TYPES_FOR_ENCODING = [
    CardType.NUMBER,
    CardType.SKIP,
    CardType.REVERSE,
    CardType.DRAW_TWO,
    CardType.WILD,
    CardType.WILD_DRAW_FOUR,
    CardType.DISCARD_ALL,
]
SPECIAL_TYPES = [
    CardType.SKIP,
    CardType.REVERSE,
    CardType.DRAW_TWO,
    CardType.WILD,
    CardType.WILD_DRAW_FOUR,
    CardType.DISCARD_ALL,
]
PENDING_TYPES = [PendingActionType.DRAWN_CARD_PLAY_WINDOW, PendingActionType.COLOR_CHOICE, PendingActionType.DRAW_STACK]
BOT_ORDER = ["OracleBot", "AggressorBot", "SupporterBot", "ConservativeBot", "RandomBot"]
SCENARIO_ORDER = list(ScenarioType)
MODE_ORDER = [GameMode.CLASSIC_2V2, GameMode.GO_WILD_2V2]


class StateEncoder:
    def __init__(self) -> None:
        self.feature_size = self._compute_feature_size()

    def encode(self, labeled: LabeledScenario) -> np.ndarray:
        return self.encode_components(
            state=labeled.scenario.state,
            target_player=labeled.scenario.target_player,
            bot_name=labeled.bot_name,
            scenario_type=labeled.scenario.scenario_type,
            metadata=labeled.scenario.metadata,
        )

    def encode_components(
        self,
        *,
        state: GameState,
        target_player: str,
        bot_name: str,
        scenario_type: ScenarioType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        metadata = metadata or {}
        features: List[float] = []

        # Player-centric features (ordered P0..P3)
        for pid in PLAYER_SEQUENCE:
            hand = next(player.hand for player in state.players if player.player_id == pid)
            features.append(len(hand) / 20.0)
            color_counts = {color: 0 for color in STANDARD_COLORS}
            special_counts = {stype: 0 for stype in SPECIAL_TYPES}
            numeric_sum = 0
            for card in hand:
                if card.color in color_counts:
                    color_counts[card.color] += 1
                if card.type in special_counts:
                    special_counts[card.type] += 1
                if card.type == CardType.NUMBER and card.number is not None:
                    numeric_sum += card.number
            features.extend(color_counts[color] / 10.0 for color in STANDARD_COLORS)
            features.extend(special_counts[stype] / 5.0 for stype in SPECIAL_TYPES)
            features.append(numeric_sum / 40.0)

        # Discard / top card
        top_card = state.discard_pile[-1]
        features.extend(self._encode_color(top_card.color))
        features.extend(self._encode_card_type(top_card.type))
        features.append((top_card.number or 0) / 9.0)
        features.append(1.0 if top_card.type == CardType.NUMBER else 0.0)

        # Current color context
        features.extend(self._encode_color(state.current_color))

        # Pending action signals
        pending = state.pending_action
        features.extend(self._encode_pending_type(pending.type if pending else None))
        draw_penalty = pending.draw_penalty if pending else 0
        features.append(draw_penalty / 12.0)
        features.append((len(pending.allowed_cards) if pending else 0) / 5.0)
        features.append(state.draw_stack_total / 12.0)

        # Scenario info
        features.extend(self._encode_scenario_type(scenario_type))
        features.append(metadata.get("distance_to_victory", 0) / 10.0)
        features.append(metadata.get("draw_stack_penalty", 0) / 12.0)
        features.append(1.0 if metadata.get("available_wild") else 0.0)
        features.append(1.0 if metadata.get("high_stakes") else 0.0)
        features.extend(self._encode_color(metadata.get("teammate_preferred_color")))
        features.extend(self._encode_color(metadata.get("weak_opponent_color")))

        # Target player & persona identity
        target_one_hot = [1.0 if pid == target_player else 0.0 for pid in PLAYER_SEQUENCE]
        features.extend(target_one_hot)
        bot_one_hot = [1.0 if bot_name == name else 0.0 for name in BOT_ORDER]
        features.extend(bot_one_hot)

        # Mode
        features.extend(self._encode_mode(state.mode))

        return np.array(features, dtype=np.float32)

    def _compute_feature_size(self) -> int:
        dummy_state = ScenarioForge(rng=random.Random(0)).generate(
            ScenarioParameters(scenario_type=ScenarioType.FINISHER)
        )
        dummy_labeled = LabeledScenario(
            scenario=dummy_state,
            bot_name="OracleBot",
            action=BotAction(ActionType.DRAW),
            resulting_state=dummy_state.state,
            action_success=True,
        )
        return len(self.encode(dummy_labeled))

    def _encode_color(self, color: Optional[Color]) -> List[float]:
        if isinstance(color, str):
            try:
                color = Color[color]
            except KeyError:
                color = None
        bucket = [0.0] * (len(STANDARD_COLORS) + 1)
        if color in STANDARD_COLORS:
            bucket[STANDARD_COLORS.index(color)] = 1.0
        elif color == Color.WILD:
            bucket[-1] = 1.0
        return bucket

    def _encode_card_type(self, card_type: CardType) -> List[float]:
        bucket = [0.0] * len(CARD_TYPES_FOR_ENCODING)
        if card_type in CARD_TYPES_FOR_ENCODING:
            bucket[CARD_TYPES_FOR_ENCODING.index(card_type)] = 1.0
        return bucket

    def _encode_pending_type(self, pending_type: Optional[PendingActionType]) -> List[float]:
        bucket = [0.0] * len(PENDING_TYPES)
        if pending_type in PENDING_TYPES:
            bucket[PENDING_TYPES.index(pending_type)] = 1.0
        return bucket

    def _encode_scenario_type(self, scenario_type: ScenarioType) -> List[float]:
        bucket = [0.0] * len(SCENARIO_ORDER)
        bucket[SCENARIO_ORDER.index(scenario_type)] = 1.0
        return bucket

    def _encode_mode(self, mode: GameMode) -> List[float]:
        bucket = [0.0] * len(MODE_ORDER)
        if mode in MODE_ORDER:
            bucket[MODE_ORDER.index(mode)] = 1.0
        return bucket


class ActionEncoder:
    def __init__(self) -> None:
        self.lookup: Dict[str, int] = {}
        self.reverse: Dict[int, str] = {}
        for token in self._enumerate_tokens():
            self.lookup[token] = len(self.lookup)
            self.reverse[self.lookup[token]] = token

    def encode(self, labeled: LabeledScenario) -> int:
        token = self._to_token(labeled.action)
        return self.lookup.get(token, self.lookup["DRAW"])

    def decode(self, index: int) -> str:
        return self.reverse[index]

    def token_to_action(self, token: str, state: GameState, player_id: str) -> BotAction:
        token = token.upper()
        if token == "DRAW":
            return BotAction(ActionType.DRAW)
        if token == "PASS":
            return BotAction(ActionType.PASS)
        hand = next(player.hand for player in state.players if player.player_id == player_id)
        parts = token.split("_")
        if len(parts) < 3:
            return BotAction(ActionType.DRAW)
        if parts[1] == "NUMBER" and len(parts) == 4:
            color = Color[parts[2]]
            number = int(parts[3])
            card = self._find_card(hand, lambda c: c.type == CardType.NUMBER and c.color == color and c.number == number)
            if card:
                return BotAction(ActionType.PLAY, card=card)
        elif parts[1] in {"SKIP", "REVERSE", "DRAW", "DISCARD"}:
            card_type = CardType["_".join(parts[1:len(parts)-1]) if parts[1] == "DRAW" and parts[2] == "TWO" else parts[1]]
            if card_type == CardType.DRAW_TWO:
                color = Color[parts[3]]
            elif card_type == CardType.DISCARD_ALL:
                color = Color[parts[3]]
            else:
                color = Color[parts[2]]
            card = self._find_card(hand, lambda c: c.type == card_type and c.color == color)
            if card:
                return BotAction(ActionType.PLAY, card=card)
        elif parts[1] == "WILD" and len(parts) == 3:
            chosen_color = Color[parts[2]]
            card = self._find_card(hand, lambda c: c.type == CardType.WILD)
            if card:
                return BotAction(ActionType.PLAY, card=card, chosen_color=chosen_color)
        elif parts[1] == "WILD" and parts[2] == "DRAW" and len(parts) == 5:
            chosen_color = Color[parts[4]]
            card = self._find_card(hand, lambda c: c.type == CardType.WILD_DRAW_FOUR)
            if card:
                return BotAction(ActionType.PLAY, card=card, chosen_color=chosen_color)
        return BotAction(ActionType.DRAW)

    def _find_card(self, hand: Sequence[Card], predicate: Callable[[Card], bool]) -> Optional[Card]:
        for card in hand:
            if predicate(card):
                return card
        return None

    def _to_token(self, action: BotAction) -> str:
        if action.action_type == ActionType.DRAW:
            return "DRAW"
        if action.action_type == ActionType.PASS:
            return "PASS"
        card = action.card
        if card is None:
            return "UNKNOWN"
        if card.type == CardType.NUMBER:
            return f"PLAY_NUMBER_{card.color.value}_{card.number}"
        if card.type in {CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO, CardType.DISCARD_ALL}:
            return f"PLAY_{card.type.name}_{card.color.value}"
        if card.type == CardType.WILD:
            chosen = action.chosen_color.value if action.chosen_color else "NONE"
            return f"PLAY_WILD_{chosen}"
        if card.type == CardType.WILD_DRAW_FOUR:
            chosen = action.chosen_color.value if action.chosen_color else "NONE"
            return f"PLAY_WILD_DRAW_FOUR_{chosen}"
        return "UNKNOWN"

    def _enumerate_tokens(self) -> Iterable[str]:
        yield "DRAW"
        yield "PASS"
        for color in STANDARD_COLORS:
            for number in range(0, 10):
                yield f"PLAY_NUMBER_{color.value}_{number}"
            for card_type in [CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO, CardType.DISCARD_ALL]:
                yield f"PLAY_{card_type.name}_{color.value}"
        for color in STANDARD_COLORS:
            yield f"PLAY_WILD_{color.value}"
            yield f"PLAY_WILD_DRAW_FOUR_{color.value}"

# %% [code cell 7]
class ScenarioDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.metadata = metadata

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - trivial
        return self.features[index], self.labels[index]

# %% [code cell 8]
class StateBayesianNN(PyroModule):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        device = device or torch.device("cpu")
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = PyroModule[nn.Linear](hidden_dim // 2, num_classes)
        self.register_buffer(
            "_prior_weight_loc",
            torch.zeros(num_classes, hidden_dim // 2, device=device),
        )
        self.register_buffer(
            "_prior_weight_scale",
            torch.ones(num_classes, hidden_dim // 2, device=device),
        )
        self.register_buffer(
            "_prior_bias_loc",
            torch.zeros(num_classes, device=device),
        )
        self.register_buffer(
            "_prior_bias_scale",
            torch.ones(num_classes, device=device),
        )
        self.classifier.weight = PyroSample(
            dist.Normal(self._prior_weight_loc, self._prior_weight_scale).to_event(2)
        )
        self.classifier.bias = PyroSample(
            dist.Normal(self._prior_bias_loc, self._prior_bias_scale).to_event(1)
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        pyro.module("backbone", self.backbone)
        features = self.backbone(x)
        logits = self.classifier(features)
        log_probs = F.log_softmax(logits, dim=-1)
        pyro.deterministic("logits", logits)
        with pyro.plate("data", x.size(0)):
            pyro.sample("obs", dist.Categorical(logits=log_probs), obs=y)
        return log_probs


class StateBNNGuide(PyroModule):
    def __init__(self, model: StateBayesianNN, device: Optional[torch.device] = None) -> None:
        super().__init__()
        hidden_dim = model.classifier.in_features
        num_classes = model.classifier.out_features
        device = device or next(model.parameters()).device
        self.backbone = model.backbone
        self.weight_loc = PyroParam(torch.zeros(num_classes, hidden_dim, device=device))
        self.weight_scale = PyroParam(
            torch.ones(num_classes, hidden_dim, device=device),
            constraint=constraints.positive,
        )
        self.bias_loc = PyroParam(torch.zeros(num_classes, device=device))
        self.bias_scale = PyroParam(
            torch.ones(num_classes, device=device),
            constraint=constraints.positive,
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> None:
        pyro.module("backbone", self.backbone)
        pyro.sample(
            "classifier.weight",
            dist.Normal(self.weight_loc, self.weight_scale).to_event(2),
        )
        pyro.sample(
            "classifier.bias",
            dist.Normal(self.bias_loc, self.bias_scale).to_event(1),
        )

# %% [code cell 9]
@dataclass
class TrainingConfig:
    num_epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 5e-3
    clip_norm: float = 10.0
    validation_split: float = 0.1
    log_every: int = 1
    log_dir: Optional[Path] = Path("training_logs")
    run_name: Optional[str] = None
    log_to_console: bool = True
    log_to_file: bool = True
    progress_bar: bool = True
    num_workers: int = field(default_factory=lambda: DEFAULT_NUM_WORKERS)
    log_batches: bool = False
    batch_log_interval: int = 10


@dataclass
class TrainingArtifacts:
    model: StateBayesianNN
    guide: StateBNNGuide
    history: Dict[str, List[float]]
    action_encoder: ActionEncoder
    state_encoder: StateEncoder
    metadata: Dict[str, Any] = field(default_factory=dict)
    log_path: Optional[Path] = None


def train_bnn(
    dataset: ScenarioDataset,
    *,
    action_encoder: ActionEncoder,
    state_encoder: StateEncoder,
    config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
) -> TrainingArtifacts:
    config = config or TrainingConfig()
    device = device or DEFAULT_DEVICE
    use_cuda = device.type == "cuda"

    num_samples = len(dataset)
    if num_samples < 2:
        raise ValueError("ScenarioDataset must contain at least two samples for train/validation split.")

    val_size = max(1, int(num_samples * config.validation_split))
    if val_size >= num_samples:
        val_size = max(1, num_samples // 5)
    if val_size >= num_samples:
        val_size = num_samples - 1
    train_size = num_samples - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
    }
    if use_cuda:
        loader_kwargs["pin_memory"] = True
        if config.num_workers > 0:
            loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    input_dim = dataset.features.shape[1]
    num_classes = len(action_encoder.lookup)

    model = StateBayesianNN(input_dim, num_classes, device=device).to(device)
    guide = StateBNNGuide(model, device=device).to(device)
    optimizer = ClippedAdam({"lr": config.learning_rate, "clip_norm": config.clip_norm})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_seconds": [],
    }

    base_name = config.run_name or "bnn_training"
    run_label = f"{base_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = Path(config.log_dir) if config.log_dir else None
    log_path: Optional[Path] = None
    log_file = None
    csv_writer = None
    if config.log_to_file and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_label}.csv"
        log_file = log_path.open("w", newline="")
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "epoch_seconds",
                "timestamp",
                "device",
                "batch_size",
                "learning_rate",
            ]
        )

    if config.log_to_console:
        logger.info(
            f"[{run_label}] device={device} train={train_size} val={val_size} "
            f"batch_size={config.batch_size} lr={config.learning_rate:.2e} workers={config.num_workers}"
        )

    epoch_numbers = range(1, config.num_epochs + 1)
    progress = None
    try:
        if config.progress_bar and tqdm is not None:
            progress = tqdm(epoch_numbers, desc=f"{run_label}", leave=False)
            epoch_iterator = progress
        else:
            epoch_iterator = epoch_numbers

        for epoch in epoch_iterator:
            model.train()
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            epoch_loss_total = 0.0
            epoch_count = 0
            num_batches = len(train_loader)
            batch_idx = 0
            for batch_x, batch_y in train_loader:
                batch_idx += 1
                batch_x = batch_x.to(device, non_blocking=use_cuda)
                batch_y = batch_y.to(device, non_blocking=use_cuda)
                loss = svi.step(batch_x, batch_y)
                epoch_loss_total += loss
                epoch_count += batch_x.size(0)
                if config.log_batches:
                    should_log_batch = config.batch_log_interval <= 1 or batch_idx % config.batch_log_interval == 0
                    if config.log_to_console and should_log_batch:
                        batch_loss = loss / max(1, batch_x.size(0))
                        logger.info(
                            f"[{run_label}] Epoch {epoch:03d} Batch {batch_idx:04d}/{num_batches:04d} "
                            f"| batch_loss {batch_loss:.4f} | raw_loss {loss:.2f}"
                        )
            epoch_loss = epoch_loss_total / max(1, epoch_count)
            history["train_loss"].append(epoch_loss)

            model.eval()
            guide.eval()
            with torch.no_grad():
                val_loss_total = 0.0
                val_correct = 0
                val_total = 0
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device, non_blocking=use_cuda)
                    batch_y = batch_y.to(device, non_blocking=use_cuda)
                    loss = svi.evaluate_loss(batch_x, batch_y)
                    val_loss_total += loss
                    log_probs = model(batch_x)
                    preds = log_probs.argmax(dim=-1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)
            val_loss = val_loss_total / max(1, val_total)
            val_acc = val_correct / max(1, val_total)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if use_cuda:
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - start_time
            history["epoch_seconds"].append(epoch_time)

            if progress is not None:
                progress.set_postfix(
                    train=f"{epoch_loss:.4f}",
                    val=f"{val_loss:.4f}",
                    acc=f"{val_acc:.3f}",
                    time=f"{epoch_time:.1f}s",
                )

            should_log = config.log_every <= 1 or epoch % config.log_every == 0 or epoch == config.num_epochs
            message = (
                f"[{run_label}] Epoch {epoch:03d}/{config.num_epochs:03d} "
                f"| train {epoch_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f} | {epoch_time:.2f}s"
            )
            if config.log_to_console and should_log:
                logger.info(message)

            if csv_writer:
                csv_writer.writerow(
                    [
                        epoch,
                        float(epoch_loss),
                        float(val_loss),
                        float(val_acc),
                        epoch_time,
                        datetime.now().isoformat(),
                        str(device),
                        config.batch_size,
                        config.learning_rate,
                    ]
                )
                log_file.flush()
    finally:
        if progress is not None:
            progress.close()
        if log_file is not None:
            log_file.close()

    metadata = {
        "run_id": run_label,
        "device": str(device),
        "train_samples": train_size,
        "val_samples": val_size,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }

    artifacts = TrainingArtifacts(
        model=model,
        guide=guide,
        history=history,
        action_encoder=action_encoder,
        state_encoder=state_encoder,
        metadata=metadata,
        log_path=log_path,
    )

    if config.log_to_console and log_path is not None:
        logger.info(f"[{run_label}] Metrics logged to {log_path}")

    return artifacts

# %% [code cell 10]
def mc_predict(
    artifacts: TrainingArtifacts,
    features: torch.Tensor,
    *,
    num_samples: int = 50,
    use_dropout: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or next(artifacts.model.parameters()).device
    artifacts.model.to(device)
    artifacts.guide.to(device)

    if use_dropout:
        artifacts.model.backbone.train()
    else:
        artifacts.model.backbone.eval()

    predictive = pyro.infer.Predictive(
        artifacts.model,
        guide=artifacts.guide,
        num_samples=num_samples,
        return_sites=("logits",),
    )
    samples = predictive(features.to(device))
    logits = samples["logits"]  # [num_samples, batch, num_classes]
    probs = logits.softmax(dim=-1)
    mean_probs = probs.mean(dim=0)
    predictive_entropy = -(mean_probs * mean_probs.clamp(min=1e-8).log()).sum(dim=-1)
    expected_entropy = -(
        probs * probs.clamp(min=1e-8).log()
    ).sum(dim=-1).mean(dim=0)
    mutual_information = predictive_entropy - expected_entropy
    predictions = mean_probs.argmax(dim=-1)

    return {
        "mean_probs": mean_probs.detach().cpu(),
        "predictive_entropy": predictive_entropy.detach().cpu(),
        "mutual_information": mutual_information.detach().cpu(),
        "predictions": predictions.detach().cpu(),
    }

# %% [code cell 11]
def build_synthetic_dataset(
    *,
    mode: GameMode,
    num_scenarios: int,
    scenario_mix: Optional[Dict[ScenarioType, float]] = None,
    rng_seed: int = 1024,
    include_random_bot: bool = True,
) -> Tuple[ScenarioDataset, List[LabeledScenario], StateEncoder, ActionEncoder]:
    scenario_mix = scenario_mix or {
        ScenarioType.FINISHER: 0.20,
        ScenarioType.DEFENDER: 0.20,
        ScenarioType.SETUP: 0.20,
        ScenarioType.COLOR_TRAP: 0.20,
        ScenarioType.WILD_DILEMMA: 0.20,
    }
    rng = random.Random(rng_seed)
    forge = ScenarioForge(rng=rng)
    params = ScenarioParameters(scenario_type=ScenarioType.FINISHER, mode=mode)

    scenarios: List[ScenarioExample] = []
    batch_size = max(1, num_scenarios // 10)
    while len(scenarios) < num_scenarios:
        needed = min(batch_size, num_scenarios - len(scenarios))
        scenarios.extend(
            forge.generate_batch(
                scenario_mix,
                mode=mode,
                batch_size=needed,
                base_params=params,
            )
        )

    rng_seed += 1
    bots: List[BotPolicy] = [
        OracleBot(rollout_count=8, rollout_depth=4, rng=random.Random(rng_seed + 1)),
        AggressorBot(rng=random.Random(rng_seed + 2)),
        SupporterBot(rng=random.Random(rng_seed + 3)),
        ConservativeBot(rng=random.Random(rng_seed + 4)),
    ]
    if include_random_bot:
        bots.append(RandomBot(rng=random.Random(rng_seed + 5)))

    labeler = ScenarioLabeler(rng=random.Random(rng_seed + 6))
    labeled = labeler.label(scenarios, bots)

    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()

    feature_rows: List[np.ndarray] = []
    label_rows: List[int] = []
    metadata: List[Dict[str, Any]] = []

    for labeled_example in labeled:
        feature_rows.append(state_encoder.encode(labeled_example))
        label_rows.append(action_encoder.encode(labeled_example))
        metadata.append(
            {
                "mode": labeled_example.scenario.state.mode.value,
                "scenario_type": labeled_example.scenario.scenario_type.value,
                "bot": labeled_example.bot_name,
                "action_token": action_encoder.decode(label_rows[-1]),
                "action_success": labeled_example.action_success,
            }
        )

    features = np.stack(feature_rows)
    labels = np.array(label_rows, dtype=np.int64)
    dataset = ScenarioDataset(features, labels, metadata)
    return dataset, labeled, state_encoder, action_encoder

# %% [code cell 12]
def export_artifacts(
    artifacts: TrainingArtifacts,
    *,
    output_dir: Path,
    model_tag: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    param_store_path = output_dir / f"{model_tag}_param_store.pt"
    meta_path = output_dir / f"{model_tag}_meta.json"

    pyro.get_param_store().save(str(param_store_path))

    meta = {
        "model_tag": model_tag,
        "feature_size": artifacts.state_encoder.feature_size,
        "num_actions": len(artifacts.action_encoder.lookup),
        "action_tokens": artifacts.action_encoder.reverse,
        "history": artifacts.history,
    }
    if extra_metadata:
        meta.update(extra_metadata)

    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, default=str)

    return {"param_store": str(param_store_path), "metadata": str(meta_path)}

# %% [code cell 13]
class BNNBot(BotPolicy):
    def __init__(
        self,
        artifacts: TrainingArtifacts,
        *,
        persona_hint: str = "OracleBot",
        rng: Optional[random.Random] = None,
        uncertainty_threshold: float = 0.25,
        mc_samples: int = 40,
    ) -> None:
        super().__init__("BNNBot", rng=rng)
        self.artifacts = artifacts
        self.persona_hint = persona_hint if persona_hint in BOT_ORDER else BOT_ORDER[0]
        self.uncertainty_threshold = uncertainty_threshold
        self.mc_samples = mc_samples

    def decide(self, engine: UnoEngine, state: GameState, player_id: str) -> BotAction:
        scenario_type = self._infer_scenario_type(state, player_id)
        metadata = self._infer_metadata(state, player_id)
        features = self.artifacts.state_encoder.encode_components(
            state=state,
            target_player=player_id,
            bot_name=self.persona_hint,
            scenario_type=scenario_type,
            metadata=metadata,
        )
        features_tensor = torch.from_numpy(features).unsqueeze(0)
        result = mc_predict(
            self.artifacts,
            features_tensor,
            num_samples=self.mc_samples,
            use_dropout=True,
        )
        mean_probs = result["mean_probs"][0]
        entropy = result["predictive_entropy"][0].item()
        mutual_info = result["mutual_information"][0].item()

        token_idx = int(result["predictions"][0].item())
        token = self.artifacts.action_encoder.decode(token_idx)
        action = self.artifacts.action_encoder.token_to_action(token, state, player_id)

        if action.action_type == ActionType.DRAW:
            # fallback to valid move with highest probability mass
            valid_actions = self.enumerate_actions(engine, state, player_id)
            ranked = []
            for candidate in valid_actions:
                candidate_token = self.artifacts.action_encoder._to_token(candidate)
                candidate_idx = self.artifacts.action_encoder.lookup.get(candidate_token)
                if candidate_idx is None:
                    continue
                ranked.append((mean_probs[candidate_idx].item(), candidate))
            if ranked:
                ranked.sort(key=lambda pair: pair[0], reverse=True)
                action = ranked[0][1]

        action.info.update(
            {
                "bnn_entropy": entropy,
                "bnn_mutual_information": mutual_info,
                "persona_hint": self.persona_hint,
                "scenario_type": scenario_type.value,
            }
        )
        return action

    def _infer_scenario_type(self, state: GameState, player_id: str) -> ScenarioType:
        hand = next(player.hand for player in state.players if player.player_id == player_id)
        if state.pending_action and state.pending_action.type == PendingActionType.DRAW_STACK:
            return ScenarioType.DEFENDER
        if len(hand) <= 2:
            return ScenarioType.FINISHER
        if any(card.type in {CardType.WILD, CardType.WILD_DRAW_FOUR} for card in hand):
            return ScenarioType.WILD_DILEMMA
        top_color = state.current_color or state.discard_pile[-1].color
        opponent_colors = self._opponent_color_counts(state, player_id)
        if top_color in opponent_colors and opponent_colors[top_color] <= 1:
            return ScenarioType.COLOR_TRAP
        return ScenarioType.SETUP

    def _infer_metadata(self, state: GameState, player_id: str) -> Dict[str, Any]:
        hand = next(player.hand for player in state.players if player.player_id == player_id)
        teammate = state.teammate_id(player_id)
        teammate_hand = next(player.hand for player in state.players if player.player_id == teammate)
        opponents = [p for p in state.players if state.team_map[p.player_id] != state.team_map[player_id]]
        opponent_colors = self._aggregate_color_counts([op.hand for op in opponents])
        weakest_color = min(opponent_colors.items(), key=lambda kv: kv[1])[0] if opponent_colors else None
        metadata = {
            "distance_to_victory": max(0, len(hand) - 1),
            "draw_stack_penalty": state.pending_action.draw_penalty if state.pending_action else 0,
            "available_wild": any(card.type in {CardType.WILD, CardType.WILD_DRAW_FOUR} for card in hand),
            "high_stakes": state.draw_stack_total >= 4,
            "teammate_preferred_color": self._dominant_color(teammate_hand),
            "weak_opponent_color": weakest_color,
        }
        return metadata

    def _dominant_color(self, hand: Sequence[Card]) -> Optional[Color]:
        counts = self._aggregate_color_counts([hand])
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _aggregate_color_counts(self, hands: Sequence[Sequence[Card]]) -> Dict[Optional[Color], int]:
        counts: Dict[Optional[Color], int] = {color: 0 for color in STANDARD_COLORS}
        for hand in hands:
            for card in hand:
                if card.color in counts:
                    counts[card.color] += 1
        return counts

    def _opponent_color_counts(self, state: GameState, player_id: str) -> Dict[Color, int]:
        opponents = [p for p in state.players if state.team_map[p.player_id] != state.team_map[player_id]]
        counts = {color: 0 for color in STANDARD_COLORS}
        for player in opponents:
            for card in player.hand:
                if card.color in counts:
                    counts[card.color] += 1
        return counts

# %% [code cell 14]
def evaluate_state_with_bnn(
    artifacts: TrainingArtifacts,
    *,
    state: GameState,
    target_player: str,
    persona_name: str,
    scenario_type: ScenarioType,
    metadata: Dict[str, Any],
    num_samples: int = 40,
) -> Dict[str, Any]:
    features = artifacts.state_encoder.encode_components(
        state=state,
        target_player=target_player,
        bot_name=persona_name,
        scenario_type=scenario_type,
        metadata=metadata,
    )
    features_tensor = torch.from_numpy(features).unsqueeze(0)
    mc = mc_predict(
        artifacts,
        features_tensor,
        num_samples=num_samples,
        use_dropout=True,
    )
    return {
        "features": features,
        "mc": mc,
    }

# %% [code cell 15]
@dataclass
class ActiveLog:
    state: GameState
    target_player: str
    persona_name: str
    scenario_type: ScenarioType
    metadata: Dict[str, Any]
    action: BotAction
    resulting_state: GameState
    action_token: str
    bnn_entropy: float
    bnn_mutual_information: float


class ActiveCurriculum:
    def __init__(
        self,
        artifacts: TrainingArtifacts,
        *,
        uncertainty_threshold: float = 0.20,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.artifacts = artifacts
        self.uncertainty_threshold = uncertainty_threshold
        self.engine = UnoEngine()
        self.rng = rng or random.Random()
        self.bnn_helper = BNNBot(artifacts, rng=self.rng)

    def simulate_game(
        self,
        *,
        mode: GameMode,
        bot_assignments: Dict[str, BotPolicy],
        max_turns: int = 200,
    ) -> List[ActiveLog]:
        players = [Player(player_id=pid) for pid in PLAYER_SEQUENCE]
        state = self.engine.init_game(players, mode=mode)
        logs: List[ActiveLog] = []
        turns = 0
        while not state.round_over and turns < max_turns:
            pending = state.pending_action
            if pending and pending.type == PendingActionType.COLOR_CHOICE:
                chooser_id = pending.player_id
                chooser_bot = bot_assignments[chooser_id]
                chosen_color = chooser_bot.choose_color(state, chooser_id)
                state = self.engine.choose_color(state, chooser_id, chosen_color)
                continue

            current_player = state.current_player()
            bot = bot_assignments[current_player.player_id]
            scenario_type = self.bnn_helper._infer_scenario_type(state, current_player.player_id)
            metadata = self.bnn_helper._infer_metadata(state, current_player.player_id)
            evaluation = evaluate_state_with_bnn(
                self.artifacts,
                state=state,
                target_player=current_player.player_id,
                persona_name=bot.name,
                scenario_type=scenario_type,
                metadata=metadata,
            )
            mc = evaluation["mc"]
            entropy = float(mc["predictive_entropy"][0].item())
            mutual_info = float(mc["mutual_information"][0].item())
            predicted_idx = int(mc["predictions"][0].item())
            predicted_token = self.artifacts.action_encoder.decode(predicted_idx)

            action = bot.decide(self.engine, state, current_player.player_id)
            next_state, success = self._apply_action(state, current_player.player_id, action)

            if success and (mutual_info >= self.uncertainty_threshold or entropy >= self.uncertainty_threshold):
                logs.append(
                    ActiveLog(
                        state=state,
                        target_player=current_player.player_id,
                        persona_name=bot.name,
                        scenario_type=scenario_type,
                        metadata=metadata,
                        action=action,
                        resulting_state=next_state,
                        action_token=self.artifacts.action_encoder._to_token(action),
                        bnn_entropy=entropy,
                        bnn_mutual_information=mutual_info,
                    )
                )

            state = next_state
            turns += 1
        return logs

    def _apply_action(self, state: GameState, player_id: str, action: BotAction) -> Tuple[GameState, bool]:
        try:
            if action.action_type == ActionType.PLAY and action.card is not None:
                return self.engine.play_card(state, player_id, action.card, action.chosen_color), True
            if action.action_type == ActionType.DRAW:
                return self.engine.draw_card(state, player_id), True
            if action.action_type == ActionType.PASS:
                return self.engine.pass_turn(state, player_id), True
        except InvalidMoveError:
            return state, False
        return state, True

# %% [code cell 16]
def build_dataset_from_labeled(
    labeled: Sequence[LabeledScenario],
    *,
    state_encoder: StateEncoder,
    action_encoder: ActionEncoder,
) -> ScenarioDataset:
    feature_rows: List[np.ndarray] = []
    label_rows: List[int] = []
    metadata: List[Dict[str, Any]] = []

    for labeled_example in labeled:
        feature_rows.append(state_encoder.encode(labeled_example))
        label_rows.append(action_encoder.encode(labeled_example))
        metadata.append(
            {
                "mode": labeled_example.scenario.state.mode.value,
                "scenario_type": labeled_example.scenario.scenario_type.value,
                "bot": labeled_example.bot_name,
                "action_token": action_encoder.decode(label_rows[-1]),
                "action_success": labeled_example.action_success,
            }
        )

    features = np.stack(feature_rows)
    labels = np.array(label_rows, dtype=np.int64)
    return ScenarioDataset(features, labels, metadata)

# %% [code cell 17]
def run_curriculum_loop(
    *,
    mode: GameMode,
    initial_scenarios: int = 5000,
    iterations: int = 2,
    games_per_iteration: int = 10,
    uncertainty_threshold: float = 0.25,
    epsilon: float = 0.005,
    rng_seed: int = 7,
) -> Tuple[TrainingArtifacts, List[LabeledScenario]]:
    rng = random.Random(rng_seed)
    device = DEFAULT_DEVICE
    if device.type == "cuda":
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        logger.info(f"[Curriculum] Mode={mode.value} | using GPU: {gpu_name}")
    else:
        logger.info(f"[Curriculum] Mode={mode.value} | using CPU backend")
    log_dir = Path("training_logs") / mode.value.lower()
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset, labeled, state_encoder, action_encoder = build_synthetic_dataset(
        mode=mode,
        num_scenarios=initial_scenarios,
        rng_seed=rng_seed,
    )

    pyro.clear_param_store()
    bootstrap_config = TrainingConfig(
        num_epochs=18,
        batch_size=96,
        learning_rate=2.5e-3,
        run_name=f"{mode.value.lower()}_bootstrap",
        log_dir=log_dir,
    )
    artifacts = train_bnn(
        dataset,
        action_encoder=action_encoder,
        state_encoder=state_encoder,
        config=bootstrap_config,
        device=device,
    )

    base_metadata = {
        "mode": mode.value,
        "initial_scenarios": initial_scenarios,
        "games_per_iteration": games_per_iteration,
        "uncertainty_threshold": uncertainty_threshold,
        "epsilon": epsilon,
    }
    artifacts.metadata.update(base_metadata)
    artifacts.metadata["phase"] = "bootstrap"
    artifacts.metadata["iteration"] = 0
    if artifacts.log_path:
        logger.info(f"[Curriculum] Bootstrap metrics logged to {artifacts.log_path}")

    labeler = ScenarioLabeler(rng=rng)
    iterations_completed = 0

    for iteration in range(1, iterations + 1):
        curriculum = ActiveCurriculum(
            artifacts,
            uncertainty_threshold=uncertainty_threshold,
            rng=rng,
        )
        logs: List[ActiveLog] = []
        for game_index in range(games_per_iteration):
            bot_assignments = {
                "P0": AggressorBot(rng=rng),
                "P1": SupporterBot(rng=rng),
                "P2": ConservativeBot(rng=rng),
                "P3": RandomBot(rng=rng),
            }
            logs.extend(
                curriculum.simulate_game(
                    mode=mode,
                    bot_assignments=bot_assignments,
                )
            )
        if not logs:
            logger.info(f"Iteration {iteration}: no high-uncertainty scenarios captured.")
            continue

        new_labeled = labeler.logs_to_labeled(logs)
        labeled.extend(new_labeled)
        dataset = build_dataset_from_labeled(
            labeled,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
        )

        pyro.clear_param_store()
        iter_config = TrainingConfig(
            num_epochs=12,
            batch_size=96,
            learning_rate=1.8e-3,
            run_name=f"{mode.value.lower()}_iter{iteration:02d}",
            log_dir=log_dir,
        )
        new_artifacts = train_bnn(
            dataset,
            action_encoder=action_encoder,
            state_encoder=state_encoder,
            config=iter_config,
            device=device,
        )

        previous_val_loss = artifacts.history["val_loss"][-1]
        new_val_loss = new_artifacts.history["val_loss"][-1]
        improvement = previous_val_loss - new_val_loss
        logger.info(
            f"Iteration {iteration}: captured {len(logs)} scenarios | val improvement {improvement:.4f}"
        )
        if new_artifacts.log_path:
            logger.info(f"[Curriculum] Iteration {iteration:02d} metrics logged to {new_artifacts.log_path}")

        new_artifacts.metadata.update(base_metadata)
        new_artifacts.metadata.update(
            {
                "phase": f"iteration_{iteration}",
                "iteration": iteration,
                "captured_scenarios": len(logs),
            }
        )
        artifacts = new_artifacts
        iterations_completed = iteration
        if improvement < epsilon:
            logger.info("Stopping early due to marginal improvement threshold.")
            break

    artifacts.action_encoder = action_encoder
    artifacts.state_encoder = state_encoder
    artifacts.metadata["phase"] = "completed"
    artifacts.metadata["iteration"] = iterations_completed
    artifacts.metadata["iterations_completed"] = iterations_completed
    artifacts.metadata["total_labeled_examples"] = len(labeled)
    return artifacts, labeled

# %% [script entrypoint]

def configure_root_logger(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    level_name = level.upper()
    supported_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    resolved_level = supported_levels.get(level_name, logging.INFO)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        root_logger.addHandler(file_handler)


def resolve_device(preference: str) -> torch.device:
    preference = preference.lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this system.")
        return torch.device("cuda")
    raise ValueError(f"Unknown device preference '{preference}'. Use 'auto', 'cpu', or 'cuda'.")


def parse_scenario_mix(argument: Optional[str]) -> Optional[Dict[ScenarioType, float]]:
    if not argument:
        return None
    mix: Dict[ScenarioType, float] = {}
    for part in argument.split(","):
        if "=" not in part:
            raise ValueError(f"Invalid scenario mix component '{part}'. Expected format TYPE=WEIGHT.")
        name, weight = part.split("=", 1)
        name = name.strip().upper()
        try:
            scenario = ScenarioType[name]
        except KeyError as exc:  # pragma: no cover - user input validation
            raise ValueError(f"Unknown scenario type '{name}'.") from exc
        mix[scenario] = float(weight)
    total = sum(mix.values())
    if not math.isclose(total, 1.0, rel_tol=1e-3):
        mix = {key: value / total for key, value in mix.items()}
    return mix


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Train the UNO Bayesian neural network locally with configurable logging.",
    )

    local_cpu_count = max(1, os.cpu_count() or 1)
    default_workers = min(2, local_cpu_count)

    parser.add_argument("--num-scenarios", type=int, default=2000, help="Number of synthetic scenarios to generate.")
    parser.add_argument(
        "--mode",
        type=str,
        default=GameMode.CLASSIC_2V2.value,
        choices=[mode.value for mode in GameMode],
        help="UNO ruleset to target.",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Optimizer learning rate.")
    parser.add_argument("--clip-norm", type=float, default=10.0, help="Gradient clipping norm.")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation set fraction.")
    parser.add_argument("--log-every", type=int, default=1, help="Log every N epochs (>=1).")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("training_logs"),
        help="Directory for CSV metrics and optional log files.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional explicit path for the detailed training log.",
    )
    parser.add_argument(
        "--no-text-log",
        dest="write_text_log",
        action="store_false",
        help="Disable writing a detailed text log file.",
    )
    parser.set_defaults(write_text_log=True)
    parser.add_argument("--log-level", type=str, default="INFO", help="Console log level (DEBUG, INFO, WARNING...).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device.")
    parser.add_argument("--num-workers", type=int, default=default_workers, help="DataLoader worker processes.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name for this training run.")
    parser.add_argument("--rng-seed", type=int, default=1024, help="Random seed for dataset generation.")
    parser.add_argument("--scenario-mix", type=str, default=None, help="Custom scenario mix, e.g. FINISHER=0.4,DEFENDER=0.6.")
    parser.add_argument("--no-random-bot", dest="include_random_bot", action="store_false", help="Exclude random bot persona.")
    parser.set_defaults(include_random_bot=True)
    parser.add_argument("--log-batches", action="store_true", help="Emit logs for individual batches.")
    parser.add_argument("--batch-log-interval", type=int, default=10, help="When logging batches, log every N batches.")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress bar.")
    parser.set_defaults(progress=True)
    parser.add_argument("--export", action="store_true", help="Export trained artifacts to disk after training.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store exported artifacts when --export is used.",
    )
    parser.add_argument("--model-tag", type=str, default=None, help="Tag/name for exported artifacts.")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"local_{args.mode}_{timestamp}"
    mode = GameMode(args.mode)

    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path: Optional[Path]
    if args.write_text_log:
        log_file_path = args.log_file or (log_dir / f"{run_name}.log")
    else:
        log_file_path = None

    configure_root_logger(args.log_level, log_file_path)

    logger.info("Preparing synthetic dataset...")
    scenario_mix = parse_scenario_mix(args.scenario_mix)
    dataset, labeled_examples, state_encoder, action_encoder = build_synthetic_dataset(
        mode=mode,
        num_scenarios=args.num_scenarios,
        scenario_mix=scenario_mix,
        rng_seed=args.rng_seed,
        include_random_bot=args.include_random_bot,
    )
    logger.info(
        "Dataset ready | samples=%d | features=%d | labels=%d",
        len(dataset),
        dataset.features.shape[1],
        len(action_encoder.lookup),
    )
    logger.debug("Labeled persona decisions: %d", len(labeled_examples))

    pyro.clear_param_store()

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_norm=args.clip_norm,
        validation_split=args.validation_split,
        log_every=max(1, args.log_every),
        log_dir=log_dir,
        run_name=run_name,
        log_to_console=True,
        log_to_file=True,
        progress_bar=args.progress,
        num_workers=max(0, args.num_workers),
        log_batches=args.log_batches,
        batch_log_interval=max(1, args.batch_log_interval),
    )

    artifacts = train_bnn(
        dataset,
        action_encoder=action_encoder,
        state_encoder=state_encoder,
        config=training_config,
        device=device,
    )

    logger.info(
        "Training complete | final_train_loss=%.4f | final_val_loss=%.4f | final_val_acc=%.3f",
        artifacts.history["train_loss"][-1],
        artifacts.history["val_loss"][-1],
        artifacts.history["val_acc"][-1],
    )

    if args.export:
        model_tag = args.model_tag or run_name
        export_dir = args.output_dir.resolve()
        logger.info("Exporting artifacts to %s (tag=%s)", export_dir, model_tag)
        export_paths = export_artifacts(
            artifacts,
            output_dir=export_dir,
            model_tag=model_tag,
            extra_metadata={
                "mode": mode.value,
                "num_scenarios": args.num_scenarios,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": str(device),
                "run_name": run_name,
                "timestamp": timestamp,
            },
        )
        logger.info("Artifacts exported: %s", export_paths)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli()
