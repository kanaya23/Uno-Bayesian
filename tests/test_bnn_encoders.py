from __future__ import annotations

import random

import pytest

from notebooks.uno_bnn_curriculum_converted import (
    ActionEncoder,
    ActionType,
    BotAction,
    LabeledScenario,
    ScenarioExample,
    ScenarioForge,
    ScenarioParameters,
    ScenarioType,
    StateEncoder,
    action_encoder_signature,
    state_encoder_signature,
)
from uno_engine import UnoEngine
from uno_engine.models import Card, CardType, Color, GameMode, Player


def _basic_state(mode: GameMode = GameMode.GO_WILD_2V2) -> tuple[ScenarioExample, str, Card]:
    engine = UnoEngine()
    players = [Player(player_id=f"P{i}") for i in range(4)]
    state = engine.init_game(players, mode=mode)

    target_player = state.players[0].player_id
    team_map = dict(state.team_map)
    player_order = tuple(player.player_id for player in state.players)
    params = ScenarioParameters(
        scenario_type=ScenarioType.SETUP,
        mode=mode,
        target_player=target_player,
        player_order=player_order,
        team_map=team_map,
    )
    scenario = ScenarioExample(
        state=state,
        target_player=target_player,
        scenario_type=ScenarioType.SETUP,
        parameters=params,
        metadata={},
    )

    discard_all = Card(color=Color.RED, type=CardType.DISCARD_ALL, value=40)
    state.players[0].hand.append(discard_all)
    return scenario, target_player, discard_all


def test_action_encoder_discard_all_round_trip() -> None:
    scenario, player_id, discard_all = _basic_state()

    action = BotAction(ActionType.PLAY, card=discard_all)
    labeled = LabeledScenario(
        scenario=scenario,
        bot_name="TestBot",
        action=action,
        resulting_state=scenario.state,
        action_success=True,
    )

    encoder = ActionEncoder()
    index = encoder.encode(labeled)
    token = encoder.decode(index)

    assert token == "PLAY_DISCARD_ALL_RED"

    decoded = encoder.token_to_action(token, scenario.state, player_id)
    assert decoded.action_type == ActionType.PLAY
    assert decoded.card is not None
    assert decoded.card.type == CardType.DISCARD_ALL
    assert decoded.card.color == Color.RED


def test_action_encoder_encode_unknown_raises() -> None:
    scenario, _, _ = _basic_state(mode=GameMode.CLASSIC_2V2)
    invalid_action = BotAction(ActionType.PLAY, card=None)
    labeled = LabeledScenario(
        scenario=scenario,
        bot_name="TestBot",
        action=invalid_action,
        resulting_state=scenario.state,
        action_success=False,
    )

    encoder = ActionEncoder()
    with pytest.raises(ValueError):
        encoder.encode(labeled)


def test_state_encoder_feature_size_matches_vector_length() -> None:
    forge = ScenarioForge(rng=random.Random(42))
    scenario = forge.generate(ScenarioParameters(scenario_type=ScenarioType.SETUP, mode=GameMode.CLASSIC_2V2))
    state_encoder = StateEncoder(player_order=forge.player_order)

    labeled = LabeledScenario(
        scenario=scenario,
        bot_name="OracleBot",
        action=BotAction(ActionType.DRAW),
        resulting_state=scenario.state,
        action_success=True,
    )

    vector = state_encoder.encode(labeled)
    assert vector.shape[0] == state_encoder.feature_size


def test_encoder_signatures_are_deterministic() -> None:
    state_encoder = StateEncoder(player_order=("P0", "P1", "P2", "P3"))
    team_map_a = {"P0": 0, "P1": 1, "P2": 0, "P3": 1}
    team_map_b = {"P0": 0, "P1": 0, "P2": 1, "P3": 1}

    sig_a = state_encoder_signature(state_encoder, team_map_a)
    sig_b = state_encoder_signature(state_encoder, team_map_b)
    assert sig_a != sig_b

    action_encoder_a = ActionEncoder()
    action_encoder_b = ActionEncoder()
    assert action_encoder_signature(action_encoder_a) == action_encoder_signature(action_encoder_b)
