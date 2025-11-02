from __future__ import annotations

from collections import Counter
from random import Random

import pytest

from uno_engine import (
    Card,
    CardType,
    Color,
    GameMode,
    GameState,
    PendingActionType,
    PlayDirection,
    Player,
    InvalidMoveError,
    UnoEngine,
)
from uno_engine.deck import build_go_wild_deck, build_standard_deck


def make_players(team0_score: int = 0, team1_score: int = 0) -> list[Player]:
    return [
        Player(player_id="A", score=team0_score),
        Player(player_id="B", score=team1_score),
        Player(player_id="C", score=team0_score),
        Player(player_id="D", score=team1_score),
    ]


def team_map_for(players: list[Player]) -> dict[str, int]:
    return {
        players[0].player_id: 0,
        players[1].player_id: 1,
        players[2].player_id: 0,
        players[3].player_id: 1,
    }


def team_scores_for(players: list[Player]) -> dict[int, int]:
    return {0: players[0].score, 1: players[1].score}


def total_cards_in_state(state: GameState) -> int:
    return (
        sum(len(player.hand) for player in state.players)
        + len(state.draw_pile)
        + len(state.discard_pile)
    )


def rig_deck_for_start_card(start_card: Card, mode: GameMode) -> list[Card]:
    base_deck = build_standard_deck() if mode == GameMode.CLASSIC_2V2 else build_go_wild_deck()
    base_deck.remove(start_card)
    dealt_cards = 4 * 7
    target_index = len(base_deck) - dealt_cards
    base_deck.insert(target_index, start_card)
    return base_deck


def test_build_standard_deck_distribution() -> None:
    deck = build_standard_deck()
    assert len(deck) == 108

    colored_cards = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]
    number_counts = {color: Counter() for color in colored_cards}
    action_counts = {color: Counter() for color in colored_cards}
    wild_counts = Counter()

    for card in deck:
        if card.type == CardType.NUMBER:
            number_counts[card.color][card.number] += 1
        elif card.type in {CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO}:
            action_counts[card.color][card.type] += 1
        else:
            wild_counts[card.type] += 1

    for color in colored_cards:
        assert number_counts[color][0] == 1
        for number in range(1, 10):
            assert number_counts[color][number] == 2
        assert action_counts[color][CardType.SKIP] == 2
        assert action_counts[color][CardType.REVERSE] == 2
        assert action_counts[color][CardType.DRAW_TWO] == 2

    assert wild_counts[CardType.WILD] == 4
    assert wild_counts[CardType.WILD_DRAW_FOUR] == 4
    assert CardType.DISCARD_ALL not in {card.type for card in deck}


def test_build_go_wild_deck_composition() -> None:
    deck = build_go_wild_deck()
    assert len(deck) == 224

    discard_all_counts = Counter(card.color for card in deck if card.type == CardType.DISCARD_ALL)
    for color in [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]:
        assert discard_all_counts[color] == 2


def test_init_game_requires_exactly_four_players() -> None:
    engine = UnoEngine(Random(0))
    players = [Player(player_id=pid, score=0) for pid in ("A", "B", "C")]

    with pytest.raises(ValueError):
        engine.init_game(players)


def test_init_game_rejects_mismatched_team_scores() -> None:
    engine = UnoEngine(Random(0))
    players = [
        Player(player_id="A", score=10),
        Player(player_id="B", score=0),
        Player(player_id="C", score=0),
        Player(player_id="D", score=0),
    ]

    with pytest.raises(ValueError):
        engine.init_game(players)


def test_init_game_sets_team_data_and_shared_hands() -> None:
    engine = UnoEngine(Random(0))
    state = engine.init_game(make_players())

    expected_map = {
        state.players[0].player_id: 0,
        state.players[1].player_id: 1,
        state.players[2].player_id: 0,
        state.players[3].player_id: 1,
    }
    assert state.team_map == expected_map
    assert state.team_scores == {0: 0, 1: 0}
    assert state.mode == GameMode.CLASSIC_2V2

    visible = state.visible_hands_for(state.players[0].player_id)
    assert set(visible.keys()) == {state.players[0].player_id, state.players[2].player_id}
    for pid, cards in visible.items():
        player = next(p for p in state.players if p.player_id == pid)
        assert len(cards) == len(player.hand)
    assert total_cards_in_state(state) == 108


def test_init_game_go_wild_uses_double_deck() -> None:
    engine = UnoEngine(Random(0))
    state = engine.init_game(make_players(), mode=GameMode.GO_WILD_2V2)

    assert state.mode == GameMode.GO_WILD_2V2
    assert total_cards_in_state(state) == 224

    all_cards = [card for player in state.players for card in player.hand]
    all_cards.extend(state.draw_pile)
    all_cards.extend(state.discard_pile)
    assert sum(1 for card in all_cards if card.type == CardType.DISCARD_ALL) == 8


def test_first_card_skip_advances_to_second_player(monkeypatch: pytest.MonkeyPatch) -> None:
    start_card = Card(color=Color.RED, type=CardType.SKIP, value=20)
    deck = rig_deck_for_start_card(start_card, GameMode.CLASSIC_2V2)

    monkeypatch.setattr("uno_engine.engine.build_deck_for_mode", lambda mode: list(deck))
    monkeypatch.setattr("uno_engine.engine.shuffle_in_place", lambda cards, rng: None)

    engine = UnoEngine(Random(0))
    state = engine.init_game(make_players())

    assert state.discard_pile[-1] == start_card
    assert state.current_player_index == 1


def test_first_card_draw_two_triggers_expected_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    start_card = Card(color=Color.BLUE, type=CardType.DRAW_TWO, value=20)
    deck = rig_deck_for_start_card(start_card, GameMode.CLASSIC_2V2)

    monkeypatch.setattr("uno_engine.engine.build_deck_for_mode", lambda mode: list(deck))
    monkeypatch.setattr("uno_engine.engine.shuffle_in_place", lambda cards, rng: None)

    engine = UnoEngine(Random(0))
    state = engine.init_game(make_players())

    assert state.discard_pile[-1] == start_card
    assert len(state.players[0].hand) == 9
    assert state.current_player_index == 1


def test_first_card_wild_requires_color_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    start_card = Card(color=Color.WILD, type=CardType.WILD, value=50)
    deck = rig_deck_for_start_card(start_card, GameMode.CLASSIC_2V2)

    monkeypatch.setattr("uno_engine.engine.build_deck_for_mode", lambda mode: list(deck))
    monkeypatch.setattr("uno_engine.engine.shuffle_in_place", lambda cards, rng: None)

    engine = UnoEngine(Random(0))
    state = engine.init_game(make_players())

    assert state.pending_action is not None
    assert state.pending_action.type == PendingActionType.COLOR_CHOICE
    assert state.pending_action.player_id == state.players[0].player_id
    assert state.current_color is None


def test_wild_draw_four_start_card_is_buried(monkeypatch: pytest.MonkeyPatch) -> None:
    start_card = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)
    deck = rig_deck_for_start_card(start_card, GameMode.CLASSIC_2V2)

    inserted_positions: list[int] = []

    def fake_bury(cards: list[Card], card: Card, rng: Random) -> None:
        inserted_positions.append(len(cards))
        cards.insert(0, card)

    monkeypatch.setattr("uno_engine.engine.build_deck_for_mode", lambda mode: list(deck))
    monkeypatch.setattr("uno_engine.engine.shuffle_in_place", lambda cards, rng: None)
    monkeypatch.setattr("uno_engine.engine.bury_card", fake_bury)

    engine = UnoEngine(Random(0))
    state = engine.init_game(make_players())

    assert state.discard_pile[-1].type != CardType.WILD_DRAW_FOUR
    assert inserted_positions


def test_go_wild_discard_all_discards_matching_color() -> None:
    engine = UnoEngine(Random(0))
    players = make_players()

    discard_all_blue = Card(color=Color.BLUE, type=CardType.DISCARD_ALL, value=40)
    other_blue = Card(color=Color.BLUE, type=CardType.NUMBER, number=6, value=6)
    off_color = Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)

    players[0].hand = [discard_all_blue, other_blue, off_color]
    players[1].hand = []
    players[2].hand = []
    players[3].hand = []

    state = GameState(
        players=players,
        draw_pile=[],
        discard_pile=[Card(color=Color.BLUE, type=CardType.NUMBER, number=9, value=9)],
        current_player_index=0,
        play_direction=PlayDirection.CLOCKWISE,
        current_color=Color.BLUE,
        mode=GameMode.GO_WILD_2V2,
        team_map=team_map_for(players),
        team_scores=team_scores_for(players),
    )

    new_state = engine.play_card(state, "A", discard_all_blue)

    assert other_blue not in new_state.players[0].hand
    assert off_color in new_state.players[0].hand
    assert new_state.current_color == Color.BLUE
    assert new_state.pending_action is None
    assert new_state.current_player_index == 1
    assert new_state.draw_stack_total == 0
    assert discard_all_blue in new_state.discard_pile
    assert other_blue in new_state.discard_pile


def test_go_wild_draw_two_creates_stack_and_forces_draw_two() -> None:
    engine = UnoEngine(Random(0))

    draw_two_red = Card(color=Color.RED, type=CardType.DRAW_TWO, value=20)
    draw_two_blue = Card(color=Color.BLUE, type=CardType.DRAW_TWO, value=20)
    wild_draw_four = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)

    players = make_players()
    filler_a = Card(color=Color.GREEN, type=CardType.NUMBER, number=0, value=0)
    filler_b = Card(color=Color.YELLOW, type=CardType.NUMBER, number=9, value=9)
    players[0].hand = [draw_two_red, filler_a]
    players[1].hand = [draw_two_blue, wild_draw_four, filler_b]
    players[2].hand = [Card(color=Color.YELLOW, type=CardType.NUMBER, number=5, value=5)]
    players[3].hand = [Card(color=Color.GREEN, type=CardType.NUMBER, number=7, value=7)]

    state = GameState(
        players=players,
        draw_pile=[],
        discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
        current_player_index=0,
        play_direction=PlayDirection.CLOCKWISE,
        current_color=Color.RED,
        mode=GameMode.GO_WILD_2V2,
        team_map=team_map_for(players),
        team_scores=team_scores_for(players),
    )

    after_a = engine.play_card(state, "A", draw_two_red)

    assert after_a.pending_action is not None
    assert after_a.pending_action.type == PendingActionType.DRAW_STACK
    assert after_a.pending_action.player_id == "B"
    assert after_a.pending_action.draw_penalty == 2
    assert after_a.current_player_index == 1
    assert after_a.draw_stack_total == 2

    moves = engine.get_valid_moves(after_a, "B")
    assert moves == [draw_two_blue]

    with pytest.raises(InvalidMoveError):
        engine.play_card(after_a, "B", wild_draw_four, chosen_color=Color.GREEN)

    after_b = engine.play_card(after_a, "B", draw_two_blue)

    assert after_b.draw_stack_total == 4
    assert after_b.pending_action is not None
    assert after_b.pending_action.player_id == "C"
    assert after_b.pending_action.draw_penalty == 4
    assert after_b.current_player_index == 2


def test_go_wild_stack_allows_wild_draw_four_when_no_draw_two() -> None:
    engine = UnoEngine(Random(0))

    draw_two_red = Card(color=Color.RED, type=CardType.DRAW_TWO, value=20)
    wild_draw_four = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)

    players = make_players()
    filler_a = Card(color=Color.GREEN, type=CardType.NUMBER, number=0, value=0)
    filler_b = Card(color=Color.YELLOW, type=CardType.NUMBER, number=4, value=4)
    players[0].hand = [draw_two_red, filler_a]
    players[1].hand = [wild_draw_four, filler_b]
    players[2].hand = [Card(color=Color.YELLOW, type=CardType.NUMBER, number=1, value=1)]
    players[3].hand = [Card(color=Color.GREEN, type=CardType.NUMBER, number=2, value=2)]

    state = GameState(
        players=players,
        draw_pile=[],
        discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=4, value=4)],
        current_player_index=0,
        play_direction=PlayDirection.CLOCKWISE,
        current_color=Color.RED,
        mode=GameMode.GO_WILD_2V2,
        team_map=team_map_for(players),
        team_scores=team_scores_for(players),
    )

    after_a = engine.play_card(state, "A", draw_two_red)
    moves = engine.get_valid_moves(after_a, "B")
    assert moves == [wild_draw_four]

    after_b = engine.play_card(after_a, "B", wild_draw_four, chosen_color=Color.GREEN)
    assert after_b.draw_stack_total == 6
    assert after_b.pending_action is not None
    assert after_b.pending_action.draw_penalty == 6
    assert after_b.current_color == Color.GREEN


def test_draw_two_not_allowed_on_wild_draw_four() -> None:
    engine = UnoEngine(Random(0))

    draw_two_red = Card(color=Color.RED, type=CardType.DRAW_TWO, value=20)
    wild_draw_four_b = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)
    draw_two_blue = Card(color=Color.BLUE, type=CardType.DRAW_TWO, value=20)
    wild_draw_four_c = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)

    players = make_players()
    filler_a = Card(color=Color.GREEN, type=CardType.NUMBER, number=0, value=0)
    filler_b = Card(color=Color.YELLOW, type=CardType.NUMBER, number=4, value=4)
    players[0].hand = [draw_two_red, filler_a]
    players[1].hand = [wild_draw_four_b, filler_b]
    players[2].hand = [draw_two_blue, wild_draw_four_c]
    players[3].hand = [Card(color=Color.GREEN, type=CardType.NUMBER, number=4, value=4)]

    state = GameState(
        players=players,
        draw_pile=[],
        discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=2, value=2)],
        current_player_index=0,
        play_direction=PlayDirection.CLOCKWISE,
        current_color=Color.RED,
        mode=GameMode.GO_WILD_2V2,
        team_map=team_map_for(players),
        team_scores=team_scores_for(players),
    )

    after_a = engine.play_card(state, "A", draw_two_red)
    after_b = engine.play_card(after_a, "B", wild_draw_four_b, chosen_color=Color.YELLOW)

    moves = engine.get_valid_moves(after_b, "C")
    assert moves == [wild_draw_four_c]

    with pytest.raises(InvalidMoveError):
        engine.play_card(after_b, "C", draw_two_blue)


def test_go_wild_stack_penalty_applies_when_chain_breaks() -> None:
    engine = UnoEngine(Random(0))

    draw_two_red = Card(color=Color.RED, type=CardType.DRAW_TWO, value=20)
    wild_draw_four_b = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)

    players = make_players()
    filler_a = Card(color=Color.GREEN, type=CardType.NUMBER, number=0, value=0)
    filler_b = Card(color=Color.YELLOW, type=CardType.NUMBER, number=4, value=4)
    players[0].hand = [draw_two_red, filler_a]
    players[1].hand = [wild_draw_four_b, filler_b]
    players[2].hand = [Card(color=Color.YELLOW, type=CardType.NUMBER, number=1, value=1)]
    players[3].hand = [Card(color=Color.GREEN, type=CardType.NUMBER, number=2, value=2)]

    draw_pile = [
        Card(color=Color.BLUE, type=CardType.NUMBER, number=n, value=n)
        for n in [1, 2, 3, 4, 5, 6, 7]
    ]

    state = GameState(
        players=players,
        draw_pile=list(draw_pile),
        discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
        current_player_index=0,
        play_direction=PlayDirection.CLOCKWISE,
        current_color=Color.RED,
        mode=GameMode.GO_WILD_2V2,
        team_map=team_map_for(players),
        team_scores=team_scores_for(players),
    )

    after_a = engine.play_card(state, "A", draw_two_red)
    after_b = engine.play_card(after_a, "B", wild_draw_four_b, chosen_color=Color.GREEN)

    assert after_b.pending_action is not None
    assert after_b.pending_action.draw_penalty == 6

    initial_len = len(after_b.players[2].hand)
    resolved = engine.draw_card(after_b, "C")

    assert len(resolved.players[2].hand) == initial_len + 6
    assert resolved.pending_action is None
    assert resolved.draw_stack_total == 0
    assert resolved.current_player_index == 3


def test_team_scoring_awards_points_to_both_teammates() -> None:
    engine = UnoEngine(Random(0))

    winning_card = Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)
    opponent_card_one = Card(color=Color.BLUE, type=CardType.DRAW_TWO, value=20)
    opponent_card_two = Card(color=Color.GREEN, type=CardType.NUMBER, number=9, value=9)

    players = make_players(team0_score=490, team1_score=200)
    players[0].hand = [winning_card]
    players[1].hand = [opponent_card_one]
    players[2].hand = [Card(color=Color.YELLOW, type=CardType.NUMBER, number=5, value=5)]
    players[3].hand = [opponent_card_two]

    state = GameState(
        players=players,
        draw_pile=[],
        discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=1, value=1)],
        current_player_index=0,
        play_direction=PlayDirection.CLOCKWISE,
        current_color=Color.RED,
        mode=GameMode.CLASSIC_2V2,
        team_map=team_map_for(players),
        team_scores=team_scores_for(players),
    )

    result = engine.play_card(state, "A", winning_card)

    assert result.round_over
    assert result.round_winner_id == "A"
    assert result.round_winning_team_ids == ("A", "C")

    added_points = opponent_card_one.value + opponent_card_two.value
    expected_score = 490 + added_points

    assert result.players[0].score == expected_score
    assert result.players[2].score == expected_score
    assert result.team_scores[0] == expected_score

    assert result.players[1].score == 200
    assert result.players[3].score == 200
    assert result.team_scores[1] == 200

    assert result.game_over
    assert result.game_winner_ids == ("A", "C")
