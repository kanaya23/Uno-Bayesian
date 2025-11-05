"""Comprehensive tests for UNO game mechanics and rules.

This test file ensures all game mechanics work correctly including:
- Card playability rules
- Special card effects (Skip, Reverse, Draw Two, etc.)
- Wild card handling
- Turn progression
- Color matching and number matching
- Winning conditions
"""

from __future__ import annotations

from random import Random

import pytest

from uno_engine import UnoEngine, InvalidMoveError, ColorSelectionError
from uno_engine.models import Card, CardType, Color, GameMode, GameState, PendingActionType, PlayDirection, Player


class TestCardPlayability:
    """Tests for determining which cards can be played."""

    def _create_simple_state(self, top_card: Card, current_color: Color) -> GameState:
        """Helper to create a test state."""
        players = [Player(player_id=f"P{i}", hand=[]) for i in range(4)]
        return GameState(
            players=players,
            draw_pile=[],
            discard_pile=[top_card],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=current_color,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )

    def test_matching_color_is_playable(self) -> None:
        """Cards matching current color should be playable."""
        engine = UnoEngine()
        top_card = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        state = self._create_simple_state(top_card, Color.RED)
        
        # Add matching color card
        matching = Card(color=Color.RED, type=CardType.NUMBER, number=7, value=7)
        state.players[0].hand.append(matching)
        
        valid = engine.get_valid_moves(state, "P0")
        assert matching in valid

    def test_matching_number_is_playable(self) -> None:
        """Cards matching number should be playable regardless of color."""
        engine = UnoEngine()
        top_card = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        state = self._create_simple_state(top_card, Color.RED)
        
        # Add matching number, different color
        matching = Card(color=Color.BLUE, type=CardType.NUMBER, number=5, value=5)
        state.players[0].hand.append(matching)
        
        valid = engine.get_valid_moves(state, "P0")
        assert matching in valid

    def test_non_matching_card_is_not_playable(self) -> None:
        """Cards with different color and number should not be playable."""
        engine = UnoEngine()
        top_card = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        state = self._create_simple_state(top_card, Color.RED)
        
        # Add non-matching card
        non_matching = Card(color=Color.BLUE, type=CardType.NUMBER, number=7, value=7)
        state.players[0].hand.append(non_matching)
        
        valid = engine.get_valid_moves(state, "P0")
        assert non_matching not in valid

    def test_wild_card_always_playable(self) -> None:
        """Wild cards should always be playable."""
        engine = UnoEngine()
        top_card = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        state = self._create_simple_state(top_card, Color.RED)
        
        wild = Card(color=Color.WILD, type=CardType.WILD, value=50)
        state.players[0].hand.append(wild)
        
        valid = engine.get_valid_moves(state, "P0")
        assert wild in valid

    def test_matching_action_type_is_playable(self) -> None:
        """Action cards of same type should match."""
        engine = UnoEngine()
        top_card = Card(color=Color.RED, type=CardType.SKIP, value=20)
        state = self._create_simple_state(top_card, Color.RED)
        
        matching_skip = Card(color=Color.BLUE, type=CardType.SKIP, value=20)
        state.players[0].hand.append(matching_skip)
        
        valid = engine.get_valid_moves(state, "P0")
        assert matching_skip in valid


class TestSkipCard:
    """Tests for Skip card functionality."""

    def test_skip_advances_two_players(self) -> None:
        """Skip should advance turn by 2 (skipping next player)."""
        engine = UnoEngine(Random(42))
        players = [
            Player(player_id="P0", hand=[
                Card(color=Color.RED, type=CardType.SKIP, value=20),
                Card(color=Color.RED, type=CardType.NUMBER, number=1, value=1),
            ]),
            Player(player_id="P1", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=2, value=2)]),
            Player(player_id="P2", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)]),
            Player(player_id="P3", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=4, value=4)]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        skip = state.players[0].hand[0]
        new_state = engine.play_card(state, "P0", skip)
        
        # Should skip P1, go to P2
        assert new_state.current_player_index == 2

    def test_skip_works_counter_clockwise(self) -> None:
        """Skip should work in counter-clockwise direction."""
        engine = UnoEngine(Random(42))
        players = [
            Player(player_id="P0", hand=[
                Card(color=Color.RED, type=CardType.SKIP, value=20),
                Card(color=Color.RED, type=CardType.NUMBER, number=1, value=1),
            ]),
            Player(player_id="P1", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=2, value=2)]),
            Player(player_id="P2", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)]),
            Player(player_id="P3", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=4, value=4)]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.COUNTER_CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        skip = state.players[0].hand[0]
        new_state = engine.play_card(state, "P0", skip)
        
        # Should skip P3, go to P2
        assert new_state.current_player_index == 2


class TestReverseCard:
    """Tests for Reverse card functionality."""

    def test_reverse_changes_direction(self) -> None:
        """Reverse should change play direction."""
        engine = UnoEngine(Random(42))
        players = [
            Player(player_id="P0", hand=[Card(color=Color.RED, type=CardType.REVERSE, value=20)]),
            Player(player_id="P1", hand=[]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        reverse = state.players[0].hand[0]
        new_state = engine.play_card(state, "P0", reverse)
        
        assert new_state.play_direction == PlayDirection.COUNTER_CLOCKWISE

    def test_reverse_changes_next_player(self) -> None:
        """After reverse, next player should be previous in original direction."""
        engine = UnoEngine(Random(42))
        players = [
            Player(player_id="P0", hand=[
                Card(color=Color.RED, type=CardType.REVERSE, value=20),
                Card(color=Color.RED, type=CardType.NUMBER, number=1, value=1),
            ]),
            Player(player_id="P1", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=2, value=2)]),
            Player(player_id="P2", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)]),
            Player(player_id="P3", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=4, value=4)]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        reverse = state.players[0].hand[0]
        new_state = engine.play_card(state, "P0", reverse)
        
        # Was clockwise, now counter-clockwise, so next is P3
        assert new_state.current_player_index == 3

    def test_double_reverse_returns_to_original(self) -> None:
        """Two reverses should return to original direction."""
        engine = UnoEngine(Random(42))
        players = [
            Player(player_id=f"P{i}", hand=[
                Card(color=Color.RED, type=CardType.REVERSE, value=20),
                Card(color=Color.RED, type=CardType.NUMBER, number=i, value=i),
            ])
            for i in range(4)
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        # First reverse
        reverse_card = [c for c in state.players[0].hand if c.type == CardType.REVERSE][0]
        state = engine.play_card(state, "P0", reverse_card)
        assert state.play_direction == PlayDirection.COUNTER_CLOCKWISE
        
        # Second reverse
        current_player = state.current_player()
        reverse_card2 = [c for c in current_player.hand if c.type == CardType.REVERSE][0]
        state = engine.play_card(state, current_player.player_id, reverse_card2)
        assert state.play_direction == PlayDirection.CLOCKWISE


class TestWildCards:
    """Tests for Wild and Wild Draw Four cards."""

    def test_wild_requires_color_choice(self) -> None:
        """Playing wild without color choice should raise error."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", hand=[Card(color=Color.WILD, type=CardType.WILD, value=50)]),
            Player(player_id="P1", hand=[]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        wild = state.players[0].hand[0]
        
        with pytest.raises(InvalidMoveError, match="color must be declared"):
            engine.play_card(state, "P0", wild)

    def test_wild_sets_chosen_color(self) -> None:
        """Playing wild should set the chosen color."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", hand=[Card(color=Color.WILD, type=CardType.WILD, value=50)]),
            Player(player_id="P1", hand=[]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        wild = state.players[0].hand[0]
        new_state = engine.play_card(state, "P0", wild, chosen_color=Color.BLUE)
        
        assert new_state.current_color == Color.BLUE

    def test_wild_cannot_choose_wild_color(self) -> None:
        """Cannot choose WILD as the color."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", hand=[Card(color=Color.WILD, type=CardType.WILD, value=50)]),
            Player(player_id="P1", hand=[]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        wild = state.players[0].hand[0]
        
        with pytest.raises(InvalidMoveError, match="standard color"):
            engine.play_card(state, "P0", wild, chosen_color=Color.WILD)


class TestTurnProgression:
    """Tests for turn order and progression."""

    def test_normal_turn_advances_by_one(self) -> None:
        """Normal card play should advance to next player."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", hand=[
                Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5),
                Card(color=Color.RED, type=CardType.NUMBER, number=6, value=6),
            ]),
            Player(player_id="P1", hand=[]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        card = state.players[0].hand[0]
        new_state = engine.play_card(state, "P0", card)
        
        assert new_state.current_player_index == 1
        assert not new_state.round_over  # Game should continue

    def test_turn_wraps_around(self) -> None:
        """Turn should wrap from P3 to P0."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", hand=[]),
            Player(player_id="P1", hand=[]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[
                Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5),
                Card(color=Color.RED, type=CardType.NUMBER, number=7, value=7),
            ]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=3,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        card = state.players[3].hand[0]
        new_state = engine.play_card(state, "P3", card)
        
        assert new_state.current_player_index == 0
        assert not new_state.round_over  # Game should continue

    def test_draw_advances_turn(self) -> None:
        """Drawing should advance to next player."""
        engine = UnoEngine()
        players = [Player(player_id=f"P{i}", hand=[]) for i in range(4)]
        
        state = GameState(
            players=players,
            draw_pile=[Card(color=Color.BLUE, type=CardType.NUMBER, number=7, value=7)],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        new_state = engine.draw_card(state, "P0")
        assert new_state.current_player_index == 1


class TestWinningConditions:
    """Tests for game and round winning."""

    def test_emptying_hand_wins_round(self) -> None:
        """Playing last card should win the round."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)]),
            Player(player_id="P1", hand=[Card(color=Color.BLUE, type=CardType.NUMBER, number=7, value=7)]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[Card(color=Color.GREEN, type=CardType.NUMBER, number=3, value=3)]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        card = state.players[0].hand[0]
        new_state = engine.play_card(state, "P0", card)
        
        assert new_state.round_over
        assert new_state.round_winner_id == "P0"

    def test_winning_awards_points(self) -> None:
        """Winning should award points from opponents' hands."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", score=0, hand=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)]),
            Player(player_id="P1", score=0, hand=[Card(color=Color.BLUE, type=CardType.NUMBER, number=7, value=7)]),
            Player(player_id="P2", score=0, hand=[]),
            Player(player_id="P3", score=0, hand=[Card(color=Color.GREEN, type=CardType.NUMBER, number=3, value=3)]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        card = state.players[0].hand[0]
        new_state = engine.play_card(state, "P0", card)
        
        # Team 0 (P0 + P2) should get points from Team 1 (P1 + P3)
        expected_points = 7 + 3  # P1's card + P3's card
        assert new_state.players[0].score == expected_points
        assert new_state.players[2].score == expected_points

    def test_reaching_500_wins_game(self) -> None:
        """Reaching 500 points should win the game."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", score=490, hand=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)]),
            Player(player_id="P1", score=200, hand=[Card(color=Color.BLUE, type=CardType.DRAW_TWO, value=20)]),
            Player(player_id="P2", score=490, hand=[]),
            Player(player_id="P3", score=200, hand=[Card(color=Color.GREEN, type=CardType.NUMBER, number=3, value=3)]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 490, 1: 200},
        )
        
        card = state.players[0].hand[0]
        new_state = engine.play_card(state, "P0", card)
        
        assert new_state.game_over
        assert new_state.game_winner_ids == ("P0", "P2")


class TestInvalidMoves:
    """Tests for moves that should be rejected."""

    def test_cannot_play_out_of_turn(self) -> None:
        """Non-current player cannot play."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", hand=[]),
            Player(player_id="P1", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        card = state.players[1].hand[0]
        
        with pytest.raises(InvalidMoveError):
            engine.play_card(state, "P1", card)

    def test_cannot_play_card_not_in_hand(self) -> None:
        """Cannot play a card not in player's hand."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)]),
            Player(player_id="P1", hand=[]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        # Try to play a card not in hand
        fake_card = Card(color=Color.BLUE, type=CardType.NUMBER, number=7, value=7)
        
        with pytest.raises(InvalidMoveError):
            engine.play_card(state, "P0", fake_card)

    def test_cannot_play_non_matching_card(self) -> None:
        """Cannot play card that doesn't match color or number."""
        engine = UnoEngine()
        players = [
            Player(player_id="P0", hand=[Card(color=Color.BLUE, type=CardType.NUMBER, number=7, value=7)]),
            Player(player_id="P1", hand=[]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[]),
        ]
        
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        card = state.players[0].hand[0]
        
        with pytest.raises(InvalidMoveError):
            engine.play_card(state, "P0", card)
