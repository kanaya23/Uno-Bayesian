"""Comprehensive tests for UNO game models and data structures."""

from __future__ import annotations

import copy

import pytest

from uno_engine.models import (
    Card,
    CardType,
    Color,
    GameMode,
    GameState,
    PendingAction,
    PendingActionType,
    PlayDirection,
    Player,
)


class TestCard:
    """Tests for Card model."""

    def test_card_creation_number_card(self) -> None:
        """Number cards should require a number value."""
        card = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        assert card.number == 5
        assert card.value == 5
        assert card.color == Color.RED
        assert card.type == CardType.NUMBER

    def test_card_creation_action_card(self) -> None:
        """Action cards should not have a number value."""
        card = Card(color=Color.BLUE, type=CardType.SKIP, value=20)
        assert card.number is None
        assert card.value == 20

    def test_card_number_required_for_number_type(self) -> None:
        """NUMBER type must have a number."""
        with pytest.raises(ValueError, match="Number cards must include"):
            Card(color=Color.RED, type=CardType.NUMBER, value=5, number=None)

    def test_card_number_forbidden_for_non_number_type(self) -> None:
        """Non-NUMBER types cannot have a number."""
        with pytest.raises(ValueError, match="Only number cards"):
            Card(color=Color.RED, type=CardType.SKIP, value=20, number=5)

    def test_card_equality(self) -> None:
        """Cards with same attributes should be equal."""
        card1 = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        card2 = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        assert card1 == card2

    def test_card_inequality_different_number(self) -> None:
        """Cards with different numbers should not be equal."""
        card1 = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        card2 = Card(color=Color.RED, type=CardType.NUMBER, number=6, value=6)
        assert card1 != card2

    def test_card_inequality_different_color(self) -> None:
        """Cards with different colors should not be equal."""
        card1 = Card(color=Color.RED, type=CardType.SKIP, value=20)
        card2 = Card(color=Color.BLUE, type=CardType.SKIP, value=20)
        assert card1 != card2

    def test_card_wild_type(self) -> None:
        """Wild cards should use WILD color."""
        card = Card(color=Color.WILD, type=CardType.WILD, value=50)
        assert card.color == Color.WILD
        assert card.number is None

    def test_card_wild_draw_four(self) -> None:
        """Wild Draw Four cards should be properly constructed."""
        card = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)
        assert card.color == Color.WILD
        assert card.type == CardType.WILD_DRAW_FOUR
        assert card.number is None
        assert card.value == 50


class TestPlayer:
    """Tests for Player model."""

    def test_player_creation_default(self) -> None:
        """Player should have default empty hand and zero score."""
        player = Player(player_id="P0")
        assert player.player_id == "P0"
        assert len(player.hand) == 0
        assert player.score == 0

    def test_player_creation_with_score(self) -> None:
        """Player can be created with initial score."""
        player = Player(player_id="P1", score=100)
        assert player.score == 100

    def test_player_creation_with_hand(self) -> None:
        """Player can be created with initial hand."""
        cards = [Card(color=Color.RED, type=CardType.NUMBER, number=i, value=i) for i in range(3)]
        player = Player(player_id="P2", hand=cards)
        assert len(player.hand) == 3
        assert player.hand == cards

    def test_player_hand_is_mutable(self) -> None:
        """Player hand should be modifiable."""
        player = Player(player_id="P0")
        card = Card(color=Color.BLUE, type=CardType.SKIP, value=20)
        player.hand.append(card)
        assert len(player.hand) == 1
        assert player.hand[0] == card

    def test_player_different_instances_are_independent(self) -> None:
        """Different player instances should have independent hands."""
        player1 = Player(player_id="P0")
        player2 = Player(player_id="P1")
        
        card = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        player1.hand.append(card)
        
        assert len(player1.hand) == 1
        assert len(player2.hand) == 0


class TestPendingAction:
    """Tests for PendingAction model."""

    def test_pending_action_color_choice(self) -> None:
        """Color choice pending action."""
        action = PendingAction(player_id="P0", type=PendingActionType.COLOR_CHOICE)
        assert action.player_id == "P0"
        assert action.type == PendingActionType.COLOR_CHOICE
        assert len(action.allowed_cards) == 0
        assert action.draw_penalty == 0

    def test_pending_action_draw_stack(self) -> None:
        """Draw stack pending action with penalty."""
        action = PendingAction(
            player_id="P1",
            type=PendingActionType.DRAW_STACK,
            draw_penalty=4
        )
        assert action.player_id == "P1"
        assert action.type == PendingActionType.DRAW_STACK
        assert action.draw_penalty == 4

    def test_pending_action_with_allowed_cards(self) -> None:
        """Pending action with allowed cards tuple."""
        cards = (
            Card(color=Color.RED, type=CardType.DRAW_TWO, value=20),
            Card(color=Color.BLUE, type=CardType.DRAW_TWO, value=20),
        )
        action = PendingAction(
            player_id="P2",
            type=PendingActionType.DRAW_STACK,
            allowed_cards=cards
        )
        assert len(action.allowed_cards) == 2
        assert action.allowed_cards == cards


class TestGameState:
    """Tests for GameState model and its methods."""

    def _create_basic_state(self) -> GameState:
        """Helper to create a basic game state."""
        players = [Player(player_id=f"P{i}") for i in range(4)]
        return GameState(
            players=players,
            draw_pile=[],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )

    def test_game_state_creation(self) -> None:
        """Game state should be created with all required fields."""
        state = self._create_basic_state()
        assert len(state.players) == 4
        assert state.current_player_index == 0
        assert state.play_direction == PlayDirection.CLOCKWISE
        assert not state.round_over
        assert not state.game_over

    def test_current_player_accessor(self) -> None:
        """current_player() should return the active player."""
        state = self._create_basic_state()
        current = state.current_player()
        assert current == state.players[0]
        assert current.player_id == "P0"

    def test_current_player_with_different_index(self) -> None:
        """current_player() should respect current_player_index."""
        state = self._create_basic_state()
        state.current_player_index = 2
        current = state.current_player()
        assert current == state.players[2]
        assert current.player_id == "P2"

    def test_player_ids_returns_ordered_ids(self) -> None:
        """player_ids() should return tuple of player IDs."""
        state = self._create_basic_state()
        ids = state.player_ids()
        assert ids == ("P0", "P1", "P2", "P3")
        assert isinstance(ids, tuple)

    def test_team_index_for_valid_player(self) -> None:
        """team_index_for() should return correct team."""
        state = self._create_basic_state()
        assert state.team_index_for("P0") == 0
        assert state.team_index_for("P1") == 1
        assert state.team_index_for("P2") == 0
        assert state.team_index_for("P3") == 1

    def test_team_index_for_invalid_player(self) -> None:
        """team_index_for() should raise KeyError for unknown player."""
        state = self._create_basic_state()
        with pytest.raises(KeyError, match="Unknown player"):
            state.team_index_for("P99")

    def test_teammate_id_returns_correct_teammate(self) -> None:
        """teammate_id() should return the player's teammate."""
        state = self._create_basic_state()
        assert state.teammate_id("P0") == "P2"
        assert state.teammate_id("P1") == "P3"
        assert state.teammate_id("P2") == "P0"
        assert state.teammate_id("P3") == "P1"

    def test_teammate_id_invalid_player(self) -> None:
        """teammate_id() should raise KeyError for unknown player."""
        state = self._create_basic_state()
        with pytest.raises(KeyError):
            state.teammate_id("P99")

    def test_visible_hands_for_returns_self_and_teammate(self) -> None:
        """visible_hands_for() should return hands for player and teammate."""
        state = self._create_basic_state()
        
        # Add some cards to verify
        state.players[0].hand = [Card(color=Color.RED, type=CardType.NUMBER, number=1, value=1)]
        state.players[2].hand = [Card(color=Color.BLUE, type=CardType.NUMBER, number=2, value=2)]
        
        visible = state.visible_hands_for("P0")
        
        assert "P0" in visible
        assert "P2" in visible  # Teammate
        assert "P1" not in visible
        assert "P3" not in visible
        assert len(visible["P0"]) == 1
        assert len(visible["P2"]) == 1

    def test_visible_hands_returns_tuples(self) -> None:
        """visible_hands_for() should return immutable tuples."""
        state = self._create_basic_state()
        visible = state.visible_hands_for("P0")
        
        assert isinstance(visible["P0"], tuple)
        assert isinstance(visible["P2"], tuple)

    def test_fast_clone_creates_independent_copy(self) -> None:
        """fast_clone() should create independent game state."""
        state = self._create_basic_state()
        state.players[0].hand = [Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)]
        state.draw_pile = [Card(color=Color.BLUE, type=CardType.NUMBER, number=3, value=3)]
        
        cloned = state.fast_clone()
        
        # Modify original
        state.players[0].hand.append(Card(color=Color.GREEN, type=CardType.SKIP, value=20))
        state.draw_pile.append(Card(color=Color.YELLOW, type=CardType.REVERSE, value=20))
        state.current_player_index = 2
        
        # Clone should be unaffected
        assert len(cloned.players[0].hand) == 1
        assert len(cloned.draw_pile) == 1
        assert cloned.current_player_index == 0

    def test_fast_clone_preserves_all_fields(self) -> None:
        """fast_clone() should preserve all game state fields."""
        state = self._create_basic_state()
        state.round_over = True
        state.round_winner_id = "P0"
        state.game_over = True
        state.game_winner_ids = ("P0", "P2")
        state.draw_stack_total = 6
        state.pending_action = PendingAction(
            player_id="P1",
            type=PendingActionType.DRAW_STACK,
            draw_penalty=4
        )
        
        cloned = state.fast_clone()
        
        assert cloned.round_over == state.round_over
        assert cloned.round_winner_id == state.round_winner_id
        assert cloned.game_over == state.game_over
        assert cloned.game_winner_ids == state.game_winner_ids
        assert cloned.draw_stack_total == state.draw_stack_total
        assert cloned.pending_action is not None
        assert cloned.pending_action.player_id == "P1"
        assert cloned.pending_action.draw_penalty == 4

    def test_fast_clone_player_independence(self) -> None:
        """fast_clone() should create independent player objects."""
        state = self._create_basic_state()
        cloned = state.fast_clone()
        
        # Modify cloned player
        cloned.players[0].score = 100
        cloned.players[0].hand.append(Card(color=Color.RED, type=CardType.NUMBER, number=1, value=1))
        
        # Original should be unaffected
        assert state.players[0].score == 0
        assert len(state.players[0].hand) == 0

    def test_fast_clone_with_none_pending_action(self) -> None:
        """fast_clone() should handle None pending action."""
        state = self._create_basic_state()
        state.pending_action = None
        
        cloned = state.fast_clone()
        
        assert cloned.pending_action is None

    def test_game_state_default_values(self) -> None:
        """Game state should have sensible defaults."""
        players = [Player(player_id=f"P{i}") for i in range(4)]
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=None,
        )
        
        assert state.pending_action is None
        assert not state.round_over
        assert state.round_winner_id is None
        assert not state.game_over
        assert state.game_winner_ids == ()
        assert state.mode == GameMode.CLASSIC_2V2
        assert state.draw_stack_total == 0


class TestEnumerations:
    """Tests for enum types."""

    def test_color_enum_values(self) -> None:
        """Color enum should have expected values."""
        assert Color.RED.value == "RED"
        assert Color.YELLOW.value == "YELLOW"
        assert Color.GREEN.value == "GREEN"
        assert Color.BLUE.value == "BLUE"
        assert Color.WILD.value == "WILD"

    def test_card_type_enum_values(self) -> None:
        """CardType enum should have all expected types."""
        types = {ct.value for ct in CardType}
        expected = {
            "NUMBER", "SKIP", "REVERSE", "DRAW_TWO",
            "WILD", "WILD_DRAW_FOUR", "DISCARD_ALL"
        }
        assert types == expected

    def test_game_mode_enum_values(self) -> None:
        """GameMode enum should have both modes."""
        assert GameMode.CLASSIC_2V2.value == "CLASSIC_2V2"
        assert GameMode.GO_WILD_2V2.value == "GO_WILD_2V2"

    def test_play_direction_enum(self) -> None:
        """PlayDirection should have both directions."""
        assert PlayDirection.CLOCKWISE
        assert PlayDirection.COUNTER_CLOCKWISE

    def test_pending_action_type_enum(self) -> None:
        """PendingActionType should have all action types."""
        types = list(PendingActionType)
        assert PendingActionType.DRAWN_CARD_PLAY_WINDOW in types
        assert PendingActionType.COLOR_CHOICE in types
        assert PendingActionType.DRAW_STACK in types


class TestModelEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_player_hand_can_be_large(self) -> None:
        """Player should be able to hold many cards."""
        player = Player(player_id="P0")
        cards = [Card(color=Color.RED, type=CardType.NUMBER, number=i % 10, value=i % 10) for i in range(100)]
        player.hand = cards
        assert len(player.hand) == 100

    def test_game_state_with_many_cards_in_piles(self) -> None:
        """Game state should handle large card piles."""
        players = [Player(player_id=f"P{i}") for i in range(4)]
        large_pile = [Card(color=Color.RED, type=CardType.NUMBER, number=i % 10, value=i % 10) for i in range(200)]
        
        state = GameState(
            players=players,
            draw_pile=large_pile[:100],
            discard_pile=large_pile[100:200],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
        )
        
        assert len(state.draw_pile) == 100
        assert len(state.discard_pile) == 100

    def test_game_state_team_scores_can_be_large(self) -> None:
        """Team scores can reach large values."""
        players = [Player(player_id=f"P{i}", score=10000) for i in range(4)]
        state = GameState(
            players=players,
            draw_pile=[],
            discard_pile=[],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=None,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 10000, 1: 10000},
        )
        
        assert state.team_scores[0] == 10000
        assert state.team_scores[1] == 10000

    def test_pending_action_with_many_allowed_cards(self) -> None:
        """Pending action can have many allowed cards."""
        cards = tuple(
            Card(color=Color.RED, type=CardType.NUMBER, number=i % 10, value=i % 10)
            for i in range(20)
        )
        action = PendingAction(
            player_id="P0",
            type=PendingActionType.DRAW_STACK,
            allowed_cards=cards
        )
        assert len(action.allowed_cards) == 20
