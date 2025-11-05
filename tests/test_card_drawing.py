"""Regression tests for card drawing to prevent the 3-card bug from returning.

This test file specifically focuses on the bug where drawing a card was giving
3 cards instead of 1. These tests ensure that:
1. Initial dealing gives exactly 7 cards per player
2. Drawing a single card adds exactly 1 card
3. Card counts remain consistent throughout the game
"""

from __future__ import annotations

from random import Random

import pytest

from uno_engine import UnoEngine, InvalidMoveError
from uno_engine.models import Card, CardType, Color, GameMode, GameState, PendingAction, PendingActionType, PlayDirection, Player


class TestInitialCardDealing:
    """Tests for initial card dealing to each player."""

    def test_each_player_gets_exactly_7_cards(self) -> None:
        """REGRESSION: Each player must get exactly 7 cards, not 21."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}") for i in range(4)]
        state = engine.init_game(players)
        
        for player in state.players:
            assert len(player.hand) == 7, f"{player.player_id} should have 7 cards, has {len(player.hand)}"

    def test_initial_dealing_with_different_seeds(self) -> None:
        """Initial dealing should give 7 cards before any starting card effects."""
        for seed in [0, 42, 123, 999, 54321]:
            engine = UnoEngine(Random(seed))
            players = [Player(player_id=f"P{i}") for i in range(4)]
            state = engine.init_game(players, mode=GameMode.CLASSIC_2V2)
            
            # Each player should start with at least 7 cards
            # (some may have more if starting card was Draw Two)
            for player in state.players:
                assert len(player.hand) >= 7, f"Player {player.player_id} has {len(player.hand)} cards with seed {seed}"

    def test_initial_dealing_classic_mode(self) -> None:
        """Classic mode should deal 7 cards per player."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}") for i in range(4)]
        state = engine.init_game(players, mode=GameMode.CLASSIC_2V2)
        
        for player in state.players:
            assert len(player.hand) == 7

    def test_initial_dealing_go_wild_mode(self) -> None:
        """Go Wild mode should deal at least 7 cards per player."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}") for i in range(4)]
        state = engine.init_game(players, mode=GameMode.GO_WILD_2V2)
        
        # Each player should start with at least 7 cards
        # (some may have more if starting card was Draw Two)
        for player in state.players:
            assert len(player.hand) >= 7

    def test_total_cards_after_initial_deal_classic(self) -> None:
        """Total cards should be 108 after initial deal (Classic)."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}") for i in range(4)]
        state = engine.init_game(players, mode=GameMode.CLASSIC_2V2)
        
        total = sum(len(p.hand) for p in state.players) + len(state.draw_pile) + len(state.discard_pile)
        assert total == 108

    def test_total_cards_after_initial_deal_go_wild(self) -> None:
        """Total cards should be 224 after initial deal (Go Wild)."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}") for i in range(4)]
        state = engine.init_game(players, mode=GameMode.GO_WILD_2V2)
        
        total = sum(len(p.hand) for p in state.players) + len(state.draw_pile) + len(state.discard_pile)
        assert total == 224

    def test_no_player_gets_duplicate_initial_hands(self) -> None:
        """Each player should get different cards initially."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}") for i in range(4)]
        state = engine.init_game(players)
        
        # Collect all cards from all players
        all_cards = []
        for player in state.players:
            all_cards.extend(player.hand)
        
        # In classic UNO, we have duplicates in the deck, but each player
        # should not get the exact same set of 7 cards
        hands_as_sorted_strings = [
            str(sorted([str(c) for c in p.hand]))
            for p in state.players
        ]
        # At least some hands should be different
        assert len(set(hands_as_sorted_strings)) > 1


class TestSingleCardDrawing:
    """Tests for drawing a single card during gameplay."""

    def _create_test_state(self, current_player_idx: int = 0) -> GameState:
        """Helper to create a simple game state for testing."""
        players = [
            Player(player_id=f"P{i}", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=1, value=1)])
            for i in range(4)
        ]
        return GameState(
            players=players,
            draw_pile=[Card(color=Color.BLUE, type=CardType.NUMBER, number=i % 10, value=i % 10) for i in range(20)],
            discard_pile=[Card(color=Color.GREEN, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=current_player_idx,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.GREEN,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )

    def test_draw_card_adds_exactly_one_card(self) -> None:
        """REGRESSION: Drawing must add exactly 1 card, not 3."""
        engine = UnoEngine(Random(42))
        state = self._create_test_state()
        
        before = len(state.players[0].hand)
        new_state = engine.draw_card(state, "P0")
        after = len(new_state.players[0].hand)
        
        assert after - before == 1, f"Should draw 1 card, drew {after - before}"

    def test_draw_card_multiple_times(self) -> None:
        """Drawing multiple times should add 1 card each time."""
        engine = UnoEngine(Random(42))
        state = self._create_test_state()
        
        initial_count = len(state.players[0].hand)
        
        # Draw 5 times
        for i in range(5):
            state = engine.draw_card(state, "P0")
            state.current_player_index = 0  # Reset for next draw
        
        final_count = len(state.players[0].hand)
        assert final_count == initial_count + 5

    def test_draw_card_reduces_draw_pile(self) -> None:
        """Drawing should remove exactly 1 card from draw pile."""
        engine = UnoEngine(Random(42))
        state = self._create_test_state()
        
        before = len(state.draw_pile)
        new_state = engine.draw_card(state, "P0")
        after = len(new_state.draw_pile)
        
        assert before - after == 1

    def test_draw_card_conserves_total_cards(self) -> None:
        """Total cards should remain constant after drawing."""
        engine = UnoEngine(Random(42))
        state = self._create_test_state()
        
        before_total = sum(len(p.hand) for p in state.players) + len(state.draw_pile) + len(state.discard_pile)
        new_state = engine.draw_card(state, "P0")
        after_total = sum(len(p.hand) for p in new_state.players) + len(new_state.draw_pile) + len(new_state.discard_pile)
        
        assert before_total == after_total

    def test_only_current_player_can_draw(self) -> None:
        """Only the current player should be able to draw."""
        engine = UnoEngine(Random(42))
        state = self._create_test_state(current_player_idx=0)
        
        # P0 is current player - should work
        new_state = engine.draw_card(state, "P0")
        assert len(new_state.players[0].hand) > len(state.players[0].hand)
        
        # P1 is not current player - should fail
        with pytest.raises(InvalidMoveError):
            engine.draw_card(state, "P1")

    def test_draw_from_different_positions(self) -> None:
        """Drawing should work for any player when it's their turn."""
        engine = UnoEngine(Random(42))
        
        for player_idx in range(4):
            state = self._create_test_state(current_player_idx=player_idx)
            player_id = f"P{player_idx}"
            
            before = len(state.players[player_idx].hand)
            new_state = engine.draw_card(state, player_id)
            after = len(new_state.players[player_idx].hand)
            
            assert after == before + 1


class TestDrawStackMechanic:
    """Tests for drawing multiple cards due to Draw Two/Wild Draw Four stacking."""

    def test_draw_two_penalty_classic_mode(self) -> None:
        """In Classic mode, Draw Two should force drawing exactly 2 cards."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}", hand=[]) for i in range(4)]
        draw_pile = [Card(color=Color.BLUE, type=CardType.NUMBER, number=i % 10, value=i % 10) for i in range(20)]
        
        state = GameState(
            players=players,
            draw_pile=draw_pile,
            discard_pile=[Card(color=Color.RED, type=CardType.DRAW_TWO, value=20)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        # Apply initial card effect manually
        engine._apply_initial_card_effect(state)
        
        # Player 0 should have drawn exactly 2 cards
        assert len(state.players[0].hand) == 2

    def test_wild_draw_four_penalty_classic_mode(self) -> None:
        """Wild Draw Four should force drawing exactly 4 cards."""
        engine = UnoEngine(Random(42))
        players = [
            Player(player_id="P0", hand=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)]),
            Player(player_id="P1", hand=[]),
            Player(player_id="P2", hand=[]),
            Player(player_id="P3", hand=[]),
        ]
        draw_pile = [Card(color=Color.BLUE, type=CardType.NUMBER, number=i % 10, value=i % 10) for i in range(20)]
        
        state = GameState(
            players=players,
            draw_pile=draw_pile,
            discard_pile=[Card(color=Color.GREEN, type=CardType.NUMBER, number=3, value=3)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.GREEN,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        wild_draw_four = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)
        state.players[0].hand.append(wild_draw_four)
        
        # Play the Wild Draw Four
        new_state = engine.play_card(state, "P0", wild_draw_four, chosen_color=Color.RED)
        
        # Next player (P1) should have drawn exactly 4 cards
        assert len(new_state.players[1].hand) == 4

    def test_draw_stack_resolution_go_wild(self) -> None:
        """In Go Wild, drawing from stack should add exact penalty amount."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}", hand=[]) for i in range(4)]
        draw_pile = [Card(color=Color.BLUE, type=CardType.NUMBER, number=i % 10, value=i % 10) for i in range(30)]
        
        state = GameState(
            players=players,
            draw_pile=draw_pile,
            discard_pile=[Card(color=Color.RED, type=CardType.DRAW_TWO, value=20)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.GO_WILD_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
            pending_action=PendingAction(
                player_id="P0",
                type=PendingActionType.DRAW_STACK,
                draw_penalty=6
            ),
        )
        
        # Player chooses to draw (accept penalty)
        before = len(state.players[0].hand)
        new_state = engine.draw_card(state, "P0")
        after = len(new_state.players[0].hand)
        
        assert after - before == 6, f"Should draw exactly 6 cards from stack, drew {after - before}"


class TestCardConservation:
    """Tests to ensure cards are never duplicated or lost."""

    def test_no_cards_lost_during_full_game(self) -> None:
        """Play several rounds and ensure card count stays constant."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}") for i in range(4)]
        state = engine.init_game(players, mode=GameMode.CLASSIC_2V2)
        
        initial_total = sum(len(p.hand) for p in state.players) + len(state.draw_pile) + len(state.discard_pile)
        assert initial_total == 108
        
        # Perform various actions
        for _ in range(10):
            if state.round_over:
                break
            
            current_player_id = state.current_player().player_id
            valid_moves = engine.get_valid_moves(state, current_player_id)
            
            if valid_moves and Random(42).random() > 0.3:
                # Play a card
                card = valid_moves[0]
                try:
                    if card.type in {CardType.WILD, CardType.WILD_DRAW_FOUR}:
                        state = engine.play_card(state, current_player_id, card, Color.RED)
                    else:
                        state = engine.play_card(state, current_player_id, card)
                except InvalidMoveError:
                    pass
            else:
                # Draw a card
                try:
                    state = engine.draw_card(state, current_player_id)
                except InvalidMoveError:
                    pass
            
            # Check total after each action
            current_total = sum(len(p.hand) for p in state.players) + len(state.draw_pile) + len(state.discard_pile)
            assert current_total == initial_total, f"Cards lost/duplicated: expected {initial_total}, got {current_total}"

    def test_draw_pile_recycling_conserves_cards(self) -> None:
        """When draw pile is empty and recycled, no cards should be lost."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}", hand=[]) for i in range(4)]
        
        # Start with only 3 cards in draw pile
        draw_pile = [Card(color=Color.BLUE, type=CardType.NUMBER, number=i, value=i) for i in range(3)]
        discard_pile = [Card(color=Color.RED, type=CardType.NUMBER, number=i, value=i) for i in range(10)]
        
        state = GameState(
            players=players,
            draw_pile=draw_pile,
            discard_pile=discard_pile,
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        initial_total = 3 + 10  # draw + discard
        
        # Draw all cards from draw pile, forcing recycle
        for i in range(5):
            state = engine.draw_card(state, "P0")
            state.current_player_index = 0  # Reset for next draw
        
        final_total = sum(len(p.hand) for p in state.players) + len(state.draw_pile) + len(state.discard_pile)
        assert final_total == initial_total


class TestEdgeCases:
    """Edge cases that could trigger the bug."""

    def test_draw_with_one_card_left_in_pile(self) -> None:
        """Drawing when only 1 card remains should work correctly."""
        engine = UnoEngine(Random(42))
        players = [Player(player_id=f"P{i}", hand=[]) for i in range(4)]
        
        # Create a varied discard pile to avoid winning conditions
        discard_cards = [Card(color=Color.RED, type=CardType.NUMBER, number=i % 10, value=i % 10) for i in range(10)]
        
        state = GameState(
            players=players,
            draw_pile=[Card(color=Color.BLUE, type=CardType.NUMBER, number=7, value=7)],
            discard_pile=discard_cards,
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        new_state = engine.draw_card(state, "P0")
        
        assert len(new_state.players[0].hand) == 1
        # Draw pile should be recycled from discard (at least the top card stays)
        assert len(new_state.draw_pile) + len(new_state.discard_pile) >= 1

    def test_draw_card_after_skip(self) -> None:
        """Drawing after a skip should still add only 1 card."""
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
            draw_pile=[Card(color=Color.BLUE, type=CardType.NUMBER, number=i, value=i) for i in range(10)],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        skip_card = state.players[0].hand[0]
        state = engine.play_card(state, "P0", skip_card)
        
        # P2 should be current now (P1 was skipped)
        assert state.current_player().player_id == "P2"
        
        # P2 draws
        before = len(state.players[2].hand)
        state = engine.draw_card(state, "P2")
        after = len(state.players[2].hand)
        
        assert after - before == 1

    def test_draw_card_after_reverse(self) -> None:
        """Drawing after a reverse should still add only 1 card."""
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
            draw_pile=[Card(color=Color.BLUE, type=CardType.NUMBER, number=i, value=i) for i in range(10)],
            discard_pile=[Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)],
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=Color.RED,
            mode=GameMode.CLASSIC_2V2,
            team_map={"P0": 0, "P1": 1, "P2": 0, "P3": 1},
            team_scores={0: 0, 1: 0},
        )
        
        reverse_card = state.players[0].hand[0]
        state = engine.play_card(state, "P0", reverse_card)
        
        # Direction should be counter-clockwise, P3 is next
        assert state.play_direction == PlayDirection.COUNTER_CLOCKWISE
        
        # P3 draws
        before = len(state.players[3].hand)
        state = engine.draw_card(state, "P3")
        after = len(state.players[3].hand)
        
        assert after - before == 1
