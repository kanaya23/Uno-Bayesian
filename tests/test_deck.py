"""Comprehensive tests for deck building and manipulation."""

from __future__ import annotations

import random
from collections import Counter
from typing import List

import pytest

from uno_engine.deck import build_deck_for_mode, build_go_wild_deck, build_standard_deck, bury_card, shuffle_in_place
from uno_engine.models import Card, CardType, Color, GameMode


class TestStandardDeck:
    """Tests for standard UNO deck construction."""

    def test_standard_deck_has_108_cards(self) -> None:
        """Verify standard deck contains exactly 108 cards."""
        deck = build_standard_deck()
        assert len(deck) == 108

    def test_standard_deck_has_correct_zeros(self) -> None:
        """Each color should have exactly one zero."""
        deck = build_standard_deck()
        zeros = [card for card in deck if card.type == CardType.NUMBER and card.number == 0]
        
        assert len(zeros) == 4  # One per color
        colors = {card.color for card in zeros}
        assert colors == {Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE}

    def test_standard_deck_has_correct_number_distribution(self) -> None:
        """Numbers 1-9 should appear twice per color."""
        deck = build_standard_deck()
        
        for color in [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]:
            for number in range(1, 10):
                matching = [
                    card for card in deck
                    if card.type == CardType.NUMBER
                    and card.color == color
                    and card.number == number
                ]
                assert len(matching) == 2, f"{color} {number} should appear twice"

    def test_standard_deck_has_correct_action_cards(self) -> None:
        """Each color should have 2 Skip, 2 Reverse, 2 Draw Two."""
        deck = build_standard_deck()
        
        for color in [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]:
            skips = [c for c in deck if c.type == CardType.SKIP and c.color == color]
            reverses = [c for c in deck if c.type == CardType.REVERSE and c.color == color]
            draw_twos = [c for c in deck if c.type == CardType.DRAW_TWO and c.color == color]
            
            assert len(skips) == 2, f"{color} should have 2 Skip cards"
            assert len(reverses) == 2, f"{color} should have 2 Reverse cards"
            assert len(draw_twos) == 2, f"{color} should have 2 Draw Two cards"

    def test_standard_deck_has_correct_wild_cards(self) -> None:
        """Deck should have 4 Wild and 4 Wild Draw Four."""
        deck = build_standard_deck()
        
        wilds = [c for c in deck if c.type == CardType.WILD]
        wild_draw_fours = [c for c in deck if c.type == CardType.WILD_DRAW_FOUR]
        
        assert len(wilds) == 4
        assert len(wild_draw_fours) == 4
        
        # All wild cards should have WILD color
        for card in wilds + wild_draw_fours:
            assert card.color == Color.WILD

    def test_standard_deck_has_no_discard_all_cards(self) -> None:
        """Standard deck should not contain Discard All cards."""
        deck = build_standard_deck()
        discard_all_cards = [c for c in deck if c.type == CardType.DISCARD_ALL]
        assert len(discard_all_cards) == 0

    def test_standard_deck_card_values_are_correct(self) -> None:
        """Verify point values are assigned correctly."""
        deck = build_standard_deck()
        
        for card in deck:
            if card.type == CardType.NUMBER:
                assert card.value == card.number
            elif card.type in {CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO}:
                assert card.value == 20
            elif card.type in {CardType.WILD, CardType.WILD_DRAW_FOUR}:
                assert card.value == 50


class TestGoWildDeck:
    """Tests for Go Wild 2v2 deck construction."""

    def test_go_wild_deck_has_224_cards(self) -> None:
        """Go Wild deck should have 224 cards (2 standard + 8 Discard All)."""
        deck = build_go_wild_deck()
        assert len(deck) == 224

    def test_go_wild_deck_has_double_standard_cards(self) -> None:
        """Go Wild should contain two complete standard decks."""
        deck = build_go_wild_deck()
        
        # Count non-Discard All cards
        standard_cards = [c for c in deck if c.type != CardType.DISCARD_ALL]
        assert len(standard_cards) == 216  # 2 * 108

    def test_go_wild_deck_has_discard_all_cards(self) -> None:
        """Go Wild should have 2 Discard All per color."""
        deck = build_go_wild_deck()
        
        for color in [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]:
            discard_alls = [
                c for c in deck
                if c.type == CardType.DISCARD_ALL and c.color == color
            ]
            assert len(discard_alls) == 2, f"{color} should have 2 Discard All cards"

    def test_go_wild_deck_discard_all_values(self) -> None:
        """Discard All cards should have value of 40."""
        deck = build_go_wild_deck()
        discard_alls = [c for c in deck if c.type == CardType.DISCARD_ALL]
        
        for card in discard_alls:
            assert card.value == 40

    def test_go_wild_deck_number_distribution(self) -> None:
        """Numbers should appear in correct quantities (double of standard)."""
        deck = build_go_wild_deck()
        
        # Zeros: 2 per color (1 * 2 decks)
        zeros = [c for c in deck if c.type == CardType.NUMBER and c.number == 0]
        assert len(zeros) == 8  # 4 colors * 2 decks
        
        # Numbers 1-9: 4 per color (2 * 2 decks)
        for number in range(1, 10):
            number_cards = [c for c in deck if c.type == CardType.NUMBER and c.number == number]
            assert len(number_cards) == 16  # 4 colors * 2 per standard * 2 decks


class TestDeckBuilding:
    """Tests for the deck building factory function."""

    def test_build_deck_for_classic_mode(self) -> None:
        """Classic mode should return standard deck."""
        deck = build_deck_for_mode(GameMode.CLASSIC_2V2)
        assert len(deck) == 108

    def test_build_deck_for_go_wild_mode(self) -> None:
        """Go Wild mode should return extended deck."""
        deck = build_deck_for_mode(GameMode.GO_WILD_2V2)
        assert len(deck) == 224

    def test_build_deck_returns_new_instance(self) -> None:
        """Each call should return a fresh deck instance."""
        deck1 = build_deck_for_mode(GameMode.CLASSIC_2V2)
        deck2 = build_deck_for_mode(GameMode.CLASSIC_2V2)
        
        # They should be equal in content but different objects
        assert deck1 is not deck2
        assert len(deck1) == len(deck2)


class TestShuffling:
    """Tests for deck shuffling operations."""

    def test_shuffle_in_place_maintains_card_count(self) -> None:
        """Shuffling should not change number of cards."""
        deck = build_standard_deck()
        original_length = len(deck)
        rng = random.Random(42)
        
        shuffle_in_place(deck, rng)
        
        assert len(deck) == original_length

    def test_shuffle_in_place_maintains_card_identity(self) -> None:
        """Shuffling should not change which cards are present."""
        deck = build_standard_deck()
        original_types = Counter(c.type for c in deck)
        original_colors = Counter(c.color for c in deck)
        rng = random.Random(42)
        
        shuffle_in_place(deck, rng)
        
        shuffled_types = Counter(c.type for c in deck)
        shuffled_colors = Counter(c.color for c in deck)
        
        assert shuffled_types == original_types
        assert shuffled_colors == original_colors

    def test_shuffle_in_place_changes_order(self) -> None:
        """Shuffling should actually change card order."""
        deck1 = build_standard_deck()
        deck2 = list(deck1)  # Copy
        rng = random.Random(42)
        
        shuffle_in_place(deck1, rng)
        
        # Extremely unlikely to be the same order
        assert deck1 != deck2

    def test_shuffle_is_deterministic_with_same_seed(self) -> None:
        """Same seed should produce same shuffle."""
        deck1 = build_standard_deck()
        deck2 = build_standard_deck()
        
        shuffle_in_place(deck1, random.Random(123))
        shuffle_in_place(deck2, random.Random(123))
        
        assert deck1 == deck2

    def test_shuffle_is_different_with_different_seeds(self) -> None:
        """Different seeds should produce different shuffles."""
        deck1 = build_standard_deck()
        deck2 = build_standard_deck()
        
        shuffle_in_place(deck1, random.Random(123))
        shuffle_in_place(deck2, random.Random(456))
        
        # Extremely unlikely to be the same
        assert deck1 != deck2

    def test_shuffle_empty_deck_does_not_crash(self) -> None:
        """Shuffling empty deck should not raise error."""
        deck: List[Card] = []
        rng = random.Random(42)
        
        shuffle_in_place(deck, rng)
        assert len(deck) == 0

    def test_shuffle_single_card_deck(self) -> None:
        """Shuffling single card should work."""
        deck = [Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)]
        rng = random.Random(42)
        
        shuffle_in_place(deck, rng)
        assert len(deck) == 1


class TestBuryCard:
    """Tests for the bury_card operation."""

    def test_bury_card_increases_deck_size(self) -> None:
        """Burying a card should add it to the deck."""
        deck = build_standard_deck()[:10]  # Take first 10 cards
        card = Card(color=Color.RED, type=CardType.WILD_DRAW_FOUR, value=50)
        original_length = len(deck)
        rng = random.Random(42)
        
        bury_card(deck, card, rng)
        
        assert len(deck) == original_length + 1

    def test_bury_card_adds_correct_card(self) -> None:
        """Buried card should be present in deck."""
        deck = build_standard_deck()[:10]
        card = Card(color=Color.RED, type=CardType.NUMBER, number=7, value=7)
        rng = random.Random(42)
        
        bury_card(deck, card, rng)
        
        assert card in deck

    def test_bury_card_into_empty_deck(self) -> None:
        """Burying into empty deck should work."""
        deck: List[Card] = []
        card = Card(color=Color.BLUE, type=CardType.SKIP, value=20)
        rng = random.Random(42)
        
        bury_card(deck, card, rng)
        
        assert len(deck) == 1
        assert deck[0] == card

    def test_bury_card_random_position(self) -> None:
        """Buried card can appear at different positions."""
        positions = set()
        card = Card(color=Color.GREEN, type=CardType.REVERSE, value=20)
        
        for seed in range(50):
            deck = [Card(color=Color.RED, type=CardType.NUMBER, number=i % 10, value=i % 10) for i in range(20)]
            rng = random.Random(seed)
            
            bury_card(deck, card, rng)
            
            position = deck.index(card)
            positions.add(position)
        
        # Should have multiple different positions
        assert len(positions) > 5

    def test_bury_card_maintains_other_cards(self) -> None:
        """Burying should not affect existing cards."""
        original_cards = [
            Card(color=Color.RED, type=CardType.NUMBER, number=i, value=i)
            for i in range(5)
        ]
        deck = list(original_cards)
        card = Card(color=Color.BLUE, type=CardType.WILD, value=50)
        rng = random.Random(42)
        
        bury_card(deck, card, rng)
        
        # Remove the buried card
        deck.remove(card)
        
        # Should have all original cards
        assert deck == original_cards


class TestDeckEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_number_card_validation(self) -> None:
        """Number cards must have a number value."""
        with pytest.raises(ValueError):
            Card(color=Color.RED, type=CardType.NUMBER, value=5, number=None)

    def test_non_number_card_cannot_have_number(self) -> None:
        """Non-number cards cannot have a number value."""
        with pytest.raises(ValueError):
            Card(color=Color.RED, type=CardType.SKIP, value=20, number=5)

    def test_all_cards_in_standard_deck_are_valid(self) -> None:
        """Every card in standard deck should be properly constructed."""
        deck = build_standard_deck()
        
        for card in deck:
            # Should not raise any errors
            assert isinstance(card, Card)
            assert isinstance(card.color, Color)
            assert isinstance(card.type, CardType)
            assert isinstance(card.value, int)

    def test_all_cards_in_go_wild_deck_are_valid(self) -> None:
        """Every card in Go Wild deck should be properly constructed."""
        deck = build_go_wild_deck()
        
        for card in deck:
            assert isinstance(card, Card)
            assert isinstance(card.color, Color)
            assert isinstance(card.type, CardType)
            assert isinstance(card.value, int)
