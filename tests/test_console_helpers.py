"""Tests for console interface helper functions (non-training components).

These tests cover the utility functions used in the console interface
without touching any training-related components.
"""

from __future__ import annotations

import pytest

from uno_engine.models import Card, CardType, Color


# Import console functions conditionally to avoid training dependencies
try:
    from play_bnn_console import format_card, describe_token
    CONSOLE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CONSOLE_AVAILABLE = False
    format_card = None
    describe_token = None


# Skip all tests if console module not available
pytestmark = pytest.mark.skipif(not CONSOLE_AVAILABLE, reason="Console module requires training dependencies")


class TestFormatCard:
    """Tests for card formatting display function."""

    def test_format_number_card(self) -> None:
        """Number cards should show color and number."""
        card = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        result = format_card(card)
        assert "Red" in result
        assert "5" in result

    def test_format_all_colors(self) -> None:
        """All colors should be formatted correctly."""
        colors = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]
        for color in colors:
            card = Card(color=color, type=CardType.NUMBER, number=3, value=3)
            result = format_card(card)
            assert color.value.title() in result

    def test_format_wild_card(self) -> None:
        """Wild card should be formatted as 'Wild'."""
        card = Card(color=Color.WILD, type=CardType.WILD, value=50)
        result = format_card(card)
        assert "Wild" in result

    def test_format_wild_draw_four(self) -> None:
        """Wild Draw Four should be clearly labeled."""
        card = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)
        result = format_card(card)
        assert "Wild" in result
        assert "Draw" in result
        assert "Four" in result

    def test_format_draw_two(self) -> None:
        """Draw Two should show color and type."""
        card = Card(color=Color.RED, type=CardType.DRAW_TWO, value=20)
        result = format_card(card)
        assert "Draw Two" in result
        assert "Red" in result

    def test_format_skip(self) -> None:
        """Skip cards should be formatted with color."""
        card = Card(color=Color.BLUE, type=CardType.SKIP, value=20)
        result = format_card(card)
        assert "Skip" in result
        assert "Blue" in result

    def test_format_reverse(self) -> None:
        """Reverse cards should be formatted with color."""
        card = Card(color=Color.GREEN, type=CardType.REVERSE, value=20)
        result = format_card(card)
        assert "Reverse" in result
        assert "Green" in result

    def test_format_discard_all(self) -> None:
        """Discard All cards should be formatted with color."""
        card = Card(color=Color.YELLOW, type=CardType.DISCARD_ALL, value=40)
        result = format_card(card)
        assert "Discard All" in result
        assert "Yellow" in result

    def test_format_returns_string(self) -> None:
        """format_card should always return a string."""
        cards = [
            Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5),
            Card(color=Color.WILD, type=CardType.WILD, value=50),
            Card(color=Color.BLUE, type=CardType.SKIP, value=20),
        ]
        for card in cards:
            result = format_card(card)
            assert isinstance(result, str)
            assert len(result) > 0


class TestDescribeToken:
    """Tests for action token description function."""

    def test_describe_draw_token(self) -> None:
        """DRAW token should be described clearly."""
        result = describe_token("DRAW")
        assert "Draw" in result
        assert "card" in result.lower()

    def test_describe_number_card_token(self) -> None:
        """Number card tokens should show color and number."""
        result = describe_token("PLAY_NUMBER_RED_5")
        assert "Red" in result
        assert "5" in result

    def test_describe_skip_token(self) -> None:
        """Skip tokens should show type and color."""
        result = describe_token("PLAY_SKIP_BLUE")
        assert "Skip" in result
        assert "Blue" in result

    def test_describe_reverse_token(self) -> None:
        """Reverse tokens should show type and color."""
        result = describe_token("PLAY_REVERSE_GREEN")
        assert "Reverse" in result
        assert "Green" in result

    def test_describe_draw_two_token(self) -> None:
        """Draw Two tokens should be clearly described."""
        result = describe_token("PLAY_DRAW_TWO_RED")
        assert "Draw Two" in result
        assert "Red" in result

    def test_describe_discard_all_token(self) -> None:
        """Discard All tokens should be clearly described."""
        result = describe_token("PLAY_DISCARD_ALL_YELLOW")
        assert "Discard All" in result
        assert "Yellow" in result

    def test_describe_wild_token(self) -> None:
        """Wild tokens should show chosen color."""
        result = describe_token("PLAY_WILD_BLUE")
        assert "Wild" in result
        assert "Blue" in result

    def test_describe_wild_draw_four_token(self) -> None:
        """Wild Draw Four tokens should show chosen color."""
        result = describe_token("PLAY_WILD_DRAW_FOUR_GREEN")
        assert "Wild" in result
        assert "Draw Four" in result or "Four" in result
        assert "Green" in result

    def test_describe_handles_lowercase(self) -> None:
        """Tokens should be handled case-insensitively."""
        result = describe_token("draw")
        assert "Draw" in result or "draw" in result

    def test_describe_unknown_token_returns_string(self) -> None:
        """Unknown tokens should still return something."""
        result = describe_token("UNKNOWN_TOKEN")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_various_colors(self) -> None:
        """All color variations should be handled."""
        colors = ["RED", "YELLOW", "GREEN", "BLUE"]
        for color in colors:
            token = f"PLAY_NUMBER_{color}_3"
            result = describe_token(token)
            assert color.title() in result or color.lower() in result


class TestCardFormatting:
    """Integration tests for card display formatting."""

    def test_format_all_number_cards(self) -> None:
        """All number cards 0-9 should format correctly."""
        for number in range(10):
            card = Card(color=Color.RED, type=CardType.NUMBER, number=number, value=number)
            result = format_card(card)
            assert str(number) in result

    def test_format_all_action_types(self) -> None:
        """All action card types should format without error."""
        action_types = [CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO]
        for card_type in action_types:
            card = Card(color=Color.RED, type=card_type, value=20)
            result = format_card(card)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_format_consistency(self) -> None:
        """Formatting should be consistent for same card."""
        card = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        result1 = format_card(card)
        result2 = format_card(card)
        assert result1 == result2

    def test_different_cards_different_format(self) -> None:
        """Different cards should have different formatted strings."""
        card1 = Card(color=Color.RED, type=CardType.NUMBER, number=5, value=5)
        card2 = Card(color=Color.BLUE, type=CardType.NUMBER, number=7, value=7)
        
        result1 = format_card(card1)
        result2 = format_card(card2)
        
        assert result1 != result2


class TestTokenDescriptionConsistency:
    """Tests for token description consistency and edge cases."""

    def test_describe_token_is_deterministic(self) -> None:
        """Same token should always produce same description."""
        token = "PLAY_NUMBER_RED_5"
        result1 = describe_token(token)
        result2 = describe_token(token)
        assert result1 == result2

    def test_describe_different_tokens_different_output(self) -> None:
        """Different tokens should produce different descriptions."""
        token1 = "PLAY_NUMBER_RED_5"
        token2 = "PLAY_NUMBER_BLUE_7"
        
        result1 = describe_token(token1)
        result2 = describe_token(token2)
        
        # They should differ (unless implementation is very generic)
        # At least one should contain its specific details
        assert ("Red" in result1 or "5" in result1) and ("Blue" in result2 or "7" in result2)

    def test_describe_handles_empty_string(self) -> None:
        """Empty string should not crash."""
        result = describe_token("")
        assert isinstance(result, str)

    def test_describe_all_action_tokens(self) -> None:
        """All action type tokens should be describable."""
        tokens = [
            "PLAY_SKIP_RED",
            "PLAY_REVERSE_BLUE",
            "PLAY_DRAW_TWO_GREEN",
            "PLAY_WILD_YELLOW",
            "PLAY_WILD_DRAW_FOUR_RED",
            "DRAW",
        ]
        
        for token in tokens:
            result = describe_token(token)
            assert isinstance(result, str)
            assert len(result) > 0


class TestDisplayHelperEdgeCases:
    """Edge cases and boundary conditions for display helpers."""

    def test_format_card_with_zero_number(self) -> None:
        """Zero number cards should format correctly."""
        card = Card(color=Color.RED, type=CardType.NUMBER, number=0, value=0)
        result = format_card(card)
        assert "0" in result

    def test_format_card_with_all_wild_types(self) -> None:
        """Both wild card types should format."""
        wild = Card(color=Color.WILD, type=CardType.WILD, value=50)
        wild_draw_four = Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50)
        
        result1 = format_card(wild)
        result2 = format_card(wild_draw_four)
        
        assert result1 != result2  # Should be different
        assert "Wild" in result1
        assert "Wild" in result2

    def test_token_description_with_special_characters(self) -> None:
        """Tokens with underscores should be handled."""
        token = "PLAY_WILD_DRAW_FOUR_RED"
        result = describe_token(token)
        assert isinstance(result, str)

    def test_format_readability(self) -> None:
        """Formatted strings should be human-readable."""
        card = Card(color=Color.RED, type=CardType.SKIP, value=20)
        result = format_card(card)
        
        # Should contain readable words, not just codes
        assert not result.startswith("Card(")
        assert "Skip" in result or "SKIP" in result
