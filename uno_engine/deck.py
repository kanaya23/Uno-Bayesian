"""Deck construction and shuffling helpers for the UNO engine."""

from __future__ import annotations

from random import Random
from typing import List

from .models import Card, CardType, Color, GameMode


def build_standard_deck() -> List[Card]:
    """Return a freshly constructed, ordered 108-card UNO deck."""

    deck: List[Card] = []
    colored_cards = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]

    for color in colored_cards:
        # Single zero per color.
        deck.append(Card(color=color, type=CardType.NUMBER, number=0, value=0))

        # Two copies of each digit 1-9 per color.
        for number in range(1, 10):
            deck.extend(
                Card(color=color, type=CardType.NUMBER, number=number, value=number)
                for _ in range(2)
            )

        # Two of each action per color.
        for _ in range(2):
            deck.append(Card(color=color, type=CardType.SKIP, value=20))
            deck.append(Card(color=color, type=CardType.REVERSE, value=20))
            deck.append(Card(color=color, type=CardType.DRAW_TWO, value=20))

    # Four wilds and four wild draw fours.
    for _ in range(4):
        deck.append(Card(color=Color.WILD, type=CardType.WILD, value=50))
        deck.append(Card(color=Color.WILD, type=CardType.WILD_DRAW_FOUR, value=50))

    if len(deck) != 108:
        raise AssertionError("Standard UNO deck must contain exactly 108 cards.")

    return deck


def build_go_wild_deck() -> List[Card]:
    """Return the 2v2 Go Wild deck (double deck + Discard All cards)."""

    deck: List[Card] = []

    # Two full standard decks combined.
    for _ in range(2):
        deck.extend(build_standard_deck())

    colored_cards = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]
    for color in colored_cards:
        for _ in range(2):
            deck.append(Card(color=color, type=CardType.DISCARD_ALL, value=40))

    expected_size = (108 * 2) + (len(colored_cards) * 2)
    if len(deck) != expected_size:
        raise AssertionError("Go Wild deck must contain 224 cards.")

    return deck


def build_deck_for_mode(mode: GameMode) -> List[Card]:
    """Return the appropriate deck for the supplied game mode."""

    if mode == GameMode.CLASSIC_2V2:
        return build_standard_deck()
    if mode == GameMode.GO_WILD_2V2:
        return build_go_wild_deck()
    raise ValueError(f"Unsupported game mode '{mode}'.")


def shuffle_in_place(cards: List[Card], rng: Random) -> None:
    """In-place Fisher-Yates shuffle powered by the supplied RNG."""

    for index in range(len(cards) - 1, 0, -1):
        swap_index = rng.randint(0, index)
        cards[index], cards[swap_index] = cards[swap_index], cards[index]


def bury_card(cards: List[Card], card: Card, rng: Random) -> None:
    """Reinsert *card* at a random index within *cards* (assumes card absent)."""

    insertion_point = rng.randint(0, len(cards))
    cards.insert(insertion_point, card)
