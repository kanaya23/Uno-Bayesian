"""Domain models and enumerations for the UNO game engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


class Color(Enum):
    """Possible colors that a UNO card can have."""

    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    BLUE = "BLUE"
    WILD = "WILD"


class CardType(Enum):
    """All supported UNO card types."""

    NUMBER = "NUMBER"
    SKIP = "SKIP"
    REVERSE = "REVERSE"
    DRAW_TWO = "DRAW_TWO"
    WILD = "WILD"
    WILD_DRAW_FOUR = "WILD_DRAW_FOUR"
    DISCARD_ALL = "DISCARD_ALL"


class GameMode(Enum):
    """Available game modes supported by the engine."""

    CLASSIC_2V2 = "CLASSIC_2V2"
    GO_WILD_2V2 = "GO_WILD_2V2"


class PlayDirection(Enum):
    """Directional state for turn progression."""

    CLOCKWISE = auto()
    COUNTER_CLOCKWISE = auto()


class PendingActionType(Enum):
    """Marker for special, out-of-band obligations owed by a player."""

    DRAWN_CARD_PLAY_WINDOW = auto()
    COLOR_CHOICE = auto()
    DRAW_STACK = auto()


@dataclass(slots=True)
class Card:
    """Represents a single UNO card."""

    color: Color
    type: CardType
    value: int
    number: Optional[int] = None

    def __post_init__(self) -> None:
        if self.type == CardType.NUMBER and self.number is None:
            raise ValueError("Number cards must include their numeric face value.")
        if self.type != CardType.NUMBER and self.number is not None:
            raise ValueError("Only number cards may carry a numeric face value.")


@dataclass(slots=True)
class Player:
    """Encapsulates a single player's public and private state."""

    player_id: str
    hand: List[Card] = field(default_factory=list)
    score: int = 0


@dataclass(slots=True)
class PendingAction:
    """State describing an outstanding requirement for a specific player."""

    player_id: str
    type: PendingActionType
    allowed_cards: Tuple[Card, ...] = ()
    draw_penalty: int = 0


@dataclass(slots=True)
class GameState:
    """Represents the full state for an in-progress UNO round."""

    players: List[Player]
    draw_pile: List[Card]
    discard_pile: List[Card]
    current_player_index: int
    play_direction: PlayDirection
    current_color: Optional[Color]
    pending_action: Optional[PendingAction] = None
    round_over: bool = False
    round_winner_id: Optional[str] = None
    game_over: bool = False
    game_winner_ids: Tuple[str, ...] = ()
    mode: GameMode = GameMode.CLASSIC_2V2
    team_map: Dict[str, int] = field(default_factory=dict)
    team_scores: Dict[int, int] = field(default_factory=dict)
    round_winning_team_ids: Tuple[str, ...] = ()
    draw_stack_total: int = 0

    def current_player(self) -> Player:
        """Convenience accessor for the active player object."""

        return self.players[self.current_player_index]

    def player_ids(self) -> Sequence[str]:
        """Ordered list of player identifiers."""

        return tuple(player.player_id for player in self.players)

    def team_index_for(self, player_id: str) -> int:
        """Return the team index for the supplied player id."""

        try:
            return self.team_map[player_id]
        except KeyError as exc:
            raise KeyError(f"Unknown player id '{player_id}' in team map") from exc

    def teammate_id(self, player_id: str) -> str:
        """Return the teammate's player id for the supplied player id."""

        team_index = self.team_index_for(player_id)
        for candidate in self.players:
            if candidate.player_id != player_id and self.team_map.get(candidate.player_id) == team_index:
                return candidate.player_id
        raise KeyError(f"No teammate registered for player id '{player_id}'")

    def visible_hands_for(self, player_id: str) -> Mapping[str, Sequence[Card]]:
        """Return the hands visible to the supplied player (self + teammate)."""

        teammate = self.teammate_id(player_id)
        owner_hand = next(player.hand for player in self.players if player.player_id == player_id)
        teammate_hand = next(player.hand for player in self.players if player.player_id == teammate)
        return {
            player_id: tuple(owner_hand),
            teammate: tuple(teammate_hand),
        }
