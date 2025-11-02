"""Rule-accurate UNO engine implementing state transitions and validations."""

from __future__ import annotations

import copy
from random import Random
from typing import List, Optional, Sequence

from .deck import build_deck_for_mode, bury_card, shuffle_in_place
from .models import (
    Card,
    CardType,
    Color,
    GameState,
    GameMode,
    PendingAction,
    PendingActionType,
    PlayDirection,
    Player,
)


class UnoError(Exception):
    """Base class for UNO engine exceptions."""


class InvalidMoveError(UnoError):
    """Raised when a player attempts an illegal move."""


class ColorSelectionError(UnoError):
    """Raised when a color choice is requested at an invalid time."""


class UnoEngine:
    """Core engine responsible for UNO game state management."""

    def __init__(
        self,
        rng: Optional[Random] = None,
        default_mode: GameMode = GameMode.CLASSIC_2V2,
    ) -> None:
        self._rng = rng or Random()
        self._default_mode = default_mode

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------
    def init_game(
        self,
        players: Sequence[Player],
        *,
        mode: Optional[GameMode] = None,
    ) -> GameState:
        """Create a freshly initialized round preserving player scores.

        Args:
            players: Ordered sequence of players (must be exactly four).
            mode: Optional override for the game mode (defaults to engine setting).

        Returns:
            GameState: The initialized round state.
        """

        if len(players) != 4:
            raise ValueError("UNO 2v2 requires exactly four players per round.")

        selected_mode = mode or self._default_mode

        # Copy the public player metadata but reset hands.
        ordered_players = [
            Player(player_id=p.player_id, score=p.score, hand=list()) for p in players
        ]

        team_map, team_scores = self._build_team_metadata(ordered_players)

        deck = build_deck_for_mode(selected_mode)
        shuffle_in_place(deck, self._rng)

        # Deal 7 cards to each player in sequence.
        for _ in range(7):
            for player in ordered_players:
                player.hand.append(deck.pop())

        discard_pile: List[Card] = []

        # Draw the starting card, avoiding Wild Draw Four.
        while True:
            top_card = deck.pop()
            if top_card.type == CardType.WILD_DRAW_FOUR:
                bury_card(deck, top_card, self._rng)
                continue
            discard_pile.append(top_card)
            break

        current_color: Optional[Color] = (
            None if top_card.type == CardType.WILD else top_card.color
        )

        state = GameState(
            players=ordered_players,
            draw_pile=deck,
            discard_pile=discard_pile,
            current_player_index=0,
            play_direction=PlayDirection.CLOCKWISE,
            current_color=current_color,
            mode=selected_mode,
            team_map=team_map,
            team_scores=team_scores,
        )

        self._apply_initial_card_effect(state)
        return state

    # ------------------------------------------------------------------
    # Player-facing API
    # ------------------------------------------------------------------
    def get_valid_moves(self, state: GameState, player_id: str) -> List[Card]:
        """Return the list of cards the player may legally play."""

        self._ensure_round_active(state)
        player_index = self._player_index(state, player_id)

        if state.pending_action:
            if state.pending_action.type == PendingActionType.COLOR_CHOICE:
                return []
            if state.pending_action.type == PendingActionType.DRAW_STACK:
                if state.pending_action.player_id != player_id:
                    return []
                return self._stack_response_options(state, player_index)
            if state.pending_action.type == PendingActionType.DRAWN_CARD_PLAY_WINDOW:
                if state.pending_action.player_id != player_id:
                    return []
                return list(state.pending_action.allowed_cards)

        if player_index != state.current_player_index:
            return []

        if state.pending_action and state.pending_action.player_id != player_id:
            return []

        current_color = self._current_color(state)
        top_card = state.discard_pile[-1]

        valid_moves: List[Card] = []
        for card in state.players[player_index].hand:
            if self._is_card_playable(card, state.players[player_index].hand, top_card, current_color):
                valid_moves.append(card)

        return valid_moves

    def play_card(
        self,
        state: GameState,
        player_id: str,
        card: Card,
        chosen_color: Optional[Color] = None,
    ) -> GameState:
        """Execute a card play for the active player."""

        self._ensure_round_active(state)
        self._ensure_no_color_choice_pending(state)

        player_index = self._player_index(state, player_id)

        pending = state.pending_action
        if pending:
            if pending.type == PendingActionType.DRAWN_CARD_PLAY_WINDOW and pending.player_id != player_id:
                raise InvalidMoveError("Only the drawing player may act right now.")
            if pending.type == PendingActionType.DRAW_STACK and pending.player_id != player_id:
                raise InvalidMoveError("Only the targeted player may respond to the draw stack.")

        if pending is None and player_index != state.current_player_index:
            raise InvalidMoveError("It is not this player's turn.")
        if pending and player_index != state.current_player_index:
            raise InvalidMoveError("It is not this player's turn.")

        cloned_state = self._clone_state(state)
        working_player = cloned_state.players[player_index]

        hand_index, in_hand_card = self._find_card_in_hand(working_player.hand, card)

        valid_moves = self.get_valid_moves(state, player_id)
        if in_hand_card not in valid_moves:
            raise InvalidMoveError("Card is not playable right now.")

        # Remove card from hand and place onto discard pile.
        del working_player.hand[hand_index]
        cloned_state.discard_pile.append(in_hand_card)

        # Resolve current color.
        if in_hand_card.type in (CardType.WILD, CardType.WILD_DRAW_FOUR):
            if chosen_color is None:
                raise InvalidMoveError("A color must be declared when playing a wild card.")
            if chosen_color == Color.WILD:
                raise InvalidMoveError("Chosen color must be a standard color.")
            cloned_state.current_color = chosen_color
        else:
            cloned_state.current_color = in_hand_card.color

        # Clear any transient pending action satisfied by this play.
        if cloned_state.pending_action and cloned_state.pending_action.type in (
            PendingActionType.DRAWN_CARD_PLAY_WINDOW,
            PendingActionType.DRAW_STACK,
        ):
            cloned_state.pending_action = None

        advance_steps = 1
        stacking_card_played = False

        if in_hand_card.type == CardType.DISCARD_ALL:
            self._execute_discard_all(cloned_state, working_player, in_hand_card)

        if in_hand_card.type == CardType.REVERSE:
            cloned_state.play_direction = self._reverse_direction(cloned_state.play_direction)

        elif in_hand_card.type == CardType.SKIP:
            advance_steps = 2

        elif in_hand_card.type == CardType.DRAW_TWO:
            if cloned_state.mode == GameMode.GO_WILD_2V2:
                stacking_card_played = True
                self._queue_draw_stack(cloned_state, player_index, penalty=2)
            else:
                self._force_draw(cloned_state, self._next_player_index(cloned_state, 1), 2)
                advance_steps = 2

        elif in_hand_card.type == CardType.WILD_DRAW_FOUR:
            if cloned_state.mode == GameMode.GO_WILD_2V2:
                stacking_card_played = True
                self._queue_draw_stack(cloned_state, player_index, penalty=4)
            else:
                self._force_draw(cloned_state, self._next_player_index(cloned_state, 1), 4)
                advance_steps = 2

        if not stacking_card_played:
            self._reset_draw_stack(cloned_state)

        # After resolving the card effect, check for round completion.
        if not working_player.hand:
            self._finalize_round(cloned_state, working_player.player_id)
            return cloned_state

        self._advance_turn(cloned_state, advance_steps)
        return cloned_state

    def draw_card(self, state: GameState, player_id: str) -> GameState:
        """Allow the active player to draw one card."""

        self._ensure_round_active(state)
        self._ensure_no_color_choice_pending(state)

        player_index = self._player_index(state, player_id)

        pending = state.pending_action
        if pending and pending.type == PendingActionType.DRAW_STACK:
            if pending.player_id != player_id:
                raise InvalidMoveError("Only the targeted player may resolve the draw stack.")
            cloned_state = self._clone_state(state)
            self._force_draw(cloned_state, player_index, pending.draw_penalty)
            self._reset_draw_stack(cloned_state)
            self._advance_turn(cloned_state, 1)
            return cloned_state

        if player_index != state.current_player_index:
            raise InvalidMoveError("Only the active player may draw a card.")

        if state.pending_action and state.pending_action.type == PendingActionType.DRAWN_CARD_PLAY_WINDOW:
            raise InvalidMoveError("Player must resolve drawn-card window first.")

        if self.get_valid_moves(state, player_id):
            raise InvalidMoveError("Player must play a valid card before drawing.")

        cloned_state = self._clone_state(state)
        drawn_card = self._draw_single(cloned_state, player_index)

        current_color = self._current_color(cloned_state)
        top_card = cloned_state.discard_pile[-1]
        if self._is_card_playable(drawn_card, cloned_state.players[player_index].hand, top_card, current_color):
            cloned_state.pending_action = PendingAction(
                player_id=player_id,
                type=PendingActionType.DRAWN_CARD_PLAY_WINDOW,
                allowed_cards=(drawn_card,),
            )
        else:
            self._advance_turn(cloned_state, 1)

        return cloned_state

    def pass_turn(self, state: GameState, player_id: str) -> GameState:
        """Explicitly pass after drawing when no play is made."""

        self._ensure_round_active(state)

        if not state.pending_action or state.pending_action.type != PendingActionType.DRAWN_CARD_PLAY_WINDOW:
            raise InvalidMoveError("No drawn-card window is active.")

        if state.pending_action.player_id != player_id:
            raise InvalidMoveError("Only the drawing player may pass.")

        cloned_state = self._clone_state(state)
        cloned_state.pending_action = None
        self._advance_turn(cloned_state, 1)
        return cloned_state

    def choose_color(self, state: GameState, player_id: str, color: Color) -> GameState:
        """Resolve an outstanding color declaration (e.g., initial wild)."""

        self._ensure_round_active(state)

        if color == Color.WILD:
            raise ColorSelectionError("Chosen color must be RED, YELLOW, GREEN, or BLUE.")

        if not state.pending_action or state.pending_action.type != PendingActionType.COLOR_CHOICE:
            raise ColorSelectionError("No color choice is currently pending.")

        if state.pending_action.player_id != player_id:
            raise ColorSelectionError("This player is not responsible for choosing the color.")

        cloned_state = self._clone_state(state)
        cloned_state.current_color = color
        cloned_state.pending_action = None
        return cloned_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_initial_card_effect(self, state: GameState) -> None:
        """Apply the effect of the opening discard card."""

        top_card = state.discard_pile[-1]

        if top_card.type == CardType.REVERSE:
            state.play_direction = self._reverse_direction(state.play_direction)
            state.current_player_index = self._next_player_index(state, 1)

        elif top_card.type == CardType.SKIP:
            state.current_player_index = self._next_player_index(state, 1)

        elif top_card.type == CardType.DRAW_TWO:
            self._force_draw(state, state.current_player_index, 2)
            state.current_player_index = self._next_player_index(state, 1)

        elif top_card.type == CardType.WILD:
            state.pending_action = PendingAction(
                player_id=state.players[state.current_player_index].player_id,
                type=PendingActionType.COLOR_CHOICE,
            )

    def _build_team_metadata(
        self, players: Sequence[Player]
    ) -> tuple[dict[str, int], dict[int, int]]:
        team_pairs = ((0, 2), (1, 3))
        team_map: dict[str, int] = {}
        team_scores: dict[int, int] = {}

        for team_index, (first_idx, second_idx) in enumerate(team_pairs):
            first_player = players[first_idx]
            second_player = players[second_idx]
            if first_player.score != second_player.score:
                raise ValueError("Teammates must begin a round with matching scores.")
            team_map[first_player.player_id] = team_index
            team_map[second_player.player_id] = team_index
            team_scores[team_index] = first_player.score

        return team_map, team_scores

    def _clone_state(self, state: GameState) -> GameState:
        return copy.deepcopy(state)

    def _player_index(self, state: GameState, player_id: str) -> int:
        for index, player in enumerate(state.players):
            if player.player_id == player_id:
                return index
        raise InvalidMoveError(f"Unknown player id '{player_id}'.")

    def _find_card_in_hand(self, hand: List[Card], target: Card) -> tuple[int, Card]:
        for index, card in enumerate(hand):
            if card == target:
                return index, card
        raise InvalidMoveError("Card is not present in hand.")

    def _reverse_direction(self, direction: PlayDirection) -> PlayDirection:
        return (
            PlayDirection.COUNTER_CLOCKWISE
            if direction == PlayDirection.CLOCKWISE
            else PlayDirection.CLOCKWISE
        )

    def _next_player_index(
        self, state: GameState, steps: int, start_index: Optional[int] = None
    ) -> int:
        base_index = state.current_player_index if start_index is None else start_index
        direction_multiplier = 1 if state.play_direction == PlayDirection.CLOCKWISE else -1
        return (base_index + (steps * direction_multiplier)) % len(state.players)

    def _advance_turn(self, state: GameState, steps: int) -> None:
        state.current_player_index = self._next_player_index(state, steps)

    def _draw_single(self, state: GameState, player_index: int) -> Card:
        if not state.draw_pile:
            self._recycle_discard_into_draw(state)
        card = state.draw_pile.pop()
        state.players[player_index].hand.append(card)
        return card

    def _force_draw(self, state: GameState, player_index: int, count: int) -> None:
        for _ in range(count):
            self._draw_single(state, player_index)

    def _recycle_discard_into_draw(self, state: GameState) -> None:
        if len(state.discard_pile) <= 1:
            raise UnoError("Cannot recycle discard pile with fewer than two cards.")

        top_card = state.discard_pile[-1]
        recyclable = state.discard_pile[:-1]
        shuffle_in_place(recyclable, self._rng)
        state.draw_pile[:] = recyclable
        state.discard_pile[:] = [top_card]

    def _queue_draw_stack(self, state: GameState, source_index: int, penalty: int) -> None:
        state.draw_stack_total += penalty
        target_index = self._next_player_index(state, 1, start_index=source_index)
        allowed_cards = self._stack_response_options(state, target_index)
        state.pending_action = PendingAction(
            player_id=state.players[target_index].player_id,
            type=PendingActionType.DRAW_STACK,
            allowed_cards=tuple(allowed_cards),
            draw_penalty=state.draw_stack_total,
        )

    def _reset_draw_stack(self, state: GameState) -> None:
        state.draw_stack_total = 0
        if state.pending_action and state.pending_action.type == PendingActionType.DRAW_STACK:
            state.pending_action = None

    def _stack_response_options(self, state: GameState, player_index: int) -> List[Card]:
        top_card = state.discard_pile[-1]
        hand = state.players[player_index].hand

        if top_card.type == CardType.DRAW_TWO:
            draw_twos = [card for card in hand if card.type == CardType.DRAW_TWO]
            if draw_twos:
                return draw_twos
            return [card for card in hand if card.type == CardType.WILD_DRAW_FOUR]

        if top_card.type == CardType.WILD_DRAW_FOUR:
            return [card for card in hand if card.type == CardType.WILD_DRAW_FOUR]

        return []

    def _execute_discard_all(self, state: GameState, player: Player, card: Card) -> None:
        matching_color = card.color
        additional_cards = [c for c in list(player.hand) if c.color == matching_color]
        for extra in additional_cards:
            player.hand.remove(extra)
            state.discard_pile.append(extra)

    def _is_card_playable(
        self,
        card: Card,
        hand: Sequence[Card],
        top_card: Card,
        current_color: Optional[Color],
    ) -> bool:
        if card.type == CardType.WILD:
            return True

        if card.type == CardType.WILD_DRAW_FOUR:
            return self._can_play_wild_draw_four(card, hand, current_color)

        if current_color is not None and card.color == current_color:
            return True

        if top_card.type == CardType.NUMBER and card.type == CardType.NUMBER:
            return card.number == top_card.number

        if card.type in {CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO, CardType.DISCARD_ALL}:
            return card.type == top_card.type

        return False

    def _can_play_wild_draw_four(
        self, card: Card, hand: Sequence[Card], current_color: Optional[Color]
    ) -> bool:
        if current_color is None:
            return True
        for other in hand:
            if other == card:
                continue
            if other.type == CardType.WILD:
                continue
            if other.color == current_color:
                return False
        return True

    def _current_color(self, state: GameState) -> Optional[Color]:
        if state.current_color is not None:
            return state.current_color
        if state.discard_pile:
            return state.discard_pile[-1].color
        return None

    def _ensure_round_active(self, state: GameState) -> None:
        if state.round_over:
            raise UnoError("Round has ended; no further actions are allowed.")

    def _ensure_no_color_choice_pending(self, state: GameState) -> None:
        if state.pending_action and state.pending_action.type == PendingActionType.COLOR_CHOICE:
            raise ColorSelectionError("Pending color choice must be resolved first.")

    def _finalize_round(self, state: GameState, winner_id: str) -> None:
        winner_team_index = state.team_index_for(winner_id)
        opponent_team_index = 1 - winner_team_index

        opponent_players = [
            player
            for player in state.players
            if state.team_map[player.player_id] == opponent_team_index
        ]
        total = sum(sum(card.value for card in player.hand) for player in opponent_players)

        for player in state.players:
            if state.team_map[player.player_id] == winner_team_index:
                player.score += total

        state.team_scores[winner_team_index] = next(
            player.score
            for player in state.players
            if state.team_map[player.player_id] == winner_team_index
        )
        state.team_scores[opponent_team_index] = next(
            player.score
            for player in state.players
            if state.team_map[player.player_id] == opponent_team_index
        )

        state.round_over = True
        state.round_winner_id = winner_id
        state.round_winning_team_ids = tuple(
            player.player_id
            for player in state.players
            if state.team_map[player.player_id] == winner_team_index
        )

        if state.team_scores[winner_team_index] >= 500:
            state.game_over = True
            state.game_winner_ids = state.round_winning_team_ids

    # Specification-friendly method aliases ---------------------------------
    InitGame = init_game
    GetValidMoves = get_valid_moves
    PlayCard = play_card
    DrawCard = draw_card
    PassTurn = pass_turn
    ChooseColor = choose_color


__all__ = ["UnoEngine", "UnoError", "InvalidMoveError", "ColorSelectionError"]
