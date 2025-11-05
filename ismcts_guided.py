"""Guided ISMCTS integration leveraging BNN priors for UNO gameplay.

This module couples the Bayesian neural network (BNN) predictors with an
Information Set Monte Carlo Tree Search (ISMCTS) planner. The resulting
searcher uses BNN guidance to prune, bias and evaluate branches while keeping
the public BNN interface unchanged.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from uno_engine.engine import InvalidMoveError, UnoEngine
from uno_engine.models import GameState

from notebooks.uno_bnn_curriculum_converted import (
    ActionType,
    BNNBot,
    BotAction,
    ScenarioType,
    TrainingArtifacts,
)


def clone_bot_action(action: BotAction) -> BotAction:
    """Return a shallow clone of *action* preserving metadata."""

    return BotAction(
        action_type=action.action_type,
        card=action.card,
        chosen_color=action.chosen_color,
        info=dict(action.info),
    )


@dataclass
class BNNActionPrior:
    """Container representing a BNN prior over an action."""

    token: str
    action: BotAction
    probability: float
    raw_probability: float
    std_dev: Optional[float]
    index: int
    pruned: bool = False

    def replicate(self) -> "BNNActionPrior":
        """Return a deep-copy style clone safe for tree reuse."""

        return BNNActionPrior(
            token=self.token,
            action=clone_bot_action(self.action),
            probability=self.probability,
            raw_probability=self.raw_probability,
            std_dev=self.std_dev,
            index=self.index,
            pruned=self.pruned,
        )


@dataclass
class BNNContext:
    """Captured BNN evaluation used to seed and inform ISMCTS."""

    player_id: str
    scenario_type: ScenarioType
    entropy: float
    normalized_entropy: float
    mutual_information: float
    mc_samples: int
    mean_probability: float
    probability_std: float
    priors: List[BNNActionPrior]
    token_to_prior: Dict[str, BNNActionPrior]
    valid_actions: Tuple[BotAction, ...]
    evaluation: Dict[str, Any]
    mean_probs: torch.Tensor
    mean_probs_masked: torch.Tensor
    std_probs_masked: Optional[torch.Tensor]
    mask: torch.Tensor

    def replicable_priors(self) -> List[BNNActionPrior]:
        return [prior.replicate() for prior in self.priors]


def _clone_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    cloned = value.detach().clone()
    if cloned.device.type != "cpu":
        cloned = cloned.cpu()
    return cloned


def build_bnn_context(
    artifacts: TrainingArtifacts,
    evaluation: Dict[str, Any],
    player_id: str,
) -> BNNContext:
    """Convert a BNN evaluation dictionary into a structured context."""

    candidates: Sequence[Dict[str, Any]] = evaluation.get("candidates", [])

    priors: List[BNNActionPrior] = []
    token_to_prior: Dict[str, BNNActionPrior] = {}
    for position, candidate in enumerate(candidates):
        prior = BNNActionPrior(
            token=candidate.get("token", ""),
            action=clone_bot_action(candidate["action"]),
            probability=float(candidate.get("prob", candidate.get("raw_prob", 0.0))),
            raw_probability=float(candidate.get("raw_prob", 0.0)),
            std_dev=candidate.get("std"),
            index=int(candidate.get("idx", position)),
            pruned=bool(candidate.get("pruned", False)),
        )
        priors.append(prior)
        token_to_prior[prior.token] = prior

    valid_actions: Tuple[BotAction, ...] = tuple(
        clone_bot_action(action) for action in evaluation.get("valid_actions", [])
    )

    mean_probs = _clone_tensor(evaluation.get("mean_probs"))
    if mean_probs is None:
        mean_probs = torch.zeros(len(artifacts.action_encoder.lookup), dtype=torch.float32)
    mean_probs = mean_probs.reshape(-1)

    mean_probs_masked = _clone_tensor(evaluation.get("mean_probs_masked"))
    if mean_probs_masked is None:
        mean_probs_masked = mean_probs.clone()
    mean_probs_masked = mean_probs_masked.reshape(-1)

    std_probs_masked = _clone_tensor(evaluation.get("std_probs_masked"))
    if std_probs_masked is not None:
        std_probs_masked = std_probs_masked.reshape(-1)

    mask = _clone_tensor(evaluation.get("mask"))
    if mask is None:
        mask = torch.ones_like(mean_probs)
    mask = mask.reshape(-1)

    return BNNContext(
        player_id=player_id,
        scenario_type=evaluation.get("scenario_type", ScenarioType.SETUP),
        entropy=float(evaluation.get("entropy", 0.0)),
        normalized_entropy=float(evaluation.get("normalized_entropy", evaluation.get("entropy", 0.0))),
        mutual_information=float(evaluation.get("mutual_information", 0.0)),
        mc_samples=int(evaluation.get("mc_samples", 0)),
        mean_probability=float(evaluation.get("prob_mean", 0.0)),
        probability_std=float(evaluation.get("prob_std", 0.0)),
        priors=priors,
        token_to_prior=token_to_prior,
        valid_actions=valid_actions,
        evaluation=dict(evaluation),
        mean_probs=mean_probs,
        mean_probs_masked=mean_probs_masked,
        std_probs_masked=std_probs_masked,
        mask=mask,
    )


def select_action_from_evaluation(
    evaluation: Dict[str, Any],
    bnn_helper: BNNBot,
    artifacts: TrainingArtifacts,
    state: GameState,
    player_id: str,
) -> BotAction:
    """Reproduce the BNNBot decision logic from a cached evaluation."""

    entropy = float(evaluation.get("entropy", 0.0))
    mutual_info = float(evaluation.get("mutual_information", 0.0))
    normalized_entropy = float(evaluation.get("normalized_entropy", entropy))
    scenario_type: ScenarioType = evaluation.get("scenario_type", ScenarioType.SETUP)
    candidates: List[Dict[str, Any]] = list(evaluation.get("candidates", []))
    valid_actions: Sequence[BotAction] = evaluation.get("valid_actions", [])

    if not valid_actions:
        fallback = BotAction(ActionType.DRAW)
        fallback.info.update(
            {
                "bnn_entropy": entropy,
                "bnn_entropy_normalized": normalized_entropy,
                "bnn_mutual_information": mutual_info,
                "persona_hint": bnn_helper.persona_hint,
                "scenario_type": scenario_type.value,
                "selection_reason": "no_actions",
            }
        )
        return fallback

    if not candidates:
        token = evaluation.get("prediction_token")
        try:
            fallback = artifacts.action_encoder.token_to_action(token, state, player_id)
        except Exception:
            fallback = BotAction(ActionType.DRAW)
        fallback.info.update(
            {
                "bnn_entropy": entropy,
                "bnn_entropy_normalized": normalized_entropy,
                "bnn_mutual_information": mutual_info,
                "persona_hint": bnn_helper.persona_hint,
                "scenario_type": scenario_type.value,
                "selection_reason": "encoder_miss",
            }
        )
        return fallback

    selection_reason = "probability"
    selected = candidates[0]
    if mutual_info > bnn_helper.mi_explore_threshold:
        exploratory = bnn_helper._select_exploration_candidate(state, candidates)
        if exploratory is not None:
            selected = exploratory
            selection_reason = "explore_high_mi"
    elif entropy < bnn_helper.entropy_exploit_threshold:
        exploit = bnn_helper._select_exploitation_candidate(candidates)
        if exploit is not None:
            selected = exploit
            selection_reason = "exploit_low_entropy"

    action = clone_bot_action(selected["action"])
    has_play_option = any(candidate["action"].action_type == ActionType.PLAY for candidate in candidates)

    action.info.update(
        {
            "bnn_entropy": entropy,
            "bnn_entropy_normalized": normalized_entropy,
            "bnn_mutual_information": mutual_info,
            "persona_hint": bnn_helper.persona_hint,
            "scenario_type": scenario_type.value,
            "bnn_probability": float(selected.get("prob", selected.get("raw_prob", 0.0))),
            "bnn_probability_raw": float(selected.get("raw_prob", 0.0)),
            "bnn_probability_std": selected.get("std"),
            "selection_reason": selection_reason,
            "mi_threshold": bnn_helper.mi_explore_threshold,
            "entropy_threshold": bnn_helper.entropy_exploit_threshold,
            "mc_samples": int(evaluation.get("mc_samples", 0)),
            "candidate_count": len(candidates),
            "pruned_candidates": int(evaluation.get("pruned_count", 0)),
            "prune_prob_threshold": bnn_helper.prune_prob_threshold,
            "prune_entropy_threshold": bnn_helper.prune_entropy_threshold,
            "mi_bonus_weight": bnn_helper.mi_bonus_weight,
            "probability_normalization": evaluation.get("normalization_kind"),
            "bnn_token": selected.get("token"),
        }
    )

    if action.action_type == ActionType.DRAW:
        action.info["forfeits_turn"] = has_play_option

    return action


class BNNMixin:
    """Mixin supplying BNN evaluation helpers to guided planners."""

    def __init__(
        self,
        artifacts: TrainingArtifacts,
        *,
        persona_hint: str = "OracleBot",
        mc_samples: int = 40,
        rng: Optional[random.Random] = None,
        **kwargs: Any,
    ) -> None:
        self.artifacts = artifacts
        self.persona_hint = persona_hint
        self.mc_samples = mc_samples
        self._guidance_rng = rng or random.Random()
        bnn_rng = random.Random(self._guidance_rng.random())
        self.bnn_helper = BNNBot(
            artifacts,
            persona_hint=self.persona_hint,
            rng=bnn_rng,
            mc_samples=mc_samples,
        )
        super().__init__(rng=self._guidance_rng, **kwargs)

    def evaluate_state(
        self,
        engine: UnoEngine,
        state: GameState,
        player_id: str,
    ) -> Dict[str, Any]:
        return self.bnn_helper.evaluate_candidates(engine, state, player_id)

    def build_context_from_state(
        self,
        engine: UnoEngine,
        state: GameState,
        player_id: str,
    ) -> BNNContext:
        evaluation = self.evaluate_state(engine, state, player_id)
        return build_bnn_context(self.artifacts, evaluation, player_id)

    def context_from_evaluation(
        self,
        evaluation: Dict[str, Any],
        player_id: str,
    ) -> BNNContext:
        return build_bnn_context(self.artifacts, evaluation, player_id)

    def bnn_policy_action(
        self,
        _engine: UnoEngine,
        state: GameState,
        player_id: str,
        evaluation: Dict[str, Any],
    ) -> BotAction:
        return select_action_from_evaluation(
            evaluation,
            self.bnn_helper,
            self.artifacts,
            state,
            player_id,
        )


class ISMCTSBase:
    """Lightweight ISMCTS scaffold supporting guided specialisations."""

    @dataclass
    class TreeNode:
        player_id: Optional[str]
        team_index: int
        action: Optional[BotAction] = None
        token: Optional[str] = None
        prior: float = 1.0
        parent: Optional["ISMCTSBase.TreeNode"] = None
        children: Dict[str, "ISMCTSBase.TreeNode"] = field(default_factory=dict)
        untried_actions: List[BNNActionPrior] = field(default_factory=list)
        visits: int = 0
        value_sum: float = 0.0
        context: Optional[BNNContext] = None

        def add_untried(self, priors: Iterable[BNNActionPrior]) -> None:
            self.untried_actions.extend(prior.replicate() for prior in priors)

    def __init__(
        self,
        *,
        engine: Optional[UnoEngine] = None,
        rng: Optional[random.Random] = None,
        exploration_constant: float = 1.5,
        rollout_depth: int = 10,
    ) -> None:
        self.engine = engine or UnoEngine()
        self.rng = rng or random.Random()
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth

    def _clone_state(self, state: GameState) -> GameState:
        return copy.deepcopy(state)

    def _apply_action(self, state: GameState, player_id: str, action: BotAction) -> GameState:
        if action.action_type == ActionType.PLAY and action.card is not None:
            return self.engine.play_card(state, player_id, action.card, action.chosen_color)
        if action.action_type == ActionType.DRAW:
            try:
                return self.engine.draw_card(state, player_id)
            except InvalidMoveError:
                return state
        return state

    def _determinize_state(self, state: GameState, _: str) -> GameState:
        """Return a determinized view of *state* (identity for now)."""

        return self._clone_state(state)

    def _select_child(self, node: "ISMCTSBase.TreeNode") -> Optional["ISMCTSBase.TreeNode"]:
        best_score = -float("inf")
        best_child: Optional[ISMCTSBase.TreeNode] = None
        parent_visits = max(1, node.visits)
        sqrt_parent = math.sqrt(parent_visits)
        for child in node.children.values():
            prior = max(child.prior, 0.0)
            if child.visits > 0:
                value = child.value_sum / child.visits
            else:
                value = 0.0
            exploration = self.exploration_constant * prior * sqrt_parent / (1 + child.visits)
            score = value + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backpropagate(
        self,
        path: Sequence["ISMCTSBase.TreeNode"],
        reward: float,
        root_team: int,
    ) -> None:
        for node in reversed(path):
            node.visits += 1
            if node.team_index == root_team:
                node.value_sum += reward
            else:
                node.value_sum -= reward

    def _evaluate_reward(self, state: GameState, root_player: str) -> float:
        if state.round_over:
            winner = state.round_winner_id
            if winner is None:
                return 0.0
            try:
                winner_team = state.team_index_for(winner)
            except KeyError:
                return 0.0
            root_team = state.team_index_for(root_player)
            return 1.0 if winner_team == root_team else -1.0

        root_team = state.team_index_for(root_player)
        opp_team = 1 - root_team
        my_cards = sum(
            len(player.hand) for player in state.players if state.team_map[player.player_id] == root_team
        )
        opp_cards = sum(
            len(player.hand) for player in state.players if state.team_map[player.player_id] == opp_team
        )
        diff = opp_cards - my_cards
        return math.tanh(diff / 6.0)

    def _rollout(self, state: GameState, current_player: Optional[str], root_player: str) -> float:
        raise NotImplementedError


class GuidedISMCTS(BNNMixin, ISMCTSBase):
    """ISMCTS planner guided by BNN priors for pruning and rollouts."""

    def __init__(
        self,
        artifacts: TrainingArtifacts,
        *,
        persona_hint: str = "OracleBot",
        mc_samples: int = 40,
        engine: Optional[UnoEngine] = None,
        rng: Optional[random.Random] = None,
        max_branching: int = 3,
        probability_mass_limit: float = 0.6,
        rollout_alpha: float = 0.75,
        exploration_constant: float = 1.5,
        rollout_depth: int = 10,
    ) -> None:
        self.max_branching = max(1, int(max_branching))
        self.probability_mass_limit = float(max(probability_mass_limit, 0.0))
        self.rollout_alpha = float(min(max(rollout_alpha, 0.0), 1.0))
        super().__init__(
            artifacts,
            persona_hint=persona_hint,
            mc_samples=mc_samples,
            rng=rng,
            engine=engine,
            exploration_constant=exploration_constant,
            rollout_depth=rollout_depth,
        )

    # pylint: disable=arguments-differ
    def __init_subclass__(cls, **kwargs: Any) -> None:  # pragma: no cover - defensive
        return super().__init_subclass__(**kwargs)

    def _filtered_priors(self, priors: Sequence[BNNActionPrior]) -> List[BNNActionPrior]:
        actionable = [prior for prior in priors if not prior.pruned and prior.probability > 0.0]
        if not actionable:
            actionable = [prior for prior in priors if not prior.pruned]
        if not actionable:
            actionable = list(priors)

        selected: List[BNNActionPrior] = []
        cumulative = 0.0
        for prior in actionable:
            selected.append(prior.replicate())
            cumulative += max(prior.probability, 0.0)
            if len(selected) >= self.max_branching:
                break
            if self.probability_mass_limit > 0 and cumulative >= self.probability_mass_limit:
                break

        if not selected and actionable:
            selected.append(actionable[0].replicate())
        return selected

    def _rollout(self, state: GameState, current_player: Optional[str], root_player: str) -> float:
        if current_player is None:
            return self._evaluate_reward(state, root_player)

        depth = self.rollout_depth
        working_state = state
        player = current_player
        while depth > 0 and not working_state.round_over and player is not None:
            context = self.build_context_from_state(self.engine, working_state, player)
            action = self._select_rollout_action(context)
            working_state = self._apply_action(working_state, player, action)
            if working_state.round_over:
                break
            player = working_state.current_player().player_id
            depth -= 1
        return self._evaluate_reward(working_state, root_player)

    def _select_rollout_action(self, context: BNNContext) -> BotAction:
        if not context.priors:
            if context.valid_actions:
                return clone_bot_action(self.rng.choice(context.valid_actions))
            return BotAction(ActionType.DRAW)

        if self.rng.random() < self.rollout_alpha:
            weights = [max(prior.probability, 0.0) for prior in context.priors]
            if sum(weights) <= 0:
                weights = [1.0 for _ in context.priors]
            choice = self.rng.choices(context.priors, weights=weights, k=1)[0]
        else:
            choice = self.rng.choice(context.priors)
        return clone_bot_action(choice.action)

    def search(
        self,
        game_state: GameState,
        bnn_context: BNNContext,
        num_simulations: int,
    ) -> BotAction:
        if num_simulations <= 0:
            num_simulations = 1

        player_id = bnn_context.player_id
        root_team = game_state.team_index_for(player_id)
        root_state = self._determinize_state(game_state, player_id)

        priors = self._filtered_priors(bnn_context.priors)
        if not priors:
            return select_action_from_evaluation(
                bnn_context.evaluation,
                self.bnn_helper,
                self.artifacts,
                game_state,
                player_id,
            )

        root_node = ISMCTSBase.TreeNode(
            player_id=player_id,
            team_index=root_team,
            action=None,
            token=None,
            prior=1.0,
            parent=None,
            context=bnn_context,
        )
        root_node.add_untried(priors)

        actual_simulations = 0
        for _ in range(num_simulations):
            node = root_node
            state = self._clone_state(root_state)
            player_to_move = player_id
            path = [node]

            # Selection
            while not state.round_over and not node.untried_actions and node.children:
                child = self._select_child(node)
                if child is None:
                    break
                action = child.action if child.action is not None else None
                if action is None or player_to_move is None:
                    break
                state = self._apply_action(state, player_to_move, action)
                node = child
                path.append(node)
                player_to_move = state.current_player().player_id if not state.round_over else None

            # Expansion
            if node.untried_actions and not state.round_over and player_to_move is not None:
                prior = node.untried_actions.pop(0)
                action = prior.action
                state = self._apply_action(state, player_to_move, action)
                next_player = state.current_player().player_id if not state.round_over else None

                child_context: Optional[BNNContext] = None
                child_priors: List[BNNActionPrior] = []
                if not state.round_over and next_player is not None:
                    evaluation = self.evaluate_state(self.engine, state, next_player)
                    child_context = self.context_from_evaluation(evaluation, next_player)
                    child_priors = self._filtered_priors(child_context.priors)

                child_team = state.team_index_for(next_player) if next_player is not None else node.team_index
                child_node = ISMCTSBase.TreeNode(
                    player_id=next_player,
                    team_index=child_team,
                    action=action,
                    token=prior.token,
                    prior=prior.probability,
                    parent=node,
                    context=child_context,
                )
                child_node.add_untried(child_priors)
                node.children[prior.token] = child_node
                node = child_node
                path.append(node)
                player_to_move = next_player

            # Simulation / Rollout
            if state.round_over:
                reward = self._evaluate_reward(state, player_id)
            else:
                reward = self._rollout(state, player_to_move, player_id)

            self._backpropagate(path, reward, root_team)
            actual_simulations += 1

        if not root_node.children:
            return select_action_from_evaluation(
                bnn_context.evaluation,
                self.bnn_helper,
                self.artifacts,
                game_state,
                player_id,
            )

        best_child = max(
            root_node.children.values(),
            key=lambda child: (child.visits, child.value_sum / child.visits if child.visits else 0.0),
        )
        recommended = clone_bot_action(best_child.action) if best_child.action is not None else BotAction(ActionType.DRAW)

        prior = bnn_context.token_to_prior.get(best_child.token or "")
        probability = prior.probability if prior is not None else max(best_child.prior, 0.0)
        raw_probability = prior.raw_probability if prior is not None else probability
        std_dev = prior.std_dev if prior is not None else None

        recommended.info.update(
            {
                "bnn_entropy": bnn_context.entropy,
                "bnn_entropy_normalized": bnn_context.normalized_entropy,
                "bnn_mutual_information": bnn_context.mutual_information,
                "persona_hint": self.persona_hint,
                "scenario_type": bnn_context.scenario_type.value,
                "bnn_probability": probability,
                "bnn_probability_raw": raw_probability,
                "bnn_probability_std": std_dev,
                "selection_reason": "ismcts_guided",
                "mi_threshold": self.bnn_helper.mi_explore_threshold,
                "entropy_threshold": self.bnn_helper.entropy_exploit_threshold,
                "mc_samples": bnn_context.mc_samples,
                "candidate_count": len(bnn_context.evaluation.get("candidates", [])),
                "pruned_candidates": int(bnn_context.evaluation.get("pruned_count", 0)),
                "prune_prob_threshold": self.bnn_helper.prune_prob_threshold,
                "prune_entropy_threshold": self.bnn_helper.prune_entropy_threshold,
                "mi_bonus_weight": self.bnn_helper.mi_bonus_weight,
                "probability_normalization": bnn_context.evaluation.get("normalization_kind"),
                "bnn_token": best_child.token,
                "ismcts_simulations": actual_simulations,
                "ismcts_rollout_alpha": self.rollout_alpha,
                "ismcts_max_branching": self.max_branching,
                "ismcts_probability_mass_limit": self.probability_mass_limit,
                "ismcts_exploration_constant": self.exploration_constant,
                "ismcts_rollout_depth": self.rollout_depth,
                "ismcts_visit_count": best_child.visits,
                "ismcts_total_root_visits": root_node.visits,
                "ismcts_mean_value": best_child.value_sum / best_child.visits if best_child.visits else 0.0,
            }
        )

        if recommended.action_type == ActionType.DRAW:
            candidates = bnn_context.evaluation.get("candidates", [])
            has_play_option = any(candidate["action"].action_type == ActionType.PLAY for candidate in candidates)
            recommended.info["forfeits_turn"] = has_play_option

        return recommended


__all__ = [
    "BNNActionPrior",
    "BNNContext",
    "BNNMixin",
    "GuidedISMCTS",
    "ISMCTSBase",
    "build_bnn_context",
    "clone_bot_action",
    "select_action_from_evaluation",
]
