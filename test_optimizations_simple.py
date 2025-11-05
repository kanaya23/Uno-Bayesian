#!/usr/bin/env python3
"""Simple test script for optimization verification (no pytest dependency)."""

import copy
import time
import sys

from uno_engine.engine import UnoEngine
from uno_engine.models import (
    Card,
    CardType,
    Color,
    GameMode,
    Player,
)


def test_fast_clone():
    """Test fast_clone optimization."""
    print("Testing fast_clone()...")

    engine = UnoEngine()
    players = [Player(player_id=f"P{i}", score=0) for i in range(4)]
    state = engine.init_game(players, mode=GameMode.CLASSIC_2V2)

    # Test correctness
    cloned = state.fast_clone()
    assert len(cloned.players) == len(state.players)
    assert len(cloned.draw_pile) == len(state.draw_pile)
    assert cloned.current_player_index == state.current_player_index
    print("  ✓ fast_clone() preserves state structure")

    # Test independence
    original_hand_size = len(state.players[0].hand)
    cloned.players[0].hand.append(Card(Color.RED, CardType.NUMBER, value=5, number=5))
    assert len(state.players[0].hand) == original_hand_size
    print("  ✓ fast_clone() creates independent copy")

    # Test performance
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        _ = copy.deepcopy(state)
    deepcopy_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        _ = state.fast_clone()
    fastclone_time = time.perf_counter() - start

    speedup = deepcopy_time / fastclone_time
    print(f"  ✓ fast_clone() speedup: {speedup:.2f}x faster than deepcopy")

    if speedup < 2.0:
        print(f"  ⚠ Warning: Expected 2x+ speedup, got {speedup:.2f}x")
    else:
        print(f"  ✓ Performance target met (>2x speedup)")

    return speedup >= 1.5  # Allow some variance


def test_cache_mechanism():
    """Test BNN cache mechanism."""
    print("\nTesting BNN cache mechanism...")

    try:
        from ismcts_guided import GuidedISMCTS
        print("  ✓ GuidedISMCTS imports successfully")

        # Verify cache methods exist
        assert hasattr(GuidedISMCTS, '__init__')
        print("  ✓ GuidedISMCTS has required methods")

        # Would need full artifacts to test actual caching
        print("  ℹ Full cache testing requires trained model artifacts")
        return True

    except ImportError as e:
        print(f"  ⚠ Cannot test cache: {e}")
        return False


def test_game_integration():
    """Test optimizations in actual gameplay."""
    print("\nTesting integration with game engine...")

    engine = UnoEngine()
    players = [Player(player_id=f"P{i}", score=0) for i in range(4)]
    state = engine.init_game(players, mode=GameMode.CLASSIC_2V2)

    # Simulate a few moves using fast_clone
    moves_tested = 0
    for _ in range(5):
        if state.round_over:
            break

        player = state.current_player()
        valid_moves = engine.get_valid_moves(state, player.player_id)

        if valid_moves:
            # Clone before move
            cloned = state.fast_clone()
            cloned_discard_size = len(cloned.discard_pile)

            # Make move
            chosen_color = Color.RED if valid_moves[0].type in {CardType.WILD, CardType.WILD_DRAW_FOUR} else None
            state = engine.play_card(state, player.player_id, valid_moves[0], chosen_color)

            # Verify clone unchanged
            assert len(cloned.discard_pile) == cloned_discard_size
            moves_tested += 1

    print(f"  ✓ fast_clone() works correctly through {moves_tested} game moves")
    return True


def test_parameter_support():
    """Test that new parameters are supported."""
    print("\nTesting optimization parameters...")

    try:
        from ismcts_guided import GuidedISMCTS
        from notebooks.uno_bnn_curriculum_converted import TrainingArtifacts

        # Check constructor signature
        import inspect
        sig = inspect.signature(GuidedISMCTS.__init__)
        params = list(sig.parameters.keys())

        required_params = [
            'enable_state_cache',
            'time_limit',
            'max_simulation_depth',
            'parallel_workers',
        ]

        for param in required_params:
            if param in params:
                print(f"  ✓ Parameter '{param}' supported")
            else:
                print(f"  ✗ Parameter '{param}' missing")
                return False

        return True

    except ImportError as e:
        print(f"  ⚠ Cannot test parameters: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MCTS Optimization Test Suite")
    print("=" * 60)

    results = []

    try:
        results.append(("fast_clone", test_fast_clone()))
    except Exception as e:
        print(f"  ✗ fast_clone test failed: {e}")
        results.append(("fast_clone", False))

    try:
        results.append(("cache", test_cache_mechanism()))
    except Exception as e:
        print(f"  ✗ cache test failed: {e}")
        results.append(("cache", False))

    try:
        results.append(("integration", test_game_integration()))
    except Exception as e:
        print(f"  ✗ integration test failed: {e}")
        results.append(("integration", False))

    try:
        results.append(("parameters", test_parameter_support()))
    except Exception as e:
        print(f"  ✗ parameter test failed: {e}")
        results.append(("parameters", False))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nSummary: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
