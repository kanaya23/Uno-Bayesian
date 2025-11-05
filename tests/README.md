# UNO Game Engine Test Suite

This directory contains comprehensive tests for all features in the UNO game engine repository. These tests ensure bugs like the "3-card drawing bug" don't happen again.

## Test Files

### Core Game Engine Tests

1. **`test_deck.py`** (40 tests)
   - Standard deck construction and composition
   - Go Wild deck construction with Discard All cards
   - Deck shuffling operations
   - Card burying mechanics
   - Edge cases and validation

2. **`test_models.py`** (43 tests)
   - Card model creation and validation
   - Player model and hand management
   - Pending action states
   - Game state creation and manipulation
   - Fast cloning for MCTS performance
   - Enumerations (Color, CardType, GameMode, etc.)
   - Edge cases with large hands and high scores

3. **`test_engine.py`** (16 tests - existing)
   - Game initialization
   - Team management
   - Initial card effects (Skip, Reverse, Draw Two, Wild)
   - Go Wild mode features (Discard All, Draw stacking)
   - Round completion and scoring

4. **`test_card_drawing.py`** (20 tests)
   - **REGRESSION TESTS** for the 3-card drawing bug
   - Initial dealing (exactly 7 cards per player)
   - Single card drawing (exactly 1 card)
   - Draw stack mechanics (Draw Two, Wild Draw Four)
   - Card conservation throughout gameplay
   - Edge cases (skip, reverse, empty pile)

5. **`test_game_mechanics.py`** (12 tests)
   - Card playability rules (color/number matching)
   - Special card effects (Skip, Reverse, Draw Two)
   - Wild card handling and color selection
   - Turn progression and wrapping
   - Winning conditions and point scoring
   - Invalid move detection

### Optional Tests (Training-Related)

6. **`test_bnn_encoders.py`** (existing)
   - BNN state and action encoders
   - *Requires training dependencies (numpy, pyro, torch)*

7. **`test_console_helpers.py`** (created but skipped)
   - Console display formatting functions
   - *Skipped when training dependencies not available*

8. **`test_optimizations.py`** (existing)
   - Performance optimizations
   - *May require training dependencies*

## Running Tests

### Run All Core Tests (No Training Dependencies)
```bash
pytest tests/ --ignore=tests/test_bnn_encoders.py --ignore=tests/test_optimizations.py --ignore=tests/test_console_helpers.py -v
```

### Run Specific Test File
```bash
pytest tests/test_deck.py -v
pytest tests/test_card_drawing.py -v
pytest tests/test_game_mechanics.py -v
```

### Run Tests with Coverage
```bash
pytest tests/ --ignore=tests/test_bnn_encoders.py --cov=uno_engine --cov-report=html
```

### Run Regression Tests Only
```bash
pytest tests/test_card_drawing.py -v
```

## Test Coverage Summary

- **Total Tests**: 131 core tests (excluding training-related tests)
- **Coverage Areas**:
  - ✅ Deck construction and manipulation
  - ✅ Card and player models
  - ✅ Game state management
  - ✅ Card drawing mechanics (**REGRESSION PROTECTED**)
  - ✅ Game rules and mechanics
  - ✅ Turn progression
  - ✅ Winning conditions
  - ✅ Edge cases and error handling

## Key Features Tested

### Bug Prevention
- **Card Drawing Bug**: Multiple tests ensure drawing always adds exactly 1 card (not 3)
- **Initial Dealing**: Tests verify each player gets exactly 7 cards at start
- **Card Conservation**: Tests ensure cards are never duplicated or lost

### Game Mechanics
- Color and number matching rules
- Wild card color selection
- Skip card behavior (advance by 2)
- Reverse card direction changes
- Draw Two and Wild Draw Four effects
- Turn progression and wrapping (P3 → P0)

### Data Integrity
- Fast cloning for MCTS (independent copies)
- Team score tracking
- Visible hands (player + teammate)
- Round and game winning conditions

## Continuous Integration

These tests should be run:
1. Before every commit
2. In CI/CD pipeline
3. After any engine or model changes
4. Before releasing new versions

## Adding New Tests

When adding features:
1. Add tests to appropriate test file
2. For new files, add to this README
3. Ensure tests are independent and can run in any order
4. Use descriptive test names explaining what is tested
5. Include docstrings explaining the test purpose

## Test Structure

All tests follow this structure:
```python
class TestFeatureName:
    """Tests for specific feature."""
    
    def test_specific_behavior(self) -> None:
        """Clear description of what is tested."""
        # Arrange
        # Act
        # Assert
```

## Dependencies

Core tests require only:
- Python 3.12+
- pytest
- uno_engine package (local)

Training-related tests additionally require:
- numpy
- pyro-ppl
- torch

## Notes

- Tests avoid training code to keep them fast and dependency-light
- Each test is independent and creates its own game state
- Tests use fixed random seeds for reproducibility
- Comprehensive coverage prevents regression of known bugs
