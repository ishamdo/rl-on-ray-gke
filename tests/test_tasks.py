"""Unit tests for the RL Trainer logic (no GPU or Ray cluster needed)."""

import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
import tasks


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is #### 42", "42"),
        ("Reasoning here... #### 125", "125"),
        ("So we get #### -5.5\n", "-5.5"),
        ("Therefore, the answer is \\boxed{16}", "16"),
        ("Using the formula: \\boxed{ 42.5 }.", "42.5"),
        ("We conclude that the answer is 12.", "12"),
        ("No numbers here.", None),
    ]
)
def test_extract_answer(text, expected):
    # Instantiate trainer without loading heavy model for simple test
    # We will test the class method by calling the underlying logic or using a mock trainer
    # Actually, extract_answer is a method on RLTrainer, but it doesn't use self.
    # Let's inspect if we can call it on a dummy class or just test the logic.
    # Let's define a mock class or call the method directly.
    class DummyTrainer:
        extract_answer = tasks.RLTrainer.extract_answer
        reward_fn = tasks.RLTrainer.reward_fn
        
    trainer = DummyTrainer()
    assert trainer.extract_answer(text) == expected


@pytest.mark.parametrize(
    "completion,target,expected_reward",
    [
        ("The final result is #### 50", "The answer is #### 50", 1.0),
        ("We boxed the answer: \\boxed{12}", "#### 12", 1.0),
        ("The count is 5", "#### 10", 0.0),
        ("No answer", "#### 5", 0.0),
        ("The answer is #### 10.5", "#### 10.5", 1.0),
    ]
)
def test_reward_fn(completion, target, expected_reward):
    class DummyTrainer:
        extract_answer = tasks.RLTrainer.extract_answer
        reward_fn = tasks.RLTrainer.reward_fn
        
    trainer = DummyTrainer()
    assert trainer.reward_fn(completion, target) == expected_reward
