import unittest

from scripts.generate_controller_oracle_dataset import (
    _has_strong_optimal_margins,
    _optimal_stop_step,
    _optimal_values_and_actions,
    _validate_margin_request,
)


class GenerateControllerOracleDatasetTests(unittest.TestCase):
    def test_optimal_values_and_actions_continue_then_halt(self):
        values, actions = _optimal_values_and_actions([-0.5, 0.2, 0.2], 0.001)
        self.assertEqual(actions, [0, 1, 1])
        self.assertAlmostEqual(values[0], 0.199)
        self.assertAlmostEqual(values[1], 0.2)

    def test_strong_margin_filter_rejects_tiny_step_one_gap(self):
        self.assertEqual(_optimal_stop_step([-0.5, 0.0, 0.0], 0.001), 1)
        self.assertFalse(_has_strong_optimal_margins([-0.5, 0.0, 0.0], 0.001, 0.05))

    def test_strong_margin_filter_accepts_clear_decisions(self):
        self.assertEqual(_optimal_stop_step([-0.5, 0.2, 0.2], 0.001), 1)
        self.assertFalse(_has_strong_optimal_margins([-0.5, 0.2, 0.2], 0.001, 0.05))
        self.assertEqual(_optimal_stop_step([-0.5, -0.2, 0.2], 0.001), 2)
        self.assertTrue(_has_strong_optimal_margins([-0.5, -0.2, 0.2], 0.001, 0.05))

    def test_validate_margin_request_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            _validate_margin_request([], continue_cost=0.001, min_decision_margin=0.05)
        with self.assertRaises(ValueError):
            _validate_margin_request([0, 1, 2], continue_cost=-0.1, min_decision_margin=0.05)
        with self.assertRaises(ValueError):
            _validate_margin_request([0, 1, 2], continue_cost=0.1, min_decision_margin=-0.01)


if __name__ == "__main__":
    unittest.main()
