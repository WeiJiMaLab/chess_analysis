import unittest

from scripts.controller_synthetic_diagnostic import SCENARIOS, SyntheticHaltEnv, _optimal_stop_step


class ControllerSyntheticDiagnosticTests(unittest.TestCase):
    def test_oracle_stop_steps_match_named_scenarios(self):
        for scenario in SCENARIOS.values():
            self.assertEqual(
                _optimal_stop_step(scenario.halt_rewards, scenario.continue_cost),
                scenario.expected_stop_step,
            )

    def test_always_halt_scenario_halt_is_better_than_continue(self):
        scenario = SCENARIOS["always-halt"]
        env = SyntheticHaltEnv(scenario.halt_rewards, scenario.continue_cost)
        env.reset()
        halt_result = env.step(1)
        self.assertTrue(halt_result.done)
        self.assertAlmostEqual(halt_result.reward, scenario.halt_rewards[0])

        env.reset()
        continue_result = env.step(0)
        self.assertFalse(continue_result.done)
        self.assertLess(continue_result.reward, halt_result.reward)

    def test_always_continue_scenario_reaches_final_step(self):
        scenario = SCENARIOS["always-continue"]
        env = SyntheticHaltEnv(scenario.halt_rewards, scenario.continue_cost)
        env.reset()
        result = None
        for _ in range(len(scenario.halt_rewards) - 1):
            result = env.step(0)
            self.assertFalse(result.done)
        result = env.step(1)
        self.assertIsNotNone(result)
        self.assertTrue(result.done)
        self.assertEqual(result.info["expansions"], len(scenario.halt_rewards) - 1)

    def test_continue_once_scenario_oracle_prefers_step_one(self):
        scenario = SCENARIOS["continue-once"]
        self.assertEqual(_optimal_stop_step(scenario.halt_rewards, scenario.continue_cost), 1)

    def test_continue_twice_scenario_oracle_prefers_step_two(self):
        scenario = SCENARIOS["continue-twice"]
        self.assertEqual(_optimal_stop_step(scenario.halt_rewards, scenario.continue_cost), 2)

    def test_halt_late_drop_scenario_stops_before_drop(self):
        scenario = SCENARIOS["halt-late-drop"]
        self.assertEqual(_optimal_stop_step(scenario.halt_rewards, scenario.continue_cost), 1)

    def test_borderline_halt_scenario_prefers_immediate_halt(self):
        scenario = SCENARIOS["borderline-halt"]
        self.assertEqual(_optimal_stop_step(scenario.halt_rewards, scenario.continue_cost), 0)

    def test_small_gains_always_continue_scenario_reaches_end(self):
        scenario = SCENARIOS["small-gains-always-continue"]
        self.assertEqual(_optimal_stop_step(scenario.halt_rewards, scenario.continue_cost), 3)


if __name__ == "__main__":
    unittest.main()
