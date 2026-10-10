"""Learning behavior and Streamlit interactions; run with unittest."""
import unittest
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
import app

CONFIG = {"rates": [.10, .20, .05], "epsilon": .10, "ucb_scale": 1.0, "seed": 42}

class SimulationTests(unittest.TestCase):
    def test_pending_decision_stays_fixed_until_outcome(self):
        for i, name in enumerate(app.STRATEGIES):
            run = app.new_run(42, i)
            decision = app.choose(run, name, CONFIG)
            self.assertIs(decision, app.choose(run, name, CONFIG))
            arm = decision["arm"]
            app.record(run, name, CONFIG, 1)
            self.assertEqual(run["counts"][arm], 1)
            self.assertEqual(run["clicks"][arm], 1)
            self.assertIsNone(run["pending"])

    def test_ucb_tries_every_arm(self):
        run = app.new_run(42, 1)
        for arm in range(3):
            self.assertEqual(app.choose(run, "UCB", CONFIG)["arm"], arm)
            app.record(run, "UCB", CONFIG, 0)
        np.testing.assert_array_equal(run["counts"], [1, 1, 1])

    def test_same_seed_reproduces_history(self):
        for i, name in enumerate(app.STRATEGIES):
            first, second = app.new_run(42, i), app.new_run(42, i)
            for _ in range(100):
                app.record(first, name, CONFIG)
                app.record(second, name, CONFIG)
            self.assertEqual(first["history"], second["history"])
            self.assertEqual(int(first["counts"].sum()), 100)
            self.assertEqual(int(first["clicks"].sum()), sum(r["Click (0/1)"] for r in first["history"]))

    def test_context_switch_preserves_model_and_changes_example_winner(self):
        run = app.new_contextual_run(42)
        new = app.contextual_decision(run, 0)
        returning = app.contextual_decision(run, 1)
        self.assertEqual(new["arm"], 1)
        self.assertEqual(returning["arm"], 0)
        np.testing.assert_array_equal(new["weights"], returning["weights"])
        app.record_contextual(run, 1, 1)
        self.assertEqual(run["clicks"][0, 1], 1)
        self.assertEqual(run["counts"][0, 0], 0)
        for _ in range(100):
            decision = app.contextual_decision(run, 1)
            self.assertTrue(np.all((decision["scores"] >= 0) & (decision["scores"] <= 1)))
            app.record_contextual(run, 1)

    def test_session_limit_preserves_history(self):
        run = app.new_run(42, 0)
        run["history"] = [None] * app.MAX_VISITORS
        app.record(run, app.STRATEGIES[0], CONFIG)
        self.assertEqual(len(run["history"]), app.MAX_VISITORS)
        self.assertEqual(int(run["counts"].sum()), 0)
        ctx = app.new_contextual_run(42)
        ctx["history"] = [None] * app.MAX_VISITORS
        app.record_contextual(ctx, 0)
        self.assertEqual(int(ctx["counts"].sum()), 0)

class InterfaceTests(unittest.TestCase):
    def test_boot_manual_contextual_comparison_and_reset(self):
        at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        at.button(key="click_Thompson_sampling").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.session_state["runs"]["Thompson sampling"]["history"]), 1)
        at.button(key="ctx_auto").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.session_state["contextual"]["history"]), 1)
        at.button(key="compare_run").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.session_state["runs"]["UCB"]["history"]), 100)
        at.button(key="reset_all").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(all(not r["history"] for r in at.session_state["runs"].values()))
        self.assertFalse(at.session_state["contextual"]["history"])
        # A separate viewer starts with an independent session.
        other = AppTest.from_file(str(Path(app.__file__)), default_timeout=30).run()
        self.assertEqual(len(other.exception), 0)
        self.assertTrue(all(not r["history"] for r in other.session_state["runs"].values()))

if __name__ == "__main__":
    unittest.main()
