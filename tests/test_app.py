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

    def test_formula_table_uses_frozen_thompson_draw(self):
        run = app.new_run(42, 2)
        run["counts"][:] = [4, 3, 1]
        run["clicks"][:] = [1, 1, 0]
        decision = app.choose(run, "Thompson sampling", CONFIG)
        rng_before = repr(run["rng"].bit_generator.state)
        table = app.table_for(run, decision, "Thompson sampling", CONFIG)
        self.assertEqual(table.iloc[0]["α = 1 + clicks"], "1 + 1 = 2")
        self.assertEqual(table.iloc[0]["β = 1 + non-clicks"], "1 + 3 = 4")
        self.assertEqual(table.iloc[0]["Mean = α / (α + β)"], "2 / (2 + 4) = 0.3333")
        for i in range(3):
            self.assertAlmostEqual(float(table.iloc[i]["Next decision value"]), decision["scores"][i], places=4)
        self.assertEqual(repr(run["rng"].bit_generator.state), rng_before)
        self.assertIs(run["pending"], decision)

    def test_ucb_table_resolves_score_and_handles_untried_arms(self):
        run = app.new_run(42, 1)
        table = app.table_for(run, app.choose(run, "UCB", CONFIG), "UCB", CONFIG)
        self.assertTrue(all(table["Next decision value"] == "∞ (untried: priority)"))
        run["counts"][:] = [4, 3, 1]
        run["clicks"][:] = [1, 1, 0]
        run["history"] = [None] * 8
        run["pending"] = None
        table = app.table_for(run, app.choose(run, "UCB", CONFIG), "UCB", CONFIG)
        self.assertEqual(table.iloc[0]["Next visitor (t)"], 9)
        self.assertEqual(table.iloc[0]["CTR calculation"], "1 / 4 = 0.2500")
        self.assertEqual(table.iloc[0]["Decision step"], "0.2500 + 1.0481 = 1.2981")
        self.assertEqual(table.iloc[0]["Next decision value"], "1.2981")

    def test_epsilon_table_explains_both_branches_and_old_sessions(self):
        for epsilon, branch in [(0.0, "u ≥ ε"), (1.0, "u < ε")]:
            config = dict(CONFIG, epsilon=epsilon)
            run = app.new_run(42, 0)
            decision = app.choose(run, "Epsilon-greedy", config)
            draw = decision["exploration_draw"]
            table = app.table_for(run, decision, "Epsilon-greedy", config)
            self.assertTrue(all(table["Shared random draw (u)"] == f"{draw:.4f}"))
            self.assertTrue(all(table["Decision step"].str.startswith(branch)))
            del decision["exploration_draw"]
            old_table = app.table_for(run, decision, "Epsilon-greedy", config)
            self.assertTrue(all(old_table["Shared random draw (u)"] == "Recorded on next decision"))
            self.assertIs(app.choose(run, "Epsilon-greedy", config), decision)

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
        ctx_table = next(df.value for df in at.dataframe if "Returning feature (x)" in df.value.columns)
        self.assertEqual(ctx_table.iloc[0]["Returning feature (x)"], 1)
        self.assertEqual(ctx_table.iloc[0]["Contribution = x × θ₁"], "1 × (+0.0800) = +0.0800")
        self.assertEqual(ctx_table.iloc[0]["Next decision value"], "0.1700")
        at.radio(key="contextual_type").set_value("New").run()
        self.assertEqual(len(at.exception), 0)
        ctx_table = next(df.value for df in at.dataframe if "Returning feature (x)" in df.value.columns)
        self.assertEqual(ctx_table.iloc[0]["Next decision value"], "0.0900")
        self.assertEqual(ctx_table.iloc[1]["Next visitor"], "← selected")
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
