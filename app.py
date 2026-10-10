"""Interactive multi-armed bandits lesson. Run: streamlit run app.py.

All simulation logic is in this file. Rewards are Bernoulli (click/no click).
No real visitor data is collected; histories live in server-side Streamlit session state.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARMS = ("Ad A", "Ad B", "Ad C")
STRATEGIES = ("Epsilon-greedy", "UCB", "Thompson sampling")
COLORS = ("#2563eb", "#e07818", "#8b5cf6")
MAX_VISITORS = 20_000


def new_run(seed: int, strategy_index: int) -> dict:
    return {"counts": np.zeros(3, dtype=int), "clicks": np.zeros(3, dtype=int),
            "history": [], "rng": np.random.default_rng(np.random.SeedSequence([seed, strategy_index, 31])),
            "pending": None, "previous": None}


def choose(run: dict, strategy: str, config: dict) -> dict:
    """Freeze a decision until its visitor's outcome is recorded."""
    if run["pending"] is not None:
        return run["pending"]
    counts, clicks, rng = run["counts"], run["clicks"], run["rng"]
    means = np.divide(clicks, counts, out=np.zeros(3), where=counts > 0)
    bonus = np.zeros(3)
    if strategy == "Epsilon-greedy":
        scores = means.copy()
        explore = rng.random() < config["epsilon"]
        if explore:
            arm = int(rng.integers(3))
            reason = "Random exploration (epsilon coin chose explore)"
        else:
            arm = int(rng.choice(np.flatnonzero(means == means.max())))
            reason = "Exploit highest observed CTR (random tie-break if tied)"
    elif strategy == "UCB":
        untried = np.flatnonzero(counts == 0)
        tried = counts > 0
        t = len(run["history"]) + 1
        bonus[tried] = config["ucb_scale"] * np.sqrt(2 * np.log(max(2, t)) / counts[tried])
        scores = means + bonus
        scores[~tried] = np.inf
        if len(untried):
            arm = int(untried[0])
            reason = "Initialization: try each ad once"
        else:
            arm = int(rng.choice(np.flatnonzero(scores == scores.max())))
            reason = "Highest observed CTR + uncertainty bonus"
    else:
        scores = rng.beta(clicks + 1, counts - clicks + 1)
        arm = int(np.argmax(scores))
        reason = "Highest sampled plausible CTR"
    run["pending"] = {"arm": arm, "scores": scores, "means": means,
                      "bonus": bonus, "reason": reason}
    return run["pending"]


def record(run: dict, strategy: str, config: dict, outcome: int | None = None) -> None:
    if len(run["history"]) >= MAX_VISITORS:
        return
    decision = choose(run, strategy, config)
    arm = decision["arm"]
    visitor = len(run["history"]) + 1
    manual = outcome is not None
    if not manual:
        # Same potential outcome per visitor/ad across strategies. Reward RNG is
        # separate from decision RNG, so epsilon/Thompson draws cannot affect it.
        env = np.random.default_rng(np.random.SeedSequence([config["seed"], visitor, 17]))
        uniforms = env.random(3)
        outcome = int(uniforms[arm] < config["rates"][arm])
    if outcome not in (0, 1):
        raise ValueError("A visitor outcome must be 0 or 1.")
    run["previous"] = {"counts": run["counts"].copy(), "clicks": run["clicks"].copy(), "arm": arm}
    before_n, before_c = int(run["counts"][arm]), int(run["clicks"][arm])
    run["counts"][arm] += 1
    run["clicks"][arm] += outcome
    regret = float(max(config["rates"]) - config["rates"][arm])
    previous_regret = run["history"][-1]["Cumulative expected regret"] if run["history"] else 0.0
    row = {"Visitor": visitor, "Ad": ARMS[arm], "Outcome": "Click" if outcome else "No click",
           "Click (0/1)": outcome, "Source": "Manual" if manual else "Simulated",
           "Decision": decision["reason"], "Views before": before_n, "Clicks before": before_c,
           "Views after": int(run["counts"][arm]), "Clicks after": int(run["clicks"][arm]),
           "CTR after": float(run["clicks"][arm] / run["counts"][arm]),
           "Beta alpha after": int(run["clicks"][arm] + 1),
           "Beta beta after": int(run["counts"][arm] - run["clicks"][arm] + 1),
           "Cumulative clicks": int(run["clicks"].sum()),
           "Cumulative expected regret": previous_regret + regret}
    for i, label in enumerate(ARMS):
        row[f"{label} decision value"] = float(decision["scores"][i])
    run["history"].append(row)
    run["pending"] = None


def beta_density(clicks: int, nonclicks: int, x: np.ndarray) -> np.ndarray:
    """Beta(1+clicks, 1+nonclicks) density, computed in log space."""
    normalizer = math.lgamma(clicks + nonclicks + 2) - math.lgamma(clicks + 1) - math.lgamma(nonclicks + 1)
    return np.exp(normalizer + clicks * np.log(x) + nonclicks * np.log1p(-x))


def plot_layout(fig: go.Figure, x_title: str, y_title: str, height: int = 370) -> go.Figure:
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, height=height,
                      margin=dict(l=30, r=20, t=25, b=45),
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    return fig


def decision_chart(run: dict, strategy: str, decision: dict) -> go.Figure:
    fig = go.Figure()
    if strategy == "Thompson sampling":
        return beta_arm_chart(run, decision, decision["arm"])
    fig.add_trace(go.Bar(x=ARMS, y=decision["means"], name="Observed CTR", marker_color=COLORS))
    if strategy == "UCB":
        fig.add_trace(go.Bar(x=ARMS, y=decision["bonus"], name="Uncertainty bonus", marker_color="#9ca3af"))
        fig.update_layout(barmode="stack")
        for i in np.flatnonzero(run["counts"] == 0):
            fig.add_annotation(x=ARMS[i], y=0, text="Untried: priority", showarrow=False, yshift=18)
        y_title = "Decision score (CTR + bonus)"
    else:
        y_title = "Observed click-through rate"
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return plot_layout(fig, "Ad", y_title)


def new_contextual_run(seed: int) -> dict:
    """Exact binary-context Bernoulli model, expressed as linear weights.

    p_new and p_return have independent Beta priors. Linear weights are
    (baseline=p_new, effect=p_return-p_new), so the weights are dependent.
    """
    return {"counts": np.zeros((3, 2), dtype=int), "clicks": np.zeros((3, 2), dtype=int),
            "history": [], "rng": np.random.default_rng(np.random.SeedSequence([seed, 91])),
            "pending": {"weights": np.array([[.09, .08], [.11, .02], [.07, .03]]),
                        "source": "Worked-example illustration"},
            "previous": None, "seed": seed,
            "rates": np.array([[.09, .17], [.11, .13], [.07, .10]])}


def contextual_draw(run: dict) -> dict:
    if run["pending"] is None:
        rates = run["rng"].beta(run["clicks"] + 1, run["counts"] - run["clicks"] + 1)
        run["pending"] = {"weights": np.column_stack([rates[:, 0], rates[:, 1] - rates[:, 0]]),
                          "source": "Posterior sample"}
    return run["pending"]


def contextual_decision(run: dict, returning: int) -> dict:
    draw = contextual_draw(run)
    weights = draw["weights"]
    scores = weights @ np.array([1, returning])
    return {"weights": weights, "scores": scores, "arm": int(np.argmax(scores)), "source": draw["source"]}


def record_contextual(run: dict, returning: int, outcome: int | None = None) -> None:
    if len(run["history"]) >= MAX_VISITORS:
        return
    if returning not in (0, 1):
        raise ValueError("Context must be new (0) or returning (1).")
    decision = contextual_decision(run, returning)
    arm, visitor = decision["arm"], len(run["history"]) + 1
    manual = outcome is not None
    if not manual:
        env = np.random.default_rng(np.random.SeedSequence([run["seed"], visitor, 92]))
        outcome = int(env.random() < run["rates"][arm, returning])
    if outcome not in (0, 1):
        raise ValueError("A visitor outcome must be 0 or 1.")
    run["previous"] = {"counts": run["counts"].copy(), "clicks": run["clicks"].copy(), "arm": arm}
    run["counts"][arm, returning] += 1
    run["clicks"][arm, returning] += outcome
    weights = decision["weights"]
    row = {"Visitor": visitor, "Visitor type": "Returning" if returning else "New", "Ad": ARMS[arm],
           "Outcome": "Click" if outcome else "No click", "Click (0/1)": outcome,
           "Source": "Manual" if manual else "Simulated", "Weight source": decision["source"],
           "Sampled baseline": float(weights[arm, 0]), "Sampled returning effect": float(weights[arm, 1]),
           "Sampled score": float(decision["scores"][arm]),
           "Views after (ad + context)": int(run["counts"][arm, returning]),
           "Clicks after (ad + context)": int(run["clicks"][arm, returning])}
    for i, label in enumerate(ARMS):
        row[f"{label} sampled baseline"] = float(weights[i, 0])
        row[f"{label} sampled effect"] = float(weights[i, 1])
        row[f"{label} sampled score"] = float(decision["scores"][i])
    run["history"].append(row)
    run["pending"] = None


def contextual_marginals(counts: np.ndarray, clicks: np.ndarray, bins: int = 2048) -> tuple:
    """Deterministic quadrature of marginals; sampling itself uses exact Betas."""
    x = (np.arange(bins) + .5) / bins
    masses = []
    for context in (0, 1):
        c, n = int(clicks[context]), int(counts[context])
        # Normalize in log space, even with many observations near 0 or 1.
        log_mass = c * np.log(x) + (n-c) * np.log1p(-x)
        mass = np.exp(log_mass - log_mass.max())
        masses.append(mass / mass.sum())
    effect_x = np.arange(-(bins-1), bins) / bins
    effect_density = np.convolve(masses[1], masses[0][::-1]) * bins
    baseline_x = np.unique(np.concatenate([x, np.geomspace(.000001, .1, 300),
                                           1-np.geomspace(.000001, .1, 300)]))
    c, n = int(clicks[0]), int(counts[0])
    return baseline_x, beta_density(c, n-c, baseline_x), effect_x, effect_density


def highlight_chart(fig: go.Figure, selected: bool) -> go.Figure:
    if selected:
        fig.update_layout(plot_bgcolor="rgba(59,130,246,0.12)", paper_bgcolor="rgba(59,130,246,0.06)")
    return fig


def beta_arm_chart(run: dict, decision: dict, arm: int, y_max: float | None = None) -> go.Figure:
    x = np.unique(np.concatenate([np.linspace(.00001, .99999, 1500),
                                  np.geomspace(.000001, .1, 300), 1-np.geomspace(.000001, .1, 300)]))
    c, n = int(run["clicks"][arm]), int(run["counts"][arm])
    fig = go.Figure(go.Scatter(x=x, y=beta_density(c, n-c, x), mode="lines", name=f"Beta({c+1}, {n-c+1})",
                              line=dict(color=COLORS[arm]), hovertemplate="CTR: %{x:.2%}<br>Density: %{y:.3f}<extra></extra>"))
    sample = float(decision["scores"][arm])
    clipped = np.clip(sample, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    height = float(beta_density(c, n-c, np.array([clipped]))[0])
    fig.add_trace(go.Scatter(x=[sample], y=[height], mode="markers", name=f"Draw: {sample:.2%}",
                            marker=dict(size=12, color=COLORS[arm], symbol="diamond"),
                            hovertemplate="Next visitor's draw: %{x:.2%}<extra></extra>"))
    before = run["previous"]
    if before is not None and before["arm"] == arm:
        old_c, old_n = int(before["clicks"][arm]), int(before["counts"][arm])
        fig.add_trace(go.Scatter(x=x, y=beta_density(old_c, old_n-old_c, x), mode="lines",
                                name="Before last visitor", line=dict(color="#808080", dash="dash")))
    plot_layout(fig, "Possible true CTR", "Density", 340)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", nticks=4)
    if y_max is not None:
        fig.update_yaxes(range=[0, y_max])
    return highlight_chart(fig, decision["arm"] == arm)


def show_beta_panels(run: dict, decision: dict) -> None:
    figures = [beta_arm_chart(run, decision, i) for i in range(3)]
    # Common y scale makes the relative concentration directly comparable.
    y_max = max(float(np.max(t.y)) for f in figures for t in f.data) * 1.12
    for i, col in enumerate(st.columns(3)):
        with col:
            label = f"{ARMS[i]} — NEXT VISITOR" if decision["arm"] == i else ARMS[i]
            st.markdown(f"**{label}**")
            figures[i].update_yaxes(range=[0, y_max])
            st.plotly_chart(figures[i], width="stretch", key=f"beta_arm_{i}")
            st.caption(f"Draw: {decision['scores'][i]:.2%} · {int(run['clicks'][i])} clicks / {int(run['counts'][i])} views")
    st.caption("Blue-tinted panel = ad that receives your next Click / No click outcome. Diamonds = sampled CTRs. All three panels use the same scales.")


def weight_chart(run: dict, decision: dict, arm: int, weight: int) -> go.Figure:
    x, baseline, effect_x, effect = contextual_marginals(run["counts"][arm], run["clicks"][arm])
    values, density = (x, baseline) if weight == 0 else (effect_x, effect)
    fig = go.Figure(go.Scatter(x=values, y=density, mode="lines", name="Current belief",
                              line=dict(color=COLORS[arm]), hovertemplate="Weight: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>"))
    sample = float(decision["weights"][arm, weight])
    sample_y = float(np.interp(sample, values, density))
    fig.add_trace(go.Scatter(x=[sample], y=[sample_y], mode="markers", name=f"Draw: {sample:+.3f}",
                            marker=dict(size=11, color=COLORS[arm], symbol="diamond"),
                            hovertemplate="Sampled weight: %{x:.3f}<extra></extra>"))
    before = run["previous"]
    if before is not None and before["arm"] == arm:
        old = contextual_marginals(before["counts"][arm], before["clicks"][arm])
        old_x, old_y = (old[0], old[1]) if weight == 0 else (old[2], old[3])
        fig.add_trace(go.Scatter(x=old_x, y=old_y, mode="lines", name="Before last visitor",
                                line=dict(color="#808080", dash="dash")))
    plot_layout(fig, "Baseline CTR" if weight == 0 else "Returning effect (CTR difference)", "Density", 290)
    fig.update_xaxes(range=[0, 1] if weight == 0 else [-1, 1], tickformat=".0%", nticks=4)
    return highlight_chart(fig, decision["arm"] == arm)


def show_contextual(show_truth: bool) -> None:
    if "contextual" not in st.session_state:
        st.session_state.contextual = new_contextual_run(st.session_state.config["seed"])
    run = st.session_state.contextual
    st.subheader("Contextual Thompson: sample weights, then use the visitor's features")
    st.write("Each ad gets a sampled baseline and a sampled returning effect. Switching visitor type keeps those weights fixed; only the input changes.")
    st.latex(r"\text{sampled score}_i=\widetilde\theta_{i,0}+\mathrm{returning}\times\widetilde\theta_{i,1}")
    visitor = st.radio("Next visitor type", ["New", "Returning"], index=1, horizontal=True, key="contextual_type")
    returning = int(visitor == "Returning")
    buttons = st.columns(2)
    if buttons[0].button("Draw another plausible model", key="ctx_redraw"):
        run["pending"] = None
        contextual_draw(run)
    if buttons[1].button("Use the worked-example weights", key="ctx_example"):
        run["pending"] = {"weights": np.array([[.09, .08], [.11, .02], [.07, .03]]),
                          "source": "Worked-example illustration"}
    decision = contextual_decision(run, returning)
    if decision["source"] == "Worked-example illustration":
        st.info("The marked weights are the fixed teaching example (A: 0.09 + 0.08; B: 0.11 + 0.02). Use 'Draw another plausible model' to sample from the current beliefs. The example does not change those beliefs.")
    st.success(f"Next visitor: {visitor.lower()} → show {ARMS[decision['arm']]} (sampled score {decision['scores'][decision['arm']]:.2%}). This is the ad your next outcome updates.")
    rows = []
    for i, label in enumerate(ARMS):
        b, e = decision["weights"][i]
        rows.append({"Ad": label, "Sampled baseline": f"{b:.3f}", "Sampled returning effect": f"{e:+.3f}",
                     "Calculation": f"{b:.3f} + {returning} × ({e:+.3f}) = {decision['scores'][i]:.3f}",
                     "Next visitor": "← selected" if i == decision["arm"] else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Distributions behind the sampled weights")
    st.caption("Each column is one ad; each row is one weight. Diamonds mark the weights used above. Blue tint identifies the selected ad. Effect is a change in probability: +0.08 means +8 percentage points.")
    for weight, heading in [(0, "Baseline-weight distributions"), (1, "Returning-effect distributions")]:
        st.markdown(f"**{heading}**")
        figures = [weight_chart(run, decision, i, weight) for i in range(3)]
        y_max = max(float(np.max(t.y)) for f in figures for t in f.data) * 1.12
        for i, col in enumerate(st.columns(3)):
            with col:
                st.markdown(f"**{ARMS[i]}{' — SELECTED' if i == decision['arm'] else ''}**")
                figures[i].update_yaxes(range=[0, y_max])
                st.plotly_chart(figures[i], width="stretch", key=f"ctx_weight_{weight}_{i}")
    with st.expander("How these weight distributions are learned"):
        st.write("This is an exact click model for the special case of one binary feature (new or returning). Each ad learns a new-visitor CTR and a returning-visitor CTR, each with a Beta(1,1) starting prior. It draws those two CTRs, then sets baseline = new CTR and returning effect = returning CTR − new CTR.")
        st.write("The plotted effect distribution is the distribution of that difference. The two weight distributions are marginals of a joint belief: baseline and effect are related, so they are not drawn independently. Their sum always gives a valid returning-visitor CTR. Curves for the effect use deterministic numerical approximation; CTR sampling uses the exact Beta models.")
        st.write("A click or non-click updates only the selected ad's model for that visitor type. A new visitor changes the baseline belief and therefore the derived effect belief; a returning visitor updates the returning CTR and therefore the effect belief. The dashed curve shows the previous marginal for the last updated ad.")
        st.write("With richer features, linear Thompson implementations often use Gaussian regression models; Gaussian weights and linear scores are not automatically valid click probabilities. This binary-feature example avoids that issue and preserves the baseline + effect arithmetic.")
        st.markdown("Linear contextual bandits also have UCB strategies: [LinUCB / linear contextual UCB](https://proceedings.mlr.press/v15/chu11a.html) and [linear Thompson sampling](https://proceedings.mlr.press/v28/agrawal13.html) are both established approaches.")
    st.write("**Record this visitor's outcome**")
    c1, c2, c3 = st.columns(3)
    full = len(run["history"]) >= MAX_VISITORS
    if c1.button("Click", key="ctx_click", disabled=full):
        record_contextual(run, returning, 1)
        st.rerun()
    if c2.button("No click", key="ctx_miss", disabled=full):
        record_contextual(run, returning, 0)
        st.rerun()
    if c3.button("Simulate 1 visitor", key="ctx_auto", disabled=full):
        record_contextual(run, returning)
        st.rerun()
    batch = st.select_slider("Automatic contextual visitors", options=[10, 50, 100, 500, 1000], value=100, key="ctx_batch")
    mixed = st.checkbox("Alternate new and returning visitors in the batch", value=True, key="ctx_mixed")
    if st.button(f"Simulate {batch} contextual visitors", key="ctx_run", disabled=full):
        for _ in range(min(batch, MAX_VISITORS-len(run["history"]))):
            context = len(run["history"]) % 2 if mixed else returning
            record_contextual(run, context)
        st.rerun()
    if full:
        st.info("Session limit reached. Download history before resetting.")
    if show_truth:
        st.caption("Contextual simulator's true rates: " + " · ".join(f"{a}: new {run['rates'][i,0]:.0%}, returning {run['rates'][i,1]:.0%}" for i, a in enumerate(ARMS)) + ". These are separate from the regular bandit settings.")
    with st.expander("Contextual simulator settings"):
        with st.form("ctx_settings"):
            rates = np.zeros((3, 2))
            for i, col in enumerate(st.columns(3)):
                with col:
                    st.markdown(f"**{ARMS[i]}**")
                    for j, label in enumerate(("New", "Returning")):
                        rates[i,j] = st.slider(f"{label} true CTR (%)", 0, 100, int(round(run["rates"][i,j]*100)), key=f"ctx_rate_{i}_{j}") / 100
            apply = st.form_submit_button("Apply contextual rates & reset this tab")
        if apply:
            fresh = new_contextual_run(st.session_state.config["seed"])
            fresh["rates"] = rates
            st.session_state.contextual = fresh
            st.rerun()
    if st.button("Reset contextual history", key="ctx_reset"):
        fresh = new_contextual_run(st.session_state.config["seed"])
        fresh["rates"] = run["rates"].copy()
        st.session_state.contextual = fresh
        st.rerun()
    st.caption("This tab has its own history. Applying the global settings or resetting all histories also resets it. Download CSV before resetting.")
    st.subheader("Contextual click history")
    if not run["history"]:
        st.info("Record an outcome to see learning for the selected ad and visitor type.")
        return
    last = run["history"][-1]
    st.write(f"**Last visitor #{last['Visitor']}: {last['Visitor type']} → {last['Ad']} → {last['Outcome']}**")
    st.caption(f"For that ad and context: {last['Clicks after (ad + context)']} clicks / {last['Views after (ad + context)']} views.")
    history = pd.DataFrame(run["history"])
    st.dataframe(history[["Visitor", "Visitor type", "Ad", "Outcome", "Source", "Sampled baseline", "Sampled returning effect", "Sampled score"]].iloc[::-1], hide_index=True, width="stretch", height=320)
    st.download_button("Download contextual history (CSV)", history.to_csv(index=False).encode(), "contextual_thompson_history.csv", "text/csv", key="ctx_download")


def table_for(run: dict, decision: dict) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(ARMS):
        n, c = int(run["counts"][i]), int(run["clicks"][i])
        score = decision["scores"][i]
        rows.append({"Ad": name, "Views": n, "Clicks": c, "Non-clicks": n-c,
                     "Observed CTR": f"{c/n:.2%}" if n else "No observations",
                     "Beta belief": f"Beta({c+1}, {n-c+1})",
                     "Beta mean": f"{(c+1)/(n+2):.2%}",
                     "Next decision value": "Untried: priority" if np.isinf(score) else f"{score:.4f}",
                     "Next visitor": "← selected" if decision["arm"] == i else ""})
    return pd.DataFrame(rows)


def apply_settings(config: dict) -> None:
    st.session_state.config = config
    st.session_state.runs = {name: new_run(config["seed"], i) for i, name in enumerate(STRATEGIES)}
    st.session_state.contextual = new_contextual_run(config["seed"])


def show_strategy(strategy: str, show_truth: bool) -> None:
    config = st.session_state.config
    run = st.session_state.runs[strategy]
    decision = choose(run, strategy, config)
    prefix = strategy.replace(" ", "_")
    if strategy == "Epsilon-greedy":
        st.write(f"With probability {config['epsilon']:.0%}, choose an ad uniformly at random. Otherwise choose the highest observed CTR.")
        st.caption("Untried ads start with an estimate of 0; ties are broken randomly. A random exploration choice can also pick the current leader.")
    elif strategy == "UCB":
        st.write("Choose the largest observed CTR + uncertainty bonus. First, show each ad once.")
        st.latex(r"\mathrm{score}_i=\widehat p_i+c\sqrt{\frac{2\ln(\max(2,t))}{n_i}}")
        st.caption("t is the upcoming visitor number; n is that ad's views. The score is an optimistic index and can exceed 1; it is not a probability.")
    else:
        st.write("Draw one plausible click rate for each ad, then choose the largest draw. The diamonds show the next visitor's draws.")
        st.caption("Horizontal axis: possible true CTR. Higher density means more belief per unit of CTR; interval probability is area under the curve. The curves each have total area 1. The dashed line shows the last updated ad before its latest update.")
        st.latex(r"p_i\mid\mathrm{history}\sim\operatorname{Beta}(1+\mathrm{clicks}_i,\;1+\mathrm{nonclicks}_i)")
    st.subheader(f"Next visitor: show {ARMS[decision['arm']]}")
    st.caption(decision["reason"] + ". This choice stays fixed until you record an outcome.")
    if strategy == "Thompson sampling":
        show_beta_panels(run, decision)
    else:
        st.plotly_chart(decision_chart(run, strategy, decision), width="stretch", key=f"chart_{prefix}")
    st.dataframe(table_for(run, decision), hide_index=True, width="stretch")
    if show_truth:
        st.caption("Simulator's true rates: " + " · ".join(f"{a}: {p:.1%}" for a, p in zip(ARMS, config["rates"])) + ". The algorithms never read these rates.")
    st.write("**Record the next visitor**")
    c1, c2, c3 = st.columns(3)
    full = len(run["history"]) >= MAX_VISITORS
    if c1.button("Click", key=f"click_{prefix}", disabled=full):
        record(run, strategy, config, 1)
        st.rerun()
    if c2.button("No click", key=f"miss_{prefix}", disabled=full):
        record(run, strategy, config, 0)
        st.rerun()
    if c3.button("Simulate 1 visitor", key=f"auto_{prefix}", disabled=full):
        record(run, strategy, config)
        st.rerun()
    batch = st.select_slider("Automatic visitors", options=[10, 50, 100, 500, 1000], value=100, key=f"batch_{prefix}")
    if st.button(f"Simulate {batch} visitors", key=f"run_{prefix}", disabled=full):
        for _ in range(min(batch, MAX_VISITORS-len(run["history"]))):
            record(run, strategy, config)
        st.rerun()
    if full:
        st.info("Session limit reached (20,000 visitors per strategy). Download history before resetting.")
    st.subheader("Click history")
    if not run["history"]:
        st.info("Record a click or non-click to see the first update.")
        return
    latest = run["history"][-1]
    st.write(f"**Last visitor #{latest['Visitor']}: {latest['Ad']} → {latest['Outcome']}** ({latest['Source'].lower()})")
    st.caption(f"For that ad: {latest['Clicks before']}/{latest['Views before']} → {latest['Clicks after']}/{latest['Views after']} clicks/views. All previous observations remain in the update.")
    total = len(run["history"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Visitors", f"{total:,}")
    m2.metric("Clicks", int(run["clicks"].sum()))
    m3.metric("Overall observed CTR", f"{run['clicks'].sum()/total:.2%}")
    history = pd.DataFrame(run["history"])
    display = history[["Visitor", "Ad", "Outcome", "Source", "Views after", "Clicks after", "CTR after", "Decision"]].copy()
    display["CTR after"] = display["CTR after"].map(lambda p: f"{p:.2%}")
    st.dataframe(display.iloc[::-1], hide_index=True, width="stretch", height=320)
    st.caption("Newest visitor first. CTR after is cumulative for the selected ad, not the latest batch's CTR.")
    st.download_button("Download complete history (CSV)", history.to_csv(index=False).encode(),
                       file_name=f"{prefix.lower()}_history.csv", mime="text/csv", key=f"download_{prefix}")


def show_comparison(show_truth: bool) -> None:
    config, runs = st.session_state.config, st.session_state.runs
    st.write("Run all three strategies through the same simulated visitor stream. Each strategy learns only from the ad it actually shows.")
    st.caption("At matching visitor numbers, an ad has the same potential click outcome for every strategy. Decision randomness is separate. No strategy is guaranteed to win any one run.")
    batch = st.select_slider("Visitors to add to each strategy", options=[10, 100, 500, 1000, 5000], value=100, key="compare_batch")
    if st.button("Run all three", type="primary", key="compare_run"):
        for name in STRATEGIES:
            run = runs[name]
            for _ in range(min(batch, MAX_VISITORS-len(run["history"]))):
                record(run, name, config)
        st.rerun()
    if st.button("Reset all histories", key="reset_all"):
        apply_settings(config)
        st.rerun()
    st.caption("Resetting or applying settings clears session histories. Download them first if you want to keep them.")
    rows, combined = [], []
    has_manual = any(row["Source"] == "Manual" for r in runs.values() for row in r["history"])
    if has_manual:
        st.warning("These histories include manually supplied outcomes. For a comparison generated entirely from the simulated rates, reset and use automatic visitors.")
    totals = [len(r["history"]) for r in runs.values()]
    if len(set(totals)) > 1:
        st.info("Strategies currently have different visitor totals. Compare curves at the same visitor number, or reset for a fresh equal-length run.")
    fig_clicks, fig_regret, fig_alloc = go.Figure(), go.Figure(), go.Figure()
    for i, name in enumerate(STRATEGIES):
        run = runs[name]
        n, c = len(run["history"]), int(run["clicks"].sum())
        row = {"Strategy": name, "Visitors": n, "Clicks": c, "Observed CTR": f"{c/n:.2%}" if n else "—"}
        if show_truth:
            row["Expected regret"] = round(run["history"][-1]["Cumulative expected regret"], 3) if n else 0.0
        rows.append(row)
        h = pd.DataFrame(run["history"])
        if n:
            fig_clicks.add_trace(go.Scatter(x=h["Visitor"], y=h["Cumulative clicks"], name=name, mode="lines", line=dict(color=COLORS[i])))
            fig_regret.add_trace(go.Scatter(x=h["Visitor"], y=h["Cumulative expected regret"], name=name, mode="lines", line=dict(color=COLORS[i])))
            h.insert(0, "Strategy", name)
            combined.append(h)
        fig_alloc.add_trace(go.Bar(x=ARMS, y=run["counts"], name=name, marker_color=COLORS[i]))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if combined:
        st.plotly_chart(plot_layout(fig_clicks, "Visitors", "Cumulative actual clicks"), width="stretch", key="compare_clicks")
        st.plotly_chart(plot_layout(fig_alloc, "Ad", "Visitors assigned"), width="stretch", key="compare_alloc")
        if show_truth:
            st.plotly_chart(plot_layout(fig_regret, "Visitors", "Cumulative expected clicks forgone"), width="stretch", key="compare_regret")
            st.caption("Expected regret adds (best true CTR − selected ad's true CTR) each visitor. Lower is better. It measures selection quality, rather than unlucky clicks, and uses true rates only for evaluation.")
        combined_df = pd.concat(combined, ignore_index=True)
        st.download_button("Download all strategies' histories (CSV)", combined_df.to_csv(index=False).encode(),
                           "all_bandit_histories.csv", "text/csv", key="download_all")


def main() -> None:
    st.set_page_config(page_title="Bandit Strategy Lab", page_icon="🎰", layout="wide")
    st.title("Bandit Strategy Lab")
    st.write("Choose an ad, observe a click or non-click, and watch the algorithm learn. Try one visitor at a time or compare thousands of simulated visitors.")
    if "config" not in st.session_state:
        apply_settings({"rates": [.10, .20, .05], "epsilon": .10, "ucb_scale": 1.0, "seed": 42})
    with st.sidebar:
        st.header("Simulation settings")
        with st.form("settings"):
            rates = [st.slider(f"{a} true CTR (%)", 0, 100, int(round(st.session_state.config["rates"][i]*100)), key=f"rate_{i}") / 100 for i, a in enumerate(ARMS)]
            epsilon = st.slider("Epsilon: random exploration", 0.0, 1.0, st.session_state.config["epsilon"], .01)
            ucb_scale = st.slider("UCB uncertainty multiplier", 0.0, 3.0, st.session_state.config["ucb_scale"], .1)
            seed = st.number_input("Random seed", min_value=0, max_value=2**32-1, value=st.session_state.config["seed"], step=1)
            submitted = st.form_submit_button("Apply settings & reset all")
        if submitted:
            apply_settings({"rates": rates, "epsilon": epsilon, "ucb_scale": ucb_scale, "seed": int(seed)})
            st.rerun()
        st.caption("True rates generate automatic outcomes. Manual Click / No click buttons let you supply your own outcomes. Rates stay fixed during a run.")
        show_truth = st.checkbox("Reveal true rates and expected regret", value=True)
        st.caption("Thompson sampling starts with Beta(1,1) for every ad. Each session has its own history; download CSV to keep or share it.")
    tabs = st.tabs([*STRATEGIES, "Contextual Thompson", "Compare all three", "How to use this in class"])
    for tab, strategy in zip(tabs[:3], STRATEGIES):
        with tab:
            show_strategy(strategy, show_truth)
    with tabs[3]:
        show_contextual(show_truth)
    with tabs[4]:
        show_comparison(show_truth)
    with tabs[5]:
        st.markdown("""
### A short classroom experiment
1. In **Thompson sampling**, record a click, then a non-click. Watch how the curve and Beta counts change. The next visitor's draws may change even when the curves barely move.
2. In **UCB**, simulate the first three visitors. Each ad gets a turn; then the algorithm uses the CTR plus its uncertainty bonus.
3. In **Epsilon-greedy**, compare epsilon = 0, 0.1, and 1 in separate fresh runs. Zero can stop exploration too early; one keeps choosing randomly.
4. In **Contextual Thompson**, switch New / Returning with the worked-example weights. B wins for a new visitor; A wins for a returning visitor. Then draw a new model and record outcomes to see the weight distributions update.
5. In **Compare all three**, reset, simulate 1,000 visitors, and inspect allocation and expected regret. Repeat with different seeds and closer true CTRs. One lucky run does not prove a strategy is best.

### What is a Beta distribution?
It is a curve over possible click rates from 0 to 1. A flat Beta(1,1) prior treats every rate as equally plausible. Each click adds 1 to alpha; each non-click adds 1 to beta. Greater accumulated evidence usually narrows the curve. Its height is density, not the probability of one exact click rate.

The Beta mean is `(clicks + 1) / (views + 2)`. Observed CTR is `clicks / views`. These differ because the Beta model includes the starting prior. A new click is one extra observation; a tiny batch with a high CTR never replaces the accumulated history.

### Reading the three strategies
- **Epsilon-greedy:** use the highest empirical CTR most of the time; choose randomly with probability epsilon.
- **UCB:** use an optimistic score that adds an uncertainty bonus to empirical CTR. This score can exceed 100%.
- **Thompson sampling:** sample a plausible CTR from each posterior, then choose the largest. Broad uncertain distributions sometimes produce winning draws; confidently poor ads rarely win.

All three balance exploration and exploitation differently. Thompson sampling and UCB both adapt exploration to evidence; neither always beats the other. This lesson assumes independent clicks and stationary true rates.

### Share or run
Share `app.py` and install `streamlit`, `numpy`, `pandas`, and `plotly` (or use the included requirements file), then run `streamlit run app.py`. The app uses no external data or credentials. Histories are stored in the current Streamlit session, not written to disk automatically.

Implementation references: [Streamlit charts](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart) · [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
""")


if __name__ == "__main__":
    main()
