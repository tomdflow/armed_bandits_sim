# Bandit Strategy Lab

A single-file Streamlit teaching simulation of epsilon-greedy, UCB, and Thompson sampling for three ads with click/no-click rewards. All application logic lives in `app.py`.

## Run locally

Use Python 3.13, matching the tested environment and Cloud deployment setting.
From this repository's root:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the localhost address printed by Streamlit. With uv, an alternative is
`uv run --python 3.13 --with-requirements requirements.txt streamlit run app.py`.

## Deploy and maintain

See [DEPLOYMENT.md](DEPLOYMENT.md) for GitHub setup, exact Streamlit Community
Cloud settings, and the update/rollback workflow. Entry point: `app.py`.
No API keys, external datasets, database, or Linux system packages are required.

Run checks before committing changes:

```sh
python -m unittest discover -s tests -v
```

GitHub Actions runs the same tests on pushes and pull requests using Python 3.13
on Linux. Tests cover decision stability, UCB initialization, repeatable runs,
contextual updates, session limits, and Streamlit interactions. The four direct
runtime dependencies are pinned to tested versions; update them deliberately.

## Explore

- Each strategy has its own tab and session history. Its next decision stays fixed until you supply or simulate the outcome.
- Click **Click** or **No click** for a manual outcome, or simulate one or many visitors.
- Thompson sampling shows three separate Beta charts side by side on common axes. The next selected ad has a tinted background and explicit label; diamonds mark the next decision's sampled CTRs, and a dashed line shows the latest updated ad's previous curve.
- **Contextual Thompson** adds new/returning visitor features, three baseline-weight distributions, and three returning-effect distributions. Toggle visitor type without changing the sampled model; record manual or automatic outcomes and download the contextual history. The default marked weights reproduce the worked example, and are labeled as an illustration until you draw a posterior sample.
- UCB shows empirical CTR and its uncertainty bonus. Untried ads are prioritized once; the UCB index can exceed 1.
- Epsilon-greedy shows empirical CTR and whether the next decision is exploration or exploitation. Untried estimates are 0; ties are broken randomly.
- History records every selected ad, click/non-click, decision values, counts before/after, and outcome source. CSV downloads preserve the entire session.
- **Compare all three** adds visitors to every strategy. At the same visitor number, each ad has the same potential outcome across strategies; decision randomness uses independent generators.
- Change settings and apply to start fresh, or reset all histories. Downloads are the only persistence; reset or session loss clears history.

## Classroom notes

The algorithms never access true CTRs. True rates are used only to simulate rewards and evaluate expected regret (best true rate minus selected rate, accumulated). Manual outcomes can contradict those rates; reset before an automatic comparison. Unequal visitor totals should be compared at matching visitor numbers.

The Thompson prior is Beta(1,1), so the posterior mean is `(clicks+1)/(views+2)`, while empirical CTR is `clicks/views`. Updates retain all previous observations. A distribution's height is density; interval probability is area under the curve. Fixed Bernoulli rates are a teaching assumption. Compare multiple seeds, since a single run cannot establish a universally best strategy.

Session limit: 20,000 visitors per strategy. Share `app.py` and `requirements.txt` with a colleague; no credentials, external datasets, or existing repository notebooks are needed.

## Contextual model

For this binary-feature lesson, each ad has independent Beta(1,1) priors over its new-visitor and returning-visitor CTRs. These become linear weights via `baseline = p_new` and `effect = p_return - p_new`. Draw the two CTRs jointly and transform them; the resulting baseline and effect are dependent and must not be sampled independently from their displayed marginal curves. The model yields valid click probabilities for both contexts. Outcomes update only the selected ad/context Beta counts; derived weight marginals update accordingly. This special case does not pool evidence across visitor types.

Baseline curves use exact Beta densities. Effect curves use deterministic grid convolution (2,048 midpoint bins per underlying CTR), so their rendering is a numerical approximation. Actual CTR draws use NumPy's Beta sampler, not the plot grid.

The contextual tab has its own true-rate controls and session history, separate from the regular three-strategy comparison. Global apply/reset also clears contextual history; its tab-specific reset preserves its true rates. Manual outcomes and fixed worked-example decisions are labeled in the CSV.

Linear contextual bandits are used with both [UCB](https://proceedings.mlr.press/v15/chu11a.html) and [Thompson sampling](https://proceedings.mlr.press/v28/agrawal13.html). Richer linear models often use Gaussian weight beliefs; this lesson uses the exact binary-context Bernoulli construction to preserve click-probability interpretation.
