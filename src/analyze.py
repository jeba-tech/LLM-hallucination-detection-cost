"""Aggregate results/raw_runs.csv into per-(method, model, n) accuracy vs cost,
then plot the cost-accuracy Pareto frontier.

Run from src/: python analyze.py

Three accuracy figures are reported per configuration:
  f1            - at the detector's a-priori threshold (the shipped operating point)
  f1_best       - best F1 reachable by sweeping the threshold post hoc
  auroc         - threshold-free ranking quality, from the continuous score
`f1` is the honest headline number; `f1_best` shows how much of a method's
weakness is the threshold rather than the signal. Reporting only `f1_best`
would be tuning on the test set, so both are kept side by side.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import config

MARKERS = {
    "verbalized_confidence": "o",
    "llm_judge": "s",
    "self_consistency": "^",
    "semantic_entropy": "D",
}

# Which direction of the raw score indicates hallucination. Self-consistency
# agreement and verbalized confidence are inverted (high = trustworthy);
# entropy is not (high = disagreement = hallucination). llm_judge emits a bare
# verdict with no continuous score, so it is absent here.
HIGHER_MEANS_HALLUCINATION = {
    "verbalized_confidence": False,
    "self_consistency": False,
    "semantic_entropy": True,
}


def _oriented_score(method: str, scores: pd.Series) -> pd.Series | None:
    """Flip scores so that larger always means 'more likely hallucinated'."""
    if method not in HIGHER_MEANS_HALLUCINATION or scores.isna().all():
        return None
    return scores if HIGHER_MEANS_HALLUCINATION[method] else -scores


def _best_f1_over_thresholds(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Best F1 achievable at any threshold on this score."""
    best = 0.0
    for threshold in np.unique(scores):
        best = max(best, f1_score(y_true, scores >= threshold, zero_division=0))
    return best


def _auroc_ci(
    y_true: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap 95% CI for AUROC.

    Load-bearing at this sample size: point estimates here look decisive but
    mostly have intervals straddling 0.5, and reporting AUROC alone would
    invite reading noise as a finding.
    """
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], scores[idx]))
    if not boots:
        return float("nan"), float("nan")
    return tuple(np.percentile(boots, [2.5, 97.5]))


def summarize() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_RUNS_PATH)

    rows = []
    for (method, model, n), group in df.groupby(["method", "model", "n_samples"]):
        y_true = group["label"].astype(int).to_numpy()
        y_pred = group["prediction"].astype(int).to_numpy()

        f1 = f1_score(y_true, y_pred, zero_division=0)

        auroc, f1_best = float("nan"), float("nan")
        auroc_lo, auroc_hi = float("nan"), float("nan")
        oriented = _oriented_score(method, group["score"])
        both_classes_present = len(np.unique(y_true)) > 1
        if oriented is not None and both_classes_present:
            scores = oriented.to_numpy()
            auroc = roc_auc_score(y_true, scores)
            f1_best = _best_f1_over_thresholds(y_true, scores)
            if len(np.unique(scores)) > 1:
                auroc_lo, auroc_hi = _auroc_ci(y_true, scores)

        rows.append(
            {
                "method": method,
                "model": model,
                "n_samples": n,
                "f1": f1,
                "f1_best": f1_best,
                "auroc": auroc,
                "auroc_lo": auroc_lo,
                "auroc_hi": auroc_hi,
                # Does the CI exclude chance? The only claim the data supports.
                "beats_chance": bool(auroc_lo > 0.5) if auroc_lo == auroc_lo else False,
                "total_cost_usd": group["cost_usd"].sum(),
                "avg_latency_s": group["latency_s"].mean(),
                "flag_rate": y_pred.mean(),
                "hallucination_rate": y_true.mean(),
                "n_questions": len(group),
            }
        )

    return pd.DataFrame(rows).sort_values(["method", "model", "n_samples"])


def plot_pareto(summary: pd.DataFrame) -> None:
    """Cost vs. AUROC with bootstrap intervals and an explicit chance line.

    Deliberately *not* plotted against F1. At the shipped thresholds F1 ranks
    self-consistency highest, which is an artifact of it flagging ~75% of
    answers against a 28% base rate; a reader would take the opposite of the
    supported conclusion from that figure. AUROC is threshold-free, and the
    error bars show directly that only one configuration clears chance.
    """
    scored = summary.dropna(subset=["auroc_lo"])
    models = sorted(scored["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        subset = scored[scored["model"] == model]

        for method, group in subset.groupby("method"):
            group = group.sort_values("total_cost_usd")
            significant = group["beats_chance"]
            ax.errorbar(
                group["total_cost_usd"],
                group["auroc"],
                yerr=[
                    group["auroc"] - group["auroc_lo"],
                    group["auroc_hi"] - group["auroc"],
                ],
                marker=MARKERS.get(method, "x"),
                markersize=9,
                capsize=4,
                linewidth=1.2,
                label=method,
                alpha=0.9,
            )
            # ring the configurations whose interval actually clears chance
            if significant.any():
                ax.scatter(
                    group.loc[significant, "total_cost_usd"],
                    group.loc[significant, "auroc"],
                    s=260,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.6,
                    zorder=5,
                )
            for _, row in group.iterrows():
                if row["n_samples"] > 1:
                    ax.annotate(
                        f"N={int(row['n_samples'])}",
                        (row["total_cost_usd"], row["auroc"]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=7,
                        alpha=0.75,
                    )

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2)
        ax.text(
            0.99, 0.5, " chance", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, style="italic",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Total cost, USD (simulated, whole run)")
        n_q = int(subset["n_questions"].max())
        ax.set_title(f"{model}  (n={n_q})")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("AUROC (threshold-free), 95% bootstrap CI")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Hallucination detection on TruthfulQA: cost vs. detection quality\n"
        "circled = 95% CI excludes chance; LLM-as-judge omitted (binary verdict, no score)",
        fontsize=11,
    )
    fig.tight_layout()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PARETO_PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {config.PARETO_PLOT_PATH}")


def main():
    summary = summarize()

    shown = summary.drop(columns=["hallucination_rate"])
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    rate = summary["hallucination_rate"].mean()
    print(f"\nBase rate: {rate:.1%} of graded answers were labelled hallucinations.")
    print("(F1 for a detector that flags everything would be "
          f"{2 * rate / (1 + rate):.3f}; beat this, or the method is noise.)")

    scored = summary.dropna(subset=["auroc_lo"])
    winners = scored[scored["beats_chance"]]
    print(f"\nConfigurations whose AUROC 95% CI excludes chance: "
          f"{len(winners)}/{len(scored)}")
    if winners.empty:
        print("None. At this sample size the experiment cannot distinguish any "
              "detector from random guessing; differences below are not "
              "interpretable as findings.")
    else:
        for _, r in winners.iterrows():
            print(f"  {r['method']} / {r['model']} / N={int(r['n_samples'])}: "
                  f"AUROC {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}]")

    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    plot_pareto(summary)


if __name__ == "__main__":
    main()
