from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch
from sklearn.calibration import calibration_curve

from pipeline.settings import OUTPUT_DIR


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Plot calibration curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        n_bins: Number of bins for calibration
        save_path: Path to save the plot (if None, auto-generated)

    Returns:
        Path where plot was saved
    """
    fig, ax = plt.subplots()

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=model_name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(f"Calibration Curve: {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(output_dir) / f"{model_name}_calibration_curve.png"
    else:
        save_path = Path(save_path)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return str(save_path)


def plot_logistic_curve(
    model: Any,
    feature_name: str,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Plot logistic regression curve showing the fitted model.

    Args:
        model: Trained model
        feature_name: Name of the feature being plotted
        X: Feature values (1D array)
        y: True labels
        model_name: Name of the model
        save_path: Path to save the plot (if None, auto-generated)

    Returns:
        Path where plot was saved
    """
    import pandas as pd

    fig, ax = plt.subplots()

    # Create range for smooth curve
    x_min, x_max = X.min(), X.max()
    x_range = np.linspace(x_min, x_max, 300)

    # Predict probabilities for the range
    X_range_df = pd.DataFrame({feature_name: x_range})
    y_pred = model.predict(X_range_df)

    # Plot the logistic curve
    ax.plot(x_range, y_pred, "b-", linewidth=2, label="Logistic Curve")

    # Calculate binned actual rates for overlay
    n_bins = 20
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_indices = np.digitize(X, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = []
    actual_rates = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 5:  # Only show bins with enough samples
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            actual_rates.append(y[mask].mean())

    # Plot actual rates as scatter points
    ax.scatter(
        bin_centers,
        actual_rates,
        color="red",
        s=50,
        alpha=0.6,
        label="Actual Goal Rate (binned)",
        zorder=5,
    )

    ax.set_xlabel(feature_name.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Goal Probability", fontsize=12)
    ax.set_title(f"Logistic Curve: {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(output_dir) / f"{model_name}_logistic_curve.png"
    else:
        save_path = Path(save_path)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return str(save_path)


def plot_feature_vs_probability(
    model: Any,
    feature_name: str,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Plot feature values against predicted goal probabilities with actual outcomes.

    Args:
        model: Trained model
        feature_name: Name of the feature being plotted
        X: Feature values (1D array)
        y: True labels (0 or 1)
        model_name: Name of the model
        save_path: Path to save the plot (if None, auto-generated)

    Returns:
        Path where plot was saved
    """
    import pandas as pd

    fig, ax = plt.subplots()

    # Get predictions
    X_df = pd.DataFrame({feature_name: X})
    y_pred = model.predict(X_df)

    # Separate goals and non-goals
    goals_mask = y == 1
    non_goals_mask = y == 0

    # Plot scatter points with jitter for better visibility
    jitter = 0.02
    y_jitter_goals = y[goals_mask] + np.random.uniform(
        -jitter, jitter, goals_mask.sum()
    )
    y_jitter_non_goals = y[non_goals_mask] + np.random.uniform(
        -jitter, jitter, non_goals_mask.sum()
    )

    ax.scatter(
        X[goals_mask],
        y_jitter_goals,
        c="green",
        alpha=0.3,
        s=20,
        label="Goals",
        marker="^",
    )
    ax.scatter(
        X[non_goals_mask],
        y_jitter_non_goals,
        c="red",
        alpha=0.2,
        s=20,
        label="Non-Goals",
        marker="v",
    )

    # Plot predicted probabilities as a line
    sorted_idx = np.argsort(X)
    ax.plot(
        X[sorted_idx],
        y_pred[sorted_idx],
        "b-",
        linewidth=2,
        label="Predicted Probability",
        alpha=0.8,
    )

    ax.set_xlabel(feature_name.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Goal Probability / Outcome", fontsize=12)
    ax.set_title(
        f"{feature_name.replace('_', ' ').title()} vs Goal Probability: {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(output_dir) / f"{model_name}_feature_vs_probability.png"
    else:
        save_path = Path(save_path)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return str(save_path)


def plot_shots(data):
    """Plot shot locations on a soccer pitch."""
    if data.empty:
        return

    pitch = Pitch(pitch_type="statsbomb", pitch_color="grass", line_color="white")
    fig, ax = pitch.draw(figsize=(10, 7), dpi=300)

    # Plot shots
    goals = data[data["goal"] == 1]
    misses = data[data["goal"] == 0]

    pitch.scatter(
        goals["x"],
        goals["y"],
        ax=ax,
        color="green",
        edgecolors="black",
        s=100,
        label="Goal",
        alpha=0.7,
    )
    pitch.scatter(
        misses["x"],
        misses["y"],
        ax=ax,
        color="red",
        edgecolors="black",
        s=100,
        label="Miss",
        alpha=0.7,
    )

    plt.legend(loc="upper right")
    plt.title("Shot Locations")

    plt.savefig(f"{OUTPUT_DIR}/shot_locations.png")
    plt.close()
