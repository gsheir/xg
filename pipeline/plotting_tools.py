from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

from pipeline.settings import OUTPUT_DIR


def plot_curve_against_binned(
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

    # Convert to degrees if feature is angle
    is_angle = feature_name.lower() == "angle"
    X_display = np.degrees(X) if is_angle else X

    # Create range for smooth curve
    x_min, x_max = X.min(), X.max()
    x_range = np.linspace(x_min, x_max, 300)

    # Predict probabilities for the range
    X_range_df = pd.DataFrame({feature_name: x_range})
    y_pred = model.predict(X_range_df)

    # Convert x_range to degrees for display if needed
    x_range_display = np.degrees(x_range) if is_angle else x_range

    # Plot the logistic curve
    ax.plot(x_range_display, y_pred, "b-", linewidth=2, label="Curve vs binned shots")

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
            center = (bins[i] + bins[i + 1]) / 2
            bin_centers.append(np.degrees(center) if is_angle else center)
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

    # Set xlabel with units
    xlabel = feature_name.replace("_", " ").title()
    if is_angle:
        xlabel += " (degrees)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Goal Probability", fontsize=12)
    ax.set_title(f"Curve vs Binned Shots: {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(output_dir) / f"{model_name}_curve_vs_binned_shots.png"
    else:
        save_path = Path(save_path)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return str(save_path)


def plot_curve_against_actual(
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

    # Convert to degrees if feature is angle
    is_angle = feature_name.lower() == "angle"
    X_display = np.degrees(X) if is_angle else X

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
        X_display[goals_mask],
        y_jitter_goals,
        c="green",
        alpha=0.3,
        s=20,
        label="Goals",
        marker="^",
    )
    ax.scatter(
        X_display[non_goals_mask],
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
        X_display[sorted_idx],
        y_pred[sorted_idx],
        "b-",
        linewidth=2,
        label="Predicted Probability",
        alpha=0.8,
    )

    # Set xlabel with units
    xlabel = feature_name.replace("_", " ").title()
    if is_angle:
        xlabel += " (degrees)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Goal Probability / Outcome", fontsize=12)
    ax.set_title(
        f"{feature_name.replace('_', ' ').title()} vs Actual Shot Outcome: {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(output_dir) / f"{model_name}_vs_actual_outcome.png"
    else:
        save_path = Path(save_path)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return str(save_path)


def plot_shots(data, output_dir: str = OUTPUT_DIR):
    """Plot shot locations on a soccer pitch.

    Args:
        data: DataFrame with shot data including x, y, and goal columns
        output_dir: Directory to save the plot

    Returns:
        Path where plot was saved
    """
    if data.empty:
        return None

    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
    fig, ax = pitch.draw(figsize=(10, 7))

    # Plot shots
    goals = data[data["goal"] == 1]
    misses = data[data["goal"] == 0]

    pitch.scatter(
        misses["x"],
        misses["y"],
        ax=ax,
        color="grey",
        s=10,
        label="Miss",
        alpha=0.2,
    )
    pitch.scatter(
        goals["x"],
        goals["y"],
        ax=ax,
        color="red",
        s=20,
        label="Goal",
        alpha=0.8,
    )

    plt.legend(loc="upper right")
    plt.title("Shot Locations")

    save_path = Path(output_dir) / "shot_locations.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    return str(save_path)


def plot_shot_heatmap(data, output_dir: str = OUTPUT_DIR):
    """Plot binned goal probability heatmap on a soccer pitch.

    Args:
        data: DataFrame with shot data including x, y, and goal columns
        output_dir: Directory to save the plot

    Returns:
        Path where plot was saved
    """
    if data.empty:
        return None

    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
    fig, ax = pitch.draw(figsize=(10, 7))

    # Calculate binned goal probability using bin_statistic
    bin_statistic = pitch.bin_statistic(
        data["x"],
        data["y"],
        values=data["goal"],
        statistic="mean",
        bins=(25, 25),  # (x_bins, y_bins)
    )

    # Plot the heatmap
    heatmap = pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Reds",
        alpha=0.8,
    )

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Goal Probability", rotation=270, labelpad=20)

    plt.title("Binned Goal Probability Heatmap")

    save_path = Path(output_dir) / "shot_heatmap.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    return str(save_path)


def plot_model_prediction_heatmap(
    model: Any, model_name: str, features: list, output_dir: str = OUTPUT_DIR
):
    """Plot model's predicted goal probability heatmap across the pitch.

    Args:
        model: Trained model with predict method
        model_name: Name of the model for the filename
        features: List of feature names the model uses
        output_dir: Directory to save the plot

    Returns:
        Path where plot was saved
    """
    import pandas as pd

    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
    fig, ax = pitch.draw(figsize=(10, 7))

    # Create a grid of points across the pitch
    # StatsBomb pitch dimensions: 120 x 80
    x_range = np.linspace(0, 120, 50)
    y_range = np.linspace(0, 80, 50)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Flatten the grid for prediction
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Calculate angle and distance for each point
    from pipeline.data_preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()
    angles = []
    distances = []
    for x, y in zip(x_flat, y_flat):
        angle, distance = preprocessor.calculate_angle_and_distance(x, y)
        angles.append(angle)
        distances.append(distance)

    # Create dataframe with all features
    grid_df = pd.DataFrame({"angle": angles, "distance": distances})

    # Select only the features the model uses
    grid_df = grid_df[features]

    # Get model predictions
    predictions = model.predict(grid_df)

    # Reshape predictions back to grid
    predictions_grid = predictions.reshape(x_grid.shape)

    # Plot the heatmap using imshow
    heatmap = ax.imshow(
        predictions_grid,
        extent=[0, 120, 0, 80],
        origin="lower",
        cmap="Reds",
        alpha=0.8,
        aspect="auto",
    )

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Predicted Goal Probability", rotation=270, labelpad=20)

    plt.title(f"Model Predicted Goal Probability: {model_name}")

    save_path = Path(output_dir) / f"{model_name}_prediction_heatmap.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    return str(save_path)
