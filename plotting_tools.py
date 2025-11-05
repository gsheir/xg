import matplotlib.pyplot as plt
import numpy as np

from settings import OUTPUT_DIR


def plot_models(data, angle_model, distance_model):
    """Plot the logistic regression models."""
    if data.empty or angle_model is None or distance_model is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot angle model
    angles = np.linspace(data["angle"].min(), data["angle"].max(), 100)
    prob_angle = angle_model.predict(angles.reshape(-1, 1))

    ax1.scatter(data["angle"], data["goal"], alpha=0.5, s=20)
    ax1.plot(angles, prob_angle, "r-", linewidth=2, label="Logistic Regression")
    ax1.set_xlabel("Angle to Goal (radians)")
    ax1.set_ylabel("Goal Probability")
    ax1.set_title("xG Model: Angle Only")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot distance model
    distances = np.linspace(data["distance"].min(), data["distance"].max(), 100)
    prob_distance = distance_model.predict(distances.reshape(-1, 1))

    ax2.scatter(data["distance"], data["goal"], alpha=0.5, s=20)
    ax2.plot(distances, prob_distance, "r-", linewidth=2, label="Logistic Regression")
    ax2.set_xlabel("Distance to Goal")
    ax2.set_ylabel("Goal Probability")
    ax2.set_title("xG Model: Distance Only")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/xg_models.png")
    plt.close()
