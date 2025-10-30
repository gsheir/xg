from settings import COMPETITION_ID, SEASON_IDS
from data_loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import math


def calculate_angle_and_distance(x, y):
    """
    Calculate angle and distance to goal from shot position.
    Goal is at (120, 40) with posts at (120, 36) and (120, 44).
    """
    # Goal coordinates (center of goal)
    goal_x, goal_y = 120, 40

    # Distance to goal center
    distance = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)

    # Calculate angle to goal
    # Use the goal posts to calculate the angle
    post1_y, post2_y = 36, 44  # Goal posts

    # Calculate angles to each post
    angle1 = math.atan2(post1_y - y, goal_x - x)
    angle2 = math.atan2(post2_y - y, goal_x - x)

    # Angle is the absolute difference between the two angles
    angle = abs(angle1 - angle2)

    return distance, angle


def get_shots_data(force_refresh: bool = False):
    """Fetch all shots data from the specified seasons using cached data."""
    data_manager = DataLoader()

    # Show cache info
    cache_info = data_manager.get_cache_info()
    if not cache_info.empty:
        print("Current cache status:")
        print(cache_info.to_string(index=False))
        print()

    # Get shots data (will use cache if available)
    shots_df = data_manager.get_shots_data(
        competition_id=COMPETITION_ID,
        season_ids=SEASON_IDS,
        force_refresh=force_refresh,
    )

    return shots_df


def prepare_xg_data(shots_df):
    """Prepare shots data for xG modeling."""
    if shots_df.empty:
        print("No shots data available")
        return pd.DataFrame()

    # Extract shot coordinates
    def extract_coordinates(loc):
        if loc is None:
            return None, None
        try:
            # Handle different formats: list, tuple, or numpy array
            if hasattr(loc, "__len__") and len(loc) >= 2:
                return float(loc[0]), float(loc[1])
            else:
                return None, None
        except (TypeError, IndexError, ValueError):
            return None, None

    coords = shots_df["location"].apply(extract_coordinates)
    shots_df[["x", "y"]] = pd.DataFrame(coords.tolist(), index=shots_df.index)

    # Remove shots without location data
    shots_df = shots_df.dropna(subset=["x", "y"])

    # Calculate distance and angle
    shots_df[["distance", "angle"]] = shots_df.apply(
        lambda row: calculate_angle_and_distance(row["x"], row["y"]),
        axis=1,
        result_type="expand",
    )

    # Create binary goal variable
    shots_df["goal"] = (shots_df["shot_outcome"] == "Goal").astype(int)

    # Remove penalty shots for more realistic xG
    shots_df = shots_df[shots_df["shot_type"] != "Penalty"]

    return shots_df[
        ["distance", "angle", "goal", "x", "y", "shot_outcome", "season_name"]
    ].copy()


def create_logistic_models(data):
    """Create logistic regression models for angle and distance."""
    if data.empty:
        print("No data available for modeling")
        return None, None

    print(f"Training models with {len(data)} shots")
    print(f"Goals: {data['goal'].sum()}, Non-goals: {len(data) - data['goal'].sum()}")

    # Model 1: Angle only
    X_angle = data[["angle"]].values
    y = data["goal"].values

    model_angle = LogisticRegression(random_state=42)
    model_angle.fit(X_angle, y)

    # Model 2: Distance only
    X_distance = data[["distance"]].values

    model_distance = LogisticRegression(random_state=42)
    model_distance.fit(X_distance, y)

    return model_angle, model_distance


def evaluate_models(data, model_angle, model_distance):
    """Evaluate the logistic regression models."""
    if data.empty or model_angle is None or model_distance is None:
        return

    X_angle = data[["angle"]].values
    X_distance = data[["distance"]].values
    y = data["goal"].values

    # Predictions
    pred_angle = model_angle.predict_proba(X_angle)[:, 1]
    pred_distance = model_distance.predict_proba(X_distance)[:, 1]

    # Evaluation
    print("\n=== MODEL EVALUATION ===")
    print(f"Angle Model AUC: {roc_auc_score(y, pred_angle):.3f}")
    print(f"Distance Model AUC: {roc_auc_score(y, pred_distance):.3f}")

    # Model coefficients
    print(f"\nAngle Model Coefficient: {model_angle.coef_[0][0]:.3f}")
    print(f"Distance Model Coefficient: {model_distance.coef_[0][0]:.3f}")

    return pred_angle, pred_distance


def plot_models(data, model_angle, model_distance):
    """Plot the logistic regression models."""
    if data.empty or model_angle is None or model_distance is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot angle model
    angles = np.linspace(data["angle"].min(), data["angle"].max(), 100)
    prob_angle = model_angle.predict_proba(angles.reshape(-1, 1))[:, 1]

    ax1.scatter(data["angle"], data["goal"], alpha=0.5, s=20)
    ax1.plot(angles, prob_angle, "r-", linewidth=2, label="Logistic Regression")
    ax1.set_xlabel("Angle to Goal (radians)")
    ax1.set_ylabel("Goal Probability")
    ax1.set_title("xG Model: Angle Only")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot distance model
    distances = np.linspace(data["distance"].min(), data["distance"].max(), 100)
    prob_distance = model_distance.predict_proba(distances.reshape(-1, 1))[:, 1]

    ax2.scatter(data["distance"], data["goal"], alpha=0.5, s=20)
    ax2.plot(distances, prob_distance, "r-", linewidth=2, label="Logistic Regression")
    ax2.set_xlabel("Distance to Goal")
    ax2.set_ylabel("Goal Probability")
    ax2.set_title("xG Model: Distance Only")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


def main():
    plt.ioff()
    print("Fetching shots data...")
    shots_df = get_shots_data()
    

    if shots_df.empty:
        print("No shots data found!")
        return

    print("Preparing data for xG modeling...")
    xg_data = prepare_xg_data(shots_df)

    if xg_data.empty:
        print("No valid shots data for modeling!")
        return

    print("Creating logistic regression models...")
    model_angle, model_distance = create_logistic_models(xg_data)

    if model_angle is not None and model_distance is not None:
        evaluate_models(xg_data, model_angle, model_distance)
        plot_models(xg_data, model_angle, model_distance)


    return xg_data, model_angle, model_distance


if __name__ == "__main__":
    data, angle_model, distance_model = main()
