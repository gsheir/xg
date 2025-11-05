import math

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import DataLoader
from models.logistic_regression import LogisticRegressionModel
from plotting_tools import plot_models
from settings import COMPETITION_ID, SEASON_IDS


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

    angle_model = LogisticRegressionModel()
    angle_model.fit(X_angle, y)

    # Model 2: Distance only
    X_distance = data[["distance"]].values

    distance_model = LogisticRegressionModel()
    distance_model.fit(X_distance, y)

    return angle_model, distance_model


def evaluate_models(data, angle_model, distance_model):
    """Evaluate the logistic regression models."""
    if data.empty or angle_model is None or distance_model is None:
        return

    X_angle = data[["angle"]].values
    X_distance = data[["distance"]].values
    y = data["goal"].values

    pred_angle = angle_model.calculate_auc_score(X_angle, y)
    pred_distance = distance_model.calculate_auc_score(X_distance, y)

    return pred_angle, pred_distance


def run_analysis():
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
    angle_model, distance_model = create_logistic_models(xg_data)

    if angle_model is not None and distance_model is not None:
        evaluate_models(xg_data, angle_model, distance_model)
        plot_models(xg_data, angle_model, distance_model)


if __name__ == "__main__":
    run_analysis()
