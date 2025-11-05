import math

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import DataLoader
from experiment_manager import ExperimentManager
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
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


def run_experiments():
    """Run the xG model experiment with multiple models and cross-validation."""
    plt.ioff()

    print("=" * 60)
    print("Expected Goals (xG) Model Comparison Experiment")
    print("=" * 60)

    # 1. Load and prepare data
    print("\n1. Loading data...")
    shots_df = get_shots_data()

    if shots_df.empty:
        print("No shots data found!")
        return

    print("\n2. Preparing data for xG modeling...")
    xg_data = prepare_xg_data(shots_df)

    if xg_data.empty:
        print("No valid shots data for modeling!")
        return

    print(f"Prepared {len(xg_data)} shots for modeling")

    # 3. Set up experiment
    print("\n3. Setting up experiment...")
    experiment = ExperimentManager("xg_model_comparison")

    # Prepare features and target
    feature_columns = ["distance", "angle"]
    X = xg_data[feature_columns]
    y = xg_data["goal"]

    experiment.set_data(X, y)

    # 4. Register models
    print("\n4. Registering models...")

    # Model 1: Logistic Regression - Angle only
    experiment.register_model(
        model_class=LogisticRegressionModel,
        model_name="logistic_angle",
        features=["angle"],
    )

    # Model 2: Logistic Regression - Distance only
    experiment.register_model(
        model_class=LogisticRegressionModel,
        model_name="logistic_distance",
        features=["distance"],
    )

    # Model 3: Logistic Regression - Combined
    experiment.register_model(
        model_class=LogisticRegressionModel,
        model_name="logistic_combined",
        features=["distance", "angle"],
    )

    # Model 4: Random Forest
    experiment.register_model(
        model_class=RandomForestModel,
        model_name="random_forest",
        features=["distance", "angle"],
    )

    # Model 5: XGBoost
    experiment.register_model(
        model_class=XGBoostModel,
        model_name="xgboost",
        features=["distance", "angle"],
    )

    # 5. Run the experiment
    print("\n5. Running experiment...")
    experiment.run_experiment(force_retrain=False)

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print(f"Results saved to: {experiment.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_experiments()
