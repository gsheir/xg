import matplotlib.pyplot as plt

from models.logistic_regression import LogisticRegressionModel
from pipeline.data_loader import DataLoader
from pipeline.experiment_manager import ExperimentManager
from pipeline.settings import COMPETITION_ID, EXPERIMENTS_DIR, N_FOLDS, SEASON_IDS


def run_compare_models_experiment():
    """Run the xG model experiment with multiple models and cross-validation."""
    plt.ioff()

    print("Running Compare Models Experiment")

    # Load and prepare data
    print("Loading data...")
    data_manager = DataLoader()

    cache_info = data_manager.get_cache_info()
    if not cache_info.empty:
        print("Current cache status:")
        print(cache_info.to_string(index=False))
        print()

    df = data_manager.get_shots_data(
        competition_id=COMPETITION_ID,
        season_ids=SEASON_IDS,
        force_refresh=False,
    )

    if df.empty:
        print("No data found!")
        return

    experiment_settings = {
        "feature_columns": ["distance", "angle"],
        "target_column": "goal",
        "output_dir": EXPERIMENTS_DIR,
        "split_settings": {
            "stratify": True,
            "split_type": "cross_validation",
            "n_folds": N_FOLDS,
        },
    }

    experiment = ExperimentManager("compare_models", experiment_settings)
    experiment.register_dataset(df)

    # Register models
    print("Registering models...")

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

    # Run the experiment
    print("Running experiment...")
    experiment.run(force_retrain=True)

    print("Experiment completed successfully!")
    print(f"Results saved to: {experiment.output_dir}")
