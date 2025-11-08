# Data loading settings
COMPETITION_ID = 37
SEASON_IDS = {
    "2018/2019": 4,
    "2019/2020": 42,
    "2020/2021": 90,
}

# Directory settings
OUTPUT_DIR = "output/"
MODELS_DIR = "models/saved"
EXPERIMENTS_DIR = "output/experiments"

# Model training settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "random_state": RANDOM_SEED,
    "max_iter": 1000,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_SEED,
    "max_depth": 10,
    "min_samples_split": 5,
}

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_SEED,
    "max_depth": 5,
    "learning_rate": 0.1,
}

# Plotting settings
PLOT_DPI = 300
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE_SINGLE = (10, 8)
FIGURE_SIZE_DOUBLE = (15, 6)
FIGURE_SIZE_COMPARISON = (12, 10)
