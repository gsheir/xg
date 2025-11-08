import math

import pandas as pd


class DataPreprocessor:
    def __init__(self):
        self.df = None

    def load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df
        return df

    def prepare_xg_data(self):
        """Prepare shots data for xG modeling."""
        if self.df.empty:
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

        coords = self.df["location"].apply(extract_coordinates)
        self.df[["x", "y"]] = pd.DataFrame(coords.tolist(), index=self.df.index)

        # Remove shots without location data
        self.df = self.df.dropna(subset=["x", "y"])

        # Calculate distance and angle
        self.df[["distance", "angle"]] = self.df.apply(
            lambda row: self.calculate_angle_and_distance(row["x"], row["y"]),
            axis=1,
            result_type="expand",
        )

        # Create binary goal variable
        self.df["goal"] = (self.df["shot_outcome"] == "Goal").astype(int)

        # Remove penalty shots for more realistic xG
        self.df = self.df[self.df["shot_type"] != "Penalty"]

        return self.df[
            ["distance", "angle", "goal", "x", "y", "shot_outcome", "season_name"]
        ].copy()

    @staticmethod
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

    def run(self):
        if self.df is None:
            raise ValueError(
                "Data not loaded. Please load data before running preprocessing."
            )

        self.df = self.prepare_xg_data(self.df)
        return self.df
