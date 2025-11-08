"""
Data management module for StatsBomb data with Parquet caching.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from statsbombpy import sb


class DataLoader:
    """Manages StatsBomb data fetching and caching with Parquet files."""

    def __init__(self, cache_dir: str = "data"):
        """Initialize the data manager with a cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._suppress_warnings()

    def _suppress_warnings(self):
        """Suppress warnings from StatsBomb API."""
        warnings.simplefilter("ignore")

    def _get_season_dir(self, season_id: int) -> Path:
        """Get the directory path for a specific season."""
        season_dir = self.cache_dir / f"season_{season_id}"
        season_dir.mkdir(exist_ok=True)
        return season_dir

    def _get_match_file_path(self, season_id: int, match_id: int) -> Path:
        """Get the file path for a specific match."""
        season_dir = self._get_season_dir(season_id)
        return season_dir / f"match_{match_id}.parquet"

    def is_match_cached(self, season_id: int, match_id: int) -> bool:
        """Check if a match is already cached."""
        return self._get_match_file_path(season_id, match_id).exists()

    def cache_match_events(
        self, season_id: int, match_id: int, events_df: pd.DataFrame
    ):
        """Cache events for a specific match as Parquet."""
        if events_df.empty:
            return

        # Add metadata
        events_df = events_df.copy()
        events_df["match_id"] = match_id
        events_df["season_id"] = season_id
        events_df["cached_at"] = pd.Timestamp.now()

        # Save as Parquet with compression
        file_path = self._get_match_file_path(season_id, match_id)
        events_df.to_parquet(file_path, compression="snappy", index=False)

        print(f"Cached {len(events_df)} events for match {match_id}")

    def load_match_events(self, season_id: int, match_id: int) -> pd.DataFrame:
        """Load cached events for a specific match."""
        file_path = self._get_match_file_path(season_id, match_id)

        if not file_path.exists():
            return pd.DataFrame()

        return pd.read_parquet(file_path)

    def fetch_and_cache_match_events(
        self, season_id: int, match_id: int, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Fetch events for a match, using cache if available."""
        if not force_refresh and self.is_match_cached(season_id, match_id):
            return self.load_match_events(season_id, match_id)

        try:
            print(f"Fetching events for match {match_id} from StatsBomb API...")
            events = sb.events(match_id=match_id)

            if not events.empty:
                self.cache_match_events(season_id, match_id, events)

            return events

        except Exception as e:
            print(f"Error fetching events for match {match_id}: {e}")
            return pd.DataFrame()

    def get_matches_for_season(
        self, competition_id: int, season_id: int
    ) -> pd.DataFrame:
        """Get all matches for a specific season."""
        try:
            matches = sb.matches(competition_id=competition_id, season_id=season_id)
            return matches
        except Exception as e:
            print(f"Error fetching matches for season {season_id}: {e}")
            return pd.DataFrame()

    def fetch_all_events(
        self,
        competition_id: int,
        season_ids: Dict[str, int],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch all events for multiple seasons."""
        all_events = []

        for season_name, season_id in season_ids.items():
            print(f"\nProcessing season: {season_name} (ID: {season_id})")

            # Get matches for this season
            matches = self.get_matches_for_season(competition_id, season_id)

            if matches.empty:
                print(f"No matches found for season {season_name}")
                continue

            print(f"Found {len(matches)} matches for season {season_name}")

            # Fetch events for each match
            season_events = []
            for _, match in matches.iterrows():
                match_id = match["match_id"]
                events = self.fetch_and_cache_match_events(
                    season_id, match_id, force_refresh
                )

                if not events.empty:
                    # Add season name for easier filtering
                    events["season_name"] = season_name
                    season_events.append(events)

            if season_events:
                season_df = pd.concat(season_events, ignore_index=True)
                all_events.append(season_df)
                print(f"Collected {len(season_df)} events for season {season_name}")

        if all_events:
            return pd.concat(all_events, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_shots_data(
        self,
        competition_id: int,
        season_ids: Dict[str, int],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Get all shots data from the specified seasons."""
        print("Fetching all events data...")
        all_events = self.fetch_all_events(competition_id, season_ids, force_refresh)

        if all_events.empty:
            print("No events data found!")
            return pd.DataFrame()

        # Filter for shots
        shots = all_events[all_events["type"] == "Shot"].copy()

        if shots.empty:
            print("No shots found in the data!")
            return pd.DataFrame()

        print(f"Found {len(shots)} shots across all seasons")
        return shots

    def get_cache_info(self) -> pd.DataFrame:
        """Get information about cached data."""
        cache_info = []

        for season_dir in self.cache_dir.iterdir():
            if season_dir.is_dir() and season_dir.name.startswith("season_"):
                season_id = season_dir.name.replace("season_", "")
                match_files = list(season_dir.glob("match_*.parquet"))

                total_size = sum(f.stat().st_size for f in match_files)

                cache_info.append(
                    {
                        "season_id": season_id,
                        "matches_cached": len(match_files),
                        "total_size_mb": round(total_size / 1024 / 1024, 2),
                        "cache_dir": str(season_dir),
                    }
                )

        return pd.DataFrame(cache_info)

    def clear_cache(self, season_id: Optional[int] = None):
        """Clear cached data for a specific season or all seasons."""
        if season_id:
            season_dir = self._get_season_dir(season_id)
            if season_dir.exists():
                for file in season_dir.glob("*.parquet"):
                    file.unlink()
                print(f"Cleared cache for season {season_id}")
        else:
            for season_dir in self.cache_dir.iterdir():
                if season_dir.is_dir():
                    for file in season_dir.glob("*.parquet"):
                        file.unlink()
            print("Cleared all cached data")
