"""
Feature Engineering Pipeline (Netflix-Style)
--------------------------------------------

Purpose:
Build ranking features for (user, candidate_item) pairs.
This stage encodes personalization, context, and business logic.

Output:
Feature matrix ready for Learning-to-Rank models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from generator import CandidateGenerator



class FeatureBuilder:
    def __init__(
        self,
        interactions_path: str,
        metadata_path: str,
        user_context_path: str,
        decay_lambda: float = 0.01,
    ):
        self.interactions = pd.read_csv(interactions_path)
        self.metadata = pd.read_csv(metadata_path)
        self.user_context = pd.read_csv(user_context_path)

        self.metadata["release_date"] = pd.to_datetime(
            self.metadata["release_date"]
        )

        self.decay_lambda = decay_lambda

    # -----------------------------
    # Helper functions
    # -----------------------------
    def _get_user_history(self, user_id):
        return self.interactions[
            self.interactions["user_id"] == user_id
        ]

    def _get_user_avg_watch_pct(self, user_id):
        history = self._get_user_history(user_id)
        if len(history) == 0:
            return 0.0
        return history["watched_pct"].mean()

    def _get_user_context(self, user_id):
        return self.user_context[
            self.user_context["user_id"] == user_id
        ].iloc[0]

    def _popularity_decay(self, release_date):
        days_since_release = (datetime.now() - release_date).days
        return np.exp(-self.decay_lambda * days_since_release)

    # -----------------------------
    # Main feature builder
    # -----------------------------
    def build_features(self, user_id, candidate_items):
        user_history = self._get_user_history(user_id)
        context = self._get_user_context(user_id)

        recent_genres = set(context["recent_genres"].split("|"))
        avg_watch_pct = self._get_user_avg_watch_pct(user_id)
        device = context["device"]
        time_of_day = context["time_of_day"]

        rows = []

        for item_id in candidate_items:
            item = self.metadata[self.metadata["item_id"] == item_id].iloc[0]

            genre_match = int(item["genre"] in recent_genres)

            popularity = item["popularity"]
            popularity_decay = self._popularity_decay(item["release_date"])

            # Device vs duration heuristic
            duration_match = 1
            if device == "mobile" and item["duration"] > 120:
                duration_match = 0

            # Time-of-day heuristic
            time_match = 1
            if time_of_day == "night" and item["genre"] in ["comedy", "romance"]:
                time_match = 0

            rows.append({
                "user_id": user_id,
                "item_id": item_id,
                "genre_match": genre_match,
                "popularity": popularity,
                "popularity_decay": popularity_decay,
                "avg_user_watch_pct": avg_watch_pct,
                "duration_match": duration_match,
                "time_of_day_match": time_match
            })

        return pd.DataFrame(rows)


# -----------------------------
# Example standalone usage
# -----------------------------
if __name__ == "__main__":
    from generator import CandidateGenerator

    generator = CandidateGenerator("data/interactions.csv")
    candidates = generator.generate_candidates(user_id=1, top_k=20)

    feature_builder = FeatureBuilder(
        interactions_path="data/interactions.csv",
        metadata_path="data/metadata.csv",
        user_context_path="data/user_context.csv"
    )

    X = feature_builder.build_features(user_id=1, candidate_items=candidates)

    print(X.head())
