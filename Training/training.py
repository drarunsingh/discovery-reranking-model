"""
Training Dataset Builder (Netflix-Style)
----------------------------------------

Purpose:
Build training data for Learning-to-Rank models by mixing
positive and negative examples per user.

Each user = one query group.
"""

import pandas as pd
import numpy as np


class TrainingDataBuilder:
    def __init__(
        self,
        interactions_path: str,
        metadata_path: str,
        user_context_path: str,
        feature_builder,
        min_positive_watch_pct: float = 0.5,
        negatives_per_user: int = 20,
        random_state: int = 42,
    ):
        self.interactions = pd.read_csv(interactions_path)
        self.metadata = pd.read_csv(metadata_path)
        self.user_context = pd.read_csv(user_context_path)

        self.feature_builder = feature_builder
        self.min_positive_watch_pct = min_positive_watch_pct
        self.negatives_per_user = negatives_per_user
        self.random_state = random_state

        np.random.seed(self.random_state)

    def _get_positive_items(self, user_id):
        return self.interactions[
            (self.interactions["user_id"] == user_id)
            & (self.interactions["watched_pct"] >= self.min_positive_watch_pct)
        ]["item_id"].unique().tolist()

    def _sample_negative_items(self, user_id, exclude_items):
        all_items = self.metadata["item_id"].unique().tolist()

        candidate_negatives = list(set(all_items) - set(exclude_items))

        if len(candidate_negatives) == 0:
            return []

        num_samples = min(self.negatives_per_user, len(candidate_negatives))

        return np.random.choice(
            candidate_negatives, size=num_samples, replace=False
        ).tolist()

    def build_training_data(self, user_ids):
        """
        Build full training dataset across multiple users.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix
        y : list[int]
            Relevance labels
        group : list[int]
            Query group sizes
        """
        X_all = []
        y_all = []
        group = []

        for user_id in user_ids:
            positives = self._get_positive_items(user_id)

            # Skip users with no positives (cold-start)
            if len(positives) == 0:
                continue

            negatives = self._sample_negative_items(
                user_id=user_id,
                exclude_items=positives
            )

            training_items = list(set(positives + negatives))

            # Build features
            X_user = self.feature_builder.build_features(
                user_id=user_id,
                candidate_items=training_items
            )

            # Build labels
            X_user = X_user.merge(
                self.interactions[["user_id", "item_id", "watched_pct"]],
                on=["user_id", "item_id"],
                how="left"
            ).fillna({"watched_pct": 0.0})

            X_user["label"] = X_user["watched_pct"].apply(
                lambda x: 3 if x >= 0.8 else
                          2 if x >= 0.5 else
                          1 if x >= 0.2 else
                          0
            )

            X_all.append(X_user.drop(columns=["watched_pct"]))
            y_all.extend(X_user["label"].tolist())
            group.append(len(X_user))

        X = pd.concat(X_all, ignore_index=True)

        return X, y_all, group


# -----------------------------
# Example standalone usage
# -----------------------------
if __name__ == "__main__":
    from feature_builder import FeatureBuilder

    fb = FeatureBuilder(
        interactions_path="data/interactions.csv",
        metadata_path="data/metadata.csv",
        user_context_path="data/user_context.csv"
    )

    builder = TrainingDataBuilder(
        interactions_path="data/interactions.csv",
        metadata_path="data/metadata.csv",
        user_context_path="data/user_context.csv",
        feature_builder=fb,
        negatives_per_user=15
    )

    users = builder.interactions["user_id"].unique()[:10]

    X, y, group = builder.build_training_data(users)

    print("Feature matrix shape:", X.shape)
    print("First 10 labels:", y[:10])
    print("Query groups:", group)
