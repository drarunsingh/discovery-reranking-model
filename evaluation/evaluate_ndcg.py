"""
Offline NDCG Evaluation (Netflix-Style)
--------------------------------------

Purpose:
Compare baseline vs learned ranking using NDCG@K.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import joblib


class NDCGEvaluator:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def _get_feature_matrix(self, X):
        return X.drop(columns=["user_id", "item_id", "label"])

    def evaluate(self, X, y, group, k=10):
        """
        Compute average NDCG@K across users
        """
        ndcg_model = []
        ndcg_baseline = []

        start = 0

        for group_size in group:
            end = start + group_size

            Xg = X.iloc[start:end]
            yg = np.array(y[start:end]).reshape(1, -1)

            # Skip groups with no positives
            if yg.max() == 0:
                start = end
                continue

            # -------------------------
            # Model ranking
            # -------------------------
            features = self._get_feature_matrix(Xg)
            scores_model = self.model.predict(features).reshape(1, -1)
            ndcg_model.append(ndcg_score(yg, scores_model, k=k))

            # -------------------------
            # Baseline ranking (popularity)
            # -------------------------
            popularity_scores = Xg["popularity"].values.reshape(1, -1)
            ndcg_baseline.append(ndcg_score(yg, popularity_scores, k=k))

            start = end

        return {
            f"NDCG@{k}_model": np.mean(ndcg_model),
            f"NDCG@{k}_baseline": np.mean(ndcg_baseline),
            "lift_pct": (
                (np.mean(ndcg_model) - np.mean(ndcg_baseline))
                / max(np.mean(ndcg_baseline), 1e-6)
            ) * 100
        }


# -----------------------------
# Example standalone usage
# -----------------------------
if __name__ == "__main__":
    from training_data import TrainingDataBuilder
    from feature_builder import FeatureBuilder

    # Build features
    fb = FeatureBuilder(
        "data/interactions.csv",
        "data/metadata.csv",
        "data/user_context.csv"
    )

    # Build training-style evaluation data
    builder = TrainingDataBuilder(
        interactions_path="data/interactions.csv",
        metadata_path="data/metadata.csv",
        user_context_path="data/user_context.csv",
        feature_builder=fb,
        negatives_per_user=15
    )

    users = builder.interactions["user_id"].unique()[:10]
    X, y, group = builder.build_training_data(users)

    evaluator = NDCGEvaluator("lgbm_ranker.pkl")
    results = evaluator.evaluate(X, y, group, k=10)

    print("📊 Offline Evaluation Results")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
