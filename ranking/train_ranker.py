"""
LightGBM LambdaRank Training (Netflix-Style)
--------------------------------------------

Purpose:
Train a Learning-to-Rank model using LambdaRank
with user-level query groups.
"""

import lightgbm as lgb
import pandas as pd
import joblib


class RankerTrainer:
    def __init__(self):
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="gbdt",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            random_state=42
        )

    def train(self, X: pd.DataFrame, y: list, group: list):
        """
        Train LambdaRank model
        """
        # Drop non-feature columns
        feature_cols = [
            col for col in X.columns
            if col not in ["user_id", "item_id", "label"]
                ]
        

        X_train = X[feature_cols]

        self.model.fit(
            X_train,
            y,
            group=group
        )

        return feature_cols

    def save_model(self, path="lgbm_ranker.pkl"):
        joblib.dump(self.model, path)


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

    # Build training data
    builder = TrainingDataBuilder(
        interactions_path="data/interactions.csv",
        metadata_path="data/metadata.csv",
        user_context_path="data/user_context.csv",
        feature_builder=fb,
        negatives_per_user=15
    )

    users = builder.interactions["user_id"].unique()[:10]
    X, y, group = builder.build_training_data(users)

    # Train ranker
    trainer = RankerTrainer()
    feature_cols = trainer.train(X, y, group)

    print("✅ Ranker trained successfully")
    print("Features used:")
    for f in feature_cols:
        print(" -", f)

    # Save model
    trainer.save_model()
    print("💾 Model lgbm_ranker.pkl")
