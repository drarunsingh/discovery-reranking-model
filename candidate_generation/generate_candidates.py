"""
Candidate Generator (Netflix-Style)
----------------------------------

Purpose:
Generate a high-recall candidate set for a given user.
This stage optimizes recall and latency, NOT ranking precision.

Key principles:
- Recall-first, ranking-later
- Simple, explainable logic
- Modular and replaceable
"""

import pandas as pd
from typing import List, Set


class CandidateGenerator:
    def __init__(self, interactions_path: str):
        """
        Parameters
        ----------
        interactions_path : str
            Path to interactions.csv
        """
        self.interactions = pd.read_csv(interactions_path)

        # Basic validation
        required_cols = {"user_id", "item_id"}
        if not required_cols.issubset(self.interactions.columns):
            raise ValueError(
                f"interactions.csv must contain columns: {required_cols}"
            )

    def _get_user_history(self, user_id: int) -> Set[int]:
        """
        Returns items already interacted with by the user
        """
        return set(
            self.interactions[
                self.interactions["user_id"] == user_id
            ]["item_id"].unique()
        )

    def _find_similar_users(self, user_id: int, top_n: int = 10) -> List[int]:
        """
        Finds similar users based on item co-occurrence.

        Similarity metric:
        - Number of overlapping items watched
        """
        user_items = self._get_user_history(user_id)

        if len(user_items) == 0:
            return []

        similarities = []

        for other_user in self.interactions["user_id"].unique():
            if other_user == user_id:
                continue

            other_items = set(
                self.interactions[
                    self.interactions["user_id"] == other_user
                ]["item_id"].unique()
            )

            overlap = len(user_items.intersection(other_items))
            if overlap > 0:
                similarities.append((other_user, overlap))

        # Sort by overlap (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [user for user, _ in similarities[:top_n]]

    def generate_candidates(self, user_id: int, top_k: int = 100) -> List[int]:
        """
        Generates candidate items for a given user.

        Steps:
        1. Fetch user's watch history
        2. Find similar users
        3. Collect their watched items
        4. Remove already seen items
        5. Return top-K candidates

        Parameters
        ----------
        user_id : int
            Target user
        top_k : int
            Number of candidates to return

        Returns
        -------
        List[int]
            Candidate item_ids
        """
        user_history = self._get_user_history(user_id)
        similar_users = self._find_similar_users(user_id)

        candidate_items = set()

        for sim_user in similar_users:
            items = self.interactions[
                self.interactions["user_id"] == sim_user
            ]["item_id"].unique()
            candidate_items.update(items)

        # Remove items already seen by the user
        candidate_items -= user_history

        return [int(item) for item in list(candidate_items)[:top_k]]



# -------------------------
# Example standalone usage
# -------------------------
if __name__ == "__main__":
    generator = CandidateGenerator("data/interactions.csv")

    user_id = 1
    candidates = generator.generate_candidates(user_id=user_id, top_k=50)

    print(f"Generated {len(candidates)} candidates for user {user_id}")
    print(candidates)
