# 🎬 Discovery Re-Ranking Model (Netflix-Style)

A Netflix-inspired Discovery Re-Ranking system that re-orders a candidate set of titles
for a given user context to improve relevance, discovery, and long-term member joy.

**Key idea:**  
Candidate generation maximizes recall; ranking optimizes user experience.

## Why Re-Ranking Is Not Recommendation

Traditional recommender systems focus on *what* to recommend.
Discovery systems focus on *how content is ordered and surfaced*.

In Netflix-style systems:
- Candidate generation ensures coverage
- Re-ranking balances relevance, freshness, diversity, and context
- Business constraints are enforced explicitly

## System Architecture

User Context  
↓  
Candidate Generation (Collaborative Filtering)  
↓  
Feature Engineering (Context + Business Logic)  
↓  
Learning-to-Rank Model (LambdaRank)  
↓  
Re-Ranked Content List

## Dataset Design

Synthetic but realistic datasets inspired by MovieLens and TMDB-style metadata:

- interactions.csv – implicit user feedback
- metadata.csv – content attributes (genre, popularity, freshness)
- user_context.csv – session-level context

Datasets are intentionally small to emphasize system design over scale.

## Candidate Generation (Recall-First)

Candidates are generated using item-based collaborative filtering based on user co-occurrence.

This stage maximizes recall and does not perform ranking.

## Feature Engineering

Most of the system’s intelligence lives in feature design:

- genre_match
- popularity
- popularity_decay
- avg_user_watch_pct
- duration_match
- time_of_day_match

Business logic is encoded explicitly for transparency.

## Ranking Model

- Model: LightGBM LGBMRanker
- Objective: LambdaRank
- Metric: NDCG

The model is intentionally simple; complexity lives in the pipeline.

## Offline Evaluation

Metric: NDCG@10

Results (synthetic dataset):
- Model: 0.91
- Popularity baseline: 0.36

These results indicate strong directional improvement, not production guarantees.

## Tradeoffs & Limitations

- Offline metrics may not translate to online impact
- Synthetic data inflates absolute gains
- No real-time feedback loop implemented

These are explicitly acknowledged by design.

## Interview Positioning Statement
“This system optimizes for member joy, not just clicks.
Most of the intelligence in ranking systems lives in feature design and evaluation, not the model itself.”

