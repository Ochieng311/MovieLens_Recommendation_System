# 🎬 MovieLens Top-5 Movie Recommender

A hybrid recommendation system that takes a user's movie ratings and returns the **5 unseen movies they are most likely to watch and enjoy**. It is built on the MovieLens `ml-latest-small` dataset (GroupLens, University of Minnesota).

Collaborative filtering is the primary mechanism (matrix factorisation, item-kNN, user-kNN, implicit ALS). A content-based model covers rarely-rated movies, which addresses the cold-start problem.

> **Everything lives in one documented Jupyter notebook:** `MovieLens_Recommendation_System.ipynb`. For the full technical write-up, see [`DOCUMENTATION.md`](DOCUMENTATION.md).

---

## Table of contents
1. [Highlights](#highlights)
2. [Results](#results)
3. [How it works](#how-it-works)
4. [Repository contents](#repository-contents)
5. [Quick start](#quick-start)
6. [Using the recommender](#using-the-recommender)
7. [Notebook structure](#notebook-structure)
8. [Configuration](#configuration)
9. [Limitations](#limitations)
10. [Dataset, license and citation](#dataset-license-and-citation)

---

## Highlights

- **Objective:** a *"Top 5 picks for you"* list: unseen movies the user is likely to watch and rate **≥ 4.0**.
- **Hybrid design:** a learned, support-aware blend of four rating models, plus an implicit-feedback model that estimates *"will they watch it?"*.
- **Cold start handled on both sides:**
  - *new movies:* the content model scores any movie that has metadata, even with zero ratings;
  - *new users:* "fold-in" scores any rating vector without retraining.
- **Rigorous evaluation:** per-user 70/10/20 train/validation/test split, tuning on validation only, a single final test evaluation, RMSE/MAE **and** ranking metrics (Precision, Recall, Hit-rate, NDCG@5, coverage, popularity bias), bootstrap confidence intervals and segment analysis.
- **Fully documented notebook:** every code cell is followed by an explanation (what, why, how to read the output, decision taken).

## Results

Held-out **test set** (20% of each user's ratings, 610 users, models fitted on train + validation):

| Model | RMSE | MAE | Precision@5 | Hit-rate@5 | NDCG@5 | Catalogue coverage |
|---|---|---|---|---|---|---|
| Random | n/a | n/a | 0.003 | 1.5% | 0.004 | 26.9% |
| Bias model (μ + b_u + b_i) | 0.866 | 0.663 | 0.000 | 0.2% | 0.000 | 0.1% |
| Popularity (most rated) | n/a | n/a | 0.141 | 43.5% | 0.158 | 0.4% |
| Best single rating model (content-based) | 0.831 | 0.635 | 0.017 | 7.6% | 0.016 | 1.8% |
| Implicit CF only (iALS) | n/a | n/a | 0.246 | 65.1% | 0.272 | 3.2% |
| **Final hybrid** | **0.813** | **0.618** | **0.265** | **65.4%** | **0.298** | 2.8% |

- Rating accuracy improves by about **6%** (RMSE) over the bias baseline.
- The Top-5 list scores **almost twice popularity's NDCG@5**. The gain over popularity is **+0.140** (95% bootstrap CI: 0.123 to 0.156).
- **Key insight:** ranking by predicted rating alone is poor (NDCG@5 ≈ 0.02) even when the predictions are accurate, because users mostly rate what they chose to watch. Adding the implicit "will they watch it?" signal fixes this.

> Absolute Top-5 scores are higher on the test set than on validation because the test set holds twice as many hold-out ratings per user. Compare models within the same table, not across sets.

## How it works

```
 ratings ──► bias model (μ + b_u + b_i) ──► residuals ──┬─► Matrix factorisation (ALS)  ─┐
                                                         ├─► Item-based kNN               ├─► blend (weights per item-support bucket) ─► predicted rating r̂
 movie metadata (genres, decade, tags) ──────────────────┼─► Content-based ridge  ────────┤
                                                         └─► User-based kNN  ─────────────┘
 "who rated what" (binary) ──► implicit ALS ──► watch-likelihood p̂

                          final score = r̂ + 8 · p̂   →   drop already-rated movies   →   Top-5
```

| Component | Role |
|---|---|
| Bias model | Removes user generosity and movie quality, so the other models learn *taste* |
| Matrix factorisation (ALS-WR), item-kNN, user-kNN | Collaborative filtering on the residuals |
| Content-based ridge (genres + decade + tag-SVD) | Cold-start: works with no ratings for the movie |
| NNLS blend per support bucket (cold < 3, mid 3 to 19, warm 20+ ratings) | Trusts content for rare movies and CF for well-rated ones |
| Implicit ALS | Estimates the probability that the user watches the movie |

## Repository contents

```
.
├── MovieLens_Recommendation_System.ipynb   # the full, executed, documented notebook
├── top5_recommendations.csv                # generated: Top-5 titles for all 610 users
├── README.md                               # this file
├── DOCUMENTATION.md                        # full technical documentation
└── ml-latest-small/  (or ml-latest-small.zip)   # dataset: you provide this
```

`top5_recommendations.csv` has the columns `userId, rec1, rec2, rec3, rec4, rec5` (titles, best first). It comes from the final model retrained on 100% of the ratings, and excludes movies the user has already rated and movies with fewer than 5 ratings.

## Quick start

**1. Get the data**
Download `ml-latest-small.zip` from <https://grouplens.org/datasets/movielens/latest/> and put either the unzipped folder `ml-latest-small/` or the zip **next to the notebook**. The notebook unzips it automatically if only the zip is present.

**2. Install requirements**
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel
```
Tested with: NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-learn 1.9, matplotlib 3.11, seaborn 0.13. No GPU is needed; all models are implemented in NumPy/SciPy.

**3. Run**
```bash
jupyter notebook MovieLens_Recommendation_System.ipynb
```
Run all cells (Kernel → Restart & Run All). A full run takes roughly **5 to 6 minutes** on a standard CPU, and most of that is hyperparameter search. Results are deterministic (`SEED = 42`), so small floating-point differences across machines are possible, but the conclusions do not change.

## Using the recommender

The model class `HybridRecommender` is defined inside the notebook (Step 9.1). After running Step 10.1 (which creates the production model `prod`), you can use the following.

**A brand-new user, supplying ratings by title** (no retraining needed):
```python
rated = {
    "Matrix, The (1999)": 5,
    "Blade Runner (1982)": 5,
    "Alien (1979)": 4.5,
    "Titanic (1997)": 2,
}
R_row, M_row = user_vector(rated)          # resolves titles to movie ids
prod.recommend_row(R_row, M_row, n=5)      # DataFrame: title, genres, predicted rating, watch score, ...
```
Use the exact MovieLens title including the year (search with `find_movie("matrix")`). If a title is not found, the helper prints a warning and skips it.

**An existing user:**
```python
u = u2i[461]                               # userId -> internal row index
prod.recommend_row(R_all[u], M_all[u], n=5, self_row=u)
```

Each returned row contains: `title`, `genres`, `pred. rating` (clipped to 0.5 to 5), `watch score`, `#ratings` and `because you liked` (a heuristic explanation).

## Notebook structure

| Step | Content |
|---|---|
| **Intro cell** | Business objective, data understanding, approach, roadmap |
| **1** | Setup, data location |
| **2** | Data understanding: ratings, users, long tail, genres, tags, time |
| **3** | Preparation: index, per-user 70/10/20 split, matrices, content features |
| **4** | Evaluation framework (RMSE, MAE, Top-5 metrics) with oracle/random sanity checks |
| **5** | Baselines: global mean, bias model, popularity, random |
| **6** | Collaborative filtering: item-kNN, user-kNN, matrix factorisation |
| **7** | Content-based model, feature ablation and cold-start diagnosis |
| **8** | Hybrid: blend, implicit ALS, β sweep, **model card** |
| **9** | Final class, test evaluation, bootstrap significance, segment and calibration analysis |
| **10** | Demos for existing users and brand-new users, CSV export |
| **11** | Conclusions, limitations, next steps |

## Configuration

Frozen hyperparameters (tuned on validation only):

| Part | Setting |
|---|---|
| Bias model | λ_item = 2, λ_user = 1 |
| Matrix factorisation (ALS-WR) | k = 128, λ = 0.1, 10 iterations |
| Item-kNN | K = 200, shrink = 25, damping = 0.25 |
| User-kNN | k = 200, shrink = 20, damping = 0.25 |
| Content-based ridge | λ = 20 (48 features) |
| Implicit ALS | k = 64, α = 1, λ = 20, 8 iterations |
| Final ranking | `score = r̂ + β·p̂`, **β = 8**, minimum 5 ratings to be recommended |

To experiment, change the grids in Steps 5 to 8, `K_TOP` / `REL_THR` in Step 4, or the split fractions in `split_per_user` (Step 3.2). Then re-run from that point onward.

## Limitations

- **Offline metrics are conservative and exposure-biased:** movies a user never rated count as "not relevant". Real-world quality needs an online A/B test.
- **Small development dataset:** 610 users, and the median movie has only 3 ratings. Results may differ on larger MovieLens versions.
- **Very short histories** (about 7 ratings or fewer) lead to mostly popular, well-liked recommendations.
- **No time or context awareness:** random per-user split, no recency or session signals.
- **"Because you liked…"** explanations come from latent-space similarity and are heuristic.
- The class lives in the notebook. It is not (yet) packaged as an installable library.

## Dataset, license and citation

Data: **MovieLens `ml-latest-small`** (100,836 ratings, 3,683 tags, 9,742 movies, 610 users; generated 26 Sep 2018), © GroupLens Research, University of Minnesota.
The dataset is intended for **research use only**. Do not use it commercially without permission, and do not imply endorsement by the University of Minnesota or GroupLens. See the dataset's `README.txt` for the full terms.

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>
