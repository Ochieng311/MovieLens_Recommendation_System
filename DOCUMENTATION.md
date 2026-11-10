# MovieLens Top-5 Recommender: Full Technical Documentation

This document describes the data, the methodology, the mathematics, the evaluation protocol, the results and the usage of the hybrid movie recommendation system implemented in `MovieLens_Recommendation_System.ipynb`. For a short overview and quick start, see [`README.md`](README.md).

## Contents
1. [Problem definition](#1-problem-definition)
2. [Data](#2-data)
3. [Experimental protocol](#3-experimental-protocol)
4. [Evaluation metrics](#4-evaluation-metrics)
5. [Models and mathematics](#5-models-and-mathematics)
6. [Hyperparameter search](#6-hyperparameter-search)
7. [Results](#7-results)
8. [Cold-start handling](#8-cold-start-handling)
9. [Final system and API reference](#9-final-system-and-api-reference)
10. [Worked examples](#10-worked-examples)
11. [Design decisions and rationale (FAQ)](#11-design-decisions-and-rationale-faq)
12. [Limitations](#12-limitations)
13. [Extending the system](#13-extending-the-system)
14. [Troubleshooting](#14-troubleshooting)
15. [Reproducibility](#15-reproducibility)
16. [Glossary](#16-glossary)
17. [Data license and citation](#17-data-license-and-citation)

---

## 1. Problem definition

**Business problem.** A streaming service wants a *"Top 5 picks for you"* row. Given a user's rating history, recommend **five movies the user has not rated, and is likely to watch and enjoy**. "Enjoy" is operationalised as a rating **≥ 4.0** on the 0.5 to 5 scale.

**Technical task.** Rank all unseen movies for a user and return the 5 best. The system answers two questions at once:

| Question | Quantity | Models |
|---|---|---|
| How much would the user **like** movie *i*? | predicted rating `r̂(u,i)` | bias model, matrix factorisation, item-kNN, user-kNN, content-based ridge, blended |
| Would the user **watch** movie *i* at all? | watch-likelihood `p̂(u,i)` | implicit-feedback ALS |

**Final ranking score**

```
score(u, i) = r̂(u, i) + β · p̂(u, i)        β = 8
```

**Assignment requirements and where they are met**

| Requirement | Where |
|---|---|
| Top-5 recommendations from a user's ratings of other movies | Steps 8 to 10; `HybridRecommender.recommend_row` |
| Must use collaborative filtering | Step 6 (item-kNN, user-kNN, MF) and Step 8.2 (implicit CF) |
| Hybrid with content-based filtering for cold start | Step 7 (content model), Step 8.1 (support-aware blend) |
| Thoughtful evaluation metrics | Step 4 and Steps 5 to 9 (RMSE, MAE, ranking metrics, coverage, popularity bias, bootstrap CIs) |
| Specify how the user provides ratings | A `{title: rating}` dictionary (new users) or the stored history (existing users); see §10 |

## 2. Data

**Source.** MovieLens `ml-latest-small`, from GroupLens (University of Minnesota), generated 26 September 2018.

| File | Rows | Content |
|---|---|---|
| `ratings.csv` | 100,836 | `userId, movieId, rating, timestamp` (0.5 to 5.0 in half-star steps, 1996 to 2018) |
| `movies.csv` | 9,742 | `movieId, title (with year), genres (pipe-separated)` |
| `tags.csv` | 3,683 | free-text tags: `userId, movieId, tag, timestamp` |
| `links.csv` | 9,742 | IMDb / TMDb ids (not used for modelling; 8 missing TMDb ids) |

### 2.1 Key facts (from the exploratory analysis)

| Aspect | Finding |
|---|---|
| Size | 610 users, 9,742 movies (9,724 rated, **18 never rated**) |
| Integrity | no nulls in ratings/movies/tags, no duplicate (user, movie) pairs, no orphan movie ids |
| Sparsity | **98.30%** of the 610 × 9,724 matrix is empty |
| Rating level | mean **3.50**, median 3.5; **48.2%** of ratings ≥ 4.0; 13.4% ≤ 2.0; 69.9% whole stars |
| User generosity | user-mean ratings: std **0.48**, range 1.27 to 5.00 |
| User activity | every user has ≥ 20 ratings; median **70.5**, mean 165, max **2,698**; 36.9% of users have < 50; the top 10% of users supply 47.7% of ratings |
| Movie popularity (long tail) | median **3** ratings per movie; **35.4%** have exactly 1; 66.4% have ≤ 5; the top 10% of movies receive **60.0%** of ratings; max 329 |
| Popularity vs. quality | Spearman 0.25 between rating count and mean rating (movies with ≥ 10 ratings) |
| Genres | 19 genres + "(no genres listed)" (34 movies); 2.27 genres per movie on average; Drama and Comedy dominate; genres differ in mean rating (Drama 3.66, Comedy 3.38) |
| Year | 13 movies have no parsable year (parsed from the title) |
| Tags | 3,683 tags by only **58 users** on 1,572 movies (**16.1%** of the catalogue); most frequent tag is "in netflix queue" |
| Time | bursty activity (e.g. 10,061 ratings in 2000, 507 in 1998); yearly mean rating stays roughly within 3.3 to 3.9 |

### 2.2 Implications for modelling

1. **Sparsity and long tail:** similarity-based CF is unreliable for most movies, so a content model is blended in, with trust set by how many ratings a movie has.
2. **Different rating habits:** model user and item biases first; CF models learn residual *taste* only.
3. **Exposure bias** (users rate what they chose to watch): evaluate with ranking metrics and add an implicit watch-likelihood model.
4. **Skewed positive ratings:** "liked" is defined as ≥ 4.0 (about the top half of ratings).
5. **Light users and power users:** regularisation scales with the number of ratings, and evaluation is per user with a segment analysis.

### 2.3 Preprocessing and features

- Index **all 9,742 catalogue movies** and the 610 users as contiguous integers; never-rated movies stay in the model.
- **Dense matrices:** `R` (ratings, float32) and `M` (rated mask), each 610 × 9,742 (24 MB).
- **Content feature matrix `X` (9,742 × 48):**

| Block | Columns | Construction |
|---|---|---|
| Genres | 19 | multi-hot (the "no genres listed" label is dropped, giving an all-zero vector) |
| Release decade | 9 | one-hot over ≤1949, 50s, 60s, 70s, 80s, 90s, 00s, 10s, plus "unknown"; scaled by **0.5** |
| Tags | 20 | lower-cased tags → TF-IDF (`min_df = 2`, vocabulary 542, 1,343 movies with ≥ 1 tag) → truncated SVD to 20 dims (24.4% of variance) → L2-normalised rows |

Tag text is item metadata (not ratings), so using all tags does not leak test ratings.

## 3. Experimental protocol

**Per-user random split** (seed 42): for every user, about 70% of their ratings go to *train*, 10% to *validation*, 20% to *test* (at least 1 validation and 1 test rating per user).

| Set | Ratings | Users | Share on movies unseen in train | Relevant (≥ 4) ratings |
|---|---|---|---|---|
| Train | 70,579 | 610 | 0% | 33,934 |
| Validation | 10,093 | 610 | 4.3% | 4,912 |
| Test | 20,164 | 610 | 4.4% | 9,734 |

Smallest per-user sizes: 14 / 2 / 4. 1,164 catalogue movies have no training rating.

**Workflow**
1. **Tune** every hyperparameter by fitting on *train* and scoring on *validation*.
2. **Fit blend weights and β** on validation.
3. **Refit** everything on *train + validation* with frozen settings and evaluate **once** on *test*.
4. **Refit on 100%** of the data for the demo/export model.

> Because the test set holds twice as many hold-out ratings per user as validation, absolute Top-5 scores are higher on test than on validation. Compare models only within the same set.

**Ranking evaluation rules.** For each user, movies already seen (in the fitted data) are never recommended; the Top-5 by score is compared with the user's held-out movies rated ≥ 4. Unrated movies are treated as non-relevant (a conservative assumption; see §12). No minimum-support filter is applied during evaluation, so all models are judged on the full unseen catalogue.

## 4. Evaluation metrics

**Rating accuracy** (predictions clipped to [0.5, 5]):

```
RMSE = sqrt( mean( (r̂ − r)² ) )        MAE = mean( |r̂ − r| )
```

**Top-5 ranking** (per user *u* with at least one relevant held-out movie; averaged over users). Let `hits_k` = 1 if the item at rank *k* is relevant, and `n_rel` the number of relevant held-out items:

| Metric | Definition |
|---|---|
| Precision@5 | `Σ_k hits_k / 5` |
| Recall@5 | `Σ_k hits_k / n_rel` |
| Hit-rate@5 | 1 if at least one of the 5 is relevant, else 0 |
| NDCG@5 | `DCG / IDCG`, with `DCG = Σ_{k=1..5} hits_k / log2(k+1)` and `IDCG = Σ_{k=1..min(n_rel,5)} 1/log2(k+1)` |
| Coverage | distinct movies recommended to anyone ÷ catalogue size (9,742) |
| AvgLogPop | mean `log(1 + #train ratings)` of recommended movies (popularity-bias indicator) |

**Primary metrics:** RMSE (rating accuracy) and **NDCG@5** (list quality). **Sanity check:** an oracle that knows the answers scores Hit@5 = 1 and NDCG@5 = 1 (Precision@5 is capped at 0.73 on validation because many users have fewer than 5 relevant items), and random scores about 0.

## 5. Models and mathematics

Notation: `u` user, `i` movie, `m_ui ∈ {0,1}` rated indicator, `n_u`, `n_i` rating counts, `μ` global mean.

### 5.1 Baselines

- **Global mean:** `r̂ = μ`.
- **Damped bias model:** `r̂_ui = μ + b_i + b_u`, fitted by 10 alternating updates:
  ```
  b_i = Σ_u m_ui (r_ui − μ − b_u) / (n_i + λ_i)
  b_u = Σ_i m_ui (r_ui − μ − b_i) / (n_u + λ_u)
  ```
  The λ's shrink the offsets of rarely-rated movies and users toward 0. For a **new user**, `b_u` is computed from the supplied ratings with the same formula.
- **Popularity:** score = number of training ratings. **Random:** uniform scores.

### 5.2 Residuals

All CF and content models predict the **residual** `e_ui = r_ui − μ − b_i − b_u` (0 where unrated; std 0.80 on train, the error the models try to reduce). The final rating is `bias + predicted residual`.

### 5.3 Item-based kNN CF

Adjusted cosine on residuals, shrunk by the number of co-raters `n_ij`:

```
s_ij = [ Σ_u e_ui e_uj / (‖e_·i‖ ‖e_·j‖) ] · n_ij / (n_ij + shrink)     (n_ij ≥ 2, negative similarities set to 0)
ê_ui = Σ_{j ∈ N_K(i)} s_ij e_uj / ( Σ_{j ∈ N_K(i)} s_ij m_uj + damp )
```

`N_K(i)` is the set of the K most similar movies. The `damp` term pulls predictions with little evidence toward 0. Computed in chunks of 500 movies with sparse neighbour storage. Sanity check: neighbours of *Toy Story* are *Toy Story 2* and *3*; of *Pulp Fiction*: *Usual Suspects*, *Fight Club*, *Kill Bill Vol. 1*.

### 5.4 User-based kNN CF

Same similarity between user residual vectors, shrunk by overlap; keep the k most similar users (never the user themselves) and:

```
ê_ui = Σ_{v ∈ N_k(u)} s_uv e_vi / ( Σ_{v ∈ N_k(u)} s_uv m_vi + damp )
```

### 5.5 Matrix factorisation (ALS-WR)

Each user and movie gets a latent vector (`p_u`, `q_i` ∈ ℝᵏ); `ê_ui = p_u · q_i`. Alternating exact solutions with regularisation proportional to the number of ratings:

```
p_u = ( Q_uᵀ Q_u + λ n_u I )⁻¹ Q_uᵀ e_u        (Q_u: factors of the movies user u rated)
q_i = ( P_iᵀ P_i + λ n_i I )⁻¹ P_iᵀ e_i
```

Movies with no ratings keep a zero vector. **Fold-in:** for any new rating row, `p_u` is obtained by solving the first equation with the movie factors fixed.

### 5.6 Content-based model (per-user ridge)

For each user, regress their residuals on the content features of the movies they rated, then apply the profile to *all* movies:

```
w_u = ( X_uᵀ X_u + λ I )⁻¹ X_uᵀ e_u        ê_ui = x_i · w_u
```

Works for any movie with metadata, including movies nobody has rated.

### 5.7 Hybrid rating blend

```
r̂_ui = μ + b_i + b_u + Σ_k w_{k, bucket(i)} · ê^{(k)}_ui ,     k ∈ {MF, item-kNN, user-kNN, content}
```

`bucket(i)` depends on the number of training ratings of movie *i*: **cold** (< 3), **mid** (3 to 19), **warm** (≥ 20). The weights are fitted by **non-negative least squares** on validation residuals, one vector per bucket. Fitted weights:

| Bucket | MF | Item-kNN | User-kNN | Content | Validation pairs |
|---|---|---|---|---|---|
| cold (< 3) | 0.141 | 0.257 | 0.737 | **1.719** | 1,097 |
| mid (3 to 19) | 0.446 | 0.455 | 0.000 | **0.834** | 3,106 |
| warm (20+) | 0.000 | **0.904** | 0.459 | 0.692 | 5,890 |

Weights above 1 are expected: each component is deliberately shrunk, so the blend rescales it. MF gets weight 0 in the warm bucket because the kNN models already explain the same signal.

### 5.8 Implicit ALS: watch-likelihood

Trained on the **binary** matrix "did the user rate this movie?" (preference 1 for rated, 0 otherwise; confidence `c_ui = 1 + α m_ui`). User step (and, symmetrically, item step):

```
x_u = ( YᵀY + α Y_uᵀ Y_u + λ I )⁻¹ · (1 + α) Σ_{i rated by u} y_i        p̂_ui = x_u · y_i
```

Sanity check (validation): mean watch score is **0.437** for movies the user rated vs **0.011** for those they did not.

### 5.9 Final ranking

```
score(u, i) = r̂_ui + 8 · p̂_ui
```

Already-rated movies are removed; for deployment, movies with fewer than 5 ratings are also excluded (`min_ratings_to_recommend = 5`). Why β is needed is explained in §11.

## 6. Hyperparameter search

All tuned on **validation**, minimising RMSE (rating models) or maximising NDCG@5 (iALS, β). After each grid the notebook reports whether the winner lies on a grid edge.

| Model | Grid | Winner | Validation score |
|---|---|---|---|
| Bias | λ_i, λ_u ∈ {0.5, 1, 2, 5, 10, 20} (36) | λ_i = 2, λ_u = 1 | RMSE 0.8552 |
| Item-kNN | shrink {10, 25, 50} × K {50, 100, 200, 400} × damp {0.05, 0.25, 1} (36) | 25 / 200 / 0.25 | RMSE 0.8337 |
| User-kNN | k {50, 100, 200, 400} × shrink {5, 20, 50} × damp {0.05, 0.25, 1} (36) | 200 / 20 / 0.25 | RMSE 0.8445 |
| MF (ALS-WR) | k {32, 64, 128} × λ {0.05, 0.1, 0.2}, 10 iterations (9) | k = 128, λ = 0.1 | RMSE 0.8434 |
| Content ridge | features {genres, + decade, + decade + tags} × λ {5, 10, 20, 40, 80} (15) | all features, λ = 20 | RMSE 0.8297 |
| Implicit ALS | k {16, 32, 64} × α {0.5, 1, 2} × λ {10, 20, 40}, 8 iterations (27) | k = 64, α = 1, λ = 20 | NDCG@5 0.1405 |
| β | {0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 32} | β = 8 | NDCG@5 0.1493 |

Notes:
- **MF k = 128** sits at the top of its grid, but gains beyond k = 64 are only in the third decimal, so the search stops there. The ALS learning curve flattens after about 5 iterations (0.8544 → 0.8427 → 0.8433 at 10 → 0.8437 at 20).
- **Implicit ALS k = 64** sits at the top of its k range. An extra check with k = 128 did not help (NDCG@5 0.139, about 4× slower), and λ = 40 was clearly worse.
- **Content features ablation (best λ per variant):** genres only 0.8387, + decade 0.8314, + decade + tags **0.8297**.
- **β** sits on a broad plateau (β from 6 to 16 within 0.003 NDCG@5), so the choice is not sensitive.

## 7. Results

### 7.1 Validation (used for model selection)

| Model | RMSE | MAE | P@5 | NDCG@5 |
|---|---|---|---|---|
| Global mean | 1.0438 | 0.8290 | n/a | n/a |
| Bias model | 0.8552 | 0.6600 | 0.0021 | 0.0018 |
| Item-kNN | 0.8337 | 0.6406 | 0.0131 | 0.0181 |
| User-kNN | 0.8445 | 0.6490 | 0.0055 | 0.0067 |
| Matrix factorisation | 0.8434 | 0.6479 | 0.0090 | 0.0103 |
| Content-based | 0.8297 | 0.6389 | 0.0065 | 0.0059 |
| Hybrid rating blend | **0.8134** (5-fold CV: 0.8145) | 0.6232 | 0.0117 | 0.0142 |
| Popularity | n/a | n/a | 0.0761 | 0.0948 |
| Implicit CF only | n/a | n/a | 0.1105 | 0.1405 |
| **Final hybrid** | 0.8134 | 0.6232 | **0.1198** | **0.1493** |

### 7.2 Test (final, single evaluation)

| Model | RMSE | MAE | P@5 | R@5 | Hit@5 | NDCG@5 | Coverage |
|---|---|---|---|---|---|---|---|
| Random | n/a | n/a | 0.0033 | 0.0011 | 1.5% | 0.0040 | 26.9% |
| Popularity | n/a | n/a | 0.1414 | 0.0597 | 43.5% | 0.1580 | 0.4% |
| Global mean | 1.0442 | 0.8266 | n/a | n/a | n/a | n/a | n/a |
| Bias model | 0.8663 | 0.6630 | 0.0003 | 0.0000 | 0.2% | 0.0002 | 0.1% |
| Item-kNN | 0.8403 | 0.6397 | 0.0238 | 0.0133 | 9.9% | 0.0244 | 1.7% |
| User-kNN | 0.8509 | 0.6477 | 0.0132 | 0.0050 | 5.6% | 0.0119 | 0.7% |
| Matrix factorisation | 0.8503 | 0.6482 | 0.0132 | 0.0066 | 5.8% | 0.0128 | 3.0% |
| Content-based | 0.8309 | 0.6352 | 0.0166 | 0.0054 | 7.6% | 0.0162 | 1.8% |
| Hybrid rating blend | **0.8132** | **0.6179** | 0.0258 | 0.0110 | 11.4% | 0.0244 | 3.8% |
| Implicit CF only | n/a | n/a | 0.2460 | 0.1248 | 65.1% | 0.2716 | 3.2% |
| **Final hybrid** | **0.8132** | **0.6179** | **0.2649** | **0.1316** | **65.4%** | **0.2975** | 2.8% |

**Reading the table**
- RMSE improves by **6.1%** and MAE by **6.8%** over the bias model. The model ordering on test matches validation, so tuning generalised.
- Ranking by *predicted rating alone* is poor (NDCG@5 ≈ 0.02) however accurate the ratings are. The watch-likelihood term is what makes the Top-5 list work.
- The final model has almost **twice popularity's NDCG@5**, while recommending about 8× more distinct movies (2.8% vs 0.4% of the catalogue).

### 7.3 Statistical significance (paired bootstrap over users, 2,000 resamples)

| Final hybrid minus… | Mean NDCG@5 difference | 95% CI | Users better / worse |
|---|---|---|---|
| Popularity | +0.140 | 0.123 to 0.156 | 315 / 62 |
| Rating-only hybrid | +0.273 | 0.252 to 0.295 | 386 / 14 |
| Implicit CF only | +0.026 | 0.016 to 0.036 | 214 / 118 |

All intervals are entirely above 0. The gain over implicit-only is small but reliable.

### 7.4 Segment analysis (test)

**By user activity** (ratings in train + validation)

| Group | Users | RMSE bias → hybrid | RMSE gain | NDCG@5 final | NDCG@5 popularity |
|---|---|---|---|---|---|
| Light (≤ 40) | 226 | 0.952 → 0.929 | 2.4% | 0.180 | 0.075 |
| Medium (41 to 120) | 205 | 0.904 → 0.859 | 5.0% | 0.286 | 0.138 |
| Heavy (> 120) | 173 | 0.848 → 0.789 | 6.9% | 0.466 | 0.291 |

The system beats popularity in every group. Light users, the hardest case, still score more than 2× popularity, but gains grow with history.

**By movie support** (number of ratings the movie had in train + validation)

| Ratings | Test pairs | RMSE bias → hybrid | Gain |
|---|---|---|---|
| 0 | 792 | 1.029 → 0.959 | 6.7% |
| 1 to 2 | 1,218 | 0.926 → 0.863 | 6.8% |
| 3 to 9 | 2,970 | 0.893 → 0.850 | 4.8% |
| 10 to 29 | 5,029 | 0.848 → 0.805 | 5.0% |
| 30+ | 10,155 | 0.846 → 0.787 | 6.9% |

**Calibration:** predicted and actual mean ratings agree closely in the central bins (e.g. 3.27 vs 3.26 and 3.75 vs 3.74). There is mild over-prediction at the top (4.69 predicted vs 4.54 actual in the highest bin) and about +0.1 in the 1.5 to 2.5 bins. This does not affect *ranking* (the order is preserved), but displayed star predictions could be shrunk slightly.

## 8. Cold-start handling

The cold-start diagnosis (validation RMSE by the number of training ratings of the movie) motivates the hybrid:

| Train ratings of movie | Pairs | Bias | Item-kNN | User-kNN | MF | **Content** |
|---|---|---|---|---|---|---|
| 0 | 433 | 0.988 | 0.988 | 0.988 | 0.988 | **0.933** |
| 1 to 2 | 664 | 0.932 | 0.932 | 0.931 | 0.934 | **0.892** |
| 3 to 9 | 1,573 | 0.866 | 0.860 | 0.862 | 0.860 | **0.851** |
| 10 to 29 | 2,775 | 0.847 | 0.831 | 0.840 | 0.838 | **0.824** |
| 30+ | 4,648 | 0.831 | **0.795** | 0.813 | 0.812 | 0.806 |

- **Cold movies (no ratings):** CF components give exactly the bias prediction (no information); the content model provides the gain. The blend therefore gives content the largest weight in the cold bucket.
- **Well-rated movies:** item-kNN is best, and the blend puts most weight on it.
- **Cold users:** every component can score an arbitrary rating row (fold-in), so a new user receives recommendations immediately. Quality grows with the amount of history (see §7.4).

## 9. Final system and API reference

All code is in the notebook. The packaged pipeline is the class `HybridRecommender` (Step 9.1).

### 9.1 `HybridRecommender(cfg, W_blend, X_content)`

| Method | Purpose |
|---|---|
| `fit(R, M)` | Fit bias, item-kNN neighbours, ALS factors and implicit-ALS factors on a dense rating matrix `R` (users × movies, float32) and mask `M` (bool) |
| `components(R, M, self_rows=None)` | Return `(base, comps, watch)` for any set of rating rows: bias prediction matrix, dict of residual predictions `{mf, iknn, uknn, cb}`, and watch scores. `self_rows` excludes a training user from their own neighbourhood |
| `rating_matrix(base, comps)` | Combine components into the hybrid predicted rating (unclipped) |
| `recommend_row(R_row, M_row, n=5, min_ratings=None, self_row=None)` | Top-`n` DataFrame for one rating vector |

`recommend_row` returns, per rank: `title`, `genres`, `pred. rating` (clipped to 0.5 to 5), `watch score`, `#ratings`, `because you liked` (the user's liked movie, rated ≥ 4, whose MF latent vector is most similar to the recommendation; a heuristic).

### 9.2 Helper functions

| Function | Purpose |
|---|---|
| `find_movie(query, top=5)` | Search titles; most-rated matches first |
| `user_vector(rated)` | `{title: rating}` → `(R_row, M_row)`; resolves each title with `find_movie` and warns on misses |
| `split_per_user(df, val_frac, test_frac, seed)` | Per-user random split |
| `to_dense(df)` | Ratings table → dense `(R, M)` |
| `fit_bias`, `row_bias` | Bias model fitting / user bias for arbitrary rows |
| `item_knn_neighbors`, `build_S`, `item_knn_predict` | Item-kNN |
| `user_knn_predict` | User-kNN |
| `als_fit`, `als_rows`, `mf_predict` | Matrix factorisation and fold-in |
| `content_predict` | Per-user ridge content model |
| `ials_fit`, `ials_rows`, `watch_scores` | Implicit ALS |
| `hybrid_rating_matrix`, `fit_blend`, `apply_blend`, `bucket_of` | Support-aware blend |
| `rating_metrics`, `topn_eval`, `log_result`, `edge_check` | Evaluation utilities |

### 9.3 Frozen configuration

```python
CONFIG = {
  "bias": {"lam_i": 2.0, "lam_u": 1.0},
  "iknn": {"shrink": 25, "K": 200, "damp": 0.25},
  "uknn": {"k": 200, "shrink": 20, "damp": 0.25},
  "mf":   {"k": 128, "lam": 0.1, "n_iter": 10},
  "cb":   {"lam": 20.0},
  "ials": {"k": 64, "alpha": 1.0, "lam": 20.0, "n_iter": 8},
  "beta": 8.0,
  "min_ratings_to_recommend": 5,
}
```

### 9.4 Outputs

`top5_recommendations.csv`: columns `userId, rec1, …, rec5` (titles, best first) for all 610 users, from the model fitted on 100% of the data. Across all users, **386 distinct movies** (4.0% of the catalogue) are recommended.

## 10. Worked examples

### 10.1 New user (supplies ratings)

```python
rated = {"Matrix, The (1999)": 5, "Blade Runner (1982)": 5, "Alien (1979)": 4.5,
         "Terminator 2: Judgment Day (1991)": 5,
         "Star Wars: Episode IV - A New Hope (1977)": 5,
         "Titanic (1997)": 2, "Notting Hill (1999)": 2}
R_row, M_row = user_vector(rated)
prod.recommend_row(R_row, M_row, n=5)
```

Observed output (Top-5): *Star Wars: Episode V*, *Star Wars: Episode VI*, *Raiders of the Lost Ark*, *Fight Club*, *Aliens*, with explanations pointing to *Star Wars: A New Hope*, *The Matrix* and *Alien*.

### 10.2 Existing user

```python
u = u2i[461]
prod.recommend_row(R_all[u], M_all[u], n=5, self_row=u)
```

For user 461 (27 ratings, favourites include *Wolf of Wall Street* and *Forrest Gump*), the Top-5 is *Pulp Fiction*, *American Beauty*, *Fight Club*, *The Godfather*, *American History X*.

### 10.3 Observed behaviour with very short histories

A user with only 3 ratings (*Forrest Gump* 5, *Pulp Fiction* 4, *Fight Club* 5) receives *Shawshank Redemption*, *Schindler's List*, *The Matrix*, *Braveheart*, *Silence of the Lambs*: popular, highly regarded titles. A 7-rating family/animation persona received a mix of broadly liked films (*Aladdin* among them) rather than a clearly animated list. With little evidence, the strongly regularised model leans on popularity; see §12.

## 11. Design decisions and rationale (FAQ)

**Why not just minimise RMSE?** RMSE measures how well *ratings* are predicted, but the product shows a *ranked list*. In this data the two diverge sharply: the rating-only hybrid improves RMSE by 6% yet its Top-5 NDCG@5 is ≈ 0.02, far below popularity (0.158). Both families of metrics are reported.

**Why is popularity such a strong baseline?** Users rate what they chose to watch, so held-out ratings concentrate on popular movies. Any ranking that favours popular movies scores well offline. That is why we also report coverage and average popularity, and why the implicit model must not simply collapse to popularity (it recommends about 8× more distinct movies).

**Why add an implicit-feedback model?** *Which* movies a user chose to rate is informative (exposure and taste). Rating-value models cannot see it. Implicit ALS learns it from the same user-movie matrix, so the system stays collaborative.

**Why `score = r̂ + β·p̂`?** Roughly *expected value = P(watch) × how much they like it*. β sets the trade-off and is tuned on validation. The plateau around β = 6 to 16 shows the result is robust. Higher β raises average popularity (log-popularity 2.2 → 4.8) and slightly lowers coverage (3.7% → 2.6%).

**Why model residuals?** Bias removal lets kNN/MF focus on taste, which improves accuracy and numerical behaviour on sparse data.

**Why a blend per support bucket?** The cold-start diagnosis showed content wins for rarely-rated movies and CF wins for well-rated ones. NNLS weights per bucket let the data decide, and a 5-fold CV on the validation pairs (RMSE 0.8145 vs 0.8134 in-sample) shows no meaningful overfitting.

**Why dense matrices?** At 610 × 9,742 they are only 24 MB each, so dense NumPy operations are simpler and faster than sparse code. For much larger data, switch to sparse structures.

**Why ridge per user for content?** A user's tastes over genres/eras are individual, and a small regularised linear model is stable even with 14 training ratings.

**Why a random per-user split rather than temporal?** The goal is predicting held-out ratings of known users, and yearly average ratings show no strong drift. A temporal split is a recommended extension.

## 12. Limitations

1. **Offline evaluation bias:** unrated movies are treated as non-relevant, so metrics are conservative and favour popular items. Only an online A/B test measures true impact.
2. **Small development dataset:** 610 users, extreme long tail; MovieLens itself advises against using this set for published benchmark results.
3. **Very short histories** (about 7 ratings or fewer) yield mostly popular, well-liked recommendations.
4. **No time/context:** no recency, seasonality or session signals.
5. **Blend weights** are learned on a single validation split (CV shows little overfitting, and the test results agree), but repeated splits would give tighter estimates.
6. **Explanations** ("because you liked…") are heuristic, based on latent similarity, and can be noisy for rarely-rated movies.
7. **Packaging:** the class lives in the notebook rather than an installable library; there is no persistence (save/load) of the fitted model.
8. **Never-rated movies:** scored only by the content model, and excluded from deployed recommendations by default (`min_ratings_to_recommend = 5`). Their watch score is 0, since the implicit model has no data for them.

## 13. Extending the system

| Goal | Suggestion |
|---|---|
| Recommend more or fewer items | Pass `n=` to `recommend_row`; change `K_TOP` in Step 4 for evaluation |
| Different notion of "liked" | Change `REL_THR` in Step 4 |
| Allow new/rare movies | Call `recommend_row(..., min_ratings=0)` (content model only for never-rated movies) |
| Different split | Edit the fractions in `split_per_user`, or implement a temporal split |
| Richer content | Add columns to `X_content` (e.g. plot embeddings, cast/director via `links.csv` and TMDb/IMDb), then re-tune the content ridge |
| Better ranking objective | Replace NNLS + β with a learning-to-rank model (BPR / LambdaRank) trained on the component scores |
| Diversity | Re-rank the top-N with MMR and report intra-list diversity |
| Time awareness | Add time-decayed weights or a temporal split |
| Productionisation | Move the class and helpers to a Python package, add `save`/`load`, serve `recommend_row` behind an API |
| Adding new movies | Append them to `movies`, extend `X_content` and the movie index, and refit (the CF parts need ratings; content works immediately) |

## 14. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: Place 'ml-latest-small/' ...` | The dataset is not next to the notebook. Put the folder or the zip in the notebook's working directory |
| `! not found: <title>` from `user_vector` | The title was not found. Use `find_movie("part of title")` to get the exact MovieLens title (with the year) |
| `NameError: name 'prod' is not defined` | Step 10.1 has not been run. Run the notebook top to bottom (or at least Steps 1 to 10.1) before calling the recommender |
| Long run time | Most time goes to the grid searches (kNN, ALS k = 128, iALS). Reduce the grids or `n_iter` to speed up, accepting slightly different winners |
| Slightly different numbers than documented | Different BLAS/NumPy versions can change floating-point results in the last digits; the seed is fixed (`SEED = 42`). Conclusions are unaffected |
| `IndexError`/shape errors after editing | Matrices are `n_users × n_items`; row/column indices come from `u2i`/`m2i`. Re-run the preparation steps after changing the data |

## 15. Reproducibility

- Single seed `SEED = 42` for the split, ALS initialisations, SVD, CV folds and bootstrap.
- Tested stack: NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2.
- CPU only; no external services. All algorithms are implemented in the notebook.
- The notebook delivered here was executed end to end with no errors, and the figures quoted in this document match its outputs.

## 16. Glossary

| Term | Meaning |
|---|---|
| **Collaborative filtering (CF)** | Recommending from patterns in the user-item matrix (who rated/liked what), without item content |
| **Content-based filtering** | Recommending from item attributes (genres, decade, tags) matched to a user's taste profile |
| **Hybrid** | A combination of CF and content-based methods |
| **Cold start** | Too little data for a new user or item; CF cannot score items with no ratings |
| **Residual** | Rating minus the bias-model prediction (the "taste" part) |
| **Shrinkage / damping** | Regularisation that pulls estimates built on little evidence toward a neutral value |
| **ALS / ALS-WR** | Alternating Least Squares; the "weighted regularisation" variant scales the penalty by the number of ratings |
| **Implicit feedback** | Behaviour signals (here: *which* movies a user rated) rather than rating values |
| **Fold-in** | Computing a new user's representation from their ratings with the item-side model fixed (no retraining) |
| **NNLS** | Non-negative least squares, used to learn blend weights |
| **NDCG@5** | Normalised discounted cumulative gain on the top 5; rewards relevant items placed near the top |
| **Coverage** | Share of the catalogue that is ever recommended |
| **Exposure bias** | Users only rate what they were exposed to or chose to watch, which biases offline evaluation toward popular items |

## 17. Data license and citation

The MovieLens data may be used for **research purposes** under the GroupLens conditions: do not state or imply endorsement by the University of Minnesota or GroupLens; acknowledge the dataset in publications; redistribution is allowed under the same terms; **no commercial or revenue-bearing use without permission**. See the dataset's own `README.txt`.

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>
