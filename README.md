# Türk Telekom Big Data Camp 2025

My two projects from Türk Telekom's data science bootcamp (February 2025), where I was selected among 20 of about 3,000 applicants and placed 3rd. The bootcamp repository with the course material is [husnusensoy/tt-bootcamp](https://github.com/husnusensoy/tt-bootcamp); this repository keeps only my own code.

## `churn/`: who is about to leave

Capstone task: rank 10 million customers by churn risk so that the call centre and marketing teams know whom to contact first. The churn label is rare, 0.3% of broadband customers and about 1.9% of postpaid and prepaid customers, so accuracy is meaningless and the models are judged on AUC, recall and the quality of the ranking.

- **Data.** Ten JSONL shards, read with PySpark, split by service type (broadband, postpaid, prepaid) and written to Parquet. Segment-specific column cleaning and imputation (`group_by.py`, `missing_value.py`, `parquet_check.py`).
- **Models.** One XGBoost classifier per segment with `scale_pos_weight` for the class imbalance, 5-fold stratified cross-validation, and a decision threshold chosen on the out-of-fold predictions (largest geometric mean of TPR and 1 - FPR) rather than the default 0.5 (`*_xgboost.py`).
- **Explanations.** For the 20 customers just above the threshold in each segment, DiCE counterfactuals over the features the business can act on (satisfaction score, data usage, monthly charge, support calls, ...), with per-feature weights so that a change is measured relative to the feature's range (`*_counterfactual.py`).

Test-set results (15% hold-out, threshold from cross-validation):

| Segment | AUC | Recall | Balanced accuracy |
|---|---|---|---|
| Broadband | 0.62 | 0.57 | 0.61 |
| Postpaid | 0.78 | 0.78 | 0.72 |
| Prepaid | 0.71 | 0.69 | 0.64 |

Precision is low in every segment (0.5% to 4%), as expected at these base rates. At the chosen thresholds the flagged lists catch 57% to 78% of churners while carrying 1.6 to 2.2 times the churn rate of the customer base; the lists are meant to be worked from the top, ordered by predicted probability. Plots, confusion matrices and metric files are under `churn/*_results/`, and the presentation given to the panel is `churn/presentation.pdf` (in Turkish).

## `recommender/`: a movie recommender in SQL

Exercise on a Netflix-style ratings dump (four `rating_*.txt` files and a movie catalogue), done entirely in DuckDB with Jinja-templated SQL.

- `top_movies.py`: the 30 best films by a Bayesian average that shrinks a film's mean rating towards the global mean in proportion to how few ratings it has, so that a film with three perfect ratings does not top the list.
- `user_recommendations.py`: personalised recommendations for one user. Finds the 50 most similar users by Jaccard similarity of watched films, collects what they watched and the target user did not, and ranks the candidates by similarity-weighted Bayesian score.
- `movie_similarity.py`: compares pairs of films by the KL divergence between their (Bayesian-smoothed) rating distributions, to flag near-duplicates in the catalogue.

## Running

```bash
pip install -r requirements.txt
export TT_DATA=/path/to/capstone/data        # ten capstone.N.jsonl shards
export BINGE_DATA=/path/to/binge             # rating_*.txt and movie_titles.csv
python churn/group_by.py && python churn/missing_value.py && python churn/broadband_xgboost.py
python recommender/top_movies.py
```

The data is not included; it was provided by the bootcamp.
