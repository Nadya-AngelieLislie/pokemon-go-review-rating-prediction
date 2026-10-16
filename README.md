# Pokemon GO Review Rating Prediction

> Predicting 1-5 star ratings from Google Play review text, comparing classical ML and deep learning.

## Overview

- Solo midterm project for a Text Mining / NLP course: predict a reviewer's star rating from the review text alone.
- Reviews of Pokemon GO were scraped from Google Play, cleaned, and used to train and compare machine learning and deep learning classifiers.

## Data / Scope

- English Pokemon GO reviews scraped with `google-play-scraper`, language-validated and length-filtered (very short and very long reviews removed).
- The final dataset (`data/pokemon_go_reviews_new.csv`) has about 8,800 reviews with columns `review` and `rating`.
- Classes are imbalanced and bimodal: 1-star is the largest class (3,821 reviews in the EDA sample) and 5-star the second (2,242), while 2-4 stars are much smaller.

## Workflow

1. Scraping and filtering (language check, length filter).
2. EDA: rating distribution, dominant words per rating, noisy non-standard text, word clouds.
3. Preprocessing: contraction handling, emoji converted to text (emoji carry strong sentiment), stopword handling, lemmatization, with separate pipelines for ML and DL.
4. Text representation: TF-IDF and Word2Vec for ML; tokenized sequences for DL.
5. Modelling: Random Forest, Linear SVM, Bagging, XGBoost and others (ML); Dense, LSTM, GRU and CNN (DL).
6. Hyperparameter tuning of the top four ML models and the CNN.
7. Imbalance handling: class weight, SMOTE (ML) and class weight, random oversampling (DL).

## Results

- Best overall model: tuned Random Forest on TF-IDF (accuracy 0.66, macro F1 0.57).
- Tuned Linear SVM on TF-IDF (macro F1 0.56) and tuned Random Forest on Word2Vec (macro F1 0.558) followed closely.
- Best deep learning model: CNN (baseline macro F1 0.547); tuning did not help (0.515).
- Imbalance handling: SMOTE gave the best Random Forest macro F1 (0.563); random oversampling lifted the CNN from 0.440 to 0.566 macro F1, while class weight hurt it (0.318).
- Takeaway: with roughly 9,000 short, informal reviews, classical TF-IDF models were competitive with or better than the deep learning models.

## Tech Stack

Python, google-play-scraper, NLTK, scikit-learn, imbalanced-learn (SMOTE), Gensim (Word2Vec), XGBoost, TensorFlow/Keras

## Repository Structure

```
notebook/    Full analysis notebook (scraping to conclusion)
data/        Scraped and filtered review dataset (CSV)
figures/     Key charts from the notebook
requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook/Pokemon_GO_Text_Mining.ipynb
```

Adjust the CSV path in the data-loading cell to `data/pokemon_go_reviews_new.csv`. Re-scraping is optional because the dataset is included.

## Team

Individual project by Nadya Angelie Lislie (Text Mining course, BINUS University).

## Notes

- The notebook is written in Indonesian with English code and outputs.
- Reviews are public Google Play text and contain no usernames.

## Author

Nadya Angelie Lislie, BINUS University (Computer Science and Statistics). [LinkedIn](https://www.linkedin.com/in/nadya-angelie-lislie/) | [GitHub](https://github.com/Nadya-AngelieLislie)

## Experiment Tracking

`src/experiment_tracking.py` provides an optional MLflow wrapper (`track_run`) for logging parameters, metrics and artifacts; it does nothing when MLflow is not installed. Run `mlflow ui` after logging runs to compare experiments.

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff` and `pytest` repository-health checks on every push; run them locally with `make lint test`.
