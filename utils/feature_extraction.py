import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


def tune_max_features(data, n_iter=5, random_state=42):
    """
    Tune max_features for TfidfVectorizer using RandomizedSearchCV.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'text' column.
        n_iter (int): Number of random search iterations.
        random_state (int): Random seed.

    Returns:
        int: Best max_features value.
    """

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english'))
    ])

    param_dist = {
        'tfidf__max_features': np.random.randint(100, 5000, 20),
    }

    def score_func(estimator, X, y=None):
        tfidf_matrix = estimator.named_steps['tfidf'].fit_transform(X)

        # Density / variance score
        return np.var(tfidf_matrix.data)

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=score_func,
        cv=2,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    random_search.fit(data["text"])

    best_max_features = random_search.best_params_["tfidf__max_features"]

    print(f"Best max_features: {best_max_features}")

    return best_max_features


def train_feature_extractor(
    data,
    save_path=VECTORIZER_PATH,
    tune=True
):
    """
    Train TF-IDF vectorizer and save it.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'text'
        save_path (str): path to save model
        tune (bool): whether to tune max_features

    Returns:
        X_tfidf
        feature_names
        top_words_dict
        vectorizer
    """

    # Tune max_features
    if tune:
        best_max_features = tune_max_features(data)
    else:
        best_max_features = 5000

    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=best_max_features,
        stop_words="english"
    )

    # Fit + transform
    X_tfidf = vectorizer.fit_transform(data["text"])

    # Save vectorizer
    joblib.dump(vectorizer, save_path)

    print(f"Vectorizer saved to: {save_path}")

    # Feature names
    feature_names = vectorizer.get_feature_names_out()

    # Mean TF-IDF score
    tfidf_scores = np.mean(X_tfidf.toarray(), axis=0)

    # Top words
    top_words_indices = np.argsort(tfidf_scores)[::-1]

    top_words_dict = {
        feature_names[i]: float(tfidf_scores[i])
        for i in top_words_indices
    }

    return X_tfidf, feature_names, top_words_dict, vectorizer


def load_vectorizer(model_path=VECTORIZER_PATH):
    """
    Load saved TF-IDF vectorizer.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Vectorizer not found at {model_path}"
        )

    vectorizer = joblib.load(model_path)

    print("Vectorizer loaded successfully.")

    return vectorizer


def transform_text(texts, model_path=VECTORIZER_PATH):
    """
    Transform new production texts using saved vectorizer.

    Parameters:
        texts (list[str] or str)

    Returns:
        TF-IDF transformed matrix
    """

    vectorizer = load_vectorizer(model_path)

    # Handle single string
    if isinstance(texts, str):
        texts = [texts]

    X = vectorizer.transform(texts)

    return X
