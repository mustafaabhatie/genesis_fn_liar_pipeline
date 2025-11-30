#!/usr/bin/env python3
# genesis_fn_liar_pipeline.py
#
# End‑to‑end Genesis‑FN style fake‑news detection pipeline for the LIAR dataset.
# Steps:
#   1. Load LIAR train.tsv
#   2. Text cleaning
#   3. Feature construction:
#        - TF‑IDF on statement text
#        - One‑hot categorical metadata
#        - Numeric history counts
#        - BERT sentence embeddings (bert-base-uncased)
#   4. Three‑stage feature selection:
#        - Decision Tree importance
#        - RFE with linear SVM
#        - Linear Regression coefficients
#   5. Train final classifier (linear SVM) and report metrics.
#
# All random seeds, hyperparameters, and file paths are fixed for reproducibility.

import os
import re
import string
import json
import random
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder

import torch
from transformers import BertTokenizer, BertModel

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ----------------------------------------------------------------------
# Configuration and reproducibility
# ----------------------------------------------------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DATA_FILE = "train.tsv"          # path to LIAR training split
BERT_MODEL_NAME = "bert-base-uncased"
BERT_CACHE_DIR = "./bert_cache"  # created automatically if missing

OUTPUT_DIR = "./outputs"         # where to store selected features, metrics, etc.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default hyperparameters (make sure they match the paper)
TEST_SIZE = 0.2
BATCH_BERT = 32               # batch size for BERT encoding
IMPORTANCE_THRESHOLD = 0.01   # DecisionTree feature-importance threshold
RFE_N_FEATURES = 50           # number of features kept by RFE
LR_PERCENTILE = 80            # top 20% features by |coef|


# ----------------------------------------------------------------------
# Utility: text preprocessing
# ----------------------------------------------------------------------

def init_nltk():
    """Download NLTK resources if not already present."""
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


def build_preprocess_fn():
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r"\d+", " ", text)
        tokens = [w for w in text.split() if w not in stop_words]
        tokens = [stemmer.stem(w) for w in tokens]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)

    return preprocess_text


# ----------------------------------------------------------------------
# Step 1: Data loading and label binarization
# ----------------------------------------------------------------------

def load_liar_data(path: str) -> pd.DataFrame:
    """
    Load LIAR train.tsv with the canonical 14 columns.

    Columns:
    id, label, statement, subject, speaker, job, state, party,
    barely_true_c, false_c, half_true_c, mostly_true_c, pants_on_fire_c, venue
    """
    col_names = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "job",
        "state",
        "party",
        "barely_true_c",
        "false_c",
        "half_true_c",
        "mostly_true_c",
        "pants_on_fire_c",
        "venue",
    ]
    df = pd.read_csv(path, sep="\t", names=col_names)
    return df


def binarize_labels(series: pd.Series) -> pd.Series:
    """
    Map LIAR labels to binary:
        {true, mostly-true, half-true} -> 1
        others -> 0
    """
    positive = {"true", "mostly-true", "half-true"}
    return series.apply(lambda x: 1 if x in positive else 0)


# ----------------------------------------------------------------------
# Step 2: Feature construction
# ----------------------------------------------------------------------

def build_tfidf_features(statements):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(statements)
    return X_tfidf, vectorizer


def build_categorical_features(df):
    categorical_cols = ["subject", "speaker", "job", "state", "party", "venue"]
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[categorical_cols])
    return X_cat, encoder, categorical_cols


def build_numeric_features(df):
    numeric_cols = [
        "barely_true_c",
        "false_c",
        "half_true_c",
        "mostly_true_c",
        "pants_on_fire_c",
    ]
    X_num = df[numeric_cols].astype(float).values
    return csr_matrix(X_num), numeric_cols


def load_bert():
    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL_NAME, cache_dir=BERT_CACHE_DIR
    )
    model = BertModel.from_pretrained(
        BERT_MODEL_NAME, cache_dir=BERT_CACHE_DIR
    )
    model.eval()
    return tokenizer, model


def encode_with_bert(statements, tokenizer, model, batch_size=BATCH_BERT):
    """
    Mean-pool last hidden state to obtain one vector per statement.
    """
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(statements), batch_size):
            batch = list(statements[i : i + batch_size])
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_emb)
    return np.vstack(embeddings)


def build_all_features(df, use_bert=True):
    """
    Build TF‑IDF + categorical + numeric + optional BERT features.
    Returns:
        X_combined (scipy.sparse CSR)
        feature_metadata (dict) – for documentation and reuse
    """
    preprocess_text = build_preprocess_fn()
    X_text_clean = df["statement"].apply(preprocess_text)

    # TF‑IDF
    X_tfidf, tfidf_vectorizer = build_tfidf_features(X_text_clean)

    # Categorical
    X_cat, cat_encoder, categorical_cols = build_categorical_features(df)

    # Numeric
    X_num, numeric_cols = build_numeric_features(df)

    # BERT
    if use_bert:
        tokenizer, bert_model = load_bert()
        X_bert = encode_with_bert(
            X_text_clean.tolist(), tokenizer, bert_model
        )
        X_bert = csr_matrix(X_bert)
    else:
        X_bert = csr_matrix((df.shape[0], 0))

    # Combine
    X_combined = hstack(
        [csr_matrix(X_tfidf), X_bert, X_cat, X_num],
        format="csr",
    )

    feature_metadata = {
        "tfidf_vocab_size": int(X_tfidf.shape[1]),
        "use_bert": use_bert,
        "bert_dim": 0 if not use_bert else int(X_bert.shape[1]),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }

    return X_combined, feature_metadata, tfidf_vectorizer, cat_encoder


# ----------------------------------------------------------------------
# Step 3: Feature selection (three methods combined)
# ----------------------------------------------------------------------

def three_stage_feature_selection(X_train, y_train):
    """
    Apply:
      1) Decision Tree importance
      2) RFE with linear SVM
      3) Linear Regression coefficients
    Then take the union of selected indices.
    """
    # Convert to dense for models that do not support sparse
    X_train_dense = X_train.toarray()

    # 1) Decision Tree
    dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
    dt.fit(X_train_dense, y_train)
    dt_importance = dt.feature_importances_
    selected_dt = np.where(dt_importance > IMPORTANCE_THRESHOLD)[0]

    # 2) RFE + SVM
    svm = SVC(kernel="linear", random_state=RANDOM_SEED)
    # Limit n_features_to_select to number of available features
    n_features_rfe = min(RFE_N_FEATURES, X_train_dense.shape[1])
    rfe = RFE(estimator=svm, n_features_to_select=n_features_rfe, step=1)
    rfe.fit(X_train_dense, y_train)
    selected_rfe = np.where(rfe.support_)[0]

    # 3) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_dense, y_train)
    lr_coeff = np.abs(lr.coef_)
    threshold = np.percentile(lr_coeff, LR_PERCENTILE)
    selected_lr = np.where(lr_coeff >= threshold)[0]

    # Union
    selected_all = np.union1d(selected_dt, np.union1d(selected_rfe, selected_lr))

    # Save selection statistics
    stats = {
        "n_features_total": int(X_train_dense.shape[1]),
        "n_features_dt": int(len(selected_dt)),
        "n_features_rfe": int(len(selected_rfe)),
        "n_features_lr": int(len(selected_lr)),
        "n_features_union": int(len(selected_all)),
        "importance_threshold": float(IMPORTANCE_THRESHOLD),
        "rfe_n_features": int(n_features_rfe),
        "lr_percentile": int(LR_PERCENTILE),
    }

    return selected_all, stats


# ----------------------------------------------------------------------
# Step 4: Final classifier training and evaluation
# ----------------------------------------------------------------------

def train_and_evaluate(X_train_sel, y_train, X_test_sel, y_test):
    """
    Train a linear SVM on the selected features and return evaluation metrics.
    """
    clf = SVC(kernel="linear", random_state=RANDOM_SEED)
    clf.fit(X_train_sel, y_train)

    y_pred = clf.predict(X_test_sel)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return clf, metrics


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Genesis‑FN style fake‑news detection pipeline on LIAR."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_FILE,
        help="Path to LIAR train.tsv file.",
    )
    parser.add_argument(
        "--no_bert",
        action="store_true",
        help="Disable BERT embeddings (for faster, text‑only runs).",
    )
    parsed = parser.parse_args(args)

    # 1) Data loading
    print("Loading data...")
    df = load_liar_data(parsed.data_path)
    y = binarize_labels(df["label"])

    # 2) Feature construction
    print("Building features (TF‑IDF + categorical + numeric + BERT)...")
    init_nltk()
    X_all, feat_meta, tfidf_vec, cat_enc = build_all_features(
        df, use_bert=not parsed.no_bert
    )

    # 3) Train‑test split
    print("Splitting into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # 4) Feature selection
    print("Running three‑stage feature selection...")
    selected_idx, fs_stats = three_stage_feature_selection(X_train, y_train)
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    # 5) Final classifier training and evaluation
    print("Training final classifier and evaluating...")
    clf, eval_metrics = train_and_evaluate(X_train_sel, y_train, X_test_sel, y_test)

    # 6) Save artifacts for reuse
    artifacts = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "bert_model_name": BERT_MODEL_NAME,
        "use_bert": not parsed.no_bert,
        "feature_metadata": feat_meta,
        "feature_selection_stats": fs_stats,
        "selected_feature_indices": selected_idx.tolist(),
        "metrics": eval_metrics,
    }
    with open(os.path.join(OUTPUT_DIR, "run_summary.json"), "w") as f:
        json.dump(artifacts, f, indent=2)

    print("\n=== Run summary ===")
    print(json.dumps(artifacts["metrics"], indent=2))
    print(f"\nSelected {fs_stats['n_features_union']} features out of "
          f"{fs_stats['n_features_total']} total.")
    print(f"Full summary written to {os.path.join(OUTPUT_DIR, 'run_summary.json')}.")


if __name__ == "__main__":
    main()
