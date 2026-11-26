from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


def processing(path: Path | str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    mapping = {"feature": 0, "bug": 1, "doc": 2, "question": 3}
    y = df["label"].map(mapping)
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna(""))
    X = df.drop(columns=["title","body","label"])
    if y.isna().any():
        raise ValueError(f"There  is a NA value {df.loc[y.isna(), "label"].unique()}")
    return X, y

def encoding_model(model: SentenceTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input the text column and returns the embedding
    """
    train_temp = X_train["text"].tolist()
    val_temp = X_val["text"].tolist()
    test_temp = X_test["text"].tolist()
    train_emb = model.encode(train_temp, batch_size=32, show_progress_bar=True)
    if len(train_emb) != len(X_train):
        raise ValueError("length of embedding differs from X_train")
    val_emb = model.encode(val_temp, batch_size=32, show_progress_bar=True)
    if len(val_emb) != len(X_val):
        raise ValueError("length of embedding differs from X_val")
    test_emb = model.encode(test_temp, batch_size=32, show_progress_bar=True)
    if len(test_emb) != len(X_test):
        raise ValueError("length of embedding differs from X_test")
    return train_emb,val_emb,test_emb

def split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.3,random_state=42,stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,y_temp, test_size=0.5,random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


