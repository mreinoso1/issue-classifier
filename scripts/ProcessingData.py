from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from issue_classifier.processing.processing import encoding_model, processing, split

root = Path(__file__).resolve().parents[1]
X, y = processing(path = root/ "data" / "raw"/ "Issues.parquet")
train_full,val_full,test_full,y_train,y_val,y_test = split(X,y)
model = SentenceTransformer("all-MiniLM-L6-v2")
train_emb, val_emb, test_emb = encoding_model(model,train_full,val_full,test_full)
# ---- Saving values ----
train_full.to_parquet(path = root / "data" / "processed" / "train_full.parquet", index=False)
val_full.to_parquet(path = root / "data" / "processed" / "val_full.parquet", index=False)
test_full.to_parquet(path=root / "data" / "processed" / "test_full.parquet", index=False)

np.save(root / "data" / "processed" / "train_emb.npy", train_emb)
np.save(root / "data" / "processed" / "val_emb.npy", val_emb)
np.save(root / "data" / "processed" / "test_emb.npy", test_emb)

np.save(root / "data" / "processed" / "y_train.npy", y_train)
np.save(root / "data" / "processed" / "y_val.npy", y_val)
np.save(root / "data" / "processed" / "y_test.npy", y_test)