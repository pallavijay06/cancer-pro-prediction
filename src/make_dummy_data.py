"""Create a stand-in dataset (same shape/format as the real one) so the modeling
code can be built and tested before the real dataset.csv exists.
NOT real data. Output is git-ignored."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

rng = np.random.default_rng(42)
X, y = make_classification(n_samples=217, n_features=30, n_informative=6,
                           weights=[0.71, 0.29], flip_y=0.1, random_state=42)
df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(30)])
df["cat_surgery"] = rng.choice(["BCS", "Mastectomy"], len(df))
df["cat_stage"] = rng.choice(["I", "II", "III"], len(df), p=[0.3, 0.6, 0.1])
df = df.mask(rng.random(df.shape) < 0.05)          # ~5% missing values (NaN)
df["y_drop"] = y
df["y_threshold"] = (y + (rng.random(len(y)) < 0.15)) % 2
df.to_csv("data/processed/dummy_dataset.csv", index=False)
print(df.shape, df[["y_drop", "y_threshold"]].mean().round(2).to_dict())
