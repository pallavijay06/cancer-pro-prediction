"""CP1 smoke test: one model, end to end, on the dummy data."""
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate import nested_cv, summarize

df = pd.read_csv("data/processed/dummy_dataset.csv")
y = df["y_drop"]
X = df.drop(columns=["y_drop", "y_threshold"])

# Stand-in for features.get_preprocessor() that your teammate will provide
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), sel(dtype_include="number")),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]),
     sel(dtype_exclude="number")),
])
pipe = Pipeline([("prep", prep),
                 ("clf", LogisticRegression(max_iter=2000))])  # default penalty = L2 (Ridge)
grid = {"clf__C": [0.01, 0.1, 1, 10]}

res = nested_cv(pipe, grid, X, y)
print(summarize("Ridge (dummy)", res))
