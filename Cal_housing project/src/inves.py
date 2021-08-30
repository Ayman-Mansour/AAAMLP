import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
x = data["data"]
cols_names = data["feature_names"]
y = data["target"]

df = pd.DataFrame(x, columns=cols_names)

df.loc[:, "MedInc_Sqrt"] = df.MedInc.apply(np.sqrt)

df.corr