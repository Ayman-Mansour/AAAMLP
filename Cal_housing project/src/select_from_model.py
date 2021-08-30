import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]

model = RandomForestRegressor()

sfm = SelectFromModel(estimator=model)

x_transformed = sfm.fit(X, y)

support = sfm.get_support()

print(
    [
        x for x,y in zip(col_names, support) if y == True
    ]
)


