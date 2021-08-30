# src/create_folds.py.strafiedFold

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/cat_train.csv")
    
    df["fold"] = -1
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    y = df.target.values
    
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
        
    df.to_csv("../input/cat_train_folds.csv", index=False)