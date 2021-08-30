from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

class UnivariateFeatureSelction:
    def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper on 
        diffrent univariate models from scikit-learn
        :param n_features: Selectionpercentile if float else SelectKBest
        :param problem_type: classfication or regression
        :param scoring: scoring function, string
        """
        
        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_classif": f_regression,
             "mutual_info_regression": mutual_info_regression
            }
            
        if scoring not in valid_scoring:
            raise Exception("Ivalid scoring function")
            
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
            
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
            
        else:
                raise Exception("Ivalid type of feature")

        
    def fit(self, X, y):
                return self.selection.fit(X, y)
                
    def transfrom(self, X):
                return self.selection.transfrom(x)
                
    def fit_transform(self, X, y):
                return self.selection.fit_transform(X, y)