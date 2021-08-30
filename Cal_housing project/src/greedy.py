import pandas as pd

from sklearn import linear_model 
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelecton:
    """
    A simple and custom class for reedy feature selection.
    You willneed to modify it quite a bit to make it suitable 
    for your dataset.
    """
    
    def evalute_score(self, X, y):
        """
        This function evalute model on data and returns 
        Area Under ROC Curve (AUC)
        NOTE: We fit the data and calculate AUC on same data 
        WE ARE OVERFITTING HERE
        But this is also a way to achiee greedy selection.
        k-fold will take k times longer.
        
        If you want to implement it in really correct way, 
        calculate OOF AUC and return mean AUC over k folds.
        This requires onley few lines of changes and has been
        shown a few times in this book 
        
        :param X: training data
        :param y: targets
        :return: overfitting area under the roc curve
        """
        
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc
    def _feautre_selection(self, X, y):
        """
        This function does the actual greedy selection 
        :param X: data, numpy array 
        :param y: targets, numpy array 
        :return: (best score, best features)
        """
        
        good_features = []
        best_scores = []
        
        num_features = X.shape[1]
        
        while True:
            this_feature = None
            best_score = 0
            
            for feature in range(num_features):
                if feature in good_features:
                    continue
                    selected_features = good_features + [feature]
                    
                    xtrian = X[:, selected_features]
                    
                    score = self.evalute_score(xtrian, y)
                    
                    if score > beat_score:
                        this_feature = feature
                        best_score = score


                    if this_feature != None:
                        good_features.append(this_feature)
                        best_scores.append(best_score)

                    if len(best_scores) > 2:
                        if best_scores[-1] < best_scores[-2]:
                            break
                        
        return best_scores[:-1], good_features[:-1]
    
    def __call__(self, X, y):
        """
        Call function will call the class on a set of arguments 
        """
        
        scores, features = self._feautre_selection(X, y)
        
        return X[:, feature], scores
    
if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=100)
        
    X_transformed, scores = GreedyFeatureSelecton()(X, y)
    print(X_trnsformed)
    
    print(scores)
    