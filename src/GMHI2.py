import numpy as np
from sklearn.base import BaseEstimator

class GMHI2(BaseEstimator):
    def __init__(self):
        self.fitted = False
    
    def fit(self, X, y):
        """X is dataframe (num_examples, num_features). y is dataframe (num_examples, 1)"""
        self.fitted = True
        
        # convert to numpy
        X = X.values
        y = y.values
        
        m = X.shape[0] # examples
        n = X.shape[1] # features
        
        # get healthy and unhealthy samples
        healthies = X[y.flatten(), :]
        unhealthies = X[~y.flatten(), :]
        
        # get number of healthy and unhealthy samples
        num_healthy = healthies.shape[0]
        num_unhealthy = unhealthies.shape[0]
        
        # for each feature, see the proportion of samples with a 1 as its value for that feature
        prop_healthy = healthies.mean(axis=0)
        prop_unhealthy = unhealthies.mean(axis=0)
        
        # to avoid divide by zero, replace zero with smallest possible nonzero proportion for each feature
        prop_healthy[prop_healthy == 0] = 1 / num_healthy
        prop_unhealthy[prop_unhealthy == 0] = 1 / num_unhealthy
        
        # calculate theta with difference and fold changes
        diff = prop_healthy - prop_unhealthy
        foldh = prop_healthy / prop_unhealthy
        foldn = prop_unhealthy / prop_healthy
        
        
        theta = diff * np.log(np.maximum(foldh, foldn))
        theta_positive = theta.copy()
        theta_negative = theta.copy()
        
        theta_positive[theta_positive < 0] = 0
        theta_negative[theta_negative > 0] = 0
        
        score_positive = (healthies @ theta_positive).mean()
        score_negative = (-1 * (unhealthies @ theta_negative)).mean()
        
        theta_positive /= score_positive
        theta_negative /= score_negative
        
        self.theta = theta_positive + theta_negative
    
    def predict(self, X):
        """Returns predictions for X"""
        if not self.fitted:
            return None
        return (X @ self.theta) > 0
