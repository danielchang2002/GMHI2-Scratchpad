from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

"""
Class that implements the GMHI algorithm. Extends sklearn base estimator for
cross validation compatibility
"""

class GMHI(BaseEstimator):

    def __init__(self, use_shannon = False, theta_f = 1, theta_d = 0):
        self.use_shannon = use_shannon
        self.fitted = False
        self.thresh = 0.00001
        self.health_abundant = []
        self.health_scarce = []
        self.theta_f = theta_f
        self.theta_d = theta_d

    def fit(self, X, y):
        """
        Based on theta_f and theta_d, 
        """
        self.fitted = True

        difference, fold_change = self.get_proportion_comparisons(X, y)

        self.select_features(difference, fold_change)

    def get_proportion_comparisons(self, X, y):
        """
        Returns difference and fold changes between healthy and unhealthy
        nonzero proportions for each feature
        """

        # get healthy and unhealthy samples
        healthies = X.iloc[y.values, :]
        unhealthies = X.iloc[~y.values, :]
        
        # get proportions for each species
        proportion_healthy = self.get_proportions(healthies)
        proportion_unhealthy = self.get_proportions(unhealthies)
        
        # get differences and fold change
        diff = proportion_healthy - proportion_unhealthy
        fold = proportion_healthy / proportion_unhealthy
        return diff, fold

    def select_features(self, difference, fold_change):
        """
            sets health_abundant and health_scarce
        """
        
        # based on proportion differences and fold change, select health abundant
        # and health scarce
        self.health_abundant = self.cutoff(difference, fold_change)
        self.health_scarce = self.cutoff(-1 * difference, 1 / fold_change)

    def get_psi(self, X):
        psi = self.richness(X)
        if self.use_shannon:
            shan = self.shannon(X)
            psi *= shan
        return psi

    def cutoff(self, diff, fold):
        return list(diff[
            (diff['Proportion'] > self.theta_d) & (fold['Proportion'] > self.theta_f)
        ].index)

    def get_proportions(self, df):
        p = (df > self.thresh).sum() / df.shape[0]
        proportion = pd.DataFrame({"Proportion" : p})
        return proportion

    def predict_raw(self, X):
        if not self.fitted:
            return None
        X[X < self.thresh] = 0
        X_healthy_features = X[self.health_abundant]
        X_unhealthy_features = X[self.health_scarce]
        psi_MH = self.get_psi(X_healthy_features) / (
                X_healthy_features.shape[1])
        psi_MN = self.get_psi(X_unhealthy_features) / (
                X_unhealthy_features.shape[1])
        num = psi_MH + self.thresh
        dem = psi_MN + self.thresh
        return np.log10(num / dem)

    def predict(self, X):
        return self.predict_raw(X) > 0

    def score(self, X, y):
        """
            Returns balanced accuracy, chi
        """
        pred = self.predict(X)
        score = balanced_accuracy_score(y, pred)
        return score

    def get_features(self):
        """
            Returns the lists of health abundant and health scarce 
            features as a tuple, if fitted
        """
        if not self.fitted:
            return None
        return self.health_abundant, self.health_scarce

    def richness(self, X):
        """
        Returns the number of nonzero values for each sample (row) in X
        """
        frame = pd.DataFrame((X > self.thresh).sum(axis=1))
        return frame

    def shannon(self, X):
        """
        Returns the shannon diversity for each sample (row) in X
        """
        logged = np.log(X[X > 0])
        logged.fillna(0, inplace=True)
        shan = logged * X * -1
        return pd.DataFrame(shan.sum(axis=1))

    def get_thetas(self):
        """
            Returns the best fold change and difference thresholds
        """
        return self.theta_f, self.theta_d
