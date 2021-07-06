from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

"""
Class that implements the GMHI algorithm. Extends sklearn base estimator for
cross validation compatibility
"""

class GMHI(BaseEstimator):
    def __init__(self, thresh = 0.00001, use_shannon = True):
        self.theta_f = 1.4
        self.theta_d = 0.1
        self.thresh = thresh
        self.use_shannon = use_shannon
        self.fitted = False
    
    def fit(self, X, y):
        """
        Tries different values for theta_f and theta_d. Calls
        fit_species and then scores the model. Keeps the best combination of 
        theta_f and theta_d based on balanced accuracy on the training set, 
        then recalls fit_species with those parameters
        """
        best_score = -1
        best_theta_f = -1
        best_theta_d = -1
        # grid search theta
        for theta_f in [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
            for theta_d in [0, 0.05, 0.1, 0.15, 0.2]:
                self.theta_f = theta_f
                self.theta_d = theta_d
                self.select_features(X, y)
                score = self.score(X, y)
                if score > best_score:
                    best_theta_f = theta_f
                    best_theta_d = theta_d
                    best_score = score
        # save the best theta values based on training set score
        self.theta_f = best_theta_f
        self.theta_d = best_theta_d
        self.select_features(X, y)
        return best_score, best_theta_f, best_theta_d

    def select_features(self, X, y):
        """
            X is a df, (num_examples, num_features)
            y is a df, (num_examples, 1)
            X and y have column names indicating species names
            Selects health abundant and health scarce species based on differences and fold changes
        """
        
        self.fitted = True
        
        # get healthy and unhealthy samples
        healthies = X.iloc[y.values, :]
        unhealthies = X.iloc[~y.values, :]
        
        # get proportions for each species
        proportion_healthy = self.get_proportions(healthies)
        proportion_unhealthy = self.get_proportions(unhealthies)
        
        # get differences and fold change
        diff = proportion_healthy - proportion_unhealthy
        fold = proportion_healthy / proportion_unhealthy
        
        # based on proportion differences and fold change, select health abundant
        # and health scarce
        self.health_abundant = self.cutoff(diff, fold)
        self.health_scarce = self.cutoff(-1 * diff, 1 / fold)
        
    def cutoff(self, diff, fold):
        return list(diff[
            (diff['Proportion'] > self.theta_d) & (fold['Proportion'] > self.theta_f)
        ].index)
        
    def get_proportions(self, df):
        p = (df > self.thresh).sum() / df.shape[0]
        proportion = pd.DataFrame({"Proportion" : p})
        return proportion
    
    def predict_raw(self, X):
        """
            X is a df, (num_examples, num_features)
            X has column names indicating species names
        """
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
        
    def get_psi(self, X):
        psi = self.richness(X)
        if self.use_shannon:
            shan = self.shannon(X)
            psi *= shan
        return psi

    def score(self, X, y):
        """
            Returns balanced accuracy, chi,  after predicting
        """
        pred = self.predict_raw(X)
        healthy_samples = pred.iloc[y.values, :]
        unhealthy_samples = pred.iloc[~y.values, :]
        p_h = np.sum(healthy_samples > 0) / len(healthy_samples)
        p_n = np.sum(unhealthy_samples < 0) / len(unhealthy_samples)
        return 0.5 * (p_h + p_n)[0]
        
    def get_species(self):
        """
            Returns the lists of health abundant and health scarce species as a tuple, if fitted
        """
        if not self.fitted:
            return None
        return self.health_abundant, self.health_scarce

    def get_thetas(self):
        """
            Returns theta_f and theta_d
        """
        return self.theta_f, self.theta_d
    
    def richness(self, X):
        frame = pd.DataFrame((X > self.thresh).sum(axis=1))
        return frame
    
    def shannon(self, X):
        logged = np.log(X[X > 0])
        logged.fillna(0, inplace=True)
        shan = logged * X * -1
        return pd.DataFrame(shan.sum(axis=1))
