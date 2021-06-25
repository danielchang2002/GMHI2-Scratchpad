from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

class GMHI(BaseEstimator):
    def __init__(self, theta_f = 1.4, theta_d = 0.1, thresh = 0.00001, use_shannon = True,
                R_MH = 7, R_MN = 31):
        self.theta_f = theta_f
        self.theta_d = theta_d
        self.thresh = 0.00001
        self.use_shannon = use_shannon
        self.fitted = False
        self.R_MH = R_MH
        self.R_MN = R_MN
    
    def fit(self, X, y):
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
        self.health_abundant = self.select_species(diff, fold)
        self.health_scarce = self.select_species(-1 * diff, 1 / fold)
        
    def select_species(self, diff, fold):
        return list(diff[
            (diff['Proportion'] > self.theta_d) & (fold['Proportion'] > self.theta_f)
        ].index)
        
    def get_proportions(self, df):
        p = (df > self.thresh).sum() / df.shape[0]
        proportion = pd.DataFrame({"Proportion" : p})
        return proportion
    
    def predict(self, X):
        """
            X is a df, (num_examples, num_features)
            X has column names indicating species names
        """
        if not self.fitted:
            return None
        X_healthy_features = X[self.health_abundant]
        X_unhealthy_features = X[self.health_scarce]
        psi_MH = self.get_psi(X_healthy_features) / self.R_MH
        psi_MN = self.get_psi(X_unhealthy_features) / self.R_MN
        return np.log((psi_MH + 0.00001) / (psi_MN) + 0.00001) > 0
        
    def get_psi(self, X):
        psi = self.richness(X)
        if self.use_shannon:
            psi *= self.shannon(X)
        return psi
        
    def get_species(self):
        """
            Returns the lists of health abundant and health scarce species as a tuple, if fitted
        """
        if not self.fitted:
            return None
        return self.health_abundant, self.health_scarce
    
    def richness(self, X):
        frame = pd.DataFrame((X > 0).sum(axis=1))
        return frame
    
    def shannon(self, X):
        logged = X.copy()
        logged[logged > 0] = np.log(logged[logged > 0])
        shannoned = logged * X * -1
        sums = shannoned.sum(axis=1)
        sums = pd.DataFrame(sums)
        return sums
