from msilib.schema import Error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from typing import List
from operator import add
from toolz import reduce, partial
from scipy.optimize import fmin_slsqp
import random

class SyntheticControl:
    def __init__(self, data0, data1, col, method = 'linear'):
                
        self.X0 = data0.drop(col, axis = 1)
        self.X1 = data1.drop(col, axis = 1)
        self.y0 = data0[col]
        self.y1 = data1[col]
        
        self.col = col
        self.donor_pool = self.X0.columns
        self.method = method
        if self.method in ['robust_l1', 'robust_l2']:
            self._scaling
        
    def get(self, k = 3):
        
        if self.method == 'linear':
            self.W = self._linear_control()
            synthetic_y = np.asarray(self.X1).dot(self.W)
        elif self.method == 'robust_l1':
            self.W = self._robust_l1_control(k)
            synthetic_y = np.asarray(self.M1).dot(self.W)
            synthetic_y= self._inv_scaling(synthetic_y)
        elif self.method == 'robust_l2':
            self.W = self._robust_l2_control(k)
            synthetic_y = np.asarray(self.M1).dot(self.W)
            synthetic_y= self._inv_scaling(synthetic_y)
            pass
        else:
            raise Error("Only 'linear', 'robust_l1', 'robust_l2' are available.")
        
        return synthetic_y
        
    def _linear_control(self):
        
        def loss_w(W, X, y):
            return np.sqrt(np.mean((y - X.dot(W))**2))
        
        w_start = [1/self.X0.shape[1]]*self.X0.shape[1]

        weights = fmin_slsqp(partial(loss_w, X=self.X0, y=self.y0),
                            np.array(w_start),
                            f_eqcons=lambda x: np.sum(x) - 1,
                            bounds=[(0.0, 1.0)]*len(w_start),
                            disp=False)
        return weights
    
    def _robust_l1_control(self, k, eta = "Auto"):
        
        self._scaling()
        self.M0, self.M1 = self._denoising(k)
        if eta == "Auto":
            opt_eta = self._chaining(method = 'l1')
            lasso = Lasso(alpha = opt_eta, 
                        fit_intercept = False,
                        max_iter = 2000,
                        )
            lasso.fit(self.M0, self.y0)
        else:
            lasso = Lasso(alpha = eta, 
                        fit_intercept = False,
                        max_iter = 2000,
                        )
            lasso.fit(self.M0, self.y0)
            
        return np.array(lasso.coef_)
    
    def _robust_l2_control(self, k, eta = "Auto"):
        
        self._scaling()
        self.M0, self.M1 = self._denoising(k)
        if eta == "Auto":
            opt_eta = self._chaining(method = 'l2')
            ridge = Ridge(alpha = opt_eta, 
                        fit_intercept = False,
                        max_iter = 2000,
                        )
            ridge.fit(self.M0, self.y0)
        else:
            ridge = Ridge(alpha = eta, 
                        fit_intercept = False,
                        max_iter = 2000,
                        )
            ridge.fit(self.M0, self.y0)
            
        return np.array(ridge.coef_)
    
    def _denoising(self,k):
        u, s, v = np.linalg.svd(pd.concat([self.X0, self.X1], axis = 0))
        reduced = u[:,:k].dot(np.diag(s[:k])).dot(v[:k,:])
        return reduced[:len(self.X0), :], reduced[len(self.X0):, :]
        
        
    def svd_ploting(self):
        
        M = pd.concat([self.X0, self.X1], axis = 0)
        res = np.linalg.svd(M)[1]
        
        plt.plot(np.arange(1, len(res)+1), res)
        plt.show()
        
        return res
        
    def _scaling(self):
        data = pd.concat([
            pd.concat([self.X0, self.y0], axis = 1), 
            pd.concat([self.X1, self.y1], axis = 1)
        ], axis= 0 )
        
        b = data.max().max()
        a = data.min().min()
        
        self.m = (a + b)/2
        self.s = (b - a)/2
        
        self.X0 = (self.X0 - self.m)/self.s
        self.X1 = (self.X1 - self.m)/self.s
        self.y0 = (self.y0 - self.m)/self.s
        self.y1 = (self.y1 - self.m)/self.s
        
        return
    
    def _inv_scaling(self, data):
        try :
            self.m
            self.s
        except:
            NameError("Scaling method is not applied.")
        return data * self.s + self.m
    
    
    def _chaining(self, method):
        
        n = len(self.M0)
        max_eta = max(self.M0.T.dot(self.y))
        candidate_eta = [10**i for i in range(-2, 10)]
        candidate_eta = [eta for eta in candidate_eta if eta <max_eta]
        
        train_X = self.M0[:(n-1), :].copy()
        train_y = self.y0.iloc[:(n-1)].copy()
        
        valid_X = self.M0[-1:,:].copy()
        valid_y = self.y0.iloc[-1].copy()
        
        idx_list = [i for i in range(n-1)]
        random.shuffle(idx_list)
        knots = [int(i/6*(n-1)) for i in range(6)] + [n-1]
        cv_idx = [idx_list[:knots[i]] + idx_list[knots[i+1]: ] for i in range(5)]
        
        
        min_loss = 1e8
        opt_eta = None
        for eta in candidate_eta:
            if method == 'l1':
                model = Lasso(alpha = eta, 
                            fit_intercept = False,
                            max_iter = 2000,
                            )
            elif method == 'l2':
                model = Ridge(alpha = eta, 
                            fit_intercept = False,
                            max_iter = 2000,
                            )
            loss = []
            for i in range(5):
                model.fit(train_X[cv_idx, :], train_y[cv_idx])
                
                loss.append((model.predict(valid_X)[0] - valid_y)**2)
            if np.mean(loss) < min_loss:
                min_loss = np.mean(loss)
                opt_eta = eta
                
        return opt_eta
        
        
        

        
        