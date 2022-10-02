import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from toolz import partial
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
        
    def get(self, k = 3, eta = 'Auto'):
        
        if self.method == 'linear':
            synthetic_y0, synthetic_y1 = self._linear_control()
            return synthetic_y0, synthetic_y1
            
        elif self.method == 'robust_l1':
            synthetic_y0, synthetic_y1 = self._robust_l1_control(k, eta)
            return synthetic_y0, synthetic_y1
            
        elif self.method == 'robust_l2':
            synthetic_y0, synthetic_y1= self._robust_l2_control(k, eta)
            return synthetic_y0, synthetic_y1
        else:
            raise Exception("Only 'linear', 'robust_l1', 'robust_l2' are available.")
        
        
    def _linear_control(self):
        
        def loss_w(W, X, y):
            return np.sqrt(np.mean((y - X.dot(W))**2))
        
        w_start = [1/self.X0.shape[1]]*self.X0.shape[1]

        weights = fmin_slsqp(partial(loss_w, X=self.X0, y=self.y0),
                            np.array(w_start),
                            f_eqcons=lambda x: np.sum(x) - 1,
                            bounds=[(0.0, 1.0)]*len(w_start),
                            disp=False)
        
        self.W = weights
        synthetic_y0 = np.asarray(self.X0).dot(self.W)
        synthetic_y1 = np.asarray(self.X1).dot(self.W)
        
        return synthetic_y0, synthetic_y1    
    
    def _linear_control_SS(self):
        '''
        Synthetic Control with normal linear regression.
        This method has a very high risk of overfitting and is designed for experiments.
        '''
        
        lr_ss = LinearRegression(fit_intercept = False)
        lr_ss.fit(self.X0, self.y0)
            
        self.W = np.array(lr_ss.coef_)
        synthetic_y0 = np.asarray(self.X0).dot(self.W)
        synthetic_y1 = np.asarray(self.X1).dot(self.W)
        self._inv_scaling()
        
        return synthetic_y0, synthetic_y1    
    
    def _robust_l1_control(self, k, eta = "Auto"):
        
        self._scaling()
        self.M0, self.M1 = self._denoising(k)
        if eta == "Auto":
            opt_eta = self._forward_chain_CV(method = 'l1')
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
            
        self.W = np.array(lasso.coef_)
        synthetic_y0 = np.asarray(self.M0).dot(self.W) * self.s + self.m
        synthetic_y1 = np.asarray(self.M1).dot(self.W) * self.s + self.m
        self._inv_scaling()
        
        return synthetic_y0, synthetic_y1    
    
    def _robust_l2_control(self, k, eta = "Auto"):
        
        self._scaling()
        self.M0, self.M1 = self._denoising(k)
        if eta == "Auto":
            opt_eta = self._forward_chain_CV(method = 'l2')
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
            
        self.W = np.array(ridge.coef_)
        synthetic_y0 = np.asarray(self.M0).dot(self.W) * self.s + self.m
        synthetic_y1 = np.asarray(self.M1).dot(self.W) * self.s + self.m
        self._inv_scaling()
        
        return synthetic_y0, synthetic_y1
    
    def _denoising(self,k):
        u, s, v = np.linalg.svd(pd.concat([self.X0, self.X1], axis = 0))
        reduced = u[:,:k].dot(np.diag(s[:k])).dot(v[:k,:])
        return reduced[:len(self.X0), :], reduced[len(self.X0):, :]
        
        
    def svd_ploting(self):
        
        M = pd.concat([self.X0, self.X1], axis = 0)
        res = np.linalg.svd(M)[1]
        
        plt.plot(np.arange(1, len(res)+1), res)
        plt.title("values plot of SVD ")
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
    
    def _inv_scaling(self):
        self.X0 = self.X0 * self.s + self.m
        self.X1 = self.X1 * self.s + self.m
        self.y0 = self.y0 * self.s + self.m
        self.y1 = self.y1 * self.s + self.m
        
        return
    
    
    def _forward_chain_CV(self, method):
        
        n = len(self.M0)
        max_eta = max(self.M0.T.dot(self.y0))
        candidate_eta = [10**i for i in range(-2, 10)]
        candidate_eta = [eta for eta in candidate_eta if eta <max_eta]
        
        train_X = self.M0[:(n-1), :].copy()
        train_y = self.y0.iloc[:(n-1)].values.copy()
        
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
                model.fit(train_X[cv_idx[i], :], train_y[cv_idx[i]])
                
                loss.append((model.predict(valid_X)[0] - valid_y)**2)
            if np.mean(loss) < min_loss:
                min_loss = np.mean(loss)
                opt_eta = eta
                
        return opt_eta
        
        
        

        
        