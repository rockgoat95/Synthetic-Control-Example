import pandas as pd
from SyntheticControl import SyntheticControl
import numpy as np

data = pd.read_csv('DAT/job_pivot_table.csv').set_index('create_week')
T0 = 202221

sc_model = SyntheticControl(data0 = data.loc[data.index<T0,:],
                            data1 = data.loc[data.index>=T0,:],
                            col = '마법사(여)_소환사' ,
                            method = 'linear')

sc_y1 = sc_model.get()

robust_sc_model = SyntheticControl(data0 = data.loc[data.index<T0,:],
                            data1 = data.loc[data.index>=T0,:],
                            col = '마법사(여)_소환사' ,
                            method = 'robust_l2')
robust_sc_model.svd_ploting()
robust_sc_model.get(4)


robust_sc_model.m
u, s, v = np.linalg.svd(pd.concat([robust_sc_model.X0, robust_sc_model.X1], axis = 0))
reduced = u[:,:k].dot(np.diag(s[:k])).dot(v[:k,:])