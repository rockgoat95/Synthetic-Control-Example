import pandas as pd
from SyntheticControl import SyntheticControl
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('DAT/job_pivot_table.csv').set_index('create_week')
T0 = 202221
col = "마법사(여)_소환사"

sc_model = SyntheticControl(data0 = data.loc[data.index<T0,:],
                            data1 = data.loc[data.index>=T0,:],
                            col = col,
                            method = 'linear')

sc_y0, sc_y1 = sc_model.get()

robust_sc_model = SyntheticControl(data0 = data.loc[data.index<T0,:],
                            data1 = data.loc[data.index>=T0,:],
                            col = col,
                            method = 'robust_l2')
robust_sc_model.svd_ploting()
rsc_y0, rsc_y1 = robust_sc_model.get(2)


def synthetic_plot(x, y, s_y, v_idx, label = ""):
    plt.plot(x, s_y, label = 'Sythetic')
    plt.plot(x, y, label = 'Real')
    plt.vlines(x = v_idx, ymax = max([s_y.max(), y.max()]), linestyle=":" , ymin = min([s_y.min(), y.min()]) )
    plt.ylabel(label)
    plt.legend()
    plt.show()
    return 

synthetic_plot(data.index, data[col], np.concatenate((sc_y0, sc_y1)) , v_idx = T0-0.5, label = "소환사의 주간 캐릭터 생성")

synthetic_plot(data.index, data[col], np.concatenate((rsc_y0, rsc_y1)) , v_idx = T0-0.5, label = "소환사의 주간 캐릭터 생성")
