import pandas as pd
from SyntheticControl import SyntheticControl
import numpy as np
import matplotlib.pyplot as plt
from src.utils import *
plt.rc('font', family='Malgun Gothic')

data = pd.read_csv('DAT/job_pivot_table_week.csv').set_index('create_date')

data.index = data.index - 202200
T0 = 20
T1_term = 3
MSE_threshold = 80
treated_jobs = ["마법사(여)_소환사", "도적_쿠노이치"]
data.columns.__len__()

job_create_freq_plot(data, treated_jobs[0], T0, save = True)
job_create_freq_plot(data, treated_jobs[1], T0, save = True)
job_create_freq_plot(data, '도적_로그', T0, save = True)
job_create_freq_plot(data, '프리스트(남)_크루세이더', T0, save = True)

donor_pool = [job for job in data.columns if job not in treated_jobs]
empirical_pvs = []
for i in range(len(treated_jobs)):

    treated = treated_jobs[i]
    columns = donor_pool + [treated_jobs[i]]

    sc_model = SyntheticControl(data0 = data.loc[data.index<T0,columns],
                                data1 = data.loc[data.index>=T0,columns],
                                col = treated,
                                method = 'linear')

    sc_y0, sc_y1 = sc_model.get()

    robust_sc_model = SyntheticControl(data0 = data.loc[data.index<T0,columns],
                                data1 = data.loc[data.index>=T0,columns],
                                col = treated,
                                method = 'robust_l2')
    # robust_sc_model.svd_ploting()
    rsc_y0, rsc_y1 = robust_sc_model.get(2)


    def synthetic_plot(x, y, s_y, v_idx, label = ""):
        plt.plot(x, s_y, label = 'Sythetic')
        plt.plot(x, y, label = 'Real')
        plt.vlines(x = v_idx, ymax = max([s_y.max(), y.max()]), ymin = min([s_y.min(), y.min()]),  linestyles = '--' )
        plt.ylabel(label)
        plt.legend()
        plt.show()
        return 

    # synthetic_plot(data.index, data[treated], np.concatenate((sc_y0, sc_y1)) , v_idx = T0-0.5, label = treated + "의 주간 캐릭터 생성")

    synthetic_plot(data.index, data[treated], np.concatenate((rsc_y0, rsc_y1)) , v_idx = T0-0.5, job_name = treated, title = treated + "의 주간 캐릭터 생성")

    ymax = -10
    ymin = 10

    after_treated = []
    after_treated_names = []
    empirical_ATE_sum =[]
    for placebo_treated in columns:
        
        placebo_sc_model = SyntheticControl(data0 = data.loc[data.index<T0,columns],
                                    data1 = data.loc[data.index>=T0,columns],
                                    col = placebo_treated,
                                    method = 'robust_l2')
        
        placebo_sc_y = np.concatenate(placebo_sc_model.get(2))

        alpha = 1 if placebo_treated == treated else 0.2
        color = 'red' if placebo_treated == treated else 'gray'
        
        true_sc_diff = data[placebo_treated].values - placebo_sc_y 
        
        if np.mean(true_sc_diff[:-T1_term]**2) > MSE_threshold:
            continue
        after_treated.append(true_sc_diff[-T1_term:].sum())
        after_treated_names.append(placebo_treated)
        ymax = max(ymax, true_sc_diff.max())
        ymin = min(ymin, true_sc_diff.min())
        
        if placebo_treated == treated:
            empirical_ATE_sum.append(true_sc_diff[-T1_term:].sum())
            
        
        plt.plot(data.index, true_sc_diff, alpha = alpha, color = color)
        
    plt.vlines(x = T0-0.5, ymax = ymax, ymin = ymin, linestyles = '--', alpha = 0.5 )
    plt.title('Treated v.s. Placebo - ' + treated)
    plt.savefig('plots/with_placebo_'+treated + '.png')
    plt.show()

    after_treated = pd.Series(after_treated)
    after_treated.index = after_treated_names
    empirical_pvs.append(np.mean(after_treated >= after_treated[treated]) )
    ## bootstrap 결과를 통한 검정 추가 