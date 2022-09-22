import os
import pickle
os.chdir('C:/sion/dnf-job-analysis')

import pandas as pd
import datetime

path = 'DAT/chr_info'

data_list = []

def get_update_week(x):
    res = x - datetime.timedelta(days = 4)
    res = list(res.date().isocalendar())
    res = res[0]*100 + res[1] 
    return res

for file_name in os.listdir(path):
    with open(os.path.join(path, file_name), "rb") as fp: 
        chr_dat = pickle.load(fp)
        
    df = pd.DataFrame(chr_dat)
    df = df.loc[df['create_date'] != '', ]
    df['create_week'] = pd.to_datetime(df['create_date']).apply(get_update_week)

    df = df.groupby('create_week').size().reset_index()

    df.columns = ['create_week', 'size']

    df['job'] = file_name[:-5]
    
    data_list.append(df)
    
data = pd.concat(data_list, axis = 0 )

for dat in data_list:
    print(len(dat))
    
data.head()

