import os
import pickle

from src.utils import *

import pandas as pd
import datetime

path = "DAT/chr_info"

## for week
T0_before = 8
T0_after = 3
T0 = 202220 # Treament

# for day
T0_day_before = '2022-05-01'
T0_day_after = '2022-05-29'
per_day = False

data_list = []

for file_name in os.listdir(path):
    with open(os.path.join(path, file_name), "rb") as fp:
        chr_dat = pickle.load(fp)

    df = pd.DataFrame(chr_dat)
    df = df.loc[
        df["create_date"] != "",
    ]
    if per_day:
        df["create_date"] = df["create_date"].apply(lambda x: x[:10])
    else:
        df["create_date"] = pd.to_datetime(df["create_date"]).apply(get_update_week)

    df = df.groupby("create_date").size().reset_index()

    df = df.sort_values(by = 'create_date')
    df.columns = ["create_date", "size"]

    df["job"] = file_name[:-5]

    data_list.append(df)

data = pd.concat(data_list, axis=0)
job_names = [name[:-5] for name in os.listdir(path)]
unable_to_get = [job_names[i] for i in range(len(job_names))  if len(data_list[i]) == 0 ]

with open("DAT/job_unused.txt", "w") as output:
    output.write(', '.join(unable_to_get))
# get_update_week(datetime.datetime.strptime('2022-05-17', "%Y-%m-%d"))

# 이전 8주간 이용 
# 2022년 21번째 주에 소환사와 스트리트파이터 (여) 리뉴얼되었슴. 
# 4주간


if per_day:
    data = data.loc[(data['create_date'] >= T0_day_before) & (data['create_date'] <= T0_day_after), :]
    save_path = 'DAT/job_pivot_table_day.csv'
else:
    data = data.loc[(data['create_date'] >= (T0-T0_before)) & (data['create_date'] < (T0+T0_after)), :]
    save_path = 'DAT/job_pivot_table_week.csv'

data['job'] = data['job'].str.replace("眞 ","")

data = data.pivot_table(index = ['create_date'], columns = 'job', values = 'size')
data = data.fillna(0)

data.to_csv(save_path)