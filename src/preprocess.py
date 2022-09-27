import os
import pickle


import pandas as pd
import datetime

path = "DAT/chr_info"

T0_before = 8
T0_after = 3
T0 = 202219 # Treament

data_list = []

def get_update_week(x):
    res = x - datetime.timedelta(days=2)
    res = list(res.date().isocalendar())
    res = res[0] * 100 + res[1]
    return res


for file_name in os.listdir(path):
    with open(os.path.join(path, file_name), "rb") as fp:
        chr_dat = pickle.load(fp)

    df = pd.DataFrame(chr_dat)
    df = df.loc[
        df["create_date"] != "",
    ]
    df["create_week"] = pd.to_datetime(df["create_date"]).apply(get_update_week)

    df = df.groupby("create_week").size().reset_index()

    df.columns = ["create_week", "size"]

    df["job"] = file_name[:-5]

    data_list.append(df)

data = pd.concat(data_list, axis=0)
job_names = [name[:-5] for name in os.listdir(path)]
unable_to_get = [job_names[i] for i in range(len(job_names))  if len(data_list[i]) == 0 ]

# get_update_week(datetime.datetime.strptime('2022-05-25', "%Y-%m-%d"))

# 이전 8주간 이용 
# 2022년 21번째 주에 소환사와 스트리트파이터 (여) 리뉴얼되었슴. 
# 4주간


data = data.loc[(data['create_week'] >= (T0-T0_before)) & (data['create_week'] < (T0+T0_after)), :]
data['job'] = data['job'].str.replace("眞 ","")

data = data.pivot_table(index = ['create_week'], columns = 'job', values = 'size')
data = data.fillna(0)

data.to_csv('DAT/job_pivot_table.csv')
