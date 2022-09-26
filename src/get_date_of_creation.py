import json
import requests

import pandas as pd

import time 
import re
import pickle
import os

from dotenv import load_dotenv

load_dotenv()

from copy import deepcopy

import tqdm
import datetime 

import numpy as np

from utils import get_chracter_creation_date, create_continuous_time

api_key_val = os.environ.get("API_KEY")

api_url = 'https://api.neople.co.kr/df/'

end_point = 'servers'

url = api_url + end_point + '?apikey='+api_key_val
res = requests.get(url)
server_res = json.loads(res.content)

server_df = pd.DataFrame(server_res['rows'])

server_name2id = dict([[x2,x1] for x1, x2 in server_df.values.tolist()])


file_names = os.listdir('DAT/job_info')

len_files = len(file_names)
for i, file_name in enumerate(file_names):
    with open('C:/Neople_Anal/DAT/job_info/' +file_name, "rb") as fp: 
        chr_dict = pickle.load(fp)
    
    chr_dict['create_date'] = []
    chr_dict['adventureName'] = []
    print('Now : ' + file_name + ' \t'+ str(i) +'/' + str(len_files))
    for name, server in tqdm.tqdm(zip(chr_dict['nick_name'], chr_dict['server'])):
        date, adv = get_chracter_creation_date(name, server, server_name2id, api_key_val)
        chr_dict['create_date'].append(date)
        chr_dict['adventureName'].append(adv)
    chr_dict['job'] = [file_name]*len(chr_dict['create_date'][:-5])

    with open('C:/Neople_Anal/DAT/chr_info/' + file_name, "wb") as fp: 
        pickle.dump(chr_dict, fp)
    
