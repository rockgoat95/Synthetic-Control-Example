import urllib
import json
import datetime
import requests
import time
import matplotlib.pyplot as plt
import pandas as pd

api_url = "https://api.neople.co.kr/df/"

def job_create_freq_plot(data, job_name, T0, save = False):
    freq_series = data[job_name]
    plt.plot(freq_series.index, freq_series)
    plt.vlines(x = T0, ymax = freq_series.max(), ymin = 0, linestyles = '--', alpha = 0.5 )
    plt.title('생성 빈도 - ' + job_name )
    plt.xlabel('주차')
    plt.ylabel('빈도수')
    if save:
        plt.savefig('plots/'+ job_name + '_생성빈도.png')
    plt.show()

def synthetic_plot(x, y, s_y, v_idx, label = ""):
    plt.plot(x, s_y, label = 'Sythetic')
    plt.plot(x, y, label = 'Real')
    plt.vlines(x = v_idx, ymax = max([s_y.max(), y.max()]), ymin = min([s_y.min(), y.min()]),  linestyles = '--' )
    plt.ylabel(label)
    plt.legend()
    plt.show()
    return 

def get_update_week(x):
    res = x - datetime.timedelta(days=2)
    res = list(res.date().isocalendar())
    res = res[0] * 100 + res[1]
    return res

def get_chracter_creation_date(chr_name, server, server_name2id, api_key_val):
    try:
        chr_url = (
            api_url
            + "servers/"
            + server_name2id[server]
            + "/characters?characterName="
            + urllib.parse.quote(chr_name)
            + "&apikey="
            + api_key_val
        )
        chr_info = json.loads(requests.get(chr_url).content)["rows"][0]

        limit = "2017-09-21 00:00"
        current = datetime.datetime.now()
        current_str = current.strftime("%Y-%m-%d %H:%M")
        stop_idx = False
        first = True
        while not stop_idx:
            time.sleep(0.01)
            current = current - datetime.timedelta(days=90)
            past_str = current.strftime("%Y-%m-%d %H:%M")

            if past_str < limit:
                past_str = limit
                stop_idx = True

            timeline_url = (
                api_url
                + "servers/"
                + server_name2id[server]
                + "/characters/"
                + chr_info["characterId"]
                + "/timeline?code=101&startDate="
                + past_str
                + "&endDate="
                + current_str
                + "&apikey="
                + api_key_val
            )
            timeline = json.loads(requests.get(timeline_url).content)
            create_info = timeline["timeline"]["rows"]

            current_str = past_str

            if first:
                adv_name = timeline["adventureName"]
                first = False
            if len(create_info) == 0:
                continue
            else:
                return create_info[0]["date"], adv_name

        return "", adv_name
    except:
        return "", ""


def create_continuous_time(min_time_str, max_time_str, format_):
    min_time = datetime.datetime.strptime(min_time_str, format_)
    max_time = datetime.datetime.strptime(max_time_str, format_)

    continuous_time = [min_time]
    current_time = min_time + datetime.timedelta(days=1)
    while True:
        continuous_time.append(current_time)
        current_time += datetime.timedelta(days=1)
        if current_time > max_time:
            break
    continuous_time = [x.strftime(format_) for x in continuous_time]
    return continuous_time
