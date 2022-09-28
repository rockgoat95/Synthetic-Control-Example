import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver

import pandas as pd

import tqdm
import time
import re
import pickle
import os

from selenium.webdriver.common.by import By

## infighter url
url = "https://dundam.xyz/damage_ranking?page=1&type=7&job=%E7%9C%9E%20%EC%9B%A8%ED%8E%80%EB%A7%88%EC%8A%A4%ED%84%B0&baseJob=%EA%B7%80%EA%B2%80%EC%82%AC(%EB%82%A8)&weaponType=%EC%A0%84%EC%B2%B4&weaponDetail=%EC%A0%84%EC%B2%B4"

driver = webdriver.Chrome("C:/Chromedriver/chromedriver.exe")
driver.get(url)
page_src = BeautifulSoup(driver.page_source)

job_box = page_src.find_all("ul", "job")

re_base_job = re.compile("[가-힣()]+")
re_job = re.compile("\n")

base_job_list = []
for i in range(3):
    job_line = job_box[i].find_all("label")
    for j in range(len(job_line)):
        base_job_list.extend(re_base_job.findall(job_line[j].get_text()))

job_dict = {}
for i in range(len(base_job_list)):
    time.sleep(1)
    job_page = driver.find_element(
        By.XPATH,
        "/html/body/section/div[1]/div[1]/div/div[2]/div[2]/ul["
        + str(i // 5 + 1)
        + "]/div["
        + str(i % 5 + 1)
        + "]",
    )
    job_page.click()
    page_src = BeautifulSoup(driver.page_source)
    job_dict[base_job_list[i]] = [
        re_job.sub("", x.get_text())
        for x in page_src.find("div", class_="ch_evol tble").find_all(
            "div", class_="rkcbox"
        )
        if "none" not in x["style"]
    ]

for base_job in tqdm.tqdm(job_dict):
    for job in job_dict[base_job]:
        job_url = (
            "https://dundam.xyz/damage_ranking?page=1&type=7&job="
            + job
            + "&baseJob="
            + base_job
            + "&weaponType=%EC%A0%84%EC%B2%B4&weaponDetail=%EC%A0%84%EC%B2%B4"
        )

        driver.get(job_url)

        chr_dict = {"nick_name": [], "server": []}
        stop_idx = False
        while not stop_idx:
            for j in range(1, 11):
                time.sleep(1)

                page_src = BeautifulSoup(driver.page_source)

                rkt_tr_list = page_src.find_all("div", class_="rkt-tr")
                for l in range(1, 11):
                    try:
                        nick_name = rkt_tr_list[l].find("span", class_="nik").get_text()
                        chr_dict["nick_name"].append(nick_name)

                        server = rkt_tr_list[l].find("span", class_="svname").get_text()
                        chr_dict["server"].append(server)

                    except:
                        print("Stop")
                        stop_idx = True

                ## 다음페이지에 해당하는 버튼 누르기
                next_page = driver.find_element_by_xpath(
                    '//*[@id="ranking"]/div[1]/div[5]/div[3]/div/ul/li[' + str(2 + j) + "]"
                )
                next_page.click()

                if stop_idx:
                    break
            # 중간 저장
            with open("DAT/" + base_job + "_" + job + "_info", "wb") as fp:
                pickle.dump(chr_dict, fp)

        # 최종 저장
        with open("DAT/" + base_job + "_" + job + "_info", "wb") as fp:
            pickle.dump(chr_dict, fp)
