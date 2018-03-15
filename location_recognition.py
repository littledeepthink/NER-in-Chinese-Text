# coding=utf-8
from __future__ import print_function, unicode_literals
import numpy as np
import pandas as pd
import json
import requests
import argparse

data = pd.read_csv('/home/purvar/Downloads/location/t_sup_complaint.csv',
                   names=np.arange(27))
# text = data.iloc[:, 11]

NER_URL = 'http://api.bosonnlp.com/ner/analysis' # BosonNLP

parser = argparse.ArgumentParser()
parser.add_argument('--index',
                    type=int,
                    default=100,
                    help='Please input an index')
FLAGS, unparsed = parser.parse_known_args()

s = [ data.iloc[FLAGS.index, 11] ] #投诉内容位于第十二列
print(s)
# s = ['硚口区汉西三路香江家居对面常青国际小区，光头卤店铺24小时抽烟机噪音扰民，\
#      油烟扰民，区局已派第三方检查公司进行检测，投诉人等待测试结果的回复。多次来电，请重点处理。']
data = json.dumps(s)
headers = {'X-Token': 'LkwQR-rW.21981.qz7z9JKCk9A9'}
resp = requests.post(NER_URL, headers=headers, data=data.encode('utf-8'))

for item in resp.json():
    for entity in item['entity']:
        if entity[2] in ['location', 'org_name', 'company_name']:
            print(''.join(item['word'][entity[0]:entity[1]]), entity[2])

# print(resp.text)
