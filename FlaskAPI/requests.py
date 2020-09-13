# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 01:10:09 2020

@author: tigra
"""

import requests
from data_input import data_in

URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}
data = {"input": data_in}

r = requests.get(URL,headers=headers, json=data) 

r.json()