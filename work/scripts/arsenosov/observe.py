# -*- coding: utf-8 -*-
"""
МИСИС
Программная инженерия

@author: Карнаушко В. А.
"""


from common import cwd_work
cwd_work()
import numpy as np
import pandas as pd


pth = 'data/familiesandhouseholds20221.xlsx'
frame = pd.read_excel(pth)
