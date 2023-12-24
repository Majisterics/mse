# -*- coding: utf-8 -*-
"""
МИСИС
Программная инженерия

@author: Карнаушко В. А.
@theme: Предобработка данных.
"""


from common import cwd_work
cwd_work()
import numpy as np
import pandas as pd


pth_b = './data/AUTO21053B.xlsx'
CARSB = pd.read_excel(pth_b)
print(CARSB.head(10))

# К каким шкалам относятся переменные (измерения) соответствующих
# признаков?
#
#    age - относительная шкала (или может порядковая, т.к. число символов
#          алфавита сравнительно невелико)
#    music - номинальная шкала
#    signal - номинальная шкала
#    run - относительная шкала
#    price - относительная шкала
