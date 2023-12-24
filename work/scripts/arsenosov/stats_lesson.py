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

# Чтение данных
pth_b = './data/AUTO21053B.xlsx'
CARSB = pd.read_excel(pth_b)
print(CARSB.head())
print(CARSB.tail())

pth_b_w = './data/AUTO21053B.csv'
CARSB_W = pd.read_table(pth_b_w, sep=';', header=0, decimal=',',
                        encoding='utf-8')

CB = CARSB.copy()
print(CB.dtypes)
CB['music'].replace({1:"есть", 0:"нет"}, inplace=True)
print(CB.dtypes)
CB['music'].head()
# CARSB.head()
CB['music']=CB['music'].astype("category")
CB.loc[0, 'music'] = 'yes'
print(CB.dtypes)
CB['music'].head()


from pandas.api.types import CategoricalDtype
cat_type = CategoricalDtype(categories=["нет", "есть"], ordered=True)
CB['music'] = CB['music'].astype(cat_type)
CB['music'].head()

#*******************************************************

pth_a = './data/AUTO21053A.xlsx' # Путь относительно рабочего каталога
CARSA = pd.read_excel(pth_a)

CARSA = CARSA.astype({'age':np.float64, 'music':'category', 
                      'signal':'category', 'price':np.float64})

CA = CARSA.select_dtypes(include='float')
CA_STAT = CA.describe()
#-------------------------------------
W = CA.quantile(q=0.75) - CA.quantile(q=0.25) # Получается pandas.Series
# Создаем pandas.DataFrame из новых статистик
CA_irq = pd.DataFrame([W], index=['IQR'])
# Объединяем CA_STAT и W
CA_STAT = pd.concat([CA_STAT, CA_irq])

# Проверить формулы
irq = CA_irq['price']
wisker_u = (CA_STAT.loc['50%', 'price'] + 1.5*irq).values[0]
wisker_l = (CA_STAT.loc['50%', 'price'] - 1.5*irq).values[0]
out_1 = (CA['price'] > wisker_u) + (CA['price'] <= wisker_l)

# Анализ корреляции между количественными переменными
# Используем библиотеку scipy
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# Здесь будут значения оценок коэффициента корреляции Пирсона
C_P = pd.DataFrame([], index=CA.columns, columns=CA.columns) 
# Здесь будут значения значимости оценок коэффициента корреляции Пирсона
P_P = pd.DataFrame([], index=CA.columns, columns=CA.columns)
# Здесь будут значения оценок коэффициента корреляции Спирмена
C_S = pd.DataFrame([], index=CA.columns, columns=CA.columns)
# Здесь будут значения значимости оценок коэффициента корреляции Спирмена
P_S = pd.DataFrame([], index=CA.columns, columns=CA.columns)
for x in CA.columns:
    for y in CA.columns:
        C_P.loc[x,y], P_P.loc[x,y] = pearsonr(CA[x], CA[y])
        C_S.loc[x,y], P_S.loc[x,y] = spearmanr(CA[x], CA[y])

# Сохраняем текстовый отчет на разные листы Excel файла
with pd.ExcelWriter('./output/CARS_STAT.xlsx', engine="openpyxl") as wrt:
# Общая статистика
    CA_STAT.to_excel(wrt, sheet_name='stat')
# Корреляция Пирсона
    C_P.to_excel(wrt, sheet_name='Pearson')
    dr = C_P.shape[0] + 2
    P_P.to_excel(wrt, startrow=dr, sheet_name='Pearson') # Значимость
# Корреляция Спирмена
    C_S.to_excel(wrt, sheet_name='Spirmen')
    dr = C_S.shape[0] + 2
    P_S.to_excel(wrt, startrow=dr, sheet_name='Spirmen') # Значимость
    
