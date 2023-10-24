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
# read_table вместо read_csv - некоторый дополнительный способ задать параметры
# чтения таблицы, в read_csv что-то задано по умолчанию
CARSB_W = pd.read_table(pth_b_w, sep=';', header=0, decimal=',', encoding='utf-8')

CB = CARSB.copy()
print(CB.dtypes)
CB['music'].replace({1:"есть", 0:"нет"}, inplace=True)
print(CB.dtypes)
CB['music'].head()
# CARSB.head()
# Выдаст ошибку - мы пытаемся присвоить значение, не
# входящее в алфавит переменной (можно только "есть", "нет")
# CB['music'] = CB['music'].astype("category")
# CB.loc[0, 'music'] = 'yes'
print(CB.dtypes)
CB['music'].head()


from pandas.api.types import CategoricalDtype
# Для того же задания категории можно использовать внутренне
# определенный класс CategoricalDtype. Хотя по-моему
# легче использовать функцию astype с параметром "category".
# Если только вам не нужно дополнительно задать точно отношение порядка,
# все элементы алфавита, т. д.
cat_type = CategoricalDtype(categories=["нет", "есть"], ordered=True)
CB['music'] = CB['music'].astype(cat_type)
CB['music'].head()

# Кодирование переменных в качестве упражнения
EMPLOYEE = pd.read_excel('./data/EMPLOYEE_01.xlsx')
ED = EMPLOYEE.copy()
ED['Gender'].replace({"ж":"F", "м":"M"}, inplace=True)
ED['Gender'] = ED['Gender'].astype("category")
ED['Level'] = ED['Level'].astype("category")
cat_type = CategoricalDtype(categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ordered=True)
ED['  Experience'] = ED['  Experience'].astype(cat_type)
