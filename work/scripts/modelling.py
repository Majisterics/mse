# -*- coding: utf-8 -*-
"""
Методы анализа стохастических взаимосвязей
Моделирование проверка гипотез

@author: Карнаушко В. А.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from pathlib import Path
from pandas.api.types import CategoricalDtype

# Очищаем файл перед запуском
with open('./output/modelling.txt', 'w') as fln:
    print()

pWork: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent
logging.basicConfig(level=logging.INFO, filename=pWork.joinpath("logs").joinpath("modelling.log"), filemode="w")

# **********************************************************
# ============= МОДЕЛИРОВАНИЕ ==============================
# **********************************************************

# Читаем и преобразуем данные
pData = pWork.joinpath("data")
dfData = pd.read_csv(pData.joinpath("processed-realtor-data").with_suffix(".csv"))

status_cat_type = CategoricalDtype(categories=["for_sale", "second_sale"], ordered=True)
dfData["status"] = dfData["status"].astype(status_cat_type)
dfData["bed"] = dfData["bed"].astype(np.int32)
dfData["bath"] = dfData["bath"].astype(np.int32)
# уплотняем качественные переменные
dfData["bed"] = pd.cut(
    dfData["bed"],
    labels=["1", "2", "3", "4", "5"],
    bins=[0, 2, 3, 4, 5, max(dfData["bed"])]
)
dfData["bath"] = pd.cut(
    dfData["bath"],
    labels=["1", "2", "3", "4"],
    bins=[0, 1, 2, 3, max(dfData["bath"])]
)
# определяем переменные, как порядковый тип
dfData["bed"] = dfData["bed"].astype(CategoricalDtype(["1", "2", "3", "4", "5"], ordered=True))
dfData["bath"] = dfData["bath"].astype(CategoricalDtype(["1", "2", "3", "4"], ordered=True))

CA = dfData.copy()
# print(CA.head())
# Разбиение данных на тренировочное и тестовое множество
# frac- доля данных в тренировочном множестве
# random_state - для повторного отбора тех же элементов
CA_train = CA.sample(frac=0.8, random_state=42) 
# Символ ~ обозначает отрицание (not)
CA_test = CA.loc[~CA.index.isin(CA_train.index)]

# Будем накапливать данные о качестве постреонных моделей
# Используем  adjR^2 и AIC
mq = pd.DataFrame([], columns=['adjR^2', 'AIC']) # Данные о качестве

"""
Построение базовой модели
Базовая модель - линейная регрессия, которая включает в себя 
все количественные переменные и фиктивные переменные дял качественных 
переменных с учетом коллинеарности. Для каждого качетсвенного показателя
включаются все уровни за исключением одного - базового. 
"""
# Формируем целевую переменную
Y = CA_train['price']
# Формируем фиктивные (dummy) переменные для всех качественных переменных
DUM = pd.get_dummies(CA_train[['status', 'bed', 'bath']])
# Выбираем переменные для уровней, которые войдут в модель
# Будет исключен один - базовый. ВЛияние включенных уровней на зависимую 
# переменную отсчитывается от него
DUM = DUM[['status_second_sale',
           'bed_2', 'bed_3', 'bed_4', 'bed_5',
           'bath_2', 'bath_3', 'bath_4']]
# Формируем pandas.DataFramee содержащий матрицу X объясняющих переменных 
# Добавляем слева фиктивные переменные
X = pd.concat([DUM, CA_train[['acre_lot', 'house_size']]], axis=1)
# Добавляем переменную равную единице для учета константы
X = sm.add_constant(X)
X = X.astype({'const':'uint8'}) # Сокращаем место для хранения константы
# Формируем объект, содержащй все исходные данные и методы для оценивания
linreg00 = sm.OLS(Y,X)
# Оцениваем модель
fitmod00 = linreg00.fit()
# Сохраняем результаты оценки в файл
with open('./output/modelling.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod00.summary(), file=fln)
    rss = fitmod00.ssr
    print('Сумма квадратов остатков: ', rss, file=fln)
    n = X['acre_lot'].size
    k = X.size/n
    hqc = n * math.log(rss/n)+2*k*math.log(math.log(n))
    print('HQC: ', hqc, file=fln)


# Проверяем степень мультиколлинеарности только базовой модели
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
X_q = X.drop(columns=["const"])
# X_q = X_q[["acre_lot", "house_size"]]
vif["vars"] = X_q.columns
vif["VIF"] = [variance_inflation_factor(X_q.values, i)
              for i in range(X_q.shape[1])]
# Сохраняем полученные результаты
with pd.ExcelWriter('./output/modelling.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    vif.to_excel(wrt, sheet_name='vif')

# Проверяем гетероскедастичность базовой модели
# помощью коритерия White(а) и F критерия
from statsmodels.stats.diagnostic import het_white
e = fitmod00.resid

WHT = pd.DataFrame(het_white(e, X), index= ['LM', 'LM_P', 'F', 'F_P'])
# Сохраняем полученные результаты
with pd.ExcelWriter('./output/modelling.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    WHT.to_excel(wrt, sheet_name='het')

print(vif)
print()
print(WHT)

# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod00.rsquared_adj, fitmod00.aic], 
                 index=['adjR^2', 'AIC'], columns=['base_00']).T
mq = pd.concat([mq, q])    

# ****************** Проверки гипотез ******************

"""
Статус постройки влияет на цену
Чем новее постройка, тем она дешевле
Модель для проверки:
price = a0 + a11*status_for_sale + a12*status_second_sale + a2*bed + a3*bath + a4*acre_lot + a5*house_size + v
*****************
Если гипотеза справедлива, то a11>0, a11<a12 и значима
*****************

Целевая переменная не меняется.

Результат: гипотеза отвергается.

"""

X_1 = X.copy()
linreg01 = sm.OLS(Y,X_1)
fitmod01 = linreg01.fit()
# Сохраняем результаты оценки в файл
with open('./output/modelling.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели для гипотезы №1 ******',
          file=fln)
    print(fitmod01.summary(), file=fln)
    rss = fitmod01.ssr
    print('Сумма квадратов остатков: ', rss, file=fln)
    n = X['acre_lot'].size
    k = X.size/n
    hqc = n * math.log(rss/n)+2*k*math.log(math.log(n))
    print('HQC: ', hqc, file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod01.rsquared_adj, fitmod01.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_01']).T
mq = pd.concat([mq, q])    

# Коэфициент при первом статусе выше, чем при втором - следовательно, гипотеза подтверждается

"""
Сила влияния количества спален на цену зависит от количества ванных комнат
Рост количества спален приводит к росту стоимости недвижимости
Чем больше ванных комнат, тем привлекательнее большим семьям
Модель для проверки:
price = a0 + a11*status_for_sale + a12*status_second_sale + (a20 + a21*bath)*bed + a3*bath + a4*acre_lot + a5*house_size + v
Раскрывая скобки:
price = a0 + a11*status_for_sale + a12*status_second_sale + a20*bed + a21*bath*bed + a3*bath + a4*acre_lot + a5*house_size + v
*****************
Если гипотеза справедлива, то a21>a20 (?), a21>0 и значима
(a20 + a21*bath)*bed > 0, при этом bed > 0, bath > 0 => (a20 + a21*bath) > 0 и a21 > 0 => a21 > a20
*****************

Целевая переменная не меняется.

Результат: гипотеза отвергается.

"""

X_2 = X.copy()

X_2["bed_2 * bath_2"] = X_2["bath_2"] * X_2["bed_2"]
X_2["bed_2 * bath_3"] = X_2["bath_3"] * X_2["bed_2"]
X_2["bed_2 * bath_4"] = X_2["bath_4"] * X_2["bed_2"]

X_2["bed_3 * bath_2"] = X_2["bath_2"] * X_2["bed_3"]
X_2["bed_3 * bath_3"] = X_2["bath_3"] * X_2["bed_3"]
X_2["bed_3 * bath_4"] = X_2["bath_4"] * X_2["bed_3"]

X_2["bed_4 * bath_2"] = X_2["bath_2"] * X_2["bed_4"]
X_2["bed_4 * bath_3"] = X_2["bath_3"] * X_2["bed_4"]
X_2["bed_4 * bath_4"] = X_2["bath_4"] * X_2["bed_4"]

X_2["bed_5 * bath_2"] = X_2["bath_2"] * X_2["bed_5"]
X_2["bed_5 * bath_3"] = X_2["bath_3"] * X_2["bed_5"]
X_2["bed_5 * bath_4"] = X_2["bath_4"] * X_2["bed_5"]

linreg02 = sm.OLS(Y, X_2)
fitmod02 = linreg02.fit()
# Сохраняем результаты оценки в файл
with open('./output/modelling.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели для гипотезы №2 ******',
          file=fln)
    print(fitmod02.summary(), file=fln)
    rss = fitmod02.ssr
    print('Сумма квадратов остатков: ', rss, file=fln)
    n = X['acre_lot'].size
    k = X.size/n
    hqc = n * math.log(rss/n)+2*k*math.log(math.log(n))
    print('HQC: ', hqc, file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod02.rsquared_adj, fitmod02.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_02']).T
mq = pd.concat([mq, q])

# Гипотеза отвергается

"""
Сила влияния размера участка на цену зависит от размера дома:
Чем больше площадь и меньше дом тем влияние меньше
Для больших площадей участков с небольшими домами цены растут не так сильно из-за необходимости ухода за землей
Введем переменную
acre_thr = 1, если acre_lot-house_size >= thr и 0, если нет
thr - неизвестный порог
Модель для проверки:
price = a0 + a11*status_for_sale + a12*status_second_sale + a2*bed + a3*bath + (a40 + a41*acre_thr)*acre_lot + (a50 + a51*acre_thr)*house_size + v
Раскрывая скобки:
price = a0 + a11*status_for_sale + a12*status_second_sale + a2*bed + a3*bath + a40*acre_lot + a41*acre_thr*acre_lot + a50*house_size + a51*acre_thr*house_size + v
*****************
Если гипотеза справедлива, то a40>0, a41<0, a50>0, a51<0 и значима 
*****************

Целевая переменная не меняется.

Результат: гипотеза принимается.

"""
thr = 134.26853707414827 # Порог пробега - подобранный
X_3 = X.copy()
# False == 0, True == 1
acre_thr = X_3['acre_lot'] - X_3['house_size'] >= thr
X_3['ath'] = X_3['acre_lot']*acre_thr # Взаимодействие
X_3['hth'] = X_3['house_size']*acre_thr
linreg03 = sm.OLS(Y,X_3)
fitmod03 = linreg03.fit()
# Сохраняем результаты оценки в файл
with open('./output/modelling.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели: проверка гипотезы №3 ******',
          file=fln)
    print(fitmod03.summary(), file=fln)
    rss = fitmod03.ssr
    print('Сумма квадратов остатков: ', rss, file=fln)
    n = X['acre_lot'].size
    k = X.size/n
    hqc = n * math.log(rss/n)+2*k*math.log(math.log(n))
    print('HQC: ', hqc, file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod03.rsquared_adj, fitmod03.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_03']).T
mq = pd.concat([mq, q])   

"""
Промежуточная модель совмещает в себе альтернативную гипотезу 2 и
гипотезу 3.
"""

thr = 134.26853707414827 # Порог пробега - подобранный
X_4 = X.copy()

acre_thr = X_4['acre_lot'] - X_4['house_size'] >= thr
X_4['ath'] = X_4['acre_lot']*acre_thr # Взаимодействие
X_4['hth'] = X_4['house_size']*acre_thr

X_4["bed_2 * bath_2"] = X_4["bath_2"] * X_4["bed_2"]
X_4["bed_2 * bath_3"] = X_4["bath_3"] * X_4["bed_2"]
X_4["bed_2 * bath_4"] = X_4["bath_4"] * X_4["bed_2"]

X_4["bed_3 * bath_2"] = X_4["bath_2"] * X_4["bed_3"]
X_4["bed_3 * bath_3"] = X_4["bath_3"] * X_4["bed_3"]
X_4["bed_3 * bath_4"] = X_4["bath_4"] * X_4["bed_3"]

X_4["bed_4 * bath_2"] = X_4["bath_2"] * X_4["bed_4"]
X_4["bed_4 * bath_3"] = X_4["bath_3"] * X_4["bed_4"]
X_4["bed_4 * bath_4"] = X_4["bath_4"] * X_4["bed_4"]

X_4["bed_5 * bath_2"] = X_4["bath_2"] * X_4["bed_5"]
X_4["bed_5 * bath_3"] = X_4["bath_3"] * X_4["bed_5"]
X_4["bed_5 * bath_4"] = X_4["bath_4"] * X_4["bed_5"]

linreg04 = sm.OLS(Y,X_4)
fitmod04 = linreg04.fit()
# Сохраняем результаты оценки в файл
with open('./output/modelling.txt', 'a') as fln:
    print('\n ****** Оптимизация модели: №1 ******',
          file=fln)
    print(fitmod04.summary(), file=fln)
    rss = fitmod04.ssr
    print('Сумма квадратов остатков: ', rss, file=fln)
    n = X['acre_lot'].size
    k = X.size/n
    hqc = n * math.log(rss/n)+2*k*math.log(math.log(n))
    print('HQC: ', hqc, file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod04.rsquared_adj, fitmod04.aic], 
                 index=['adjR^2', 'AIC'], columns=['opt_1']).T
mq = pd.concat([mq, q])

"""
Оптимизация модели. Среди всех колонок была выбрана та,
которая увеличивает R-squared модели и уменьшает AIC. Показатель
adj. R-squared немного ухудшился по сравнению с предыдущей версией,
но в общем можно наблюдать улучшение остальных показателей и
уменьшение мультиколлинеарности. Далее было проверено, что
дальнейшее удаление будет только уменьшать показатели модели.
Итоговая и в то же время оптимальная модель представлена ниже.
Мультиколлинеарность высокая, но показатели у модели выше, чем
у базовой.
"""

X_5 = X_4.copy()

X_5.drop(columns=["bath_4"], inplace=True)

linreg05 = sm.OLS(Y, X_5)
fitmod05 = linreg05.fit()
# Сохраняем результаты оценки в файл
with open('./output/modelling.txt', 'a') as fln:
    print('\n ****** Оптимизация модели: №2 ******',
        file=fln)
    print(fitmod05.summary(), file=fln)
    rss = fitmod05.ssr
    print('Сумма квадратов остатков: ', rss, file=fln)
    n = X['acre_lot'].size
    k = X.size/n
    hqc = n * math.log(rss/n)+2*k*math.log(math.log(n))
    print('HQC: ', hqc, file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod05.rsquared_adj, fitmod05.aic], 
                index=['adjR^2', 'AIC'], columns=['opt_2']).T
mq = pd.concat([mq, q])

# Оптимизация далее идет во вред R-squared
# Уменьшаем мультиколлинеарность и отслеживаем AIC

X_6 = X_5.copy()
X_6.drop(columns=["acre_lot"], inplace=True)

linreg06 = sm.OLS(Y, X_6)
fitmod06 = linreg06.fit()
# Сохраняем результаты оценки в файл
with open('./output/modelling.txt', 'a') as fln:
    print('\n ****** Оптимизация модели: №3 ******',
        file=fln)
    print(fitmod06.summary(), file=fln)
    rss = fitmod06.ssr
    print('Сумма квадратов остатков: ', rss, file=fln)
    n = X['acre_lot'].size
    k = X.size/n
    hqc = n * math.log(rss/n)+2*k*math.log(math.log(n))
    print('HQC: ', hqc, file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod06.rsquared_adj, fitmod06.aic], 
                index=['adjR^2', 'AIC'], columns=['opt_3']).T
mq = pd.concat([mq, q])

X_7 = X_6.copy()
X_7.drop(columns=["bed_5"], inplace=True)

linreg07 = sm.OLS(Y, X_7)
fitmod07 = linreg07.fit()
# Сохраняем результаты оценки в файл
with open('./output/modelling.txt', 'a') as fln:
    print('\n ****** Оптимизация модели: №4 ******',
        file=fln)
    print(fitmod06.summary(), file=fln)
    rss = fitmod07.ssr
    print('Сумма квадратов остатков: ', rss, file=fln)
    n = X['acre_lot'].size
    k = X.size/n
    hqc = n * math.log(rss/n)+2*k*math.log(math.log(n))
    print('HQC: ', hqc, file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod07.rsquared_adj, fitmod07.aic], 
                index=['adjR^2', 'AIC'], columns=['opt_4']).T
mq = pd.concat([mq, q])

print(mq)
