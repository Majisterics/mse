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
dfData["bed"] = dfData["bed"].astype(CategoricalDtype([0, 2, 3, 4, 5, 6, max(dfData["bed"])], ordered=True))
dfData["bath"] = dfData["bath"].astype(CategoricalDtype([0, 1, 2, 3, max(dfData["bath"])], ordered=True))

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
# DUM = DUM[['status_for_sale', 'status_second_sale']]
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
vif = pd.DataFrame() # Для хранения 
# X_q = X.select_dtypes(include='float64')# Только количественные регрессоры
# X_q = X_q[["acre_lot", "house_size"]]
vif["vars"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) 
              for i in range(X.shape[1])]
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

# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod00.rsquared_adj, fitmod00.aic], 
                 index=['adjR^2', 'AIC'], columns=['base_00']).T
mq = pd.concat([mq, q])    

# """
# Исключаем из базовой модели переменные, которые считаем ненужными
# На самом деле исключаем по одной, понимаем, что 

# """
# X_1 = X.drop('bed', axis=1)
# # Формируем объект, содержащий все исходные данные и методы для оценивания
# linreg01 = sm.OLS(Y,X_1)
# # Оцениваем модель
# fitmod01 = linreg01.fit()
# # Сохраняем результаты оценки в файл
# with open('./output/modelling.txt', 'a') as fln:
#     print('\n ****** Оценка базовой модели без bed ******',
#           file=fln)
#     print(fitmod01.summary(), file=fln)
    
# # Сохраняем данные о качестве модели
# q = pd.DataFrame([fitmod01.rsquared_adj, fitmod01.aic], 
#                  index=['adjR^2', 'AIC'], columns=['base_no_bed']).T
# mq = pd.concat([mq, q])

# X_1 = X.drop('bath', axis=1)
# # Формируем объект, содержащий все исходные данные и методы для оценивания
# linreg01 = sm.OLS(Y,X_1)
# # Оцениваем модель
# fitmod01 = linreg01.fit()
# # Сохраняем результаты оценки в файл
# with open('./output/modelling.txt', 'a') as fln:
#     print('\n ****** Оценка базовой модели без bath ******',
#           file=fln)
#     print(fitmod01.summary(), file=fln)
    
# # Сохраняем данные о качестве модели
# q = pd.DataFrame([fitmod01.rsquared_adj, fitmod01.aic], 
#                  index=['adjR^2', 'AIC'], columns=['base_no_bath']).T
# mq = pd.concat([mq, q])

# X_1 = X.drop('acre_lot', axis=1)
# # Формируем объект, содержащий все исходные данные и методы для оценивания
# linreg01 = sm.OLS(Y,X_1)
# # Оцениваем модель
# fitmod01 = linreg01.fit()
# # Сохраняем результаты оценки в файл
# with open('./output/modelling.txt', 'a') as fln:
#     print('\n ****** Оценка базовой модели без acre_lot ******',
#           file=fln)
#     print(fitmod01.summary(), file=fln)
    
# # Сохраняем данные о качестве модели
# q = pd.DataFrame([fitmod01.rsquared_adj, fitmod01.aic], 
#                  index=['adjR^2', 'AIC'], columns=['base_no_acre_lot']).T
# mq = pd.concat([mq, q])

# X_1 = X.drop('house_size', axis=1)
# # Формируем объект, содержащий все исходные данные и методы для оценивания
# linreg01 = sm.OLS(Y,X_1)
# # Оцениваем модель
# fitmod01 = linreg01.fit()
# # Сохраняем результаты оценки в файл
# with open('./output/modelling.txt', 'a') as fln:
#     print('\n ****** Оценка базовой модели без house_size ******',
#           file=fln)
#     print(fitmod01.summary(), file=fln)
    
# # Сохраняем данные о качестве модели
# q = pd.DataFrame([fitmod01.rsquared_adj, fitmod01.aic],
#                  index=['adjR^2', 'AIC'], columns=['base_no_house_size']).T
# mq = pd.concat([mq, q])    

# # Обратите внимание - модель стала хуже. Лучше вернуться к предыдущей.

# ****************** Проверки гипотез ******************

"""
Статус постройки влияет на цену
Чем новее постройка, тем она дороже
Модель для проверки:
price = a0 + a11*status_for_sale + a12*status_second_sale + a2*bed + a3*bath + a4*acre_lot + a5*house_size + v
*****************
Если гипотеза справедлива, то a11>0, a11>a12 и значима
*****************

Целевая переменная не меняется.

Результат: гипотеза принимается и считается значимой.

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
X_2["bed_bath"] = list(zip(CA_train["bed"], CA_train["bath"]))
X_2 = pd.concat([pd.get_dummies(X_2["bed_bath"]), X_2], axis=1)
X_2.drop(columns=["bed_bath"], inplace=True)
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
# Формируем dummy из качественных переменных
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
print(mq)
quit()

# Коэффициент при переменной взаимодействия не значим. Надо подбирать порог

# Предсказательная сила
Y_test = CA_test['price']
DUM = pd.get_dummies(CA_test[['music', 'signal']])
# Выбираем переменные для уровней, которые войдут в модель
# Будет исключен один - базовый. ВЛияние включенных уровней на зависимую 
# переменную отсчитывается от него
DUM = DUM[['music_есть', 'signal_есть']]
# Формируем pandas.DataFramee содержащий матрицу X объясняющих переменных 
# Добавляем слева фиктивные переменные
X_test = pd.concat([DUM, CA_test[['age', 'mlg']]], axis=1)
# Добавляем переменную равную единице для учета константы
X_test = sm.add_constant(X_test)
X_test = X_test.astype({'const':'uint8'})
# Генерация предсказаний на тестовом множестве 
pred_ols = fitmod00.get_prediction(X_test)
# Генерация доверительных интервалов с доверительной вероятностью alpha
frm = pred_ols.summary_frame(alpha=0.05)
iv_l = frm["obs_ci_lower"] # Нижняя граница доверительных интервалов
iv_u = frm["obs_ci_upper"] # Верхняя граница доверительных интервалов
fv = frm['mean'] # Предсказанное значение целевой переменной
# Построение графиков
name = 'mlg' # Имя переменной относительно которой строим прогноз
Z = X_test.loc[:, name]
dfn = pd.DataFrame([Z, Y_test, fv, iv_u, iv_l]).T
dfn = dfn.sort_values(by=name)
fig, ax = plt.subplots(figsize=(8, 6))
for z in dfn.columns[1:]:
    dfn.plot(x=dfn.columns[0], y=z, ax=ax)
ax.legend(loc="best")
plt.show()

# Подсчет среднеквадратической ошибки
dif = np.sqrt((dfn.iloc[:,1] - dfn.iloc[:,2]).pow(2).sum()/dfn.shape[0])

# Доля выходов за границы доверительых интервалов
# Сортируем, чтобы индексы во всех рядах совпадали
mn = dfn.iloc[:,1].sort_index() 
out = ((mn > iv_u) + (mn < iv_l)).sum()/dfn.shape[0]

# Гипотеза не отвергается. Модель стала заметно лучше.

"""
Сила влияния площади на цену зависит от 
Сила влияния пробега на цену зависит от пробега.
Скорость падения цены с ростом пробега до определенного значения больше, 
чем после него. 
Модель для проверки:
Вводим переменную
mlg_thr = 1, если mlg >= thr и 0, если нет
thr - неизвестный порог
!!! Порог надо подбирать экспериментально, увеличивая adjR^2 !!!
price = a0 + a1*'signal_есть'+ a2*'music_есть' + 
+ (a30 + a31*mlg_thr)*mlg + a4*age + v 
раскрывая скобки
price = a0 + a1*'signal_есть'+ a2*'music_есть' + 
+ a30*mlg + a31*mlg_thr*mlg + a4*age + v
*****************
Если гипотеза справедлива, то a30<0, a31>0 и значим 
*****************

Целевая переменная не меняется.

"""
thr = 40 # Порог пробега - вариант
X_3 = X.copy()
# Формируем dummy из качественных переменных
mlg_thr = X_3['mlg'] >= thr
X_3['mth'] = X_3['mlg']*mlg_thr # Взаимодействие
linreg04 = sm.OLS(Y,X_3)
fitmod04 = linreg04.fit()
# Сохраняем результаты оценки в файл
with open('./output/modelling.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod04.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod04.rsquared_adj, fitmod04.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_03']).T
mq = pd.concat([mq, q])   

# Коэффициент при переменной взаимодействия не значим. Надо подбирать порог

# Предсказательная сила
Y_test = CA_test['price']
DUM = pd.get_dummies(CA_test[['music', 'signal']])
# Выбираем переменные для уровней, которые войдут в модель
# Будет исключен один - базовый. ВЛияние включенных уровней на зависимую 
# переменную отсчитывается от него
DUM = DUM[['music_есть', 'signal_есть']]
# Формируем pandas.DataFramee содержащий матрицу X объясняющих переменных 
# Добавляем слева фиктивные переменные
X_test = pd.concat([DUM, CA_test[['age', 'mlg']]], axis=1)
# Добавляем переменную равную единице для учета константы
X_test = sm.add_constant(X_test)
X_test = X_test.astype({'const':'uint8'})
# Генерация предсказаний на тестовом множестве 
pred_ols = fitmod00.get_prediction(X_test)
# Генерация доверительных интервалов с доверительной вероятностью alpha
frm = pred_ols.summary_frame(alpha=0.05)
iv_l = frm["obs_ci_lower"] # Нижняя граница доверительных интервалов
iv_u = frm["obs_ci_upper"] # Верхняя граница доверительных интервалов
fv = frm['mean'] # Предсказанное значение целевой переменной
# Построение графиков
name = 'mlg' # Имя переменной относительно которой строим прогноз
Z = X_test.loc[:, name]
dfn = pd.DataFrame([Z, Y_test, fv, iv_u, iv_l]).T
dfn = dfn.sort_values(by=name)
fig, ax = plt.subplots(figsize=(8, 6))
for z in dfn.columns[1:]:
    dfn.plot(x=dfn.columns[0], y=z, ax=ax)
ax.legend(loc="best")
plt.show()

# Подсчет среднеквадратической ошибки
dif = np.sqrt((dfn.iloc[:,1] - dfn.iloc[:,2]).pow(2).sum()/dfn.shape[0])

# Доля выходов за границы доверительых интервалов
# Сортируем, чтобы индексы во всех рядах совпадали
mn = dfn.iloc[:,1].sort_index() 
out = ((mn > iv_u) + (mn < iv_l)).sum()/dfn.shape[0]


