# -*- coding: utf-8 -*-
"""
МИСИС
Программная инженерия

@author: Карнаушко В. А.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.api.types import CategoricalDtype
import numpy as np
from scipy.stats import norm


pWork: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent

pData: Path = pWork.joinpath("data")
pImages: Path = pWork.joinpath("images")
dfData = pd.read_csv(pData.joinpath("processed-realtor-data").with_suffix(".csv"))

print(dfData.head())
print(dfData.shape[0])

# acre_lot, house_size, price -- количественные переменные
# status, bed, bath -- порядковые переменные

# 1. Гистограмма для acre_lot
ax = dfData["acre_lot"].plot.hist(density=True, bins=30, xlabel="land size in square meters", ylabel="frequency")
x = np.arange(min(dfData["acre_lot"]), max(dfData["acre_lot"]), 0.001)
plt.plot(x, norm.pdf(x, dfData["acre_lot"].mean(), dfData["acre_lot"].std()), "--")
plt.axvline(x=dfData["acre_lot"].mean(), color="orange")
plt.axvline(x=dfData["acre_lot"].median(), color="orangered")
ax.get_figure().savefig(pImages.joinpath("acre_lot_hist").with_suffix(".png"))
ax.get_figure().clear()

# 2. Гистограмма для house_size
ax = dfData["house_size"].plot.hist(density=True, bins=30, xlabel="house size in square meters", ylabel="frequency")
x = np.arange(min(dfData["house_size"]), max(dfData["house_size"]), 0.001)
plt.plot(x, norm.pdf(x, dfData["house_size"].mean(), dfData["house_size"].std()), "--")
plt.axvline(x=dfData["house_size"].mean(), color="orange")
plt.axvline(x=dfData["house_size"].median(), color="orangered")
ax.get_figure().savefig(pImages.joinpath("house_size_hist").with_suffix(".png"))
ax.get_figure().clear()

# 3. Гистограмма для price
ax = dfData["price"].plot.hist(density=True, bins=30, xlabel="price", ylabel="frequency")
x = np.arange(min(dfData["price"]), max(dfData["price"]), 0.1)
plt.plot(x, norm.pdf(x, dfData["price"].mean(), dfData["price"].std()), "--")
plt.axvline(x=dfData["price"].mean(), color="orange")
plt.axvline(x=dfData["price"].median(), color="orangered")
ax.get_figure().savefig(pImages.joinpath("price_hist").with_suffix(".png"))
ax.get_figure().clear()

# Графики для качественных переменных

dfData["bed"] = dfData["bed"].astype("category")
dfData["bath"] = dfData["bath"].astype("category")

# 4. Диаграмма для bed
ax = dfData["bed"].value_counts(sort=False, normalize=True).plot.bar(rot=0, figsize=(12, 6), ylabel="frequency", xlabel="number of bedrooms")
ax.get_figure().savefig(pImages.joinpath("bed_bar").with_suffix(".png"))
ax.get_figure().clear()

# 5. Диаграмма для bath
ax = dfData["bath"].value_counts(sort=False, normalize=True).plot.bar(rot=0, figsize=(12, 6), ylabel="frequency", xlabel="number of bathrooms")
ax.get_figure().savefig(pImages.joinpath("bath_bar").with_suffix(".png"))
ax.get_figure().clear()

# 6. Диаграмма для status
status_cat_type = CategoricalDtype(categories=["for_sale", "second_sale"], ordered=True)
dfData["status"] = dfData["status"].astype(status_cat_type)

ax = dfData["status"].value_counts(sort=False, normalize=True).plot.bar(ylabel="frequency", xlabel="status", rot=0, figsize=(6,4))
ax.get_figure().savefig(pImages.joinpath("status_bar").with_suffix(".png"))
ax.get_figure().clear()

# Группировка категорий

dfBed = pd.cut(dfData["bed"], bins=[0, 1, 2, 3, 4, 5, 6, 7, max(dfData["bed"])])
dfBedNorm = dfBed.value_counts(normalize=True, sort=False)
ax = dfBedNorm.plot.bar(rot=0, figsize=(12, 6), ylabel="frequency", xlabel="range of bedrooms")
ax.get_figure().savefig(pImages.joinpath("bed_binned_bar").with_suffix(".png"))
ax.get_figure().clear()

dfBath = pd.cut(dfData["bath"], bins=[0, 1, 2, 3, max(dfData["bath"])])
dfBathNorm = dfBath.value_counts(normalize=True, sort=False)
ax = dfBathNorm.plot.bar(rot=0, ylabel="frequency", xlabel="range of bathrooms")
ax.get_figure().savefig(pImages.joinpath("bath_binned_bar").with_suffix(".png"))
ax.get_figure().clear()

# Статистические взаимосвязи

firstsalePrice = dfData[dfData["status"] == "for_sale"]["price"]
secondsalePrice = dfData[dfData["status"] == "second_sale"]["price"]

dfPrice = pd.DataFrame({
    "for_sale": firstsalePrice,
    "second_sale": secondsalePrice
})

# visually detected huge and strange prices originally here
ax = dfPrice.plot.box(column=["for_sale", "second_sale"], ylabel="price", xlabel="status")
ax.get_figure().savefig(pImages.joinpath("status_box").with_suffix(".png"))
ax.get_figure().clear()

from scipy.stats import kruskal # 3 и более групп
from scipy.stats import mannwhitneyu # 2 группы

kruskal_res = kruskal(firstsalePrice, secondsalePrice, nan_policy="omit")
print("-- kruskal --")
print(kruskal_res)

mannwhitneyu_res = mannwhitneyu(firstsalePrice, secondsalePrice, nan_policy="omit")
print("-- mannwhitneyu --")
print(mannwhitneyu_res)

# bed -- price

group1 = dfData["bed"] == 1
group2 = dfData["bed"] == 2
group3 = dfData["bed"] == 3
group4 = dfData["bed"] == 4
group5 = dfData["bed"] == 5
group6 = dfData["bed"] == 6
group7 = dfData["bed"] == 7
singular_groups = group1 | group2 | group3 | group4 | group5 | group6 | group7

bed_group1 = dfData[group1]["price"]
bed_group2 = dfData[group2]["price"]
bed_group3 = dfData[group3]["price"]
bed_group4 = dfData[group4]["price"]
bed_group5 = dfData[group5]["price"]
bed_group6 = dfData[group6]["price"]
bed_group7 = dfData[group7]["price"]
bed_group8 = dfData[~singular_groups]["price"]

dfBedGroup = pd.DataFrame({
    "1": bed_group1,
    "2": bed_group2,
    "3": bed_group3,
    "4": bed_group4,
    "5": bed_group5,
    "6": bed_group6,
    "7": bed_group7,
    "8+": bed_group8
})

ax = dfBedGroup.plot.box(column=["1", "2", "3", "4", "5", "6", "7", "8+"], ylabel="price", xlabel="range of bedrooms")
ax.get_figure().savefig(pImages.joinpath("bed_box").with_suffix(".png"))
ax.get_figure().clear()

# bath -- price

group1 = dfData["bath"] == 1
group2 = dfData["bath"] == 2
group3 = dfData["bath"] == 3
singular_groups = group1 | group2 | group3

bath_group1 = dfData[group1]["price"]
bath_group2 = dfData[group2]["price"]
bath_group3 = dfData[group3]["price"]
bath_group4 = dfData[~singular_groups]["price"]

dfBathGroup = pd.DataFrame({
    "1": bath_group1,
    "2": bath_group2,
    "3": bath_group3,
    "4+": bath_group4,
})

ax = dfBathGroup.plot.box(column=["1", "2", "3", "4+"], ylabel="price", xlabel="range of bathrooms")
ax.get_figure().savefig(pImages.joinpath("bath_box").with_suffix(".png"))
ax.get_figure().clear()
