# -*- coding: utf-8 -*-
"""
МИСИС
Программная инженерия

@author: Карнаушко В. А.
"""
import os
import pandas as pd
import logging
from pathlib import Path
from pandas.api.types import CategoricalDtype
from scipy.stats import pearsonr
from scipy.stats import spearmanr


pWork: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent
logging.basicConfig(level=logging.INFO, filename=pWork.joinpath("logs").joinpath("data_description.txt"), filemode="w")

pData: Path = pWork.joinpath("data")
dfData = pd.read_csv(pData.joinpath("processed-realtor-data").with_suffix(".csv"))

# bed and bath contain NaN values, so they cannot be casted to int64 dtype
logging.info(f"Interesting\nColumns bed and bath contain NaN values, so they cannot be casted to int64 dtype.\n")

status_cat_type = CategoricalDtype(categories=["ready_to_build", "for_sale", "second_sale"], ordered=True)
dfData["status"] = dfData["status"].astype(status_cat_type)

print(dfData)
print(dfData.dtypes)

# preparing statistics
dfDataFloat = dfData.select_dtypes(include='float')
dfDescribe = dfDataFloat.describe()
IQR: pd.Series = dfDataFloat.quantile(q=0.75) - dfDataFloat.quantile(q=0.25)
dfIQR = pd.DataFrame([IQR], index=['IQR'])
dfStats = pd.concat([dfDescribe, dfIQR])

logging.info(f"Results\n{dfStats}\n")

C_P = pd.DataFrame([], index=dfDataFloat.columns, columns=dfDataFloat.columns) 
P_P = pd.DataFrame([], index=dfDataFloat.columns, columns=dfDataFloat.columns)
C_S = pd.DataFrame([], index=dfDataFloat.columns, columns=dfDataFloat.columns)
P_S = pd.DataFrame([], index=dfDataFloat.columns, columns=dfDataFloat.columns)
for x in dfDataFloat.columns:
    for y in dfDataFloat.columns:
        C_P.loc[x, y], P_P.loc[x, y] = pearsonr(dfDataFloat[x], dfDataFloat[y])
        C_S.loc[x, y], P_S.loc[x, y] = spearmanr(dfDataFloat[x], dfDataFloat[y])

# saving statistics and corellation to output folder
with pd.ExcelWriter('./output/realtor-stats.xlsx', engine="openpyxl") as wrt:
# Общая статистика
    dfStats.to_excel(wrt, sheet_name='stat')
# Корреляция Пирсона
    C_P.to_excel(wrt, sheet_name='Pearson')
    dr = C_P.shape[0] + 2
    P_P.to_excel(wrt, startrow=dr, sheet_name='Pearson') # Значимость
# Корреляция Спирмена
    C_S.to_excel(wrt, sheet_name='Spirmen')
    dr = C_S.shape[0] + 2
    P_S.to_excel(wrt, startrow=dr, sheet_name='Spirmen') # Значимость
