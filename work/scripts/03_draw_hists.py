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


pWork: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent

pData: Path = pWork.joinpath("data")
pImages: Path = pWork.joinpath("images")
dfData = pd.read_csv(pData.joinpath("processed-realtor-data").with_suffix(".csv"))

print(dfData.head())
print(dfData.shape[0])

# acre_lot, house_size, price -- количественные переменные
# status, bed, bath -- порядковые переменные

# 1. Гистограмма для acre_lot
ax = dfData["acre_lot"].plot.hist()
ax.get_figure().savefig(pImages.joinpath("acre_lot_hist").with_suffix(".png"))
ax.get_figure().clear()

# 2. Гистограмма для house_size
ax = dfData["house_size"].plot.hist()
ax.get_figure().savefig(pImages.joinpath("house_size_hist").with_suffix(".png"))
ax.get_figure().clear()

# 3. Гистограмма для price
ax = dfData["price"].plot.hist()
ax.get_figure().savefig(pImages.joinpath("price_hist").with_suffix(".png"))
ax.get_figure().clear()

# Дополнительно и не необходимо

# 4. Гистограмма для bed
ax = dfData["bed"].plot.hist()
ax.get_figure().savefig(pImages.joinpath("bed_hist").with_suffix(".png"))
ax.get_figure().clear()

# 5. Гистограмма для bath
ax = dfData["bath"].plot.hist()
ax.get_figure().savefig(pImages.joinpath("bath_hist").with_suffix(".png"))
ax.get_figure().clear()

# 6. Гистограмма для status
status_cat_type = CategoricalDtype(categories=["for_sale", "second_sale"], ordered=True)
dfData["status"] = dfData["status"].astype(status_cat_type)
# print(dfData.dtypes)
# dfData["status_codes"] = dfData["status"].cat.codes

dfStatus = pd.DataFrame({"status": ["for_sale", "second_sale"], "values": [len(dfData[dfData["status"] == "for_sale"]), len(dfData[dfData["status"] == "second_sale"])]})
ax = dfStatus.plot.bar(x="status", rot=0)
ax.get_figure().savefig(pImages.joinpath("status_bar").with_suffix(".png"))
ax.get_figure().clear()
