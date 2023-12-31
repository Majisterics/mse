import os
import pandas as pd
import logging
from pathlib import Path
from pandas.api.types import CategoricalDtype


pWork: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent
logging.basicConfig(level=logging.INFO, filename=pWork.joinpath("logs").joinpath("out_removal.log"), filemode="w")

pData: Path = pWork.joinpath("data")
dfData = pd.read_csv(pData.joinpath("processed-realtor-data").with_suffix(".csv"))

status_cat_type = CategoricalDtype(categories=["ready_to_build", "for_sale", "second_sale"], ordered=True)
dfData["status"] = dfData["status"].astype(status_cat_type)

dfDataFloat = dfData.select_dtypes(include='float')
dfDescribe = dfDataFloat.describe()
IQR: pd.Series = dfDataFloat.quantile(q=0.75) - dfDataFloat.quantile(q=0.25)
dfIQR = pd.DataFrame([IQR], index=['IQR'])
dfStats = pd.concat([dfDescribe, dfIQR])

irq_acre_lot = dfIQR['acre_lot']
wisker_u_acre_lot = (dfStats.loc['50%', 'acre_lot'] + 1.5*irq_acre_lot).values[0]
wisker_l_acre_lot = (dfStats.loc['50%', 'acre_lot'] - 1.5*irq_acre_lot).values[0]

irq_house_size = dfIQR['house_size']
wisker_u_house_size = (dfStats.loc['50%', 'house_size'] + 1.5*irq_house_size).values[0]
wisker_l_house_size = (dfStats.loc['50%', 'house_size'] - 1.5*irq_house_size).values[0]

irq_price = dfIQR['price']
wisker_u_price = (dfStats.loc['50%', 'price'] + 1.5*irq_price).values[0]
wisker_l_price = (dfStats.loc['50%', 'price'] - 1.5*irq_price).values[0]

bdfAcre = (dfData['acre_lot'] > wisker_u_acre_lot) | (dfData['acre_lot'] <= wisker_l_acre_lot)
bdfHouse = (dfData['house_size'] > wisker_u_house_size) | (dfData['house_size'] <= wisker_l_house_size)
bdfPrice = (dfData['price'] > wisker_u_price) | (dfData['price'] <= wisker_l_price)

out = bdfAcre | bdfHouse | bdfPrice

logging.info(f"Out values\n" +
             f"acre_lot: {len(dfData[bdfAcre])}, {len(dfData[bdfAcre]) * 100 / len(dfData):.2f}%\n" +
             f"house_size: {len(dfData[bdfHouse])}, {len(dfData[bdfHouse]) * 100 / len(dfData):.2f}%\n" +
             f"price: {len(dfData[bdfPrice])}, {len(dfData[bdfPrice]) * 100 / len(dfData):.2f}%\n" +
             f"combined all: {len(dfData[out])}, {len(dfData[out]) * 100 / len(dfData):.2f}%\n")

logging.info(f"After removing\nwere: {len(dfData)}\nnow are: {len(dfData[~out])}\n")

dfData = dfData[~out]

dfData.to_csv(pData.joinpath("processed-realtor-data").with_suffix(".csv"), index=False)
