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


pWork: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent
logging.basicConfig(level=logging.INFO, filename=pWork.joinpath("logs").joinpath("preprocessing.txt"), filemode="w")

pData: Path = pWork.joinpath("data")
dfData = pd.read_csv(pData.joinpath("realtor-data").with_suffix(".csv"))

# logging example for data in dataset
logging.info("\n" + str(dfData.head()) + "\n")
# logging count of not NaN values
logging.info("\n" + str(dfData.count()) + "\n")

# If we exclude NaN from prev_sold_date we leave almost 51% of data
# print(dfData[dfData["prev_sold_date"].notna()]["prev_sold_date"].size / dfData["prev_sold_date"].size)

# --- DISABLED LOGS FOR EFFICIENCY ---
# check if the connection city to zip_code is one-to-one
# dfCitiesZips = dfData.replace([np.inf, -np.inf], np.nan)
# dfCitiesZips.dropna(subset=["city", "zip_code"], how="any", inplace=True)

# used_city: list = []
# city_zip: dict = {}
# for city in dfCitiesZips["city"]:
#     if city in used_city:
#         continue
#     city_zip[city] = []
#     dfCurr = dfCitiesZips[dfCitiesZips["city"] == city]
#     for zip_code in dfCurr["zip_code"]:
#         if zip_code not in city_zip[city]:
#             city_zip[city].append(zip_code)
#     used_city.append(city)

# output = ""
# for city, zip_list in city_zip.items():
#     if len(zip_list) > 1:
#         output += f"{city}\t{zip_list}\n"

# logging that city to zip_code is not one-to-one
# logging.info(f"\n{output}")
# --- DISABLED LOGS FOR EFFICIENCY ---

# State <- City <- Zip Code
# That is the relationship between space or location defining features
# We don't use them in our hypothesis
# So we can leave all of them

# dropping unused columns
dfProcessed = dfData.drop(["state", "city", "zip_code"], axis=1)
# adding third category to status
nan_mask = (dfProcessed["status"] == "for_sale") & (~dfProcessed["prev_sold_date"].notna())
dfProcessed.loc[nan_mask, "status"] = "second_sale"

logging.info(f"\n{dfProcessed['status'].unique()}\n")
logging.info(f"\nready_to_build:\t{dfProcessed[dfProcessed['status'] == 'ready_to_build']['status'].count() / dfProcessed['status'].count() * 100:.1f}%\n")
logging.info(f"\nfor_sale:\t{dfProcessed[dfProcessed['status'] == 'for_sale']['status'].count() / dfProcessed['status'].count() * 100:.1f}%\n")
logging.info(f"\nsecond_sale:\t{dfProcessed[dfProcessed['status'] == 'second_sale']['status'].count() / dfProcessed['status'].count() * 100:.1f}%\n")

dfProcessed.to_csv(pData.joinpath("processed-realtor-data").with_suffix(".csv"), index=False)
