#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 20:24:19 2023

@author: vill
"""


import numpy as np
import pandas as pd
import os
from pathlib import Path


path: str = str(Path.home().joinpath("Major2023/SoftwareEngineering/work"))
os.chdir(path)

x = np.arange(16)
x.shape = (4, 4)
print(x)

z = np.array([[1, 'a'], [2, 'b']], dtype=object)
print(z)

ind = z[:, 1] == 'a'

Stuff = np.array([['Номер', 'Фамилия', 'Департамент', 'Зарплата'], 
                  [1, 'Петров', 'маркетинг', 34000], 
                  [2, 'Федоров', 'финансы', 35000],
                  [3, 'Ткачева', 'финансы', 22000],
                  [4, 'Самсонова', 'маркетинг', 36000],
                  [5, 'Каштанов', 'маркетинг', 26000]], dtype='O')

def get_attr(arr: np.ndarray, name: str):
    for i, arr_name in enumerate(arr[0]):
        if arr_name == name:
            return i
    return -1

attr_ind = get_attr(Stuff, 'Номер')
print(Stuff[1:, attr_ind])

def get_attr(arr, name):
    return arr[1:, arr[0, :] == name]

print(get_attr(Stuff, 'Фамилия'))
