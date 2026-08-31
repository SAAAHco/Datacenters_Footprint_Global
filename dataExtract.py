# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:04:17 2026

@author: [anonymized for double-blind review]
"""

#%% External libraries

import pandas as pd

#%% Parameters

years = ['2022', '2023', '2024', '2025']
base_map = 'Texas' # Write 'World', 'Northwestern', 'Spain', or 'Texas'
cell_size = 1

file = f"D:\{base_map}_cellSize_{cell_size}.csv"

list_WS = pd.read_csv(file, sep=',')

dfData_list = []

for year in years:
    for ws in list_WS['WS_id']:
        file_WS = f"D:\{year}\GHCNh_{ws}_{year}.psv"
        print(file_WS)
        data = pd.read_csv(file_WS, sep='|', usecols=['STATION', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'temperature', 'dew_point_temperature', 'relative_humidity', 'station_level_pressure'])
        dfData_list.append(data)

AllData_df = pd.concat(dfData_list, ignore_index=True)

AllData_df.to_csv(f"D:\\allData_{base_map}_cellSize_{cell_size}.csv", index=False)

#%% Open data

# df = pd.read_csv(f"D:\\allData_Texas_cellSize_1.csv", sep=',')
