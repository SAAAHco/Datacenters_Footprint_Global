# -*- coding: utf-8 -*-
"""
Hourly data for all the available weather stations WS are in separate files in folders.
There is a folder for each year. A file for each WS is in the corresponding year folder.
The numbers of files (WS) in each of the folders are different. The common WS in the list of years is calculated.
Each WS has an unique 11 characters identifier like "AEI0000OMAA".
"""

#%% External libraries

import sys, pandas as pd, geopandas as gpd, numpy as np, matplotlib as mpl
from shapely.geometry import Point, Polygon

#%% Parameters

years = ['2022', '2023', '2024', '2025']

cell_size = 1 # Longitude and latitude resolution in sexagesimal grades 
base_map = 'Spain' # Write 'World', 'Northwestern', 'Spain', or 'Texas'

# Geographic coordinates system
epsg = 4326 # WGS84 geographic coordinates

# WS with elevation diference respect to median are dropped
elevThreshold = 10

#%% Common WS in a set of years
"""
# List of weather stations common in years and with required sensors

dataframes = {} # List of dataframes, one for each year containing the names of the WS with available data
df_intersec = pd.DataFrame()
 
for year in years:    
    dataframes[f"df_{year}"] = pd.DataFrame(os.listdir(f"D:\{year}\."), columns=['WS']) # List of files are converted into a dataframe
    dataframes[f"df_{year}"]['WS'] = dataframes[f"df_{year}"]['WS'].str[6:17] # Characters different to WS id are removed
    
    if df_intersec.empty:
        df_intersec = dataframes[f"df_{year}"]
    else:
        df_intersec = pd.merge(df_intersec['WS'], dataframes[f"df_{year}"]['WS']) # Inner join of two tables
    
print(f"{df_intersec.size} WS in the year list {years}.")


#%% Check if WS have required sensors (temperature, temp. dew point or relative humidity, pressure). Rows are dropped

ws_No_sensors=[]
n=0

df_intersec[['numNoTemp', 'numNoHum', 'numNoPres']] = None

for i in range(df_intersec.index.size):
    ws = df_intersec.iloc[i]['WS']
    file_WS = f"D:\\2022\GHCNh_{ws}_2022.psv"

    df_WS = pd.read_csv(file_WS, sep='|', usecols=['temperature', 'dew_point_temperature', 'relative_humidity', 'station_level_pressure'])
    
    # sensor_availabe == 1, all sensors needed are present
    tempSensor = df_WS['temperature'].isna().all()
    num_No_temp = df_WS['temperature'].isna().sum()
    dewSensor = df_WS['dew_point_temperature'].isna().all()
    num_No_dew = df_WS['dew_point_temperature'].isna().sum()
    relhumSensor = df_WS['relative_humidity'].isna().all()
    num_No_rel = df_WS['relative_humidity'].isna().sum()
    num_No_hum = min(num_No_dew, num_No_rel)
    preSensor = df_WS['station_level_pressure'].isna().all()
    num_No_pre = df_WS['station_level_pressure'].isna().sum()
    
    print(i)
    if (tempSensor | (dewSensor & relhumSensor)) | preSensor == True:
        ws_No_sensors.append(i)
        n=n+1
    else:
        df_intersec.loc[i, ['numNoTemp', 'numNoHum', 'numNoPres']] = [num_No_temp, num_No_hum, num_No_pre]
    
    print(tempSensor, dewSensor, relhumSensor, preSensor, n)
    
df_intersec = df_intersec.drop(ws_No_sensors).reset_index(drop=True)
 
df_intersec.to_csv("D:\\dataframe_WS_intersect.csv", index=False)

"""
df_intersec = pd.read_csv("D:\\dataframe_WS_intersect.csv", sep=',')

#%% Region definition

# Map Texas
file_Texas = r"C:\Users\raulsanchez\1_UPM\NubeUPM\Personal\2025_TAMU\Work_group_TAMU\01_DC_heat_energy_and_water_evaluation\Publication_attemps\model_script\Texas_State_Boundary_3751460527680346532\State_Boundary.shp"
map_Texas = gpd.read_file(file_Texas)

# World map
file_countries = r"C:\Users\raulsanchez\1_UPM\NubeUPM\Personal\2025_TAMU\Work_group_TAMU\06_DC_footprint_map\World_Countries_(Generalized)_-7542350475356456390\World_Countries_Generalized.shp"
worldmap_countries = gpd.read_file(file_countries)

# List with weather stations    
weather_stations = pd.read_excel('WS_list.xlsx', usecols=['WS_id', 'Latitude',	'Longitude',	'Elevation (m)',	'Location'])
weather_stations = pd.merge(weather_stations, df_intersec, left_on='WS_id', right_on='WS')
weather_stations = weather_stations.drop(columns=['WS'])

# Regions classified by climate (Spherical squares 0.5º x 0.5º)
# fileRegions = r"C:\Users\raulsanchez\1_UPM\NubeUPM\Personal\2025_TAMU\Work_group_TAMU\06_DC_footprint_map\c1976_2000_0\c1976_2000.shp"
# map_regions = gpd.read_file(fileRegions)
# Adding area of the polygons
# map_regions['area'] = map_regions['geometry'].area
# map_regions['centroid'] = map_regions['geometry'].centroid

# Flat area in km^2 A=R^2*(Long2-Long1)*(sinLat2-sinLat1)
# map_regions['flat_area'] = 6371**2*np.abs(map_regions['geometry'].bounds.maxx-map_regions['geometry'].bounds.minx)*np.pi/180*(np.sin(map_regions['geometry'].bounds.maxy*np.pi/180)-np.sin(map_regions['geometry'].bounds.miny*np.pi/180))

# Planar coordinate reference system CRS
# map_regions = map_regions.to_crs(epsg=epsg)
worldmap_countries = worldmap_countries.to_crs(epsg=epsg)
map_Texas = map_Texas.to_crs(epsg=epsg)

geometry = [Point(xy) for xy in zip(weather_stations['Longitude'], weather_stations['Latitude'])]
weather_stations_df = gpd.GeoDataFrame(weather_stations, 
                          crs = epsg, 
                          geometry = geometry)
# Weather stations in each climatic region
# ws_in_regions = gpd.sjoin(weather_stations_df, map_regions, how='inner', predicate='within')

# 1. Bounding box limits (min/max longitude and latitude)

if base_map == 'World':
    # World
    minx, maxx = -180, 180  # Longitude
    miny, maxy = -90, 90    # Latitude
    
elif base_map == 'Northwestern':
    # Northwestern
    minx, maxx = -130, 50  # Longitude
    miny, maxy = 20, 60    # Latitude
    
elif base_map == 'Spain':    
    # Spain
    minx, maxx = -9.8, 4.5  # Longitude
    miny, maxy = 35, 44    # Latitude

elif base_map == 'Texas':
    # Texas
    minx, maxx = -107, -91  # Longitude
    miny, maxy = 25, 37.5    # Latitude

else:
    sys.exit('Write "World" or "Texas" defining base_map variable')
    
# Weather stations in each climatic region
weather_stations_df_in_region = weather_stations_df[((weather_stations_df['Latitude']>=miny) & (weather_stations_df['Latitude']<=maxy)) |
                                          ((weather_stations_df['Latitude']<=-miny) & (weather_stations_df['Latitude']>=-maxy))]

# 2. Set the step size (grid cell size in degrees)

cell_size = cell_size

#%% Weather stations in the grid

# 3. Create arrays of X and Y coordinate boundaries
x_coords = np.arange(minx, maxx, cell_size)
# y_coords = np.arange(miny, maxy, cell_size)

# --- Start here ---
# Latitude for equal area, which equals to max. area in the equator. Earth radious 6371 km
Area_max = 6371**2*np.abs(cell_size/2-(-cell_size/2))*np.pi/180*(np.sin(cell_size/2*np.pi/180)-np.sin(-cell_size/2*np.pi/180))

thetaMax = 70
theta1 = -thetaMax
theta2 = np.array([-thetaMax])

while theta2[-1] < thetaMax:
    new_theta2 = np.arcsin(np.sin(theta1*np.pi/180)+Area_max/6371**2/(np.abs(cell_size/2-(-cell_size/2))*np.pi/180))*180/np.pi
    theta2 = np.append(theta2, new_theta2)
    theta1 = new_theta2

y_coords = theta2

xx, yy = np.meshgrid(x_coords, y_coords)
points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]

# --- End here ---

# 4. Grid cells
grid_cells = []
for i in range(len(x_coords)-1):
    for j in range(len(y_coords)-1):
        grid_cells.append(Polygon([(x_coords[i], y_coords[j]), (x_coords[i+1], y_coords[j]), (x_coords[i+1], y_coords[j+1]), (x_coords[i], y_coords[j+1])]))

# 5. Convert to a GeoPandas GeoDataFrame with WGS84 CRS (EPSG:4326)
grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=epsg)

# grid_gdf['area'] = grid_gdf['geometry'].area
# grid_gdf['centroid'] = grid_gdf['geometry'].centroid

# Flat area in km2 A=R^2*(Long2-Long1)*(sinLat2-sinLat1)
grid_gdf['flat_area'] = 6371**2*np.abs(grid_gdf['geometry'].bounds.maxx-grid_gdf['geometry'].bounds.minx)*np.pi/180*(np.sin(grid_gdf['geometry'].bounds.maxy*np.pi/180)-np.sin(grid_gdf['geometry'].bounds.miny*np.pi/180))

#%% WS in selected region

if ((base_map == 'World') | (base_map=='Northwestern')) | (base_map=='Spain'):
    # WS World
    ws_in_World = gpd.sjoin(weather_stations_df_in_region, worldmap_countries, how='inner', predicate='within')
    ws_in_World = ws_in_World.drop(columns=['index_right', 'COUNTRYAFF', 'AFF_ISO'])
    
    # WS in grid
    grids_join_WS = gpd.sjoin(grid_gdf, ws_in_World, how='left', predicate='contains')
    
else:
    # WS only Texas
    ws_in_Texas = gpd.sjoin(weather_stations_df_in_region, map_Texas, how='inner', predicate='within')
    ws_in_Texas = ws_in_Texas.drop(columns=['index_right', 'FIPS', 'CREATE_USE', 'CMNT'])
    
    # WS in grid
    grids_join_WS = gpd.sjoin(grid_gdf, ws_in_Texas, how='left', predicate='contains')
    
# Number of WS in each cell of the grid
grid_gdf.loc[grids_join_WS.index[grids_join_WS['index_right'].notna()], 'num_WS'] = 1
grid_gdf.loc[grids_join_WS.index.value_counts()>1, 'num_WS'] = grids_join_WS.index.value_counts()

# Relation WS/km2
grid_gdf['WS_km2'] = grid_gdf['num_WS']/grid_gdf['flat_area']

# Elevation difference of WS in a cell
grid_gdf['min_Elevation'] = grids_join_WS.groupby(level=0)['Elevation (m)'].min()
grid_gdf['max_Elevation'] = grids_join_WS.groupby(level=0)['Elevation (m)'].max()
grid_gdf['median_Elevation'] = grids_join_WS.groupby(level=0)['Elevation (m)'].median()

# Asumption of minimum data fail
grids_join_WS['numNoData'] = grids_join_WS[['numNoTemp', 'numNoHum','numNoPres']].max(axis=1)

# Elevation difference respect to mean value of WS in the cell
# Drop WS with |elevDiff|>200 m and 3 or more
elevThreshold = elevThreshold
grids_join_WS['elevDiff'] = grids_join_WS['Elevation (m)']-grids_join_WS.groupby(level=0)['Elevation (m)'].median()
grids_join_WS['allElevDiff'] = (grids_join_WS['elevDiff'] < -elevThreshold) | (grids_join_WS['elevDiff'] > elevThreshold)

allBigThaThre = (grids_join_WS.groupby(level=0)['allElevDiff'].all() == True)

grids_join_WS['num_WS'] = grid_gdf.loc[grids_join_WS.index]['num_WS']

# WS selection in each cell (Random or WS with less fails)

#_1_ Random sample of 1 (n=1) WS in each cell
# grid_sample_WS = grids_join_WS.groupby(grids_join_WS.index).sample(n=1, random_state=None)

#-2_ Selection of the WS with less fails
grids_join_WS = grids_join_WS.reset_index()

# Drop WS with |elevDiff|>200 m and 3 or more

greaterThan2 = grids_join_WS['num_WS'] >= 3
greaterElevThres = (grids_join_WS['elevDiff']>elevThreshold) | (grids_join_WS['elevDiff']<(-elevThreshold))

grids_join_WS = grids_join_WS.drop(grids_join_WS[(greaterThan2 & greaterElevThres) & allBigThaThre].index)

grid_sample_WS = grids_join_WS.groupby('index', dropna=True)['numNoData'].idxmin()
grid_sample_WS = grid_sample_WS.dropna()
grid_sample_WS = grid_sample_WS.astype(int)
grid_sample_WS = grid_sample_WS.to_list()
list_WS = grids_join_WS.loc[grid_sample_WS][['WS_id']]
list_WS = list_WS.reset_index(drop=True)
print(list_WS)

# Save in a file

# Remove comment for saving
# list_WS.to_csv(f"D:\{base_map}_cellSize_{cell_size}.csv", index=False)

#%% Representación

fig, ax = mpl.pyplot.subplots(dpi=300, figsize=(10, 10))

ax.set_xlabel('$Longitude(^{\circ})$')
ax.set_ylabel('$Latitude(^{\circ})$')
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_aspect('equal')

grid_gdf.to_crs(epsg=epsg).plot(column='num_WS', ax=ax)

if ((base_map == 'World') | (base_map=='Northwestern')) | (base_map=='Spain'):
    filtered_map_countries_fondo = worldmap_countries
    if (maxx-minx)<20:
        ws_in_World.to_crs(epsg=epsg).plot(ax=ax, color='k', marker='+') # Too much WS. Grid representation fails for large scales
else:
    filtered_map_countries_fondo = map_Texas
    ws_in_Texas.to_crs(epsg=epsg).plot(ax=ax, color='k', marker='+')

filtered_map_countries_fondo.to_crs(epsg=epsg).plot(ax=ax, color='none', edgecolor='black')

mpl.pyplot.show()

#%% Dataframe with hourly data



