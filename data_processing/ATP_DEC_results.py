import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from calculator_core import *

import pandas as pd
from scipy.stats import zscore
from calculator_core.ATP_DEC import *

routes = [
    "BKK-CPH",
    "CPH-BKK",
    "FRA-ICN",
    "HEL-NRT",
    "ICN-FRA",
    "ICN-LHR",
    "LHR-ICN",
    "LHR-PVG",
    "NRT-HEL",
    "PVG-LHR",
    "SIN-ZRH",
    "ZRH-SIN"
]

# ATP-DEC inputs
seat_class = "First"
certification = "Legend"
aircraft_code = "B788"
aircraft_age = 5
plf = 0.835
cargo = 0
seat_data = pd.DataFrame(data={
    "First": [8, 80.0, 22.0],
    "Business": [48, 72.0, 20.0],
    "Premium Economy": [35, 38.0, 18.5],
    "Economy": [146, 31.0, 17.5]
}, index=["Number", "Pitch (cm)", "Width (cm)"])

# compute rolling averages excluding the current flight and filtering by airline
def compute_rolling_averages_with_airline(row, column_name, df, window='7D'):

    # check airline code valid
    if pd.isna(row['airline_code']) or len(row['airline_code']) < 2:
        return None

    # airline prefix is first two characters of the airline code
    airline_prefix = row['airline_code'][:2]
    
    # filter out current flight and include rows with the same airline prefix
    mask = (df.index < row.name) & (
        (df['date'] < row['date']) | 
        ((df['date'] == row['date']) & (df.index < row.name))
    ) & (df['airline_code'].str[:2] == airline_prefix)
    
    relevant_data = df.loc[mask]
    
    # compute rolling average
    return relevant_data.set_index('date')[column_name].rolling(window, closed='both').mean().iloc[-1] if not relevant_data.empty else None

########### For loop to run

for year in [2019, 2023]:
    for route in routes:
        path = fr"ATP-DEC-Nature-Communications-Earth-and-Environment\results\{year}\{route}\results_CF_filtered_7D.csv"
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['z_score_dist'] = zscore(df['actual_distance'])
        # df['z_score_zeta_2'] = zscore(df['zeta_2'])
        threshold = 1
        sorted_df = df.sort_values(by='date')
        filtered_df = sorted_df[sorted_df['z_score_dist'].abs() < threshold]
        # filtered_df = sorted_df[sorted_df['z_score'].abs() < threshold]
        df = filtered_df
        df = df.copy()

        for wind in ['30D', '7D']:

            df[f'rolling_geta_{wind}'] = df.apply(lambda row: compute_rolling_averages_with_airline(row, 'geta', df, window=wind), axis=1)
            df[f'rolling_zeta_1_{wind}'] = df.apply(lambda row: compute_rolling_averages_with_airline(row, 'zeta_1', df, window=wind), axis=1)
            df[f'rolling_zeta_2_{wind}'] = df.apply(lambda row: compute_rolling_averages_with_airline(row, 'zeta_2', df, window=wind), axis=1)
            df = df.copy()

            origin = route[:3]
            dest = route[-3:]
            pre_footprint = CarbonEmissionsCalc(origin, dest, seat_class, certification, aircraft_code, aircraft_age, seat_data, plf, cargo, 1, 1, 1, 1).epic(HAF=True)

            print(f"Starting CFC for {route} {year} {wind}")
            for index, row in df.iterrows():
                post_footprint = CarbonEmissionsCalc(row['origin'], row['destination'], seat_class, certification, aircraft_code, aircraft_age, seat_data, plf, cargo, row[f'rolling_geta_{wind}'], 1, row[f'rolling_zeta_1_{wind}'], row[f'rolling_zeta_2_{wind}']).epic(HAF=True)
                actual_footprint = CarbonEmissionsCalc(row['origin'], row['destination'], seat_class, certification, aircraft_code, aircraft_age, seat_data, plf, cargo, row['geta'], 1, row['zeta_1'], row['zeta_2']).epic(HAF=True)
                
                if wind == "30D":
                    df.loc[index, f'pre_footprint_first'] = pre_footprint[0]
                    df.loc[index, f'pre_footprint_business'] = pre_footprint[1]
                    df.loc[index, f'pre_footprint_premium_economy'] = pre_footprint[2]
                    df.loc[index, f'pre_footprint_economy'] = pre_footprint[3]
                
                df.loc[index, f'post_footprint_first_{wind}'] = post_footprint[0]
                df.loc[index, f'post_footprint_business_{wind}'] = post_footprint[1]
                df.loc[index, f'post_footprint_premium_economy_{wind}'] = post_footprint[2]
                df.loc[index, f'post_footprint_economy_{wind}'] = post_footprint[3]
                
                if wind == "7D":
                    df.loc[index, f'actual_footprint_first'] = actual_footprint[0]
                    df.loc[index, f'actual_footprint_business'] = actual_footprint[1]
                    df.loc[index, f'actual_footprint_premium_economy'] = actual_footprint[2]
                    df.loc[index, f'actual_footprint_economy'] = actual_footprint[3]

            print(f'Completed and saved CFC results for {route} {year} {wind}') 

        df.to_csv(rf"results\{year}\{route}\results_CF.csv")
        print(f'**** Completed and saved CFC results for {route} {year} ****')