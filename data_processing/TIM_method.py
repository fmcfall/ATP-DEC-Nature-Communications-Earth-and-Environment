import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from calculator_core import *

import pandas as pd
import os
from calculator_core.ATP_DEC import *

def linear_interpolation(desired, distances, fuel_consumption_values):

    # linear interpolation to get fuel consumption for desired distance

    x = desired
    for i in range(len(distances)-1):
        if distances[i] <= x <= distances[i + 1]:
            x0, y0 = distances[i], fuel_consumption_values[i]
            x1, y1 = distances[i + 1], fuel_consumption_values[i + 1]
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        elif distances[i] == x:
            return fuel_consumption_values[i]
        
def TIM(origin, destination, seat_data, plf, actual_distance = 0):

    # aircraft: B788, consistent with ATP-DEC results
    distances = [125, 200, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
    data = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'calculator_core', 'seat_data.csv'), sheet_name="aircrafts")
    first_column_data = data["B788"]
    fuel_list = [x for x in first_column_data[10:].tolist() if str(x) != 'nan']
    distances = distances[0:len(fuel_list)]

    if actual_distance == 0:
        distance = GCD_airports(origin, destination)[0] / 1.852
    else:
        distance = actual_distance / 1.852

    fuel_burn = linear_interpolation(distance, distances, fuel_list) + first_column_data[1]
    ttl_emissions = fuel_burn * (3.1894 + 0.6465)

    seats = seat_data.iloc[0].tolist()
    ttl_area = (seats[0] * 5) + (seats[1] * 4) + (seats[2] * 1.5) + (seats[3] * 1)
    
    economy_emissions = ttl_emissions / ttl_area

    first_emissions = (economy_emissions * 5) / plf
    business_emissions = (economy_emissions * 4) / plf
    premium_economy_emissions = (economy_emissions * 1.5) / plf
    economy_emissions = economy_emissions / plf

    return [first_emissions, business_emissions, premium_economy_emissions, economy_emissions]

if __name__ == "__main__":
    
    # seat data from seat guru, same as used in ATP-DEC
    seat_data = pd.DataFrame(data={
        "First": [8, 80.0, 22.0],
        "Business": [48, 72.0, 20.0],
        "Premium Economy": [35, 38.0, 18.5],
        "Economy": [146, 31.0, 17.5]
    }, index=["Number", "Pitch (cm)", "Width (cm)"])

    print(TIM("LHR", "PVG", seat_data, 0.835))