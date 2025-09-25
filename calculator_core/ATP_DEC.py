''' COPYRIGHT AND LICENSE NOTICE '''
# ATP-DEC: Air Travel Passenger Dynamic Emissions Calculator
# Copyright (C) 2025 Therme Group UK
# SPDX-License-Identifier: AGPL-3.0-or-later

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

''' Library imports '''

from pathlib import Path
import pandas as pd
import numpy as np
import os
import math
from numpy.polynomial.polynomial import Polynomial

''' Load data '''

MODULE_DIR = Path(__file__).resolve().parent
def load_airport_data():
    file_path = MODULE_DIR / "airports.csv"
    df = pd.read_csv(file_path)
    return df

''' Fuel burn functions '''

def find_airport_coordinates(airport_code_1, airport_code_2, file_path):
   
    df = load_airport_data()

    airport_info = {}

    airport_1 = df[df['iata_code'] == airport_code_1]
    airport_2 = df[df['iata_code'] == airport_code_2]

    for airport in [airport_1, airport_2]:
        if not airport.empty:
            lat = airport['latitude_deg'].values[0]
            long = airport['longitude_deg'].values[0]
            airport_info[airport['iata_code'].values[0]] = (lat, long)
    
    return list(airport_info.values())

def GCD(origin, destination):
    
    lat1, lon1 = origin
    lat2, lon2 = destination

    # deg to rad
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)

    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6372.8

    return c * r, lat1, lat2

def GCD_airports(origin, destination):
    file_path = "calculator_core/airports.csv"
    airport_info = find_airport_coordinates(origin, destination, file_path)
    return GCD(airport_info[0], airport_info[1])

def polynomial_regression(desired, distances, fuel_consumption_values, degree):

    # polynomial regression function to get fuel consumption for desired distance
    if desired in distances:
        return fuel_consumption_values[distances.index(desired)]
    else:
        coefs = np.polyfit(distances, fuel_consumption_values, degree)
        poly = Polynomial(coefs[::-1])
        return poly(desired)

''' Age, airport, aircraft, IFS functions '''

def aircraft_age_factor(aircraft_width, aircraft_age):
    if aircraft_width == "narrow":
        if aircraft_age < 1:
            return 1
        elif aircraft_age < 2:
            return 1.02
        elif aircraft_age < 3:
            return 1.04
        elif aircraft_age <= 10:
            return 1.05
        else:
            return 1.06
    else:
        if aircraft_age < 1:
            return 1
        elif aircraft_age < 2:
            return 1.01
        elif aircraft_age < 3:
            return 1.015
        elif aircraft_age <= 10:
            return 1.018
        else:
            return 1.02
        
def airport_factor(distance):
    if distance >= 4000:
        return distance * 0.00149
    elif distance > 1500:
        return distance * 0.00432
    elif distance > 800:
        return distance * 0.0103
    else:
        return distance * 0.0198

def aircraft_factor(distance):
    if distance > 4000:
        return distance * 0.000153
    elif distance > 1500:
        return distance * 0.000253
    elif distance > 800:
        return distance * 0.00027
    else:
        return distance * 0.000287

def inflight_services_factor(distance):
    if distance > 1500:
        return [10, 7, 5, 3]
    else:
        return [7, 5, 2, 1]

''' A-DEC Method Class '''

class CarbonEmissionsCalc:

    def __init__(self, origin, destination, seat_class, certification, aircraft, aircraft_age, seat_data, plf, cargo, geta, veta, zeta_1, zeta_2):

        # client inputs
        self.origin = origin
        self.destination = destination
        self.seat_class = seat_class
        self.certifcation = certification

        # SP data
        self.aircraft = aircraft
        self.seat_data = seat_data
        self.plf = plf
        self.cargo = cargo

        # funcs & params
        self.gcd, self.lat1, self.lat2 = GCD_airports(origin, destination)
        self.capacity = sum(self.seat_data[x].iloc[0] for x in self.seat_data)
        self.occupancy = round(self.capacity * self.plf)

        # data
        self.distances = [125, 200, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
        self.data = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'calculator_core', 'seat_data.csv'), sheet_name="aircrafts")
        first_column_data = self.data[self.aircraft]
        self.fuel_list = [x for x in first_column_data[10:].tolist() if str(x) != 'nan']
        self.distances = self.distances[0:len(self.fuel_list)]
        self.LTO = first_column_data[1]
        self.aircraft_width = first_column_data[0]
        self.deterioration = aircraft_age_factor(self.aircraft_width, aircraft_age)
        self.passenger_mass = {
            "First": 80.9,
            "Business": 80.9,
            "Premium Economy": 75.4,
            "Economy": 75.4
        }
        self.carry_on_luggage_mass = {
            "First": 8.7,
            "Business": 8.7,
            "Premium Economy": 7.6,
            "Economy": 7.6
        }
        self.checked_luggage_mass = {
            "First": 18.7,
            "Business": 18.7,
            "Premium Economy": 17.3,
            "Economy": 17.3
        }

        # HAF
        self.geta = geta
        self.veta = veta
        self.zeta_1 = zeta_1
        self.zeta_2 = zeta_2

    def get_distance(self, adjustment="HAF"):
        if adjustment == "DAF":
            if self.gcd < 1000:
                distance = self.gcd * 1.143
            elif self.gcd < 4000:
                distance = self.gcd * 1.073
            else:
                distance = self.gcd * 1.048
        elif adjustment == "HAF_for_show":
            distance = self.gcd * self.geta
        elif adjustment == "HAF":
            distance = self.gcd
        return distance # in km
    
    def fuel_burn(self):
        self.distance = self.get_distance() / 1.852  # km to nautical miles conversion
        return polynomial_regression(self.distance, self.distances, self.fuel_list, 2) + self.LTO

    def well_to_tank(self, deterioration=True):
        fuel = self.fuel_burn()
        factor = 0.48 * self.deterioration if deterioration else 0.48
        return fuel * factor
    
    def tank_to_wake(self, deterioration=True):
        fuel = self.fuel_burn()
        factor = 3.16 * self.deterioration if deterioration else 3.16
        return fuel * factor
    
    def WTT_TTW(self):
        return self.well_to_tank() + self.tank_to_wake()
    
    def class_weights(self):
        ttl_area = sum(np.prod(self.seat_data[x]) for x in self.seat_data)
        areas = [self.seat_data[x].iloc[1] * self.seat_data[x].iloc[2] for x in self.seat_data]
        return [x / ttl_area if self.seat_data[y].iloc[0] > 0 else 0 for x, y in zip(areas, self.seat_data)]
    
    def carry_on_class_weights(self):
        total_carry_on = sum(self.seat_data.loc["Number", seat_class] * self.carry_on_luggage_mass[seat_class] for seat_class in self.seat_data.columns)
        return [self.carry_on_luggage_mass[seat_class] / total_carry_on for seat_class in self.seat_data.columns]

    def checked_class_weights(self):
        total_checked = sum(self.seat_data.loc["Number", seat_class] * self.checked_luggage_mass[seat_class] for seat_class in self.seat_data.columns)
        return [self.checked_luggage_mass[seat_class] / total_checked for seat_class in self.seat_data.columns]

    def total_passenger_mass(self):
        # from EASA "Review of Standard Passenger Weights"
        return self.plf * sum(self.seat_data.loc["Number", seat_class] * self.passenger_mass[seat_class] for seat_class in self.seat_data.columns)
    
    def total_carry_on_luggage_mass(self):
        return self.plf * sum(self.seat_data.loc["Number", seat_class] * self.carry_on_luggage_mass[seat_class] for seat_class in self.seat_data.columns)

    def total_checked_luggage_mass(self):
        return self.plf * sum(self.seat_data.loc["Number", seat_class] * self.checked_luggage_mass[seat_class] for seat_class in self.seat_data.columns)

    def total_flight_mass(self):
        return self.total_passenger_mass() + self.total_carry_on_luggage_mass() + self.total_checked_luggage_mass() + self.cargo

    def cargo_factor(self):
        CF = 1 - (self.cargo / self.total_flight_mass())
        return CF
    
    def passenger_factor(self):
        PF = 1 - (self.total_passenger_mass() / self.total_flight_mass())
        return PF

################################################### LUGGAGE EMISSIONS ###################################################

    def carry_on_luggage_factor(self):
        CLF = 1 - ((self.total_carry_on_luggage_mass()) / self.total_flight_mass())
        return CLF

    def checked_luggage_factor(self):
        CHLF = 1 - ((self.total_checked_luggage_mass()) / self.total_flight_mass())
        return CHLF 

    def carry_on_luggage_emissions(self):
        return  self.WTT_TTW() * (1- ((self.cargo + self.total_checked_luggage_mass() + self.total_passenger_mass()) / self.total_flight_mass()))
    
    def checked_luggage_emissions(self):
        return self.WTT_TTW() * (1- ((self.cargo + self.total_carry_on_luggage_mass() + self.total_passenger_mass()) / self.total_flight_mass()))

    def carry_on_just_co2(self):
        return self.fuel_burn() * 3.15 * self.deterioration * (1- ((self.cargo + self.total_checked_luggage_mass() + self.total_passenger_mass()) / self.total_flight_mass())) /self.plf

    def checked_just_co2(self):
        return self.fuel_burn() * 3.15 * self.deterioration * (1- ((self.cargo + self.total_carry_on_luggage_mass() + self.total_passenger_mass()) / self.total_flight_mass())) /self.plf

    def get_carry_on_NOx(self, HAF=False):
        if HAF:
            NOx, _, _ = self.non_kyoto_multipliers(HAF=True)
            carry_on_co2 = self.carry_on_just_co2() * self.zeta_1
        else:
            NOx, _, _ = self.non_kyoto_multipliers(HAF=False)
            carry_on_co2 = self.carry_on_just_co2()
        return [(x * carry_on_co2 * NOx) for x in self.carry_on_class_weights()]

    def get_carry_on_CiC(self, HAF=False):
        if HAF:
            _, CiC, _ = self.non_kyoto_multipliers(HAF=True)
            carry_on_co2 = self.carry_on_just_co2() * self.zeta_1
        else:
            _, CiC, _ = self.non_kyoto_multipliers(HAF=False)
            carry_on_co2 = self.carry_on_just_co2()
        return [(x * carry_on_co2 * CiC) for x in self.carry_on_class_weights()]

    def get_carry_on_H2O(self, HAF=False):
        if HAF:
            _, _, H2O = self.non_kyoto_multipliers(HAF=True)
            carry_on_co2 = self.carry_on_just_co2() * self.zeta_1
        else:
            _, _, H2O = self.non_kyoto_multipliers(HAF=False)
            carry_on_co2 = self.carry_on_just_co2()
        return [(x * carry_on_co2 * H2O) for x in self.carry_on_class_weights()]

    def get_checked_NOx(self, HAF=False):
        if HAF:
            NOx, _, _ = self.non_kyoto_multipliers(HAF=True)
            checked_co2 = self.checked_just_co2() * self.zeta_1
        else:
            NOx, _, _ = self.non_kyoto_multipliers(HAF=False)
            checked_co2 = self.checked_just_co2()
        return [(x * checked_co2 * NOx) for x in self.checked_class_weights()]

    def get_checked_CiC(self, HAF=False):
        if HAF:
            _, CiC, _ = self.non_kyoto_multipliers(HAF=True)
            checked_co2 = self.checked_just_co2() * self.zeta_1
        else:
            _, CiC, _ = self.non_kyoto_multipliers(HAF=False)
            checked_co2 = self.checked_just_co2()
        return [(x * checked_co2 * CiC) for x in self.checked_class_weights()]

    def get_checked_H2O(self, HAF=False):
        if HAF:
            _, _, H2O = self.non_kyoto_multipliers(HAF=True)
            checked_co2 = self.checked_just_co2() * self.zeta_1
        else:
            _, _, H2O = self.non_kyoto_multipliers(HAF=False)
            checked_co2 = self.checked_just_co2()
        return [(x * checked_co2 * H2O) for x in self.checked_class_weights()]
    
    def carry_on_ethic(self, HAF=False):
        if HAF:
            e = [((x * self.carry_on_luggage_emissions()) / self.plf) * self.geta for x in self.carry_on_class_weights()]
        else:
            e = [((x * self.carry_on_luggage_emissions()) / self.plf) for x in self.carry_on_class_weights()]
        return [g if w != 0 else 0 for g, w in zip(e, self.carry_on_class_weights())]

    def carry_on_epic(self, HAF=False):
        if HAF:
            e = [((x * self.carry_on_luggage_emissions()) / self.plf) * self.geta for x in self.carry_on_class_weights()]
        else:
            e = [((x * self.carry_on_luggage_emissions()) / self.plf) for x in self.carry_on_class_weights()]
        if HAF:
            e = [sum(x) for x in zip(e, self.get_carry_on_NOx(HAF=True))]
        else:
            e = [sum(x) for x in zip(e, self.get_carry_on_NOx(HAF=False))]
        return [g if w != 0 else 0 for g, w in zip(e, self.carry_on_class_weights())]

    def carry_on_legend(self, HAF=False):
        if HAF:
            e = [((x * self.carry_on_luggage_emissions()) / self.plf) * self.geta for x in self.carry_on_class_weights()]
        else:
            e = [((x * self.carry_on_luggage_emissions()) / self.plf) for x in self.carry_on_class_weights()]
        if HAF:
            e = [sum(x) for x in zip(e, self.get_carry_on_NOx(HAF=True), self.get_carry_on_H2O(HAF=True), self.get_carry_on_H2O(HAF=True))]
        else:
            e = [sum(x) for x in zip(e, self.get_carry_on_NOx(HAF=False), self.get_carry_on_H2O(HAF=False), self.get_carry_on_H2O(HAF=False))]
        return [g if w != 0 else 0 for g, w in zip(e, self.carry_on_class_weights())]

    def checked_ethic(self, HAF=False):
        if HAF:
            e = [((x * self.checked_luggage_emissions()) / self.plf) * self.geta for x in self.checked_class_weights()]
        else:
            e = [((x * self.checked_luggage_emissions()) / self.plf) for x in self.checked_class_weights()]
        return [g if w != 0 else 0 for g, w in zip(e, self.checked_class_weights())]

    def checked_epic(self, HAF=False):
        if HAF:
            e = [((x * self.checked_luggage_emissions()) / self.plf) * self.geta for x in self.checked_class_weights()]
        else:
            e = [((x * self.checked_luggage_emissions()) / self.plf) for x in self.checked_class_weights()]
        if HAF:
            e = [sum(x) for x in zip(e, self.get_checked_NOx(HAF=True))]
        else:
            e = [sum(x) for x in zip(e, self.get_checked_NOx(HAF=False))]
        return [g if w != 0 else 0 for g, w in zip(e, self.checked_class_weights())]

    def checked_legend(self, HAF=False):
        if HAF:
            e = [((x * self.checked_luggage_emissions()) / self.plf) * self.geta for x in self.checked_class_weights()]
        else:
            e = [((x * self.checked_luggage_emissions()) / self.plf) for x in self.checked_class_weights()]
        if HAF:
            e = [sum(x) for x in zip(e, self.get_checked_NOx(HAF=True), self.get_checked_H2O(HAF=True), self.get_checked_H2O(HAF=True))]
        else:
            e = [sum(x) for x in zip(e, self.get_checked_NOx(HAF=False), self.get_checked_H2O(HAF=False), self.get_checked_H2O(HAF=False))]
        return [g if w != 0 else 0 for g, w in zip(e, self.checked_class_weights())]
    
########################################################################################################################

    def total_passenger_emissions(self):
        return self.WTT_TTW() * (1- ((self.cargo + self.total_carry_on_luggage_mass() + self.total_checked_luggage_mass()) / self.total_flight_mass()))

    def WTT_per_person(self):
        return [(x * self.well_to_tank() * (1- ((self.cargo + self.total_carry_on_luggage_mass() + self.total_checked_luggage_mass()) / self.total_flight_mass()))/self.plf) for x in self.class_weights()]
    
    def TTW_per_person(self):
        return [(x * self.tank_to_wake() * (1- ((self.cargo + self.total_carry_on_luggage_mass() + self.total_checked_luggage_mass()) / self.total_flight_mass()))/self.plf) for x in self.class_weights()]

    def non_kyoto_multipliers(self, HAF=False):
        if HAF:
            mean_lat = (self.lat1 + self.lat2)/2 * self.zeta_2
            D = self.get_distance() * self.geta /1000
        else:
            mean_lat = (self.lat1 + self.lat2)/2
            D = self.get_distance() /1000
        
        # NOx = max(0, (2.3*np.arctan(3.1*D) - 2)*(1.6*10**(-4)*mean_lat**2 + -1.6*10**(-3)*mean_lat + 0.86))
        # CiC = max(0, 1.1*np.arctan(0.5*D)*(2.8*10**(-7)*mean_lat**4 + 1.9*10**(-6)*mean_lat**3 + -1.2*10**(-3)*mean_lat**2 + -7.7*10**(-4)*mean_lat + 1.7))
        # H2O = max(0, 0.2*np.arctan(D)*(-7.6*10**(-6)*mean_lat**3 + 8.2*10**(-4)*mean_lat**2 + 1.4*10**(-3)*mean_lat + 0.15))

        c_nox, d_nox, e_nox = 1.6 * (10**(-4)), -1.6 * (10**(-3)), 0.86
        a_cic, b_cic, c_cic, d_cic, e_cic = 2.8 * (10**(-7)), 1.9 * (10**(-6)), -1.2 * (10**(-3)), -7.7 * (10**(-4)), 1.7
        b_h2o, c_h2o, d_h2o, e_h2o = -7.6 * (10**(-6)), 8.2 * (10**(-4)), 1.4 * (10**(-3)), 0.15

        NOx = max(0, (2.3*np.arctan(3.1*D) - 2)*(c_nox*mean_lat**2 + d_nox*mean_lat + e_nox))
        CiC = max(0, 1.1*np.arctan(0.5*D)*(a_cic*mean_lat**4 + b_cic*mean_lat**3 + c_cic*mean_lat**2 + d_cic*mean_lat + e_cic))
        H2O = max(0, 0.2*np.arctan(D)*(b_h2o*mean_lat**3 + c_h2o*mean_lat**2 + d_h2o*mean_lat + e_h2o))
        return NOx, CiC, H2O

    def passenger_just_co2(self):
        return self.fuel_burn() * 3.15 * self.deterioration * (1- ((self.cargo + self.total_carry_on_luggage_mass() + self.total_checked_luggage_mass()) / self.total_flight_mass()))/self.plf

    def get_NOx(self, HAF=False):
        if HAF:
            NOx, _, _ = self.non_kyoto_multipliers(HAF=True)
            passenger_co2 = self.passenger_just_co2() * self.zeta_1
        else:
            NOx, _, _ = self.non_kyoto_multipliers(HAF=False)
            passenger_co2 = self.passenger_just_co2()
        return [(x * passenger_co2 * NOx) for x in self.class_weights()]

    def get_CiC(self, HAF=False):
        if HAF:
            _, CiC, _ = self.non_kyoto_multipliers(HAF=True)
            passenger_co2 = self.passenger_just_co2() * self.zeta_1
        else:
            _, CiC, _ = self.non_kyoto_multipliers(HAF=False)
            passenger_co2 = self.passenger_just_co2()
        return [(x * passenger_co2 * CiC) for x in self.class_weights()]

    def get_H2O(self, HAF=False):
        if HAF:
            _, _, H2O = self.non_kyoto_multipliers(HAF=True)
            passenger_co2 = self.passenger_just_co2() * self.zeta_1
        else:
            _, _, H2O = self.non_kyoto_multipliers(HAF=False)
            passenger_co2 = self.passenger_just_co2()
        return [(x * passenger_co2 * H2O) for x in self.class_weights()]

    def airport_emissions(self):
        airport = [airport_factor(self.get_distance()) for i in range(4)]
        return [g if w != 0 else 0 for g, w in zip(airport, self.class_weights())]
    
    def aircraft_emissions(self):
        aircraft = [aircraft_factor(self.get_distance()) for i in range(4)]
        return [g if w != 0 else 0 for g, w in zip(aircraft, self.class_weights())]

    def inflight_emissions(self):
        ifs = inflight_services_factor(self.get_distance())
        return [g if w != 0 else 0 for g, w in zip(ifs, self.class_weights())]
   
    def ethic(self, carry_on, checked, HAF=False):
        if HAF:
            e = [((x * self.total_passenger_emissions()) / self.plf) * self.geta for x in self.class_weights()]
        else:
            e = [((x * self.total_passenger_emissions()) / self.plf) for x in self.class_weights()]
        e = [sum(x) for x in zip(self.inflight_emissions(), e)]
        e = [sum(x) for x in zip((v * carry_on for v in self.carry_on_ethic()), e)]
        e = [sum(x) for x in zip((v * checked for v in self.checked_ethic()), e)]
        return [g if w != 0 else 0 for g, w in zip(e, self.class_weights())]

    def epic(self, carry_on, checked, HAF=False):
        if HAF:
            e = [((x * self.total_passenger_emissions()) / self.plf) * self.geta for x in self.class_weights()]
        else:
            e = [((x * self.total_passenger_emissions()) / self.plf) for x in self.class_weights()]
        e = [sum(x) for x in zip(self.inflight_emissions(), e)]
        e = [sum(x) for x in zip(self.airport_emissions(), e)]
        e = [sum(x) for x in zip(self.aircraft_emissions(), e)]
        e = [sum(x) for x in zip((v * carry_on for v in self.carry_on_epic()), e)]
        e = [sum(x) for x in zip((v * checked for v in self.checked_epic()), e)]
        if HAF:
            e = [sum(x) for x in zip(e, self.get_NOx(HAF=True))]
        else:
            e = [sum(x) for x in zip(e, self.get_NOx(HAF=False))]
        return [g if w != 0 else 0 for g, w in zip(e, self.class_weights())]

    def legend(self, carry_on, checked, HAF=False):
        if HAF:
            e = [((x * self.total_passenger_emissions()) / self.plf) * self.geta for x in self.class_weights()]
        else:
            e = [((x * self.total_passenger_emissions()) / self.plf) for x in self.class_weights()]
        e = [sum(x) for x in zip(self.inflight_emissions(), e)]
        e = [sum(x) for x in zip(self.airport_emissions(), e)]
        e = [sum(x) for x in zip(self.aircraft_emissions(), e)]
        e = [sum(x) for x in zip((v * carry_on for v in self.carry_on_legend()), e)]
        e = [sum(x) for x in zip((v * checked for v in self.checked_legend()), e)]
        if HAF:
            e = [sum(x) for x in zip(e, self.get_NOx(HAF=True), self.get_H2O(HAF=True), self.get_CiC(HAF=True))]
        else:
            e = [sum(x) for x in zip(e, self.get_NOx(HAF=False), self.get_H2O(HAF=False), self.get_CiC(HAF=False))]
        return [g if w != 0 else 0 for g, w in zip(e, self.class_weights())]

    def ethic_breakdown(self, carry_on, checked, HAF=False, dataframe=True):
        if HAF:
            breakdown = {
                "TTW": [x * self.geta for x in self.TTW_per_person()],
                "WTT": [x * self.geta for x in self.WTT_per_person()],
                "In-Flight Services": [x * self.veta for x in self.inflight_emissions()],
                "Carry On Luggage": [x * carry_on for x in self.carry_on_ethic()],
                "Checked Luggage": [x * checked for x in self.checked_ethic()],
                "Total": self.ethic(carry_on, checked)
                }
        else:
            breakdown = {
                "TTW": self.TTW_per_person(),
                "WTT": self.WTT_per_person(),
                "In-Flight Services": self.inflight_emissions(),
                "Carry On Luggage": [x * carry_on for x in self.carry_on_ethic()],
                "Checked Luggage": [x * checked for x in self.checked_ethic()],
                "Total": self.ethic(carry_on, checked)
                }
        if dataframe:
            breakdown = pd.DataFrame(breakdown, index=["First", "Business", "Premium Economy", "Economy"])
        return breakdown

    def epic_breakdown(self, carry_on, checked, HAF=False, dataframe=True):
        if HAF:
            breakdown = {
                "TTW": [x * self.geta for x in self.TTW_per_person()],
                "WTT": [x * self.geta for x in self.WTT_per_person()],
                "In-Flight Services": [x * self.veta for x in self.inflight_emissions()],
                "Carry On Luggage": [x * carry_on for x in self.carry_on_epic()],
                "Checked Luggage": [x * checked for x in self.checked_epic()],
                "Airport": self.airport_emissions(),
                "Aircraft": self.aircraft_emissions(),
                "NOx": self.get_NOx(HAF=True),
                "Total": self.epic(carry_on, checked, HAF=True)
                }
        else:
            breakdown = {
                "TTW": self.TTW_per_person(),
                "WTT": self.WTT_per_person(),
                "In-Flight Services": self.inflight_emissions(),
                "Carry On Luggage": [x * carry_on for x in self.carry_on_epic()],
                "Checked Luggage": [x * checked for x in self.checked_epic()],
                "Airport": self.airport_emissions(),
                "Aircraft": self.aircraft_emissions(),
                "NOx": self.get_NOx(HAF=False),
                "Total": self.epic(carry_on, checked, HAF=False)
                }
        if dataframe:
            breakdown = pd.DataFrame(breakdown, index=["First", "Business", "Premium Economy", "Economy"])
        return breakdown

    def legend_breakdown(self, carry_on, checked, luggage_cert_method, HAF=False, dataframe=True):
        if luggage_cert_method == "Legend":
            carry_on_method = self.carry_on_legend()
            checked_method = self.checked_legend()
        if luggage_cert_method == "Epic":
            carry_on_method = self.carry_on_epic()
            checked_method = self.checked_epic()
        if luggage_cert_method == "Ethic":
            carry_on_method = self.carry_on_ethic()
            checked_method = self.checked_ethic()
        if HAF:
            breakdown = {
                "TTW": [x * self.geta for x in self.TTW_per_person()],
                "WTT": [x * self.geta for x in self.WTT_per_person()],
                "In-Flight Services": [x * self.veta for x in self.inflight_emissions()],
                "Carry On Luggage": [x * carry_on for x in carry_on_method],
                "Checked Luggage": [x * checked for x in checked_method],
                "Airport": self.airport_emissions(),
                "Aircraft": self.aircraft_emissions(),
                "NOx": self.get_NOx(HAF=True),
                "H2O": self.get_H2O(HAF=True),
                "CiC": self.get_CiC(HAF=True),
                "Total": self.legend(carry_on, checked, HAF=True)
                }
        else:
            breakdown = {
                "TTW": self.TTW_per_person(),
                "WTT": self.WTT_per_person(),
                "In-Flight Services": self.inflight_emissions(),
                "Carry On Luggage": [x * carry_on for x in carry_on_method],
                "Checked Luggage": [x * checked for x in checked_method],
                "Airport": self.airport_emissions(),
                "Aircraft": self.aircraft_emissions(),
                "NOx": self.get_NOx(HAF=False),
                "H2O": self.get_H2O(HAF=False),
                "CiC": self.get_CiC(HAF=False),
                "Total": self.legend(carry_on, checked, HAF=False)
                }
        if dataframe:
            breakdown = pd.DataFrame(breakdown, index=["First", "Business", "Premium Economy", "Economy"])
        return breakdown
    
if __name__ == "__main__":

    origin = "LHR"
    destination = "JFK"
    seat_class = "First"
    certification = "Legend"
    aircraft_code = "B788"
    aircraft_age = 5
    plf = 0.835
    cargo = 10000
    seat_data = pd.DataFrame(data={
        "First": [8, 80.0, 22.0],
        "Business": [48, 72.0, 20.0],
        "Premium Economy": [35, 38.0, 18.5],
        "Economy": [146, 31.0, 17.5]
    }, index=["Number", "Pitch (cm)", "Width (cm)"])

    cfc = CarbonEmissionsCalc(origin, destination, seat_class, certification, aircraft_code, aircraft_age, seat_data, plf, cargo, 1, 1, 1, 1)
    print(cfc.legend_breakdown(1, 2, "Legend", HAF=True))
