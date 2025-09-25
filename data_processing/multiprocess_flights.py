import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from calculator_core import *

import pandas as pd
import os
import numpy as np
from zipfile import ZipFile
from multiprocessing import Pool, Process, Manager
from calculator_core.ATP_DEC import *

def calculate_actual_distance_interp(data):
    
    # check file or Dataframe
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    # ensure numeric data
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
    df['snapshot_id'] = pd.to_numeric(df['snapshot_id'], errors='coerce')

    # drop NaNs
    df.dropna(subset=['latitude', 'longitude', 'altitude', 'snapshot_id'], inplace=True)

    # unwrap longitudes
    df['longitude_unwrapped'] = np.unwrap(np.radians(df['longitude'])) * (180 / np.pi)

    # interpolate over time
    time_stamps = df['snapshot_id']
    interpolated_times = np.arange(time_stamps.iloc[0], time_stamps.iloc[-1])
    altitude_interp = np.interp(interpolated_times, time_stamps, df['altitude'])
    latitude_interp = np.interp(interpolated_times, time_stamps, df['latitude'])
    longitude_interp = np.interp(interpolated_times, time_stamps, df['longitude_unwrapped'])

    interp_df = pd.DataFrame({
        "snapshot_id": interpolated_times,
        "latitude": latitude_interp,
        "longitude": longitude_interp,
        "altitude": altitude_interp
    })

    # GCD
    positions = interp_df[['latitude', 'longitude']].values
    total_distance = 0.0
    cumulative_distances = [0]

    for i in range(1, len(positions)):
        lat1, lon1 = positions[i - 1]
        lat2, lon2 = positions[i]
        segment_distance = GCD((lat1, lon1), (lat2, lon2))[0]  # Assuming GCD returns (distance, _, _)
        total_distance += segment_distance
        cumulative_distances.append(total_distance)

    interp_df['cumulative_dist'] = cumulative_distances

    return total_distance, interp_df

def interpolate_latitude(df):

    interpolated_latitudes = []

    for i in range(len(df) - 1):

        current_latitude = df.iloc[i]['latitude']
        next_latitude = df.iloc[i + 1]['latitude']
        current_time = df.iloc[i]['snapshot_id']
        next_time = df.iloc[i + 1]['snapshot_id']

        time_diff = next_time - current_time

        # if time diff is greater than 1, interpolate linearly between latitudes
        if time_diff > 1:
            latitudes = np.linspace(current_latitude, next_latitude, time_diff + 1)[1:]
            interpolated_latitudes.extend(latitudes)
        else:
            # add current latitude if no gap
            interpolated_latitudes.append(current_latitude)

    # add last latitude point
    interpolated_latitudes.append(df.iloc[-1]['latitude'])

    return interpolated_latitudes

def calculate_mean_latitude(data):

    # calculate mean latitude

    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
    df['snapshot_id'] = pd.to_numeric(df['snapshot_id'], errors='coerce')

    df.dropna(subset=['latitude', 'longitude', 'altitude', 'snapshot_id'], inplace=True)

    # filter sustained altitude (non-zero altitude sustained for min threshold)
    min_sustained_altitude = 4
    df['sustained'] = (df['altitude'] > 0).ne((df['altitude'] > 0).shift()).cumsum()
    sustained_groups = df.groupby('sustained')

    valid_indices = []
    for _, group in sustained_groups:
        if (group['altitude'] > 0).all() and len(group) >= min_sustained_altitude:
            valid_indices.extend(group.index)

    if not valid_indices:
        return None

    # range of sustained altitude
    first_idx, last_idx = valid_indices[0], valid_indices[-1]
    start_idx, end_idx = max(0, first_idx - 1), min(len(df) - 1, last_idx + 1)
    filtered_df = df.iloc[start_idx:end_idx + 1].copy()

    # interpolated latitudes
    interpolated_latitudes = interpolate_latitude(filtered_df)

    return np.mean(interpolated_latitudes)

def merge_overnight_flight_data(flight_id, origin, destination, date, year_dir, processed_flights, cutoff_altitude=5000):

    # merge flight data for overnight flights spanning multiple days

    # check if flight has already been processed
    if flight_id in processed_flights:
        return None

    # current day file
    current_day_file = os.path.join(year_dir, f"{date}_positions.zip")
    if not os.path.exists(current_day_file):
        return None

    # check flight id exists
    with ZipFile(current_day_file, 'r') as zf:
        flight_file = f"{date}_{flight_id}.csv"
        if flight_file not in zf.namelist():
            return None

        with zf.open(flight_file) as file:
            current_data = pd.read_csv(file)

    # check if flight starts mid-air but allow if below cutoff altitude (took off on previous day)
    if current_data.iloc[0]["altitude"] > cutoff_altitude:
        return None

    # if the flight ends below the cutoff altitude, return it as-is
    if current_data.iloc[-1]["altitude"] <= cutoff_altitude:
        processed_flights.append(f"{flight_id}-{date}")
        return current_data

    # if the flight ends above the cutoff altitude, merge with the next day's data
    next_day = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    next_day_file = os.path.join(year_dir, f"{next_day}_positions.zip")
    if not os.path.exists(next_day_file):
        processed_flights.append(f"{flight_id}-{date}")
        return current_data

    with ZipFile(next_day_file, 'r') as zf:
        next_flight_file = f"{next_day}_{flight_id}.csv"
        if next_flight_file not in zf.namelist():
            processed_flights.append(f"{flight_id}-{date}")
            return current_data

        with zf.open(next_flight_file) as file:
            next_day_data = pd.read_csv(file)

    # merge current and next day's data
    merged_data = pd.concat([current_data, next_day_data], ignore_index=True)
    merged_data = merged_data.sort_values(by="snapshot_id").reset_index(drop=True)

    # if the merged data still ends above the cutoff altitude, accept it as complete
    if merged_data.iloc[-1]["altitude"] > cutoff_altitude:
        pass

    # to avoid duplication, always mark the flight as processed
    processed_flights.append(f"{flight_id}-{date}")
    return merged_data

def process_flight_task(date, task_id, flights_batch, year_dir, year, routes, base_output_dir, cutoff_altitude, checkpoint_queue):

    # process batch of flights

    try:
        results = []
        for _, row in flights_batch.iterrows():
            flight_id = row["flight_id"]
            route = f"{row['schd_from']}-{row['schd_to']}"

            # skip unwanted route (anoms)
            if route not in routes:
                continue

            # process flight data
            full_data = merge_overnight_flight_data(
                flight_id, row['schd_from'], row['schd_to'], date, year_dir, []
            )
            if full_data is None:
                continue
            
            actual_distance, interpolated_data = calculate_actual_distance_interp(full_data)
            mean_latitude = calculate_mean_latitude(interpolated_data)

            if actual_distance is None:
                continue
            if mean_latitude is None:
                continue 

            planned_gcd = routes[route][0]
            planned_lat = routes[route][1]
            geta = actual_distance / planned_gcd
            zeta_1 = actual_distance / planned_gcd
            zeta_2 = mean_latitude / planned_lat

            # append result
            results.append({
                "flight_id": flight_id,
                "date": date,
                "origin": row['schd_from'],
                "destination": row['schd_to'],
                "airline_code": row['flight'],
                "GCD": planned_gcd,
                "actual_distance": actual_distance,
                "mean_latitude": planned_lat,
                "actual_latitude": mean_latitude,
                "geta": geta,
                "zeta_1": zeta_1,
                "zeta_2": zeta_2,
            })

        # save results incrementally
        if results:
            for result in results:
                route_folder = os.path.join(base_output_dir, str(year), f"{result['origin']}-{result['destination']}")
                os.makedirs(route_folder, exist_ok=True)
                output_file = os.path.join(route_folder, "results.csv")
                pd.DataFrame([result]).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                
        # update checkpoint
        checkpoint_queue.put(task_id)

    except Exception as e:
        print(f"Error processing batch {task_id}: {e}")


def checkpoint_writer(checkpoint_queue, checkpoint_file):
    
    # write completed tasks to checkpoint file, taken from stackexchange

    with open(checkpoint_file, "a") as f:
        while True:
            task_id = checkpoint_queue.get()
            if task_id is None:
                break
            f.write(f"{task_id}\n")
            f.flush()


def process_flight_data_parallel(base_input_dir,base_output_dir, routes, checkpoint_file="checkpoint.log", cutoff_altitude=5000):
    
    # parallel flight processing - much faster

    tasks = []
    batch_number = 0
    completed_batches = set()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            completed_batches = set(f.read().splitlines())

    for year in [2019, 2023]:
        year_dir = os.path.join(base_input_dir, str(year))
        for summary_file in os.listdir(year_dir):
            if summary_file.endswith("_flights.csv"):
                date = summary_file.split("_")[0]
                summary_df = pd.read_csv(os.path.join(year_dir, summary_file))
                batch_size = 10  # batch size of 10 flights, too high causes memory issues
                for i in range(0, len(summary_df), batch_size):
                    flights_batch = summary_df.iloc[i:i + batch_size].copy()
                    task_id = f"{summary_file}_batch{batch_number}"
                    batch_number += 1
                    # skip completed batch (multiprocessing sometimes unordered)
                    if task_id in completed_batches:
                        continue
                    tasks.append((date, task_id, flights_batch, year_dir, year, routes, base_output_dir, cutoff_altitude))

    manager = Manager()
    checkpoint_queue = manager.Queue()
    writer_process = Process(target=checkpoint_writer, args=(checkpoint_queue, checkpoint_file))
    writer_process.start()

    try:
        # processes = 6 for ASUS laptop, higher causes memory issues
        with Pool(processes=6) as pool:
            pool.starmap(
                process_flight_task,
                [(date, task_id, flights_batch, year_dir, year, routes, base_output_dir, cutoff_altitude, checkpoint_queue)
                 for date, task_id, flights_batch, year_dir, year, routes, base_output_dir, cutoff_altitude in tasks]
            )
    finally:
        checkpoint_queue.put(None)
        writer_process.join()


if __name__ == "__main__":

    # replace with input/output/checkpoint paths
    base_input_dir = r"input.csv"
    base_output_dir = r"output.csv"
    checkpoint_file = r"checkpoint.log"

    # flight routes
    routes = {
        "LHR-ICN": [8863.736032710949, 44.4698505],
        "ICN-LHR": [8863.736032710949, 44.4698505],
        "LHR-PVG": [9244.034182063435, 41.307],
        "PVG-LHR": [9244.034182063435, 41.307],
        "HEL-NRT": [7831.487310198503, 48.0415325],
        "NRT-HEL": [7831.487310198503, 48.0415325],
        "SIN-ZRH": [10308.666419056577, 24.404123],
        "ZRH-SIN": [10308.666419056577, 24.404123],
        "BKK-CPH": [8639.395948887284, 34.6495003700258],
        "CPH-BKK": [8639.395948887284, 34.6495003700258],
        "FRA-ICN": [8546.290590793822, 43.749671],
        "ICN-FRA": [8546.290590793822, 43.749671]
    }

    process_flight_data_parallel(base_input_dir, base_output_dir, routes, checkpoint_file)