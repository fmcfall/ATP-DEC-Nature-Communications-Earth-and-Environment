# ATP-DEC: Air Travel Passenger Dynamic Emissions Calculator

This repository contains the core code used in McFall et al. (2025), Nature Communications Earth & Environment. 

## Contents
- `calculator_core/`: Core ATP-DEC methodology scripts.
- `data_processing/`: Example workflow for preparing flight data (**synthetic example dataset only**; proprietary raw data used in the study cannot be shared).
- `figures/`: Scripts to reproduce figures from the article using calculator outputs.
- `results/`: Example outputs generated with the synthetic dataset (not the proprietary raw data).

## Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run example calculator:
   ```bash
   python calculator_core/ATP_DEC.py
3. Use `data_processing/` to prepare flight data and process mas flight calculations.
4. Example results found in `results/`.
5. Use notebooks in `figures/` to recreate figures shown in the article.

## Licence
ATP-DEC: Air Travel Passenger Dynamic Emissions Calculator
Copyright (C) 2025 Therme Group UK
SPDX-License-Identifier: AGPL-3.0-only

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
The Free Software Foundation, either version 3 of the License only.
