Merge scripts

Purpose
- Combine Bluebikes trip CSVs for 2019 and 2020 into a single Excel file for analysis.

Files
- `merge_trips.py`: reads `Data/bluebikes_tripdata_2019.csv` and `Data/bluebikes_tripdata_2020.csv`, harmonizes columns, and writes `Data/bluebikes_trips_2019_2020.xlsx`.

Requirements
- Python 3.8+
- pandas
- openpyxl

Quick run (PowerShell):

```powershell
# from repo root
python -m pip install --user pandas openpyxl
python scripts\merge_trips.py
```

Notes / assumptions
- The script normalizes `postal code` -> `postal_code` (present in 2020 but not 2019).
- A `source_year` column is added (prefers existing `year` column if present).
- If other columns differ, the script keeps the union and fills missing values with blank/NA.
