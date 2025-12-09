import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def create_cyclical_features(df, col, max_val, prefix=None):
    """Create sine and cosine encodings for cyclical features"""
    name = prefix if prefix else col
    df[f'{name}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{name}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def engineer_temporal_features(df):
    """Extract and encode temporal features matching specification"""
    # Extract basic temporal components
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Cyclical encodings
    df = create_cyclical_features(df, 'hour_of_day', 24)
    df = create_cyclical_features(df, 'day_of_week', 7)
    df = create_cyclical_features(df, 'month', 12)

    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Peak hour indicators (morning: 7-9, evening: 16-19)
    df['is_morning_peak'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 9)).astype(int)
    df['is_evening_peak'] = ((df['hour_of_day'] >= 16) & (df['hour_of_day'] <= 19)).astype(int)
    df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)

    # Holiday flag (will be added later with actual dates)

    return df

def create_station_clusters(station_coords, n_clusters=10):
    """Create neighborhood clusters using K-means on station coordinates"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(station_coords)
    return clusters

def engineer_spatial_features(df, station_info):
    """Encode spatial features matching specification"""
    # Merge station metadata
    df = df.merge(station_info[['station_id', 'station_latitude', 'station_longitude',
                                 'neighborhood_cluster_id', 'station_capacity', 'station_type']],
                  on='station_id', how='left')

    return df

def calculate_historical_demand(df):
    """Create lag and rolling features for demand"""
    # Sort by station and time
    df = df.sort_values(['station_id', 'timestamp'])

    # Lag features
    df['demand_t_minus_1'] = df.groupby('station_id')['demand'].shift(1)
    df['demand_t_minus_1'] = df['demand_t_minus_1'].fillna(0)
    df['demand_t_minus_24'] = df.groupby('station_id')['demand'].shift(24)
    df['demand_t_minus_24'] = df['demand_t_minus_24'].fillna(0)
    df['demand_t_minus_168'] = df.groupby('station_id')['demand'].shift(168)
    df['demand_t_minus_168'] = df['demand_t_minus_168'].fillna(0)

    # Same hour previous week
    df['same_hour_previous_week'] = df.groupby('station_id')['demand'].shift(168)

    # Rolling statistics (7 days = 168 hours)
    df['rolling_mean_7d'] = df.groupby('station_id')['demand'].transform(
        lambda x: x.shift(1).rolling(window=168, min_periods=24).mean()
    )
    df['rolling_std_7d'] = df.groupby('station_id')['demand'].transform(
        lambda x: x.shift(1).rolling(window=168, min_periods=24).std()
    )

    # Month-to-date average
    df['month_year'] = df['timestamp'].dt.to_period('M')
    df['month_to_date_average'] = df.groupby(['station_id', 'month_year'])['demand'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # Day of week average over last 4 weeks
    df['day_of_week_average_4w'] = df.groupby(['station_id', 'day_of_week'])['demand'].transform(
        lambda x: x.shift(1).rolling(window=4*24, min_periods=24).mean()
    )

    # Trend coefficient (linear regression slope over last 7 days)
    # Using a simpler, more stable method to avoid MKL issues
    def calculate_trend(series):
        try:
            if len(series) < 2:
                return 0
            # Remove NaN values
            clean_series = series.dropna()
            if len(clean_series) < 2:
                return 0
            # Check for constant series
            if np.std(clean_series) < 1e-10:
                return 0

            # Manual calculation of slope (more stable than polyfit)
            x = np.arange(len(clean_series))
            y = clean_series.values
            n = len(x)

            # Calculate slope using simple linear regression formula
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator < 1e-10:
                return 0

            slope = numerator / denominator
            return slope
        except (ValueError, RuntimeError):
            return 0

    df['trend_coefficient_7d'] = df.groupby('station_id')['demand'].transform(
        lambda x: x.shift(1).rolling(window=168, min_periods=24).apply(calculate_trend, raw=False)
    )

    return df

def calculate_weather_features(weather_df):
    """Engineer weather features from raw weather data"""
    # Calculate feels-like temperature (simplified wind chill/heat index)
    def feels_like(temp, wind_speed, humidity):
        # Simplified formula
        if temp < 10:  # Wind chill for cold
            return 13.12 + 0.6215*temp - 11.37*(wind_speed**0.16) + 0.3965*temp*(wind_speed**0.16)
        elif temp > 27:  # Heat index for warm
            return temp + 0.5555 * ((humidity/100) * 6.112 * np.exp(17.67*temp/(temp+243.5)) - 10)
        else:
            return temp

    weather_df['feels_like_temperature'] = weather_df.apply(
        lambda row: feels_like(row['temp'], row['wind_speed'], row['humidity']), axis=1
    )

    # Rename columns
    weather_df = weather_df.rename(columns={
        'temp': 'temperature',
        'precip': 'precipitation_mm',
        'wind_speed': 'wind_speed_mph'
    })

    # Weather category based on conditions
    def categorize_weather(row):
        if row['precipitation_mm'] > 5:
            return 'heavy_rain'
        elif row['precipitation_mm'] > 0.5:
            return 'rain'
        elif row['wind_speed_mph'] > 20:
            return 'windy'
        elif row['temperature'] < 0:
            return 'freezing'
        elif row['temperature'] > 30:
            return 'hot'
        else:
            return 'clear'

    weather_df['weather_category'] = weather_df.apply(categorize_weather, axis=1)

    # Weather severity score (0-10 scale, higher = worse conditions)
    def severity_score(row):
        score = 0
        # Temperature extremes
        if row['temperature'] < -5:
            score += 3
        elif row['temperature'] < 5:
            score += 2
        elif row['temperature'] > 35:
            score += 3
        elif row['temperature'] > 30:
            score += 2

        # Precipitation
        if row['precipitation_mm'] > 10:
            score += 4
        elif row['precipitation_mm'] > 2:
            score += 2
        elif row['precipitation_mm'] > 0:
            score += 1

        # Wind
        if row['wind_speed_mph'] > 25:
            score += 3
        elif row['wind_speed_mph'] > 15:
            score += 1

        return min(score, 10)

    weather_df['weather_severity_score'] = weather_df.apply(severity_score, axis=1)

    return weather_df

def calculate_user_demographics(trips_df):
    """Aggregate user demographics at station-hour level"""
    # Subscriber ratio
    trips_df['is_subscriber'] = (trips_df['usertype'] == 'Subscriber').astype(int)

    # Calculate age with safeguards
    trips_df['age'] = trips_df['year'] - trips_df['birth year']
    trips_df['age'] = trips_df['age'].clip(lower=16, upper=80)

    # Age bracket (1: <25, 2: 25-40, 3: 40-60, 4: 60+)
    trips_df['age_bracket'] = pd.cut(trips_df['age'],
                                      bins=[0, 25, 40, 60, 100],
                                      labels=[1, 2, 3, 4])
    trips_df['age_bracket'] = trips_df['age_bracket'].astype(float)

    # Return trip probability (same start and end station)
    trips_df['is_return_trip'] = (
        trips_df['start station id'] == trips_df['end station id']
    ).astype(int)

    return trips_df

def get_station_metadata(trips_df):
    """Extract station-level metadata from trips data"""
    # Get unique stations with coordinates
    stations = trips_df.groupby('start station id').agg({
        'start station name': 'first',
        'start station latitude': 'first',
        'start station longitude': 'first'
    }).reset_index()

    stations.columns = ['station_id', 'station_name', 'station_latitude', 'station_longitude']

    # Estimate station capacity based on trip volume
    trip_counts = trips_df.groupby('start station id').size()
    stations['station_capacity'] = trip_counts.values
    # Normalize to reasonable capacity range (10-50 bikes)
    stations['station_capacity'] = (
        10 + 40 * (stations['station_capacity'] - stations['station_capacity'].min()) /
        (stations['station_capacity'].max() - stations['station_capacity'].min())
    ).round().astype(int)

    # Create neighborhood clusters
    coords = stations[['station_latitude', 'station_longitude']].values
    stations['neighborhood_cluster_id'] = create_station_clusters(coords, n_clusters=15)

    # Station type based on location characteristics
    # Simple heuristic: downtown vs suburban based on distance from city center
    boston_center_lat, boston_center_lon = 42.3601, -71.0589
    stations['dist_from_center'] = np.sqrt(
        (stations['station_latitude'] - boston_center_lat)**2 +
        (stations['station_longitude'] - boston_center_lon)**2
    )
    stations['station_type'] = pd.cut(stations['dist_from_center'],
                                      bins=[0, 0.02, 0.05, 1],
                                      labels=['downtown', 'urban', 'suburban'])

    return stations

def process_bluebikes_data(trips_path, weather_path, output_path='bluebikes_ml_ready.csv'):
    """
    Main pipeline to create ML-ready dataset matching exact specification
    """

    print("=" * 60)
    print("BlueBikes Feature Engineering Pipeline")
    print("=" * 60)

    print("\n[1/8] Loading raw data...")
    trips_df = pd.read_csv(trips_path)
    trips_df['starttime'] = pd.to_datetime(trips_df['starttime'])
    trips_df['stoptime'] = pd.to_datetime(trips_df['stoptime'])
    print(f"   ✓ Loaded {len(trips_df):,} trip records")

    weather_df = pd.read_csv(weather_path)
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    print(f"   ✓ Loaded {len(weather_df):,} weather records")

    print("\n[2/8] Extracting station metadata...")
    station_info = get_station_metadata(trips_df)
    print(f"   ✓ Processed {len(station_info)} stations")
    print(f"   ✓ Created {station_info['neighborhood_cluster_id'].nunique()} neighborhood clusters")

    print("\n[3/8] Engineering user demographics...")
    trips_df = calculate_user_demographics(trips_df)

    # Round to hourly timestamp (using 'h' instead of deprecated 'H')
    trips_df['timestamp'] = trips_df['starttime'].dt.floor('h')

    print("\n[4/8] Aggregating to station × hour level...")
    # Aggregate trips
    agg_dict = {
        'tripduration': 'count',  # This is demand
        'is_subscriber': 'mean',
        'tripduration': ['count', 'mean'],
        'is_return_trip': 'mean',
        'age_bracket': 'mean'
    }

    station_hour = trips_df.groupby(['start station id', 'timestamp']).agg({
        'tripduration': ['count', 'mean'],
        'is_subscriber': 'mean',
        'is_return_trip': 'mean',
        'age_bracket': 'mean'
    }).reset_index()

    # Flatten column names
    station_hour.columns = ['station_id', 'timestamp', 'demand',
                            'average_trip_duration', 'subscriber_ratio',
                            'return_trip_probability', 'average_age_bracket']

    print(f"   ✓ Created {len(station_hour):,} station-hour records")

    print("\n[5/8] Engineering temporal features...")
    station_hour = engineer_temporal_features(station_hour)

    # Add holidays for 2020
    us_holidays_2020 = pd.to_datetime([
        '2020-01-01', '2020-01-20', '2020-02-17', '2020-05-25',
        '2020-07-04', '2020-09-07', '2020-10-12', '2020-11-11',
        '2020-11-26', '2020-12-25'
    ])
    station_hour['is_holiday'] = station_hour['timestamp'].dt.date.isin(
        us_holidays_2020.date
    ).astype(int)

    # Special events flag (simplified: major sporting events, marathons)
    special_events_2020 = pd.to_datetime([
        '2020-02-02',  # Super Bowl
        '2020-03-17',  # St. Patrick's Day
        '2020-07-04',  # Independence Day
    ])
    station_hour['special_event_flag'] = station_hour['timestamp'].dt.date.isin(
        special_events_2020.date
    ).astype(int)

    print(f"   ✓ Added cyclical encodings and flags")

    print("\n[6/8] Engineering spatial features...")
    station_hour = engineer_spatial_features(station_hour, station_info)
    print(f"   ✓ Added station metadata and clusters")

    print("\n[7/8] Calculating historical demand features...")
    station_hour = calculate_historical_demand(station_hour)
    print(f"   ✓ Added lag features and rolling statistics")

    print("\n[8/8] Merging weather data...")
    weather_df = calculate_weather_features(weather_df)
    station_hour = pd.merge(station_hour, weather_df,
                            left_on='timestamp', right_on='datetime', how='left')

    # Forward fill missing weather
    weather_cols = ['temperature', 'feels_like_temperature', 'precipitation_mm',
                    'wind_speed_mph', 'humidity', 'pressure',
                    'weather_severity_score']
    station_hour[weather_cols] = station_hour[weather_cols].ffill()

    # One-hot encode weather category
    weather_dummies = pd.get_dummies(station_hour['weather_category'],
                                     prefix='weather')
    station_hour = pd.concat([station_hour, weather_dummies], axis=1)

    print(f"   ✓ Merged weather features")

    # Fill remaining NaN in historical features (early observations)
    historical_cols = [col for col in station_hour.columns
                      if any(x in col for x in ['lag', 'rolling', 'average', 'trend', 'same_hour'])]
    station_hour[historical_cols] = station_hour[historical_cols].fillna(0)

    # Select final columns in specified order
    final_columns = [
        # Primary keys
        'station_id', 'timestamp',
        # Temporal
        'hour_of_day_sin', 'hour_of_day_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'is_weekend', 'is_peak_hour', 'is_holiday', 'special_event_flag',
        # Spatial
        'station_latitude', 'station_longitude',
        'station_capacity', 'neighborhood_cluster_id', 'station_type',
        # Historical demand
        'demand_t_minus_1', 'demand_t_minus_24', 'demand_t_minus_168',
        'rolling_mean_7d', 'rolling_std_7d',
        'same_hour_previous_week', 'month_to_date_average',
        'day_of_week_average_4w', 'trend_coefficient_7d',
        # Weather
        'temperature', 'feels_like_temperature',
        'precipitation_mm', 'wind_speed_mph',
        'weather_severity_score',
        # User/demographic
        'subscriber_ratio', 'average_trip_duration',
        'return_trip_probability', 'average_age_bracket',
        # Target
        'demand'
    ]

    # Add weather category dummies
    weather_dummy_cols = [col for col in station_hour.columns if col.startswith('weather_')]
    final_columns.extend(weather_dummy_cols)

    # Filter to final columns
    final_df = station_hour[final_columns].copy()

    # Sort by timestamp and station
    final_df = final_df.sort_values(['timestamp', 'station_id']).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n-------Dataset Summary:-----------")
    print(f"   • Total records: {len(final_df):,}")
    print(f"   • Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
    print(f"   • Number of stations: {final_df['station_id'].nunique()}")
    print(f"   • Total features: {len(final_df.columns) - 3}  (excluding keys + target)")

    print(f"\n----------Feature Breakdown:-----------------")
    temporal_features = [c for c in final_df.columns if any(x in c for x in
                        ['hour', 'day', 'month', 'weekend', 'peak', 'holiday', 'event'])]
    print(f"   • Temporal: {len(temporal_features)}")

    spatial_features = [c for c in final_df.columns if any(x in c for x in
                       ['latitude', 'longitude', 'capacity', 'cluster', 'type'])]
    print(f"   • Spatial: {len(spatial_features)}")

    historical_features = [c for c in final_df.columns if any(x in c for x in
                          ['minus', 'rolling', 'same_hour', 'average', 'trend'])]
    print(f"   • Historical: {len(historical_features)}")

    weather_features = [c for c in final_df.columns if any(x in c for x in
                       ['temperature', 'precipitation', 'wind', 'weather', 'humidity', 'pressure'])]
    print(f"   • Weather: {len(weather_features)}")

    demographic_features = [c for c in final_df.columns if any(x in c for x in
                           ['subscriber', 'trip_duration', 'return_trip', 'age_bracket'])]
    print(f"   • User/Demographic: {len(demographic_features)}")

    print(f"\n!!!!Saving to: {output_path}!!!!!!")
    final_df.to_csv(output_path, index=False)
    print(f"   ✓ Saved successfully!")

    print(f"\n-----------Sample records:----------------")
    print(final_df.head(3).to_string())

    return final_df

# Usage
if __name__ == "__main__":
    df = process_bluebikes_data(
        trips_path='Data/bluebikes_tripdata_2020.csv',
        weather_path='Data/boston_weather_2020.csv',
        output_path='Data/bluebikes_ml_ready.csv'
    )