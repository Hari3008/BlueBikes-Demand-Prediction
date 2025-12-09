import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# DATASET DESCRIPTION FOR PROJECT DOCUMENTATION
# ==========================================
print("="*80)
print("BIKE DEMAND FORECASTING PROJECT - DATASET DESCRIPTION")
print("="*80)

print("""
PROJECT GOAL:
Predict hourly or daily bike rental counts at each station using regression models
to optimize bike redistribution and ensure availability.

DATASET: Boston BlueBikes 2020 Trip Data
""")

# ==========================================
# 1. LOAD AND DESCRIBE DATASET
# ==========================================
print("\n" + "="*60)
print("1. DATASET OVERVIEW")
print("="*60)

# Load your data (adjust path)
FILE_PATH = 'D:\Course_stuff_M\Machine Learning\Project\Data\\bluebikes_tripdata_2020.csv'
SAMPLE_SIZE = 200000  # Use sample for EDA, set to None for full data

if SAMPLE_SIZE:
    print(f"Loading {SAMPLE_SIZE:,} samples for EDA...")
    df = pd.read_csv(FILE_PATH, nrows=SAMPLE_SIZE)
else:
    print("Loading full dataset...")
    df = pd.read_csv(FILE_PATH)

# Basic dataset statistics
total_samples = len(df)
print(f"\n📊 DATASET STATISTICS:")
print(f"   • Total Samples (trips): {total_samples:,}")
print(f"   • Time Period: {df['year'].iloc[0]}-{df['month'].min():02d} to {df['year'].iloc[0]}-{df['month'].max():02d}")
print(f"   • Number of Features: {df.shape[1]}")
print(f"   • Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Convert datetime
df['starttime'] = pd.to_datetime(df['starttime'])
df['stoptime'] = pd.to_datetime(df['stoptime'])

# Station analysis
unique_start_stations = df['start station id'].nunique()
unique_end_stations = df['end station id'].nunique()
all_unique_stations = pd.concat([df['start station id'], df['end station id']]).nunique()

print(f"\n🚴 STATION INFORMATION:")
print(f"   • Unique Start Stations: {unique_start_stations}")
print(f"   • Unique End Stations: {unique_end_stations}")
print(f"   • Total Unique Stations: {all_unique_stations}")

# User type distribution
print(f"\n👥 USER TYPES:")
user_dist = df['usertype'].value_counts()
for utype, count in user_dist.items():
    print(f"   • {utype}: {count:,} ({count/total_samples*100:.1f}%)")

# ==========================================
# 2. TARGET VARIABLE CREATION FOR DEMAND FORECASTING
# ==========================================
print("\n" + "="*60)
print("2. TARGET VARIABLE: DEMAND COUNTS")
print("="*60)

# Create hourly demand counts per station
print("\nCreating target variable: Hourly demand per station...")

# Extract hour and date
df['hour'] = df['starttime'].dt.hour
df['date'] = df['starttime'].dt.date

# Calculate hourly demand for each station
hourly_demand = df.groupby(['start station id', 'date', 'hour']).size().reset_index(name='demand_count')
print(f"\nHourly Demand Statistics:")
print(f"   • Total hourly station observations: {len(hourly_demand):,}")
print(f"   • Mean hourly demand per station: {hourly_demand['demand_count'].mean():.2f}")
print(f"   • Median hourly demand: {hourly_demand['demand_count'].median():.0f}")
print(f"   • Max hourly demand: {hourly_demand['demand_count'].max()}")
print(f"   • Stations with zero demand hours: {(hourly_demand['demand_count']==0).sum()}")

# Daily demand aggregation
daily_demand = df.groupby(['start station id', 'date']).size().reset_index(name='daily_demand')
print(f"\nDaily Demand Statistics:")
print(f"   • Mean daily demand per station: {daily_demand['daily_demand'].mean():.2f}")
print(f"   • Max daily demand: {daily_demand['daily_demand'].max()}")

# ==========================================
# 3. FEATURE AVAILABILITY ASSESSMENT
# ==========================================
print("\n" + "="*60)
print("3. FEATURE AVAILABILITY & EXTRACTION PLAN")
print("="*60)

print("\n✅ TEMPORAL FEATURES (Already Available/Extractable):")
print("   • Hour of day: Extracted from 'starttime'")
print("   • Day of week: Extractable (0=Monday, 6=Sunday)")
print("   • Month: Available in dataset")
print("   • Year: Available in dataset")
print("   • Weekend indicator: Derivable from day of week")
print("   • Time period (morning rush, midday, evening rush, night): Derivable")

df['day_of_week'] = df['starttime'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Time period classification
def get_time_period(hour):
    if 6 <= hour < 10:
        return 'morning_rush'
    elif 10 <= hour < 16:
        return 'midday'
    elif 16 <= hour < 20:
        return 'evening_rush'
    else:
        return 'night'

df['time_period'] = df['hour'].apply(get_time_period)

print("\n✅ SPATIAL FEATURES (Available):")
print("   • Station ID: Available for all trips")
print("   • Latitude: Available (start & end stations)")
print("   • Longitude: Available (start & end stations)")
print("   • Postal code: Available (can indicate neighborhood characteristics)")

# Check postal code coverage
postal_coverage = df['postal code'].notna().sum() / len(df) * 100
print(f"   • Postal code coverage: {postal_coverage:.1f}%")

print("\n⚠️ FEATURES REQUIRING EXTERNAL DATA:")
print("   • Weather data: Need to integrate from external source")
print("   • Holiday indicators: Need to add from calendar data")
print("   • Special events: Need external event calendar")

print("\n📈 LAG FEATURES (To be engineered):")
print("   • Previous hour demand")
print("   • Same hour yesterday demand")
print("   • Same hour last week demand")
print("   • Rolling averages (3-hour, 24-hour, 7-day)")

# ==========================================
# 4. DATA QUALITY ASSESSMENT
# ==========================================
print("\n" + "="*60)
print("4. DATA QUALITY & PREPROCESSING NEEDS")
print("="*60)

# Missing values
print("\n🔍 Missing Values Analysis:")
missing = df.isnull().sum()
for col in missing[missing > 0].index:
    pct = missing[col] / len(df) * 100
    print(f"   • {col}: {missing[col]:,} ({pct:.2f}%)")

if missing.sum() == 0:
    print("   ✅ No missing values in core features!")

# Data quality checks
print("\n🔍 Data Quality Checks:")

# Check for duplicate trips
duplicates = df.duplicated().sum()
print(f"   • Duplicate rows: {duplicates}")

# Check for invalid coordinates
invalid_coords = df[(df['start station latitude'] == 0) | 
                    (df['start station longitude'] == 0)].shape[0]
print(f"   • Invalid coordinates: {invalid_coords}")

# Check trip duration distribution
df['tripduration_min'] = df['tripduration'] / 60
print(f"\n   • Trip Duration Analysis:")
print(f"     - Mean: {df['tripduration_min'].mean():.1f} minutes")
print(f"     - Median: {df['tripduration_min'].median():.1f} minutes")
print(f"     - Trips < 1 min: {(df['tripduration'] < 60).sum():,}")
print(f"     - Trips > 24 hours: {(df['tripduration'] > 86400).sum():,}")

# ==========================================
# 5. DEMAND PATTERNS ANALYSIS
# ==========================================
print("\n" + "="*60)
print("5. DEMAND PATTERNS ANALYSIS")
print("="*60)

# Hourly pattern
hourly_pattern = df.groupby('hour').size()
print("\n📊 Hourly Demand Pattern:")
peak_hour = hourly_pattern.idxmax()
print(f"   • Peak hour: {peak_hour}:00 ({hourly_pattern.max():,} trips)")
print(f"   • Lowest hour: {hourly_pattern.idxmin()}:00 ({hourly_pattern.min():,} trips)")

# Day of week pattern
daily_pattern = df.groupby('day_of_week').size()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print("\n📊 Day of Week Pattern:")
for day, count in daily_pattern.items():
    print(f"   • {days[day]}: {count:,} trips")

# Weekend vs Weekday
weekend_trips = df[df['is_weekend'] == 1].shape[0]
weekday_trips = df[df['is_weekend'] == 0].shape[0]
print(f"\n📊 Weekend vs Weekday:")
print(f"   • Weekday trips: {weekday_trips:,} ({weekday_trips/total_samples*100:.1f}%)")
print(f"   • Weekend trips: {weekend_trips:,} ({weekend_trips/total_samples*100:.1f}%)")

# Top stations by demand
top_stations = df['start station id'].value_counts().head(10)
print("\n📊 Top 10 Stations by Demand:")
for idx, (station_id, count) in enumerate(top_stations.items(), 1):
    station_name = df[df['start station id'] == station_id]['start station name'].iloc[0]
    print(f"   {idx}. Station {station_id} ({station_name[:30]}): {count:,} trips")

# ==========================================
# 6. FEATURE ENGINEERING RECOMMENDATIONS
# ==========================================
print("\n" + "="*60)
print("6. FEATURE ENGINEERING PLAN")
print("="*60)

print("""
RECOMMENDED FEATURE SET FOR REGRESSION MODEL:

1. TEMPORAL FEATURES (High Priority):
   ✓ hour (0-23)
   ✓ day_of_week (0-6)
   ✓ month (1-12)
   ✓ is_weekend (0/1)
   ✓ is_rush_hour (0/1)
   ✓ season (derived from month)

2. SPATIAL FEATURES:
   ✓ station_id (one-hot encoded or embedding)
   ✓ station_lat
   ✓ station_lon
   ✓ station_cluster (from K-means clustering)
   ✓ distance_to_city_center

3. LAG FEATURES (Critical for time series):
   ✓ demand_last_hour
   ✓ demand_same_hour_yesterday
   ✓ demand_same_hour_last_week
   ✓ rolling_mean_3h
   ✓ rolling_mean_24h

4. INTERACTION FEATURES:
   ✓ hour × is_weekend
   ✓ hour × station_cluster
   ✓ month × is_weekend

5. EXTERNAL DATA (if available):
   ○ temperature
   ○ precipitation
   ○ is_holiday
   ○ special_event_nearby
""")

# ==========================================
# 7. VISUALIZATION CODE
# ==========================================
print("\n" + "="*60)
print("7. VISUALIZATION CODE (Run separately)")
print("="*60)

print("""
# Run this code separately to generate visualizations:

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Hourly demand pattern
hourly_pattern.plot(kind='bar', ax=axes[0,0], color='steelblue')
axes[0,0].set_title('Hourly Demand Pattern')
axes[0,0].set_xlabel('Hour of Day')
axes[0,0].set_ylabel('Number of Trips')

# 2. Daily demand pattern
daily_pattern.plot(kind='bar', ax=axes[0,1], color='coral')
axes[0,1].set_title('Daily Demand Pattern')
axes[0,1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)

# 3. Trip duration distribution
axes[0,2].hist(df['tripduration_min'][df['tripduration_min'] < 60], bins=30, color='green', alpha=0.7)
axes[0,2].set_title('Trip Duration Distribution')
axes[0,2].set_xlabel('Duration (minutes)')

# 4. Top stations
top_stations.plot(kind='barh', ax=axes[1,0], color='purple')
axes[1,0].set_title('Top 10 Stations by Demand')

# 5. User type distribution
user_dist.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
axes[1,1].set_title('User Type Distribution')

# 6. Time period distribution
time_period_dist = df['time_period'].value_counts()
time_period_dist.plot(kind='bar', ax=axes[1,2], color='orange')
axes[1,2].set_title('Trips by Time Period')

plt.tight_layout()
plt.show()
""")

# ==========================================
# 8. NEXT STEPS
# ==========================================
print("\n" + "="*60)
print("8. NEXT STEPS FOR MODEL DEVELOPMENT")
print("="*60)

print("""
1. DATA PREPARATION:
   • Remove outliers (trips < 1 min or > 24 hours)
   • Handle missing postal codes
   • Create train/test split (80/20) with time-based splitting

2. FEATURE ENGINEERING:
   • Create all temporal features
   • Generate lag features (crucial for demand forecasting)
   • Cluster stations based on location and demand patterns
   • Create interaction features

3. MODEL SELECTION:
   • Baseline: Linear Regression with regularization
   • Tree-based: Random Forest, XGBoost (typically best for this type of data)
   • Time Series: ARIMA or Prophet for station-level forecasting
   • Deep Learning: LSTM if sequential patterns are strong

4. EVALUATION METRICS:
   • Primary: RMSE (Root Mean Square Error)
   • Secondary: MAE, MAPE, R²
   • Business Metric: % of hours with stockouts/overflow

5. VALIDATION STRATEGY:
   • Time-based split (no random shuffling!)
   • Cross-validation with time series split
   • Separate validation for peak vs off-peak hours
""")

print("\n" + "="*60)
print("EDA COMPLETE - Ready for Model Development")
print("="*60)