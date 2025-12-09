import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def sanity_check_bluebikes_data(ml_ready_path, original_trips_path, weather_path):
    """
    Comprehensive sanity checks to verify the ML-ready dataset
    """
    print("="*80)
    print("BLUEBIKES DATA SANITY CHECK")
    print("="*80)
    
    # Load data
    print("\n[1] Loading datasets...")
    ml_df = pd.read_csv(ml_ready_path)
    ml_df['timestamp'] = pd.to_datetime(ml_df['timestamp'])
    
    original_df = pd.read_csv(original_trips_path)
    original_df['starttime'] = pd.to_datetime(original_df['starttime'])
    
    weather_df = pd.read_csv(weather_path)
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    
    print(f"   ✓ ML-ready data: {len(ml_df):,} records")
    print(f"   ✓ Original trips: {len(original_df):,} records")
    print(f"   ✓ Weather data: {len(weather_df):,} records")
    
    # ========================================================================
    # CHECK 1: Demand Calculation Verification
    # ========================================================================
    print("\n" + "="*80)
    print("[2] DEMAND CALCULATION VERIFICATION")
    print("="*80)
    
    # Sample a few station-hour combinations and verify demand manually
    sample_checks = ml_df.sample(10, random_state=42)
    
    print("\nVerifying 10 random station-hour demands against original data...")
    all_match = True
    
    for idx, row in sample_checks.iterrows():
        station_id = row['station_id']
        timestamp = row['timestamp']
        ml_demand = row['demand']
        
        # Count trips in original data for this station-hour
        original_df['hour_start'] = original_df['starttime'].dt.floor('h')
        actual_demand = len(original_df[
            (original_df['start station id'] == station_id) & 
            (original_df['hour_start'] == timestamp)
        ])
        
        match = (ml_demand == actual_demand)
        status = "✓" if match else "✗"
        
        print(f"  {status} Station {station_id}, {timestamp}: ML={ml_demand}, Actual={actual_demand}")
        
        if not match:
            all_match = False
    
    if all_match:
        print("\n✅ PASS: All demand values match original trip counts!")
    else:
        print("\n⚠️  WARNING: Some demand values don't match!")
    
    # ========================================================================
    # CHECK 2: Total Trips Conservation
    # ========================================================================
    print("\n" + "="*80)
    print("[3] TOTAL TRIPS CONSERVATION")
    print("="*80)
    
    total_ml_demand = ml_df['demand'].sum()
    total_original_trips = len(original_df)
    
    print(f"\n  Total demand in ML dataset:  {total_ml_demand:,}")
    print(f"  Total trips in original:     {total_original_trips:,}")
    print(f"  Difference:                  {abs(total_ml_demand - total_original_trips):,}")
    
    if total_ml_demand == total_original_trips:
        print("\n✅ PASS: Total trips conserved perfectly!")
    else:
        pct_diff = abs(total_ml_demand - total_original_trips) / total_original_trips * 100
        print(f"\n⚠️  Difference: {pct_diff:.2f}%")
    
    # ========================================================================
    # CHECK 3: Date Range Verification
    # ========================================================================
    print("\n" + "="*80)
    print("[4] DATE RANGE VERIFICATION")
    print("="*80)
    
    ml_start = ml_df['timestamp'].min()
    ml_end = ml_df['timestamp'].max()
    original_start = original_df['starttime'].min()
    original_end = original_df['starttime'].max()
    
    print(f"\n  ML dataset range:       {ml_start} to {ml_end}")
    print(f"  Original data range:    {original_start} to {original_end}")
    
    # Check if ML range is within original range
    if ml_start >= original_start and ml_end <= original_end:
        print("\n✅ PASS: ML date range is within original data range")
    else:
        print("\n⚠️  WARNING: ML date range extends beyond original data!")
    
    # ========================================================================
    # CHECK 4: Station Count Verification
    # ========================================================================
    print("\n" + "="*80)
    print("[5] STATION COUNT VERIFICATION")
    print("="*80)
    
    ml_stations = set(ml_df['station_id'].unique())
    original_stations = set(original_df['start station id'].unique())
    
    print(f"\n  Stations in ML dataset:    {len(ml_stations)}")
    print(f"  Stations in original:      {len(original_stations)}")
    
    missing_stations = original_stations - ml_stations
    extra_stations = ml_stations - original_stations
    
    if len(missing_stations) == 0 and len(extra_stations) == 0:
        print("\n✅ PASS: All stations present, no extras")
    else:
        if missing_stations:
            print(f"\n⚠️  Missing {len(missing_stations)} stations from original")
        if extra_stations:
            print(f"\n⚠️  Found {len(extra_stations)} extra stations not in original")
    
    # ========================================================================
    # CHECK 5: Weather Data Merge Verification
    # ========================================================================
    print("\n" + "="*80)
    print("[6] WEATHER DATA MERGE VERIFICATION")
    print("="*80)
    
    # Check if weather columns exist and have reasonable values
    weather_cols = ['temperature', 'precipitation_mm', 'wind_speed_mph']
    
    for col in weather_cols:
        if col in ml_df.columns:
            non_null = ml_df[col].notna().sum()
            null_count = ml_df[col].isna().sum()
            print(f"\n  {col}:")
            print(f"    Non-null values: {non_null:,} ({non_null/len(ml_df)*100:.1f}%)")
            print(f"    Null values: {null_count:,}")
            print(f"    Range: [{ml_df[col].min():.2f}, {ml_df[col].max():.2f}]")
    
    # Verify weather values are reasonable for Boston
    temp_min = ml_df['temperature'].min()
    temp_max = ml_df['temperature'].max()
    
    # Check if temperature is in Fahrenheit (32-100°F) vs Celsius (-20 to 45°C)
    if temp_min >= 32 and temp_max <= 100:
        print("\n⚠️  WARNING: Temperature appears to be in FAHRENHEIT (not Celsius)")
        print(f"    Range: [{temp_min:.1f}°F, {temp_max:.1f}°F]")
        print(f"    Equivalent: [{(temp_min-32)*5/9:.1f}°C, {(temp_max-32)*5/9:.1f}°C]")
        print("    This is NORMAL if your weather data source uses Fahrenheit")
    elif temp_min >= -20 and temp_max <= 45:
        print("\n✅ PASS: Weather values are reasonable for Boston (Celsius)")
        print(f"    Range: [{temp_min:.1f}°C, {temp_max:.1f}°C]")
    else:
        print("\n⚠️  WARNING: Weather values seem outside reasonable range")
        print(f"    Range: [{temp_min:.1f}, {temp_max:.1f}]")
    
    # ========================================================================
    # CHECK 6: Temporal Features Sanity
    # ========================================================================
    print("\n" + "="*80)
    print("[7] TEMPORAL FEATURES SANITY")
    print("="*80)
    
    # Check cyclical encodings are bounded [-1, 1]
    cyclical_features = [col for col in ml_df.columns if '_sin' in col or '_cos' in col]
    
    print(f"\n  Checking {len(cyclical_features)} cyclical features...")
    cyclical_ok = True
    
    for col in cyclical_features:
        min_val = ml_df[col].min()
        max_val = ml_df[col].max()
        if min_val < -1.01 or max_val > 1.01:  # Small tolerance
            print(f"  ✗ {col}: range [{min_val:.3f}, {max_val:.3f}] - OUT OF BOUNDS")
            cyclical_ok = False
    
    if cyclical_ok:
        print("  ✅ PASS: All cyclical features in valid range [-1, 1]")
    
    # Check weekend flag (should be 0 or 1)
    if 'is_weekend' in ml_df.columns:
        weekend_values = ml_df['is_weekend'].unique()
        if set(weekend_values).issubset({0, 1}):
            print("  ✅ PASS: Weekend flag is binary (0 or 1)")
        else:
            print(f"  ✗ Weekend flag has unexpected values: {weekend_values}")
    
    # ========================================================================
    # CHECK 7: Historical Features Sanity
    # ========================================================================
    print("\n" + "="*80)
    print("[8] HISTORICAL FEATURES SANITY")
    print("="*80)
    
    # Check lag features
    lag_features = ['demand_t_minus_1', 'demand_t_minus_24', 'demand_t_minus_168']
    
    print("\n  Testing lag feature logic...")
    # Sort by station and time
    test_df = ml_df.sort_values(['station_id', 'timestamp']).head(200)
    
    # For each station, check if lag-1 matches previous demand
    lag_check_passed = 0
    lag_check_total = 0
    
    for station in test_df['station_id'].unique()[:5]:
        station_data = test_df[test_df['station_id'] == station].reset_index(drop=True)
        
        for i in range(2, min(10, len(station_data))):
            expected_lag1 = station_data.loc[i-1, 'demand']
            actual_lag1 = station_data.loc[i, 'demand_t_minus_1']
            
            if pd.notna(actual_lag1) and expected_lag1 == actual_lag1:
                lag_check_passed += 1
            lag_check_total += 1
    
    if lag_check_total > 0:
        lag_accuracy = lag_check_passed / lag_check_total * 100
        print(f"  Lag-1 accuracy: {lag_check_passed}/{lag_check_total} ({lag_accuracy:.1f}%)")
        
        if lag_accuracy > 90:
            print("  ✅ PASS: Lag features appear correct")
        else:
            print("  ⚠️  WARNING: Some lag features may be incorrect")
    
    # ========================================================================
    # CHECK 8: Missing Values Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("[9] MISSING VALUES ANALYSIS")
    print("="*80)
    
    missing_summary = ml_df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    
    if len(missing_summary) == 0:
        print("\n  ✅ PASS: No missing values in dataset!")
    else:
        print(f"\n  Found missing values in {len(missing_summary)} columns:")
        for col, count in missing_summary.head(10).items():
            pct = count / len(ml_df) * 100
            
            # Explain expected missing values in lag features
            if 'minus_1' in col and count == ml_df['station_id'].nunique():
                status = "✅ EXPECTED"
                reason = "(first hour per station)"
            elif 'minus_24' in col and count < ml_df['station_id'].nunique() * 25:
                status = "✅ EXPECTED"
                reason = "(first 24 hours per station)"
            elif 'minus_168' in col and count < ml_df['station_id'].nunique() * 170:
                status = "✅ EXPECTED"
                reason = "(first 7 days per station)"
            elif 'rolling' in col or 'average' in col or 'trend' in col:
                status = "✅ EXPECTED"
                reason = "(early observations)"
            else:
                status = "⚠️ "
                reason = ""
            
            print(f"    {status} {col}: {count:,} ({pct:.2f}%) {reason}")
    
    # ========================================================================
    # CHECK 9: Demand Distribution Reasonableness
    # ========================================================================
    print("\n" + "="*80)
    print("[10] DEMAND DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print(f"\n  Demand statistics:")
    print(f"    Mean:   {ml_df['demand'].mean():.2f}")
    print(f"    Median: {ml_df['demand'].median():.2f}")
    print(f"    Std:    {ml_df['demand'].std():.2f}")
    print(f"    Min:    {ml_df['demand'].min():.0f}")
    print(f"    Max:    {ml_df['demand'].max():.0f}")
    print(f"    25th:   {ml_df['demand'].quantile(0.25):.2f}")
    print(f"    75th:   {ml_df['demand'].quantile(0.75):.2f}")
    
    # Check for negative values (should not exist)
    negative_demand = (ml_df['demand'] < 0).sum()
    if negative_demand == 0:
        print("\n  ✅ PASS: No negative demand values")
    else:
        print(f"\n  ✗ FAIL: Found {negative_demand} negative demand values!")
    
    # ========================================================================
    # CHECK 10: Duplicate Records
    # ========================================================================
    print("\n" + "="*80)
    print("[11] DUPLICATE RECORDS CHECK")
    print("="*80)
    
    duplicates = ml_df.duplicated(subset=['station_id', 'timestamp']).sum()
    
    if duplicates == 0:
        print("\n  ✅ PASS: No duplicate station-hour combinations")
    else:
        print(f"\n  ✗ WARNING: Found {duplicates} duplicate station-hour records!")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("[12] GENERATING VALIDATION VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Demand distribution
    axes[0, 0].hist(ml_df['demand'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Demand (trips/hour)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Demand Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Demand over time
    daily_demand = ml_df.groupby(ml_df['timestamp'].dt.date)['demand'].sum()
    axes[0, 1].plot(daily_demand.index, daily_demand.values, linewidth=1)
    axes[0, 1].set_xlabel('Date', fontsize=11)
    axes[0, 1].set_ylabel('Total Daily Demand', fontsize=11)
    axes[0, 1].set_title('Daily Demand Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Hourly pattern
    hourly_avg = ml_df.groupby(ml_df['timestamp'].dt.hour)['demand'].mean()
    axes[0, 2].bar(hourly_avg.index, hourly_avg.values, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Hour of Day', fontsize=11)
    axes[0, 2].set_ylabel('Average Demand', fontsize=11)
    axes[0, 2].set_title('Average Demand by Hour', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # 4. Temperature distribution
    axes[1, 0].hist(ml_df['temperature'], bins=40, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Temperature (°C)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Temperature Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Demand vs Temperature
    sample_for_plot = ml_df.sample(min(10000, len(ml_df)), random_state=42)
    axes[1, 1].scatter(sample_for_plot['temperature'], sample_for_plot['demand'], 
                       alpha=0.1, s=1, color='blue')
    axes[1, 1].set_xlabel('Temperature (°C)', fontsize=11)
    axes[1, 1].set_ylabel('Demand', fontsize=11)
    axes[1, 1].set_title('Demand vs Temperature', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Weekly pattern
    weekly_avg = ml_df.groupby(ml_df['timestamp'].dt.dayofweek)['demand'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 2].bar(range(7), weekly_avg.values, edgecolor='black', alpha=0.7, color='green')
    axes[1, 2].set_xticks(range(7))
    axes[1, 2].set_xticklabels(day_names)
    axes[1, 2].set_xlabel('Day of Week', fontsize=11)
    axes[1, 2].set_ylabel('Average Demand', fontsize=11)
    axes[1, 2].set_title('Average Demand by Day of Week', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('sanity_check_visualizations.png', dpi=300, bbox_inches='tight')
    print("\n  ✓ Saved: sanity_check_visualizations.png")
    plt.show()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SANITY CHECK SUMMARY")
    print("="*80)
    
    print("\n✓ Checks Completed:")
    print("  [1] Demand calculation verification")
    print("  [2] Total trips conservation")
    print("  [3] Date range verification")
    print("  [4] Station count verification")
    print("  [5] Weather data merge")
    print("  [6] Temporal features sanity")
    print("  [7] Historical features logic")
    print("  [8] Missing values analysis")
    print("  [9] Demand distribution reasonableness")
    print("  [10] Duplicate records check")
    print("  [11] Validation visualizations")
    
    print("\n" + "="*80)
    print("✅ SANITY CHECK COMPLETE!")
    print("="*80)
    
    return ml_df

# Run sanity check
if __name__ == "__main__":
    results = sanity_check_bluebikes_data(
        ml_ready_path='Data/bluebikes_ml_ready.csv',
        original_trips_path='Data/bluebikes_tripdata_2020.csv',
        weather_path='Data/boston_weather_2020.csv'
    )