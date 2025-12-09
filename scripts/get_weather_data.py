import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

def get_weather_openmeteo():
    """
    Fetch Boston 2020 weather using Open-Meteo free API
    """
    # Boston coordinates
    latitude = 42.3601
    longitude = -71.0589
    
    # API endpoint for historical data
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    all_data = []
    
    # Fetch data month by month (API limitation)
    for month in range(1, 13):
        start_date = f"2020-{month:02d}-01"
        if month == 12:
            end_date = "2020-12-31"
        else:
            end_date = f"2020-{month+1:02d}-01"
            end_date = (pd.to_datetime(end_date) - timedelta(days=1)).strftime('%Y-%m-%d')
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,windspeed_10m,relativehumidity_2m,surface_pressure",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "America/New_York"
        }
        
        print(f"Fetching weather for {start_date} to {end_date}...")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            hourly_data = data['hourly']
            
            # Create dataframe for this month
            month_df = pd.DataFrame({
                'datetime': pd.to_datetime(hourly_data['time']),
                'temp': hourly_data['temperature_2m'],
                'precip': [p * 25.4 if p else 0 for p in hourly_data['precipitation']],  # Convert to mm
                'wind_speed': hourly_data['windspeed_10m'],
                'humidity': hourly_data['relativehumidity_2m'],
                'pressure': hourly_data['surface_pressure']
            })
            
            all_data.append(month_df)
            time.sleep(1)  # Be respectful to the free API
        else:
            print(f"Error fetching data for month {month}: {response.status_code}")
    
    if all_data:
        weather_df = pd.concat(all_data, ignore_index=True)
        print(f"Downloaded {len(weather_df)} hours of weather data")
        return weather_df
    else:
        return None

# Main function to get weather data
def get_boston_weather_2020(method):
    """
    Get Boston 2020 weather data using OpenMeteo API.
    
    Args:
        method: 'openmeteo'
    
    Returns:
        DataFrame with weather data
    """
    if method == 'openmeteo':
        return get_weather_openmeteo()
    else:
        print(f"Unknown method: {method}")
        return None

if __name__ == "__main__":
    print("Attempting to fetch weather data from Open-Meteo API...")
    weather_df = get_boston_weather_2020('openmeteo')
    
    if weather_df is not None:
        # Save the data
        weather_df.to_csv('Data/boston_weather_2020.csv', index=False)
        print(f"Weather data saved to 'boston_weather_2020.csv'")
        
        # Display summary
        print("\nWeather Data Summary:")
        print(weather_df.info())
        print("\nFirst few rows:")
        print(weather_df.head())
        print("\nTemperature statistics:")
        print(weather_df['temp'].describe())
    else:
        print("\nFailed to fetch data.")