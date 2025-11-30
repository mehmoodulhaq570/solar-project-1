# import openmeteo_requests
# import pandas as pd
# import requests_cache
# from retry_requests import retry
# import os

# # ===============================================================
# # Setup the Open-Meteo API client with caching and retry
# # ===============================================================
# cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
# retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
# openmeteo = openmeteo_requests.Client(session=retry_session)

# # ===============================================================
# # API request parameters
# # ===============================================================
# url = "https://api.open-meteo.com/v1/forecast"
# params = {
#     "latitude": 31.558,     # Lahore latitude
#     "longitude": 74.3507,   # Lahore longitude
#     "hourly": [
#         "temperature_2m",
#         "wind_speed_10m",
#         "wind_direction_10m",
#         "relative_humidity_2m",
#         "shortwave_radiation",
#         "direct_radiation",
#         "diffuse_radiation",
#         "surface_pressure",
#         "direct_normal_irradiance"
#     ],
#     "timezone": "Asia/Bangkok",
#     "past_days": 3652,
#     "wind_speed_unit": "ms"
# }

# # ===============================================================
# # Fetch data
# # ===============================================================
# responses = openmeteo.weather_api(url, params=params)
# response = responses[0]

# print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
# print(f"Elevation: {response.Elevation()} m asl")
# print(f"Timezone: {response.Timezone()} ({response.TimezoneAbbreviation()})")
# print(f"UTC Offset: {response.UtcOffsetSeconds()} seconds")

# # ===============================================================
# # Process hourly data
# # ===============================================================
# hourly = response.Hourly()
# hourly_data = {
#     "date": pd.date_range(
#         start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
#         end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
#         freq=pd.Timedelta(seconds=hourly.Interval()),
#         inclusive="left"
#     ),
#     "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
#     "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
#     "wind_direction_10m": hourly.Variables(2).ValuesAsNumpy(),
#     "relative_humidity_2m": hourly.Variables(3).ValuesAsNumpy(),
#     "shortwave_radiation": hourly.Variables(4).ValuesAsNumpy(),
#     "direct_radiation": hourly.Variables(5).ValuesAsNumpy(),
#     "diffuse_radiation": hourly.Variables(6).ValuesAsNumpy(),
#     "surface_pressure": hourly.Variables(7).ValuesAsNumpy(),
#     "direct_normal_irradiance": hourly.Variables(8).ValuesAsNumpy()
# }

# hourly_df = pd.DataFrame(hourly_data)

# # ===============================================================
# # Save to CSV
# # ===============================================================
# output_filename = "lahore_weather_data_10_years.csv"
# hourly_df.to_csv(output_filename, index=False)
# print(f"\nWeather data successfully saved to: {os.path.abspath(output_filename)}")

# # Display first few rows for verification
# print("\nHourly data sample:")
# print(hourly_df.head())



import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Make sure all required weather variables are listed here
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
    "latitude": 33.7215,
    "longitude": 73.0433,
    "start_date": "2016-01-01",
    "end_date": "2025-10-28",
    "daily": ["sunset", "uv_index_max", "apparent_temperature_max", "snowfall_sum"],
    "hourly": ["temperature_2m", "uv_index_clear_sky", "direct_radiation", "direct_normal_irradiance"],
    "timezone": "Asia/Bangkok",
}

# API request
responses = openmeteo.weather_api(url, params=params)
response = responses[0]

print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_uv_index_clear_sky = hourly.Variables(1).ValuesAsNumpy()
hourly_direct_radiation = hourly.Variables(2).ValuesAsNumpy()
hourly_direct_normal_irradiance = hourly.Variables(3).ValuesAsNumpy()

hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "temperature_2m": hourly_temperature_2m,
    "uv_index_clear_sky": hourly_uv_index_clear_sky,
    "direct_radiation": hourly_direct_radiation,
    "direct_normal_irradiance": hourly_direct_normal_irradiance
}
hourly_dataframe = pd.DataFrame(data=hourly_data)
print("\nHourly data\n", hourly_dataframe.head())

# Process daily data
daily = response.Daily()
daily_sunset = daily.Variables(0).ValuesInt64AsNumpy()
daily_uv_index_max = daily.Variables(1).ValuesAsNumpy()
daily_apparent_temperature_max = daily.Variables(2).ValuesAsNumpy()
daily_snowfall_sum = daily.Variables(3).ValuesAsNumpy()

daily_data = {
    "date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    ),
    "sunset": daily_sunset,
    "uv_index_max": daily_uv_index_max,
    "apparent_temperature_max": daily_apparent_temperature_max,
    "snowfall_sum": daily_snowfall_sum
}
daily_dataframe = pd.DataFrame(data=daily_data)
print("\nDaily data\n", daily_dataframe.head())

# 💾 Save data to CSV files
hourly_dataframe.to_csv("hourly_weather_data.csv", index=False)
daily_dataframe.to_csv("daily_weather_data.csv", index=False)

print("\nData saved successfully:")
print(" - hourly_weather_data.csv")
print(" - daily_weather_data.csv")
