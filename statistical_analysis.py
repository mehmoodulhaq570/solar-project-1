"""
Statistical Analysis of Solar Data (2000-2014)

This script reads the merged NSRDB solar dataset and performs statistical analysis on key variables.
Units for each column are included for clarity.

Data file: merged_nsrdb.csv

Column Units:
- Year, Month, Day, Hour, Minute: (no units)
- DHI: Diffuse Horizontal Irradiance (W/m^2)
- DNI: Direct Normal Irradiance (W/m^2)
- GHI: Global Horizontal Irradiance (W/m^2)
- Clearsky DHI: (W/m^2)
- Clearsky DNI: (W/m^2)
- Clearsky GHI: (W/m^2)
- Dew Point: (°C)
- Temperature: (°C)
- Pressure: (mbar)
- Relative Humidity: (%)
- Solar Zenith Angle: (Degree)
- Precipitable Water: (cm)
- Snow Depth: (m)
- Wind Direction: (Degrees)
- Wind Speed: (m/s)

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
file_path = 'merged_nsrdb.csv'
df = pd.read_csv(file_path)  # Skip metadata rows

# Rename columns for clarity (if needed)
df.columns = [
    'Year', 'Month', 'Day', 'Hour', 'Minute',
    'DHI (W/m^2)', 'DNI (W/m^2)', 'GHI (W/m^2)',
    'Clearsky DHI (W/m^2)', 'Clearsky DNI (W/m^2)', 'Clearsky GHI (W/m^2)',
    'Dew Point (°C)', 'Temperature (°C)', 'Pressure (mbar)',
    'Relative Humidity (%)', 'Solar Zenith Angle (Degree)',
    'Precipitable Water (cm)', 'Snow Depth (m)',
    'Wind Direction (Degrees)', 'Wind Speed (m/s)'
]

# Show basic info
df.info()
print('\nSummary statistics:')
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Solar Data (2000-2014)')
plt.tight_layout()
plt.show()

# Histograms for key variables
key_vars = [
    'DHI (W/m^2)', 'DNI (W/m^2)', 'GHI (W/m^2)',
    'Temperature (°C)', 'Pressure (mbar)', 'Relative Humidity (%)',
    'Wind Speed (m/s)'
]
df[key_vars].hist(bins=30, figsize=(14,10), layout=(3,3))
plt.suptitle('Histograms of Key Variables')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Boxplots for outlier detection
plt.figure(figsize=(14,8))
df[key_vars].boxplot()
plt.title('Boxplots of Key Variables')
plt.ylabel('Value (see units in column names)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Time series plot for GHI (example)
df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
plt.figure(figsize=(15,5))
plt.plot(df['Datetime'], df['GHI (W/m^2)'], label='GHI (W/m^2)', alpha=0.5)
plt.title('Global Horizontal Irradiance (GHI) Over Time')
plt.xlabel('Datetime')
plt.ylabel('GHI (W/m^2)')
plt.legend()
plt.tight_layout()
plt.show()

# Save summary statistics to CSV
df.describe().to_csv('solar_data_summary_statistics.csv')
print('Summary statistics saved to solar_data_summary_statistics.csv')
