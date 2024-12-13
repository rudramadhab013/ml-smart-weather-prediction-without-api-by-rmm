import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the data
df_2019_2023 = pd.read_csv('gunupur_weather_2019_2023.csv')
df_2024_2026 = pd.read_csv('gunupur_weather_2024_2026_prediction.csv')

# Combine the datasets
df = pd.concat([df_2019_2023, df_2024_2026], ignore_index=True)

# Convert 'Month' to numerical values
month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
df['Month_Num'] = df['Month'].map(month_map)

# Prepare the features and target
X = df[['Month_Num', 'Year']]
y = df['Avg_Temp (°C)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Function to predict temperature
def predict_temperature(month, year, current_temp):
    month_num = month_map[month]
    predicted_avg = model.predict([[month_num, year]])[0]  # Ensure proper input format
    adjusted_temp = (predicted_avg + current_temp) / 2
    return adjusted_temp

# Function to generate daily temperatures for a month
def generate_daily_temps(month, year, avg_temp):
    days_in_month = 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30
    if month == 2:
        days_in_month = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    
    daily_temps = np.random.normal(avg_temp, 2, days_in_month)
    return daily_temps

# Main function to generate predictions and plots
def generate_temperature_forecast(current_month, current_temp):
    current_year = datetime.now().year
    
    # Predict temperatures
    predicted_temp_today = predict_temperature(current_month, current_year, current_temp)
    
    # Generate daily temperatures for the current month
    monthly_temps = generate_daily_temps(month_map[current_month], current_year, predicted_temp_today)
    
    # Calculate weekly average
    current_day = datetime.now().day
    week_start = max(0, current_day - 3)
    week_end = min(len(monthly_temps), current_day + 4)
    weekly_avg = np.mean(monthly_temps[week_start:week_end])
    
    # Print predictions
    print(f"Predicted temperature for today: {predicted_temp_today:.2f}°C")
    print(f"Predicted average for this week: {weekly_avg:.2f}°C")
    print(f"Predicted average for {current_month}: {np.mean(monthly_temps):.2f}°C")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Daily temperatures for the month
    days = range(1, len(monthly_temps) + 1)
    ax1.plot(days, monthly_temps, marker='o')
    ax1.set_title(f'Daily Temperatures for {current_month} {current_year}')
    ax1.set_xlabel('Day of Month')
    ax1.set_ylabel('Temperature (°C)')
    ax1.axhline(y=np.mean(monthly_temps), color='r', linestyle='--', label='Monthly Average')
    ax1.axvline(x=current_day, color='g', linestyle='--', label='Current Day')
    ax1.legend()
    
    # Plot 2: Temperature distribution
    sns.histplot(monthly_temps, kde=True, ax=ax2)
    ax2.set_title(f'Temperature Distribution for {current_month} {current_year}')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(x=predicted_temp_today, color='r', linestyle='--', label='Today\'s Prediction')
    ax2.axvline(x=weekly_avg, color='g', linestyle='--', label='Weekly Average')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Get user input for current month and temperature
current_month = input("Enter the current month (e.g., October): ")
current_temp = float(input("Enter the current temperature (in °C): "))

# Generate the temperature forecast
generate_temperature_forecast(current_month, current_temp)
