# soil.py (enhanced with air temperature analysis)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import joblib
import warnings
from pandas.api.types import is_datetime64tz_dtype

warnings.filterwarnings('ignore')

# -----------------------
# Visual defaults
# -----------------------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# -----------------------
# Load data
# -----------------------
CSV_FILENAME = 'daily_weather_data_sonha_2014-01-01_to_2025-09-02.csv'  # update path/name if different
print("Loading and preparing data for Sohna, Gurgaon...")
df = pd.read_csv(CSV_FILENAME)

# parse date and set index
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).copy()
df.set_index('date', inplace=True)

# remove timezone if present -> make tz-naive
if is_datetime64tz_dtype(df.index.dtype):
    df.index = df.index.tz_convert(None)

# Save the original last observed date (important!)
original_last_date = df.index.max()

# -----------------------
# Soil temperature proxy
# -----------------------
window_size = 7
df['soil_temp_proxy'] = (
    0.7 * df['temperature_2m_mean']
    + 0.2 * df['temperature_2m_mean'].shift(1)
    + 0.1 * df['temperature_2m_mean'].shift(2)
).rolling(window=window_size).mean()

# -----------------------
# Seasonal adjustment (safe for future dates)
# -----------------------
def apply_seasonal_adjustment(dates, temps, rain_series=None):
    """
    Apply seasonal adjustments. If rain_series provided, it's a pandas Series indexed by date.
    For dates missing in rain_series (e.g., future), rain defaults to 0.
    Returns numpy array of adjusted temps.
    """
    adjusted = []
    for date, temp in zip(dates, temps):
        m = date.month
        if m in (12, 1, 2):
            adj = -1.5
        elif m in (3, 4, 5):
            adj = 2.0
        elif m in (6, 7, 8):
            adj = 0.5
        else:
            adj = -0.5

        rain = 0.0
        if rain_series is not None:
            try:
                rain = float(rain_series.loc[date])
            except Exception:
                try:
                    rain = float(rain_series.get(date, 0.0))
                except Exception:
                    rain = 0.0

        rain_effect = -min(rain / 8.0, 3.0)
        adjusted.append(temp + adj + rain_effect)
    return np.array(adjusted)

# apply seasonal adjustment on historical data (we have rain for history)
df['soil_temp_proxy'] = apply_seasonal_adjustment(
    df.index, df['soil_temp_proxy'].values,
    rain_series=df['rain_sum'] if 'rain_sum' in df.columns else None
)

# -----------------------
# Feature engineering
# -----------------------
# Use df_model as the dataset for modelling (so we can dropna there but keep original df intact)
df_model = df.copy()

df_model['day_of_year'] = df_model.index.dayofyear
df_model['day_of_year_sin'] = np.sin(2 * np.pi * df_model['day_of_year'] / 365.25)
df_model['day_of_year_cos'] = np.cos(2 * np.pi * df_model['day_of_year'] / 365.25)
df_model['month'] = df_model.index.month
df_model['season'] = df_model['month'].apply(lambda x: (x % 12 + 3) // 3)

for lag in [1, 2, 3, 7, 14, 30]:
    df_model[f'temp_mean_lag_{lag}'] = df_model['temperature_2m_mean'].shift(lag)
    df_model[f'rain_lag_{lag}'] = df_model['rain_sum'].shift(lag) if 'rain_sum' in df_model.columns else np.nan
    df_model[f'humidity_lag_{lag}'] = df_model['relative_humidity_2m_mean'].shift(lag)

for window in [7, 14, 30]:
    df_model[f'temp_rolling_mean_{window}'] = df_model['temperature_2m_mean'].rolling(window=window).mean()
    if 'rain_sum' in df_model.columns:
        df_model[f'rain_rolling_sum_{window}'] = df_model['rain_sum'].rolling(window=window).sum()
    df_model[f'humidity_rolling_mean_{window}'] = df_model['relative_humidity_2m_mean'].rolling(window=window).mean()

# Create target: 90-day future soil temp, then dropna only on df_model
df_model['soil_temp_future_90d'] = df_model['soil_temp_proxy'].shift(-90)
df_model = df_model.dropna()  # Only used for training/testing

# -----------------------
# Prepare features & target
# -----------------------
features = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'rain_sum', 'relative_humidity_2m_mean', 'pressure_msl_mean',
    'wind_speed_10m_mean', 'wind_speed_10m_max',
    'day_of_year_sin', 'day_of_year_cos', 'month', 'season'
]
for lag in [1, 2, 3, 7, 14, 30]:
    features.extend([f'temp_mean_lag_{lag}', f'rain_lag_{lag}', f'humidity_lag_{lag}'])
for window in [7, 14, 30]:
    features.extend([f'temp_rolling_mean_{window}', f'rain_rolling_sum_{window}' if 'rain_sum' in df.columns else '', f'humidity_rolling_mean_{window}'])
features = [f for f in features if f and f in df_model.columns]

X = df_model[features]
y = df_model['soil_temp_future_90d']

# -----------------------
# Train-test split (time series)
# -----------------------
train_size = int(0.8 * len(X))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# -----------------------
# Model pipeline & train
# -----------------------
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])
print("Training model with seasonal constraints...")
model.fit(X_train, y_train)
joblib.dump(model, 'soil_temp_rf_pipeline.joblib')

# evaluate on test
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Model Performance:")
print(f"Mean Absolute Error for 90-day prediction: {mae:.2f} °C")
print(f"R² Score: {r2:.2f}")

# -----------------------
# Historical trend + forecast (use original_last_date for forecast anchor)
# -----------------------
print("Creating historical trend visualization with prediction...")
fig, ax = plt.subplots(figsize=(16, 10))

# Plot full historical soil proxy from original df (not df_model)
ax.plot(df.index, df['soil_temp_proxy'], label='Historical Soil Temperature', linewidth=1.5, alpha=0.7)

# Plot test period (these come from df_model indices)
test_dates = y_test.index
ax.plot(test_dates, y_test.values, label='Actual (Test Period)', linewidth=2, color='green')
ax.plot(test_dates, y_pred, label='Predicted (Test Period)', linewidth=2, color='red')

# Generate future dates starting from original_last_date
future_dates = [original_last_date + timedelta(days=i) for i in range(1, 91)]

# Create future predictions using historical same-day-of-year averages + small noise
rng = np.random.default_rng(42)
future_predictions = []
for future_date in future_dates:
    same_period = df[(df.index.month == future_date.month) & (df.index.day == future_date.day)]
    if len(same_period) > 0:
        pred = same_period['soil_temp_proxy'].mean() + rng.normal(0, 1.5)
    else:
        pred = df['soil_temp_proxy'].iloc[-1]
    future_predictions.append(pred)

# Seasonal adjust future predictions (no rain series for future)
future_predictions = apply_seasonal_adjustment(future_dates, future_predictions, rain_series=None)

ax.plot(future_dates, future_predictions, label='90-Day Soil Forecast', linewidth=3, color='darkorange')

# Mark forecast start at the real last observation (original_last_date)
ax.axvline(x=original_last_date, color='black', linestyle='--', alpha=0.7, label='Forecast Start')

ax.set_title('Soil Temperature Trends for Sohna, Gurgaon\nHistorical Data and 90-Day Forecast', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Soil Temperature (°C)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('historical_trend_with_forecast.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------
# NEW: Combined Soil and Air Temperature Analysis
# -----------------------
print("Creating combined soil and air temperature analysis...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Plot 1: Soil Temperature
ax1.plot(df.index, df['soil_temp_proxy'], label='Historical Soil Temperature', linewidth=1.5, color='brown', alpha=0.7)
ax1.plot(test_dates, y_test.values, label='Actual Soil (Test Period)', linewidth=2, color='green')
ax1.plot(test_dates, y_pred, label='Predicted Soil (Test Period)', linewidth=2, color='red')
ax1.plot(future_dates, future_predictions, label='90-Day Soil Forecast', linewidth=3, color='darkorange')
ax1.axvline(x=original_last_date, color='black', linestyle='--', alpha=0.7, label='Forecast Start')
ax1.set_title('Soil Temperature Analysis for Sohna, Gurgaon', fontsize=14, fontweight='bold')
ax1.set_ylabel('Soil Temperature (°C)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator())

# Plot 2: Air Temperature
# Plot historical air temperature
ax2.plot(df.index, df['temperature_2m_mean'], label='Historical Air Temperature', linewidth=1.5, color='blue', alpha=0.7)

# Generate air temperature forecast using similar method
future_air_predictions = []
for future_date in future_dates:
    same_period = df[(df.index.month == future_date.month) & (df.index.day == future_date.day)]
    if len(same_period) > 0:
        pred = same_period['temperature_2m_mean'].mean() + rng.normal(0, 1.2)
    else:
        pred = df['temperature_2m_mean'].iloc[-1]
    future_air_predictions.append(pred)

ax2.plot(future_dates, future_air_predictions, label='90-Day Air Forecast', linewidth=3, color='darkred')
ax2.axvline(x=original_last_date, color='black', linestyle='--', alpha=0.7, label='Forecast Start')
ax2.set_title('Air Temperature Analysis for Sohna, Gurgaon', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Air Temperature (°C)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('combined_soil_air_temperature_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------
# Seasonal comparison chart (fixed to use original_last_date/present_date)
# -----------------------
print("Creating seasonal comparison chart...")

# yearly composite (1..365) from entire df
df['year'] = df.index.year
df['day_of_year'] = df.index.dayofyear
yearly_avg = df.groupby('day_of_year')['soil_temp_proxy'].agg(['mean', 'std']).reset_index()

# Determine present_date: earlier of original_last_date and today
now = pd.Timestamp.now().normalize()
present_date = min(original_last_date, now)

# Anchor composite year to present_date.year for plotting months properly
baseline_year = present_date.year
baseline_start = pd.Timestamp(year=baseline_year, month=1, day=1)

# map day_of_year -> date in baseline year for plotting the composite average
yearly_avg['plot_date'] = baseline_start + pd.to_timedelta(yearly_avg['day_of_year'] - 1, unit='D')

fig, ax = plt.subplots(figsize=(14, 8))

# Historical mean +/- std band
ax.fill_between(yearly_avg['plot_date'],
                yearly_avg['mean'] - yearly_avg['std'],
                yearly_avg['mean'] + yearly_avg['std'],
                alpha=0.3, color='gray', label='Historical Range (±1 STD)')

# Historical average line
ax.plot(yearly_avg['plot_date'], yearly_avg['mean'], label='Historical Average', linewidth=2, color='blue')

# Current-year data: use original df up to present_date (so it stops at present_date)
current_year_val = original_last_date.year
current_year_df = df[(df.index.year == current_year_val) & (df.index <= present_date)].copy()
if not current_year_df.empty:
    current_year_df = current_year_df.assign(
        plot_date = baseline_start + pd.to_timedelta(current_year_df['day_of_year'] - 1, unit='D')
    )
    ax.plot(current_year_df['plot_date'], current_year_df['soil_temp_proxy'],
            label=f'Current Year ({current_year_val})', linewidth=2, color='green')

# Forecast: use future_dates (which start AFTER original_last_date) plotted on real dates
future_dates_series = pd.to_datetime(future_dates)
ax.plot(future_dates_series, future_predictions, label='90-Day Forecast', linewidth=3, color='red')

# Title indicates data span
ax.set_title(f'Seasonal Soil Temperature Patterns for Sohna, Gurgaon\nHistorical vs Current vs Forecast ({df.index.year.min()}–{present_date.year})',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Soil Temperature (°C)')

# Set x-limits from Jan 1 baseline year to last forecast date
x_min = baseline_start
x_max = future_dates_series[-1]
ax.set_xlim([x_min, x_max + pd.Timedelta(days=1)])

# Month ticks
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.xticks(rotation=0)

ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('seasonal_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------
# Regional planting timetable (keeps year in title)
# -----------------------
print("Creating focused planting timetable...")
crop_requirements = {
    'Wheat': {'min_temp': 10, 'optimal_min': 12, 'optimal_max': 25, 'max_temp': 30,
              'planting_season': 'Oct-Nov', 'harvest_season': 'Mar-Apr'},
    'Mustard': {'min_temp': 10, 'optimal_min': 15, 'optimal_max': 30, 'max_temp': 35,
                'planting_season': 'Oct-Nov', 'harvest_season': 'Feb-Mar'},
    'Chickpea': {'min_temp': 10, 'optimal_min': 15, 'optimal_max': 30, 'max_temp': 35,
                 'planting_season': 'Oct-Nov', 'harvest_season': 'Feb-Mar'},
    'Barley': {'min_temp': 8, 'optimal_min': 12, 'optimal_max': 25, 'max_temp': 30,
               'planting_season': 'Oct-Nov', 'harvest_season': 'Mar-Apr'},
    'Vegetables': {'min_temp': 15, 'optimal_min': 18, 'optimal_max': 30, 'max_temp': 35,
                   'planting_season': 'Feb-Mar', 'harvest_season': 'Varies'},
}

# Determine optimal windows from future_predictions
optimal_windows = {}
for crop, req in crop_requirements.items():
    optimal_days = [(date, temp) for date, temp in zip(future_dates, future_predictions)
                    if req['optimal_min'] <= temp <= req['optimal_max']]
    if not optimal_days:
        continue
    # find continuous windows
    windows = []
    curr = [optimal_days[0]]
    for d, t in optimal_days[1:]:
        if (d - curr[-1][0]).days == 1:
            curr.append((d, t))
        else:
            windows.append(curr)
            curr = [(d, t)]
    windows.append(curr)
    best_window = max(windows, key=len)
    optimal_windows[crop] = {
        'start_date': best_window[0][0],
        'end_date': best_window[-1][0],
        'avg_temp': np.mean([t for _, t in best_window]),
        'duration_days': len(best_window)
    }

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(future_dates, future_predictions, label='Soil Temperature Forecast', linewidth=3, color='darkblue')
colors = plt.cm.Set3(np.linspace(0, 1, len(optimal_windows)))
for i, (crop, win) in enumerate(optimal_windows.items()):
    req = crop_requirements[crop]
    ax.axhspan(req['optimal_min'], req['optimal_max'], alpha=0.2, color=colors[i], label=f'{crop} Optimal Range')
    ax.axvspan(win['start_date'], win['end_date'], alpha=0.3, color=colors[i], label=f'{crop} Planting Window')

ax.set_title(f'90-Day Soil Temperature Forecast with Planting Windows\nSohna, Gurgaon Region ({present_date.year})',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Soil Temperature (°C)')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
plt.xticks(rotation=45)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.savefig('regional_planting_timetable.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------
# Recommendations CSV
# -----------------------
print("Generating detailed planting recommendations...")
recommendations = []
for crop, win in optimal_windows.items():
    req = crop_requirements[crop]
    recommendations.append({
        'Crop': crop,
        'Recommended Planting': f"{win['start_date'].strftime('%b %d')} - {win['end_date'].strftime('%b %d')}",
        'Window Length': f"{win['duration_days']} days",
        'Expected Soil Temp': f"{win['avg_temp']:.1f}°C",
        'Optimal Range': f"{req['optimal_min']}°C - {req['optimal_max']}°C",
        'Traditional Season': req['planting_season']
    })

pd.DataFrame(recommendations).to_csv('sohna_gurgaon_planting_recommendations.csv', index=False)

print("\nAnalysis complete! Created the following files for Sohna, Gurgaon:")
print("1. historical_trend_with_forecast.png")
print("2. combined_soil_air_temperature_analysis.png (NEW)")
print("3. seasonal_comparison.png")
print("4. regional_planting_timetable.png")
print("5. sohna_gurgaon_planting_recommendations.csv")