import pandas as pd
import requests
import json
from io import StringIO
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import base64
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_INSTALLED = True
except ImportError:
    print("WARNING: statsmodels library not found. SARIMAX predictions will be skipped.")
    print("Install it using: pip install statsmodels")
    STATSMODELS_INSTALLED = False

NATIONAL_TEMP_CSV = 'climdiv_national_year.csv'
CO2_CSV = 'co2_concentration.csv'
API_WEATHER_JSON = 'api_weather_data.json'

def getLocation(location_name):
    """Fetches latitude and longitude for a given location name."""
    try:
        response = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=1&language=en&format=json", timeout=10)
        response.raise_for_status()
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return (result["name"], result["latitude"], result["longitude"])
        else:
            print(f"Warning: No geocoding results found for {location_name}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to get location data for {location_name}: {e}")
        return None

def getWeather(latitude, longitude, start_date, end_date):
    """Fetches daily weather data from Open-Meteo archive API."""
    if not all([latitude, longitude, start_date, end_date]):
        print("Error: Missing parameters for getWeather function.")
        return None
    try:
        api_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={latitude}&longitude={longitude}&"
            f"start_date={start_date}&end_date={end_date}&"
            f"daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
            f"apparent_temperature_mean,precipitation_sum,precipitation_hours&"
            f"timezone=auto"
        )
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error getting weather data from Open-Meteo: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from Open-Meteo.")
        return None

# --- Data Loading and Preprocessing ---

def load_national_temp_data(filepath=NATIONAL_TEMP_CSV):
    """Loads and preprocesses the national temperature data."""
    try:
        df = pd.read_csv(filepath)
        df.columns = [col.lower() for col in df.columns]
        if 'year' not in df.columns:
             raise ValueError("CSV file must contain a 'year' column.")
        if 'temp' in df.columns:
            df = df[['year', 'temp']].copy()
            df.rename(columns={'temp': 'AvgTemp_National'}, inplace=True)
        elif 'tempc' in df.columns:
             df = df[['year', 'tempc']].copy()
             df.rename(columns={'tempc': 'AvgTemp_National_C'}, inplace=True)
             print("Warning: Using Celsius temperature column ('tempc') as national average.")
        else:
            raise ValueError("CSV file must contain a 'temp' or 'tempc' column.")

        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

        temp_col_name = [col for col in df.columns if 'AvgTemp' in col][0]
        df[temp_col_name] = pd.to_numeric(df[temp_col_name], errors='coerce')
        df = df.dropna(subset=['year', temp_col_name])

        print(f"Loaded national temperature data: {df.shape[0]} rows")
        print(f"National temp 'year' dtype: {df['year'].dtype}")
        return df

    except FileNotFoundError:
        print(f"Error: National temperature file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading national temperature data: {e}")
        return None

def load_co2_data(filepath=CO2_CSV):
    """Loads and preprocesses the CO2 concentration data."""
    try:
        df = pd.read_csv(filepath)
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'year' in col_lower:
                rename_map[col] = 'year'
            elif 'co2' in col_lower:
                 rename_map[col] = 'CO2_ppm'

        if 'year' not in rename_map.values() or 'CO2_ppm' not in rename_map.values():
             raise ValueError("CO2 CSV must contain columns identifiable as 'Year' and 'CO2'.")

        df.rename(columns=rename_map, inplace=True)
        df = df[['year', 'CO2_ppm']]
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
        df['CO2_ppm'] = pd.to_numeric(df['CO2_ppm'], errors='coerce')
        df = df.dropna(subset=['year', 'CO2_ppm'])

        print(f"Loaded CO2 data: {df.shape[0]} rows")
        print(f"CO2 'year' dtype: {df['year'].dtype}")
        return df
    except FileNotFoundError:
        print(f"Error: CO2 file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading CO2 data: {e}")
        return None

def process_api_weather_data(api_data):
    """Processes the JSON response from Open-Meteo into a DataFrame."""
    if not api_data or 'daily' not in api_data or not api_data['daily']:
        print("Warning: No valid daily weather data received from API.")
        return None
    try:
        df = pd.DataFrame(api_data['daily'])
        df['time'] = pd.to_datetime(df['time'])
        for col in df.columns:
            if col != 'time':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Processed API weather data: {df.shape[0]} days")
        return df.dropna(subset=['time'])
    except Exception as e:
        print(f"Error processing API weather data: {e}")
        return None

# --- Analysis and Prediction ---

def analyze_trends(df_merged, df_location_weather):
    """Performs trend analysis on merged national data and location data."""
    analysis_summary = []
    trends = {}

    # --- National Trend Analysis (using lowercase 'year') ---
    if df_merged is not None and not df_merged.empty:
        temp_col_name = [col for col in df_merged.columns if 'AvgTemp' in col]
        if not temp_col_name:
            analysis_summary.append("Could not find temperature column in merged data.")
        else:
            temp_col_name = temp_col_name[0]
            df_merged = df_merged.dropna(subset=['year', temp_col_name, 'CO2_ppm'])
            if not df_merged.empty:
                # National Temperature Trend
                X_nat_temp = df_merged[['year']]
                y_nat_temp = df_merged[temp_col_name]
                model_nat_temp = LinearRegression()
                model_nat_temp.fit(X_nat_temp, y_nat_temp)
                nat_temp_trend = model_nat_temp.coef_[0]
                temp_unit = '(°F)' if 'AvgTemp_National' == temp_col_name else '(°C)'
                analysis_summary.append(f"National Avg Temp Trend ({df_merged['year'].min()}-{df_merged['year'].max()}): {nat_temp_trend:.3f} degrees {temp_unit}/year.") # Use 'year'
                trends['national_temp_slope'] = nat_temp_trend
                trends['national_temp_intercept'] = model_nat_temp.intercept_

                # CO2 Trend
                X_co2 = df_merged[['year']]
                y_co2 = df_merged['CO2_ppm']
                model_co2 = LinearRegression()
                model_co2.fit(X_co2, y_co2)
                co2_trend = model_co2.coef_[0]
                analysis_summary.append(f"Global CO2 Trend ({df_merged['year'].min()}-{df_merged['year'].max()}): {co2_trend:.3f} ppm/year.")
                trends['co2_slope'] = co2_trend
                trends['co2_intercept'] = model_co2.intercept_

                # Correlation
                correlation = df_merged[temp_col_name].corr(df_merged['CO2_ppm'])
                analysis_summary.append(f"Correlation between National Avg Temp and CO2: {correlation:.3f}.")
            else:
                analysis_summary.append("No overlapping data after dropping NAs for national trend analysis.")
    else:
        analysis_summary.append("National/CO2 data not available or merge failed for trend analysis.")

    # --- Specific Location Trend Analysis ---
    if df_location_weather is not None and not df_location_weather.empty:
        df_location_weather = df_location_weather.sort_values('time').dropna(subset=['time', 'temperature_2m_mean'])
        if df_location_weather.shape[0] > 1:
            start_day = df_location_weather['time'].min()
            df_location_weather['days_since_start'] = (df_location_weather['time'] - start_day).dt.days

            X_loc_temp = df_location_weather[['days_since_start']]
            y_loc_temp = df_location_weather['temperature_2m_mean']
            model_loc_temp = LinearRegression()
            model_loc_temp.fit(X_loc_temp, y_loc_temp)
            loc_temp_trend_per_day = model_loc_temp.coef_[0]
            loc_temp_trend_per_year = loc_temp_trend_per_day * 365.25
            analysis_summary.append(f"Location Avg Temp Trend ({df_location_weather['time'].min().date()} to {df_location_weather['time'].max().date()}): {loc_temp_trend_per_year:.3f} degrees (°C)/year.")
            trends['location_temp_slope_per_year'] = loc_temp_trend_per_year
            trends['location_temp_intercept'] = model_loc_temp.intercept_
            trends['location_start_day'] = start_day.isoformat()
        else:
             analysis_summary.append("Not enough data points for location temperature trend analysis.")
    else:
        analysis_summary.append("Location weather data not available for trend analysis.")

    return "\n".join(analysis_summary), trends

def predict_future_models(df_merged_all_years, trends, years_ahead=10):
    """
    Predicts future CO2 (Polynomial Regression trained on all data) and
    National Temperature (SARIMAX model trained on all data with CO2 as
    exogenous regressor) based on historical merged data.
    """
    predictions = {"co2": {}, "national_temp": {}}
    if df_merged_all_years is None or df_merged_all_years.empty:
        print("Warning: Cannot run predictions, merged historical data is empty.")
        return {"message": "Cannot run predictions, historical data unavailable or empty."}

    temp_col_name = trends.get('temp_col_used',
                               next((col for col in df_merged_all_years.columns if 'AvgTemp' in col), None))
    co2_col_name = 'CO2_ppm'

    required_cols = ['year', co2_col_name]
    if temp_col_name:
        required_cols.append(temp_col_name)
    else:
         print("Warning: Cannot predict temperature, temperature column not identified.")
         predictions["national_temp"] = {"error": "Temperature column not found in data."}

    if not all(col in df_merged_all_years.columns for col in required_cols):
         print(f"Warning: Missing columns in merged data needed for prediction. Required: {required_cols}")
         missing_cols_msg = f"Cannot run predictions, missing required columns ({required_cols})."
         if not temp_col_name and 'CO2_ppm' in df_merged_all_years.columns:
             missing_cols_msg = "Cannot predict temperature (missing temp column), will attempt CO2 prediction."
         elif temp_col_name and 'CO2_ppm' not in df_merged_all_years.columns:
              missing_cols_msg = "Cannot predict CO2 or Temp (missing CO2 column)."

         if 'CO2_ppm' not in df_merged_all_years.columns:
            return {"message": missing_cols_msg}
         else:
             predictions["national_temp"] = {"error": "Temperature column not found or CO2 missing."}

    if 'year' not in df_merged_all_years.columns:
         print("Error: 'year' column missing from merged data.")
         return {"message": "Cannot run predictions, 'year' column is missing."}

    df_train_full = df_merged_all_years[required_cols].dropna().copy()

    min_rows_required_ts = 10
    if df_train_full.shape[0] < min_rows_required_ts:
        print(f"Warning: Insufficient data ({df_train_full.shape[0]} rows) after dropping NAs for time series prediction.")
        predictions["national_temp"] = {"error": f"Insufficient data ({df_train_full.shape[0]} rows) for Temp prediction."}
    elif not STATSMODELS_INSTALLED:
         predictions["national_temp"] = {"error": "statsmodels library not installed, cannot run SARIMAX."}


    last_hist_year = df_train_full['year'].max()
    future_years = np.arange(last_hist_year + 1, last_hist_year + 1 + years_ahead)
    future_years_reshape = future_years.reshape(-1, 1)

    # --- 1. Predict Future CO2 (Polynomial Regression Degree 2 - Trained on ALL data) ---
    pred_co2_future = None
    try:
        if df_train_full.shape[0] < 3:
             raise ValueError(f"Not enough data ({df_train_full.shape[0]} rows) for CO2 polynomial fit.")

        X_co2_hist = df_train_full[['year']]
        y_co2_hist = df_train_full[co2_col_name]

        poly_features_co2 = PolynomialFeatures(degree=2, include_bias=False)
        X_co2_hist_poly = poly_features_co2.fit_transform(X_co2_hist)

        model_co2_poly = LinearRegression()
        model_co2_poly.fit(X_co2_hist_poly, y_co2_hist)

        future_years_poly_co2 = poly_features_co2.transform(future_years_reshape)
        pred_co2_future = model_co2_poly.predict(future_years_poly_co2)
        pred_co2_future = np.maximum(pred_co2_future, 0)

        predictions['co2'] = {int(year): round(float(co2), 2) for year, co2 in zip(future_years, pred_co2_future)}
        print(f"CO2 predictions generated for {len(future_years)} years using Polynomial Regression (d=2).")

    except Exception as e:
        print(f"Error during CO2 prediction: {e}")
        predictions['co2'] = {"error": f"Failed to predict CO2: {e}"}

    # --- 2. Predict Future National Temperature (SARIMAX) ---
    can_predict_temp = (STATSMODELS_INSTALLED and
                        temp_col_name and
                        pred_co2_future is not None and
                        df_train_full.shape[0] >= min_rows_required_ts)

    if can_predict_temp:
        try:
            print(f"Preparing SARIMAX model for temperature ({temp_col_name})...")
            df_train_indexed = df_train_full.set_index('year')
            temp_series = df_train_indexed[temp_col_name]
            co2_series_hist = df_train_indexed[[co2_col_name]]

            model_order = (1, 1, 1)
            seasonal_order = (0, 0, 0, 0)

            model_temp_sarimax = SARIMAX(endog=temp_series,
                                         exog=co2_series_hist,
                                         order=model_order,
                                         seasonal_order=seasonal_order,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False,
                                         trend=None)

            print("Fitting SARIMAX model...")
            results_sarimax = model_temp_sarimax.fit(disp=False, maxiter=200)
            print("SARIMAX model fitting complete.")

            future_co2_df = pd.DataFrame(pred_co2_future,
                                         index=future_years,
                                         columns=[co2_col_name])

            print("Generating SARIMAX forecast...")
            forecast_obj = results_sarimax.get_forecast(steps=years_ahead, exog=future_co2_df)

            pred_temp_future = forecast_obj.predicted_mean.values
            print("SARIMAX forecast complete.")

            temp_unit = '°F' if 'AvgTemp_National' == temp_col_name else '°C'
            predictions['national_temp'] = {
                int(year): {'value': round(float(temp), 2), 'unit': temp_unit}
                for year, temp in zip(future_years, pred_temp_future)
            }


        except Exception as e:
            import traceback
            print(f"Error during SARIMAX Temperature prediction: {e}")
            print(traceback.format_exc())
            predictions['national_temp'] = {"error": f"Failed to predict National Temp using SARIMAX: {e}"}

    elif temp_col_name and pred_co2_future is not None:
        if "error" not in predictions["national_temp"]:
             predictions["national_temp"] = {"error": "Skipped due to insufficient data or missing library."}


    return predictions

# --- Visualization ---

def create_plots(df_merged, df_location_weather, trends):
    """Generates plots and returns them as base64 encoded strings."""
    plots = {}
    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Plot 1: National Temp & CO2 vs Year ---
    if df_merged is not None and not df_merged.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Plot National Temperature
        ax1.plot(df_merged['year'], df_merged['temp'], label='National Avg Temp (°F)', color='tab:blue', marker='o', linestyle='-', markersize=4)
        ax1.set_xlabel('year')
        ax1.set_ylabel('National Avg Temp (°F)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Plot CO2
        ax2.plot(df_merged['year'], df_merged['CO2 (ppm)'], label='Global CO2 (ppm)', color='tab:red', marker='x', linestyle='--', markersize=4)
        ax2.set_ylabel('Global CO2 (ppm)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        if 'national_temp_slope' in trends:
             years = df_merged['year']
             trend_line_temp = trends['national_temp_intercept'] + trends['national_temp_slope'] * years
             ax1.plot(years, trend_line_temp, color='blue', linestyle=':', label='Nat Temp Trend')
        if 'co2_slope' in trends:
             years = df_merged['year']
             trend_line_co2 = trends['co2_intercept'] + trends['co2_slope'] * years
             ax2.plot(years, trend_line_co2, color='red', linestyle=':', label='CO2 Trend')

        ax1.set_title('National Avg Temperature and Global CO2 Concentration Over Time')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        fig1.tight_layout()
        buf = StringIO()
        fig1.savefig(buf, format='png')
        plots['national_plot'] = base64.b64encode(buf.getvalue().encode('utf-8')).decode('utf-8')
        plt.close(fig1)

    # --- Plot 2: Location Temperature vs Time ---
    if df_location_weather is not None and not df_location_weather.empty:
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_location_weather['time'], df_location_weather['temperature_2m_mean'], label='Mean Daily Temp (°C)', color='tab:green', marker='.', linestyle='-')

        if 'location_temp_slope_per_year' in trends and df_location_weather.shape[0] > 1:
             start_day_ts = pd.to_datetime(trends['location_start_day'])
             days = (df_location_weather['time'] - start_day_ts).dt.days
             trend_line_loc = trends['location_temp_intercept'] + (trends['location_temp_slope_per_year'] / 365.25) * days
             ax.plot(df_location_weather['time'], trend_line_loc, color='darkgreen', linestyle=':', label='Location Temp Trend')

        ax.set_xlabel('Date')
        ax.set_ylabel('Mean Daily Temperature (°C)')
        ax.set_title(f'Mean Daily Temperature for Location ({df_location_weather["time"].min().date()} to {df_location_weather["time"].max().date()})')
        ax.legend()
        plt.xticks(rotation=45)
        fig2.tight_layout()
        buf = StringIO()
        fig2.savefig(buf, format='png')
        plots['location_plot'] = base64.b64encode(buf.getvalue().encode('utf-8')).decode('utf-8')
        plt.close(fig2)

    return plots

# --- Main Processing Function ---

def process_climate_data(start_date_str, end_date_str, latitude, longitude, years_ahead=10):
    """Main function to load, process, analyze, and visualize climate data."""
    results = {
        "inputs": {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "latitude": latitude,
            "longitude": longitude,
            "years_predicted": years_ahead
        },
        "analysis_summary": "Analysis did not run.",
        "plot_data": {},
        "predictions": {"message": "Predictions did not run."},
        "error": None
    }

    try:
        # --- Load Static Data ---
        df_national_temp = load_national_temp_data()
        df_co2 = load_co2_data()

        # --- Fetch API Data ---
        try:
            valid_latitude = float(latitude)
            valid_longitude = float(longitude)
        except (ValueError, TypeError):
            print(f"Invalid latitude ({latitude}) or longitude ({longitude}). Skipping weather API call.")
            valid_latitude, valid_longitude = None, None
            results["analysis_summary"] = "Skipped location analysis due to invalid coordinates."

        api_weather_raw = None
        df_location_weather = None
        if valid_latitude is not None and valid_longitude is not None:
            api_weather_raw = getWeather(valid_latitude, valid_longitude, start_date_str, end_date_str)
            df_location_weather = process_api_weather_data(api_weather_raw)
        else:
             df_location_weather = None

        # --- Merge National Data ---
        df_merged = None
        df_merged_filtered = None
        if df_national_temp is not None and df_co2 is not None:
            df_merged = pd.merge(df_national_temp, df_co2, on='year', how='inner')
            try:
                start_year = pd.to_datetime(start_date_str).year
                end_year = pd.to_datetime(end_date_str).year
                df_merged_filtered = df_merged[
                    (df_merged['year'] >= start_year) & (df_merged['year'] <= end_year)
                ].copy()
                print(f"Filtered merged data ({start_year}-{end_year}): {df_merged_filtered.shape[0]} rows")
            except Exception as date_e:
                 print(f"Warning: Could not parse input dates ('{start_date_str}', '{end_date_str}') to filter national data by year. Using all merged data for plots/analysis. Error: {date_e}")
                 df_merged_filtered = df_merged.copy()
        else:
            print("Warning: Could not merge national temperature and CO2 data.")

        # --- Analyze Trends ---
        analysis_summary, trends = analyze_trends(df_merged_filtered, df_location_weather)
        results["analysis_summary"] = analysis_summary

        # --- Predict Future (Simple Extrapolation) ---
        print(f"Generating predictions for {years_ahead} years ahead...")
        predictions = predict_future_models(df_merged, trends, years_ahead=years_ahead)
        results["predictions"] = predictions

        # --- Create Plots ---
        plot_data_json = {}
        if df_merged_filtered is not None and not df_merged_filtered.empty:
             df_plot_nat = df_merged_filtered.copy()
             for col in df_plot_nat.columns:
                 if pd.api.types.is_numeric_dtype(df_plot_nat[col]):
                     df_plot_nat[col] = df_plot_nat[col].round(3)
             plot_data_json['national_data'] = df_plot_nat.to_dict(orient='list')

        if df_location_weather is not None and not df_location_weather.empty:
             df_plot_loc = df_location_weather.copy()
             df_plot_loc['time'] = df_plot_loc['time'].dt.strftime('%Y-%m-%d')
             for col in df_plot_loc.columns:
                  if col != 'time' and pd.api.types.is_numeric_dtype(df_plot_loc[col]):
                      df_plot_loc[col] = df_plot_loc[col].astype(float).round(3).where(pd.notnull(df_plot_loc[col]), None)

             plot_data_json['location_data'] = df_plot_loc.to_dict(orient='list')

        results["plot_data"] = plot_data_json

    except Exception as e:
        import traceback
        print(f"An unexpected error occurred in process_climate_data: {e}")
        print(traceback.format_exc())
        results["error"] = f"An unexpected server error occurred: {e}"

    def default_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    return json.dumps(results, indent=4, default=default_serializer)