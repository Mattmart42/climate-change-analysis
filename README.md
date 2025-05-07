# Climate Data Analysis and Prediction Tool

## Overview

This is a full-stack web application designed to fetch, analyze, visualize, and predict climate trends. It integrates historical US national temperature data, global CO2 concentrations, and location-specific weather data fetched from the Open-Meteo API. Users can interactively select locations via a map, choose date ranges, specify the number of years for predictions, and view historical trends alongside model-based future forecasts for temperature and CO2.

The project includes:
* A React frontend for user interaction and visualization (using Leaflet and Chart.js).
* A Python Flask backend API to handle data processing, analysis, and predictions.
* Core Python logic for data loading, cleaning, trend analysis (linear regression), and time series forecasting (Polynomial Regression for CO2, SARIMAX for National Temperature).
* An optional Python script for scraping historical CO2 data from Statista (requires login/specific network).

## Features

* **Interactive Map:** Select analysis location using Leaflet map clicks or by entering latitude/longitude coordinates.
* **Custom Date Range:** Choose start and end dates for historical data analysis and visualization.
* **Adjustable Predictions:** Specify the number of years (1-50) into the future for predictions.
* **National Trends:** Displays historical US National Average Temperature and Global CO2 concentration.
* **Location Trends:** Displays historical mean temperature for the selected location, with dynamic aggregation (daily/monthly/yearly) based on the selected date range duration.
* **Trend Analysis:** Provides a summary including calculated linear trends for temperature and CO2, and their correlation.
* **Forecasting:**
    * Predicts future Global CO2 levels using Polynomial Regression (degree 2).
    * Predicts future US National Average Temperature using a SARIMAX time series model incorporating predicted CO2 as an external regressor.
* **Integrated Visualization:** Displays historical data and future predictions on the same chart for comparison, with distinct styling for forecasts.
* **Data Scraping (Optional):** Includes a script (`data_prep_script.py`) to scrape historical CO2 data from Statista using Playwright.

## Prerequisites

* Python 3.7+
* Node.js (v16+) and npm (or yarn)
* Access to a terminal or command prompt
* Web Browser
* *(Optional: For CO2 Scraper)* Access to Statista (requires login or specific network like school WiFi).

## Installation & Setup

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/Mattmart42/climate-change-analysis
    cd climate-change-analysis
    ```

2.  **Backend Setup:**
    ```bash
    cd src/backend
    # Create and activate a virtual environment (recommended)
    python -m venv venv
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate

    # Install Python dependencies
    pip install Flask Flask-Cors pandas requests numpy scikit-learn statsmodels

    # --- Data Files ---
    # Ensure 'climdiv_national_year.csv' is present in the 'backend' directory.
    # Obtain or generate 'co2_concentration.csv'. You can:
    #   a) Provide your own CSV file with 'Year' and 'CO2' columns.
    #   b) Use the optional scraper (see step 4 below).
    ```

3.  **Frontend Setup:**
    ```bash
    cd ../frontend
    # Install Node dependencies
    npm install
    # or: yarn install
    ```

4.  **(Optional) CO2 Scraper Setup & Usage:**
    * Navigate to the directory containing `data_prep_script.py` (e.g., project root or `backend`).
    * Ensure the backend virtual environment is activated.
    * Install scraper dependencies: `pip install playwright beautifulsoup4`
    * Install Playwright browsers: `playwright install`
    * **Important:** Ensure you meet the login/network requirements for Statista.
    * Run the scraper (this may take 20-30 seconds):
        ```bash
        python ../backend/scraper.py --scrape
        ```
    * This should generate/update `co2_concentration.csv` in the `backend` directory.

## Running the Application

1.  **Start the Backend Server:**
    * Open a terminal in the `backend` directory.
    * Activate the virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`).
    * Run the Flask server:
        ```bash
        python server.py
        ```
    * The server should start, typically listening on `http://localhost:5002`. Keep this terminal open.

2.  **Start the Frontend Application:**
    * Open a *new* terminal in the `frontend` directory.
    * Run the React development server:
        ```bash
        npm run dev
        # or: yarn start
        ```
    * This should automatically open the application in your default web browser, usually at `http://localhost:3000`. If not, navigate to that URL manually.

3.  **Interact:** Use the map, date inputs, and prediction years input. Click "Analyze Climate Data" to fetch and display results.

## Data Sources

* **US National Average Temperature:** `climdiv_national_year.csv` (Source: Ideally specify source, e.g., NOAA National Centers for Environmental Information - Climate Divisional Data)
* **Global CO2 Concentration:** `co2_concentration.csv` (Generated via Statista scraper / Provided File. Original data often sourced from NOAA's Earth System Research Laboratories - Global Monitoring Laboratory)
* **Location-Specific Weather:** Open-Meteo Historical Weather API (`https://open-meteo.com/`)
