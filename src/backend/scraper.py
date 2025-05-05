import sys
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests
import json
import csv

def default_function():
    scrape_function()

    api_weather_data = getWeather("New York", "2023-01-01", "2023-01-31")
    if api_weather_data:
        with open('api_weather_data.json', 'w') as json_file:
            json.dump(api_weather_data, json_file, indent=4)
        print(api_weather_data)
    else:
        print("Failed to get weather data")
    
    static_function()

def scrape_function():
    URL = "https://www.statista.com/statistics/1091926/atmospheric-concentration-of-co2-historic/"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        try:
            print("Loading page...")
            page.goto(URL, wait_until="networkidle")

            print("Handling cookies...")
            try:
                page.locator("#onetrust-reject-all-handler").click(timeout=3000)
            except:
                print("No cookie popup found")

            print("Opening table view...")
            page.locator('button[aria-label="Settings"]').click()
            page.locator('button[aria-label="Display table"]').click()
            
            print("Waiting for table content...")
            page.wait_for_function("""() => {
                return document.querySelector('.dataTable tbody tr');
            }""", timeout=15000)
            
            print("Parsing table content...")
            html = page.inner_html('.dataTable')
            soup = BeautifulSoup(html, 'html.parser')
            
            data = []
            for row in soup.select('tbody tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    data.append({
                        'Year': cells[0].get_text(strip=True),
                        'CO2 (ppm)': cells[1].get_text(strip=True)
                    })
            
            while True:
                next_button = page.locator('button[aria-label="Show next page"]:not([disabled])')
                if not next_button.count():
                    break
                
                next_button.click()
                page.wait_for_timeout(2000)
                
                html = page.inner_html('.dataTable')
                soup = BeautifulSoup(html, 'html.parser')
                for row in soup.select('tbody tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        new_entry = {
                            'Year': cells[0].get_text(strip=True),
                            'CO2 (ppm)': cells[1].get_text(strip=True)
                        }
                        if new_entry not in data:
                            data.append(new_entry)
            
            df = pd.DataFrame(data)
            df.to_csv("co2_concentration.csv", index=False)
            print(f"Saved {len(df)} rows.")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            browser.close()

def static_function():
    print("#-----Scraping-results-----#\n")
    with open('co2_concentration.csv', 'r') as file:
      reader = csv.reader(file)
      header = next(reader)
      print(header)
      for i, row in enumerate(reader):
          if i < 4:
              print(row)
          else:
              break
    print("\n")

    print("#-----API-call-results-----#\n")
    with open('api_weather_data.json', 'r') as file:
        data = json.load(file)

    daily_data = data.get('daily', {})
    time_list = daily_data.get('time', [])
    mean_temp_list = daily_data.get('temperature_2m_mean', [])
    max_temp_list = daily_data.get('temperature_2m_max', [])
    min_temp_list = daily_data.get('temperature_2m_min', [])
    precip_list = daily_data.get('precipitation_sum', [])

    print(f"{'Date':<12} {'Mean Temp':>10} {'Max Temp':>10} {'Min Temp':>10} {'Precipitation':>12}")
    print("-" * 60)

    for i in range(min(5, len(time_list))):
        date = time_list[i]
        mean_temp = mean_temp_list[i]
        max_temp = max_temp_list[i]
        min_temp = min_temp_list[i]
        precip = precip_list[i]
        
        print(f"{date:<12} {mean_temp:>10.1f} {max_temp:>10.1f} {min_temp:>10.1f} {precip:>12.1f}")
        print("\n")

    print("#------Static-results------#\n")
    with open('climdiv_national_year.csv', 'r') as file:
      reader = csv.reader(file)
      header = next(reader)
      print(header)
      for i, row in enumerate(reader):
          if i < 4:
              print(row)
          else:
              break
    print("\n")

def getLocation(location):
    response = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json")
    if response.status_code == 200:
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return (result["name"], result["latitude"], result["longitude"])
    print(f"Error: Failed to get location data for {location}")
    return None
        
def getWeather(location, start_date, end_date):
    location_data = getLocation(location)
    if not location_data:
        return None
        
    name, lat, lon = location_data
    response = requests.get(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum")
    
    if response.status_code == 200:
        data = response.json()
        return {
            "daily": data.get("daily", {})
        }
    else:
        print(f"Error getting weather data: {response.status_code}")
        return None

if __name__ == '__main__':
    if len(sys.argv) == 1: # pass no arguments to the command line
        default_function()

    elif sys.argv[1] == '--scrape': # pass '--scrape' to the command line
        scrape_function()

    elif sys.argv[1] == '--static': # pass '--static' to the command line
        static_function()