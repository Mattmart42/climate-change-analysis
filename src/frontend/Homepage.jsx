import React, { useState, useEffect, useRef, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const DEFAULT_LAT = 34.0522;
const DEFAULT_LNG = -118.2437;
const DEFAULT_POSITION = [DEFAULT_LAT, DEFAULT_LNG];

const aggregateLocationData = (dailyData, level) => {
  if (!dailyData || !dailyData.time || !dailyData.temperature_2m_mean || level === 'daily') {
    return { ...dailyData, aggregationLevel: 'daily' };
  }

  const averages = new Map();

  dailyData.time.forEach((dateStr, index) => {
    const temp = dailyData.temperature_2m_mean[index];
    if (temp === null || temp === undefined || isNaN(parseFloat(temp))) return;

    try {
        const date = new Date(dateStr);
        if (isNaN(date.getTime())) {
            console.warn(`Skipping invalid date string: ${dateStr}`);
            return;
        }

        let key;
        if (level === 'yearly') {
            key = date.getFullYear().toString();
        } else {
            const month = (date.getMonth() + 1).toString().padStart(2, '0');
            key = `${date.getFullYear()}-${month}`;
        }

        const current = averages.get(key) || { sum: 0, count: 0 };
        current.sum += parseFloat(temp);
        current.count += 1;
        averages.set(key, current);

    } catch (e) {
         console.error(`Error processing date: ${dateStr}`, e);
    }

  });

  const sortedKeys = Array.from(averages.keys()).sort();
  const aggregatedTime = sortedKeys;
  const aggregatedTemp = sortedKeys.map(key => {
    const { sum, count } = averages.get(key);
    return count > 0 ? (sum / count) : null;
  });

  return {
    time: aggregatedTime,
    temperature_2m_mean: aggregatedTemp,
    aggregationLevel: level
  };
};

function LocationMarker({ onPositionChange }) {
  const map = useMapEvents({
      click(e) {
          const { lat, lng } = e.latlng;
          onPositionChange([lat, lng]);
          map.flyTo(e.latlng, map.getZoom());
      },
  });
  return null;
}

const NationalChart = ({ histData, predData }) => {

    // --- Basic Validation ---
    const tempColName = histData ? Object.keys(histData).find(k => k.startsWith('AvgTemp_National')) : null;
  
    if (!histData || !histData.year || !tempColName || !histData[tempColName] || !histData.CO2_ppm) {
        return <p>Insufficient historical national data for chart.</p>;
    }
  
    // --- Process Historical Data ---
    const histYears = histData.year.map(Number);
    const histTemp = histData[tempColName];
    const histCO2 = histData.CO2_ppm;
    const tempUnit = tempColName.includes('_C') ? '°C' : '°F';
  
    let lastHistYear = null;
    let lastHistTemp = null;
    let lastHistCO2 = null;
    if (histYears.length > 0) {
        lastHistYear = histYears[histYears.length - 1];
        lastHistTemp = histTemp[histTemp.length - 1];
        lastHistCO2 = histCO2[histCO2.length - 1];
    }
  
    // --- Process Prediction Data ---
    let predYears = [];
    let predTemp = [];
    let predCO2 = [];
    let predTempUnit = tempUnit;
  
    const hasValidCO2Pred = predData?.co2 && typeof predData.co2 !== 'string' && !predData.co2.error;
    const hasValidTempPred = predData?.national_temp && typeof predData.national_temp !== 'string' && !predData.national_temp.error;
  
    if (hasValidCO2Pred || hasValidTempPred) {
        const rawPredYears = Object.keys(hasValidCO2Pred ? predData.co2 : predData.national_temp);
        predYears = rawPredYears.map(Number).sort((a, b) => a - b);
  
        if (hasValidCO2Pred) {
            predCO2 = predYears.map(year => predData.co2[year] ?? null);
        } else {
            predCO2 = predYears.map(() => null);
        }
  
        if (hasValidTempPred) {
            predTemp = predYears.map(year => predData.national_temp[year]?.value ?? null);
            const firstValidTempPred = predData.national_temp[predYears[0]];
            if (firstValidTempPred?.unit) {
                predTempUnit = firstValidTempPred.unit;
            }
        } else {
             predTemp = predYears.map(() => null);
        }
    }
  
    // --- Combine Labels ---
    const allYears = [...new Set([...histYears, ...predYears])]
                      .sort((a, b) => a - b)
                      .map(String);
  
    // --- Create Datasets ---
    const datasets = [];
  
    // Historical Temperature
    datasets.push({
        label: `National Avg Temp (${tempUnit})`,
        data: histTemp,
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        yAxisID: 'y',
        tension: 0.1,
        pointRadius: 3,
        data: allYears.map(year => {
            const index = histYears.indexOf(Number(year));
            return index !== -1 ? histTemp[index] : null;
        })
    });
  
    if (predTemp.length > 0 && lastHistYear !== null) {
       datasets.push({
            label: `Predicted Temp (${predTempUnit})`,
            data: allYears.map(year => {
                const yearNum = Number(year);
                if(yearNum === lastHistYear) return lastHistTemp;
                const index = predYears.indexOf(yearNum);
                return index !== -1 ? predTemp[index] : null;
            }),
            borderColor: 'rgb(135, 206, 250)',
            backgroundColor: 'rgba(135, 206, 250, 0.5)',
            borderDash: [5, 5],
            yAxisID: 'y',
            tension: 0.1,
            pointRadius: 2,
        });
    }
  
    // Historical CO2
     datasets.push({
        label: 'Global CO2 (ppm)',
        data: allYears.map(year => {
              const index = histYears.indexOf(Number(year));
              return index !== -1 ? histCO2[index] : null;
        }),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        yAxisID: 'y1',
        tension: 0.1,
        pointRadius: 3,
     });
  
     // Predicted CO2 (connects from last historical point)
    if (predCO2.length > 0 && lastHistYear !== null) {
         datasets.push({
            label: 'Predicted CO2 (ppm)',
            data: allYears.map(year => {
                  const yearNum = Number(year);
                  if(yearNum === lastHistYear) return lastHistCO2;
                  const index = predYears.indexOf(yearNum);
                  return index !== -1 ? predCO2[index] : null;
              }),
            borderColor: 'rgb(255, 182, 193)',
            backgroundColor: 'rgba(255, 182, 193, 0.5)',
            borderDash: [5, 5],
            yAxisID: 'y1',
            tension: 0.1,
            pointRadius: 2,
        });
    }
  
  
    // --- Chart Configuration ---
    const chartData = {
        labels: allYears,
        datasets: datasets,
    };
  
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        stacked: false,
        plugins: {
            title: {
                display: true,
                text: `National Avg Temp & Global CO2 Over Time ${predYears.length > 0 ? '(with Predictions)' : ''}`,
            },
            legend: {
                 position: 'top',
            },
            tooltip: {
                shared: true,
                mode: 'index',
                intersect: false,
            }
        },
        scales: {
            x: {
                title: { display: true, text: 'Year' }
            },
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: {
                    display: true,
                    text: `Avg Temp (${tempUnit})`,
                    color: 'rgb(54, 162, 235)',
                },
                ticks: {
                     color: 'rgb(54, 162, 235)',
                }
            },
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                title: {
                    display: true,
                    text: 'CO2 (ppm)',
                    color: 'rgb(255, 99, 132)',
                },
                ticks: {
                    color: 'rgb(255, 99, 132)',
                },
                grid: {
                    drawOnChartArea: false,
                },
            },
        },
        spanGaps: false,
    };
  
    return <Line options={options} data={chartData} />;
  };


const LocationChart = ({ data }) => {
  if (!data || !data.time || !data.temperature_2m_mean || data.time.length === 0) {
      return <p>Insufficient location data for chart.</p>;
  }

   const validIndices = data.temperature_2m_mean
       .map((temp, index) => (temp !== null && !isNaN(temp) ? index : -1))
       .filter(index => index !== -1);

   const labels = validIndices.map(index => data.time[index]);
   const temps = validIndices.map(index => data.temperature_2m_mean[index]);

   if (labels.length === 0) {
        return <p>No valid location temperature data points for chart.</p>;
   }


  const chartData = {
      labels: labels,
      datasets: [
          {
              label: 'Mean Temp (°C)',
              data: temps,
              borderColor: 'rgb(75, 192, 192)',
              backgroundColor: 'rgba(75, 192, 192, 0.5)',
              yAxisID: 'y',
              tension: 0.1
          },
      ],
  };

  let chartTitle = 'Location Mean Temperature';
  let timeUnit = 'Date';
  if (data.aggregationLevel === 'monthly') {
      chartTitle = 'Location Monthly Mean Temperature';
      timeUnit = 'Month (YYYY-MM)';
  } else if (data.aggregationLevel === 'yearly') {
      chartTitle = 'Location Yearly Mean Temperature';
      timeUnit = 'Year';
  }


   const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
          legend: {
              position: 'top',
          },
          title: {
              display: true,
              text: chartTitle,
          },
           tooltip: {
              mode: 'index',
              intersect: false,
          }
      },
       scales: {
          x: {
              title: { display: true, text: timeUnit }
          },
          y: {
              title: { display: true, text: 'Temperature (°C)' }
          }
      }
  };

   return <Line options={options} data={chartData} />;
};

export default function Homepage() {
  const [predictionYears, setPredictionYears] = useState(10);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [longitude, setLongitude] = useState(DEFAULT_LNG.toString());
  const [latitude, setLatitude] = useState(DEFAULT_LAT.toString());

  const [markerPosition, setMarkerPosition] = useState(DEFAULT_POSITION);
  const mapRef = useRef(null);
  const markerRef = useRef(null);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [resultsData, setResultsData] = useState(null);

  const handleStartDateChange = (event) => setStartDate(event.target.value);
  const handleEndDateChange = (event) => setEndDate(event.target.value);

  const handleMapClick = useCallback((newPosition) => {
    const [lat, lng] = newPosition;
    setMarkerPosition([lat, lng]);
    setLatitude(lat.toFixed(6));
    setLongitude(lng.toFixed(6));
  }, []);

  const handleLongitudeChange = (event) => {
    const newLngStr = event.target.value;
    setLongitude(newLngStr);
    const newLng = parseFloat(newLngStr);
    const currentLat = parseFloat(latitude);
    if (!isNaN(newLng) && !isNaN(currentLat) && newLng >= -180 && newLng <= 180) {
        setMarkerPosition([currentLat, newLng]);
    }
  };

  const handleLatitudeChange = (event) => {
    const newLatStr = event.target.value;
    setLatitude(newLatStr);
    const newLat = parseFloat(newLatStr);
    const currentLng = parseFloat(longitude);
    if (!isNaN(newLat) && !isNaN(currentLng) && newLat >= -90 && newLat <= 90) {
        setMarkerPosition([newLat, currentLng]);
    }
  };

  const handleSearch = async () => {
    if (!startDate || !endDate || !latitude || !longitude) {
        setError("Please fill in all date and coordinate fields.");
        return;
    }
    if (isNaN(parseFloat(latitude)) || isNaN(parseFloat(longitude))) {
        setError("Invalid latitude or longitude values.");
        return;
    }

    const start = new Date(startDate);
    const end = new Date(endDate);
    if (isNaN(start.getTime()) || isNaN(end.getTime()) || start > end) {
        setError("Invalid date range. Ensure Start Date is not after End Date.");
        return;
    }

    setIsLoading(true);
    setError(null);
    setResultsData(null);

    const requestData = {
        start_date: startDate,
        end_date: endDate,
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
        prediction_years: predictionYears,
    };

    try {
        const response = await fetch('http://localhost:5002/api/climate-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(requestData),
        });

        if (!response.ok) {
            let errorMsg = `HTTP error! Status: ${response.status}`;
            try {
                 const errorData = await response.json();
                 errorMsg = errorData.error || errorData.message || errorMsg;
            } catch (e) {
            }
            throw new Error(errorMsg);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(`Backend error: ${data.error}`);
        }

        // --- Data Aggregation Logic ---
        let processedLocationData = null;
        let aggregationLevel = 'daily';

        if (data.plot_data && data.plot_data.location_data) {
            const originalLocationData = data.plot_data.location_data;

            const diffTime = Math.abs(end - start);
            const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)) + 1;
            const diffMonths = (end.getFullYear() - start.getFullYear()) * 12 + (end.getMonth() - start.getMonth()) + (end.getDate() >= start.getDate() ? 0 : -1) ;

            if (diffMonths > 60) {
                aggregationLevel = 'yearly';
            } else if (diffDays > 60) {
                aggregationLevel = 'monthly';
            } else {
                aggregationLevel = 'daily';
            }
             console.log(`Date range: ${diffDays} days, ~${diffMonths} months. Aggregation level: ${aggregationLevel}`);

            processedLocationData = aggregateLocationData(originalLocationData, aggregationLevel);
        } else {
            console.log("No location data received from backend.");
        }

        // --- Set State with Processed Data ---
        const finalResults = {
            ...data,
            plot_data: {
                national_data: data.plot_data?.national_data,
                location_data: processedLocationData
            }
        };

        setResultsData(finalResults);
        console.log("Setting processed results state:", finalResults);

    } catch (err) {
        console.error("Fetch error:", err);
        setError(err.message || "Failed to fetch data from backend.");
    } finally {
        setIsLoading(false);
    }
  };

  useEffect(() => {
    if (mapRef.current && markerPosition) {
       const map = mapRef.current;
       const currentCenter = map.getCenter();
       if (Array.isArray(markerPosition) && markerPosition.length === 2) {
           if (Math.abs(currentCenter.lat - markerPosition[0]) > 0.0001 ||
               Math.abs(currentCenter.lng - markerPosition[1]) > 0.0001) {
               map.flyTo(markerPosition, map.getZoom());
           }
       }
    }
  }, [markerPosition]);

  return (
    <div>
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
         integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
         crossOrigin=""/>
      <div className='content'>
        
        {/* Input Section */}
        <div className='input-container'>
          <div className='map'>
            <MapContainer
                center={markerPosition || DEFAULT_POSITION}
                zoom={10}
                scrollWheelZoom={true}
                style={{ height: '100%', width: '100%' }}
                ref={mapRef}
             >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <LocationMarker onPositionChange={handleMapClick} />
              {Array.isArray(markerPosition) && markerPosition.length === 2 && (
                <Marker position={markerPosition} ref={markerRef}>
                  <Popup>
                    Lat: {markerPosition[0].toFixed(4)} <br />
                    Lng: {markerPosition[1].toFixed(4)}
                  </Popup>
                </Marker>
              )}
            </MapContainer>
          </div>
          <div className='inputs'>
            <div>
              <label htmlFor="start-date">Start Date: </label>
              <input
                id="start-date"
                className='input'
                type="date"
                value={startDate}
                onChange={handleStartDateChange}
                max={endDate || ''}
                />
            </div>
            <div>
              <label htmlFor="end-date">End Date: </label>
              <input
                id="end-date"
                className='input'
                type="date"
                value={endDate}
                onChange={handleEndDateChange}
                min={startDate || ''}
                />
            </div>
            <div>
              <label htmlFor="longitude">Longitude: </label>
              <input
                id="longitude"
                className='input'
                type="number"
                placeholder="-180 to 180"
                value={longitude}
                onChange={handleLongitudeChange}
                step="any"
                />
            </div>
            <div>
              <label htmlFor="latitude">Latitude: </label>
              <input
                id="latitude"
                className='input'
                type="number"
                placeholder="-90 to 90"
                value={latitude}
                onChange={handleLatitudeChange}
                step="any"
                />
            </div>
            <div>
            <label htmlFor="prediction-years">Prediction Years Ahead: </label>
            <input
                id="prediction-years"
                className='input'
                type="number"
                min="1"
                max="50"
                step="1"
                value={predictionYears}
                onChange={(e) => {
                    const val = parseInt(e.target.value);
                    setPredictionYears(isNaN(val) || val < 1 ? 1 : val);
                }}
            />
            </div>
          </div>
          <button className='search-button' onClick={handleSearch} disabled={isLoading}>
            {isLoading ? 'Searching...' : 'Search'}
          </button>
          {error && <div className="error-message">{error}</div>}
        </div>
        
        {/* Results Section */}
        <div className='results-container'>
          <h3>Analysis Results</h3>
          {isLoading && <div className="loading-indicator">Loading data...</div>}
          {resultsData && resultsData.analysis_summary && (
            <div className="analysis-summary">
                <h4>Summary:</h4>
                {resultsData.analysis_summary.split('\n').map((line, index) => (
                    <p key={index}>{line}</p>
                ))}
            </div>
          )}
          <div className='graph-area'>
             {resultsData && resultsData.plot_data ? (
                <>
                    {resultsData.plot_data.national_data && (
                        <div className="chart-container">
                            <NationalChart
                                histData={resultsData.plot_data.national_data}
                                predData={resultsData.predictions}
                            />
                        </div>
                    )}
                     {resultsData.plot_data.location_data && (
                        <div className="chart-container">
                             <LocationChart data={resultsData.plot_data.location_data} />
                        </div>
                    )}
                    {!resultsData.plot_data.national_data && !resultsData.plot_data.location_data && (
                        <p>No data available to display charts for the selected criteria.</p>
                    )}
                </>
             ) : (
                 !isLoading && <p>Graph(s) will appear here after searching.</p>
             )}
          </div>
        </div>
      </div>
    </div>
  )
}