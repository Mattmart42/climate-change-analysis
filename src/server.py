import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

try:
    from climate_backend_py_v1 import process_climate_data
    backend_loaded = True
except ImportError:
    print("Error: Could not import 'process_climate_data'. Make sure 'climate_backend_py_v1.py' is in the same directory.")
    backend_loaded = False
    def process_climate_data(*args, **kwargs):
        return json.dumps({"error": "Backend processing function not loaded."})

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/climate-data', methods=['POST'])
def get_climate_data():
    """
    API endpoint to process climate data requests.
    Expects JSON payload with: start_date, end_date, latitude, longitude
    Optionally accepts: prediction_years (defaults to 10)
    """
    print(f"[{datetime.datetime.now()}] Received request at /api/climate-data")

    if not request.is_json:
        print("Error: Request body is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    print(f"Request data: {data}")

    start_date = data.get('start_date')
    end_date = data.get('end_date')
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if not all([start_date, end_date]):
        print("Error: Missing start_date or end_date")
        return jsonify({"error": "Missing required parameters: start_date, end_date"}), 400
    if latitude is None or longitude is None:
         print("Error: Missing latitude or longitude")
         return jsonify({"error": "Missing required parameters: latitude, longitude"}), 400

    try:
        years_param = data.get('prediction_years', 10)
        prediction_years_int = int(years_param)

        min_pred_years = 1
        max_pred_years = 50
        if prediction_years_int < min_pred_years:
            print(f"Warning: Received prediction_years ({prediction_years_int}) < {min_pred_years}. Clamping to {min_pred_years}.")
            prediction_years_int = min_pred_years
        elif prediction_years_int > max_pred_years:
            print(f"Warning: Received prediction_years ({prediction_years_int}) > {max_pred_years}. Clamping to {max_pred_years}.")
            prediction_years_int = max_pred_years

    except (ValueError, TypeError):
        default_years = 10
        print(f"Warning: Invalid value received for prediction_years ('{years_param}'). Defaulting to {default_years}.")
        prediction_years_int = default_years

    if not backend_loaded:
         print("Error: Backend module not loaded.")
         error_json = process_climate_data()
         return Response(error_json, status=500, mimetype='application/json')

    try:
        print(f"Calling process_climate_data with years_ahead={prediction_years_int}")
        result_json_string = process_climate_data(
            start_date_str=start_date,
            end_date_str=end_date,
            latitude=latitude,
            longitude=longitude,
            years_ahead=prediction_years_int
        )

        print("Processing successful, sending response.")
        return Response(result_json_string, status=200, mimetype='application/json')

    except Exception as e:
        import traceback
        print(f"Error during processing call: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred during processing: {e}"}), 500

if __name__ == '__main__':
    import datetime
    print(f"[{datetime.datetime.now()}] Starting Flask server...")
    app.run(host='0.0.0.0', port=5002, debug=True)