import os
import ee
import numpy as np
import datetime
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Inisialisasi kredensial Google Earth Engine
google_credentials = "/etc/secrets/ee-ulikhasanah16-743b3ec3e985.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

try:
    service_account = "ee-ulikhasanah16-743b3ec3e985@developer.gserviceaccount.com"
    credentials = ee.ServiceAccountCredentials(service_account, google_credentials)
    ee.Initialize(credentials)
    logging.info("Google Earth Engine initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Earth Engine: {e}")
    raise

# Load model
try:
    catboost_model = joblib.load("catboost_chlor_a.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

# Define satellite data sources
SATELLITES = {"Sentinel-2": "COPERNICUS/S2_SR_HARMONIZED"}
BANDS = {"Sentinel-2": {"Red": "B4", "NIR": "B8", "SWIR1": "B11", "SWIR2": "B12", "Blue": "B2", "Green": "B3"}}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Location(BaseModel):
    lat: float
    lon: float

def get_nearest_data(lat, lon, collection_id, bands):
    try:
        point = ee.Geometry.Point(lon, lat)
        start_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(point)
            .filterDate(ee.Date(start_date).advance(-30, 'day'), ee.Date(start_date))
            .sort("system:time_start", False)
        )
        image = collection.first()
        if image is not None and image.getInfo():
            date_info = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            data = image.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
            return {band: data.get(bands[band], None) for band in bands}, date_info
    except Exception as e:
        logging.error(f"Failed to retrieve data from Earth Engine: {e}")
    return None, None

def get_nearest_sst(lat, lon):
    try:
        point = ee.Geometry.Point(lon, lat)
        collection = (
            ee.ImageCollection("NOAA/CDR/OISST/V2_1")
            .filterBounds(point)
            .filterDate(ee.Date(datetime.datetime.utcnow().strftime('%Y-%m-%d')).advance(-30, 'day'), ee.Date(datetime.datetime.utcnow().strftime('%Y-%m-%d')))
            .sort("system:time_start", False)
        )
        image = collection.first()
        if image is not None and image.getInfo():
            date_info = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            data = image.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
            return data.get("sst", None), date_info
    except Exception as e:
        logging.error(f"Failed to retrieve SST data: {e}")
    return None, None

def calculate_ndci(red, nir):
    if red is None or nir is None:
        return None
    return (nir - red) / (nir + red) if (nir + red) != 0 else None

def process_request(lat, lon):
    data_sources = {}
    dates = {}
    
    for sat in ["Sentinel-2"]:
        data, date = get_nearest_data(lat, lon, SATELLITES[sat], BANDS[sat])
        if data:
            data_sources[sat] = data
            dates[sat] = date
    
    sst_value, sst_date = get_nearest_sst(lat, lon)
    
    selected_data = next((data_sources[sat] for sat in data_sources if data_sources[sat]), None)
    
    if not selected_data:
        raise HTTPException(status_code=404, detail=f"No satellite data available for location ({lat}, {lon})")
    
    features = {
        "latitude": lat,
        "longitude": lon,
        "SWIR1": selected_data.get("SWIR1"),
        "SWIR2": selected_data.get("SWIR2"),
        "Blue": selected_data.get("Blue"),
        "Green": selected_data.get("Green"),
        "Red": selected_data.get("Red"),
        "NIR": selected_data.get("NIR"),
        "sst": sst_value if sst_value is not None else 0
    }
    
    features["dayofyear"] = datetime.datetime.now().timetuple().tm_yday
    features["day_sin"] = np.sin(2 * np.pi * features["dayofyear"] / 365)
    features["day_cos"] = np.cos(2 * np.pi * features["dayofyear"] / 365)
    features["NDCI"] = calculate_ndci(features["Red"], features["NIR"])
    
    missing_keys = [k for k, v in features.items() if v is None]
    if missing_keys:
        raise HTTPException(status_code=400, detail=f"Data missing for location ({lat}, {lon}): {missing_keys}")
    
    input_data = np.array([[features[f] for f in features.keys()]])
    chl_a_prediction = catboost_model.predict(input_data)[0]
    
    return {
        "lat": lat,
        "lon": lon,
        "Chlorophyll-a": chl_a_prediction,
        "dates": dates,
        "sst_date": sst_date
    }

@app.post("/predict")
def predict_chlorophyll(data: Location):
    return process_request(data.lat, data.lon)

@app.get("/")
def home():
    return {"message": "FastAPI is running. Use /predict for predictions."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
