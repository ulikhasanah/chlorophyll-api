import os
import ee
import numpy as np
import datetime
import joblib
import logging
from typing import List
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

try:
    catboost_model = joblib.load("catboost_chlor_a.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

scaler = joblib.load("scaler.pkl")

SATELLITES = {"Sentinel-2": "COPERNICUS/S2_SR"}
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
    date: str = None

class MultiLocation(BaseModel):
    locations: List[Location]

def get_nearest_data(lat, lon, collection_id, bands, target_date):
    try:
        point = ee.Geometry.Point(lon, lat)
        start_date = ee.Date(target_date) if target_date else ee.Date(datetime.datetime.utcnow().strftime('%Y-%m-%d'))
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(point)
            .filterDate(start_date.advance(-30, 'day'), start_date.advance(30, 'day'))
        )
        
        def add_diff(image):
            return image.set("date_diff", ee.Number(image.date().difference(start_date, "day")).abs())
        
        collection = collection.map(add_diff).sort("date_diff")
        image = collection.first()
        
        if image is not None and image.getInfo():
            date_info = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            data = image.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
            return {band: data.get(bands[band], None) for band in bands}, date_info
    except Exception as e:
        logging.error(f"Failed to retrieve data from Earth Engine: {e}")
    return None, None

def get_nearest_sst(lat, lon, target_date):
    try:
        point = ee.Geometry.Point(lon, lat)
        start_date = ee.Date(target_date) if target_date else ee.Date(datetime.datetime.utcnow().strftime('%Y-%m-%d'))
        collection = (
            ee.ImageCollection("NOAA/CDR/OISST/V2_1")
            .filterBounds(point)
            .filterDate(start_date.advance(-30, 'day'), start_date.advance(30, 'day'))
            .sort("system:time_start")
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

def process_request(lat, lon, date):
    data_sources = {}
    dates = {}
    
    for sat in ["Sentinel-2"]:
        data, date_info = get_nearest_data(lat, lon, SATELLITES[sat], BANDS[sat], date)
        if data:
            data_sources[sat] = data
            dates[sat] = date_info
    
    sst_value, sst_date = get_nearest_sst(lat, lon, date)
    selected_data = next((data_sources[sat] for sat in data_sources if data_sources[sat]), None)
    
    if not selected_data:
        raise HTTPException(status_code=404, detail=f"No satellite data available for location ({lat}, {lon}) on {date}")
    
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
    
    features["dayofyear"] = datetime.datetime.strptime(dates["Sentinel-2"], "%Y-%m-%d").timetuple().tm_yday if "Sentinel-2" in dates else datetime.datetime.now().timetuple().tm_yday
    features["day_sin"] = np.sin(2 * np.pi * features["dayofyear"] / 365)
    features["day_cos"] = np.cos(2 * np.pi * features["dayofyear"] / 365)
    features["NDCI"] = calculate_ndci(features["Red"], features["NIR"])
    
    input_data = np.array([[features[f] for f in features.keys()]])
    normalized_input = scaler.transform(input_data)
    chl_a_prediction = catboost_model.predict(normalized_input)[0]
    
    return {
        "lat": lat,
        "lon": lon,
        "Chlorophyll-a": chl_a_prediction * 1000,
        "dates": dates,
        "sst_date": sst_date
    }

@app.post("/predict")
def predict_chlorophyll(data: Location):
    return process_request(data.lat, data.lon, data.date)
  
@app.post("/predict-multi")
def predict_multiple_chlorophyll(data: MultiLocation):
    results = [process_request(loc.lat, loc.lon, loc.date) for loc in data.locations]
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
