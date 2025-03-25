import os
import ee
import numpy as np
import datetime
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Inisialisasi Google Earth Engine
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
    satellite: str = Field("Sentinel-2", description="Satelit yang digunakan")
    date: str = Field(None, description="Tanggal format YYYY-MM-DD (opsional)")

def get_nearest_data(lat, lon, satellite, date):
    try:
        point = ee.Geometry.Point(lon, lat)
        if date:
            start_date = ee.Date(date).advance(-15, 'day')
            end_date = ee.Date(date).advance(15, 'day')
        else:
            start_date = ee.Date(datetime.datetime.utcnow().strftime('%Y-%m-%d')).advance(-30, 'day')
            end_date = ee.Date(datetime.datetime.utcnow().strftime('%Y-%m-%d'))
        
        collection = (
            ee.ImageCollection(SATELLITES[satellite])
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .sort("system:time_start", False)
        )
        image = collection.first()
        if image is not None and image.getInfo():
            date_info = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            data = image.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
            return {band: data.get(BANDS[satellite][band], None) for band in BANDS[satellite]}, date_info
    except Exception as e:
        logging.error(f"Failed to retrieve data from Earth Engine: {e}")
    return None, None

def process_request(lat, lon, satellite, date):
    data, retrieved_date = get_nearest_data(lat, lon, satellite, date)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"No satellite data available for ({lat}, {lon})")
    
    features = {
        "latitude": lat,
        "longitude": lon,
        "SWIR1": data.get("SWIR1"),
        "SWIR2": data.get("SWIR2"),
        "Blue": data.get("Blue"),
        "Green": data.get("Green"),
        "Red": data.get("Red"),
        "NIR": data.get("NIR"),
    }
    
    features["dayofyear"] = datetime.datetime.strptime(retrieved_date, "%Y-%m-%d").timetuple().tm_yday
    features["day_sin"] = np.sin(2 * np.pi * features["dayofyear"] / 365)
    features["day_cos"] = np.cos(2 * np.pi * features["dayofyear"] / 365)
    features["NDCI"] = (features["NIR"] - features["Red"]) / (features["NIR"] + features["Red"]) if (features["NIR"] + features["Red"]) != 0 else None
    
    missing_keys = [k for k, v in features.items() if v is None]
    if missing_keys:
        raise HTTPException(status_code=400, detail=f"Data missing for ({lat}, {lon}): {missing_keys}")
    
    input_data = np.array([[features[f] for f in features.keys()]])
    chl_a_prediction = catboost_model.predict(input_data)[0]
    
    return {
        "lat": lat,
        "lon": lon,
        "Chlorophyll-a": chl_a_prediction * 1000,
        "date_used": retrieved_date,
        "satellite": satellite
    }

@app.post("/predict")
def predict_chlorophyll(data: Location):
    return process_request(data.lat, data.lon, data.satellite, data.date)

@app.get("/")
def home():
    return {"message": "FastAPI is running. Use /predict for predictions."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
