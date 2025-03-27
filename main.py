import os
import io
import ee
import numpy as np
import datetime
import joblib
import logging
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

try:
    catboost_model = joblib.load("catboost_chlor_a.pkl")
    scaler = joblib.load("scaler.pkl")
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model or scaler: {e}")
    raise

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
        collection = (ee.ImageCollection(collection_id)
                      .filterBounds(point)
                      .filterDate(start_date.advance(-30, 'day'), start_date.advance(30, 'day'))
                      .sort("system:time_start"))
        image = collection.first()
        if image and image.getInfo():
            date_info = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            data = image.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
            return {band: data.get(bands[band], None) for band in bands}, date_info
    except Exception as e:
        logging.error(f"Failed to retrieve data from Earth Engine: {e}")
    return None, None

def calculate_ndci(red, nir):
    return (nir - red) / (nir + red) if (red is not None and nir is not None and (nir + red) != 0) else None

def process_request(lat, lon, date):
    data, date_info = get_nearest_data(lat, lon, SATELLITES["Sentinel-2"], BANDS["Sentinel-2"], date)
    if not data:
        raise HTTPException(status_code=404, detail=f"No satellite data available for ({lat}, {lon}) on {date}")
    
    features = {
        "latitude": lat, "longitude": lon,
        "SWIR1": data.get("SWIR1"), "SWIR2": data.get("SWIR2"),
        "Blue": data.get("Blue"), "Green": data.get("Green"),
        "Red": data.get("Red"), "NIR": data.get("NIR"),
        "NDCI": calculate_ndci(data.get("Red"), data.get("NIR"))
    }
    
    if any(v is None for v in features.values()):
        missing = [k for k, v in features.items() if v is None]
        raise HTTPException(status_code=400, detail=f"Missing data for {missing} at ({lat}, {lon})")
    
    input_data = np.array([[features[f] for f in features.keys()]])
    normalized_input = scaler.transform(input_data)
    chl_a_prediction = catboost_model.predict(normalized_input)[0]
    
    return {"lat": lat, "lon": lon, "Chlorophyll-a": chl_a_prediction * 1000, "date": date_info}

@app.post("/predict")
def predict_chlorophyll(data: Location):
    return process_request(data.lat, data.lon, data.date)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_extension not in ["csv", "xls", "xlsx"]:
            raise HTTPException(status_code=400, detail="Invalid file format. Use CSV or Excel.")
        
        df = pd.read_csv(io.BytesIO(contents)) if file_extension == "csv" else pd.read_excel(io.BytesIO(contents))
        if not {"lat", "lon", "date"}.issubset(df.columns):
            raise HTTPException(status_code=400, detail="File must contain 'lat', 'lon', 'date' columns.")
        
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise HTTPException(status_code=400, detail="Invalid dates detected.")
        
        results = [process_request(row["lat"], row["lon"], row["date"].strftime('%Y-%m-%d')) for _, row in df.iterrows()]
        return {"predictions": results}
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))