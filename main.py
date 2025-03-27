import os
import ee
import numpy as np
import datetime
import joblib
import logging
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
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

def process_request(lat, lon, date):
    # Fungsi yang mengambil data dari GEE dan memprosesnya seperti yang telah Anda buat sebelumnya
    pass  # Implementasi tetap sama seperti sebelumnya

@app.post("/predict")
def predict_chlorophyll(data: Location):
    return process_request(data.lat, data.lon, data.date)

@app.post("/predict-multi")
def predict_multiple_chlorophyll(data: MultiLocation):
    results = [process_request(loc.lat, loc.lon, loc.date) for loc in data.locations]
    return results

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        
        # Pilih parser sesuai ekstensi file
        if file_extension == "csv":
            df = pd.read_csv(pd.io.common.BytesIO(contents))
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(pd.io.common.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="File must be CSV or Excel format.")
        
        # Validasi kolom
        required_cols = {'lat', 'lon', 'date'}
        if not required_cols.issubset(df.columns):
            raise HTTPException(status_code=400, detail="File must contain 'lat', 'lon', and 'date' columns.")
        
        # Proses data
        results = [process_request(row['lat'], row['lon'], row['date']) for _, row in df.iterrows()]
        return {"predictions": results}
    
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
