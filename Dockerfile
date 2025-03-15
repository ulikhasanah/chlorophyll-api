# Gunakan image Python 3.9 sebagai dasar
FROM python:3.9

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements.txt & install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file proyek ke dalam container
COPY . .

# Jalankan FastAPI menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
