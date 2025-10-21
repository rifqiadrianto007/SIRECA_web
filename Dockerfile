# Gunakan base image Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Salin semua file proyek ke container
COPY . /app

# Instal dependensi
RUN pip install --no-cache-dir -r requirement.txt

# Ekspos port default FastAPI
EXPOSE 8000

# Jalankan server FastAPI dengan Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
