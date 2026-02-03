FROM python:3.12
# Install necessary dependencies

WORKDIR /app
RUN mkdir -p /parquet_files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./
