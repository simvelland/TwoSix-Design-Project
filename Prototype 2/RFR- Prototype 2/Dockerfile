# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files into container
COPY app.py .
COPY random_forest_model.pkl .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Start Flask app
CMD ["python", "app.py"]
