# Use a slim, stable Python image
FROM python:3.10-slim

# Install system tools
# We need 'xvfb' and 'wine' if we want to run MT5 logic, 
# but for the Brain (Signal Generator), we just need Python basics.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
# Remove Windows-only libraries from requirements before installing
RUN sed -i '/MetaTrader5/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch numpy pandas

# Copy the code
COPY 01_Systems ./01_Systems
COPY 06_Integration ./06_Integration

# Set the path so Python finds our modules
ENV PYTHONPATH=/app

# Default command: Run the Signal Generator
CMD ["python", "06_Integration/HybridBridge/etare_signal_generator.py"]
