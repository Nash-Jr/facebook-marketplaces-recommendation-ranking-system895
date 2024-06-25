FROM python:3.8

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY checkpoint.pt .
COPY embeddings.npz .
COPY index.faiss .
COPY id_to_filename.json .
COPY Facebook_Project_API.py .
COPY image_decoder.pkl .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "Facebook_Project_API:app", "--host", "0.0.0.0", "--port", "8000"]