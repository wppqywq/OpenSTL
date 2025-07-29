FROM pytorch/pytorch:2.1.1-cuda11.8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first for better caching
COPY requirements/ requirements/
COPY requirements.txt .
COPY environment.yml .

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install OpenSTL in development mode
RUN pip install -e .

# Create directory for outputs
RUN mkdir -p /workspace/outputs

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Expose port for potential web services
EXPOSE 8000

# Default command
CMD ["python", "-c", "import openstl; print('OpenSTL Docker container ready')"] 