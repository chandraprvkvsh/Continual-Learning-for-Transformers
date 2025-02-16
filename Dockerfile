FROM python:3.11.9

# Set working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache for faster rebuilds
COPY requirements.txt ./ 

# Install virtualenv and create a virtual environment
RUN apt-get update && apt-get install -y coreutils && \
    pip install --no-cache-dir virtualenv && \
    virtualenv /app/venv

# Install dependencies in the virtual environment
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set the Python environment to use the virtual environment
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH=/app/src

# Create directories for data and output
RUN mkdir -p /app/data /app/output

# Set the entrypoint and default command to use the virtual environment
ENTRYPOINT ["python", "-m", "src.main"]

# Default arguments
CMD ["--data-dir", "/app/data", "--output-dir", "/app/output", "--wandb"]