FROM python:3.9

# Set the working directory to /app (Standard for HF)
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# --- NEW: Download NLTK data during BUILD to avoid runtime errors ---
RUN python -m nltk.downloader punkt punkt_tab wordnet stopwords

# Copy all files into /app
COPY . .

# Force Streamlit to use port 7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# Disable XSRF protection (required for HF iframe)
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Inform HF of the port
EXPOSE 7860

# Run using the absolute module path
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
