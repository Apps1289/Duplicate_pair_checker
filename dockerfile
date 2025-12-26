# Use the official Python image
FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy your requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your app's code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 7860

# Command to run the app (ensure your file is named app.py)
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]