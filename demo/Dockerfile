# Use the official Python image from the Docker Hub
FROM python:3.11.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY demo/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app runs on
EXPOSE 3788

# Add a label for the application name
LABEL name="chinese_digit_classifier"

# Define the command to run the application
CMD ["python", "demo/app.py"]