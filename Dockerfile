# Use the official Python image
FROM python:3.9-slim

# Set working directory for dockerised app
WORKDIR /app

# Copy requirements.txt (from root of project to root of our app)
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (from root of project to src folder in our app)
COPY . ./src/

# Set the working directory to src in our app
WORKDIR /app/src

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]