# Use an official Python 3.8 runtime as a base image
FROM python:3.13

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set the working directory in the Docker image to /opt/program
WORKDIR /opt/program

# Upgrade pip package manager to the latest version
RUN pip install --upgrade pip

# Copy requirements.txt and install packages
COPY ./requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the source code of the application
COPY ./src /opt/program

# Copy the model into the image for local testing
COPY ./model /opt/ml/model

# Set the entry point to run Gunicorn
ENTRYPOINT ["gunicorn", "-b", ":8080", "prediction_script:app"]