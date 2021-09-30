# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Preprocess all the articles in the dataset
ENV COGTEXT_DATA_FRACTION=1.0

# Required to compile hdbscan
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    gcc python-dev

# Install pip requirements
RUN pip install pip -U
COPY requirements_hpc.txt .
RUN pip install --no-cache-dir -r requirements_hpc.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser
ENV COGTEXT_DATA_FRACTION=1.0

# Entry point
CMD ["python", "jobs/topic_embedding.py"]
