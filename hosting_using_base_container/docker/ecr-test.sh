#!/bin/bash

# Set variables
AWS_ACCOUNT_ID="794038231401"  # Replace with your AWS account ID
REGION="us-east-1"              # Replace with your region
REPO_NAME="custom_base_model_ownbuilt"   # Replace with your repository name
IMAGE_TAG="latest"              # Optional: change to a specific version like "v1"

# Full ECR image URI
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

# Log in to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Check if the repository exists, create it if it doesn't
if ! aws ecr describe-repositories --repository-names "${REPO_NAME}" --region ${REGION} &>/dev/null; then
    echo "Creating ECR repository: ${REPO_NAME}"
    aws ecr create-repository --repository-name "${REPO_NAME}" --region ${REGION}
else
    echo "ECR repository already exists: ${REPO_NAME}"
fi

# Remove all containers (optional, kept for consistency)
docker rm $(docker ps -aq)

# Clean up previous images created by this script before building
echo "Checking and removing previous images if they exist..."
if docker image inspect ${REPO_NAME}:${IMAGE_TAG} &>/dev/null; then
    echo "Removing ${REPO_NAME}:${IMAGE_TAG}"
    docker rmi -f ${REPO_NAME}:${IMAGE_TAG}
else
    echo "Image ${REPO_NAME}:${IMAGE_TAG} not found, skipping removal"
fi

if docker image inspect ${ECR_URI} &>/dev/null; then
    echo "Removing ${ECR_URI}"
    docker rmi -f ${ECR_URI}
else
    echo "Image ${ECR_URI} not found, skipping removal"
fi

# Build the Docker image
docker build -t ${REPO_NAME} .

# Tag the image for ECR
docker tag ${REPO_NAME}:${IMAGE_TAG} ${ECR_URI}

# Push the image to ECR
docker push ${ECR_URI}

echo "Docker image pushed to ECR: ${ECR_URI}"