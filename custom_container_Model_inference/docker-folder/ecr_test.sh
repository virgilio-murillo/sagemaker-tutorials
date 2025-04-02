#!/bin/bash

# Set variables
AWS_ACCOUNT_ID="794038231401"  # Replace with your AWS account ID
REGION="us-east-1"              # Replace with your region
REPO_NAME="my-sagemaker-model"  # Replace with your repository name
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

# Clean all containers
docker rmi -f $(docker images -q)
docker rm $(docker ps -aq)

# Build the Docker image
docker build -t ${REPO_NAME} .

# Tag the image for ECR
docker tag ${REPO_NAME}:${IMAGE_TAG} ${ECR_URI}

# Push the image to ECR
docker push ${ECR_URI}

echo "Docker image pushed to ECR: ${ECR_URI}"