#!/bin/bash

# Set variables
AWS_ACCOUNT_ID="794038231401"  # Replace with your AWS account ID
REGION="us-east-1"              # Replace with your region
TIMESTAMP=$(date +%Y%m%d%H%M%S)
REPO_NAME="custom-base-model-${TIMESTAMP}"  # Changed REPOSITORY_NAME to REPO_NAME
echo $REPO_NAME
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"

# Check if repository exists, create if not
if ! aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" > /dev/null 2>&1; then
    echo "Repository does not exist. Creating repository: ${REPO_NAME}"
    aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"
fi

# Generate unique tag
UNIQUE_TAG=$(date +%Y%m%d%H%M%S)-$RANDOM

# Log in to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build the Docker image with unique tag and latest
docker build -t ${REPO_NAME}:${UNIQUE_TAG} -t ${REPO_NAME}:latest .

# Tag the images for ECR
docker tag ${REPO_NAME}:${UNIQUE_TAG} ${ECR_URI}:${UNIQUE_TAG}
docker tag ${REPO_NAME}:latest ${ECR_URI}:latest

# Push the images to ECR
docker push ${ECR_URI}:${UNIQUE_TAG}
docker push ${ECR_URI}:latest

echo "Docker images pushed to ECR: ${ECR_URI}:${UNIQUE_TAG} and ${ECR_URI}:latest"