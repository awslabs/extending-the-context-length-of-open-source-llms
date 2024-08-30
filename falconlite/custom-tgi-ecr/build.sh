#!/bin/bash

REPO_NAME=${1:-custom-tgi-ecr}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)

echo "REPO_NAME: ${REPO_NAME}"
echo "REGION: ${REGION}"
echo "ACCOUNT_ID: ${ACCOUNT_ID}"

docker pull ghcr.io/huggingface/text-generation-inference:0.8.2

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

aws ecr describe-repositories --repository-names ${REPO_NAME} --region $REGION
if [ $? -ne 0 ]
then
    echo "Creating ECR repository: ${REPO_NAME}"
    aws ecr create-repository --repository-name $REPO_NAME --region $REGION
fi

docker build -t $REPO_NAME .

docker tag $REPO_NAME:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

echo "Container URI:"
echo "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest"