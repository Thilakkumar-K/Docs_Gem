#!/bin/bash

# Cloud Run Deployment Script for RAG Document QA API
# Make sure to run: chmod +x deploy-cloudrun.sh

set -e  # Exit on any error

# Configuration
PROJECT_ID="your-project-id"  # Replace with your actual project ID
SERVICE_NAME="rag-document-qa-api"
REGION="us-central1"  # Choose your preferred region
IMAGE_NAME="rag-document-qa"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Cloud Run deployment for RAG Document QA API${NC}"
echo -e "${BLUE}Project: $PROJECT_ID${NC}"
echo -e "${BLUE}Region: $REGION${NC}"
echo -e "${BLUE}Service: $SERVICE_NAME${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Set the project
echo -e "${YELLOW}🔧 Setting up gcloud project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}🔧 Enabling required Google Cloud APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com

# Create secrets (you'll be prompted to enter the values)
echo -e "${YELLOW}🔐 Creating secrets in Secret Manager...${NC}"

# Function to create secret if it doesn't exist
create_secret_if_not_exists() {
    local secret_name=$1
    local secret_description=$2

    if ! gcloud secrets describe $secret_name &> /dev/null; then
        echo -e "${YELLOW}Creating secret: $secret_name${NC}"
        echo -n "Enter $secret_description: "
        read -s secret_value
        echo ""
        echo -n "$secret_value" | gcloud secrets create $secret_name --data-file=-
        echo -e "${GREEN}✅ Secret $secret_name created${NC}"
    else
        echo -e "${GREEN}✅ Secret $secret_name already exists${NC}"
    fi
}

create_secret_if_not_exists "gemini-api-key" "your Gemini API key"
create_secret_if_not_exists "valid-token" "your API authentication token"
create_secret_if_not_exists "supabase-url" "your Supabase URL"
create_secret_if_not_exists "supabase-key" "your Supabase key"

# Build and push Docker image using Cloud Build
echo -e "${YELLOW}🐳 Building Docker image with Cloud Build...${NC}"
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG .

echo -e "${GREEN}✅ Docker image built and pushed successfully${NC}"

# Deploy to Cloud Run
echo -e "${YELLOW}🚀 Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 900 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars PORT=8080,ENVIRONMENT=production,SUPABASE_BUCKET=documents,EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2,CHUNK_SIZE=2000,CHUNK_OVERLAP=200,TOP_K_RETRIEVAL=10,MAX_CONTEXT_LENGTH=10000 \
    --set-secrets GEMINI_API_KEY=gemini-api-key:latest,VALID_TOKEN=valid-token:latest,SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-key:latest

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${GREEN}Service URL: $SERVICE_URL${NC}"
echo -e "${GREEN}Health check: $SERVICE_URL/api/v1/health${NC}"
echo -e "${GREEN}API docs: $SERVICE_URL/api/v1/docs${NC}"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo -e "1. Test your API: curl $SERVICE_URL/api/v1/health"
echo -e "2. Visit the API documentation at: $SERVICE_URL/api/v1/docs"
echo -e "3. Configure any domain or custom settings in the Cloud Console"
echo ""
echo -e "${YELLOW}💡 Useful commands:${NC}"
echo -e "View logs: gcloud run services logs read $SERVICE_NAME --region $REGION"
echo -e "Update service: gcloud run services update $SERVICE_NAME --region $REGION"
echo -e "Delete service: gcloud run services delete $SERVICE_NAME --region $REGION"