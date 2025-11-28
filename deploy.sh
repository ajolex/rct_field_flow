#!/bin/bash
# Quick deployment helper script for RCT Field Flow

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}RCT Field Flow - Deployment Helper${NC}"
echo -e "${GREEN}================================${NC}\n"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Install from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"
echo "  Version: $(docker --version)"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Install from: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker Compose found${NC}"
echo "  Version: $(docker-compose --version)"

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file not found${NC}"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env and add your SurveyCTO credentials${NC}"
    read -p "Press enter to continue..."
fi

# Check for .streamlit/secrets.toml (local development)
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo -e "${YELLOW}⚠ .streamlit/secrets.toml not found${NC}"
    echo "Creating from .streamlit/secrets.toml.example..."
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    echo -e "${YELLOW}Please edit .streamlit/secrets.toml for local development${NC}"
fi

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Choose deployment option:${NC}"
echo -e "${GREEN}================================${NC}\n"

PS3='Select option (1-4): '
options=(
    "🐳 Build & run with Docker Compose locally"
    "☁️  Prepare for Streamlit Cloud deployment"
    "🔧 Build Docker image only (no run)"
    "📋 Show deployment guide"
)

select opt in "${options[@]}"
do
    case $REPLY in
        1)
            echo -e "\n${GREEN}Starting Docker Compose...${NC}\n"
            docker-compose up
            break
            ;;
        2)
            echo -e "\n${GREEN}================================${NC}"
            echo -e "${GREEN}Streamlit Cloud Deployment${NC}"
            echo -e "${GREEN}================================${NC}\n"
            echo "Steps to deploy to Streamlit Cloud:"
            echo ""
            echo "1. Ensure all changes are committed to GitHub:"
            echo "   git add ."
            echo "   git commit -m 'Prepare for Streamlit Cloud deployment'"
            echo "   git push origin master"
            echo ""
            echo "2. Go to: https://streamlit.io/cloud"
            echo ""
            echo "3. Sign in with GitHub and authorize Streamlit"
            echo ""
            echo "4. Click 'New app' and select:"
            echo "   - Repository: ajolex/rct_field_flow"
            echo "   - Branch: master"
            echo "   - Main file path: rct_field_flow/app.py"
            echo ""
            echo "5. After deployment, add secrets in app settings:"
            echo "   Settings → Secrets → Add from .streamlit/secrets.toml"
            echo ""
            echo -e "${YELLOW}Ready? Push to GitHub now:${NC}"
            read -p "Press enter to commit and push..."
            git add .
            git commit -m "Deployment: Add Streamlit Cloud configuration"
            git push origin master
            echo -e "${GREEN}✓ Pushed to GitHub${NC}"
            echo "Now go to https://streamlit.io/cloud to deploy!"
            break
            ;;
        3)
            echo -e "\n${GREEN}Building Docker image...${NC}\n"
            docker build -t rct-field-flow:latest .
            echo -e "\n${GREEN}✓ Build complete!${NC}"
            echo "To run: docker run -p 8501:8501 --env-file .env rct-field-flow:latest"
            break
            ;;
        4)
            echo -e "\n${GREEN}Opening DEPLOYMENT.md...${NC}\n"
            if command -v cat &> /dev/null; then
                cat DEPLOYMENT.md | head -100
            else
                less DEPLOYMENT.md
            fi
            break
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done

echo -e "\n${GREEN}Done!${NC}\n"
