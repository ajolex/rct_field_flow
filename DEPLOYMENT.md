# RCT Field Flow - Deployment Guide

## Overview & Recommendations

Your app is **well-structured for containerized deployment**. I recommend the **Docker + Streamlit Cloud hybrid approach**:

1. **For Development/Testing**: Use Docker locally (already configured ✓)
2. **For Production**: Deploy to **Streamlit Cloud** (free, easiest, built for Streamlit)
3. **Alternative**: Docker on cloud servers (AWS, Azure, DigitalOcean) if you need more control

---

## Current Status

### ✅ What You Have
- `Dockerfile` - Good foundation, but needs updates
- `docker-compose.yml` - Basic setup, working
- `pyproject.toml` - Poetry configuration ✓
- `requirements.txt` - Pip dependencies ✓
- Main app: `rct_field_flow/app.py` (entry point)

### ⚠️ Issues to Fix
1. **Dockerfile references wrong entry point** (`monitor.py` instead of `app.py`)
2. **No `.streamlit/config.toml`** for Streamlit settings
3. **No `.env.example`** for environment variables documentation
4. **No `.dockerignore`** for smaller images
5. **Python 3.11 should be 3.13** (your code runs on 3.13.7)

---

## Option 1: Streamlit Cloud (Recommended for 80% of use cases)

### Why Streamlit Cloud?
- ✅ **Free** for public/private apps
- ✅ **Zero configuration** - just push to GitHub
- ✅ **Built for Streamlit** - perfect performance
- ✅ **Automatic scaling** - handles traffic
- ✅ **Built-in SSL/HTTPS**
- ✅ **Easy secrets management**
- ❌ Limited to Streamlit apps (but that's your app!)

### Steps to Deploy

1. **Push to GitHub** (you already have this)
   ```bash
   git push origin master
   ```

2. **Create `.streamlit/config.toml`**
   ```toml
   [client]
   showErrorDetails = false
   
   [logger]
   level = "info"
   
   [theme]
   base = "light"
   primaryColor = "#1f77b4"
   ```

3. **Create `.streamlit/secrets.toml`** (locally, for testing)
   ```toml
   scto_server = "your-surveycto-server"
   scto_user = "your-username"
   scto_pass = "your-password"
   ```

4. **Go to [streamlit.io/cloud](https://streamlit.io/cloud)**
   - Sign in with GitHub
   - Click "New app"
   - Select repo: `ajolex/rct_field_flow`
   - Set main file: `rct_field_flow/app.py`
   - Click "Deploy"

5. **Add Secrets in Streamlit Cloud Dashboard**
   - Settings → Secrets
   - Copy from your local `.streamlit/secrets.toml`
   - Save

### Cost: **$0** (free tier) or $7/month (premium for higher limits)

---

## Option 2: Docker + Self-Hosted (AWS/Azure/DigitalOcean)

### Why Choose This?
- More control over infrastructure
- Can handle higher traffic
- Can integrate with other services
- Cost: $5-20+/month depending on server size

### What to Update

#### Step 1: Improve Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app

USER streamlit

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "rct_field_flow/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--logger.level=info"]
```

#### Step 2: Create `.dockerignore`

```
__pycache__
*.pyc
.pytest_cache
.venv
venv
*.egg-info
.git
.gitignore
.env
*.csv
*.parquet
node_modules
.DS_Store
```

#### Step 3: Update `docker-compose.yml`

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rct-field-flow
    ports:
      - "8501:8501"
    environment:
      - SCTO_SERVER=${SCTO_SERVER}
      - SCTO_USER=${SCTO_USER}
      - SCTO_PASS=${SCTO_PASS}
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./data:/app/data           # Data persistence
      - ./uploads:/app/uploads     # User uploads
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

#### Step 4: Create `.env.example`

```env
# SurveyCTO Configuration
SCTO_SERVER=https://your-surveycto-server.surveycto.com
SCTO_USER=your-username
SCTO_PASS=your-api-token

# Streamlit Configuration (optional)
STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=false
STREAMLIT_LOGGER_LEVEL=info
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

#### Step 5: Create `.streamlit/config.toml`

```toml
[client]
showErrorDetails = false
maxMessageSize = 200

[logger]
level = "info"

[theme]
base = "light"
primaryColor = "#1f77b4"

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = true

[browser]
gatherUsageStats = false
```

### Deploy to DigitalOcean (Example)

```bash
# 1. Create droplet (Ubuntu 22.04, 2GB RAM minimum)
# 2. SSH into droplet
ssh root@your-droplet-ip

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 4. Clone repo
cd /opt
git clone https://github.com/ajolex/rct_field_flow.git
cd rct_field_flow

# 5. Create .env file
cp .env.example .env
# Edit with real credentials
nano .env

# 6. Start with Docker Compose
docker-compose up -d

# 7. View logs
docker-compose logs -f app

# 8. Set up reverse proxy (Nginx recommended)
# Point domain to droplet IP
```

### Deploy to AWS/Azure

**Option A: AWS Elastic Container Service (ECS)**
- Use `Dockerfile`
- Push to AWS ECR (Elastic Container Registry)
- Deploy via ECS
- Cost: ~$10-20/month

**Option B: Azure Container Instances**
- Use `Dockerfile`
- Push to Azure Container Registry
- Deploy via ACI
- Cost: Pay-per-second (~$0.0015/second = ~$50/month idle)

**Option C: AWS EC2 + Docker**
- Cheapest option
- t2.micro (1GB) = ~$10/month
- t3.small (2GB) = ~$20/month
- Requires more manual setup

---

## Option 3: Heroku (Deprecated but still works)

Heroku phased out free tier, but paid tier available. **Not recommended** - use Streamlit Cloud or self-hosted instead.

---

## Quick Start Commands

### Local Development
```bash
# Start with Docker Compose
docker-compose up

# Access at http://localhost:8501
```

### Build & Push to Docker Hub (optional)
```bash
# Build image
docker build -t ajolex/rct-field-flow:latest .

# Push to Docker Hub (requires account)
docker push ajolex/rct-field-flow:latest

# Pull anywhere
docker pull ajolex/rct-field-flow:latest
docker run -p 8501:8501 -e SCTO_SERVER=... ajolex/rct-field-flow:latest
```

---

## Environment Variables Setup

### For Local Development
1. Create `.env` file:
   ```env
   SCTO_SERVER=https://your-instance.surveycto.com
   SCTO_USER=your-username
   SCTO_PASS=your-api-token
   ```

2. Load in your app (already handled by `python-dotenv` ✓)

### For Streamlit Cloud
1. Go to app settings → Secrets
2. Add secrets in TOML format:
   ```toml
   scto_server = "https://your-instance.surveycto.com"
   scto_user = "your-username"
   scto_pass = "your-api-token"
   ```

### For Docker
1. Create `.env` file in project root
2. Docker Compose loads automatically (see updated `docker-compose.yml`)

---

## Performance Optimization Tips

### Memory & CPU
- **Development**: 512MB RAM minimum
- **Small team (5-10 users)**: 1-2GB RAM
- **Large team (50+ users)**: 4GB+ RAM or load balancer

### Data Handling
```python
# Add to rct_field_flow/app.py for caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_submission_data(form_id):
    # expensive operation
    return data
```

### Network
- Use CDN for static files
- Compress data transfers
- Consider regional deployment for global teams

---

## Security Checklist

- ✅ Non-root Docker user (in updated Dockerfile)
- ✅ Environment variables for secrets (no hardcoding)
- ✅ HTTPS/SSL (automatic on Streamlit Cloud)
- ✅ Health checks configured
- ✅ Error details hidden in production
- ⚠️ TODO: Add authentication layer (optional)
- ⚠️ TODO: Add rate limiting (optional)

### Add Authentication (Optional)

```python
# Add to app.py top-level
import streamlit as st

def check_password():
    """Returns True if user authenticates successfully."""
    if st.secrets.get("app_password"):
        if "password_correct" not in st.session_state:
            st.session_state.password_correct = False
        
        if not st.session_state.password_correct:
            password = st.text_input("Enter password:", type="password")
            if password == st.secrets["app_password"]:
                st.session_state.password_correct = True
            else:
                st.error("Incorrect password")
                return False
    return True

if not check_password():
    st.stop()

# Rest of app code...
```

---

## Monitoring & Logging

### Local Monitoring
```bash
# Check running containers
docker ps

# View logs
docker logs -f <container_name>

# Access shell
docker exec -it <container_name> bash
```

### Production Monitoring
- **Streamlit Cloud**: Built-in logs in dashboard
- **Self-hosted**: Use Datadog, New Relic, or CloudWatch
- **Health checks**: Configured in `docker-compose.yml`

---

## Recommended Path

### For Immediate Deployment (Today)
1. Create `.streamlit/config.toml`
2. Create `.streamlit/secrets.toml` (local)
3. Push to GitHub
4. Deploy to Streamlit Cloud (5 minutes)
5. Add secrets in Cloud dashboard

### For Production (This Week)
1. Update Dockerfile (improved version above)
2. Create `.dockerignore`
3. Create `.env.example`
4. Test locally: `docker-compose up`
5. Deploy to cloud provider of choice

### For Enterprise (Optional)
- Add authentication
- Set up monitoring
- Configure auto-scaling
- Set up CI/CD pipeline

---

## Troubleshooting

### "Module not found" errors
**Fix:** Ensure `sys.path` configuration in `app.py` is correct (already done ✓)

### Out of memory
**Fix:** Increase RAM in Docker or `docker-compose.yml`

### Secrets not loading
**Fix:** 
- Local: Ensure `.streamlit/secrets.toml` exists
- Cloud: Check Streamlit Cloud secrets dashboard
- Docker: Check `.env` file and `docker-compose.yml` environment section

### Port already in use
**Fix:** Change port in `docker-compose.yml` from 8501 to different port (8502, etc.)

---

## Next Steps

**Choose your deployment path:**

**→ Want easiest/fastest?** Go with **Streamlit Cloud**  
**→ Want more control?** Go with **Docker + DigitalOcean/AWS**  
**→ Want to test locally first?** Use **Docker Compose** locally

Let me know which option you prefer, and I'll help implement it!
