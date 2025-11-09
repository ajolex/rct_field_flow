# Deployment Quick Start

## 🎯 What Was Set Up

Your RCT Field Flow app is now **production-ready** with 3 deployment options:

### ✅ Files Created/Updated
- ✓ **Dockerfile** - Updated to Python 3.13 with best practices
- ✓ **docker-compose.yml** - Enhanced with health checks & volumes
- ✓ **.dockerignore** - Optimized Docker image size
- ✓ **.env.example** - Environment variable template
- ✓ **.streamlit/config.toml** - Production settings
- ✓ **.streamlit/secrets.toml.example** - Secrets template
- ✓ **DEPLOYMENT.md** - Comprehensive guide (read this!)
- ✓ **deploy.sh** - Bash helper (Linux/Mac)
- ✓ **deploy.ps1** - PowerShell helper (Windows)

---

## 🚀 Quick Start (Pick One)

### Option A: Streamlit Cloud (Recommended - Easiest)
**Best for:** Teams, free hosting, zero DevOps

**Steps:**
1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select repo: `ajolex/rct_field_flow`
5. Select main file: `rct_field_flow/app.py`
6. Deploy!
7. Add secrets in app settings

**Cost:** Free (or $7/month pro)  
**Time:** 5 minutes

---

### Option B: Docker Local (Testing)
**Best for:** Local development, testing before deploy

**Windows:**
```powershell
.\deploy.ps1
# Select option 1
```

**Mac/Linux:**
```bash
bash deploy.sh
# Select option 1
```

**Then visit:** http://localhost:8501

---

### Option C: Docker + Cloud Server (Full Control)
**Best for:** Enterprise, custom requirements

**Servers:**
- DigitalOcean: ~$5-20/month
- AWS EC2: ~$10-50/month
- Azure: ~$20-50/month
- Google Cloud: ~$20-50/month

**First, test locally (Option B), then:**
1. Rent a server
2. Install Docker on server
3. Clone repo on server
4. Create `.env` file with SurveyCTO credentials
5. Run: `docker-compose up -d`
6. Point domain to server IP
7. Add reverse proxy (Nginx) for HTTPS

See **DEPLOYMENT.md** for detailed instructions

---

## 📝 Before Deployment

### 1. Create `.env` file locally
```bash
# Copy template
cp .env.example .env

# Edit with your SurveyCTO details
# Windows: notepad .env
# Mac/Linux: nano .env
```

### 2. Update `.streamlit/secrets.toml` (local dev only)
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit with your credentials
```

### 3. Test locally
```powershell
# Windows
.\deploy.ps1
```

```bash
# Mac/Linux
bash deploy.sh
```

---

## 🔐 Environment Variables

Required for SurveyCTO integration:

```env
SCTO_SERVER=https://your-instance.surveycto.com
SCTO_USER=your-username
SCTO_PASS=your-api-token
```

**For Streamlit Cloud:**
1. Settings → Secrets
2. Paste from `.streamlit/secrets.toml`
3. Save

**For Docker:**
- Stored in `.env` file
- Docker Compose loads automatically
- `.env` is gitignored (won't push to GitHub)

---

## 🐳 Docker Commands Reference

```bash
# Build image
docker build -t rct-field-flow:latest .

# Run container
docker run -p 8501:8501 --env-file .env rct-field-flow:latest

# Using Docker Compose
docker-compose up                    # Start
docker-compose up -d                 # Start in background
docker-compose down                  # Stop
docker-compose logs -f               # View logs
docker-compose ps                    # Status

# View container details
docker ps                            # Running containers
docker images                        # Available images
```

---

## 📊 Recommended for Production

| Aspect | Recommendation |
|--------|-----------------|
| **Hosting** | Streamlit Cloud (easiest) or AWS (scalable) |
| **Database** | CSV files in S3 (simple) or PostgreSQL (production) |
| **Authentication** | Streamlit secrets for SurveyCTO creds |
| **Monitoring** | Streamlit Cloud logs or CloudWatch |
| **Backups** | GitHub + S3/cloud storage |
| **HTTPS** | Automatic (Streamlit Cloud) or Nginx + Let's Encrypt |

---

## 🐛 Troubleshooting

### "Port 8501 already in use"
```bash
# Use different port
docker-compose up -p 8502:8501
# Or kill existing process
lsof -i :8501
kill -9 <PID>
```

### "Module not found"
✓ Already fixed - `app.py` handles imports correctly

### "SurveyCTO credentials not working"
1. Check `.env` file has correct values
2. Check credentials are in Streamlit Cloud secrets
3. Verify server URL format: `https://...surveycto.com`

### "Out of memory"
- Increase Docker memory limit
- Reduce cache/data sizes
- Use larger server

---

## 📖 Full Documentation

See **DEPLOYMENT.md** for:
- ✓ Detailed comparison of all 3 options
- ✓ Step-by-step guides for each
- ✓ AWS/Azure/DigitalOcean deployment
- ✓ Performance optimization
- ✓ Security checklist
- ✓ Monitoring setup
- ✓ Common issues & fixes

---

## ✨ Next Steps

1. **Immediate:** Test locally with Docker Compose
   ```powershell
   .\deploy.ps1
   ```

2. **This week:** Deploy to Streamlit Cloud
   - Go to https://streamlit.io/cloud
   - Authorize GitHub
   - Deploy repo

3. **Soon:** Add monitoring & backups
   - See DEPLOYMENT.md

---

## 💡 Pro Tips

- **Test everything locally first** before deploying
- **Keep `.env` secure** - never commit to GitHub (already in `.gitignore`)
- **Monitor logs** for errors in production
- **Set up backups** for exported data
- **Use Streamlit Cloud** unless you need specific customizations

---

## Questions?

1. Read **DEPLOYMENT.md** (comprehensive guide)
2. Check **Docker documentation** (if self-hosting)
3. See **Streamlit docs** (if using Streamlit Cloud)
4. Review container logs for specific errors

---

**You're all set! 🎉 Pick your deployment path and go live!**
