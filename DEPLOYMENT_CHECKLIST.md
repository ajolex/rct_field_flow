# Streamlit Cloud Deployment Checklist

Quick reference for deploying RCT Field Flow to Streamlit Cloud with PostgreSQL.

## Pre-Deployment

- [x] PostgreSQL migration complete
- [x] `psycopg2-binary` added to requirements.txt
- [x] `.gitignore` configured
- [ ] Supabase account created
- [ ] Supabase database credentials obtained

## GitHub Setup

- [ ] All changes committed to Git
- [ ] Pushed to GitHub repository

```bash
git add .
git commit -m "PostgreSQL migration for persistent analytics"
git push origin main
```

## Streamlit Cloud Configuration

### 1. Database Secrets

Navigate to: **Your App** → **⚙️ Settings** → **Secrets**

Paste your Supabase credentials:

```toml
[database]
host = "xxxxx.supabase.co"
port = 5432
database = "postgres"
user = "postgres"
password = "your-password"
sslmode = "require"
```

### 2. Verify Auto-Deployment

- [ ] App automatically redeploys after Git push
- [ ] Check build logs for successful deployment
- [ ] Verify no errors in requirements installation

### 3. Test Database Connection

**Check App Logs:**
- [ ] Look for "Database type: PostgreSQL" message
- [ ] Confirm PostgreSQL connection successful
- [ ] No "falling back to SQLite" warnings

**Functional Tests:**
- [ ] Register a test user
- [ ] Log in successfully
- [ ] Navigate to analytics dashboard
- [ ] Record some activity
- [ ] **Restart app** (⋮ → Reboot app)
- [ ] Verify user still exists after restart
- [ ] Verify activity data persists

## Post-Deployment Verification

- [ ] Analytics dashboard shows persistent data
- [ ] User accounts survive app restarts
- [ ] Activity logs accumulate over time
- [ ] No unexpected database errors in logs

## Optional: Local PostgreSQL Testing

To test PostgreSQL locally before deploying:

1. Create `.streamlit/secrets.toml` (copy from `.streamlit/secrets.toml.example`)
2. Add your Supabase credentials
3. Run: `python database_migration_test.py`
4. Should see: "Database type: PostgreSQL"

**Remember:** Delete or rename `.streamlit/secrets.toml` to return to SQLite for local development

## Rollback (If Needed)

If issues occur:

1. **Quick fix:** Remove database secrets from Streamlit Cloud
   - App will automatically fall back to SQLite
   
2. **Full rollback:** 
   ```bash
   git revert HEAD
   git push origin main
   ```

## Success Indicators

✅ No "connection refused" errors  
✅ No "psycopg2 not found" errors  
✅ Data persists after app restarts  
✅ Analytics show cumulative data over time  
✅ Multiple users can register and their data is saved  

---

**Status:** Ready for deployment! 🚀
