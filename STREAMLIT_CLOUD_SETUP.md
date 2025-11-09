# Streamlit Cloud Deployment - Step by Step

## Prerequisites Check ✅

Before we start, verify you have:
- [ ] GitHub account (you do ✅)
- [ ] Your repo pushed to GitHub (done ✅)
- [ ] SurveyCTO credentials (username, API token, server URL)

---

## Step 1: Prepare Your Credentials

Your SurveyCTO credentials are needed for production. You should have:

```
SCTO_SERVER = https://your-instance.surveycto.com
SCTO_USER = your-username
SCTO_PASS = your-api-token
```

**Where to find them:**
- **Server URL:** Your SurveyCTO instance URL
- **Username:** Your SurveyCTO login username
- **API Token:** SurveyCTO Settings → API → Create token (or use your password)

Keep these handy - you'll need them in Step 5.

---

## Step 2: Verify Your GitHub Repo

1. Go to your repository: https://github.com/ajolex/rct_field_flow
2. Make sure all your code is pushed:
   ```powershell
   # In your terminal, check status
   git status
   # Should show: "nothing to commit, working tree clean"
   
   # If not, commit and push:
   git add .
   git commit -m "Final code before Streamlit Cloud deployment"
   git push origin master
   ```

---

## Step 3: Go to Streamlit Cloud

**In your web browser:**

1. Go to: **https://streamlit.io/cloud**
   
2. You should see a login button. Click **"Sign in"**

3. Click **"Continue with GitHub"**

4. Authorize Streamlit to access your GitHub repos

---

## Step 4: Create New App

After signing in:

1. Click **"New app"** (usually top-right button)

2. You'll see a form to fill in:

   | Field | Value |
   |-------|-------|
   | **Repository** | `ajolex/rct_field_flow` |
   | **Branch** | `master` |
   | **Main file path** | `rct_field_flow/app.py` |

3. Click **"Deploy"**

The deployment will start! This takes 2-5 minutes.

---

## Step 5: Add Your Secrets

Once deployment finishes (you'll see your app URL):

1. Click the **three-dot menu** (top-right of app)

2. Select **"Settings"**

3. Look for **"Secrets"** section on the left menu

4. Paste your credentials in the text box:

```toml
scto_server = "https://your-instance.surveycto.com"
scto_user = "your-username"
scto_pass = "your-api-token"
```

5. Click **"Save"**

---

## Step 6: Test Your App

1. Go back to your app (click the app name or logo)

2. The app will restart with your secrets

3. Test the features:
   - [ ] Try uploading data
   - [ ] Try SurveyCTO integration
   - [ ] Test randomization
   - [ ] Test case assignment

---

## That's It! 🎉

Your app is now live on the internet at: **https://share.streamlit.io/ajolex/rct_field_flow/master**

### Your Public URL
You can share this URL with your team:
```
https://share.streamlit.io/ajolex/rct_field_flow/master
```

(The actual URL format may vary based on Streamlit's current setup)

---

## Troubleshooting

### "Module not found" error?
- Check that `rct_field_flow/app.py` exists ✓
- Verify all imports in `app.py` are correct ✓
- This should already be fixed

### "SurveyCTO not connecting"?
1. Check your secrets are correct
2. Verify server URL format: `https://...surveycto.com`
3. Make sure API token is valid
4. Check Streamlit Cloud logs (Settings → Logs)

### App is slow?
- First deploy takes longer
- Streamlit Cloud caches things
- Give it 30 seconds to load

### Need to update the app?
Simply push new code to GitHub:
```powershell
git add .
git commit -m "Your changes"
git push origin master
```
Streamlit Cloud auto-deploys! ✅

---

## 🎯 You're Done!

Your RCT Field Flow app is now:
- ✅ Live on the internet
- ✅ Accessible to your team
- ✅ Automatically backed up (on GitHub)
- ✅ Production-ready

### Next Steps
1. Share the URL with your team
2. Train them on how to use it
3. Monitor performance
4. Collect feedback

---

## Support & Help

- **Streamlit Cloud docs:** https://docs.streamlit.io/streamlit-cloud
- **Your deployment:** https://share.streamlit.io/ajolex/rct_field_flow/master
- **GitHub repo:** https://github.com/ajolex/rct_field_flow

---

## 📞 Common Issues

| Problem | Solution |
|---------|----------|
| Can't see "New app" button | Make sure you're logged in |
| App won't deploy | Check main file path: `rct_field_flow/app.py` |
| SurveyCTO not working | Add secrets in Settings → Secrets |
| App is crashing | Check Streamlit Cloud logs for errors |
| Forgot to add secrets | Go back to Settings → Secrets and add them |

---

**Congratulations! Your app is live!** 🚀

Questions? Check the troubleshooting section or GitHub discussions.
