# 🚂 Railway Deployment - Step by Step

## Why Railway is Better Than Vercel for Your App

Your Flask app is getting 500 errors on Vercel because:
- ❌ Vercel is serverless (stateless)
- ❌ File system access is limited
- ❌ Complex Flask routing doesn't work well
- ❌ Template rendering has issues

Railway is perfect because:
- ✅ Traditional hosting (like Heroku)
- ✅ Full file system access
- ✅ Works perfectly with Flask
- ✅ $5 free credit monthly
- ✅ No configuration needed

## 🚀 Deploy to Railway (Web Interface - Easiest)

### Step 1: Push to GitHub (if not already)
```bash
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### Step 2: Deploy via Railway Web
1. Go to **https://railway.app**
2. Click **"Start a New Project"**
3. Choose **"Deploy from GitHub repo"**
4. Sign in with GitHub
5. Select your repository
6. Railway will automatically:
   - Detect it's a Python app
   - Install from `requirements.txt`
   - Run `python app.py`
   - Give you a live URL!

### Step 3: Get Your Live URL
- Railway will provide a URL like: `https://your-app-name.railway.app`
- Your tournament simulator will be live immediately!

## 🛠️ Alternative: CLI Method

If you want to use CLI:
```bash
# Login (opens browser)
railway login

# Initialize project
railway init

# Deploy
railway deploy

# Get your URL
railway domain
```

## 🎯 What Railway Does Automatically

1. **Detects Python**: Sees your `requirements.txt`
2. **Installs Dependencies**: Runs `pip install -r requirements.txt`
3. **Starts App**: Runs `python app.py`
4. **Provides HTTPS**: Automatic SSL certificate
5. **Auto-deploys**: Updates when you push to GitHub

## 💰 Cost

- **Free**: $5 credit monthly (enough for your app)
- **No Credit Card**: Required only if you exceed free tier
- **Usage-based**: Only pay for what you use

## 🔧 If You Need Environment Variables

In Railway dashboard:
- Go to your project
- Click "Variables" tab
- Add any needed environment variables

## 📊 Monitoring

Railway provides:
- Real-time logs
- Resource usage metrics
- Deployment history
- Easy rollbacks

---

## 🎉 Expected Result

After deployment, you'll have:
- Live URL for your tournament simulator
- All features working (Play-in, Swiss, Elimination)
- Fast loading times
- Automatic HTTPS
- Easy updates via Git push

**Your app will work perfectly on Railway!** 🚂

---

## 🌐 Custom Domain Setup (Optional)

### After Deployment
1. **Buy a domain** (suggestions: `worldssim.com`, `lolworlds.app`, `tournamentpro.com`)
2. **In Railway dashboard**:
   - Go to your project → Settings → Domains
   - Click "Custom Domain"
   - Enter your domain name
3. **Configure DNS** at your domain registrar:
   ```
   Type: CNAME
   Name: www (or @)
   Value: your-app.railway.app
   ```
4. **Wait for DNS propagation** (24-48 hours)
5. **SSL is automatic** - Railway handles HTTPS certificates!

### Domain Registrar Recommendations
- **Namecheap**: $8-12/year, developer-friendly
- **Porkbun**: $7-10/year, great pricing
- **Cloudflare**: At-cost pricing, excellent features

### Professional Domain Ideas
- `worldssim.com` - Direct and memorable
- `tournament.yourname.com` - Personal branding
- `esportssim.net` - Broader esports focus
- `montecarloleague.com` - Technical reference

**Cost**: ~$10/year for domain + Railway hosting (free tier available)