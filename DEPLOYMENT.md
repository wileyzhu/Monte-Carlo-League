# Cloud Deployment Guide for Worlds Tournament Simulator

## 1. Railway (Free Credit) ⭐ RECOMMENDED

### Prerequisites
- Git installed
- GitHub account

### Quick Deploy Steps

1. **Visit Railway**
   - Go to https://railway.app
   - Sign up with GitHub

2. **Deploy from GitHub**
   - Click "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Python and deploys!

3. **Alternative: CLI Deploy**
   ```bash
   npm install -g @railway/cli
   railway login
   railway deploy
   ```

### Railway Features
- ✅ $5 free credit monthly (no credit card required)
- ✅ Faster deployments than Heroku
- ✅ Automatic HTTPS
- ✅ Built-in monitoring
- ✅ Easy environment variables
- ✅ Custom domains

---

## 2. Render (Free Tier)

### Quick Deploy
1. Visit https://render.com
2. Connect GitHub repository
3. Choose "Web Service"
4. Use these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`

### Render Features
- ✅ Free tier (750 hours/month)
- ✅ Automatic SSL
- ✅ Global CDN
- ✅ Easy custom domains
- ✅ No credit card required

---

## 3. Vercel (Serverless Free Tier)

### Deploy Steps
1. Install Vercel CLI: `npm i -g vercel`
2. Deploy: `vercel --prod`
3. The `vercel.json` file is already configured!

### Vercel Features
- ✅ Generous free tier
- ✅ Serverless (scales to zero)
- ✅ Global edge network
- ✅ Instant deployments
- ✅ No credit card required

---

## 4. Heroku (Paid Only)

### Setup
1. **Account Verification Required**: Must add credit card
2. **Minimum Cost**: $7/month per dyno
3. Deploy steps:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Heroku Features
- ❌ No free tier (discontinued Nov 2022)
- ✅ Mature platform
- ✅ Extensive add-ons
- ✅ Easy scaling
- 💰 Starts at $7/month

---

## 5. Google Cloud Run (Pay-per-use)

### Setup
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   ```

2. Deploy:
   ```bash
   gcloud run deploy worlds-simulator --source .
   ```

### Cloud Run Features
- ✅ Pay only when used
- ✅ Scales to zero
- ✅ Handles traffic spikes
- ✅ Google's infrastructure

---

## 6. AWS App Runner (Managed)

### Setup
1. Create `apprunner.yaml`:
   ```yaml
   version: 1.0
   runtime: python3
   build:
     commands:
       build:
         - pip install -r requirements.txt
   run:
     runtime-version: 3.11
     command: python app.py
     network:
       port: 3000
   ```

2. Deploy via AWS Console or CLI

### App Runner Features
- ✅ Fully managed
- ✅ Auto-scaling
- ✅ Load balancing included
- ✅ VPC support

---

## Environment Variables (All Platforms)

Set these if needed:
- `FLASK_ENV=production`
- `PYTHONPATH=/app/src`

---

## Troubleshooting

### Common Issues
1. **Port binding errors**: Make sure app uses `PORT` env variable
2. **Module import errors**: Check `PYTHONPATH` or relative imports
3. **Memory limits**: Optimize data loading for smaller instances
4. **Cold starts**: Consider keeping one instance warm

### Performance Tips
- Use gunicorn for production: `pip install gunicorn`
- Update Procfile: `web: gunicorn app:app`
- Enable gzip compression
- Cache static assets

---

## Cost Comparison

| Platform | Free Tier | Paid Starting | Best For |
|----------|-----------|---------------|----------|
| Railway | $5 credit/month | $5/month | Modern apps, beginners |
| Render | 750 hours/month | $7/month | Traditional hosting |
| Vercel | Generous limits | $20/month | Serverless apps |
| Heroku | ❌ None | $7/month | Enterprise/legacy |
| Cloud Run | 2M requests/month | Pay-per-use | Variable traffic |
| App Runner | Pay-per-use | $0.007/hour | AWS ecosystem |

**Recommendation**: Start with Railway for the easiest experience, or Render for traditional hosting. Vercel is great for serverless. Avoid Heroku unless you need their specific features and don't mind paying.