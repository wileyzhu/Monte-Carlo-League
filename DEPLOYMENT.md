# Cloud Deployment Guide for Worlds Tournament Simulator

## 1. Heroku (Free/Easy) ⭐ RECOMMENDED

### Prerequisites
- Git installed
- Heroku CLI installed: https://devcenter.heroku.com/articles/heroku-cli

### Quick Deploy Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-worlds-simulator
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

4. **Open Your App**
   ```bash
   heroku open
   ```

### Files Created for Heroku
- `Procfile` - Tells Heroku how to run your app
- `runtime.txt` - Specifies Python version
- Updated `app.py` - Uses PORT environment variable

### Heroku Features
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Easy scaling
- ✅ Built-in monitoring
- ✅ Custom domain support

---

## 2. Railway (Modern Alternative)

### Quick Deploy
1. Visit https://railway.app
2. Connect your GitHub repository
3. Railway auto-detects Python and deploys
4. No configuration files needed!

### Railway Features
- ✅ $5/month free credit
- ✅ Faster than Heroku
- ✅ Better developer experience
- ✅ Automatic deployments from Git

---

## 3. Render (Free Tier)

### Deploy Steps
1. Visit https://render.com
2. Connect GitHub repository
3. Choose "Web Service"
4. Use these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`

### Render Features
- ✅ Free tier (with limitations)
- ✅ Automatic SSL
- ✅ Global CDN
- ✅ Easy custom domains

---

## 4. Vercel (Serverless)

### Setup
1. Install Vercel CLI: `npm i -g vercel`
2. Create `vercel.json`:
   ```json
   {
     "version": 2,
     "builds": [
       {
         "src": "app.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "app.py"
       }
     ]
   }
   ```
3. Deploy: `vercel --prod`

### Vercel Features
- ✅ Generous free tier
- ✅ Serverless (scales to zero)
- ✅ Global edge network
- ✅ Instant deployments

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
| Heroku | 550 hours/month | $7/month | Beginners |
| Railway | $5 credit/month | $5/month | Modern apps |
| Render | 750 hours/month | $7/month | Static + API |
| Vercel | Generous limits | $20/month | Serverless |
| Cloud Run | 2M requests/month | Pay-per-use | Variable traffic |
| App Runner | Pay-per-use | $0.007/hour | AWS ecosystem |

**Recommendation**: Start with Heroku or Railway for simplicity, then migrate to Cloud Run or Vercel for better performance/cost as you scale.