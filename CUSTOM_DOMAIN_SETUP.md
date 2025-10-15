# 🌐 Custom Domain Setup Guide

## Railway Custom Domain (Recommended)

### Step 1: Deploy to Railway First
1. Go to https://railway.app
2. Deploy your repository
3. Get your default Railway URL (e.g., `your-app.railway.app`)

### Step 2: Add Custom Domain in Railway
1. Go to your Railway project dashboard
2. Click on your service
3. Go to **"Settings"** tab
4. Click **"Domains"** section
5. Click **"Custom Domain"**
6. Enter your domain (e.g., `worldssim.com` or `tournament.yourname.com`)

### Step 3: Configure DNS Records
Add these DNS records at your domain registrar:

**For Root Domain (worldssim.com):**
```
Type: A
Name: @
Value: [Railway will provide the IP]
TTL: 300
```

**For Subdomain (tournament.yourname.com):**
```
Type: CNAME
Name: tournament
Value: your-app.railway.app
TTL: 300
```

### Step 4: SSL Certificate
Railway automatically provides SSL certificates for custom domains!

---

## Domain Name Suggestions

### Tournament-Themed Domains
- `worldssim.com`
- `lolworlds.app`
- `tournamentpro.com`
- `esportssim.net`
- `worldschampion.app`
- `leaguesim.com`

### Personal Branding
- `yourname-worlds.com`
- `tournament.yourname.com`
- `sim.yourname.dev`

### Creative Options
- `montecarloleague.com`
- `bayesianworlds.com`
- `probabilityesports.com`

---

## Where to Buy Domains

### Budget-Friendly Options
1. **Namecheap** - $8-12/year, good support
2. **Porkbun** - $7-10/year, developer-friendly
3. **Google Domains** - $12/year, simple interface
4. **Cloudflare** - At-cost pricing, great features

### Premium Options
1. **GoDaddy** - $15-20/year, well-known
2. **Domain.com** - $10-15/year, good tools

---

## Alternative Platforms Custom Domains

### Render Custom Domain
1. Go to Render dashboard
2. Select your service
3. Go to "Settings" → "Custom Domains"
4. Add your domain
5. Configure DNS as instructed

### Vercel Custom Domain
1. Go to Vercel dashboard
2. Select your project
3. Go to "Settings" → "Domains"
4. Add your domain
5. Follow DNS configuration

### Netlify Custom Domain
1. Go to Netlify dashboard
2. Select your site
3. Go to "Domain settings"
4. Add custom domain
5. Configure DNS records

---

## DNS Configuration Examples

### Cloudflare DNS Setup
```
Type: CNAME
Name: www
Content: your-app.railway.app
Proxy: Orange cloud (proxied)

Type: CNAME  
Name: @
Content: your-app.railway.app
Proxy: Orange cloud (proxied)
```

### Namecheap DNS Setup
```
Type: CNAME Record
Host: www
Value: your-app.railway.app
TTL: Automatic

Type: URL Redirect Record
Host: @
Value: https://www.yourdomain.com
```

---

## Free Domain Options

### Temporary/Development
- `your-app.railway.app` (Railway default)
- `your-app.onrender.com` (Render default)
- `your-app.vercel.app` (Vercel default)

### Free Subdomains
- **Freenom** - `.tk`, `.ml`, `.ga` domains (free but limited)
- **GitHub Pages** - `username.github.io/repo-name`
- **Netlify** - `app-name.netlify.app`

---

## Professional Setup Checklist

### Before Going Live
- [ ] Domain purchased and configured
- [ ] SSL certificate active (automatic on Railway)
- [ ] DNS propagation complete (24-48 hours)
- [ ] Test all routes work on custom domain
- [ ] Update any hardcoded URLs in your app
- [ ] Set up analytics (Google Analytics, etc.)

### SEO Optimization
- [ ] Add favicon.ico to static folder
- [ ] Update meta tags in templates
- [ ] Add sitemap.xml
- [ ] Configure robots.txt
- [ ] Set up Google Search Console

---

## Cost Breakdown

### Annual Costs
- **Domain**: $8-15/year
- **Railway Hosting**: $5/month ($60/year) after free credit
- **Total**: ~$75/year for professional setup

### Free Option
- Use Railway default domain: `your-app.railway.app`
- Cost: $0 (with $5 monthly free credit)

---

## Quick Setup Commands

### Check DNS Propagation
```bash
# Check if DNS has propagated
nslookup yourdomain.com
dig yourdomain.com

# Check SSL certificate
curl -I https://yourdomain.com
```

### Test Your Domain
```bash
# Test if your app responds
curl https://yourdomain.com

# Check response time
curl -w "@curl-format.txt" -o /dev/null -s https://yourdomain.com
```

---

## Troubleshooting

### Common Issues
1. **DNS not propagating**: Wait 24-48 hours
2. **SSL certificate issues**: Railway handles this automatically
3. **404 errors**: Check Railway domain configuration
4. **Mixed content warnings**: Ensure all assets use HTTPS

### Support Resources
- Railway Discord: https://discord.gg/railway
- Railway Docs: https://docs.railway.app/deploy/custom-domains
- DNS Checker: https://dnschecker.org/

---

## Recommended Setup

**For Professional Use:**
1. Buy domain from Namecheap ($10/year)
2. Deploy to Railway
3. Configure custom domain in Railway
4. Set up Cloudflare for CDN/security (optional)

**For Personal/Demo:**
1. Use Railway default domain (free)
2. Upgrade to custom domain later if needed

Your Worlds Tournament Simulator will look much more professional with a custom domain! 🚀