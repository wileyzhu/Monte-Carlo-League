#!/bin/bash

# Heroku Deployment Script for Worlds Tournament Simulator

echo "🚀 Starting Heroku deployment..."

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI not found. Please install it first:"
    echo "   https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Check if user is logged in
if ! heroku auth:whoami &> /dev/null; then
    echo "🔐 Please login to Heroku first:"
    heroku login
fi

# Get app name from user or use default
read -p "Enter your Heroku app name (or press Enter for 'worlds-tournament-sim'): " APP_NAME
APP_NAME=${APP_NAME:-worlds-tournament-sim}

echo "📱 Creating Heroku app: $APP_NAME"

# Create Heroku app (will fail gracefully if exists)
heroku create $APP_NAME 2>/dev/null || echo "App $APP_NAME already exists, continuing..."

# Add Heroku remote if not exists
if ! git remote get-url heroku &> /dev/null; then
    heroku git:remote -a $APP_NAME
fi

# Deploy to Heroku
echo "📦 Deploying to Heroku..."
git add .
git commit -m "Deploy Worlds Tournament Simulator to Heroku" || echo "No changes to commit"
git push heroku main

# Open the app
echo "🎉 Deployment complete!"
echo "🌐 Opening your app..."
heroku open -a $APP_NAME

echo "✅ Your Worlds Tournament Simulator is now live!"
echo "📊 View logs: heroku logs --tail -a $APP_NAME"
echo "⚙️  Manage app: https://dashboard.heroku.com/apps/$APP_NAME"