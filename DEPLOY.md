# Deploying to Railway

## Prerequisites
- GitHub account with this repo pushed (public or private)
- [Railway account](https://railway.com) (free tier: $5/mo credit)
- Railway CLI installed: `npm i -g @railway/cli`

## Step 1: Seed your data

Railway deploys from git, but your DB and models are gitignored (too large).
Copy them into `seed/` so the first deploy can bootstrap:

```bash
mkdir seed
cp nba.db seed/nba.db
cp -r models seed/models
```

Commit and push `seed/` to your repo. The startup script copies these to
the persistent volume on first boot, then you can remove `seed/` from the
repo to save space.

## Step 2: Create Railway project

```bash
railway login
railway init          # creates a new project
railway link          # links this directory to it
```

## Step 3: Add a persistent volume

In the Railway dashboard (or CLI):
1. Go to your service → Settings → Volumes
2. Add a volume mounted at `/data`

This is where your SQLite DB and model files live across deploys.

## Step 4: Set environment variables

In Railway dashboard → Variables (or via CLI):

```bash
railway variables set NBA_DATABASE_URL="sqlite:////data/nba.db"
railway variables set NBA_MODELS_DIR="/data/models"
railway variables set NBA_BALLDONTLIE_API_KEY="<your-api-key>"
railway variables set NBA_SECRET_KEY="<generate-a-random-string>"
```

## Step 5: Deploy

```bash
railway up
```

Or push to GitHub — Railway auto-deploys on every push if you connected
your repo in the dashboard (Settings → Source → Connect GitHub repo).

Your app will be live at `https://<project>.up.railway.app`.

## Step 6: Set up daily cron (optional)

For automatic data refresh, create a second service in the same project:

1. Railway dashboard → New Service → from same repo
2. Set the start command to: `python -m scripts.daily_cron`
3. Under Settings → Cron Schedule: `0 16 * * *` (4 PM UTC / 12 PM ET)
4. Set the same environment variables as above
5. Attach the same persistent volume at `/data`

## After first deploy

Once the app boots and the seed data is copied to `/data`, you can remove
the `seed/` directory from your repo to save space:

```bash
rm -rf seed
git add -A && git commit -m "remove seed data (already on volume)"
git push
```

## Custom domain (optional)

Railway dashboard → Settings → Networking → Add custom domain.
Point a CNAME record from your domain to Railway's provided target.

## Monitoring

- **Logs:** Railway dashboard → your service → Logs tab
- **Metrics:** CPU, memory, network visible in the Metrics tab
- **Health:** The `/` route serves as the healthcheck endpoint
