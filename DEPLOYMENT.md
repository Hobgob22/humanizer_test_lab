# Deployment Guide - Humanizer Test-Bench

This guide covers deploying the Humanizer Test-Bench using Docker with support for both development (with hot reload) and production environments.

## Prerequisites

- Docker Engine 20.10+ and Docker Compose 2.0+
- Make (optional, for using Makefile commands)
- SSL certificates (for production HTTPS)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Hobgob22/humanizer_test_lab.git
cd humanizer-testbench
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Development Mode (with Hot Reload)

```bash
# Using Make
make dev

# Or using Docker Compose directly
docker-compose --profile dev up
```

The development server will start at http://localhost:8501 with hot reload enabled. Any changes to files in the `src/` directory will automatically reload the application.

### 3. Production Mode

```bash
# Using Make
make prod

# Or using Docker Compose directly
docker-compose --profile prod up -d
```

The production server will start at http://localhost:8501

## Detailed Setup

### Environment Variables

Create a `.env` file with the following variables:

```env
# Authentication
APP_AUTH_KEY=your-secure-password

# API Keys
OPENAI_API_KEY=sk-...
HUMANIZER_OPENAI_API_KEY=sk-...
GPTZERO_API_KEY=...
SAPLING_API_KEY=...
GEMINI_API_KEY=...

# Optional Configuration
REHUMANIZE_N=5
ZERO_SHOT_THRESHOLD=0.10
MIN_WORDS_PARAGRAPH=15
MAX_ITER=5

# Worker Limits
HUMANIZER_MAX_WORKERS=50
GEMINI_MAX_WORKERS=5
DETECTOR_MAX_WORKERS=5
PARA_MAX_WORKERS=8
```

### File Structure

```
humanizer-testbench/
├── src/                    # Application source code
├── data/                   # Document folders
│   ├── ai_texts/          # AI-generated documents
│   ├── human_texts/       # Human-written documents
│   └── mixed_texts/       # Mixed documents
├── cache/                  # Detector cache (persisted)
├── logs/                   # Application logs
├── results/                # Benchmark results
├── nginx/                  # Nginx configuration
│   ├── nginx.conf         # Main config
│   └── ssl/               # SSL certificates
├── Dockerfile             # Production image
├── Dockerfile.dev         # Development image
├── docker-compose.yml     # Main compose file
├── docker-compose.override.yml  # Dev overrides
├── Makefile               # Convenience commands
└── .env                   # Environment variables
```

## Development Workflow

### Hot Reload Features

The development setup includes:
- **Automatic code reload** when Python files change
- **Volume mounting** for instant updates
- **Debug mode** enabled
- **CORS disabled** for easier testing

### Making Changes

1. Edit files in your local `src/` directory
2. Changes are automatically detected and reloaded
3. Refresh the browser to see updates

### Useful Development Commands

```bash
# View logs
make logs-dev

# Open shell in container
make shell

# Run CLI tool
make cli ARGS="--folder data/ai_texts --models gpt-4.1"

# Run tests
make test

# Format code
make format
```

## Production Deployment

### Basic Production Setup

1. **Build the production image:**
   ```bash
   make build-prod
   ```

2. **Start the production server:**
   ```bash
   make prod
   ```

3. **Verify health:**
   ```bash
   curl http://localhost:8501/_stcore/health
   ```

### Production with Nginx (Recommended)

1. **Add SSL certificates:**
   ```bash
   mkdir -p nginx/ssl
   cp /path/to/cert.pem nginx/ssl/
   cp /path/to/key.pem nginx/ssl/
   ```

2. **Update nginx.conf** with your domain:
   ```nginx
   server_name your-domain.com;
   ```

3. **Start with Nginx:**
   ```bash
   make prod-nginx
   ```

### Security Considerations

- The production image runs as non-root user
- Nginx includes security headers
- Rate limiting is configured
- SSL/TLS with modern ciphers

## Monitoring and Maintenance

### View Logs

```bash
# All logs
make logs

# Specific service
docker-compose logs -f humanizer-prod

# Last 100 lines
docker-compose logs --tail=100 humanizer-prod
```

### Monitor Resources

```bash
# Real-time stats
make monitor

# One-time snapshot
make stats
```

### Backup Data

```bash
# Create backup
make backup

# Manual backup
tar -czf backup-$(date +%Y%m%d).tar.gz data/ results/ cache/
```

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
make build-prod
make prod
```

## Troubleshooting

### Container Won't Start

1. Check logs: `make logs`
2. Verify .env file exists and has valid keys
3. Ensure ports are not in use: `lsof -i :8501`

### Hot Reload Not Working

1. Ensure using dev profile: `make dev`
2. Check file permissions
3. Verify volume mounts: `docker-compose config`

### Performance Issues

1. Increase Docker resources
2. Adjust worker limits in .env
3. Check disk space: `df -h`

### SSL Certificate Issues

1. Verify certificate paths in nginx/ssl/
2. Check certificate validity
3. Ensure proper permissions: `chmod 600 nginx/ssl/*`

## Advanced Configuration

### Custom Domain

1. Update `nginx/nginx.conf`:
   ```nginx
   server_name example.com www.example.com;
   ```

2. Set up DNS A records pointing to your server

3. Obtain SSL certificates (e.g., Let's Encrypt):
   ```bash
   certbot certonly --standalone -d example.com -d www.example.com
   ```

### Scaling

For high load, consider:
- Using external Redis for caching
- PostgreSQL instead of SQLite
- Multiple worker containers
- Load balancer setup

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to server
        run: |
          ssh user@server << 'EOF'
            cd /opt/humanizer-testbench
            git pull origin main
            make build-prod
            make prod
          EOF
```

## Support

For issues or questions:
1. Check the logs first
2. Review this documentation
3. Check the troubleshooting section
4. Create an issue in the repository