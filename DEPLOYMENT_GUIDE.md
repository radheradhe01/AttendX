# Artificial Consciousness Simulator - Deployment Guide

## Quick Start

### Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- **PostgreSQL 14+**
- **CUDA-capable GPU** (recommended)

### Installation Steps

1. **Clone the repository**:
```bash
git clone <repository-url>
cd artificial-consciousness-simulator
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install Node.js dependencies**:
```bash
npm install
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start the system**:
```bash
# Terminal 1: Start Python backend
python src/api/main.py

# Terminal 2: Start Next.js frontend
npm run dev
```

6. **Access the dashboard**:
Open http://localhost:3000 in your browser

## Detailed Setup

### Environment Configuration

Create a `.env` file with the following variables:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/consciousness_db
TIMESCALE_URL=postgresql://username:password@localhost:5432/consciousness_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WEBSOCKET_URL=ws://localhost:8000

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WEBSOCKET_URL=ws://localhost:8000

# Agent Configuration
AGENT_MEMORY_SIZE=256
AGENT_WORD_SIZE=64
AGENT_READ_HEADS=4
AGENT_WRITE_HEADS=1

# Evaluation Configuration
EVALUATION_OUTPUT_DIR=./evaluation_results
EVALUATION_LOG_LEVEL=INFO
```

### Database Setup

1. **Install PostgreSQL**:
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Windows
# Download from https://www.postgresql.org/download/windows/
```

2. **Install TimescaleDB**:
```bash
# Ubuntu/Debian
sudo apt-get install timescaledb-postgresql-14

# macOS
brew install timescaledb

# Windows
# Download from https://docs.timescale.com/install/latest/installation-windows/
```

3. **Create database**:
```sql
CREATE DATABASE consciousness_db;
\c consciousness_db;
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

### GPU Setup (Optional but Recommended)

1. **Install CUDA**:
```bash
# Follow NVIDIA CUDA installation guide for your system
# https://developer.nvidia.com/cuda-downloads
```

2. **Verify PyTorch GPU support**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Production Deployment

### Docker Deployment

1. **Build Docker images**:
```bash
# Build Python backend
docker build -f Dockerfile.backend -t consciousness-backend .

# Build Next.js frontend
docker build -f Dockerfile.frontend -t consciousness-frontend .
```

2. **Run with Docker Compose**:
```bash
docker-compose up -d
```

### Kubernetes Deployment

1. **Apply Kubernetes manifests**:
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/ingress.yaml
```

2. **Check deployment status**:
```bash
kubectl get pods -n consciousness
kubectl get services -n consciousness
```

### Cloud Deployment

#### AWS Deployment

1. **EC2 Setup**:
```bash
# Launch EC2 instance with GPU support
# Install dependencies
sudo apt-get update
sudo apt-get install python3.9 python3-pip nodejs npm postgresql

# Configure security groups for ports 3000, 8000, 5432
```

2. **RDS Setup**:
```bash
# Create RDS PostgreSQL instance
# Configure VPC security groups
# Update DATABASE_URL in .env
```

#### Google Cloud Deployment

1. **Compute Engine Setup**:
```bash
# Create VM instance with GPU
# Install dependencies
# Configure firewall rules
```

2. **Cloud SQL Setup**:
```bash
# Create Cloud SQL PostgreSQL instance
# Configure private IP
# Update connection string
```

#### Azure Deployment

1. **Virtual Machine Setup**:
```bash
# Create VM with GPU support
# Install dependencies
# Configure network security groups
```

2. **Azure Database Setup**:
```bash
# Create Azure Database for PostgreSQL
# Configure firewall rules
# Update connection string
```

## Monitoring and Maintenance

### Health Checks

1. **API Health Check**:
```bash
curl http://localhost:8000/health
```

2. **Database Health Check**:
```bash
psql $DATABASE_URL -c "SELECT 1;"
```

3. **Frontend Health Check**:
```bash
curl http://localhost:3000/api/health
```

### Logging

1. **Backend Logs**:
```bash
# View Python logs
tail -f logs/consciousness.log

# View API logs
tail -f logs/api.log
```

2. **Frontend Logs**:
```bash
# View Next.js logs
npm run logs
```

### Performance Monitoring

1. **System Metrics**:
```bash
# CPU usage
htop

# GPU usage
nvidia-smi

# Memory usage
free -h

# Disk usage
df -h
```

2. **Application Metrics**:
- Access the dashboard at http://localhost:3000
- Monitor consciousness metrics in real-time
- Check performance charts and trends

### Backup and Recovery

1. **Database Backup**:
```bash
# Create backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
psql $DATABASE_URL < backup_file.sql
```

2. **Model Backup**:
```bash
# Backup trained models
cp -r models/ backup_models_$(date +%Y%m%d_%H%M%S)/
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**:
```bash
# Find process using port
lsof -i :8000
lsof -i :3000

# Kill process
kill -9 <PID>
```

2. **Database Connection Issues**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Restart PostgreSQL
sudo systemctl restart postgresql

# Check connection
psql $DATABASE_URL -c "SELECT 1;"
```

3. **GPU Memory Issues**:
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size in configuration
# Edit src/core/dnc_memory.py
# Reduce memory_size parameter
```

4. **WebSocket Connection Issues**:
```bash
# Check WebSocket endpoint
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" http://localhost:8000/ws
```

### Performance Optimization

1. **Memory Optimization**:
```python
# In src/core/dnc_memory.py
memory_system = DNCMemorySystem(
    memory_size=128,  # Reduce from 256
    word_size=32,     # Reduce from 64
    num_read_heads=2  # Reduce from 4
)
```

2. **CPU Optimization**:
```python
# In src/core/meta_learning.py
meta_agent = MAMLAgent(
    adaptation_steps=3,  # Reduce from 5
    adaptation_lr=0.005  # Reduce from 0.01
)
```

3. **GPU Optimization**:
```python
# Enable mixed precision training
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

## Security Considerations

### API Security

1. **Authentication**:
```python
# Add JWT authentication
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

# Configure JWT
jwt_authentication = JWTAuthentication(
    secret="your-secret-key",
    lifetime_seconds=3600,
    tokenUrl="auth/jwt/login"
)
```

2. **Rate Limiting**:
```python
# Add rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
```

3. **CORS Configuration**:
```python
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Data Security

1. **Database Encryption**:
```sql
-- Enable encryption at rest
ALTER DATABASE consciousness_db SET encryption = 'on';
```

2. **Environment Variables**:
```bash
# Use secure environment variable management
# Never commit .env files to version control
echo ".env" >> .gitignore
```

3. **API Keys**:
```python
# Store API keys securely
import os
from cryptography.fernet import Fernet

# Encrypt sensitive data
key = Fernet.generate_key()
cipher_suite = Fernet(key)
```

## Scaling and Load Balancing

### Horizontal Scaling

1. **Multiple Backend Instances**:
```bash
# Run multiple backend instances
python src/api/main.py --port 8001 &
python src/api/main.py --port 8002 &
python src/api/main.py --port 8003 &
```

2. **Load Balancer Configuration**:
```nginx
# Nginx configuration
upstream consciousness_backend {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://consciousness_backend;
    }
}
```

### Vertical Scaling

1. **Resource Monitoring**:
```bash
# Monitor resource usage
htop
nvidia-smi
iostat -x 1
```

2. **Auto-scaling**:
```yaml
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: consciousness-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: consciousness-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Support and Community

### Getting Help

1. **Documentation**: Check the comprehensive documentation in `RESEARCH_DOCUMENTATION.md`
2. **Issues**: Report bugs and feature requests on GitHub Issues
3. **Discussions**: Join community discussions on GitHub Discussions
4. **Email**: Contact the development team at consciousness-simulator@example.com

### Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

### License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*This deployment guide provides comprehensive instructions for setting up and deploying the Artificial Consciousness Simulator. For additional support or questions, please refer to the project documentation or contact the development team.*
