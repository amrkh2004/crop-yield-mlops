# 🐥 Deliverable 07: Canary Rollout & Nginx Traffic Manager

This directory contains the Canary deployment infrastructure using Nginx weighted reverse proxying between **Blue (v1)** and **Green (v2)** service instances.

---

## 📅 Canary Rollout Stages Schedule

| Stage | Duration | Blue (v1) Weight | Green (v2) Weight | Health & Quality Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | 30 minutes | `95` | `5` | p95 latency stable, MAE $\le$ baseline, 0% error rate |
| **Stage 2** | 1 hour | `80` | `20` | Memory & CPU stable, 0% error rate |
| **Stage 3** | 2 hours | `50` | `50` | Full traffic load distribution test |
| **Stage 4** | Permanent | `0` | `100` | Full promotion to v2 |

---

## 🛠️ Step-by-Step Traffic Promotion

To transition from Stage 1 (95/5) to Stage 2 (80/20):
1. Edit `nginx.conf`:
   ```nginx
   upstream ride_service_backend {
       server blue_service:8000 weight=80;
       server green_service:8000 weight=20;
   }
   ```
2. Test configuration syntax:
   ```bash
   docker exec -it canary_nginx_proxy nginx -t
   ```
3. Reload Nginx without downtime:
   ```bash
   docker exec -it canary_nginx_proxy nginx -s reload
   ```

---

## 🚨 Emergency Rollback Procedures

If MAE degrades or error rate exceeds 0.1%:

### Option A: Immediate Traffic Shift to Blue (Zero Downtime)
Update `nginx.conf` weight to `100/0` and reload:
```bash
docker exec -it canary_nginx_proxy nginx -s reload
```

### Option B: Stop Green Service Container
```bash
docker compose -f docker/canary/docker-compose.yml stop green_service
```
Nginx will automatically failover 100% of incoming traffic to `blue_service`.
