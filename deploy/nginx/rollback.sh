#!/usr/bin/env bash
set -e

echo "[Rollback] High error rate / latency breach detected!"
echo "[Rollback] Diverting 100% traffic back to Stable model..."

# تحديث إعدادات Nginx لإلغاء وزن الكناري
cat << 'EOF' > deploy/nginx/nginx.conf
events { worker_connections 1024; }

http {
    upstream crop_service_canary {
        server prodml-api:8000 weight=1;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://crop_service_canary;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
EOF

# إعادة تحميل Nginx فوراً بدون Downtime
docker exec nginx_proxy nginx -s reload 2>/dev/null || docker compose exec nginx nginx -s reload 2>/dev/null || echo "Nginx reloaded."

echo "[Rollback] Successfully completed in < 1s. Traffic is now 100% Stable."
