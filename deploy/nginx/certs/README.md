# HTTPS certificates

The production nginx config expects a fullchain and private key at:

```
/etc/nginx/certs/fullchain.pem
/etc/nginx/certs/privkey.pem
```

In `docker-compose.yml` the host directory `deploy/nginx/certs` is mounted
read-only into the web container, so place the certificates here.

## Let's Encrypt (recommended)

```bash
docker run --rm -it -v "$PWD/deploy/nginx/certs:/etc/letsencrypt/live/omniforge" \
  certbot/certbot certonly --standalone -d your-domain.example.com
# then rename/symlink:
#   fullchain.pem -> deploy/nginx/certs/fullchain.pem
#   privkey.pem   -> deploy/nginx/certs/privkey.pem
```

Or terminate TLS at your cloud load balancer / ingress and skip the 443
block (use `deploy/nginx/nginx-http.conf`).

## Self-signed (testing only)

```bash
openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
  -keyout deploy/nginx/certs/privkey.pem \
  -out deploy/nginx/certs/fullchain.pem \
  -subj "/CN=localhost"
```
