# Multi-stage: build the Vue SPA, then serve it with nginx and proxy the API.

FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
# Same-origin API: an empty VITE_API_BASE makes the SPA call relative URLs
# (/chat, /knowledge, ...) that nginx proxies to the backend container.
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:1.27-alpine
# HTTPS config by default; build with --build-arg NGINX_CONF=nginx-http.conf
# for a no-certificate local test deployment.
ARG NGINX_CONF=nginx.conf
COPY deploy/nginx/${NGINX_CONF} /etc/nginx/conf.d/default.conf
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html
EXPOSE 80 443
