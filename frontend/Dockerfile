# ── Frontend Dockerfile ───────────────────────────────────────────────────────
FROM nginx:1.25-alpine

LABEL maintainer="ml-platform-team"
LABEL service="frontend"

# Copy static files
COPY index.html /usr/share/nginx/html/
COPY style.css  /usr/share/nginx/html/
COPY app.js     /usr/share/nginx/html/

# Nginx config for SPA
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -qO- http://localhost/ || exit 1
