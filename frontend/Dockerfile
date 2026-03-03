# ── Stage 1: Build ────────────────────────────────────────
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# ── Stage 2: Serve ────────────────────────────────────────
FROM nginx:alpine
COPY nginx.conf.template /etc/nginx/nginx.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
# API_URL is substituted into the nginx config at startup.
# Default: http://api:8000 (Docker Compose). Override for Railway/cloud.
ENV API_URL=http://api:8000
# envsubst replaces only $API_URL, leaving nginx variables ($host, $uri, etc.) intact.
CMD ["/bin/sh", "-c", "envsubst '$API_URL' < /etc/nginx/nginx.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
