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
# PORT: Railway sets this dynamically. Default 80 for Docker Compose.
ENV PORT=80
ENV API_URL=http://api:8000
# envsubst replaces only $PORT and $API_URL, leaving nginx vars ($host, $uri, etc.) intact.
CMD ["/bin/sh", "-c", "envsubst '$PORT $API_URL' < /etc/nginx/nginx.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
