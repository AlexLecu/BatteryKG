# Vue UI + same-origin reverse proxy for the API.
# Built by deploy/server/docker-compose.yml (context: web/, extra context "server").
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=server nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
