# The dashboard as static files behind nginx. The API address is fixed at build time:
#   docker build -t flarenet-web --build-arg VITE_API_URL=http://localhost:8000 frontend
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_URL=
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:1.27-alpine
# the app uses hash routes (#/alerts), so no rewrite rules are needed
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
