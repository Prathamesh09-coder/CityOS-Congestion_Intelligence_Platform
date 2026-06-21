FROM node:20-slim AS builder

WORKDIR /app

RUN npm install -g npm@11

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Run stage
FROM node:20-slim

WORKDIR /app

# Copy the built output from the builder stage
COPY --from=builder /app/.output ./.output

# Set host/port environment variables
ENV HOST=0.0.0.0
ENV PORT=3000

# Expose port
EXPOSE 3000

# Start server
CMD ["node", ".output/server/index.mjs"]

