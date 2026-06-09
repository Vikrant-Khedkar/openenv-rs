# Build from the repo root: docker build -f envs/echo/Dockerfile -t echo-env .
FROM rust:1.87-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p echo-env

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/echo-env /usr/local/bin/echo-env
EXPOSE 8000
ENV PORT=8000
CMD ["echo-env"]
