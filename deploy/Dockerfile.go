FROM golang:1.22 AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY cmd ./cmd
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /out/tgbot ./cmd/tgbot

FROM debian:bookworm-slim
RUN useradd --system --home /var/lib/tgbot --create-home tgbot
COPY --from=build /out/tgbot /opt/tgbot/tgbot
USER tgbot
WORKDIR /opt/tgbot
ENTRYPOINT ["/opt/tgbot/tgbot"]
