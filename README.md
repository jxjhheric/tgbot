# tgbot

## Docker 代理

设置 `PROXY_URL` 后，代理会用于 Telegram Bot API、搜索 API 和 Alist 请求；不设置时保持直连。

```bash
docker run -d \
  --name tgbot \
  -e PROXY_URL=http://host.docker.internal:7890 \
  -e TELEGRAM_TOKEN=your-token \
  -e ALIST_BASE_URL=https://alist.example.com \
  -e ALIST_TOKEN=your-alist-token \
  -e ALIST_OFFLINE_DIRS=/downloads \
  -e JAV_SEARCH_APIS=https://example.com/api \
  -e ALLOWED_USER_IDS=123456789 \
  ghcr.io/jxjhheric/tgbot:latest
```

支持带认证的 HTTP/HTTPS 代理，例如 `http://username:password@proxy.example.com:7890`。
