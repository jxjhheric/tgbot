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

## Cloudflare Worker 部署

Worker 版本位于 `worker/`，使用 Telegram Webhook，不需要常驻 Python 进程。它保留番号搜索、磁力/ed2k 添加、目录切换、Alist 刷新、清理、分类和通知功能。

```bash
cd worker
npx wrangler login
npx wrangler secret put TELEGRAM_TOKEN
npx wrangler secret put ALIST_TOKEN
npx wrangler secret put WEBHOOK_SECRET
npx wrangler deploy
```

普通配置可写入 `worker/.dev.vars`，或在 Cloudflare Worker Settings 中配置同名变量：`ALIST_BASE_URL`、`ALIST_OFFLINE_DIRS`、`JAV_SEARCH_APIS`、`ALLOWED_USER_IDS`、`SIZE_THRESHOLD`、`PREFERRED_KEYWORDS`、`CUSTOM_CATEGORIES`、`SYSTEM_FOLDERS`、`EXCLUDE_SUFFIXES`、`ADMIN_CHAT_ID`。

可选地创建 KV 保存当前目录和通知开关：

```bash
npx wrangler kv namespace create STATE
```

将命令输出的 namespace id 写入 `worker/wrangler.toml` 的 `kv_namespaces` 配置后重新部署。部署完成后设置 Telegram Webhook：

```bash
curl -X POST "https://api.telegram.org/bot<TELEGRAM_TOKEN>/setWebhook" \
  -d "url=https://<WORKER_DOMAIN>/webhook" \
  -d "secret_token=<WEBHOOK_SECRET>"
```

Worker 环境的出站请求由 Cloudflare `fetch` 发起，Python 版本的 `PROXY_URL` 不适用于 Worker；如必须经代理访问，需要在网络侧提供可公开访问的代理网关或改用 Python Docker 部署。

## Go + VPS + systemd

`go-vps-systemd` 分支包含 Go 版本，使用 Telegram 长轮询，支持代理、Alist、搜索、下载、清理、分类、目录切换和定时任务。配置模板见 `.env.go.example`。Go 版本会自动读取当前目录的 `.env`，也支持通过 `TGBOT_ENV_FILE` 指定配置文件。

在 VPS 上编译运行：

```bash
sudo apt update
sudo apt install -y golang-go git
sudo useradd --system --home /var/lib/tgbot --create-home tgbot
sudo mkdir -p /opt/tgbot /etc/tgbot /var/lib/tgbot
sudo cp .env.go.example /etc/tgbot/tgbot.env
sudo chmod 600 /etc/tgbot/tgbot.env
go build -trimpath -o /tmp/tgbot ./cmd/tgbot
sudo install -m 0755 /tmp/tgbot /opt/tgbot/tgbot
sudo cp deploy/systemd/tgbot.service /etc/systemd/system/tgbot.service
sudo systemctl daemon-reload
sudo systemctl enable --now tgbot
sudo systemctl status tgbot
```

首次编译前执行 `go mod tidy` 生成依赖校验文件，再执行上面的 `go build`。

编辑 `/etc/tgbot/tgbot.env` 后执行 `sudo systemctl restart tgbot`。日志使用 `journalctl -u tgbot -f` 查看。`PROXY_URL` 支持 HTTP/HTTPS 和 SOCKS5，例如 `socks5://127.0.0.1:7891`。

直接运行二进制时，可以这样配置：

```bash
cp .env.go.example .env
chmod 600 .env
./tgbot
```

已有系统环境变量不会被 `.env` 覆盖；systemd 模式优先使用 `/etc/tgbot/tgbot.env`。
