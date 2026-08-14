# tgbot Go

这是 tgbot 的 Go 版本，面向 Ubuntu VPS + systemd 部署。

## 功能

- Telegram 长轮询
- 用户权限控制
- 番号搜索、磁力链接和 ED2K 链接添加到 Alist
- 多搜索 API 并行搜索及 Sukebei 搜索
- `/clean` 清理小文件和空目录
- `/classify` 自动整理文件和目录
- `/setdir` 切换下载目录
- `/refresh` 刷新 Alist
- `/notify` 控制任务和清理通知
- `/reload_config` 热加载配置
- HTTP/HTTPS 和 SOCKS5 代理
- 定时清理、429 重试和状态持久化

## 文件结构

```text
cmd/tgbot/main.go                 Go 程序入口
.env.go.example                   配置模板
deploy/systemd/tgbot.service     systemd 服务文件
.github/workflows/go.yml          Go 编译检查
.github/workflows/release-go.yml  Release 构建
```

## 配置

复制配置模板：

```bash
cp .env.go.example .env
chmod 600 .env
nano .env
```

必填变量：

```dotenv
TELEGRAM_TOKEN=你的TelegramBotToken
ALIST_BASE_URL=https://alist.example.com
ALIST_TOKEN=你的AlistToken
ALIST_OFFLINE_DIRS=/downloads
JAV_SEARCH_APIS=https://example.com/api
ALLOWED_USER_IDS=123456789
```

常用可选变量：

```dotenv
PROXY_URL=socks5://127.0.0.1:7891
CLEAN_INTERVAL_MINUTES=60
SIZE_THRESHOLD=100
PREFERRED_KEYWORDS=中字,无码
CUSTOM_CATEGORIES=动漫:anime,动画;中文:chinese
SYSTEM_FOLDERS=/JAV,/其他
CLEAN_BATCH_SIZE=500
CLEAN_REQUEST_INTERVAL_MS=200
MAX_CONCURRENT_REQUESTS=20
EXCLUDE_SUFFIXES=.nfo,.txt
STATE_FILE=/var/lib/tgbot/state.json
```

程序会按以下顺序寻找配置文件：

1. `TGBOT_ENV_FILE` 指定的文件
2. 当前目录的 `.env`、`.env.go` 或 `tgbot.env`
3. 可执行文件目录的 `.env`
4. `/etc/tgbot/tgbot.env`

已经存在的系统环境变量优先级最高，不会被配置文件覆盖。

## 直接运行

需要 Go 1.22 或更高版本：

```bash
sudo apt update
sudo apt install -y golang-go git
git clone -b go-vps-systemd https://github.com/jxjhheric/tgbot.git
cd tgbot
go mod tidy
go build -trimpath -o tgbot ./cmd/tgbot
cp .env.go.example .env
nano .env
chmod 600 .env
./tgbot
```

停止程序使用 `Ctrl+C`。

## systemd 部署

推荐使用 systemd 管理进程：

```bash
sudo useradd --system --home /var/lib/tgbot --create-home tgbot
sudo mkdir -p /opt/tgbot /etc/tgbot /var/lib/tgbot
sudo install -m 0755 tgbot /opt/tgbot/tgbot
sudo cp .env /etc/tgbot/tgbot.env
sudo chmod 600 /etc/tgbot/tgbot.env
sudo cp deploy/systemd/tgbot.service /etc/systemd/system/tgbot.service
sudo systemctl daemon-reload
sudo systemctl enable --now tgbot
```

查看状态和日志：

```bash
sudo systemctl status tgbot
sudo journalctl -u tgbot -f
```

修改配置后重启：

```bash
sudo systemctl restart tgbot
```

也可以使用 Telegram 命令 `/reload_config` 重新读取配置。修改清理间隔、代理或搜索 API 后，建议同时重启服务。

## 直接下载可执行文件

Ubuntu amd64 可执行文件可从 Release 下载：

```bash
wget https://github.com/jxjhheric/tgbot/releases/download/go-v1.0.0/tgbot-linux-amd64.tar.gz
tar -xzf tgbot-linux-amd64.tar.gz
cd tgbot-linux-amd64
chmod +x tgbot
```

压缩包包含可执行文件、配置模板、systemd 文件和本说明文档。

## 代理格式

HTTP/HTTPS：

```dotenv
PROXY_URL=http://127.0.0.1:7890
```

带认证的 HTTP/HTTPS：

```dotenv
PROXY_URL=http://username:password@127.0.0.1:7890
```

SOCKS5：

```dotenv
PROXY_URL=socks5://127.0.0.1:7891
```

## GitHub Actions

推送到 `go-vps-systemd` 会运行 Go 测试和 Linux amd64 编译。推送 `go-v*` 标签会自动创建公开 Release：

```bash
git tag go-v1.0.1
git push origin go-v1.0.1
```
