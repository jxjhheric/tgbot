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

## 下载包部署

以下步骤适用于 Ubuntu amd64 VPS。最新下载包只包含 `tgbot` 可执行文件，不需要在 VPS 上安装 Go；配置文件和 systemd 服务需要按下面步骤创建。

### 1. 下载并解压

```bash
sudo apt update
sudo apt install -y wget tar

cd /tmp
wget https://github.com/jxjhheric/tgbot/releases/download/go-v1.0.3/tgbot-linux-amd64.tar.gz
tar -xzf tgbot-linux-amd64.tar.gz
cd tgbot-linux-amd64
wget -O tgbot.env.example https://raw.githubusercontent.com/jxjhheric/tgbot/go-vps-systemd/.env.go.example
wget -O tgbot.service https://raw.githubusercontent.com/jxjhheric/tgbot/go-vps-systemd/deploy/systemd/tgbot.service
```

确认文件：

```bash
ls -lh
```

应当看到：

```text
tgbot
tgbot.env.example
tgbot.service
```

### 2. 创建运行用户和目录

systemd 会使用独立的 `tgbot` 用户运行程序：

```bash
sudo useradd --system --home /var/lib/tgbot --create-home tgbot 2>/dev/null || true
sudo mkdir -p /opt/tgbot /etc/tgbot /var/lib/tgbot
sudo install -m 0755 tgbot /opt/tgbot/tgbot
sudo chown -R tgbot:tgbot /opt/tgbot /var/lib/tgbot
```

### 3. 创建配置文件

复制配置模板：

```bash
sudo cp tgbot.env.example /etc/tgbot/tgbot.env
sudo nano /etc/tgbot/tgbot.env
```

至少填写以下变量：

```dotenv
TELEGRAM_TOKEN=你的TelegramBotToken
ALIST_BASE_URL=https://你的Alist地址
ALIST_TOKEN=你的AlistToken
ALIST_OFFLINE_DIRS=/downloads
JAV_SEARCH_APIS=https://你的搜索API地址
ALLOWED_USER_IDS=你的Telegram用户数字ID
```

常用配置示例：

```dotenv
PROXY_URL=http://127.0.0.1:7890
CLEAN_INTERVAL_MINUTES=60
SIZE_THRESHOLD=100
PREFERRED_KEYWORDS=中字,无码
CUSTOM_CATEGORIES=动漫:anime,动画;中文:chinese
SYSTEM_FOLDERS=/JAV,/其他
EXCLUDE_SUFFIXES=.nfo,.txt
```

保护配置文件：

```bash
sudo chown root:tgbot /etc/tgbot/tgbot.env
sudo chmod 640 /etc/tgbot/tgbot.env
sudo chown -R tgbot:tgbot /var/lib/tgbot
```

### 4. 安装 systemd 服务

```bash
sudo cp tgbot.service /etc/systemd/system/tgbot.service
sudo systemctl daemon-reload
sudo systemctl enable --now tgbot
```

`enable` 表示 VPS 重启后自动启动，`--now` 表示立即启动。

### 5. 检查运行状态

```bash
sudo systemctl status tgbot
```

持续查看日志：

```bash
sudo journalctl -u tgbot -f
```

正常启动时日志中会显示类似：

```text
loaded environment file: /etc/tgbot/tgbot.env
tgbot Go service started
```

### 6. 修改配置

编辑配置：

```bash
sudo nano /etc/tgbot/tgbot.env
```

推荐重启服务使全部配置生效：

```bash
sudo systemctl restart tgbot
```

也可以通过 Telegram 发送 `/reload_config` 热加载配置。修改代理、搜索 API 或清理间隔后，建议使用 `systemctl restart`。

### 7. 常用 systemd 命令

```bash
# 启动
sudo systemctl start tgbot

# 停止
sudo systemctl stop tgbot

# 重启
sudo systemctl restart tgbot

# 查看状态
sudo systemctl status tgbot

# 设置开机启动
sudo systemctl enable tgbot

# 取消开机启动
sudo systemctl disable tgbot

# 查看最近 100 行日志
sudo journalctl -u tgbot -n 100 --no-pager
```

### 8. 升级程序

可以使用一键升级脚本。脚本会自动获取最新的 `go-v*` Release，备份旧程序，替换二进制并重启服务；启动失败时会自动恢复备份：

```bash
curl -fsSL https://raw.githubusercontent.com/jxjhheric/tgbot/go-vps-systemd/deploy/upgrade.sh | sudo bash
```

脚本默认升级到最新版本。如果需要指定版本：

```bash
curl -fsSL https://raw.githubusercontent.com/jxjhheric/tgbot/go-vps-systemd/deploy/upgrade.sh | sudo env TGBOT_VERSION=go-v1.0.3 bash
```

升级前会在 `/opt/tgbot/tgbot.backup.YYYYMMDDHHMMSS` 保存旧程序。配置文件 `/etc/tgbot/tgbot.env`、systemd 服务和 `/var/lib/tgbot/state.json` 不会被覆盖。

### 9. 卸载服务

```bash
sudo systemctl disable --now tgbot
sudo rm -f /etc/systemd/system/tgbot.service
sudo systemctl daemon-reload
sudo rm -rf /opt/tgbot
sudo rm -rf /var/lib/tgbot
sudo rm -f /etc/tgbot/tgbot.env
sudo userdel tgbot 2>/dev/null || true
```

### 10. 常见问题

如果出现 `missing environment variables`：

```bash
sudo grep -v '^#' /etc/tgbot/tgbot.env
sudo systemctl restart tgbot
sudo journalctl -u tgbot -n 50 --no-pager
```

检查变量名是否拼写正确，等号两侧不要添加多余空格。`ALLOWED_USER_IDS` 必须填写 Telegram 数字用户 ID，多个 ID 用逗号分隔。

如果出现 `Permission denied`：

```bash
sudo chmod 0755 /opt/tgbot/tgbot
sudo chown tgbot:tgbot /opt/tgbot/tgbot
```

如果 Telegram 无响应，先查看日志，再检查 VPS 是否能访问 Telegram API：

```bash
curl -I https://api.telegram.org
```

如果网络需要代理，在 `/etc/tgbot/tgbot.env` 中配置 `PROXY_URL` 后重启服务。

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

推送到 `go-vps-systemd` 会运行 Go 测试和 Linux amd64 编译。推送 `go-v*` 标签会自动创建公开 Release，压缩包只包含 Linux amd64 的 `tgbot` 可执行文件：

```bash
git tag go-v1.0.3
git push origin go-v1.0.3
```
