#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${TGBOT_REPO:-jxjhheric/tgbot}"
VERSION="${TGBOT_VERSION:-}"
SERVICE="${TGBOT_SERVICE:-tgbot}"
INSTALL_DIR="${TGBOT_INSTALL_DIR:-/opt/tgbot}"
PROGRAM="${INSTALL_DIR}/tgbot"
TMP_DIR="$(mktemp -d -t tgbot-upgrade.XXXXXX)"
BACKUP="${PROGRAM}.backup.$(date +%Y%m%d%H%M%S)"
ROLLED_BACK=0

cleanup() {
    rm -rf "${TMP_DIR}"
}

rollback() {
    if [[ "${ROLLED_BACK}" == "1" || ! -f "${BACKUP}" ]]; then
        return
    fi
    echo "升级失败，正在恢复备份..."
    systemctl stop "${SERVICE}" >/dev/null 2>&1 || true
    install -m 0755 "${BACKUP}" "${PROGRAM}"
    chown tgbot:tgbot "${PROGRAM}" 2>/dev/null || true
    systemctl start "${SERVICE}" >/dev/null 2>&1 || true
    ROLLED_BACK=1
}

on_error() {
    echo "升级失败，请查看：journalctl -u ${SERVICE} -n 100 --no-pager" >&2
    rollback
}

trap cleanup EXIT
trap on_error ERR

if [[ "${EUID}" -ne 0 ]]; then
    echo "请使用 root 或 sudo 执行此脚本。" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
    echo "缺少 curl 或 tar，请先安装：apt-get update && apt-get install -y curl tar" >&2
    exit 1
fi

if [[ ! -x "${PROGRAM}" ]]; then
    echo "找不到旧程序：${PROGRAM}" >&2
    echo "请先按 README 完成首次安装，或设置 TGBOT_INSTALL_DIR。" >&2
    exit 1
fi

if ! systemctl cat "${SERVICE}" >/dev/null 2>&1; then
    echo "找不到 systemd 服务：${SERVICE}" >&2
    exit 1
fi

case "$(uname -m)" in
    x86_64|amd64) asset="tgbot-linux-amd64.tar.gz" ;;
    *) echo "当前只提供 Linux amd64 程序包，检测到架构：$(uname -m)" >&2; exit 1 ;;
esac

if [[ -z "${VERSION}" ]]; then
    VERSION="$(curl -fsSL -H 'Accept: application/vnd.github+json' "https://api.github.com/repos/${REPO}/releases/latest" | sed -n 's/.*"tag_name": "\([^"]*\)".*/\1/p' | head -n 1)"
fi

if [[ -z "${VERSION}" || "${VERSION}" != go-v* ]]; then
    echo "无法确定有效版本号，请设置 TGBOT_VERSION，例如 go-v1.0.3。" >&2
    exit 1
fi

URL="https://github.com/${REPO}/releases/download/${VERSION}/${asset}"
echo "准备升级 ${SERVICE} 到 ${VERSION}..."
echo "下载：${URL}"

curl -fL --retry 3 --connect-timeout 15 -o "${TMP_DIR}/${asset}" "${URL}"
tar -xzf "${TMP_DIR}/${asset}" -C "${TMP_DIR}"

NEW_PROGRAM="${TMP_DIR}/tgbot-linux-amd64/tgbot"
if [[ ! -f "${NEW_PROGRAM}" ]]; then
    echo "压缩包中未找到 tgbot 可执行文件。" >&2
    exit 1
fi

install -m 0755 "${PROGRAM}" "${BACKUP}"
install -m 0755 "${NEW_PROGRAM}" "${PROGRAM}"
chown tgbot:tgbot "${PROGRAM}" 2>/dev/null || true

systemctl daemon-reload
systemctl restart "${SERVICE}"
sleep 2
if ! systemctl is-active --quiet "${SERVICE}"; then
    exit 1
fi

echo "升级成功：${VERSION}"
echo "备份文件：${BACKUP}"
systemctl --no-pager --full status "${SERVICE}" | sed -n '1,12p'
