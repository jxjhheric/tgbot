import sys
import re
import os
import asyncio
import ast
from datetime import datetime
from typing import List, Dict, Set, Callable, Optional, Tuple
import time
from collections import deque, defaultdict
from pathlib import Path
from telegram.error import RetryAfter
from loguru import logger
from dotenv import load_dotenv
from functools import wraps
import aiohttp
from bs4 import BeautifulSoup
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Bot version
BOT_VERSION = "END"

# 番号正则表达式
FANHAO_REGEX = re.compile(
    r'^(?:[A-Za-z]{2,5}[-_ ]?\d{2,5}(?:[-_ ]?[A-Za-z]+)?|FC2-PPV-\d{6,})$', # 稍微调整 FC2 匹配
    re.IGNORECASE
)

# 配置类
class Config:
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """加载或重新加载配置"""
        load_dotenv(override=True)
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
        self.PROXY_URL = os.getenv("PROXY_URL", "").strip() or None
        self.BASE_URL = os.getenv("ALIST_BASE_URL", "")
        self.ALIST_TOKEN = os.getenv("ALIST_TOKEN", "")
        self.ALIST_OFFLINE_DIRS = [d.strip() for d in os.getenv("ALIST_OFFLINE_DIRS", "").split(",") if d.strip()]
        self.SEARCH_URLS = [url.strip() for url in os.getenv("JAV_SEARCH_APIS", "").split(",") if url.strip()]
        self.ALLOWED_USER_IDS = self._parse_allowed_user_ids(os.getenv("ALLOWED_USER_IDS", ""))
        self.CLEAN_INTERVAL_MINUTES = self._parse_int(os.getenv("CLEAN_INTERVAL_MINUTES", "60"), "CLEAN_INTERVAL_MINUTES", 1)
        self.SIZE_THRESHOLD = self._parse_int(os.getenv("SIZE_THRESHOLD", "100"), "SIZE_THRESHOLD", 0) * 1024 * 1024
        self.PREFERRED_KEYWORDS = [k.strip().lower() for k in os.getenv("PREFERRED_KEYWORDS", "").split(",") if k.strip()]
        self.CUSTOM_CATEGORIES = self._parse_custom_categories(os.getenv("CUSTOM_CATEGORIES", ""))
        self.SYSTEM_FOLDERS = self._parse_system_folders(os.getenv("SYSTEM_FOLDERS", ""))
        self.CLEAN_BATCH_SIZE = self._parse_int(os.getenv("CLEAN_BATCH_SIZE", "500"), "CLEAN_BATCH_SIZE", 1)
        self.CLEAN_REQUEST_INTERVAL = self._parse_float(os.getenv("CLEAN_REQUEST_INTERVAL", "0.2"), "CLEAN_REQUEST_INTERVAL", 0.1)
        self.MAX_CONCURRENT_REQUESTS = self._parse_int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"), "MAX_CONCURRENT_REQUESTS", 1)
        self.EXCLUDE_SUFFIXES = [ext.strip().lower() for ext in os.getenv("EXCLUDE_SUFFIXES", "").split(",") if ext.strip()]

    def _parse_int(self, value: str, name: str, min_value: int) -> int:
        try:
            result = int(value)
            if result < min_value:
                logger.error(f"{name} 必须大于或等于 {min_value}，当前值: {value}")
                sys.exit(1)
            return result
        except (ValueError, TypeError):
            logger.error(f"{name} 格式不正确，必须为整数，当前值: {value}")
            sys.exit(1)

    def _parse_float(self, value: str, name: str, min_value: float) -> float:
        try:
            result = float(value)
            if result < min_value:
                logger.error(f"{name} 必须大于或等于 {min_value}，当前值: {value}")
                sys.exit(1)
            return result
        except (ValueError, TypeError):
            logger.error(f"{name} 格式不正确，必须为浮点数，当前值: {value}")
            sys.exit(1)

    def _parse_allowed_user_ids(self, user_ids_str: str) -> Set[int]:
        if not user_ids_str.strip():
            logger.error("ALLOWED_USER_IDS 未提供")
            sys.exit(1)
        try:
            return set(map(int, user_ids_str.split(',')))
        except (ValueError, AttributeError):
            logger.error(f"ALLOWED_USER_IDS 格式不正确: {user_ids_str}")
            sys.exit(1)

    def _parse_custom_categories(self, categories_str: str) -> List[Dict[str, List[str]]]:
        if not categories_str.strip():
            logger.debug("CUSTOM_CATEGORIES 未配置，使用默认分类")
            return []
        categories = []
        try:
            for category in categories_str.split(';'):
                if not category.strip():
                    continue
                parts = category.split(':')
                if len(parts) != 2:
                    logger.warning(f"无效分类配置（缺少名称或关键词）: {category}")
                    continue
                name, keywords = parts
                name = name.strip()
                keywords = [kw.strip().lower() for kw in keywords.split(',') if kw.strip()]
                if not name or not keywords:
                    logger.warning(f"无效分类配置（空名称或关键词）: {category}")
                    continue
                categories.append({"name": name, "keywords": keywords})
            logger.info(f"加载 {len(categories)} 个自定义分类")
            return categories
        except Exception as e:
            logger.error(f"CUSTOM_CATEGORIES 解析错误: {categories_str}, 错误: {str(e)}")
            return []

    def _parse_system_folders(self, folders_str: str) -> List[str]:
        if not folders_str.strip():
            logger.debug("SYSTEM_FOLDERS 未配置，返回空列表")
            return []
        try:
            folders = [f.strip() for f in folders_str.split(',') if f.strip()]
            normalized_folders = []
            for folder in folders:
                folder = folder.strip().replace('\\', '/')
                if not folder.startswith('/'):
                    folder = f"/{folder}"
                folder = folder.rstrip('/')
                if folder:
                    normalized_folders.append(folder)
            logger.info(f"加载 {len(normalized_folders)} 个系统文件夹")
            return normalized_folders
        except Exception as e:
            logger.error(f"SYSTEM_FOLDERS 解析错误: {folders_str}, 错误: {str(e)}")
            return []

    def validate(self):
        required = [
            (self.TELEGRAM_TOKEN, "TELEGRAM_TOKEN"),
            (self.BASE_URL, "ALIST_BASE_URL"),
            (self.ALIST_TOKEN, "ALIST_TOKEN"),
            (self.ALIST_OFFLINE_DIRS, "ALIST_OFFLINE_DIRS"),
            (self.SEARCH_URLS, "JAV_SEARCH_APIS"),
            (self.ALLOWED_USER_IDS, "ALLOWED_USER_IDS")
        ]
        for value, name in required:
            if not value:
                logger.error(f"环境变量 {name} 缺失，请检查 .env 文件")
                return False
        return True

# 初始化配置
config = Config()
if not config.validate():
    sys.exit(1)


class ProxiedClientSession:
    """ClientSession wrapper that applies the configured HTTP proxy."""

    def __init__(self, *args, **kwargs):
        self._session = aiohttp.ClientSession(*args, **kwargs)

    async def __aenter__(self):
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return await self._session.__aexit__(exc_type, exc_value, traceback)

    def get(self, url, *args, **kwargs):
        return self._session.get(url, *args, proxy=config.PROXY_URL, **kwargs)

    def post(self, url, *args, **kwargs):
        return self._session.post(url, *args, proxy=config.PROXY_URL, **kwargs)

# 配置 Loguru 日志
logger.remove()
logger.add("bot.log", rotation="10 MB", level="INFO", filter=lambda record: "Authorization" not in record["message"])
logger.add(sys.stderr, level="INFO", filter=lambda record: "Authorization" not in record["message"])

# 用户授权装饰器
def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in config.ALLOWED_USER_IDS:
            await update.effective_message.reply_text("抱歉，您没有权限使用此机器人。")
            return
        # 如果函数签名包含 token 参数，则传递，否则不传递
        if 'token' in func.__code__.co_varnames:
            return await func(update, context, token=config.ALIST_TOKEN, *args, **kwargs)
        return await func(update, context, *args, **kwargs)
    return wrapped

# 工具函数
class AsyncLimiter:
    """异步速率限制器，用于控制API请求频率"""
    def __init__(self, rate_limit: int, time_window: float = 1.0):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.times = deque()
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        async with self.lock:
            now = time.time()
            while self.times and now - self.times[0] >= self.time_window:
                self.times.popleft()
            if len(self.times) >= self.rate_limit:
                sleep_time = self.time_window - (now - self.times[0])
                await asyncio.sleep(sleep_time)
                self.times.popleft()
            self.times.append(now)

    async def __aexit__(self, exc_type, exc, tb):
        pass

def normalize_path(path: str) -> str:
    """标准化路径格式（兼容Windows/Unix）"""
    return str(Path(path).resolve()).replace('\\', '/')

def is_system_path(path: str, system_folders: List[str]) -> bool:
    """检查路径是否为系统文件夹"""
    path = path.lower().rstrip('/')
    return any(
        path == sf.lower().rstrip('/') or 
        path.startswith(f"{sf.lower().rstrip('/')}/")
        for sf in system_folders
    )

def normalize_fanhao(fanhao: str) -> str:
    """标准化番号，移除分隔符并转换为小写"""
    return re.sub(r'[-_ ]', '', fanhao).lower()

def is_fanhao_match(fanhao: str, name: str) -> bool:
    """检查名称是否包含指定的番号（支持变体）"""
    # 标准化番号：移除所有非字母数字字符并转为小写
    normalized_fanhao = re.sub(r'[^a-z0-9]', '', fanhao.lower())
    # 处理名称：移除所有非字母数字字符并转为小写
    normalized_name = re.sub(r'[^a-z0-9]', '', name.lower())
    
    # 检查标准化后的番号是否存在于处理后的名称中
    result = normalized_fanhao in normalized_name
    
    logger.debug(f"标准化番号: {normalized_fanhao}, 标准化名称: {normalized_name}, 匹配结果: {result}")
    
    if not result:
        logger.debug(f"匹配失败: 番号 {fanhao} 未在名称 {name} 中找到")
    
    return result

def parse_size_to_bytes(size_str: str) -> int:
    """将大小字符串转换为字节"""
    try:
        size_str = size_str.lower().strip()
        logger.debug(f"解析大小字符串: {size_str}")
        if not size_str:
            logger.warning("大小字符串为空")
            return 0
        units = {'kb': 1024, 'mb': 1024**2, 'gb': 1024**3, 'tb': 1024**4}
        # 匹配数字+单位，如 "1.2 GB", "500 MB"
        match = re.match(r'(\d*\.?\d+)\s*([kmgt]b)', size_str)
        if not match:
            logger.warning(f"无法解析大小字符串: {size_str}")
            return 0
        value, unit = float(match.group(1)), match.group(2)
        size_bytes = int(value * units[unit])
        logger.debug(f"解析结果: {size_bytes} 字节")
        return size_bytes
    except (ValueError, KeyError) as e:
        logger.warning(f"大小解析错误: {size_str}, 异常: {str(e)}")
        return 0

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError)
)
async def fetch_api(url: str, session: aiohttp.ClientSession, fanhao: str, headers: Dict) -> List[Dict]:
    """从API搜索磁力链接，确保结果与番号匹配"""
    entries = []
    if not url.strip():
        return entries
    try:
        async with session.get(f"{url.rstrip('/')}/{fanhao}", headers=headers, timeout=20) as response:
            response.raise_for_status()
            raw_result = await response.json()
            if not raw_result or raw_result.get("status") != "succeed":
                logger.debug(f"API {url} 返回空或失败状态")
                return entries
            total_results = len(raw_result.get("data", []))
            for entry_str in raw_result.get("data", []):
                try:
                    data = ast.literal_eval(entry_str)
                    if not isinstance(data, list) or len(data) < 4:
                        continue
                    magnet, name, size_str, date_str = data[:4]
                    if not magnet.startswith("magnet:?"):
                        continue
                    logger.debug(f"API 结果名称: {name}")
                    if not is_fanhao_match(fanhao, name):
                        logger.debug(f"API 结果忽略: {name} 不匹配番号 {fanhao}")
                        continue
                    size_bytes = parse_size_to_bytes(size_str) if size_str else 0
                    if not size_bytes:
                        continue
                    upload_date = None
                    try:
                        if date_str:
                            upload_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    except ValueError:
                        pass
                    entries.append({
                        "magnet": magnet,
                        "name": name,
                        "size_bytes": size_bytes,
                        "date": upload_date,
                        "seeders": 0,
                        "leechers": 0,
                        "source": f"API ({url})"
                    })
                except (ValueError, SyntaxError, TypeError) as e:
                    logger.debug(f"API 结果解析错误: {entry_str}, 错误: {str(e)}")
            logger.debug(f"API {url} 总返回 {total_results} 条，匹配 {len(entries)} 条")
            return entries
    except aiohttp.ClientError as e:
        logger.warning(f"API 搜索失败: {url}: {str(e)}")
        return entries

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError)
)
async def scrape_sukebei(session: aiohttp.ClientSession, fanhao: str, headers: Dict) -> List[Dict]:
    """从sukebei.nyaa.si抓取磁力链接，确保结果与番号匹配"""
    entries = []
    try:
        url = f"https://sukebei.nyaa.si/?f=0&c=0_0&q={fanhao}"
        async with session.get(url, headers=headers, timeout=20) as response:
            response.raise_for_status()
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            torrent_rows = soup.select('tr.default, tr.success')
            total_results = len(torrent_rows)
            matched_count = 0
            for tr in torrent_rows:
                name_elem = tr.select_one('td:nth-child(2) a')
                name = name_elem.text.strip() if name_elem else "未知"
                logger.debug(f"sukebei 结果名称: {name}")
                if not is_fanhao_match(fanhao, name):
                    logger.debug(f"sukebei 结果忽略: {name} 不匹配番号 {fanhao}")
                    continue
                magnet_a = tr.find('a', href=lambda x: x and x.startswith('magnet:?'))
                if not magnet_a:
                    logger.debug(f"sukebei 结果忽略: {name} 无磁力链接")
                    continue
                magnet = magnet_a['href']
                size_td = tr.select_one('td:nth-child(3)')
                size_str = size_td.text.strip() if size_td else ""
                size_bytes = parse_size_to_bytes(size_str) if size_str else 0
                # 放宽过滤条件，允许 size_bytes 为 0
                if size_bytes == 0:
                    logger.warning(f"资源 {name} 大小为 0，可能解析失败")
                date_td = tr.select_one('td:nth-child(5)[data-timestamp]')
                date_str = date_td.text.strip() if date_td else ""
                upload_date = None
                try:
                    if date_str:
                        upload_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M').date()
                except ValueError:
                    pass
                seeders = leechers = 0
                tds = tr.select('td.text-center')
                if len(tds) >= 5:
                    try:
                        seeders = int(tds[2].text.strip())
                        leechers = int(tds[3].text.strip())
                    except ValueError:
                        pass
                entries.append({
                    "magnet": magnet,
                    "name": name,
                    "size_bytes": size_bytes,
                    "date": upload_date,
                    "seeders": seeders,
                    "leechers": leechers,
                    "source": "sukebei"
                })
                matched_count += 1
            logger.debug(f"sukebei.nyaa.si 总返回 {total_results} 条，匹配 {matched_count} 条")
            return entries
    except aiohttp.ClientError as e:
        logger.warning(f"sukebei.nyaa.si 搜索失败: {str(e)}")
        return entries

async def search_magnet(fanhao: str, search_urls: List[str], context: ContextTypes.DEFAULT_TYPE) -> Tuple[Optional[str], Optional[str]]:
    """搜索磁力链接并选择最佳结果"""
    if not FANHAO_REGEX.match(fanhao):
        return None, f"❌ 无效番号格式: {fanhao}"
    
    async with ProxiedClientSession() as session:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        logger.info(f"开始并行搜索番号: {fanhao}")
        api_tasks = [fetch_api(url, session, fanhao, headers) for url in search_urls if url.strip()]
        results = await asyncio.gather(*api_tasks, scrape_sukebei(session, fanhao, headers), return_exceptions=True)
        all_entries = []
        api_results_count = 0
        sukebei_results_count = 0
        failed_sources = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_sources += 1
                logger.warning(f"搜索源 {search_urls[i] if i < len(api_tasks) else 'sukebei.nyaa.si'} 失败: {str(result)}")
                continue
            if i < len(api_tasks):
                api_results_count += len(result)
                logger.debug(f"API {search_urls[i]} 返回 {len(result)} 条匹配结果")
            else:
                sukebei_results_count = len(result)
                logger.debug(f"sukebei.nyaa.si 返回 {len(result)} 条匹配结果")
            logger.debug(f"添加结果: {len(result)} 条，来源: {'API' if i < len(api_tasks) else 'sukebei'}")
            all_entries.extend(result)
        
        logger.info(f"搜索结果统计: API 总计 {api_results_count} 条, sukebei.nyaa.si {sukebei_results_count} 条")
        
        if not all_entries:
            error_msg = f"🔍 未找到番号 {fanhao} 的相关资源（API: {api_results_count} 条匹配, sukebei: {sukebei_results_count} 条匹配）"
            if failed_sources == len(api_tasks) + 1:
                error_msg += f"\n⚠️ 所有 {failed_sources} 个搜索源均失败"
            return None, error_msg
        
        seen_magnets = set()
        unique_entries = []
        for entry in all_entries:
            magnet = entry["magnet"]
            if magnet not in seen_magnets:
                seen_magnets.add(magnet)
                unique_entries.append(entry)
        logger.info(f"去重后剩余 {len(unique_entries)} 条结果")
        
        if not unique_entries:
            return None, f"🔍 未找到有效磁力链接（API: {api_results_count} 条, sukebei: {sukebei_results_count} 条）"
        
        # 不过度过滤 size_bytes 为 0 的条目
        valid_entries = unique_entries
        logger.debug(f"有效条目: {len(valid_entries)} 条")
        if not valid_entries:
            return None, f"❌ 未找到与番号 {fanhao} 匹配的有效磁力链接（API: {api_results_count} 条, sukebei: {sukebei_results_count} 条）"
        
        if len(valid_entries) == 1:
            selected_entry = valid_entries[0]
            logger.info(f"仅一条有效结果，选择来源: {selected_entry['source']}, 名称: {selected_entry['name'][:50]}...")
            return selected_entry["magnet"], None
        
        def has_preferred_keyword(entry):
            if not config.PREFERRED_KEYWORDS:
                return 0
            name_lower = entry["name"].lower()
            return any(keyword in name_lower for keyword in config.PREFERRED_KEYWORDS)
        
        # 优先选择有种子的条目
        selected = sorted(
            valid_entries,
            key=lambda x: (
                has_preferred_keyword(x),
                x.get("seeders", 0),
                x["date"].toordinal() if x["date"] else float('-inf'),
                x["size_bytes"]
            ),
            reverse=True
        )
        selected_entry = selected[0]
        logger.info(f"选择最佳磁力链接，来源: {selected_entry['source']}, 名称: {selected_entry['name'][:50]}...")
        return selected_entry["magnet"], None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, RetryAfter))
)
async def add_offline_download(context: ContextTypes.DEFAULT_TYPE, token: str, links: List[str]) -> Tuple[bool, str]:
    """添加离线下载任务到Alist"""
    if not token or not links:
        return False, "❌ 缺少必要参数"
    try:
        url = f"{config.BASE_URL.rstrip('/')}/api/fs/add_offline_download"
        headers = {"Authorization": token, "Content-Type": "application/json"}
        post_data = {
            "path": context.bot_data.get('current_download_dir', config.ALIST_OFFLINE_DIRS[0]),
            "urls": links,
            "tool": "storage",
            "delete_policy": "delete_on_upload_succeed"
        }
        async with ProxiedClientSession() as session:
            async with session.post(url, json=post_data, headers=headers, timeout=30) as response:
                status = response.status
                result = await response.json()
                if status == 401:
                    return False, "❌ 认证失败"
                if status == 500:
                    return False, "❌ 服务器拒绝请求"
                if status == 200 and result.get("code") == 200:
                    return True, "✅ 已添加到下载队列"
                return False, f"❌ 解析失败"
    except aiohttp.ClientError as e:
        logger.error(f"添加任务失败: {str(e)}")
        raise

async def recursive_collect_files(token: str, base_url: str, root_dir: str) -> Tuple[Dict[str, List[str]], Set[str]]:
    """递归收集小文件并分组"""
    dir_files = defaultdict(list)
    known_empty_dirs = set()

    async def collect_files_worker(dir_path: str, session: aiohttp.ClientSession):
        list_url = f"{base_url.rstrip('/')}/api/fs/list"
        headers = {"Authorization": token}
        payload = {"path": dir_path, "page": 1, "per_page": 1000, "refresh": True}
        try:
            async with session.post(list_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    logger.error(f"列出目录 {dir_path} 失败: 状态码 {resp.status}")
                    return
                data = await resp.json()
                if data.get("code") != 200:
                    logger.error(f"列出目录 {dir_path} 失败: {data.get('message', '未知错误')}")
                    return
                content = data.get("data", {}).get("content")
                if content is None:
                    known_empty_dirs.add(dir_path)
                    return

                tasks = []
                for item in content:
                    item_path = os.path.join(dir_path, item["name"]).replace("\\", "/")
                    if item["is_dir"]:
                        tasks.append(collect_files_worker(item_path, session))
                    elif (
                        item["size"] < config.SIZE_THRESHOLD and
                        not any(item["name"].lower().endswith(ext) for ext in config.EXCLUDE_SUFFIXES)
                    ):
                        dir_files[dir_path].append(item["name"])
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        except aiohttp.ClientError as e:
            logger.error(f"列出目录 {dir_path} 失败: {str(e)}")

    async with ProxiedClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        await collect_files_worker(root_dir, session)
    return dict(dir_files), known_empty_dirs

async def recursive_collect_empty_dirs(token: str, base_url: str, root_dir: str) -> List[str]:
    """递归收集空文件夹"""
    empty_dirs = set()

    async def check_dir(dir_path: str, session: aiohttp.ClientSession):
        if is_system_path(dir_path, config.SYSTEM_FOLDERS):
            return
        list_url = f"{base_url.rstrip('/')}/api/fs/list"
        headers = {"Authorization": token}
        payload = {"path": dir_path, "page": 1, "per_page": 1000, "refresh": True}
        try:
            async with session.post(list_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                if data.get("code") != 200:
                    return
                content = data.get("data", {}).get("content")
                if content is None or len(content) == 0:
                    empty_dirs.add(dir_path)
                    return
                tasks = []
                for item in content:
                    if item["is_dir"]:
                        item_path = os.path.join(dir_path, item["name"]).replace("\\", "/")
                        tasks.append(check_dir(item_path, session))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        except aiohttp.ClientError as e:
            logger.error(f"列出目录 {dir_path} 失败: {str(e)}")

    async with ProxiedClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        await check_dir(root_dir, session)
    return list(empty_dirs)

async def cleanup_small_files(token: str, base_url: str, root_dir: str, progress_callback: Callable[[int], None] = None) -> Tuple[int, str]:
    """清理小文件，支持并行删除"""
    start_time = datetime.now()
    list_url = f"{base_url.rstrip('/')}/api/fs/list"
    headers = {"Authorization": token}
    payload = {"path": root_dir, "page": 1, "per_page": 0}
    try:
        async with ProxiedClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(list_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    return 0, f"❌ 无法访问路径 {root_dir}: 状态码 {resp.status}"
                data = await resp.json()
                if data.get("code") != 200:
                    return 0, f"❌ 路径无效: {data.get('message', '未知错误')}"
    except aiohttp.ClientError as e:
        return 0, f"❌ 路径访问失败: {str(e)}"

    dir_files, _ = await recursive_collect_files(token, base_url, root_dir)
    total_files = sum(len(files) for files in dir_files.values())
    logger.info(f"开始清理 {root_dir}，发现 {total_files} 个小文件")

    merged_batches = []
    current_batch = []
    current_batch_size = 0
    for parent, names in dir_files.items():
        if is_system_path(parent, config.SYSTEM_FOLDERS):
            continue
        for name in names:
            if current_batch_size < config.CLEAN_BATCH_SIZE:
                current_batch.append((parent, name))
                current_batch_size += 1
            else:
                merged_batches.append(current_batch)
                current_batch = [(parent, name)]
                current_batch_size = 1
    if current_batch:
        merged_batches.append(current_batch)

    deleted_count = 0
    errors = []
    async with ProxiedClientSession(connector=aiohttp.TCPConnector(limit=config.MAX_CONCURRENT_REQUESTS)) as session:
        for batch_idx, batch in enumerate(merged_batches, 1):
            batch_by_parent = defaultdict(list)
            for parent, name in batch:
                batch_by_parent[parent].append(name)

            async def delete_batch(parent: str, names: List[str], retry_count: int = 0) -> Tuple[int, str]:
                max_retries = 5
                base_delay = 1
                try:
                    remove_url = f"{base_url.rstrip('/')}/api/fs/remove"
                    headers = {"Authorization": token}
                    data = {"dir": parent, "names": names}
                    async with session.post(remove_url, json=data, headers=headers) as resp:
                        if resp.status == 429:
                            if retry_count < max_retries:
                                delay = base_delay * (2 ** retry_count)
                                logger.warning(f"API 速率限制，等待 {delay} 秒")
                                await asyncio.sleep(delay)
                                return await delete_batch(parent, names, retry_count + 1)
                            return 0, f"{parent}: 速率限制"
                        result = await resp.json()
                        if result.get("code") == 200:
                            if progress_callback:
                                progress_callback(len(names))
                            return len(names), ""
                        return 0, f"{parent}: {result.get('message', '未知错误')}"
                except aiohttp.ClientError as e:
                    return 0, f"{parent}: 网络错误 - {str(e)}"

            tasks = [delete_batch(parent, names) for parent, names in batch_by_parent.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (count, error), (parent, names) in zip(results, batch_by_parent.items()):
                if isinstance(count, Exception):
                    errors.append(f"{parent}: 任务失败 - {str(count)}")
                    continue
                deleted_count += count
                if error:
                    errors.append(error)

            await asyncio.sleep(config.CLEAN_REQUEST_INTERVAL)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"清理 {root_dir} 完成，{deleted_count}/{total_files} 文件，耗时 {elapsed:.2f} 秒")
    if errors:
        return deleted_count, (
            f"⚠️ 部分清理失败（成功 {deleted_count} 文件）\n" +
            "\n".join([f"• {msg}" for msg in errors[:3]])
        )
    return deleted_count, f"✅ 成功清理 {deleted_count} 个小文件"

async def cleanup_empty_dirs(token: str, base_url: str, root_dir: str, 
                          progress_callback: Callable[[int], None] = None) -> Tuple[int, str]:
    """清理空文件夹"""
    start_time = datetime.now()
    list_url = f"{base_url.rstrip('/')}/api/fs/list"
    headers = {"Authorization": token}
    payload = {"path": root_dir, "page": 1, "per_page": 0}
    try:
        async with ProxiedClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(list_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    return 0, f"❌ 无法访问路径 {root_dir}: 状态码 {resp.status}"
                data = await resp.json()
                if data.get("code") != 200:
                    return 0, f"❌ 路径无效: {data.get('message', '未知错误')}"
    except aiohttp.ClientError as e:
        return 0, f"❌ 路径访问失败: {str(e)}"

    empty_dirs = await recursive_collect_empty_dirs(token, base_url, root_dir)
    if not empty_dirs:
        logger.info(f"未找到空文件夹: {root_dir}")
        return 0, "✅ 未找到空文件夹"

    total_deleted = 0
    error_messages = []
    merged_groups = defaultdict(list)
    for dir_path in empty_dirs:
        if is_system_path(dir_path, config.SYSTEM_FOLDERS):
            continue
        parent = os.path.dirname(dir_path)
        name = os.path.basename(dir_path)
        merged_groups[parent].append(name)

    async with ProxiedClientSession(connector=aiohttp.TCPConnector(limit=config.MAX_CONCURRENT_REQUESTS)) as session:
        async def delete_empty_batch(parent: str, names: List[str], retry_count: int = 0) -> Tuple[int, str]:
            max_retries = 5
            base_delay = 1
            try:
                remove_url = f"{base_url.rstrip('/')}/api/fs/remove"
                headers = {"Authorization": token}
                data = {"dir": parent, "names": names}
                async with session.post(remove_url, json=data, headers=headers) as resp:
                    if resp.status == 429:
                        if retry_count < max_retries:
                            delay = base_delay * (2 ** retry_count)
                            logger.warning(f"API 速率限制，等待 {delay} 秒")
                            await asyncio.sleep(delay)
                            return await delete_empty_batch(parent, names, retry_count + 1)
                        return 0, f"{parent}: 速率限制"
                    result = await resp.json()
                    if result.get("code") == 200:
                        count = len(names)
                        if progress_callback:
                            progress_callback(count)
                        return count, ""
                    return 0, f"{parent}: {result.get('message', '未知错误')}"
            except aiohttp.ClientError as e:
                return 0, f"{parent}: 网络错误 - {str(e)}"

        tasks = [delete_empty_batch(parent, names) for parent, names in merged_groups.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (count, error), (parent, names) in zip(results, merged_groups.items()):
            if isinstance(count, Exception):
                error_messages.append(f"{parent}: 任务失败 - {str(count)}")
                continue
            total_deleted += count
            if error:
                error_messages.append(error)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"清理空文件夹 {root_dir} 完成，{total_deleted} 个，耗时 {elapsed:.2f} 秒")
    if error_messages:
        return total_deleted, (
            f"⚠️ 部分删除失败（成功 {total_deleted} 文件夹）\n" +
            "\n".join([f"• {msg}" for msg in error_messages[:3]])
        )
    return total_deleted, f"✅ 成功删除 {total_deleted} 个空文件夹"

async def auto_clean(context: ContextTypes.DEFAULT_TYPE) -> None:
    """自动清理任务"""
    token = config.ALIST_TOKEN
    base_url = config.BASE_URL
    results = []
    start_msg = (f"🔄 自动清理任务启动\n"
                 f"• 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 仅向启用清理通知的用户发送消息
    for user_id in config.ALLOWED_USER_IDS:
        if context.bot_data.get(f"notify_clean_{user_id}", False):
            logger.info(f"Sending auto-clean start message to user {user_id}")
            try:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=start_msg,
                    parse_mode=ParseMode.MARKDOWN
                )
                await asyncio.sleep(0.3)
            except aiohttp.ClientError as e:
                logger.error(f"发送消息到 {user_id} 失败: {str(e)}")

    # 过滤掉子目录，只清理顶层目录
    unique_dirs = []
    for dir_path in sorted(config.ALIST_OFFLINE_DIRS, key=len):
        if not any(dir_path.startswith(parent + '/') for parent in unique_dirs):
            unique_dirs.append(dir_path)

    try:
        for target_dir in unique_dirs:
            dir_msg = []
            try:
                # 验证目录存在
                async with ProxiedClientSession() as session:
                    list_url = f"{base_url.rstrip('/')}/api/fs/list"
                    headers = {"Authorization": token}
                    payload = {"path": target_dir, "page": 1, "per_page": 0}
                    async with session.post(list_url, json=payload, headers=headers) as resp:
                        if resp.status != 200:
                            results.append(f"❌ {target_dir} 不可访问: 状态码 {resp.status}")
                            continue
                        data = await resp.json()
                        if data.get("code") != 200:
                            results.append(f"❌ {target_dir} 无效: {data.get('message', '未知错误')}")
                            continue

                deleted_files, msg_files = await cleanup_small_files(token, base_url, target_dir)
                dir_msg.append(f"小文件: {msg_files}")
                deleted_dirs, msg_dirs = await cleanup_empty_dirs(token, base_url, target_dir)
                dir_msg.append(f"空目录: {msg_dirs}")
                results.append(f"📂 目录 {target_dir}:\n" + "\n".join(dir_msg))
            except aiohttp.ClientError as e:
                error_detail = f"HTTP 错误: {str(e)}"
                try:
                    async with ProxiedClientSession() as session:
                        async with session.post(list_url, json=payload, headers=headers) as resp:
                            error_detail += f", 状态码: {resp.status}, 响应: {await resp.text()[:100]}"
                except Exception as inner_e:
                    error_detail += f", 无法获取响应: {str(inner_e)}"
                results.append(f"❌ {target_dir} 清理失败: {error_detail}")
            await asyncio.sleep(1)  # 避免 API 速率限制

        summary = [
            f"✅ 自动清理完成",
            f"• 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ] + results
        max_length = 4000
        current_msg = []
        for line in summary:
            if len('\n'.join(current_msg + [line])) > max_length:
                message = '\n'.join(current_msg)
                for user_id in config.ALLOWED_USER_IDS:
                    if context.bot_data.get(f"notify_clean_{user_id}", False):
                        logger.info(f"Sending auto-clean summary message to user {user_id}")
                        try:
                            await context.bot.send_message(
                                chat_id=user_id,
                                text=message,
                                parse_mode=ParseMode.MARKDOWN
                            )
                            await asyncio.sleep(0.3)
                        except aiohttp.ClientError as e:
                            logger.error(f"发送消息到 {user_id} 失败: {str(e)}")
                current_msg = [line]
            else:
                current_msg.append(line)
        if current_msg:
            message = '\n'.join(current_msg)
            for user_id in config.ALLOWED_USER_IDS:
                if context.bot_data.get(f"notify_clean_{user_id}", False):
                    logger.info(f"Sending final auto-clean summary message to user {user_id}")
                    try:
                        await context.bot.send_message(
                            chat_id=user_id,
                            text=message,
                            parse_mode=ParseMode.MARKDOWN
                        )
                        await asyncio.sleep(0.3)
                    except aiohttp.ClientError as e:
                        logger.error(f"发送消息到 {user_id} 失败: {str(e)}")
    except Exception as e:
        error_msg = f"❌ 自动清理任务异常: {str(e)}"
        for user_id in config.ALLOWED_USER_IDS:
            if context.bot_data.get(f"notify_clean_{user_id}", False):
                logger.info(f"Sending auto-clean error message to user {user_id}")
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=error_msg,
                        parse_mode=ParseMode.MARKDOWN
                    )
                    await asyncio.sleep(0.3)
                except aiohttp.ClientError as e:
                    logger.error(f"发送消息到 {user_id} 失败: {str(e)}")

async def find_download_directory(token: str, base_url: str, parent_dir: str, original_code: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """查找包含番号的目录"""
    logger.info(f"搜索番号 '{original_code}'")
    list_url = f"{base_url.rstrip('/')}/api/fs/list"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    try:
        parent_dir = parent_dir.strip().replace('\\', '/').rstrip('/')
        if not parent_dir.startswith('/'):
            parent_dir = f'/{parent_dir}'
        list_payload = {"path": parent_dir, "page": 1, "per_page": 0}
        async with ProxiedClientSession() as session:
            async with session.post(list_url, json=list_payload, headers=headers, timeout=20) as response:
                response.raise_for_status()
                list_result = await response.json()
                if list_result.get("code") != 200:
                    return None, f"目录列表失败: {list_result.get('message', '未知错误')}"
                content = list_result.get("data", {}).get("content", [])
                target_pattern = re.sub(r'[^a-zA-Z0-9]', '', original_code).lower()
                possible_matches = []
                for item in content:
                    if item.get("is_dir"):
                        dir_name = item.get("name", "").strip()
                        normalized_dir = re.sub(r'[^a-zA-Z0-9]', '', dir_name).lower()
                        if normalized_dir.startswith(target_pattern):
                            full_path = f"{parent_dir.rstrip('/')}/{dir_name}".replace('//', '/')
                            possible_matches.append(full_path)
                return possible_matches, None
    except aiohttp.ClientError as e:
        logger.error(f"目录搜索失败: {str(e)}")
        return None, "目录搜索失败"

def extract_jav_prefix(name: str) -> Optional[str]:
    """提取 JAV 前缀"""
    logger.debug(f"原始名称: {name}")
    
    # 清理名称，保留关键番号部分
    clean_name = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U00002600-\U000026FF]+',  # Emoji
        '', name, flags=re.UNICODE)
    clean_name = re.sub(
        r'(?:[Cc]hinese|[Ee]nglish)\s*[Ss]ubtitles?|中文字幕|英文字幕|高清|无码|無碼|无水印|無水印|破解版|破解|[\(\[]\d{4}[\)\]]|'  # 年份括号
        r'FHD|HD|SD|1080p|720p|4K|Blu[Rr]ay|WEB-DL|REPACK|UNCENSORED|LEAKED|'  # 质量标签
        r'\[.*?\]|【.*?】|\(.*?\)',  # 各种括号内容
        '',
        clean_name,
        flags=re.IGNORECASE
    ).strip()
    clean_name = re.sub(r'\s+', ' ', clean_name)  # 合并多余空格
    logger.debug(f"清理后名称: {clean_name}")

    # 优先匹配更明确的模式
    patterns = [
        # FC2-PPV-数字(6+): FC2-PPV-1234567, FC2PPV123456
        r'(FC2)[-_]?(PPV)[-_]?(\d{6,})',
        # FC2-数字(6+): FC2-1234567, FC2123456 (非PPV)
        r'(FC2)[-_]?(\d{6,})',
        # 字母(2-5)-数字(2-7)，可选字母后缀: ABC-123, DEFG-1001, HIJKL-500A
        r'([A-Z]{2,5})[-_ ]?(\d{2,7})(?:[-_ ]?([A-Z]+))?',
        # 长字母串+数字(3+): HEYZO123, CARIBBEAN001
        r'([A-Z]{4,})(\d{3,})'
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_name, re.IGNORECASE)
        if match:
            groups = match.groups()
            prefix = groups[0].upper()
            logger.debug(f"匹配前缀: {prefix}, 完整匹配: {match.group(0)}")
            # 特殊处理 FC2
            if prefix == 'FC2':
                # 如果匹配到 FC2-PPV 模式，或名称包含 PPV
                if len(groups) > 1 and groups[1] and groups[1].upper() == 'PPV':
                    return 'FC2-PPV'
                elif 'PPV' in name.upper():
                    return 'FC2-PPV'
                else:
                    return 'FC2'
            # 对于其他前缀，直接返回
            return prefix

    logger.trace(f"无法从名称 '{name}' (清理后: '{clean_name}') 提取 JAV 前缀")
    return None

def get_destination_subdir(name: str, custom_categories: List[Dict[str, List[str]]]) -> str:
    """确定文件的分类子目录"""
    logger.debug(f"确定子目录，名称: {name}")
    prefix = extract_jav_prefix(name)
    logger.debug(f"提取的前缀: {prefix}")
    if prefix:
        if prefix.startswith('FC2') or prefix == 'FC2-PPV':
            logger.debug("分类到: JAV/FC2")
            return 'JAV/FC2'
        first_letter = prefix[0].upper()
        logger.debug(f"分类到: JAV/{first_letter}")
        return f"JAV/{first_letter}"
    name_lower = name.lower()
    for category in custom_categories:
        if any(kw in name_lower for kw in category["keywords"]):
            logger.debug(f"分类到: {category['name']}")
            return category["name"]
    logger.debug("分类到: 其他")
    return "其他"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError)
)
async def list_directory(token: str, base_url: str, path: str) -> List[Dict]:
    """列出目录内容，保证始终返回列表"""
    url = f"{base_url.rstrip('/')}/api/fs/list"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    payload = {"path": path, "page": 1, "per_page": 0}
    try:
        async with ProxiedClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=15) as response:
                # 处理非200状态码
                if response.status != 200:
                    logger.error(f"目录列表失败 HTTP状态码: {response.status}")
                    return []
                
                result = await response.json()
                if result.get("code") == 200:
                    return result.get("data", {}).get("content", [])
                
                logger.error(f"目录列表失败: {result.get('message', '未知错误')}")
                return []
    
    except Exception as e:
        logger.error(f"获取目录内容失败: {str(e)}")
        return []  # 确保返回空列表而不是 None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError)
)
async def create_directory_recursive(token: str, path: str) -> bool:
    """递归创建目录，完全处理路径异常"""
    try:
        # 规范化路径处理
        path = path.strip().replace('\\', '/').rstrip('/')
        
        # 处理根目录和空路径
        if not path:
            logger.error("创建目录失败: 空路径")
            return False
        if path == "/":
            logger.debug("根目录已存在")
            return True
        
        # 分割路径为层级结构
        parts = [p for p in path.lstrip('/').split('/') if p.strip()]
        if not parts:
            logger.error("无效路径格式")
            return False
        
        current_path = ""
        for part in parts:
            try:
                # 构建当前路径
                current_path = f"{current_path}/{part}".lstrip('/')
                full_path = f"/{current_path}"
                
                # 跳过空路径段
                if not part.strip():
                    continue
                
                # 检查目录是否存在
                if not await directory_exists(token, full_path):
                    # 创建目录
                    mkdir_url = f"{config.BASE_URL.rstrip('/')}/api/fs/mkdir"
                    headers = {"Authorization": token}
                    data = {"path": full_path}
                    
                    async with ProxiedClientSession() as session:
                        async with session.post(mkdir_url, json=data, headers=headers, timeout=15) as response:
                            if response.status != 200:
                                logger.error(f"创建目录失败 HTTP {response.status}: {full_path}")
                                return False
                            
                            result = await response.json()
                            if result.get("code") != 200:
                                logger.error(f"创建目录失败: {result.get('message')} ({full_path})")
                                return False
                            
                            logger.debug(f"已创建目录: {full_path}")
            
            except Exception as e:
                logger.error(f"创建目录 {full_path} 时出错: {str(e)}")
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"递归创建目录异常: {str(e)}")
        return False

async def directory_exists(token: str, path: str) -> bool:
    """检查目录是否存在"""
    parent, name = os.path.dirname(path), os.path.basename(path)
    
    # 获取目录内容时处理可能的 None 返回值
    content = await list_directory(token, config.BASE_URL, parent)
    
    # 处理 content 为 None 的情况
    if content is None:
        logger.warning(f"目录 {parent} 列表返回空结果")
        return False
    
    # 遍历内容检查目录是否存在
    for item in content:
        if item.get("name") == name and item.get("is_dir"):
            return True
    return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, RetryAfter))
)
async def ensure_directory_exists(token: str, path: str) -> Tuple[bool, str]:
    """确保目录存在，递归创建所有父目录"""
    try:
        # 调用递归创建函数
        success = await create_directory_recursive(token, path)
        if success:
            return True, ""
        else:
            return False, "无法创建目录结构，请检查路径权限或网络连接"
    except Exception as e:
        logger.error(f"创建目录 {path} 时发生意外错误: {str(e)}")
        return False, f"目录创建失败: {str(e)}"

async def move_items(token: str, src_dir: str, names: List[str], dst_dir: str) -> Tuple[bool, str]:
    dir_exists, error = await ensure_directory_exists(token, dst_dir)
    if not dir_exists:
        return False, f"目录准备失败: {dst_dir} - {error}"
    
    url = f"{config.BASE_URL.rstrip('/')}/api/fs/move"
    headers = {"Authorization": token}
    data = {
        "src_dir": normalize_path(src_dir),
        "dst_dir": normalize_path(dst_dir),
        "names": names
    }
    
    try:
        async with ProxiedClientSession() as session:
            async with session.post(url, json=data, headers=headers, timeout=30) as resp:
                result = await resp.json()
                if resp.status == 200 and result.get("code") == 200:
                    return True, ""
                return False, result.get("message", "移动失败")
    except Exception as e:
        return False, f"网络错误: {str(e)}"

async def count_items_to_move(token: str, target_dir: str, classify_type: str, top_dirs: List[str]) -> int:
    """统计需要移动的项目总数"""
    content = await list_directory(token, config.BASE_URL, target_dir) or []
    count = 0
    for item in content:
        if should_process_item(item, classify_type, top_dirs, target_dir):
            count += 1
    logger.debug(f"目录 {target_dir} 需移动项目数: {count}")
    return count

async def process_classification(token: str, target_dir: str, classify_type: str, moved_items: List[int], lock: asyncio.Lock, error_messages: List[str], update_progress: Callable) -> None:
    """优化版分类逻辑，实时更新移动进度"""
    limiter = AsyncLimiter(2, 1)

    top_dirs = ["JAV"] + [cat["name"] for cat in config.CUSTOM_CATEGORIES] + ["其他"]
    for dir_name in top_dirs:
        dest_path = f"{target_dir}/{dir_name}"
        success, error = await ensure_directory_exists(token, dest_path)
        if not success:
            error_messages.append(f"目录准备失败 {dir_name}: {error}")
        # 额外确保 JAV/FC2 目录
        if dir_name == "JAV":
            fc2_path = f"{target_dir}/JAV/FC2"
            success, error = await ensure_directory_exists(token, fc2_path)
            if not success:
                error_messages.append(f"目录准备失败 JAV/FC2: {error}")

    content = await list_directory(token, config.BASE_URL, target_dir) or []
    move_map = defaultdict(list)
    
    for item in content:
        if not should_process_item(item, classify_type, top_dirs, target_dir):
            continue
        dest = get_destination_subdir(item["name"], config.CUSTOM_CATEGORIES)
        move_map[f"{target_dir}/{dest}"].append(item["name"])

    logger.debug(f"移动映射: {dict(move_map)}")
    
    for dest_path, names in move_map.items():
        for i in range(0, len(names), 5):
            batch = names[i:i+5]
            async with limiter:
                success, error = await move_items(token, target_dir, batch, dest_path)
                if success:
                    async with lock:
                        moved_items[0] += len(batch)
                    await update_progress()
                    logger.debug(f"移动 {len(batch)} 个项目到 {dest_path}")
                else:
                    error_messages.append(f"移动失败到 {os.path.basename(dest_path)}: {error}")
                    if "object not found" in error.lower():
                        await ensure_directory_exists(token, dest_path)
                        success, error = await move_items(token, target_dir, batch, dest_path)
                        if success:
                            async with lock:
                                moved_items[0] += len(batch)
                            await update_progress()
                            error_messages.pop()
                            logger.debug(f"修复后成功移动 {len(batch)} 个项目到 {dest_path}")

def should_process_item(item: dict, classify_type: str, top_dirs: List[str], target_dir: str) -> bool:
    name = item.get("name", "")
    is_dir = item.get("is_dir", False)
    
    if classify_type == "folder" and not is_dir:
        return False
    if classify_type == "file" and is_dir:
        return False
    if is_system_path(f"{target_dir}/{name}", config.SYSTEM_FOLDERS):
        return False
    if name in top_dirs:
        return False
    return True

async def resolve_target_dirs(token: str, base_dir: str, target: Optional[str]) -> List[str]:
    if not target or target == "/":
        return [base_dir]
    
    if FANHAO_REGEX.match(target):
        dirs, _ = await find_download_directory(token, config.BASE_URL, base_dir, target)
        return dirs or []
    
    target_path = normalize_path(os.path.join(base_dir, target))
    if await directory_exists(token, target_path):
        return [target_path]
    
    return []

@restricted
async def classify_command(update: Update, context: ContextTypes.DEFAULT_TYPE, token: str) -> None:
    """支持批量操作的分类命令，优化进度条显示"""
    args = context.args or []
    classify_type = "all"
    target = None
    
    if args:
        first_arg = args[0].lower()
        if first_arg in ("all", "folder", "file"):
            classify_type = first_arg
            target = args[1] if len(args) > 1 else None
        else:
            target = first_arg
            if len(args) > 1 and args[1].lower() in ("all", "folder", "file"):
                classify_type = args[1].lower()

    current_dir = context.bot_data.get('current_download_dir', config.ALIST_OFFLINE_DIRS[0])
    target_dirs = await resolve_target_dirs(token, current_dir, target)
    if not target_dirs:
        await update.effective_message.reply_text("❌ 未找到有效目标目录")
        return

    top_dirs = ["JAV"] + [cat["name"] for cat in config.CUSTOM_CATEGORIES] + ["其他"]
    total_items = 0
    for dir_path in target_dirs:
        total_items += await count_items_to_move(token, dir_path, classify_type, top_dirs)
    logger.info(f"共需移动 {total_items} 个项目，目标目录数: {len(target_dirs)}")

    progress_msg = await update.effective_message.reply_text(
        f"🔄 开始批量整理【{classify_type}】类型，共 {total_items} 个项目...\n"
        f"▰▱▱▱▱▱▱▱▱ 0%"
    )

    semaphore = asyncio.Semaphore(15)
    moved_items = [0]
    error_messages = []
    lock = asyncio.Lock()

    async def update_progress():
        progress = int((moved_items[0] / total_items) * 100) if total_items > 0 else 100
        progress_bar = "▰" * (progress // 5) + "▱" * (20 - progress // 5)
        await safe_edit_message(
            progress_msg,
            f"🔄 批量整理中...\n"
            f"{progress_bar} {progress}%\n"
            f"• 已移动: {moved_items[0]}/{total_items} 项目\n"
            f"• 错误: {len(error_messages)}"
        )

    async def process_with_semaphore(dir_path):
        async with semaphore:
            await process_classification(token, dir_path, classify_type, moved_items, lock, error_messages, update_progress)

    tasks = [process_with_semaphore(dir_path) for dir_path in target_dirs]
    await asyncio.gather(*tasks, return_exceptions=True)

    result_msg = [
        f"✅ 整理完成！",
        f"• 处理目录: {len(target_dirs)}",
        f"• 移动项目: {moved_items[0]}",
        f"• 失败项目: {len(error_messages)}"
    ]
    
    if error_messages:
        error_samples = "\n".join([f"• {e}" for e in error_messages[:3]])
        result_msg.append(f"\n错误示例:\n{error_samples}")
        if len(error_messages) > 3:
            result_msg.append("...（更多错误详见日志）")

    user_id = update.effective_user.id
    if context.bot_data.get(f"notify_task_{user_id}", False):
        await update.effective_message.reply_text(
            f"✅ 分类任务完成: {'/'.join(target_dirs)}",
            parse_mode=ParseMode.MARKDOWN
        )
    await safe_edit_message(progress_msg, "\n".join(result_msg))
    await asyncio.sleep(3)
    await refresh_command(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """显示帮助信息"""
    user_id = update.effective_user.id
    if user_id not in config.ALLOWED_USER_IDS:
        await update.effective_message.reply_text("抱歉，您没有权限使用此机器人。")
        return
    help_text = (
        f'### JAV 下载机器人 {BOT_VERSION}\n'
        '1. 发送番号（如 `ABC-123` 或 `FC2-PPV-1010519`）\n'
        '2. 发送磁力链接（`magnet:?`）或 ed2k 链接（`ed2k://`）\n'
        '3. **清理功能**：\n'
        '   - `/clean` 清理当前目录\n'
        '   - `/clean <番号>` 清理特定目录\n'
        '4. **目录管理**：\n'
        '   - `/setdir` 列出并选择下载目录\n'
        '   - `/reload_config` 重新加载配置\n'
        '5. **分类整理**:\n'
        '   - `/classify [all|folder|file]` 智能整理\n'
        '     示例:\n'
        '     /classify       - 整理所有类型\n'
        '     /classify folder - 仅整理文件夹\n'
        '     /classify file   - 仅整理文件\n'
        '6. **刷新alist**:\n'
        '   - `/refresh`\n'
        '7. **通知设置**:\n'
        '   - `/notify 清理 <on|off>` 设置通知\n'
        '     示例:\n'
        '     /notify 清理 off - 关闭自动清理通知\n'
        f'**当前目录**：`{context.bot_data.get("current_download_dir", "未知")}`'
    )
    await update.effective_message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """启动命令"""
    user_id = update.effective_user.id
    if user_id not in config.ALLOWED_USER_IDS:
        await update.effective_message.reply_text("抱歉，您没有权限使用此机器人。")
        return
    keyboard = [[InlineKeyboardButton("查看帮助", callback_data='help')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_message.reply_text(
        '欢迎使用 JAV 下载机器人！\n'
        '发送番号或磁力/ed2k 链接以添加到 Alist。\n'
        '点击下方按钮查看帮助。',
        reply_markup=reply_markup
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理按钮回调"""
    query = update.callback_query
    await query.answer()
    if query.data == 'help':
        await help_command(update, context)

async def handle_single_entry(update: Update, context: ContextTypes.DEFAULT_TYPE, token: str, entry: str):
    """处理单条输入"""
    chat_id = update.effective_chat.id
    processing_msg = None
    try:
        if entry.startswith(("magnet:?", "ed2k://")):
            processing_msg = await update.effective_message.reply_text(f"🔗 收到{'磁力' if entry.startswith('magnet:?') else 'ed2k'}链接")
            success, result_msg = await add_offline_download(context, token, [entry])
            await processing_msg.edit_text(result_msg)
            if success:
                user_id = update.effective_user.id
                if context.bot_data.get(f"notify_task_{user_id}", False):
                    await update.effective_message.reply_text(
                        f"✅ 任务完成：{entry}",
                        parse_mode=ParseMode.MARKDOWN
                    )
                await asyncio.sleep(3)
                await refresh_command(update, context)
        elif FANHAO_REGEX.match(entry):
            processing_msg = await update.effective_message.reply_text(f"🔍 搜索番号: {entry}")
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            magnet, error_msg = await search_magnet(entry, config.SEARCH_URLS, context)
            if not magnet:
                await processing_msg.edit_text(f"❌ 搜索失败: {error_msg}")
                return
            await processing_msg.edit_text(f"✅ 找到磁力链接")
            success, add_msg = await add_offline_download(context, token, [magnet])
            await processing_msg.edit_text(add_msg)
            if success:
                user_id = update.effective_user.id
                if context.bot_data.get(f"notify_task_{user_id}", False):
                    await update.effective_message.reply_text(
                        f"✅ 任务完成：{entry}",
                        parse_mode=ParseMode.MARKDOWN
                    )
                await asyncio.sleep(3)
                await refresh_command(update, context)
        else:
            await update.effective_message.reply_text("无法识别的格式")
    except aiohttp.ClientError as e:
        error_msg = "❌ 处理失败"
        if processing_msg:
            await processing_msg.edit_text(error_msg)
        else:
            await update.effective_message.reply_text(error_msg)

async def safe_edit_message(progress_msg, text, retry_count=3):
    """安全编辑消息"""
    for attempt in range(retry_count):
        try:
            await progress_msg.edit_text(text)
            return True
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except aiohttp.ClientError as e:
            logger.error(f"编辑消息失败: {str(e)}")
            return False
    return False

@restricted
async def handle_batch_entries(update: Update, context: ContextTypes.DEFAULT_TYPE, entries: List[str], *, token: str):
    """处理批量输入"""
    chat_id = update.effective_chat.id
    total_tasks = len(entries)
    progress_msg = await update.effective_message.reply_text(f"🔄 处理 {total_tasks} 个任务")
    try:
        task_results = []
        all_urls = []
        seen_magnets = set()
        success_count = 0
        update_interval = max(1, total_tasks // 5)
        for idx, entry in enumerate(entries, 1):
            if entry.startswith(("magnet:?", "ed2k://")):
                if entry not in seen_magnets:
                    all_urls.append((entry, entry))
                    seen_magnets.add(entry)
                    task_results.append(f"{idx}. {entry[:15]}...: 待添加")
                else:
                    task_results.append(f"{idx}. {entry[:15]}...: 重复")
            elif FANHAO_REGEX.match(entry):
                magnet, error_msg = await search_magnet(entry, config.SEARCH_URLS, context)
                if magnet and magnet not in seen_magnets:
                    all_urls.append((magnet, entry))
                    seen_magnets.add(magnet)
                    task_results.append(f"{idx}. {entry}: 待添加")
                else:
                    task_results.append(f"{idx}. {entry}: ❌无资源")
            else:
                task_results.append(f"{idx}. {entry[:15]}...: 无效")
            if idx % update_interval == 0 or idx == total_tasks:
                await safe_edit_message(progress_msg, f"⏳ 处理 {idx}/{total_tasks}\n" + "\n".join(task_results[-5:]))

        url_total = len(all_urls)
        for url_idx, (url, original_entry) in enumerate(all_urls, 1):
            success, msg = await add_offline_download(context, token, [url])
            for idx, entry in enumerate(entries, 1):
                if entry == original_entry or (FANHAO_REGEX.match(entry) and entry in original_entry):
                    if success:
                        task_results[idx-1] = f"{idx}. {original_entry[:15]}...: ✅成功"
                        success_count += 1
                        user_id = update.effective_user.id
                        if context.bot_data.get(f"notify_task_{user_id}", False):
                            await update.effective_message.reply_text(
                                f"✅ 任务完成：{original_entry}",
                                parse_mode=ParseMode.MARKDOWN
                            )
                    else:
                        task_results[idx-1] = f"{idx}. {original_entry[:15]}...: {msg}"
                    break
            if url_idx % update_interval == 0 or url_idx == url_total:
                await safe_edit_message(progress_msg, f"⏳ 添加 {url_idx}/{url_total}\n成功: {success_count}")

        summary = f"✅ 完成 (成功: {success_count}/{total_tasks})\n" + "\n".join(task_results[:10])
        if len(task_results) > 10:
            summary += f"\n...（共 {len(task_results)} 条）"
        await safe_edit_message(progress_msg, summary)
        if success_count > 0:
            await asyncio.sleep(3)
            await refresh_command(update, context)
    except aiohttp.ClientError as e:
        error_msg = f"❌ 批量处理失败\n" + "\n".join(task_results[:5])
        await safe_edit_message(progress_msg, error_msg)

@restricted
async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE, token: str) -> None:
    """处理用户消息"""
    message_text = update.message.text.strip()
    entries = [line.strip() for line in message_text.split('\n') if line.strip()]
    if not entries:
        await update.effective_message.reply_text("⚠️ 输入为空")
        return
    if len(entries) == 1:
        await handle_single_entry(update, context, token, entries[0])
    else:
        await handle_batch_entries(update, context, entries)

@restricted
async def clean_command(update: Update, context: ContextTypes.DEFAULT_TYPE, token: str) -> None:
    """清理小文件和空目录"""
    if config.SIZE_THRESHOLD == 0:
        await update.effective_message.reply_text("✅ 清理功能已禁用")
        return

    target = context.args[0].strip() if context.args else None
    chat_id = update.effective_chat.id
    target_display = target if target else "当前目录"
    processing_msg = await update.effective_message.reply_text(f"🧹 清理任务（目标: {target_display}）")

    try:
        current_dir = context.bot_data.get('current_download_dir', config.ALIST_OFFLINE_DIRS[0])
        results = []
        total_files = 0
        total_dirs_cleaned = 0

        # 确定目标目录
        if not target:
            target_dirs = [current_dir]
        else:
            directories, find_error = await find_download_directory(token, config.BASE_URL, current_dir, target)
            if not directories:
                await processing_msg.edit_text(f"❌ 清理失败: {find_error}")
                return
            target_dirs = directories

        # 统计小文件总数
        total_small_files = 0
        for target_dir in target_dirs:
            dir_files, _ = await recursive_collect_files(token, config.BASE_URL, target_dir)
            total_small_files += sum(len(files) for files in dir_files.values())

        deleted_small_files = [0]
        update_interval = max(10, total_small_files // 10) if total_small_files > 0 else 1

        async def update_small_files_progress():
            if deleted_small_files[0] % update_interval == 0 or deleted_small_files[0] == total_small_files:
                progress = int((deleted_small_files[0] / total_small_files) * 100) if total_small_files > 0 else 100
                progress_bar = "▰" * (progress // 10) + "▱" * (10 - progress // 10)
                await safe_edit_message(
                    processing_msg,
                    f"🧹 清理小文件: {progress}% [{progress_bar}]"
                )

        def small_files_progress_callback(deleted: int):
            deleted_small_files[0] += deleted
            asyncio.create_task(update_small_files_progress())

        # 清理小文件
        for target_dir in target_dirs:
            deleted, msg = await cleanup_small_files(
                token, config.BASE_URL, target_dir, progress_callback=small_files_progress_callback
            )
            total_files += deleted
            results.append(f"清理 {target_dir}:\n- {msg}")

        # 清理空文件夹
        empty_dirs = await recursive_collect_empty_dirs(token, config.BASE_URL, current_dir)
        total_empty_dirs = len([d for d in empty_dirs if not is_system_path(d, config.SYSTEM_FOLDERS)])
        deleted_empty_dirs = [0]
        update_interval_empty = max(5, total_empty_dirs // 10) if total_empty_dirs > 0 else 1

        async def update_empty_dirs_progress():
            if deleted_empty_dirs[0] % update_interval_empty == 0 or deleted_empty_dirs[0] == total_empty_dirs:
                progress = int((deleted_empty_dirs[0] / total_empty_dirs) * 100) if total_empty_dirs > 0 else 100
                progress_bar = "▰" * (progress // 10) + "▱" * (10 - progress // 10)
                await safe_edit_message(
                    processing_msg,
                    f"🧹 清理空文件夹: {progress}% [{progress_bar}]"
                )

        def empty_dirs_progress_callback(deleted: int):
            deleted_empty_dirs[0] += deleted
            asyncio.create_task(update_empty_dirs_progress())

        if total_empty_dirs > 0:
            await safe_edit_message(processing_msg, f"🧹 清理空文件夹: 0% [▱▱▱▱▱▱▱▱▱▱]")
            deleted_dirs, msg_dirs = await cleanup_empty_dirs(
                token, config.BASE_URL, current_dir, progress_callback=empty_dirs_progress_callback
            )
            total_dirs_cleaned += deleted_dirs
            results.append(f"清理空文件夹 {current_dir}:\n- {msg_dirs}")
        else:
            results.append(f"清理空文件夹 {current_dir}:\n- ✅ 未找到空文件夹")

        await safe_edit_message(processing_msg, "\n\n".join(results))
        await asyncio.sleep(3)
        await refresh_command(update, context)
    except aiohttp.ClientError as e:
        error_msg = f"❌ 清理失败: {str(e)}"
        await processing_msg.edit_text(error_msg)

@restricted
async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE, token: str) -> None:
    """刷新Alist文件列表"""
    refresh_url = f"{config.BASE_URL.rstrip('/')}/api/fs/list"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    payload = {"path": context.bot_data.get('current_download_dir', config.ALIST_OFFLINE_DIRS[0]), "page": 1, "per_page": 0, "refresh": True}
    chat_id = update.effective_chat.id
    processing_msg = await update.effective_message.reply_text("🔄 正在刷新 Alist 文件列表...")
    try:
        async with ProxiedClientSession() as session:
            async with session.post(refresh_url, json=payload, headers=headers, timeout=30) as response:
                response.raise_for_status()
                result = await response.json()
                if result.get("code") == 200:
                    await processing_msg.edit_text("✅ Alist 刷新成功！")
                else:
                    error_msg = result.get("message", "未知错误")
                    await processing_msg.edit_text(f"❌ 刷新失败: {error_msg}")
    except aiohttp.ClientError as e:
        logger.error(f"Alist 刷新错误: {str(e)}")
        error_msg = "❌ 刷新失败"
        await processing_msg.edit_text(error_msg)

@restricted
async def setdir_command(update: Update, context: ContextTypes.DEFAULT_TYPE, token: str) -> None:
    """通过内联键盘显示下载目录并选择"""
    dirs = config.ALIST_OFFLINE_DIRS
    if not dirs:
        await update.effective_message.reply_text("未配置下载目录。")
        return

    # 当前目录
    current_dir = context.bot_data.get('current_download_dir', dirs[0])
    current_index = context.bot_data.get('current_download_dir_index', 0)
    
    # 创建内联键盘，按钮显示目录名称，当前目录用 👉 标记
    keyboard = [
        [InlineKeyboardButton(f"{'👉 ' if dir == current_dir else ''}{dir}", callback_data=f"setdir_{i}")]
        for i, dir in enumerate(dirs)
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # 发送消息，仅包含提示
    try:
        await update.effective_message.reply_text(
            "点击下方按钮选择目录：",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        logger.info(f"Sent /setdir message with {len(dirs)} directories for user {update.effective_user.id}")
    except Exception as e:
        logger.error(f"Failed to send /setdir message for user {update.effective_user.id}: {str(e)}")
        await update.effective_message.reply_text("❌ 发送目录列表失败，请稍后重试")

async def setdir_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /setdir 按钮回调"""
    query = update.callback_query
    try:
        await query.answer()  # 回应回调
    except Exception as e:
        logger.error(f"Failed to answer callback query for user {query.from_user.id}: {str(e)}")
        return

    logger.debug(f"Received callback query with data: {query.data} from user {query.from_user.id}")
    
    if query.data.startswith("setdir_"):
        try:
            index = int(query.data[len("setdir_"):])
            dirs = config.ALIST_OFFLINE_DIRS
            if 0 <= index < len(dirs):
                context.bot_data['current_download_dir_index'] = index
                context.bot_data['current_download_dir'] = dirs[index]
                await query.message.edit_text(
                    f"✅ 已切换到目录 {index + 1}: {dirs[index]}",
                    parse_mode=ParseMode.MARKDOWN
                )
                logger.info(f"User {query.from_user.id} switched to directory {dirs[index]} (index {index})")
            else:
                await query.message.edit_text(
                    f"❌ 无效选择：目录索引 {index} 超出范围",
                    parse_mode=ParseMode.MARKDOWN
                )
                logger.warning(f"Invalid directory index {index} selected by user {query.from_user.id}")
        except ValueError as e:
            await query.message.edit_text(
                "❌ 无效选择：无法解析目录索引",
                parse_mode=ParseMode.MARKDOWN
            )
            logger.error(f"Failed to parse callback data '{query.data}' for user {query.from_user.id}: {str(e)}")
        except Exception as e:
            await query.message.edit_text(
                "❌ 处理回调失败，请稍后重试",
                parse_mode=ParseMode.MARKDOWN
            )
            logger.error(f"Unexpected error in setdir_callback for user {query.from_user.id}: {str(e)}")
    else:
        await query.message.edit_text(
            "❌ 无效回调数据",
            parse_mode=ParseMode.MARKDOWN
        )
        logger.warning(f"Unexpected callback data '{query.data}' from user {query.from_user.id}")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理其他按钮回调"""
    query = update.callback_query
    await query.answer()
    if query.data == 'help':
        await help_command(update, context)
    else:
        logger.debug(f"Ignored callback data '{query.data}' in button_callback for user {query.from_user.id}")

@restricted
async def reload_config(update: Update, context: ContextTypes.DEFAULT_TYPE, token: str) -> None:
    """热重新加载配置"""
    global config
    old_config = config
    new_config = Config()

    # 验证新配置
    if not new_config.validate():
        await update.effective_message.reply_text("❌ 配置重载失败：请检查 .env 文件")
        return

    # 更新全局配置
    config = new_config
    logger.info("配置已热重载")

    # 检查当前下载目录
    current_dir = context.bot_data.get('current_download_dir', old_config.ALIST_OFFLINE_DIRS[0])
    current_index = context.bot_data.get('current_download_dir_index', 0)
    if current_dir in config.ALIST_OFFLINE_DIRS:
        # 当前目录仍然有效，保持不变
        new_index = config.ALIST_OFFLINE_DIRS.index(current_dir)
        context.bot_data['current_download_dir'] = current_dir
        context.bot_data['current_download_dir_index'] = new_index
    else:
        # 当前目录不再有效，重置为第一个目录
        context.bot_data['current_download_dir'] = config.ALIST_OFFLINE_DIRS[0] if config.ALIST_OFFLINE_DIRS else ""
        context.bot_data['current_download_dir_index'] = 0

    # 更新自动清理任务
    job_queue = context.application.job_queue
    if job_queue:
        # 移除现有自动清理任务
        for job in job_queue.jobs():
            if job.name == "auto_clean":
                job.schedule_removal()
        
        # 如果清理功能启用，重新调度任务
        if config.SIZE_THRESHOLD > 0 and config.ALIST_OFFLINE_DIRS:
            job_queue.run_repeating(
                auto_clean,
                interval=config.CLEAN_INTERVAL_MINUTES * 60,
                first=config.CLEAN_INTERVAL_MINUTES * 60,
                name="auto_clean"
            )
            logger.info(f"自动清理任务已更新，间隔 {config.CLEAN_INTERVAL_MINUTES} 分钟")

    # 构建响应消息
    message = [
        "✅ 配置已热重载",
        f"• 当前目录：{context.bot_data.get('current_download_dir', '未知')}",
        f"• 清理间隔：{config.CLEAN_INTERVAL_MINUTES} 分钟",
        f"• 大小阈值：{config.SIZE_THRESHOLD // (1024 * 1024)} MB",
        f"• 下载目录数：{len(config.ALIST_OFFLINE_DIRS)}",
        f"• 搜索API数：{len(config.SEARCH_URLS)}",
        f"• 允许用户数：{len(config.ALLOWED_USER_IDS)}"
    ]
    await update.effective_message.reply_text("\n".join(message), parse_mode=ParseMode.MARKDOWN)

@restricted
async def notify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """开启或关闭任务完成通知和自动清理通知"""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} triggered /notify with args: {context.args}")
    if len(context.args) != 2:
        await update.effective_message.reply_text(
            "⚠️ 格式：/notify <任务|清理> <on|off>，例如：/notify 任务 on",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    notify_type = context.args[0].strip()
    state = context.args[1].lower()
    if notify_type not in ("任务", "清理"):
        await update.effective_message.reply_text(
            "⚠️ 类型必须为 '任务' 或 '清理'",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    if state not in ("on", "off"):
        await update.effective_message.reply_text(
            "⚠️ 状态必须为 on 或 off",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    key = f"notify_task_{user_id}" if notify_type == "任务" else f"notify_clean_{user_id}"
    context.bot_data[key] = (state == "on")
    logger.info(f"Set {key} to {state == 'on'} for user {user_id}")
    await update.effective_message.reply_text(
        f"✅ {notify_type}通知已{'开启' if state == 'on' else '关闭'}",
        parse_mode=ParseMode.MARKDOWN
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理错误"""
    logger.error(f"更新 {update} 导致错误: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text("❌ 发生错误")

def main() -> None:
    """启动机器人"""
    application_builder = Application.builder().token(config.TELEGRAM_TOKEN)
    if config.PROXY_URL:
        application_builder = application_builder.proxy(config.PROXY_URL).get_updates_proxy(config.PROXY_URL)
        logger.info(f"代理已启用: {config.PROXY_URL.rsplit('@', 1)[-1]}")
    application = application_builder.build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("classify", classify_command))
    application.add_handler(CommandHandler("clean", clean_command))
    application.add_handler(CommandHandler("refresh", refresh_command))
    application.add_handler(CommandHandler("setdir", setdir_command))
    application.add_handler(CommandHandler("reload_config", reload_config))
    application.add_handler(CommandHandler("notify", notify_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))
    application.add_handler(CallbackQueryHandler(setdir_callback, pattern=r"^setdir_\d+$"))  # 优先注册 setdir_callback
    application.add_handler(CallbackQueryHandler(button_callback))  # 其他回调处理器
    application.add_error_handler(error_handler)
    if config.SIZE_THRESHOLD > 0 and config.ALIST_OFFLINE_DIRS:
        job_queue = application.job_queue
        job_queue.run_repeating(
            auto_clean,
            interval=config.CLEAN_INTERVAL_MINUTES * 60,
            first=config.CLEAN_INTERVAL_MINUTES * 60,
            name="auto_clean"
        )
        logger.info(f"自动清理任务启用，间隔 {config.CLEAN_INTERVAL_MINUTES} 分钟")
        application.bot_data['current_download_dir'] = config.ALIST_OFFLINE_DIRS[0]
        # 初始化默认通知状态
        for user_id in config.ALLOWED_USER_IDS:
            if f"notify_task_{user_id}" not in application.bot_data:
                application.bot_data[f"notify_task_{user_id}"] = False  # 默认关闭任务通知
            if f"notify_clean_{user_id}" not in application.bot_data:
                application.bot_data[f"notify_clean_{user_id}"] = True  # 默认开启清理通知
    logger.info("机器人启动...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
