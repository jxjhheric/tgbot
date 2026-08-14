const FANHAO_REGEX = /^(?:[A-Za-z]{2,5}[-_ ]?\d{2,5}(?:[-_ ]?[A-Za-z]+)?|FC2-PPV-\d{6,})$/i;
const DEFAULT_HEADERS = { "User-Agent": "tgbot-worker/1.0" };

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return json({ ok: true, service: "tgbot-worker" });
    }
    if (request.method !== "POST" || url.pathname !== "/webhook") {
      return new Response("Not found", { status: 404 });
    }
    if (env.WEBHOOK_SECRET && request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.WEBHOOK_SECRET) {
      return new Response("Unauthorized", { status: 401 });
    }
    try {
      const update = await request.json();
      ctx.waitUntil(handleUpdate(update, env));
      return new Response("ok");
    } catch (error) {
      console.error("Webhook error", error);
      return new Response("Bad request", { status: 400 });
    }
  },

  async scheduled(controller, env, ctx) {
    ctx.waitUntil(runScheduledCleanup(env));
  },
};

function json(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function config(env) {
  const directories = split(env.ALIST_OFFLINE_DIRS);
  const searchUrls = split(env.JAV_SEARCH_APIS);
  const allowedUsers = new Set(split(env.ALLOWED_USER_IDS).map(Number).filter(Number.isInteger));
  return {
    token: env.TELEGRAM_TOKEN,
    alistBaseUrl: (env.ALIST_BASE_URL || "").replace(/\/$/, ""),
    alistToken: env.ALIST_TOKEN || "",
    directories,
    searchUrls,
    allowedUsers,
    sizeThreshold: Math.max(0, Number(env.SIZE_THRESHOLD || 100)) * 1024 * 1024,
    preferredKeywords: split(env.PREFERRED_KEYWORDS).map((item) => item.toLowerCase()),
    customCategories: parseCategories(env.CUSTOM_CATEGORIES),
    systemFolders: split(env.SYSTEM_FOLDERS).map(normalizePath),
    excludeSuffixes: split(env.EXCLUDE_SUFFIXES).map((item) => item.toLowerCase()),
  };
}

function validateConfig(settings) {
  return [
    [settings.token, "TELEGRAM_TOKEN"],
    [settings.alistBaseUrl, "ALIST_BASE_URL"],
    [settings.alistToken, "ALIST_TOKEN"],
    [settings.directories.length, "ALIST_OFFLINE_DIRS"],
    [settings.searchUrls.length, "JAV_SEARCH_APIS"],
    [settings.allowedUsers.size, "ALLOWED_USER_IDS"],
  ].filter(([value]) => !value).map(([, name]) => name);
}

function split(value) {
  return String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
}

function parseCategories(value) {
  return String(value || "").split(";").map((item) => {
    const [name, keywords] = item.split(":");
    return { name: (name || "").trim(), keywords: split(keywords).map((keyword) => keyword.toLowerCase()) };
  }).filter((item) => item.name && item.keywords.length);
}

async function handleUpdate(update, env) {
  const settings = config(env);
  const missing = validateConfig(settings);
  if (missing.length) {
    console.error("Missing environment variables", missing);
    return;
  }
  if (update.callback_query) {
    await handleCallback(update.callback_query, settings, env);
    return;
  }
  const message = update.message;
  if (!message?.chat?.id) return;
  const userId = message.from?.id || message.chat.id;
  if (!settings.allowedUsers.has(userId)) {
    await sendMessage(settings, message.chat.id, "抱歉，您没有权限使用此机器人。");
    return;
  }
  const text = String(message.text || "").trim();
  if (!text) return;
  if (text.startsWith("/")) {
    await handleCommand(message, text, settings, env);
  } else {
    await handleEntries(message, text.split("\n").map((item) => item.trim()).filter(Boolean), settings, env);
  }
}

async function handleCommand(message, text, settings, env) {
  const parts = text.split(/\s+/);
  const command = parts[0].split("@")[0].toLowerCase();
  const args = parts.slice(1);
  switch (command) {
    case "/start": return start(message, settings);
    case "/help": return help(message, settings, env);
    case "/clean": return clean(message, args, settings, env);
    case "/refresh": return refresh(message, settings, env);
    case "/setdir": return setDirectory(message, settings, env);
    case "/classify": return classify(message, args, settings, env);
    case "/notify": return notify(message, args, settings, env);
    case "/reload_config": return reloadConfig(message, settings);
    default: return sendMessage(settings, message.chat.id, "未知命令，请发送 /help 查看帮助。");
  }
}

async function start(message, settings) {
  return sendMessage(settings, message.chat.id,
    "欢迎使用 JAV 下载机器人！\n发送番号或磁力/ed2k 链接以添加到 Alist。",
    { inline_keyboard: [[{ text: "查看帮助", callback_data: "help" }]] });
}

async function help(message, settings, env) {
  const current = await getCurrentDirectory(settings, env);
  return sendMessage(settings, message.chat.id,
    "JAV 下载机器人\n\n" +
    "发送番号（如 ABC-123 或 FC2-PPV-1010519）自动搜索并添加。\n" +
    "也支持 magnet:? 和 ed2k:// 链接。\n\n" +
    "/clean [番号] 清理小文件和空目录\n" +
    "/setdir 选择下载目录\n" +
    "/classify [all|folder|file] 整理目录\n" +
    "/refresh 刷新 Alist\n" +
    "/notify 任务|清理 on|off 设置通知\n" +
    `/reload_config 查看当前 Worker 配置\n\n当前目录：${current}`);
}

async function handleEntries(message, entries, settings, env) {
  if (entries.length === 1) return handleSingleEntry(message, entries[0], settings, env);
  const status = await sendMessage(settings, message.chat.id, `🔄 处理 ${entries.length} 个任务`);
  const results = [];
  let success = 0;
  const seen = new Set();
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const result = await resolveEntry(entry, settings);
    if (!result.magnet) {
      results.push(`${index + 1}. ${entry.slice(0, 20)}: ${result.error || "无资源"}`);
      continue;
    }
    if (seen.has(result.magnet)) {
      results.push(`${index + 1}. ${entry.slice(0, 20)}: 重复`);
      continue;
    }
    seen.add(result.magnet);
    const added = await addOfflineDownload(settings, env, [result.magnet]);
    if (added.ok) {
      success += 1;
      results.push(`${index + 1}. ${entry.slice(0, 20)}: ✅成功`);
      if (await notificationsEnabled(env, message.from.id, "task")) {
        await sendMessage(settings, message.chat.id, `✅ 任务完成：${entry}`);
      }
    } else {
      results.push(`${index + 1}. ${entry.slice(0, 20)}: ${added.message}`);
    }
  }
  await editMessage(settings, message.chat.id, status.result.message_id, `✅ 完成（成功: ${success}/${entries.length}）\n${results.slice(0, 10).join("\n")}`);
}

async function handleSingleEntry(message, entry, settings, env) {
  const status = await sendMessage(settings, message.chat.id, `🔍 处理：${entry}`);
  const result = await resolveEntry(entry, settings);
  if (!result.magnet) {
    return editMessage(settings, message.chat.id, status.result.message_id, result.error || "❌ 无法识别的格式");
  }
  const added = await addOfflineDownload(settings, env, [result.magnet]);
  await editMessage(settings, message.chat.id, status.result.message_id, added.message);
  if (added.ok && await notificationsEnabled(env, message.from.id, "task")) {
    await sendMessage(settings, message.chat.id, `✅ 任务完成：${entry}`);
  }
}

async function resolveEntry(entry, settings) {
  if (/^(magnet:\?|ed2k:\/\/)/i.test(entry)) return { magnet: entry };
  if (!FANHAO_REGEX.test(entry)) return { error: "❌ 无效番号格式" };
  const result = await searchMagnet(entry, settings);
  return result.magnet ? result : { error: result.error || "❌ 未找到资源" };
}

async function searchMagnet(code, settings) {
  const results = await Promise.allSettled([
    ...settings.searchUrls.map((url) => fetchSearchApi(url, code)),
    scrapeSukebei(code),
  ]);
  const entries = results.flatMap((result) => result.status === "fulfilled" ? result.value : []);
  const unique = [...new Map(entries.map((entry) => [entry.magnet, entry])).values()];
  if (!unique.length) return { error: `🔍 未找到番号 ${code} 的相关资源` };
  unique.sort((a, b) => scoreEntry(b, settings) - scoreEntry(a, settings));
  return { magnet: unique[0].magnet };
}

async function fetchSearchApi(baseUrl, code) {
  try {
    const response = await fetch(`${baseUrl.replace(/\/$/, "")}/${encodeURIComponent(code)}`, { headers: DEFAULT_HEADERS });
    if (!response.ok) return [];
    const body = await response.json();
    if (body?.status !== "succeed") return [];
    return (body.data || []).map(parseSearchEntry).filter((entry) => entry && isFanhaoMatch(code, entry.name));
  } catch (error) {
    console.error("Search API error", error);
    return [];
  }
}

async function scrapeSukebei(code) {
  try {
    const response = await fetch(`https://sukebei.nyaa.si/?f=0&c=0_0&q=${encodeURIComponent(code)}`, { headers: DEFAULT_HEADERS });
    if (!response.ok) return [];
    const html = await response.text();
    const rows = html.match(/<tr[^>]*class=["'][^"']*(?:default|success)[^"']*["'][\s\S]*?<\/tr>/gi) || [];
    return rows.map((row) => {
      const name = decodeHtml(stripTags((row.match(/<td[^>]*>\s*<a[^>]*>([\s\S]*?)<\/a>/i) || [])[1] || "未知")).trim();
      const magnet = (row.match(/href=["'](magnet:\?[^"']+)/i) || [])[1];
      const size = decodeHtml(stripTags((row.match(/<td[^>]*>([^<]*(?:[KMGT]B|[KMGT]iB)[^<]*)<\/td>/i) || [])[1] || ""));
      if (!magnet || !isFanhaoMatch(code, name)) return null;
      return { magnet, name, sizeBytes: parseSize(size), seeders: 0, date: null, source: "sukebei" };
    }).filter(Boolean);
  } catch (error) {
    console.error("Sukebei error", error);
    return [];
  }
}

function parseSearchEntry(value) {
  if (Array.isArray(value)) {
    const [magnet, name, size, date] = value;
    return magnet?.startsWith("magnet:?") ? { magnet, name: String(name || ""), sizeBytes: parseSize(size), date, seeders: 0, source: "api" } : null;
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return parseSearchEntry(parsed);
    } catch {
      const parts = value.match(/^\s*\[\s*["'](magnet:\?[^"']+)["']\s*,\s*["']([\s\S]*?)["']\s*,\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\]\s*$/);
      if (parts) return { magnet: parts[1], name: parts[2], sizeBytes: parseSize(parts[3]), date: parts[4], seeders: 0, source: "api" };
    }
  }
  return null;
}

function scoreEntry(entry, settings) {
  const keyword = settings.preferredKeywords.some((item) => entry.name.toLowerCase().includes(item)) ? 1e15 : 0;
  const date = entry.date ? Date.parse(entry.date) || 0 : 0;
  return keyword + (entry.seeders || 0) * 1e9 + date + (entry.sizeBytes || 0);
}

function isFanhaoMatch(code, name) {
  return normalizeCode(name).includes(normalizeCode(code));
}

function normalizeCode(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function parseSize(value) {
  const match = String(value || "").toLowerCase().match(/(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb)/);
  if (!match) return 0;
  return Math.floor(Number(match[1]) * ({ kb: 1024, mb: 1024 ** 2, gb: 1024 ** 3, tb: 1024 ** 4 }[match[2]]));
}

function stripTags(value) { return String(value || "").replace(/<[^>]+>/g, ""); }
function decodeHtml(value) { return String(value || "").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">"); }

async function addOfflineDownload(settings, env, links, directory = null) {
  const result = await alistRequest(settings, "/api/fs/add_offline_download", {
    path: directory || await getCurrentDirectory(settings, env),
    urls: links,
    tool: "storage",
    delete_policy: "delete_on_upload_succeed",
  });
  return result.ok && result.data?.code === 200
    ? { ok: true, message: "✅ 已添加到下载队列" }
    : { ok: false, message: result.status === 401 ? "❌ 认证失败" : `❌ 添加失败：${result.data?.message || "服务器错误"}` };
}

async function handleCallback(query, settings, env) {
  const userId = query.from?.id;
  if (!settings.allowedUsers.has(userId)) return answerCallback(settings, query.id, "无权限");
  await answerCallback(settings, query.id);
  if (query.data === "help") return help({ chat: { id: query.message.chat.id }, from: query.from }, settings, env);
  if (!query.data?.startsWith("setdir_")) return;
  const index = Number(query.data.slice(7));
  if (!Number.isInteger(index) || !settings.directories[index]) return editMessage(settings, query.message.chat.id, query.message.message_id, "❌ 无效目录");
  const globalState = await getState(env, "global");
  await setState(env, "global", { ...globalState, currentDir: settings.directories[index], currentIndex: index });
  return editMessage(settings, query.message.chat.id, query.message.message_id, `✅ 已切换到目录 ${index + 1}: ${settings.directories[index]}`);
}

async function setDirectory(message, settings, env) {
  const state = await getState(env, `user:${message.from.id}`);
  const current = state.currentDir || settings.directories[0];
  const keyboard = settings.directories.map((directory, index) => [{ text: `${directory === current ? "👉 " : ""}${directory}`, callback_data: `setdir_${index}` }]);
  return sendMessage(settings, message.chat.id, "点击下方按钮选择目录：", { inline_keyboard: keyboard });
}

async function refresh(message, settings, env) {
  const result = await alistRequest(settings, "/api/fs/list", { path: await getCurrentDirectory(settings, env), page: 1, per_page: 0, refresh: true });
  return sendMessage(settings, message.chat.id, result.ok && result.data?.code === 200 ? "✅ Alist 刷新成功！" : `❌ 刷新失败：${result.data?.message || "服务器错误"}`);
}

async function reloadConfig(message, settings) {
  const missing = validateConfig(settings);
  return sendMessage(settings, message.chat.id, missing.length ? `❌ 缺少配置：${missing.join(", ")}` : "✅ Worker 配置有效。修改 Secret/变量后新请求会自动使用新配置。");
}

async function notify(message, args, settings, env) {
  if (args.length !== 2 || !["任务", "清理"].includes(args[0]) || !["on", "off"].includes(args[1].toLowerCase())) {
    return sendMessage(settings, message.chat.id, "⚠️ 格式：/notify <任务|清理> <on|off>");
  }
  const state = await getState(env, `user:${message.from.id}`);
  state[args[0] === "任务" ? "notifyTask" : "notifyClean"] = args[1].toLowerCase() === "on";
  await setState(env, `user:${message.from.id}`, state);
  return sendMessage(settings, message.chat.id, `✅ ${args[0]}通知已${state[args[0] === "任务" ? "notifyTask" : "notifyClean"] ? "开启" : "关闭"}`);
}

async function classify(message, args, settings, env) {
  let type = "all";
  let target = null;
  if (args[0] && ["all", "folder", "file"].includes(args[0].toLowerCase())) {
    type = args[0].toLowerCase(); target = args[1] || null;
  } else {
    target = args[0] || null;
    if (args[1] && ["all", "folder", "file"].includes(args[1].toLowerCase())) type = args[1].toLowerCase();
  }
  const current = await getCurrentDirectory(settings, env);
  const targetDir = target ? normalizePath(`${current}/${target}`) : current;
  const content = await listDirectory(settings, targetDir);
  if (!content) return sendMessage(settings, message.chat.id, "❌ 未找到有效目标目录");
  const topDirs = ["JAV", ...settings.customCategories.map((item) => item.name), "其他"];
  const moves = new Map();
  for (const item of content) {
    if (!shouldProcess(item, type, topDirs, targetDir, settings.systemFolders)) continue;
    const destination = `${targetDir}/${destinationSubdir(item.name, settings.customCategories)}`;
    if (!moves.has(destination)) moves.set(destination, []);
    moves.get(destination).push(item.name);
  }
  let moved = 0;
  let failed = 0;
  for (const [destination, names] of moves) {
    if (!(await ensureDirectory(settings, destination)).ok) { failed += names.length; continue; }
    for (let index = 0; index < names.length; index += 5) {
      const result = await moveItems(settings, targetDir, names.slice(index, index + 5), destination);
      if (result.ok) moved += names.slice(index, index + 5).length;
      else failed += names.slice(index, index + 5).length;
    }
  }
  return sendMessage(settings, message.chat.id, `✅ 整理完成！\n• 处理目录: 1\n• 移动项目: ${moved}\n• 失败项目: ${failed}`);
}

function shouldProcess(item, type, topDirs, targetDir, systemFolders) {
  if (type === "folder" && !item.is_dir) return false;
  if (type === "file" && item.is_dir) return false;
  if (topDirs.includes(item.name)) return false;
  return !isSystemPath(`${targetDir}/${item.name}`, systemFolders);
}

function destinationSubdir(name, categories) {
  const clean = String(name || "").replace(/[\u{1F300}-\u{1FAFF}\u2600-\u27BF]/gu, "").replace(/(?:Chinese|English)\s*subtitles?|中文字幕|英文字幕|高清|无码|無碼|无水印|無水印|破解版|破解|\[.*?\]|【.*?】|\(.*?\)/gi, "");
  let match = clean.match(/(FC2)[-_ ]?(PPV)[-_ ]?(\d{6,})/i) || clean.match(/(FC2)[-_ ]?(\d{6,})/i);
  if (match) return "JAV/FC2";
  match = clean.match(/([A-Z]{2,5})[-_ ]?(\d{2,7})(?:[-_ ]?([A-Z]+))?/i) || clean.match(/([A-Z]{4,})(\d{3,})/i);
  if (match) return `JAV/${match[1].toUpperCase()[0]}`;
  const lower = String(name || "").toLowerCase();
  return categories.find((category) => category.keywords.some((keyword) => lower.includes(keyword)))?.name || "其他";
}

async function clean(message, args, settings, env, baseDirectory = null) {
  if (!settings.sizeThreshold) return sendMessage(settings, message.chat.id, "✅ 清理功能已禁用");
  const current = baseDirectory || await getCurrentDirectory(settings, env);
  const target = args[0] ? normalizePath(`${current}/${args[0]}`) : current;
  const files = await collectFiles(settings, target);
  let deleted = 0;
  for (const [directory, names] of files) {
    const result = await alistRequest(settings, "/api/fs/remove", { dir: directory, names });
    if (result.ok && result.data?.code === 200) deleted += names.length;
  }
  const emptyDirs = await collectEmptyDirs(settings, current);
  let removedDirs = 0;
  for (const directory of emptyDirs) {
    if (isSystemPath(directory, settings.systemFolders)) continue;
    const parent = directory.slice(0, directory.lastIndexOf("/")) || "/";
    const name = directory.slice(directory.lastIndexOf("/") + 1);
    const result = await alistRequest(settings, "/api/fs/remove", { dir: parent, names: [name] });
    if (result.ok && result.data?.code === 200) removedDirs += 1;
  }
  return sendMessage(settings, message.chat.id, `✅ 清理完成\n• 删除小文件: ${deleted}\n• 删除空目录: ${removedDirs}`);
}

async function collectFiles(settings, root) {
  const result = new Map();
  const queue = [root];
  while (queue.length) {
    const directory = queue.shift();
    const content = await listDirectory(settings, directory) || [];
    for (const item of content) {
      const path = normalizePath(`${directory}/${item.name}`);
      if (item.is_dir) queue.push(path);
      else if (Number(item.size || 0) < settings.sizeThreshold && !settings.excludeSuffixes.some((suffix) => item.name.toLowerCase().endsWith(suffix))) {
        if (!result.has(directory)) result.set(directory, []);
        result.get(directory).push(item.name);
      }
    }
  }
  return result;
}

async function collectEmptyDirs(settings, root) {
  const empty = [];
  const queue = [root];
  while (queue.length) {
    const directory = queue.shift();
    const content = await listDirectory(settings, directory) || [];
    const childDirs = content.filter((item) => item.is_dir);
    queue.push(...childDirs.map((item) => normalizePath(`${directory}/${item.name}`)));
    if (!content.length && directory !== root) empty.push(directory);
  }
  return empty.sort((a, b) => b.length - a.length);
}

async function listDirectory(settings, path) {
  const result = await alistRequest(settings, "/api/fs/list", { path, page: 1, per_page: 0, refresh: true });
  return result.ok && result.data?.code === 200 ? result.data?.data?.content || [] : null;
}

async function ensureDirectory(settings, path) {
  const parts = normalizePath(path).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current += `/${part}`;
    const parent = current.slice(0, current.lastIndexOf("/")) || "/";
    const content = await listDirectory(settings, parent);
    if (content?.some((item) => item.name === part && item.is_dir)) continue;
    const result = await alistRequest(settings, "/api/fs/mkdir", { path: current });
    if (!result.ok || result.data?.code !== 200) return { ok: false, message: result.data?.message || "创建目录失败" };
  }
  return { ok: true };
}

async function moveItems(settings, source, names, destination) {
  const result = await alistRequest(settings, "/api/fs/move", { src_dir: normalizePath(source), dst_dir: normalizePath(destination), names });
  return { ok: result.ok && result.data?.code === 200, message: result.data?.message };
}

async function alistRequest(settings, path, body) {
  try {
    const response = await fetch(`${settings.alistBaseUrl}${path}`, {
      method: "POST",
      headers: { Authorization: settings.alistToken, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let data = null;
    try { data = await response.json(); } catch { data = {}; }
    return { ok: response.ok, status: response.status, data };
  } catch (error) {
    console.error("Alist request error", path, error);
    return { ok: false, status: 0, data: { message: String(error) } };
  }
}

async function getCurrentDirectory(settings, env) {
  const state = await getState(env, "global");
  return state.currentDir && settings.directories.includes(state.currentDir) ? state.currentDir : settings.directories[0];
}

async function notificationsEnabled(env, userId, type) {
  const state = await getState(env, `user:${userId}`);
  return type === "clean" ? state.notifyClean !== false : state.notifyTask === true;
}

async function getState(env, key) {
  if (!env.STATE) return {};
  return await env.STATE.get(key, "json") || {};
}

async function setState(env, key, value) {
  if (env.STATE) await env.STATE.put(key, JSON.stringify(value));
}

async function runScheduledCleanup(env) {
  const settings = config(env);
  if (validateConfig(settings).length || !settings.sizeThreshold) return;
  for (const directory of settings.directories) {
    const fakeMessage = { chat: { id: Number(env.ADMIN_CHAT_ID || [...settings.allowedUsers][0]) }, from: { id: Number(env.ADMIN_CHAT_ID || [...settings.allowedUsers][0]) } };
    await clean(fakeMessage, [], settings, env, directory);
  }
}

async function telegram(settings, method, payload) {
  const response = await fetch(`https://api.telegram.org/bot${settings.token}/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  return { status: response.status, ...(await response.json()) };
}

async function sendMessage(settings, chatId, text, replyMarkup = null) {
  const payload = { chat_id: chatId, text, disable_web_page_preview: true };
  if (replyMarkup) payload.reply_markup = replyMarkup;
  return telegram(settings, "sendMessage", payload);
}

async function editMessage(settings, chatId, messageId, text) {
  return telegram(settings, "editMessageText", { chat_id: chatId, message_id: messageId, text, disable_web_page_preview: true });
}

async function answerCallback(settings, callbackQueryId, text = "") {
  return telegram(settings, "answerCallbackQuery", { callback_query_id: callbackQueryId, text });
}

function normalizePath(path) {
  const parts = String(path || "").replaceAll("\\", "/").split("/");
  const result = [];
  for (const part of parts) {
    if (!part || part === ".") continue;
    if (part === "..") result.pop();
    else result.push(part);
  }
  return `/${result.join("/")}`.replace(/\/$/, "") || "/";
}

function isSystemPath(path, folders) {
  const normalized = normalizePath(path).toLowerCase();
  return folders.some((folder) => normalized === folder.toLowerCase() || normalized.startsWith(`${folder.toLowerCase()}/`));
}
