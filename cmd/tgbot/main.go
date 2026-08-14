package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/proxy"
)

const (
	defaultStateFile = "/var/lib/tgbot/state.json"
	telegramTimeout = 55 * time.Second
)

var fanhaoPattern = regexp.MustCompile(`(?i)^(?:[a-z]{2,5}[-_ ]?\d{2,5}(?:[-_ ]?[a-z]+)?|FC2-PPV-\d{6,})$`)

type Config struct {
	TelegramToken    string
	ProxyURL         string
	AlistBaseURL     string
	AlistToken       string
	OfflineDirs      []string
	SearchURLs       []string
	AllowedUsers     map[int64]bool
	CleanInterval    time.Duration
	SizeThreshold    int64
	PreferredWords   []string
	CustomCategories []Category
	SystemFolders    []string
	CleanBatchSize   int
	CleanRequestWait time.Duration
	MaxConcurrent    int
	ExcludeSuffixes  []string
	StateFile        string
}

type Category struct {
	Name     string
	Keywords []string
}

type State struct {
	CurrentDirectory string           `json:"current_directory"`
	CurrentIndex     int              `json:"current_index"`
	NotifyTask       map[int64]bool   `json:"notify_task"`
	NotifyClean      map[int64]bool   `json:"notify_clean"`
}

type Bot struct {
	cfg        Config
	client     *http.Client
	state      State
	stateMu    sync.Mutex
	updateMu   sync.Mutex
	offset     int64
	cleanMu    sync.Mutex
}

type TelegramUpdate struct {
	UpdateID      int64              `json:"update_id"`
	Message       *TelegramMessage   `json:"message"`
	CallbackQuery *CallbackQuery     `json:"callback_query"`
}

type TelegramMessage struct {
	MessageID int64          `json:"message_id"`
	Chat      TelegramChat   `json:"chat"`
	From      TelegramUser   `json:"from"`
	Text      string         `json:"text"`
}

type TelegramChat struct { ID int64 `json:"id"` }
type TelegramUser struct { ID int64 `json:"id"` }

type CallbackQuery struct {
	ID      string           `json:"id"`
	From    TelegramUser     `json:"from"`
	Data    string           `json:"data"`
	Message *TelegramMessage `json:"message"`
}

type InlineKeyboardMarkup struct {
	InlineKeyboard [][]InlineKeyboardButton `json:"inline_keyboard"`
}

type InlineKeyboardButton struct {
	Text         string `json:"text"`
	CallbackData string `json:"callback_data,omitempty"`
}

type alistResponse struct {
	Code    int                    `json:"code"`
	Message string                 `json:"message"`
	Data    map[string]interface{} `json:"data"`
}

type httpStatusError struct {
	StatusCode int
	Message    string
}

func (e *httpStatusError) Error() string { return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message) }

type alistItem struct {
	Name  string `json:"name"`
	IsDir bool   `json:"is_dir"`
	Size  int64  `json:"size"`
}

type SearchEntry struct {
	Magnet   string
	Name     string
	Size     int64
	Seeders  int
	Date     time.Time
	Source   string
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}
	client, err := newHTTPClient(cfg.ProxyURL)
	if err != nil {
		log.Fatal(err)
	}
	bot := &Bot{cfg: cfg, client: client}
	if err := bot.loadState(); err != nil {
		log.Printf("state load warning: %v", err)
	}
	log.Printf("tgbot Go service started; directories=%d proxy=%t", len(cfg.OfflineDirs), cfg.ProxyURL != "")
	go bot.cleanLoop(context.Background())
	bot.poll(context.Background())
}

func loadConfig() (Config, error) {
	cfg := Config{
		TelegramToken:    os.Getenv("TELEGRAM_TOKEN"),
		ProxyURL:         strings.TrimSpace(os.Getenv("PROXY_URL")),
		AlistBaseURL:     strings.TrimRight(strings.TrimSpace(os.Getenv("ALIST_BASE_URL")), "/"),
		AlistToken:       os.Getenv("ALIST_TOKEN"),
		OfflineDirs:      splitCSV(os.Getenv("ALIST_OFFLINE_DIRS")),
		SearchURLs:       splitCSV(os.Getenv("JAV_SEARCH_APIS")),
		AllowedUsers:     parseUserIDs(os.Getenv("ALLOWED_USER_IDS")),
		PreferredWords:   lowerCSV(os.Getenv("PREFERRED_KEYWORDS")),
		CustomCategories: parseCategories(os.Getenv("CUSTOM_CATEGORIES")),
		SystemFolders:    normalizeCSVPaths(os.Getenv("SYSTEM_FOLDERS")),
		ExcludeSuffixes:  lowerCSV(os.Getenv("EXCLUDE_SUFFIXES")),
		CleanBatchSize:   envInt("CLEAN_BATCH_SIZE", 500),
		MaxConcurrent:    envInt("MAX_CONCURRENT_REQUESTS", 20),
		StateFile:        envString("STATE_FILE", defaultStateFile),
	}
	interval := envInt("CLEAN_INTERVAL_MINUTES", 60)
	thresholdMB := envInt("SIZE_THRESHOLD", 100)
	waitMS := envInt("CLEAN_REQUEST_INTERVAL_MS", 200)
	cfg.CleanInterval = time.Duration(interval) * time.Minute
	cfg.SizeThreshold = int64(thresholdMB) * 1024 * 1024
	cfg.CleanRequestWait = time.Duration(waitMS) * time.Millisecond
	if cfg.CleanBatchSize < 1 { cfg.CleanBatchSize = 500 }
	if cfg.MaxConcurrent < 1 { cfg.MaxConcurrent = 20 }
	missing := make([]string, 0)
	for name, value := range map[string]string{
		"TELEGRAM_TOKEN": cfg.TelegramToken, "ALIST_BASE_URL": cfg.AlistBaseURL,
		"ALIST_TOKEN": cfg.AlistToken, "ALIST_OFFLINE_DIRS": strings.Join(cfg.OfflineDirs, ","),
		"JAV_SEARCH_APIS": strings.Join(cfg.SearchURLs, ","),
	} {
		if value == "" { missing = append(missing, name) }
	}
	if len(cfg.AllowedUsers) == 0 { missing = append(missing, "ALLOWED_USER_IDS") }
	if len(missing) > 0 { return Config{}, fmt.Errorf("missing environment variables: %s", strings.Join(missing, ", ")) }
	return cfg, nil
}

func newHTTPClient(proxyURL string) (*http.Client, error) {
	transport := &http.Transport{}
	if proxyURL == "" {
		transport.Proxy = http.ProxyFromEnvironment
		return &http.Client{Transport: transport}, nil
	}
	u, err := url.Parse(proxyURL)
	if err != nil { return nil, fmt.Errorf("invalid PROXY_URL: %w", err) }
	if strings.HasPrefix(strings.ToLower(u.Scheme), "socks5") {
		var authentication *proxy.Auth
		if u.User != nil {
			password, _ := u.User.Password()
			authentication = &proxy.Auth{User: u.User.Username(), Password: password}
		}
		dialer, err := proxy.SOCKS5("tcp", u.Host, authentication, proxy.Direct)
		if err != nil { return nil, fmt.Errorf("create socks5 proxy: %w", err) }
		transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
			return dialer.Dial(network, address)
		}
	} else {
		transport.Proxy = http.ProxyURL(u)
	}
	return &http.Client{Transport: transport}, nil
}

func (b *Bot) poll(ctx context.Context) {
	for {
		updates, err := b.getUpdates(ctx)
		if err != nil {
			log.Printf("telegram polling error: %v", err)
			time.Sleep(5 * time.Second)
			continue
		}
		for _, update := range updates {
			if update.UpdateID >= b.offset { b.offset = update.UpdateID + 1 }
			go b.handleUpdate(ctx, update)
		}
	}
}

func (b *Bot) getUpdates(ctx context.Context) ([]TelegramUpdate, error) {
	payload := map[string]interface{}{"offset": b.offset, "timeout": 50, "allowed_updates": []string{"message", "callback_query"}}
	var result struct { OK bool `json:"ok"`; Result []TelegramUpdate `json:"result"`; Description string `json:"description"` }
	err := b.telegramRequest(ctx, "getUpdates", payload, &result, telegramTimeout)
	if err != nil { return nil, err }
	if !result.OK { return nil, errors.New(result.Description) }
	return result.Result, nil
}

func (b *Bot) handleUpdate(ctx context.Context, update TelegramUpdate) {
	if update.CallbackQuery != nil { b.handleCallback(ctx, update.CallbackQuery); return }
	message := update.Message
	if message == nil || message.Chat.ID == 0 { return }
	if !b.cfg.AllowedUsers[message.From.ID] {
		b.sendMessage(ctx, message.Chat.ID, "抱歉，您没有权限使用此机器人。")
		return
	}
	text := strings.TrimSpace(message.Text)
	if text == "" { return }
	if strings.HasPrefix(text, "/") { b.handleCommand(ctx, message, text); return }
	entries := nonEmptyLines(text)
	if len(entries) == 1 { b.handleSingle(ctx, message, entries[0]); return }
	b.handleBatch(ctx, message, entries)
}

func (b *Bot) handleCommand(ctx context.Context, message *TelegramMessage, text string) {
	parts := strings.Fields(text)
	command := strings.ToLower(strings.Split(parts[0], "@")[0])
	args := parts[1:]
	switch command {
	case "/start": b.start(ctx, message)
	case "/help": b.help(ctx, message)
	case "/clean": b.cleanCommand(ctx, message, args)
	case "/refresh": b.refresh(ctx, message)
	case "/setdir": b.setDir(ctx, message)
	case "/classify": b.classify(ctx, message, args)
	case "/reload_config": b.reloadConfig(ctx, message)
	case "/notify": b.notify(ctx, message, args)
	default: b.sendMessage(ctx, message.Chat.ID, "未知命令，请发送 /help 查看帮助。")
	}
}

func (b *Bot) start(ctx context.Context, message *TelegramMessage) {
	markup := InlineKeyboardMarkup{InlineKeyboard: [][]InlineKeyboardButton{{{Text: "查看帮助", CallbackData: "help"}}}}
	b.sendMessageWithMarkup(ctx, message.Chat.ID, "欢迎使用 JAV 下载机器人！\n发送番号或磁力/ed2k 链接以添加到 Alist。", markup)
}

func (b *Bot) help(ctx context.Context, message *TelegramMessage) {
	current := b.currentDirectory()
	text := "JAV 下载机器人\n\n" +
		"发送番号（如 ABC-123 或 FC2-PPV-1010519）自动搜索并添加。\n" +
		"也支持 magnet:? 和 ed2k:// 链接。\n\n" +
		"/clean [番号] 清理小文件和空目录\n" +
		"/setdir 选择下载目录\n" +
		"/classify [all|folder|file] 整理目录\n" +
		"/refresh 刷新 Alist\n" +
		"/notify 任务|清理 on|off 设置通知\n" +
		"/reload_config 重新加载配置\n\n当前目录：" + current
	b.sendMessage(ctx, message.Chat.ID, text)
}

func (b *Bot) handleSingle(ctx context.Context, message *TelegramMessage, entry string) {
	status := b.sendMessage(ctx, message.Chat.ID, "🔍 处理："+entry)
	magnet, err := b.resolveEntry(ctx, entry)
	if err != nil { b.editMessage(ctx, message.Chat.ID, status, "❌ "+err.Error()); return }
	_, err = b.addOfflineDownload(ctx, []string{magnet})
	if err != nil { b.editMessage(ctx, message.Chat.ID, status, "❌ "+err.Error()); return }
	b.editMessage(ctx, message.Chat.ID, status, "✅ 已添加到下载队列")
	if b.notifyEnabled(message.From.ID, true) { b.sendMessage(ctx, message.Chat.ID, "✅ 任务完成："+entry) }
}

func (b *Bot) handleBatch(ctx context.Context, message *TelegramMessage, entries []string) {
	status := b.sendMessage(ctx, message.Chat.ID, fmt.Sprintf("🔄 处理 %d 个任务", len(entries)))
	seen := make(map[string]bool)
	results := make([]string, 0, len(entries))
	success := 0
	for index, entry := range entries {
		magnet, err := b.resolveEntry(ctx, entry)
		if err != nil { results = append(results, fmt.Sprintf("%d. %s: %s", index+1, short(entry), err)); continue }
		if seen[magnet] { results = append(results, fmt.Sprintf("%d. %s: 重复", index+1, short(entry))); continue }
		seen[magnet] = true
		if _, err := b.addOfflineDownload(ctx, []string{magnet}); err != nil { results = append(results, fmt.Sprintf("%d. %s: %s", index+1, short(entry), err)); continue }
		success++
		results = append(results, fmt.Sprintf("%d. %s: ✅成功", index+1, short(entry)))
		if b.notifyEnabled(message.From.ID, true) { b.sendMessage(ctx, message.Chat.ID, "✅ 任务完成："+entry) }
	}
	b.editMessage(ctx, message.Chat.ID, status, fmt.Sprintf("✅ 完成（成功: %d/%d）\n%s", success, len(entries), strings.Join(limitStrings(results, 10), "\n")))
}

func (b *Bot) resolveEntry(ctx context.Context, entry string) (string, error) {
	if strings.HasPrefix(entry, "magnet:?") || strings.HasPrefix(entry, "ed2k://") { return entry, nil }
	if !fanhaoPattern.MatchString(entry) { return "", errors.New("无法识别的格式") }
	return b.searchMagnet(ctx, entry)
}

func (b *Bot) searchMagnet(ctx context.Context, code string) (string, error) {
	entries := make(chan []SearchEntry, len(b.cfg.SearchURLs)+1)
	for _, base := range b.cfg.SearchURLs { go func(base string) { entries <- b.fetchSearchAPI(ctx, base, code) }(base) }
	go func() { entries <- b.scrapeSukebei(ctx, code) }()
	all := make([]SearchEntry, 0)
	for range len(b.cfg.SearchURLs)+1 { all = append(all, <-entries...) }
	unique := make(map[string]SearchEntry)
	for _, entry := range all { if isFanhaoMatch(code, entry.Name) { unique[entry.Magnet] = entry } }
	if len(unique) == 0 { return "", fmt.Errorf("未找到番号 %s 的相关资源", code) }
	all = all[:0]
	for _, entry := range unique { all = append(all, entry) }
	sort.SliceStable(all, func(i, j int) bool { return b.entryScore(all[i]) > b.entryScore(all[j]) })
	return all[0].Magnet, nil
}

func (b *Bot) fetchSearchAPI(ctx context.Context, base, code string) []SearchEntry {
	var body struct { Status string `json:"status"`; Data []interface{} `json:"data"` }
	if err := b.requestJSON(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/"+url.PathEscape(code), nil, &body, 20*time.Second); err != nil || body.Status != "succeed" { return nil }
	entries := make([]SearchEntry, 0)
	for _, raw := range body.Data { if entry, ok := parseSearchValue(raw); ok && isFanhaoMatch(code, entry.Name) { entries = append(entries, entry) } }
	return entries
}

func (b *Bot) scrapeSukebei(ctx context.Context, code string) []SearchEntry {
	request, _ := http.NewRequestWithContext(ctx, http.MethodGet, "https://sukebei.nyaa.si/?f=0&c=0_0&q="+url.QueryEscape(code), nil)
	request.Header.Set("User-Agent", "tgbot-go/1.0")
	response, err := b.client.Do(request)
	if err != nil { return nil }
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body); if err != nil { return nil }
	rows := regexp.MustCompile(`(?is)<tr[^>]*class=["'][^"']*(?:default|success)[^"']*["'][^>]*>(.*?)</tr>`).FindAllSubmatch(data, -1)
	result := make([]SearchEntry, 0, len(rows))
	for _, row := range rows {
		content := string(row[1])
		name := stripTags(firstMatch(content, `(?is)<td[^>]*>\s*<a[^>]*>(.*?)</a>`))
		magnet := firstMatch(content, `(?is)href=["'](magnet:\?[^"']+)`)
		if magnet == "" || !isFanhaoMatch(code, html.UnescapeString(name)) { continue }
		result = append(result, SearchEntry{Magnet: html.UnescapeString(magnet), Name: strings.TrimSpace(html.UnescapeString(name)), Source: "sukebei"})
	}
	return result
}

func parseSearchValue(raw interface{}) (SearchEntry, bool) {
	s, ok := raw.(string); if !ok { return SearchEntry{}, false }
	parts := regexp.MustCompile(`^\s*\[\s*["'](magnet:\?[^"']+)["']\s*,\s*["'](.*?)["']\s*,\s*["'](.*?)["']\s*,\s*["'](.*?)["']\s*\]\s*$`).FindStringSubmatch(s)
	if len(parts) != 5 { return SearchEntry{}, false }
	return SearchEntry{Magnet: parts[1], Name: parts[2], Size: parseSize(parts[3]), Date: parseDate(parts[4]), Source: "api"}, true
}

func (b *Bot) entryScore(entry SearchEntry) int64 {
	score := entry.Size + entry.Date.Unix() + int64(entry.Seeders)*1000000000
	name := strings.ToLower(entry.Name)
	for _, keyword := range b.cfg.PreferredWords { if strings.Contains(name, keyword) { score += 1000000000000000; break } }
	return score
}

func (b *Bot) addOfflineDownload(ctx context.Context, links []string) (bool, error) {
	_, err := b.alist(ctx, "/api/fs/add_offline_download", map[string]interface{}{
		"path": b.currentDirectory(), "urls": links, "tool": "storage", "delete_policy": "delete_on_upload_succeed",
	})
	return err == nil, err
}

func (b *Bot) alist(ctx context.Context, endpoint string, payload interface{}) (alistResponse, error) {
	var result alistResponse
	err := b.requestJSON(ctx, http.MethodPost, b.cfg.AlistBaseURL+endpoint, payload, &result, 30*time.Second)
	if err != nil { return result, err }
	if result.Code != 200 { return result, fmt.Errorf("alist: %s", result.Message) }
	return result, nil
}

func (b *Bot) requestJSON(ctx context.Context, method, endpoint string, payload interface{}, target interface{}, timeout time.Duration) error {
	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload); if err != nil { return err }
		body = bytes.NewReader(data)
	}
	requestCtx, cancel := context.WithTimeout(ctx, timeout); defer cancel()
	request, err := http.NewRequestWithContext(requestCtx, method, endpoint, body); if err != nil { return err }
	request.Header.Set("Accept", "application/json")
	request.Header.Set("User-Agent", "tgbot-go/1.0")
	if payload != nil { request.Header.Set("Content-Type", "application/json") }
	if strings.Contains(endpoint, b.cfg.AlistBaseURL) { request.Header.Set("Authorization", b.cfg.AlistToken) }
	response, err := b.client.Do(request); if err != nil { return err }
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return &httpStatusError{StatusCode: response.StatusCode, Message: strings.TrimSpace(string(data))}
	}
	return json.NewDecoder(response.Body).Decode(target)
}

func (b *Bot) telegramRequest(ctx context.Context, method string, payload interface{}, target interface{}, timeout time.Duration) error {
	return b.requestJSON(ctx, http.MethodPost, "https://api.telegram.org/bot"+b.cfg.TelegramToken+"/"+method, payload, target, timeout)
}

func (b *Bot) sendMessage(ctx context.Context, chatID int64, text string) int64 {
	var result struct { OK bool `json:"ok"`; Result TelegramMessage `json:"result"` }
	if err := b.telegramRequest(ctx, "sendMessage", map[string]interface{}{"chat_id": chatID, "text": text}, &result, 20*time.Second); err != nil { log.Printf("send message: %v", err); return 0 }
	return result.Result.MessageID
}

func (b *Bot) sendMessageWithMarkup(ctx context.Context, chatID int64, text string, markup InlineKeyboardMarkup) int64 {
	var result struct { OK bool `json:"ok"`; Result TelegramMessage `json:"result"` }
	if err := b.telegramRequest(ctx, "sendMessage", map[string]interface{}{"chat_id": chatID, "text": text, "reply_markup": markup}, &result, 20*time.Second); err != nil { log.Printf("send message: %v", err); return 0 }
	return result.Result.MessageID
}

func (b *Bot) editMessage(ctx context.Context, chatID, messageID int64, text string) {
	if messageID == 0 { return }
	var result struct { OK bool `json:"ok"` }
	if err := b.telegramRequest(ctx, "editMessageText", map[string]interface{}{"chat_id": chatID, "message_id": messageID, "text": text}, &result, 20*time.Second); err != nil { log.Printf("edit message: %v", err) }
}

func (b *Bot) answerCallback(ctx context.Context, id string) {
	var result struct { OK bool `json:"ok"` }
	_ = b.telegramRequest(ctx, "answerCallbackQuery", map[string]interface{}{"callback_query_id": id}, &result, 20*time.Second)
}

func (b *Bot) handleCallback(ctx context.Context, query *CallbackQuery) {
	if query == nil || query.Message == nil { return }
	if !b.cfg.AllowedUsers[query.From.ID] { b.answerCallback(ctx, query.ID); return }
	b.answerCallback(ctx, query.ID)
	if query.Data == "help" { b.help(ctx, query.Message); return }
	if !strings.HasPrefix(query.Data, "setdir_") { return }
	index, err := strconv.Atoi(strings.TrimPrefix(query.Data, "setdir_"))
	if err != nil || index < 0 || index >= len(b.cfg.OfflineDirs) { b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, "❌ 无效目录"); return }
	b.stateMu.Lock()
	b.state.CurrentIndex = index
	b.state.CurrentDirectory = b.cfg.OfflineDirs[index]
	b.stateMu.Unlock()
	if err := b.saveState(); err != nil { log.Printf("save state: %v", err) }
	b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, fmt.Sprintf("✅ 已切换到目录 %d: %s", index+1, b.cfg.OfflineDirs[index]))
}

func (b *Bot) loadState() error {
	b.state = State{NotifyTask: map[int64]bool{}, NotifyClean: map[int64]bool{}}
	data, err := os.ReadFile(b.cfg.StateFile)
	if errors.Is(err, os.ErrNotExist) { b.state.CurrentDirectory = b.cfg.OfflineDirs[0]; return nil }
	if err != nil { return err }
	if err := json.Unmarshal(data, &b.state); err != nil { return err }
	if b.state.NotifyTask == nil { b.state.NotifyTask = map[int64]bool{} }
	if b.state.NotifyClean == nil { b.state.NotifyClean = map[int64]bool{} }
	if b.state.CurrentDirectory == "" || !contains(b.cfg.OfflineDirs, b.state.CurrentDirectory) { b.state.CurrentIndex = 0; b.state.CurrentDirectory = b.cfg.OfflineDirs[0] }
	return nil
}

func (b *Bot) saveState() error {
	b.stateMu.Lock(); data, err := json.MarshalIndent(b.state, "", "  "); b.stateMu.Unlock()
	if err != nil { return err }
	if err := os.MkdirAll(path.Dir(b.cfg.StateFile), 0755); err != nil { return err }
	temporary := b.cfg.StateFile + ".tmp"
	if err := os.WriteFile(temporary, data, 0600); err != nil { return err }
	return os.Rename(temporary, b.cfg.StateFile)
}

func (b *Bot) currentDirectory() string {
	b.stateMu.Lock(); defer b.stateMu.Unlock()
	if b.state.CurrentDirectory == "" { return b.cfg.OfflineDirs[0] }
	return b.state.CurrentDirectory
}

func (b *Bot) notifyEnabled(userID int64, task bool) bool {
	b.stateMu.Lock(); defer b.stateMu.Unlock()
	if task { return b.state.NotifyTask[userID] }
	if value, ok := b.state.NotifyClean[userID]; ok { return value }
	return true
}

func (b *Bot) setDir(ctx context.Context, message *TelegramMessage) {
	current := b.currentDirectory()
	keyboard := make([][]InlineKeyboardButton, 0, len(b.cfg.OfflineDirs))
	for index, directory := range b.cfg.OfflineDirs {
		label := directory
		if directory == current { label = "👉 " + label }
		keyboard = append(keyboard, []InlineKeyboardButton{{Text: label, CallbackData: fmt.Sprintf("setdir_%d", index)}})
	}
	b.sendMessageWithMarkup(ctx, message.Chat.ID, "点击下方按钮选择目录：", InlineKeyboardMarkup{InlineKeyboard: keyboard})
}

func (b *Bot) refresh(ctx context.Context, message *TelegramMessage) {
	_, err := b.alist(ctx, "/api/fs/list", map[string]interface{}{"path": b.currentDirectory(), "page": 1, "per_page": 0, "refresh": true})
	if err != nil { b.sendMessage(ctx, message.Chat.ID, "❌ 刷新失败: "+err.Error()); return }
	b.sendMessage(ctx, message.Chat.ID, "✅ Alist 刷新成功！")
}

func (b *Bot) notify(ctx context.Context, message *TelegramMessage, args []string) {
	if len(args) != 2 || (args[0] != "任务" && args[0] != "清理") || (args[1] != "on" && args[1] != "off") {
		b.sendMessage(ctx, message.Chat.ID, "⚠️ 格式：/notify <任务|清理> <on|off>")
		return
	}
	enabled := args[1] == "on"
	b.stateMu.Lock()
	if args[0] == "任务" { b.state.NotifyTask[message.From.ID] = enabled } else { b.state.NotifyClean[message.From.ID] = enabled }
	b.stateMu.Unlock()
	if err := b.saveState(); err != nil { log.Printf("save state: %v", err) }
	state := "关闭"; if enabled { state = "开启" }
	b.sendMessage(ctx, message.Chat.ID, fmt.Sprintf("✅ %s通知已%s", args[0], state))
}

func (b *Bot) reloadConfig(ctx context.Context, message *TelegramMessage) {
	newConfig, err := loadConfig()
	if err != nil { b.sendMessage(ctx, message.Chat.ID, "❌ 配置重载失败: "+err.Error()); return }
	newClient, err := newHTTPClient(newConfig.ProxyURL)
	if err != nil { b.sendMessage(ctx, message.Chat.ID, "❌ 代理配置无效: "+err.Error()); return }
	b.cfg = newConfig
	b.client = newClient
	b.stateMu.Lock()
	if b.state.CurrentDirectory == "" || !contains(newConfig.OfflineDirs, b.state.CurrentDirectory) {
		b.state.CurrentIndex = 0
		b.state.CurrentDirectory = newConfig.OfflineDirs[0]
	}
	b.stateMu.Unlock()
	_ = b.saveState()
	b.sendMessage(ctx, message.Chat.ID, fmt.Sprintf("✅ 配置已热重载\n• 当前目录：%s\n• 清理间隔：%s\n• 大小阈值：%d MB\n• 下载目录数：%d\n• 搜索API数：%d\n• 允许用户数：%d", b.currentDirectory(), newConfig.CleanInterval, newConfig.SizeThreshold/1024/1024, len(newConfig.OfflineDirs), len(newConfig.SearchURLs), len(newConfig.AllowedUsers)))
}

func (b *Bot) classify(ctx context.Context, message *TelegramMessage, args []string) {
	classifyType, target := parseClassifyArgs(args)
	base := b.currentDirectory()
	targets := b.resolveTargets(ctx, base, target)
	if len(targets) == 0 { b.sendMessage(ctx, message.Chat.ID, "❌ 未找到有效目标目录"); return }
	status := b.sendMessage(ctx, message.Chat.ID, fmt.Sprintf("🔄 开始批量整理【%s】类型，共 %d 个目录...", classifyType, len(targets)))
	moved, failed := 0, 0
	for _, targetDirectory := range targets {
		movedCount, failedCount := b.classifyDirectory(ctx, targetDirectory, classifyType)
		moved += movedCount; failed += failedCount
	}
	b.editMessage(ctx, message.Chat.ID, status, fmt.Sprintf("✅ 整理完成！\n• 处理目录: %d\n• 移动项目: %d\n• 失败项目: %d", len(targets), moved, failed))
}

func parseClassifyArgs(args []string) (string, string) {
	typeName, target := "all", ""
	if len(args) == 0 { return typeName, target }
	first := strings.ToLower(args[0])
	if first == "all" || first == "folder" || first == "file" {
		typeName = first
		if len(args) > 1 { target = args[1] }
		return typeName, target
	}
	target = args[0]
	if len(args) > 1 && (strings.ToLower(args[1]) == "all" || strings.ToLower(args[1]) == "folder" || strings.ToLower(args[1]) == "file") { typeName = strings.ToLower(args[1]) }
	return typeName, target
}

func (b *Bot) resolveTargets(ctx context.Context, base, target string) []string {
	if target == "" || target == "/" { return []string{base} }
	if fanhaoPattern.MatchString(target) {
		matches := b.findDownloadDirectories(ctx, base, target)
		return matches
	}
	targetPath := normalizePath(base + "/" + target)
	if b.directoryExists(ctx, targetPath) { return []string{targetPath} }
	return nil
}

func (b *Bot) findDownloadDirectories(ctx context.Context, parent, code string) []string {
	content := b.listDirectory(ctx, parent)
	target := normalizeCode(code)
	result := make([]string, 0)
	for _, item := range content {
		if item.IsDir && strings.HasPrefix(normalizeCode(item.Name), target) { result = append(result, normalizePath(parent+"/"+item.Name)) }
	}
	return result
}

func (b *Bot) classifyDirectory(ctx context.Context, target, classifyType string) (int, int) {
	topDirs := []string{"JAV"}
	for _, category := range b.cfg.CustomCategories { topDirs = append(topDirs, category.Name) }
	topDirs = append(topDirs, "其他")
	for _, directory := range topDirs {
		if _, err := b.ensureDirectory(ctx, target+"/"+directory); err != nil { log.Printf("prepare classification directory: %v", err) }
	}
	if _, err := b.ensureDirectory(ctx, target+"/JAV/FC2"); err != nil { log.Printf("prepare FC2 directory: %v", err) }
	content := b.listDirectory(ctx, target)
	moves := make(map[string][]string)
	for _, item := range content {
		if !shouldProcessItem(item, classifyType, topDirs, target, b.cfg.SystemFolders) { continue }
		destination := normalizePath(target + "/" + destinationSubdir(item.Name, b.cfg.CustomCategories))
		moves[destination] = append(moves[destination], item.Name)
	}
	moved, failed := 0, 0
	for destination, names := range moves {
		for start := 0; start < len(names); start += 5 {
			end := start + 5; if end > len(names) { end = len(names) }
			if err := b.moveItems(ctx, target, names[start:end], destination); err != nil { failed += end-start } else { moved += end-start }
		}
	}
	return moved, failed
}

func shouldProcessItem(item alistItem, classifyType string, topDirs []string, target string, systemFolders []string) bool {
	if classifyType == "folder" && !item.IsDir { return false }
	if classifyType == "file" && item.IsDir { return false }
	if contains(topDirs, item.Name) { return false }
	return !isSystemPath(target+"/"+item.Name, systemFolders)
}

func destinationSubdir(name string, categories []Category) string {
	clean := regexp.MustCompile(`(?i)(?:Chinese|English)\s*subtitles?|中文字幕|英文字幕|高清|无码|無碼|无水印|無水印|破解版|破解|\[.*?\]|【.*?】|\(.*?\)`).ReplaceAllString(name, "")
	if regexp.MustCompile(`(?i)FC2[-_ ]?(?:PPV[-_ ]?)?\d{6,}`).MatchString(clean) { return "JAV/FC2" }
	match := regexp.MustCompile(`(?i)([A-Z]{2,5})[-_ ]?\d{2,7}`).FindStringSubmatch(clean)
	if len(match) > 1 { return "JAV/" + strings.ToUpper(match[1][:1]) }
	match = regexp.MustCompile(`(?i)([A-Z]{4,})\d{3,}`).FindStringSubmatch(clean)
	if len(match) > 1 { return "JAV/" + strings.ToUpper(match[1][:1]) }
	lower := strings.ToLower(name)
	for _, category := range categories {
		for _, keyword := range category.Keywords { if strings.Contains(lower, keyword) { return category.Name } }
	}
	return "其他"
}

func (b *Bot) listDirectory(ctx context.Context, directory string) []alistItem {
	result, err := b.alist(ctx, "/api/fs/list", map[string]interface{}{"path": normalizePath(directory), "page": 1, "per_page": 1000, "refresh": true})
	if err != nil || result.Data == nil { return nil }
	raw, ok := result.Data["content"].([]interface{}); if !ok { return nil }
	items := make([]alistItem, 0, len(raw))
	for _, value := range raw {
		data, ok := value.(map[string]interface{}); if !ok { continue }
		item := alistItem{Name: stringValue(data["name"]), IsDir: boolValue(data["is_dir"]), Size: int64Value(data["size"])}
		items = append(items, item)
	}
	return items
}

func (b *Bot) directoryExists(ctx context.Context, directory string) bool {
	directory = normalizePath(directory)
	parent := normalizePath(path.Dir(directory))
	name := path.Base(directory)
	for _, item := range b.listDirectory(ctx, parent) { if item.IsDir && item.Name == name { return true } }
	return false
}

func (b *Bot) ensureDirectory(ctx context.Context, directory string) (bool, error) {
	directory = normalizePath(directory)
	parts := strings.Split(strings.Trim(directory, "/"), "/")
	current := ""
	for _, part := range parts {
		current += "/" + part
		if b.directoryExists(ctx, current) { continue }
		if _, err := b.alist(ctx, "/api/fs/mkdir", map[string]string{"path": current}); err != nil { return false, err }
	}
	return true, nil
}

func (b *Bot) moveItems(ctx context.Context, source string, names []string, destination string) error {
	if ok, err := b.ensureDirectory(ctx, destination); !ok { return err }
	_, err := b.alist(ctx, "/api/fs/move", map[string]interface{}{"src_dir": normalizePath(source), "dst_dir": normalizePath(destination), "names": names})
	return err
}

func (b *Bot) cleanCommand(ctx context.Context, message *TelegramMessage, args []string) {
	if b.cfg.SizeThreshold <= 0 { b.sendMessage(ctx, message.Chat.ID, "✅ 清理功能已禁用"); return }
	current := b.currentDirectory()
	targets := []string{current}
	if len(args) > 0 { targets = b.resolveTargets(ctx, current, args[0]) }
	if len(targets) == 0 { b.sendMessage(ctx, message.Chat.ID, "❌ 清理失败：未找到目标目录"); return }
	status := b.sendMessage(ctx, message.Chat.ID, "🧹 清理任务（目标: "+strings.Join(targets, ", ")+")")
	deletedFiles, deletedDirs := 0, 0
	for _, target := range targets {
		deletedFiles += b.cleanSmallFiles(ctx, target)
		deletedDirs += b.cleanEmptyDirectories(ctx, target)
	}
	b.editMessage(ctx, message.Chat.ID, status, fmt.Sprintf("✅ 清理完成\n• 删除小文件: %d\n• 删除空目录: %d", deletedFiles, deletedDirs))
	if b.notifyEnabled(message.From.ID, false) { b.sendMessage(ctx, message.Chat.ID, fmt.Sprintf("✅ 清理任务完成：删除小文件 %d 个，空目录 %d 个", deletedFiles, deletedDirs)) }
}

func (b *Bot) cleanSmallFiles(ctx context.Context, root string) int {
	groups := make(map[string][]string)
	queue := []string{root}
	for len(queue) > 0 {
		directory := queue[0]; queue = queue[1:]
		for _, item := range b.listDirectory(ctx, directory) {
			if item.IsDir { queue = append(queue, normalizePath(directory+"/"+item.Name)); continue }
			if item.Size < b.cfg.SizeThreshold && !hasSuffix(item.Name, b.cfg.ExcludeSuffixes) && !isSystemPath(directory, b.cfg.SystemFolders) { groups[directory] = append(groups[directory], item.Name) }
		}
	}
	items := make([][2]string, 0)
	for directory, names := range groups { for _, name := range names { items = append(items, [2]string{directory, name}) } }
	deleted := 0
	for start := 0; start < len(items); start += b.cfg.CleanBatchSize {
		end := start + b.cfg.CleanBatchSize; if end > len(items) { end = len(items) }
		byDirectory := make(map[string][]string)
		for _, item := range items[start:end] { byDirectory[item[0]] = append(byDirectory[item[0]], item[1]) }
		for directory, names := range byDirectory {
			if b.removeWithRetry(ctx, directory, names) == nil { deleted += len(names) }
		}
		time.Sleep(b.cfg.CleanRequestWait)
	}
	return deleted
}

func (b *Bot) cleanEmptyDirectories(ctx context.Context, root string) int {
	empty := make([]string, 0)
	queue := []string{root}
	for len(queue) > 0 {
		directory := queue[0]; queue = queue[1:]
		content := b.listDirectory(ctx, directory)
		if len(content) == 0 && directory != root && !isSystemPath(directory, b.cfg.SystemFolders) { empty = append(empty, directory); continue }
		for _, item := range content { if item.IsDir { queue = append(queue, normalizePath(directory+"/"+item.Name)) } }
	}
	sort.Slice(empty, func(i, j int) bool { return len(empty[i]) > len(empty[j]) })
	deleted := 0
	for _, directory := range empty {
		parent := normalizePath(path.Dir(directory)); name := path.Base(directory)
		if b.removeWithRetry(ctx, parent, []string{name}) == nil { deleted++ }
	}
	return deleted
}

func (b *Bot) removeWithRetry(ctx context.Context, directory string, names []string) error {
	var last error
	for attempt := 0; attempt < 6; attempt++ {
		requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		_, err := b.alist(requestCtx, "/api/fs/remove", map[string]interface{}{"dir": directory, "names": names})
		cancel()
		if err == nil { return nil }
		last = err
		statusError, ok := err.(*httpStatusError)
		if !ok || statusError.StatusCode != http.StatusTooManyRequests { break }
		time.Sleep(time.Duration(1<<attempt) * time.Second)
	}
	return last
}

func (b *Bot) cleanLoop(ctx context.Context) {
	if b.cfg.SizeThreshold <= 0 { return }
	ticker := time.NewTicker(b.cfg.CleanInterval); defer ticker.Stop()
	for range ticker.C {
		b.cleanMu.Lock()
		for _, directory := range b.cfg.OfflineDirs {
			files := b.cleanSmallFiles(ctx, directory)
			dirs := b.cleanEmptyDirectories(ctx, directory)
			log.Printf("scheduled cleanup %s: files=%d dirs=%d", directory, files, dirs)
		}
		b.cleanMu.Unlock()
	}
}

func stringValue(value interface{}) string { result, _ := value.(string); return result }
func boolValue(value interface{}) bool { result, _ := value.(bool); return result }
func int64Value(value interface{}) int64 {
	switch result := value.(type) { case float64: return int64(result); case int64: return result; case json.Number: number, _ := result.Int64(); return number }
	return 0
}

func nonEmptyLines(value string) []string { result := make([]string, 0); scanner := bufio.NewScanner(strings.NewReader(value)); for scanner.Scan() { if line := strings.TrimSpace(scanner.Text()); line != "" { result = append(result, line) } }; return result }
func short(value string) string { if len(value) > 20 { return value[:20] + "..." }; return value }
func limitStrings(values []string, limit int) []string { if len(values) > limit { return values[:limit] }; return values }
func contains(values []string, target string) bool { for _, value := range values { if value == target { return true } }; return false }
func hasSuffix(value string, suffixes []string) bool { lower := strings.ToLower(value); for _, suffix := range suffixes { if strings.HasSuffix(lower, suffix) { return true } }; return false }
func normalizeCode(value string) string { value = strings.ToLower(value); result := strings.Map(func(r rune) rune { if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') { return r }; return -1 }, value); return result }
func isFanhaoMatch(code, name string) bool { return strings.Contains(normalizeCode(name), normalizeCode(code)) }
func normalizePath(value string) string { value = strings.ReplaceAll(value, "\\", "/"); clean := path.Clean("/" + strings.TrimPrefix(value, "/")); if clean == "." { return "/" }; return clean }
func isSystemPath(value string, folders []string) bool { normalized := strings.ToLower(normalizePath(value)); for _, folder := range folders { normalizedFolder := strings.ToLower(normalizePath(folder)); if normalized == normalizedFolder || strings.HasPrefix(normalized, normalizedFolder+"/") { return true } }; return false }
func parseSize(value string) int64 { match := regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb)`).FindStringSubmatch(value); if len(match) != 3 { return 0 }; multiplier := map[string]float64{"kb": 1024, "mb": 1024 * 1024, "gb": 1024 * 1024 * 1024, "tb": 1024 * 1024 * 1024 * 1024}; return int64(parseFloat(match[1]) * multiplier[strings.ToLower(match[2])]) }
func parseFloat(value string) float64 { result, _ := strconv.ParseFloat(value, 64); return result }
func parseDate(value string) time.Time { for _, layout := range []string{"2006-01-02", "2006-01-02 15:04"} { if result, err := time.Parse(layout, value); err == nil { return result } }; return time.Time{} }
func stripTags(value string) string { return regexp.MustCompile(`(?s)<[^>]+>`).ReplaceAllString(value, "") }
func firstMatch(value, expression string) string { match := regexp.MustCompile(expression).FindStringSubmatch(value); if len(match) > 1 { return match[1] }; return "" }
func splitCSV(value string) []string { result := make([]string, 0); for _, item := range strings.Split(value, ",") { if item = strings.TrimSpace(item); item != "" { result = append(result, item) } }; return result }
func lowerCSV(value string) []string { result := splitCSV(value); for index := range result { result[index] = strings.ToLower(result[index]) }; return result }
func normalizeCSVPaths(value string) []string { result := splitCSV(value); for index := range result { result[index] = normalizePath(result[index]) }; return result }
func parseUserIDs(value string) map[int64]bool { result := map[int64]bool{}; for _, item := range splitCSV(value) { if id, err := strconv.ParseInt(item, 10, 64); err == nil { result[id] = true } }; return result }
func parseCategories(value string) []Category { result := make([]Category, 0); for _, item := range strings.Split(value, ";") { parts := strings.SplitN(item, ":", 2); if len(parts) != 2 { continue }; name := strings.TrimSpace(parts[0]); keywords := lowerCSV(parts[1]); if name != "" && len(keywords) > 0 { result = append(result, Category{Name: name, Keywords: keywords}) } }; return result }
func envString(name, fallback string) string { if value := strings.TrimSpace(os.Getenv(name)); value != "" { return value }; return fallback }
func envInt(name string, fallback int) int { if value, err := strconv.Atoi(strings.TrimSpace(os.Getenv(name))); err == nil { return value }; return fallback }
