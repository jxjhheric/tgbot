package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
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
	"path/filepath"
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
	MukakuBaseURL    string
	MukakuAppID     string
	MukakuIdentity  string
	MukakuToken     string
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
	cfgMu      sync.RWMutex
	updateMu   sync.Mutex
	offset     int64
	cleanMu    sync.Mutex
	cleanWake  chan struct{}
	apiSem     chan struct{}
	mukakuMu   sync.Mutex
	mukakuByID map[string]*MukakuSession
	mukakuUser map[int64]string
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
	Leechers int
	Date     time.Time
	Source   string
}

type MukakuMovie struct {
	ID     int64  `json:"id"`
	DoubID int64  `json:"doub_id"`
	Title  string `json:"title"`
	Year   string `json:"years"`
	Class  string `json:"class"`
	Area   string `json:"production_area"`
}

type MukakuResource struct {
	ID     int64
	Name   string
	Size   string
	Date   string
	Magnet string
}

type MukakuSession struct {
	Token     string
	UserID    int64
	Query     string
	ExpiresAt time.Time
	Movies    []MukakuMovie
	Resources map[int][]MukakuResource
	Added     map[string]bool
}

func main() {
	if envFile := loadDotEnv(); envFile != "" {
		log.Printf("loaded environment file: %s", envFile)
	}
	cfg, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}
	client, err := newHTTPClient(cfg.ProxyURL)
	if err != nil {
		log.Fatal(err)
	}
	bot := &Bot{cfg: cfg, client: client, cleanWake: make(chan struct{}, 1), apiSem: make(chan struct{}, cfg.MaxConcurrent), mukakuByID: map[string]*MukakuSession{}, mukakuUser: map[int64]string{}}
	if err := bot.loadState(); err != nil {
		log.Printf("state load warning: %v", err)
	}
	log.Printf("tgbot Go service started; directories=%d proxy=%t", len(cfg.OfflineDirs), cfg.ProxyURL != "")
	go bot.cleanLoop(context.Background())
	bot.poll(context.Background())
}

func loadDotEnv() string {
	candidates := make([]string, 0, 5)
	if configured := strings.TrimSpace(os.Getenv("TGBOT_ENV_FILE")); configured != "" {
		candidates = append(candidates, configured)
	}
	if workingDirectory, err := os.Getwd(); err == nil {
		candidates = append(candidates,
			filepath.Join(workingDirectory, ".env"),
			filepath.Join(workingDirectory, ".env.go"),
			filepath.Join(workingDirectory, "tgbot.env"),
		)
	}
	if executable, err := os.Executable(); err == nil {
		executableDirectory := filepath.Dir(executable)
		candidates = append(candidates, filepath.Join(executableDirectory, ".env"))
	}
	candidates = append(candidates, "/etc/tgbot/tgbot.env")

	seen := make(map[string]bool)
	for _, candidate := range candidates {
		candidate = filepath.Clean(candidate)
		if seen[candidate] { continue }
		seen[candidate] = true
		if _, err := os.Stat(candidate); err != nil { continue }
		if err := parseDotEnvFile(candidate); err != nil {
			log.Printf("environment file warning (%s): %v", candidate, err)
			continue
		}
		return candidate
	}
	return ""
}

func parseDotEnvFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil { return err }
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") { continue }
		line = strings.TrimPrefix(line, "export ")
		separator := strings.IndexByte(line, '=')
		if separator <= 0 { continue }
		key := strings.TrimSpace(line[:separator])
		value := strings.TrimSpace(line[separator+1:])
		if !regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`).MatchString(key) {
			log.Printf("environment file warning (%s:%d): invalid variable name %q", filename, lineNumber, key)
			continue
		}
		if len(value) >= 2 && ((value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'')) {
			value = value[1 : len(value)-1]
		}
		if _, exists := os.LookupEnv(key); !exists {
			if err := os.Setenv(key, value); err != nil { return err }
		}
	}
	return scanner.Err()
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
		MukakuBaseURL:    strings.TrimRight(envString("MUKAKU_BASE_URL", "https://web5.mukaku.com"), "/"),
		MukakuAppID:      envString("MUKAKU_APP_ID", "83768d9ad4"),
		MukakuIdentity:   envString("MUKAKU_IDENTITY", "23734adac0301bccdcb107c4aa21f96c"),
		MukakuToken:      strings.TrimSpace(os.Getenv("MUKAKU_ACCESS_TOKEN")),
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
	if !isDirectLink(entry) && !fanhaoPattern.MatchString(entry) {
		b.startMukakuSearch(ctx, message, strings.TrimSpace(entry))
		return
	}
	status := b.sendMessage(ctx, message.Chat.ID, "🔍 处理："+entry)
	magnet, err := b.resolveEntry(ctx, entry)
	if err != nil { b.editMessage(ctx, message.Chat.ID, status, "❌ "+err.Error()); return }
	_, err = b.addOfflineDownload(ctx, []string{magnet})
	if err != nil { b.editMessage(ctx, message.Chat.ID, status, "❌ "+err.Error()); return }
	b.editMessage(ctx, message.Chat.ID, status, "✅ 已添加到下载队列")
	if b.notifyEnabled(message.From.ID, true) { b.sendMessage(ctx, message.Chat.ID, "✅ 任务完成："+entry) }
	time.Sleep(3 * time.Second)
	b.refreshDirectory(ctx, message.Chat.ID, b.currentDirectory())
}

func (b *Bot) handleBatch(ctx context.Context, message *TelegramMessage, entries []string) {
	status := b.sendMessage(ctx, message.Chat.ID, fmt.Sprintf("🔄 处理 %d 个任务", len(entries)))
	seen := make(map[string]bool)
	results := make([]string, 0, len(entries))
	success := 0
	for index, entry := range entries {
		if !isDirectLink(entry) && !fanhaoPattern.MatchString(entry) {
			b.startMukakuSearch(ctx, message, strings.TrimSpace(entry))
			results = append(results, fmt.Sprintf("%d. %s: 已发送 Mukaku 搜索", index+1, short(entry)))
			continue
		}
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
	if success > 0 {
		time.Sleep(3 * time.Second)
		b.refreshDirectory(ctx, message.Chat.ID, b.currentDirectory())
	}
}

func (b *Bot) resolveEntry(ctx context.Context, entry string) (string, error) {
	if isDirectLink(entry) { return entry, nil }
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
	sort.SliceStable(all, func(i, j int) bool { return b.entryBetter(all[i], all[j]) })
	return all[0].Magnet, nil
}

func (b *Bot) startMukakuSearch(ctx context.Context, message *TelegramMessage, query string) {
	if query == "" { return }
	status := b.sendMessage(ctx, message.Chat.ID, "🔍 正在搜索 Mukaku，请稍候……")
	go func() {
		movies, err := b.mukakuSearch(ctx, query)
		if err != nil {
			log.Printf("Mukaku search failed for %q: %v", query, err)
			b.editMessage(ctx, message.Chat.ID, status, "❌ 搜索失败，请稍后重试")
			return
		}
		if len(movies) == 0 {
			b.editMessage(ctx, message.Chat.ID, status, "❌ 未找到相关影视资源")
			return
		}
		token := newMukakuToken()
		session := &MukakuSession{Token: token, UserID: message.From.ID, Query: query, ExpiresAt: time.Now().Add(30 * time.Minute), Movies: movies, Resources: map[int][]MukakuResource{}, Added: map[string]bool{}}
		b.mukakuMu.Lock()
		if old := b.mukakuUser[message.From.ID]; old != "" { delete(b.mukakuByID, old) }
		b.mukakuByID[token] = session
		b.mukakuUser[message.From.ID] = token
		b.mukakuMu.Unlock()
		keyboard := make([][]InlineKeyboardButton, 0, len(movies))
		for index, movie := range movies {
			keyboard = append(keyboard, []InlineKeyboardButton{{Text: mukakuMovieLabel(movie), CallbackData: fmt.Sprintf("mk_movie_%s_%d", token, index)}})
		}
		b.editMessageWithMarkup(ctx, message.Chat.ID, status, fmt.Sprintf("🔎 Mukaku 搜索结果：%s\n请选择影视：", query), InlineKeyboardMarkup{InlineKeyboard: keyboard})
	}()
}

func newMukakuToken() string {
	data := make([]byte, 6)
	if _, err := rand.Read(data); err == nil { return fmt.Sprintf("%x", data) }
	return fmt.Sprintf("%x", time.Now().UnixNano())[:12]
}

func mukakuMovieLabel(movie MukakuMovie) string {
	label := movie.Title
	if movie.Year != "" { label += " (" + movie.Year + ")" }
	return limitText(label, 55)
}

func (b *Bot) mukakuSearch(ctx context.Context, query string) ([]MukakuMovie, error) {
	endpoint, err := url.Parse(b.cfg.MukakuBaseURL + "/prod/api/v1/getVideoList")
	if err != nil { return nil, err }
	params := endpoint.Query()
	params.Set("app_id", b.cfg.MukakuAppID)
	params.Set("identity", b.cfg.MukakuIdentity)
	params.Set("sb", query)
	params.Set("page", "1")
	params.Set("limit", "10")
	if b.cfg.MukakuToken != "" { params.Set("access_token", b.cfg.MukakuToken) }
	endpoint.RawQuery = params.Encode()
	var response struct {
		Success bool `json:"success"`
		Message string `json:"message"`
		Data struct {
			Data []MukakuMovie `json:"data"`
		} `json:"data"`
	}
	if err := b.requestJSONRetry(ctx, http.MethodGet, endpoint.String(), nil, &response, 20*time.Second, 2); err != nil { return nil, err }
	if !response.Success { return nil, errors.New(response.Message) }
	if len(response.Data.Data) > 10 { response.Data.Data = response.Data.Data[:10] }
	return response.Data.Data, nil
}

func (b *Bot) mukakuResources(ctx context.Context, movieID int64) ([]MukakuResource, error) {
	endpoint, err := url.Parse(b.cfg.MukakuBaseURL + "/prod/api/v1/getVideoDetail")
	if err != nil { return nil, err }
	params := endpoint.Query()
	params.Set("app_id", b.cfg.MukakuAppID)
	params.Set("identity", b.cfg.MukakuIdentity)
	params.Set("id", strconv.FormatInt(movieID, 10))
	if b.cfg.MukakuToken != "" { params.Set("access_token", b.cfg.MukakuToken) }
	endpoint.RawQuery = params.Encode()
	var response struct {
		Success bool `json:"success"`
		Message string `json:"message"`
		Data struct {
			AllSeeds []struct {
				ID     int64  `json:"id"`
				Zname  string `json:"zname"`
				Zsize  string `json:"zsize"`
				Date   string `json:"ezt"`
				Zlink  string `json:"zlink"`
			} `json:"all_seeds"`
		} `json:"data"`
	}
	if err := b.requestJSONRetry(ctx, http.MethodGet, endpoint.String(), nil, &response, 20*time.Second, 2); err != nil { return nil, err }
	if !response.Success { return nil, errors.New(response.Message) }
	resources := make([]MukakuResource, 0, len(response.Data.AllSeeds))
	for _, item := range response.Data.AllSeeds {
		if strings.HasPrefix(item.Zlink, "magnet:?") { resources = append(resources, MukakuResource{ID: item.ID, Name: item.Zname, Size: item.Zsize, Date: item.Date, Magnet: item.Zlink}) }
	}
	if len(resources) > 10 { resources = resources[:10] }
	return resources, nil
}

func (b *Bot) getMukakuSession(token string, userID int64) (*MukakuSession, bool) {
	b.mukakuMu.Lock()
	defer b.mukakuMu.Unlock()
	session, ok := b.mukakuByID[token]
	if !ok || session.UserID != userID || time.Now().After(session.ExpiresAt) { return nil, false }
	return session, true
}

func limitText(value string, max int) string {
	runes := []rune(value)
	if len(runes) <= max { return value }
	return string(runes[:max-1]) + "…"
}

func mukakuResourceLabel(resource MukakuResource) string {
	resolution := firstMatch(resource.Name, `(?i)\b(?:4320|2160|1080|720|576|480)p\b`)
	if resolution == "" { resolution = "未知分辨率" }
	codec := firstMatch(resource.Name, `(?i)\b(?:x265|h[ .]?265|hevc|x264|h[ .]?264|avc|av1|xvid|vp9)\b`)
	if codec == "" { codec = "未知编码" }
	codec = strings.ToUpper(strings.ReplaceAll(codec, " ", ""))
	if resource.Size == "" { resource.Size = "未知大小" }
	return limitText(fmt.Sprintf("%s · %s · %s", strings.ToUpper(resolution), codec, resource.Size), 55)
}

func isDirectLink(entry string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(entry)), "magnet:?") || strings.HasPrefix(strings.ToLower(strings.TrimSpace(entry)), "ed2k://")
}

func (b *Bot) fetchSearchAPI(ctx context.Context, base, code string) []SearchEntry {
	var body struct { Status string `json:"status"`; Data []interface{} `json:"data"` }
	if err := b.requestJSONRetry(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/"+url.PathEscape(code), nil, &body, 20*time.Second, 3); err != nil || body.Status != "succeed" { return nil }
	entries := make([]SearchEntry, 0)
	for _, raw := range body.Data { if entry, ok := parseSearchValue(raw); ok && isFanhaoMatch(code, entry.Name) { entries = append(entries, entry) } }
	return entries
}

func (b *Bot) scrapeSukebei(ctx context.Context, code string) []SearchEntry {
	var data []byte
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		request, requestErr := http.NewRequestWithContext(ctx, http.MethodGet, "https://sukebei.nyaa.si/?f=0&c=0_0&q="+url.QueryEscape(code), nil)
		if requestErr != nil { return nil }
		request.Header.Set("User-Agent", "tgbot-go/1.0")
		response, requestErr := b.do(request)
		if requestErr == nil {
			data, err = io.ReadAll(response.Body)
			response.Body.Close()
			if err == nil && response.StatusCode >= 200 && response.StatusCode < 300 { break }
		}
		if attempt < 2 { time.Sleep(time.Duration(1<<attempt) * time.Second) }
	}
	if err != nil || len(data) == 0 { return nil }
	rows := regexp.MustCompile(`(?is)<tr[^>]*class=["'][^"']*(?:default|success)[^"']*["'][^>]*>(.*?)</tr>`).FindAllSubmatch(data, -1)
	result := make([]SearchEntry, 0, len(rows))
	for _, row := range rows {
		content := string(row[1])
		cells := regexp.MustCompile(`(?is)<td[^>]*>(.*?)</td>`).FindAllStringSubmatch(content, -1)
		name := ""
		if len(cells) > 1 { name = stripTags(cells[1][1]) }
		magnet := firstMatch(content, `(?is)href=["'](magnet:\?[^"']+)`)
		if magnet == "" || !isFanhaoMatch(code, html.UnescapeString(name)) { continue }
		entry := SearchEntry{Magnet: html.UnescapeString(magnet), Name: strings.TrimSpace(html.UnescapeString(name)), Source: "sukebei"}
		if len(cells) > 2 { entry.Size = parseSize(stripTags(cells[2][1])) }
		if len(cells) > 4 { entry.Date = parseDate(stripTags(cells[4][1])) }
		centerCells := regexp.MustCompile(`(?is)<td[^>]*class=["'][^"']*\btext-center\b[^"']*["'][^>]*>(.*?)</td>`).FindAllStringSubmatch(content, -1)
		if len(centerCells) >= 5 {
			entry.Seeders = parseIntValue(stripTags(centerCells[2][1]))
			entry.Leechers = parseIntValue(stripTags(centerCells[3][1]))
		}
		result = append(result, entry)
	}
	return result
}

func parseSearchValue(raw interface{}) (SearchEntry, bool) {
	if values, ok := raw.([]interface{}); ok && len(values) >= 4 {
		magnet, _ := values[0].(string)
		name, _ := values[1].(string)
		size, _ := values[2].(string)
		date, _ := values[3].(string)
		if strings.HasPrefix(magnet, "magnet:?") { return SearchEntry{Magnet: magnet, Name: name, Size: parseSize(size), Date: parseDate(date), Source: "api"}, true }
	}
	s, ok := raw.(string); if !ok { return SearchEntry{}, false }
	parts := regexp.MustCompile(`^\s*\[\s*["'](magnet:\?[^"']+)["']\s*,\s*["'](.*?)["']\s*,\s*["'](.*?)["']\s*,\s*["'](.*?)["']\s*\]\s*$`).FindStringSubmatch(s)
	if len(parts) != 5 { return SearchEntry{}, false }
	return SearchEntry{Magnet: parts[1], Name: parts[2], Size: parseSize(parts[3]), Date: parseDate(parts[4]), Source: "api"}, true
}

func (b *Bot) entryBetter(left, right SearchEntry) bool {
	leftPreferred := b.hasPreferredWord(left.Name)
	rightPreferred := b.hasPreferredWord(right.Name)
	if leftPreferred != rightPreferred { return leftPreferred }
	if left.Seeders != right.Seeders { return left.Seeders > right.Seeders }
	if !left.Date.Equal(right.Date) {
		if left.Date.IsZero() { return false }
		if right.Date.IsZero() { return true }
		return left.Date.After(right.Date)
	}
	return left.Size > right.Size
}

func (b *Bot) hasPreferredWord(name string) bool {
	lowerName := strings.ToLower(name)
	for _, keyword := range b.cfg.PreferredWords { if strings.Contains(lowerName, keyword) { return true } }
	return false
}

func (b *Bot) addOfflineDownload(ctx context.Context, links []string) (bool, error) {
	_, err := b.alist(ctx, "/api/fs/add_offline_download", map[string]interface{}{
		"path": b.currentDirectory(), "urls": links, "tool": "storage", "delete_policy": "delete_on_upload_succeed",
	})
	return err == nil, err
}

func (b *Bot) alist(ctx context.Context, endpoint string, payload interface{}) (alistResponse, error) {
	var result alistResponse
	err := b.requestJSONRetry(ctx, http.MethodPost, b.cfg.AlistBaseURL+endpoint, payload, &result, 30*time.Second, 3)
	if err != nil { return result, err }
	if result.Code != 200 {
		if result.Code == http.StatusTooManyRequests { return result, &httpStatusError{StatusCode: result.Code, Message: result.Message} }
		return result, fmt.Errorf("alist: %s", result.Message)
	}
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
	if b.cfg.MukakuToken != "" && strings.Contains(endpoint, b.cfg.MukakuBaseURL) { request.Header.Set("Authorization", "Bearer "+b.cfg.MukakuToken) }
	response, err := b.do(request); if err != nil { return err }
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return &httpStatusError{StatusCode: response.StatusCode, Message: strings.TrimSpace(string(data))}
	}
	return json.NewDecoder(response.Body).Decode(target)
}

func (b *Bot) requestJSONRetry(ctx context.Context, method, endpoint string, payload interface{}, target interface{}, timeout time.Duration, attempts int) error {
	var last error
	for attempt := 0; attempt < attempts; attempt++ {
		if err := b.requestJSON(ctx, method, endpoint, payload, target, timeout); err == nil { return nil } else { last = err }
		if attempt+1 < attempts { time.Sleep(time.Duration(1<<attempt) * time.Second) }
	}
	return last
}

func (b *Bot) do(request *http.Request) (*http.Response, error) {
	if b.apiSem != nil { b.apiSem <- struct{}{}; defer func() { <-b.apiSem }() }
	return b.client.Do(request)
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
	b.editMessageWithMarkup(ctx, chatID, messageID, text, InlineKeyboardMarkup{})
}

func (b *Bot) editMessageWithMarkup(ctx context.Context, chatID, messageID int64, text string, markup InlineKeyboardMarkup) {
	if messageID == 0 { return }
	var result struct { OK bool `json:"ok"` }
	payload := map[string]interface{}{"chat_id": chatID, "message_id": messageID, "text": text}
	if len(markup.InlineKeyboard) > 0 { payload["reply_markup"] = markup }
	if err := b.telegramRequest(ctx, "editMessageText", payload, &result, 20*time.Second); err != nil { log.Printf("edit message: %v", err) }
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
	if strings.HasPrefix(query.Data, "mk_movie_") { b.handleMukakuMovie(ctx, query); return }
	if strings.HasPrefix(query.Data, "mk_resource_") { b.handleMukakuResource(ctx, query); return }
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
	b.refreshDirectory(ctx, message.Chat.ID, b.currentDirectory())
}

func (b *Bot) refreshDirectory(ctx context.Context, chatID int64, directory string) bool {
	status := b.sendMessage(ctx, chatID, "🔄 正在刷新 Alist 文件列表...")
	_, err := b.alist(ctx, "/api/fs/list", map[string]interface{}{"path": directory, "page": 1, "per_page": 0, "refresh": true})
	if err != nil {
		b.editMessage(ctx, chatID, status, "❌ 刷新失败: "+err.Error())
		return false
	}
	b.editMessage(ctx, chatID, status, "✅ Alist 刷新成功！")
	return true
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
	b.cfgMu.Lock()
	b.cfg = newConfig
	b.client = newClient
	b.apiSem = make(chan struct{}, newConfig.MaxConcurrent)
	b.cfgMu.Unlock()
	b.stateMu.Lock()
	if b.state.CurrentDirectory == "" || !contains(newConfig.OfflineDirs, b.state.CurrentDirectory) {
		b.state.CurrentIndex = 0
		b.state.CurrentDirectory = newConfig.OfflineDirs[0]
	}
	b.stateMu.Unlock()
	_ = b.saveState()
	select { case b.cleanWake <- struct{}{}: default: }
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
		b.editMessage(ctx, message.Chat.ID, status, fmt.Sprintf("🔄 批量整理中...\n• 已处理目录: %d/%d\n• 已移动: %d\n• 失败: %d", movedCount, len(targets), moved, failed))
	}
	b.editMessage(ctx, message.Chat.ID, status, fmt.Sprintf("✅ 整理完成！\n• 处理目录: %d\n• 移动项目: %d\n• 失败项目: %d", len(targets), moved, failed))
	b.refreshDirectory(ctx, message.Chat.ID, base)
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
		b.editMessage(ctx, message.Chat.ID, status, fmt.Sprintf("🧹 清理中...\n• 已处理目录: %s\n• 删除小文件: %d\n• 删除空目录: %d", target, deletedFiles, deletedDirs))
		deletedDirs += b.cleanEmptyDirectories(ctx, target)
		b.editMessage(ctx, message.Chat.ID, status, fmt.Sprintf("🧹 清理中...\n• 已处理目录: %s\n• 删除小文件: %d\n• 删除空目录: %d", target, deletedFiles, deletedDirs))
	}
	renameCount := b.renameOfflineDirectories(ctx)
	b.editMessage(ctx, message.Chat.ID, status, fmt.Sprintf("✅ 清理完成\n• 删除小文件: %d\n• 删除空目录: %d\n• 重命名文件夹: %d", deletedFiles, deletedDirs, renameCount))
	b.refreshDirectory(ctx, message.Chat.ID, current)
}

func (b *Bot) handleMukakuMovie(ctx context.Context, query *CallbackQuery) {
	parts := strings.Split(query.Data, "_")
	if len(parts) != 4 { b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, "❌ 搜索结果已失效"); return }
	index, err := strconv.Atoi(parts[3])
	session, ok := b.getMukakuSession(parts[2], query.From.ID)
	if err != nil || !ok || index < 0 || index >= len(session.Movies) { b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, "❌ 搜索结果已失效"); return }
	movieID := session.Movies[index].DoubID
	if movieID == 0 { movieID = session.Movies[index].ID }
	resources, err := b.mukakuResources(ctx, movieID)
	if err != nil {
		log.Printf("Mukaku detail failed for %q: %v", session.Movies[index].Title, err)
		b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, "❌ 获取资源失败，请稍后重试")
		return
	}
	if len(resources) == 0 { b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, "❌ 未找到可用磁力链接"); return }
	b.mukakuMu.Lock()
	session.Resources[index] = resources
	b.mukakuMu.Unlock()
	keyboard := make([][]InlineKeyboardButton, 0, len(resources))
	for resourceIndex, resource := range resources {
		keyboard = append(keyboard, []InlineKeyboardButton{{Text: mukakuResourceLabel(resource), CallbackData: fmt.Sprintf("mk_resource_%s_%d_%d", session.Token, index, resourceIndex)}})
	}
	b.editMessageWithMarkup(ctx, query.Message.Chat.ID, query.Message.MessageID, fmt.Sprintf("🎬 %s\n请选择资源：", mukakuMovieLabel(session.Movies[index])), InlineKeyboardMarkup{InlineKeyboard: keyboard})
}

func (b *Bot) handleMukakuResource(ctx context.Context, query *CallbackQuery) {
	parts := strings.Split(query.Data, "_")
	if len(parts) != 5 { b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, "❌ 资源按钮已失效"); return }
	movieIndex, movieErr := strconv.Atoi(parts[3])
	resourceIndex, resourceErr := strconv.Atoi(parts[4])
	session, ok := b.getMukakuSession(parts[2], query.From.ID)
	if movieErr != nil || resourceErr != nil || !ok { b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, "❌ 资源按钮已失效"); return }
	b.mukakuMu.Lock()
	resources := session.Resources[movieIndex]
	if movieIndex < 0 || movieIndex >= len(session.Movies) || resourceIndex < 0 || resourceIndex >= len(resources) { b.mukakuMu.Unlock(); b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, "❌ 资源按钮已失效"); return }
	resource := resources[resourceIndex]
	if session.Added[resource.Magnet] { b.mukakuMu.Unlock(); return }
	b.mukakuMu.Unlock()
	if _, err := b.addOfflineDownload(ctx, []string{resource.Magnet}); err != nil {
		log.Printf("Mukaku add failed for %q: %v", resource.Name, err)
		b.sendMessage(ctx, query.Message.Chat.ID, "❌ 添加失败，请稍后重试")
		return
	}
	b.mukakuMu.Lock()
	session.Added[resource.Magnet] = true
	b.mukakuMu.Unlock()
	b.editMessage(ctx, query.Message.Chat.ID, query.Message.MessageID, "✅ 已添加到下载队列："+mukakuResourceLabel(resource))
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

func (b *Bot) renameOfflineDirectories(ctx context.Context) int {
	b.cfgMu.RLock()
	offlineDirs := append([]string(nil), b.cfg.OfflineDirs...)
	b.cfgMu.RUnlock()
	renamed := 0
	for _, directory := range offlineDirs {
		directory = normalizePath(directory)
		for _, item := range b.listDirectory(ctx, directory) {
			if !item.IsDir { continue }
			newName, ok := trimFolderPrefix(item.Name)
			if !ok { continue }
			oldPath := normalizePath(directory + "/" + item.Name)
			if b.nameExists(ctx, directory, newName) {
				log.Printf("skip folder rename due to name conflict: %s -> %s", oldPath, newName)
				continue
			}
			if err := b.renameDirectory(ctx, oldPath, newName); err != nil {
				log.Printf("rename folder failed: %s -> %s: %v", oldPath, newName, err)
				continue
			}
			renamed++
		}
	}
	return renamed
}

func (b *Bot) renameDirectory(ctx context.Context, directory, name string) error {
	_, err := b.alist(ctx, "/api/fs/rename", map[string]string{"path": normalizePath(directory), "name": name})
	return err
}

func (b *Bot) nameExists(ctx context.Context, directory, name string) bool {
	for _, item := range b.listDirectory(ctx, directory) { if item.Name == name { return true } }
	return false
}

func (b *Bot) cleanLoop(ctx context.Context) {
	var ticker *time.Ticker
	var tick <-chan time.Time
	resetTicker := func() {
		if ticker != nil { ticker.Stop() }
		b.cfgMu.RLock()
		interval := b.cfg.CleanInterval
		enabled := b.cfg.SizeThreshold > 0
		b.cfgMu.RUnlock()
		if enabled { ticker = time.NewTicker(interval); tick = ticker.C } else { ticker = nil; tick = nil }
	}
	resetTicker()
	for {
		select {
		case <-ctx.Done():
			if ticker != nil { ticker.Stop() }
			return
		case <-b.cleanWake:
			resetTicker()
		case <-tick:
			b.runScheduledCleanup(ctx)
		}
	}
}

func (b *Bot) runScheduledCleanup(ctx context.Context) {
	b.cleanMu.Lock()
	defer b.cleanMu.Unlock()
	b.cfgMu.RLock()
	directories := append([]string(nil), b.cfg.OfflineDirs...)
	users := make([]int64, 0, len(b.cfg.AllowedUsers))
	for userID := range b.cfg.AllowedUsers { users = append(users, userID) }
	b.cfgMu.RUnlock()
	directories = topLevelDirectories(directories)
	start := fmt.Sprintf("🔄 自动清理任务启动\n• 时间: %s", beijingNow().Format("2006-01-02 15:04:05"))
	for _, userID := range users {
		if b.notifyEnabled(userID, false) { b.sendMessage(ctx, userID, start); time.Sleep(300 * time.Millisecond) }
	}
	results := make([]string, 0, len(directories)+2)
	for _, directory := range directories {
		files := b.cleanSmallFiles(ctx, directory)
		dirs := b.cleanEmptyDirectories(ctx, directory)
		results = append(results, fmt.Sprintf("📂 目录 %s:\n• 小文件: 成功清理 %d 个\n• 空目录: 成功删除 %d 个", directory, files, dirs))
		log.Printf("scheduled cleanup %s: files=%d dirs=%d", directory, files, dirs)
		time.Sleep(time.Second)
	}
	renameCount := b.renameOfflineDirectories(ctx)
	results = append(results, fmt.Sprintf("• 文件夹改名: 成功 %d 个", renameCount))
	log.Printf("scheduled cleanup folder rename: renamed=%d", renameCount)
	summary := "✅ 自动清理完成\n• 时间: " + beijingNow().Format("2006-01-02 15:04:05") + "\n" + strings.Join(results, "\n")
	for _, userID := range users {
		if !b.notifyEnabled(userID, false) { continue }
		for _, part := range splitMessage(summary, 4000) { b.sendMessage(ctx, userID, part); time.Sleep(300 * time.Millisecond) }
	}
}

func topLevelDirectories(directories []string) []string {
	sort.Slice(directories, func(i, j int) bool { return len(directories[i]) < len(directories[j]) })
	result := make([]string, 0, len(directories))
	for _, directory := range directories {
		directory = normalizePath(directory)
		parentFound := false
		for _, parent := range result { if directory == parent || strings.HasPrefix(directory, parent+"/") { parentFound = true; break } }
		if !parentFound { result = append(result, directory) }
	}
	return result
}

func splitMessage(value string, max int) []string {
	if len(value) <= max { return []string{value} }
	parts := make([]string, 0)
	current := ""
	for _, line := range strings.Split(value, "\n") {
		if current != "" && len(current)+len(line)+1 > max { parts = append(parts, current); current = "" }
		if current != "" { current += "\n" }
		current += line
	}
	if current != "" { parts = append(parts, current) }
	return parts
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
func parseIntValue(value string) int { result, _ := strconv.Atoi(strings.TrimSpace(value)); return result }
func trimFolderPrefix(name string) (string, bool) {
	remaining := strings.TrimSpace(name)
	removed := false
	for strings.HasPrefix(remaining, "【") {
		end := strings.Index(remaining, "】")
		if end < 0 { break }
		remaining = strings.TrimSpace(remaining[end+len("】"):])
		removed = true
	}
	if !removed || remaining == "" { return "", false }
	return remaining, true
}
func beijingNow() time.Time { return time.Now().In(time.FixedZone("CST", 8*60*60)) }
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
