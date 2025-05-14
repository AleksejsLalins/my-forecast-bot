import requests
import json
from datetime import datetime
from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import numpy as np
import os
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

BOT_TOKEN = "8038821776:AAG2LFhNwJDX6tOJJsrvu9bFOQZRijbrDx8"
CHAT_ID = "6413269307"
CRYPTO_PANIC_KEY = "f9db8855fc9a498b3d9aa73b0a69e876c41a0a47"

COINS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "XRPUSDT": "ripple",
    "ADAUSDT": "cardano",
    "TRXUSDT": "tron",
    "LINKUSDT": "chainlink",
    "XLMUSDT": "stellar",
    "PEPEUSDT": "pepe",
    "SUIUSDT": "sui",
    "SHIBUSDT": "shiba-inu",
    "HBARUSDT": "hedera-hashgraph",
    "LTCUSDT": "litecoin",
    "ARBUSDT": "arbitrum"
}

SIGNALS_SENT = set()
LOG_FILE = "log.txt"

bot = Bot(token=BOT_TOKEN)

def get_ohlcv(symbol):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": 50}
        res = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code != 200:
            raise Exception(f"Bad response: {res.status_code} — {res.text}")
        data = res.json()
        close_prices = [float(c[4]) for c in data]
        volumes = [float(c[5]) for c in data]
        return close_prices, volumes
    except Exception as e:
        msg = f"[get_ohlcv] Ошибка (Binance): {e}"
        print(msg)
        bot.send_message(chat_id=CHAT_ID, text=msg)
        return [], []

def ema(data, period):
    return np.convolve(data, np.ones(period) / period, mode='valid')[-1] if len(data) >= period else data[-1]

def rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = prices[-i] - prices[-i - 1]
        if diff >= 0:
            gains.append(diff)
        else:
            losses.append(-diff)
    avg_gain = np.mean(gains) if gains else 0.0001
    avg_loss = np.mean(losses) if losses else 0.0001
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def log_action(message):
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {message}\n")

def get_crypto_news():
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_KEY}&kind=news&public=true"
    try:
        data = requests.get(url).json()
        headlines = [x['title'].lower() for x in data['results'][:10]]
        neg_words = ['hack', 'lawsuit', 'scam', 'ban', 'arrest', 'exploit']
        pos_words = ['partnership', 'adoption', 'surge', 'bullish', 'growth']
        neg_hits = sum(any(k in h for k in neg_words) for h in headlines)
        pos_hits = sum(any(k in h for k in pos_words) for h in headlines)
        return pos_hits, neg_hits
    except:
        return 0, 0

def analyze():
    global SIGNALS_SENT
    btc_prices, _ = get_ohlcv("BTCUSDT")
    if not btc_prices:
        return
    btc_ema20 = ema(btc_prices[-20:], 20)
    btc_ema50 = ema(btc_prices[-50:], 50)
    trend_up = btc_ema20 > btc_ema50
    pos_news, neg_news = get_crypto_news()
    if neg_news > pos_news:
        bot.send_message(chat_id=CHAT_ID, text="📰 Обнаружены негативные новости — сигналы временно приостановлены")
        return

    for sym in COINS:
        close, vol = get_ohlcv(sym)
        if len(close) < 30:
            continue
        price = close[-1]
        ema20_val = ema(close[-20:], 20)
        ema50_val = ema(close[-50:], 50)
        rsi_val = rsi(close)
        vol_now = vol[-1]
        avg_vol = np.mean(vol[-10:])
        resistance = max(close[-30:])

        score = 0
        if ema20_val > ema50_val: score += 1
        if rsi_val < 40: score += 1
        if price < resistance * 0.98: score += 1
        if close[-1] > close[-2] > close[-3]: score += 1
        if vol_now > avg_vol * 1.3: score += 1
        if trend_up: score += 1
        if pos_news > neg_news: score += 1

        confidence = (score / 7) * 100

        if confidence >= 75 and f"BUY_{sym}" not in SIGNALS_SENT:
            price_text = f"${price:.2f}" if price >= 0.01 else f"${price:.8f}"
            msg = f"🟢 Сигнал на покупку {sym}: {price_text}\nУверенность: {confidence:.1f}%"
            bot.send_message(chat_id=CHAT_ID, text=msg)
            SIGNALS_SENT.add(f"BUY_{sym}")
            log_action(msg)

        if rsi_val > 70 and ema20_val < ema50_val and f"SELL_{sym}" not in SIGNALS_SENT:
            price_text = f"${price:.2f}" if price >= 0.01 else f"${price:.8f}"
            msg = f"🔴 Сигнал на продажу {sym}: {price_text}\nУверенность: 80%+ (перекупленность, пересечение EMA)"
            bot.send_message(chat_id=CHAT_ID, text=msg)
            SIGNALS_SENT.add(f"SELL_{sym}")
            log_action(msg)

def price_command(update: Update, context: CallbackContext):
    symbol = (context.args[0].upper() + "USDT") if context.args else "BTCUSDT"
    if symbol not in COINS:
        update.message.reply_text("Монета не найдена")
        return
    close, _ = get_ohlcv(symbol)
    
    if close:
        price = close[-1]
        if price >= 0.01:
            price_text = f"${price:.2f}"
        else:
            price_text = f"${price:.8f}"
        update.message.reply_text(f"Цена {symbol[:-4]}: {price_text}")
    else:
        update.message.reply_text("Данные недоступны")

def status_command(update: Update, context: CallbackContext):
    msg = "📊 Статус монет (оценка):\n"
    for sym in COINS:
        close, vol = get_ohlcv(sym)
        if len(close) < 30:
            continue
        price = close[-1]
        ema20_val = ema(close[-20:], 20)
        ema50_val = ema(close[-50:], 50)
        rsi_val = rsi(close)
        vol_now = vol[-1]
        avg_vol = np.mean(vol[-10:])
        resistance = max(close[-30:])

        score = 0
        if ema20_val > ema50_val: score += 1
        if rsi_val < 40: score += 1
        if price < resistance * 0.98: score += 1
        if close[-1] > close[-2] > close[-3]: score += 1
        if vol_now > avg_vol * 1.3: score += 1
        btc_prices, _ = get_ohlcv("BTCUSDT")
        if len(btc_prices) >= 50 and ema(btc_prices[-20:], 20) > ema(btc_prices[-50:], 50):
            score += 1
        pos_news, neg_news = get_crypto_news()
        if pos_news > neg_news: score += 1

        confidence = (score / 7) * 100
        if confidence >= 75:
            decision = f"Купить ({confidence:.1f}%)"
        elif rsi_val > 70 and ema20_val < ema50_val:
            decision = f"Продавать (80%+)"
        else:
            decision = f"Ждать ({confidence:.1f}%)"
        msg += f"{sym[:-4]}: {decision}\n"
    update.message.reply_text(msg)

def summary_command(update: Update, context: CallbackContext):
    msg = "💰 Текущие цены:\n"
    for sym in COINS:
        close, _ = get_ohlcv(sym)
        if close:
            price = close[-1]
            if price >= 0.01:
                price_text = f"${price:.2f}"
            else:
                price_text = f"${price:.8f}"
            msg += f"{sym[:-4]}: {price_text}\n"
    update.message.reply_text(msg)

def send_prices_regularly():
    msg = "💰 Регулярное обновление цен:\n"
    for sym in COINS:
        close, _ = get_ohlcv(sym)
        if close:
            price = close[-1]
            if price >= 0.01:
                price_text = f"${price:.2f}"
            else:
                price_text = f"${price:.8f}"
            msg += f"{sym[:-4]}: {price_text}\n"
    bot.send_message(chat_id=CHAT_ID, text=msg)

def topgainer_command(update: Update, context: CallbackContext):
    changes = []
    for sym in COINS:
        close, _ = get_ohlcv(sym)
        if len(close) >= 25:
            growth = ((close[-1] - close[-24]) / close[-24]) * 100
            changes.append((sym, growth))
    changes.sort(key=lambda x: x[1], reverse=True)
    top = changes[:3]
    msg = "🚀 Топ-3 монеты за 24ч:\n"
    for sym, g in top:
        msg += f"{sym[:-4]}: +{g:.2f}%\n"
    update.message.reply_text(msg)

def help_command(update: Update, context: CallbackContext):
    text = (
        "🤖 Команды:\n"
        "/price BTC — текущая цена монеты\n"
        "/status — статус монет\n"
        "/summary — цены всех монет\n"
        "/topgainer — топ-3 по росту\n"
        "/help — команды"
    )
    update.message.reply_text(text)

def main():
    updater = Updater(token=BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("price", price_command))
    dp.add_handler(CommandHandler("status", status_command))
    dp.add_handler(CommandHandler("summary", summary_command))
    dp.add_handler(CommandHandler("topgainer", topgainer_command))
    dp.add_handler(CommandHandler("help", help_command))

    scheduler = BackgroundScheduler(timezone=pytz.utc)
    scheduler.add_job(analyze, 'interval', minutes=2)
    scheduler.add_job(send_prices_regularly, 'interval', minutes=14)
    scheduler.start()

    PORT = int(os.environ.get("PORT", 8080))
    updater.start_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=BOT_TOKEN,
        webhook_url=f"https://{os.environ['RENDER_EXTERNAL_HOSTNAME']}/{BOT_TOKEN}"
    )
    print("🟢 Бот запущен через Webhook")
    updater.idle()

if __name__ == '__main__':
    main()
