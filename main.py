import requests
import time
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

BUY_PRICES_FILE = "buy_prices.json"
LOG_FILE = "log.txt"

bot = Bot(token=BOT_TOKEN)

def get_ohlcv(symbol):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1h",
            "limit": 50
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, params=params, headers=headers)
        print(f"[DEBUG] Request URL: {res.url}")
        print(f"[DEBUG] Status Code: {res.status_code}")
        if res.status_code != 200:
            raise Exception(f"Bad response: {res.status_code} — {res.text}")
        data = res.json()
        close_prices = [float(candle[4]) for candle in data]
        volumes = [float(candle[5]) for candle in data]
        return close_prices, volumes
    except Exception as e:
        print(f"[get_ohlcv] Ошибка (Binance): {e}")
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

def load_buy_prices():
    try:
        with open(BUY_PRICES_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_buy_prices(data):
    with open(BUY_PRICES_FILE, 'w') as f:
        json.dump(data, f)

def log_action(message):
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {message}\n")

def get_crypto_news():
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_KEY}&kind=news&public=true"
    try:
        response = requests.get(url)
        data = response.json()
        headlines = [x['title'].lower() for x in data['results'][:10]]
        negative_keywords = ['hack', 'lawsuit', 'scam', 'ban', 'arrest', 'exploit']
        positive_keywords = ['partnership', 'adoption', 'surge', 'bullish', 'growth']
        negative_hits = sum(any(k in h for k in negative_keywords) for h in headlines)
        positive_hits = sum(any(k in h for k in positive_keywords) for h in headlines)
        return positive_hits, negative_hits
    except:
        return 0, 0

def analyze():
    buy_prices = load_buy_prices()
    btc_prices, _ = get_ohlcv("BTCUSDT")
    if not btc_prices:
        return
    btc_ema20 = ema(btc_prices[-20:], 20)
    btc_ema50 = ema(btc_prices[-50:], 50)
    skip_signals = btc_ema20 < btc_ema50
    pos_news, neg_news = get_crypto_news()
    if neg_news > pos_news:
        bot.send_message(chat_id=CHAT_ID, text="📰 Обнаружены негативные новости — сигналы временно приостановлены")
        skip_signals = True

    for symbol in COINS:
        close_prices, volumes = get_ohlcv(symbol)
        if not close_prices:
            continue
        price = close_prices[-1]
        ema20_val = ema(close_prices[-20:], 20)
        ema50_val = ema(close_prices[-50:], 50)
        rsi_val = rsi(close_prices)
        vol_now = volumes[-1]
        avg_vol = np.mean(volumes[-10:])
        resistance = max(close_prices[-30:])

        if symbol in buy_prices:
            entry = buy_prices[symbol]
            if price <= entry * 0.95:
                msg = f"❗️ Стоп-лосс по {symbol}: ${price:.2f} (убыток >5%)"
                bot.send_message(chat_id=CHAT_ID, text=msg)
                log_action(msg)
                del buy_prices[symbol]
                save_buy_prices(buy_prices)
                continue

        if (not skip_signals and ema20_val > ema50_val and rsi_val < 40
                and vol_now > avg_vol * 1.3
                and close_prices[-1] > close_prices[-2] > close_prices[-3]
                and price < resistance * 0.98):

            if symbol not in buy_prices:
                buy_prices[symbol] = price
                save_buy_prices(buy_prices)
                msg = (f"🟢 Сигнал на ПОКУПКУ {symbol}: ${price:.2f}\n"
                       f"EMA20>EMA50, RSI={rsi_val:.1f}, объём выше нормы\n"
                       f"BTC-тренд: Восходящий, Новости: 👍{pos_news} 👎{neg_news}")
                bot.send_message(chat_id=CHAT_ID, text=msg)
                log_action(msg)

        elif symbol in buy_prices:
            entry = buy_prices[symbol]
            if price >= entry * 1.05:
                msg = f"🔴 Сигнал на ПРОДАЖУ {symbol}: ${price:.2f} (прибыль +5%)"
                bot.send_message(chat_id=CHAT_ID, text=msg)
                log_action(msg)
                del buy_prices[symbol]
                save_buy_prices(buy_prices)

def price_command(update: Update, context: CallbackContext):
    symbol = (context.args[0].upper() + "USDT") if context.args else "BTCUSDT"
    if symbol not in COINS:
        update.message.reply_text("Монета не найдена")
        return
    close, _ = get_ohlcv(symbol)
    if not close:
        update.message.reply_text("Данные недоступны")
    else:
        update.message.reply_text(f"Цена {symbol[:-4]}: ${close[-1]:.2f}")

def status_command(update: Update, context: CallbackContext):
    buy_prices = load_buy_prices()
    msg = "📊 Статус монет:\n"
    for sym in COINS:
        close, _ = get_ohlcv(sym)
        if not close:
            continue
        current = close[-1]
        if sym in buy_prices and current >= buy_prices[sym] * 1.05:
            status = "Продавать"
        elif sym not in buy_prices:
            status = "Купить"
        else:
            status = "Ждать"
        msg += f"{sym[:-4]}: {status}\n"
    update.message.reply_text(msg)

def reset_command(update: Update, context: CallbackContext):
    save_buy_prices({})
    update.message.reply_text("✅ Все buy цены сброшены.")
    log_action("[RESET] Все buy цены сброшены пользователем.")

def summary_command(update: Update, context: CallbackContext):
    msg = "💰 Текущие цены:\n"
    for sym in COINS:
        close, _ = get_ohlcv(sym)
        if close:
            msg += f"{sym[:-4]}: ${close[-1]:.2f}\n"
    update.message.reply_text(msg)

def topgainer_command(update: Update, context: CallbackContext):
    top_symbol = ""
    top_growth = -100
    for sym in COINS:
        close, _ = get_ohlcv(sym)
        if len(close) < 25:
            continue
        growth = ((close[-1] - close[-24]) / close[-24]) * 100
        if growth > top_growth:
            top_growth = growth
            top_symbol = sym
    if top_symbol:
        update.message.reply_text(f"🚀 Топ-гейнер за 24ч: {top_symbol[:-4]} +{top_growth:.2f}%")
    else:
        update.message.reply_text("Не удалось определить лидера роста")

def help_command(update: Update, context: CallbackContext):
    help_text = ("🤖 Доступные команды:\n"
                 "/price BTC — текущая цена монеты\n"
                 "/status — статус: Купить / Ждать / Продавать\n"
                 "/summary — текущие цены всех монет\n"
                 "/topgainer — лидер роста за 24 часа\n"
                 "/reset — сбросить все buy-цены\n"
                 "/help — список всех команд")
    update.message.reply_text(help_text)

def main():
    updater = Updater(token=BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("price", price_command))
    dp.add_handler(CommandHandler("status", status_command))
    dp.add_handler(CommandHandler("reset", reset_command))
    dp.add_handler(CommandHandler("summary", summary_command))
    dp.add_handler(CommandHandler("topgainer", topgainer_command))
    dp.add_handler(CommandHandler("help", help_command))

    # ✅ Планировщик с pytz UTC
    scheduler = BackgroundScheduler(timezone=pytz.utc)
    scheduler.add_job(analyze, 'interval', minutes=2)
    scheduler.start()

    # ✅ Вебхук
    PORT = int(os.environ.get("PORT", 8443))
    updater.start_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=BOT_TOKEN,
        webhook_url=f"https://{os.environ['RENDER_EXTERNAL_HOSTNAME']}/{BOT_TOKEN}"
    )

    print("🟢 Бот запущен через Webhook")

    # 🔁 Держим процесс активным
    updater.idle()  # <--- ОБЯЗАТЕЛЬНО

