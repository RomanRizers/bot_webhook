import os, re, json, random, warnings, threading
from io import BytesIO
from pathlib import Path

import telebot
from flask import Flask, request, jsonify
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

TOKEN       = os.environ["TELEGRAM_BOT_TOKEN"]
WEBHOOK_URL = os.environ["WEBHOOK_URL"].rstrip("/")
PORT        = int(os.environ.get("PORT", 5000))

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# ── ML-бот (обучается в фоне) ─────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 0.25
FAIL_PHRASES = [
    "Не понял запрос. Переформулируй, пожалуйста.",
    "Не могу уверенно распознать смысл.",
    "Уточни, что именно тебя интересует.",
]

_vec = _X_v = _clf = _cfg = _fail_phrases = None
_model_ready = threading.Event()

def normalize(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def _train():
    global _vec, _X_v, _clf, _cfg, _fail_phrases
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.neural_network import MLPClassifier
        cfg_path = Path("edu_bot_config_lab3.json")
        if not cfg_path.exists():
            print("[MODEL] edu_bot_config_lab3.json не найден", flush=True)
            return
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        X_texts, y_labels = [], []
        for intent, data in cfg["intents"].items():
            for ex in data.get("examples", []):
                if isinstance(ex, str) and ex.strip():
                    X_texts.append(normalize(ex))
                    y_labels.append(intent)
        vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        X_v = vec.fit_transform(X_texts)
        clf = MLPClassifier(hidden_layer_sizes=(160,), max_iter=500, random_state=42)
        clf.fit(X_v, y_labels)
        _vec, _X_v, _clf, _cfg = vec, X_v, clf, cfg
        _fail_phrases = cfg.get("failure_phrases", FAIL_PHRASES)
        _model_ready.set()
        print(f"[MODEL] Готов! Интентов: {len(cfg['intents'])}", flush=True)
    except Exception as e:
        print(f"[MODEL] Ошибка обучения: {e}", flush=True)

def bot_reply(text):
    if not _model_ready.is_set():
        return "Загружаюсь, подожди 30 секунд и спроси снова."
    from sklearn.metrics.pairwise import cosine_similarity
    tn   = normalize(text)
    feat = _vec.transform([tn])
    conf = float(_clf.predict_proba(feat).max())
    sim  = float(cosine_similarity(feat, _X_v).max())
    if conf < CONFIDENCE_THRESHOLD or sim < SIMILARITY_THRESHOLD:
        return random.choice(_fail_phrases)
    intent = _clf.predict(feat)[0]
    responses = _cfg["intents"][intent].get("responses", FAIL_PHRASES)
    return random.choice(responses) if responses else random.choice(_fail_phrases)

# ── Обработчики ───────────────────────────────────────────────────────────────
@bot.message_handler(commands=["start", "help"])
def cmd_start(message):
    name = message.from_user.first_name
    bot.send_message(message.chat.id,
        f"Привет, {name}! Задай вопрос про расписание, сессию, справки и другое.")

@bot.message_handler(content_types=["text"])
def handle_text(message):
    name   = message.from_user.first_name
    answer = bot_reply(message.text)
    print(f"[TEXT] {name}: {message.text}", flush=True)
    print(f"[TEXT] Бот: {answer}", flush=True)
    bot.send_message(message.chat.id, answer)
    buf = BytesIO()
    gTTS(text=answer, lang="ru").write_to_fp(buf)
    buf.seek(0)
    bot.send_voice(message.chat.id, buf)

# ── Flask эндпоинты ───────────────────────────────────────────────────────────
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    if request.content_type == "application/json":
        update = telebot.types.Update.de_json(request.get_data().decode("utf-8"))
        bot.process_new_updates([update])
        return jsonify({"ok": True})
    return "Bad Request", 400

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": _model_ready.is_set()})

@app.route("/")
def index():
    return jsonify({"status": "running"})

@app.route("/setup")
def manual_setup():
    try:
        _do_setup_webhook()
        return jsonify({"ok": True, "webhook": f"{WEBHOOK_URL}/{TOKEN}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ── Webhook в фоне ────────────────────────────────────────────────────────────
def _do_setup_webhook():
    url = f"{WEBHOOK_URL}/{TOKEN}"
    bot.remove_webhook()
    bot.set_webhook(url=url)
    print(f"[WEBHOOK] Установлен: {url}", flush=True)

def _setup_webhook_bg():
    try:
        _do_setup_webhook()
    except Exception as e:
        print(f"[WEBHOOK] Ошибка: {e}", flush=True)

# ── Старт ─────────────────────────────────────────────────────────────────────
print(f"[STARTUP] PORT={PORT}  WEBHOOK_URL={WEBHOOK_URL}", flush=True)
threading.Thread(target=_train, daemon=True).start()
threading.Thread(target=_setup_webhook_bg, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
