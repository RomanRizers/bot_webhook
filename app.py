
import os, re, json, random, warnings
from io import BytesIO
from pathlib import Path

import telebot
from flask import Flask, request, jsonify
from gtts import gTTS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

TOKEN       = os.environ["TELEGRAM_BOT_TOKEN"]
WEBHOOK_URL = os.environ["WEBHOOK_URL"]   # https://your-app.up.railway.app
PORT        = int(os.environ.get("PORT", 5000))

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# ── ML-бот ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 0.25
FAIL_PHRASES = [
    "Не понял запрос. Переформулируй, пожалуйста.",
    "Не могу уверенно распознать смысл.",
    "Уточни, что именно тебя интересует.",
]

def normalize(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

CONFIG_PATH = Path("edu_bot_config_lab3.json")
if CONFIG_PATH.exists():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
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
    fail_phrases = cfg.get("failure_phrases", FAIL_PHRASES)

    def bot_reply(text):
        tn   = normalize(text)
        feat = vec.transform([tn])
        conf = float(clf.predict_proba(feat).max())
        sim  = float(cosine_similarity(feat, X_v).max())
        if conf < CONFIDENCE_THRESHOLD or sim < SIMILARITY_THRESHOLD:
            return random.choice(fail_phrases)
        intent = clf.predict(feat)[0]
        responses = cfg["intents"][intent].get("responses", FAIL_PHRASES)
        return random.choice(responses) if responses else random.choice(fail_phrases)
else:
    bot_reply = lambda text: random.choice(FAIL_PHRASES)

# ── Обработчики ───────────────────────────────────────────────────────────────
@bot.message_handler(commands=["start", "help"])
def cmd_start(message):
    name = message.from_user.first_name
    bot.send_message(message.chat.id,
        f"Привет, {name}! Я учебный бот. Задай вопрос про расписание, сессию, справки и другое.")

@bot.message_handler(content_types=["text"])
def handle_text(message):
    name   = message.from_user.first_name
    answer = bot_reply(message.text)
    print(f"[TEXT] {name}: {message.text}")
    print(f"[TEXT] Бот: {answer}")
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
    return jsonify({"status": "ok"})

@app.route("/")
def index():
    return jsonify({"status": "running"})

# ── Запуск ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.remove_webhook()
    bot.set_webhook(url=f"{WEBHOOK_URL}/{TOKEN}")
    print(f"Webhook: {WEBHOOK_URL}/{TOKEN}")
    app.run(host="0.0.0.0", port=PORT)
