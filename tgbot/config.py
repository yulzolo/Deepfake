"""
Конфигурация, тексты, FSM-состояния, хранилище пользователей, бот и очередь.
"""
import asyncio
import json
from pathlib import Path

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.state import State, StatesGroup

# ============================================================
# ========================= НАСТРОЙКИ ========================
# ============================================================
BOT_TOKEN = "СЮДА_ТВОЙ_ТОКЕН"

FACEFUSION_DIR = Path(r"C:\Users\user\Documents\facefusion")
FACEFUSION_PYTHON = r"C:\ProgramData\anaconda3\envs\facefusion\python.exe"

DETECTOR_DIR = Path(r"C:\Study\detector")
DETECTOR_PYTHON = r"C:\ProgramData\anaconda3\envs\detector\python.exe"
DETECTOR_CHECKPOINT = "checkpoints/detector_v3.pt"

WORK_DIR = FACEFUSION_DIR / "bot_workdir"
MAX_FILE_SIZE = 20 * 1024 * 1024
PROCESS_TIMEOUT = 60 * 15
AGREEMENT_FILE = Path(__file__).parent / "agreed_users.json"

# ============================================================
# ========================= БОТ И ОЧЕРЕДЬ ====================
# ============================================================
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
task_queue: asyncio.Queue = asyncio.Queue()

# ============================================================
# ========================== ТЕКСТЫ ==========================
# ============================================================
WELCOME_TEXT = """
Привет! Я AI-помощник по работе с лицами.

Что я умею:

<b>Face Swap</b> — замена лица на фото
→ выбери нужные опции
→ пришли фото с лицом-донором (source)
→ потом фото-цель (target)

<b>Check Deepfake</b> — анализ фото на подделку
→ пришли любое фото
→ я скажу, настоящее оно или сгенерировано ИИ
"""

AGREEMENT_TEXT = """
<b>ПОЛЬЗОВАТЕЛЬСКОЕ СОГЛАШЕНИЕ</b>

Используя бота, вы соглашаетесь:

<b>Ответственность</b>
• Вы несёте ответственность за использование сгенерированных изображений
• Бот не несёт ответственности за последствия использования

<b>Запрещено</b>
• Создавать дипфейки без согласия людей
• Использовать для клеветы, мошенничества или незаконных целей
• Создавать оскорбительный или порнографический контент
• Распространять дезинформацию

<b>Конфиденциальность</b>
• Все фото удаляются после обработки
• Данные не хранятся и не передаются третьим лицам

<b>Ограничения</b>
• Бот предоставляется "как есть"
• Мы не гарантируем 100% точность детекции
• Результаты носят ознакомительный характер

Нарушение правил влечёт блокировку.

Принимаете условия?
"""

ACK_PHRASES = [
    "Окей 👌",
    "Понял, принял 📝",
    "Сделано ✨",
    "Учтено 🤖",
    "Хорошо 👍",
    "Записал 🖊",
    "Ок 🆗",
    "Принято 🫡",
    "Есть 🫡",
    "Зафиксировал 📌",
]

# ============================================================
# ======================= FSM СОСТОЯНИЯ ======================
# ============================================================
class SwapStates(StatesGroup):
    waiting_options = State()
    waiting_source = State()
    waiting_target = State()


class CheckStates(StatesGroup):
    waiting_photo = State()

# ============================================================
# ============= ХРАНИЛИЩЕ ПОЛЬЗОВАТЕЛЕЙ (ФАЙЛ) ==============
# ============================================================
def load_agreed_users() -> set:
    if AGREEMENT_FILE.exists():
        try:
            with open(AGREEMENT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return set(data.get("users", []))
        except (json.JSONDecodeError, KeyError):
            return set()
    return set()


def save_agreed_users(users: set) -> None:
    with open(AGREEMENT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"users": list(users)}, f, ensure_ascii=False, indent=2)


agreed_users = load_agreed_users()


def is_user_agreed(user_id: int) -> bool:
    return user_id in agreed_users


def add_agreed_user(user_id: int) -> None:
    agreed_users.add(user_id)
    save_agreed_users(agreed_users)
