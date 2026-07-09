"""
Все клавиатуры бота.
"""
from typing import Set

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def get_main_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.add(
        KeyboardButton(text="Face Swap"),
        KeyboardButton(text="Check Deepfake"),
    )
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)


def get_swap_options_keyboard(selected: Set[str] = None) -> ReplyKeyboardMarkup:
    """Клавиатура выбора опций для Face Swap."""
    if selected is None:
        selected = set()
    builder = ReplyKeyboardBuilder()
    face_enhance_text = "✅ Улучшение лица" if "face_enhancer" in selected else "❌ Улучшение лица"
    frame_enhance_text = "✅ Апскейл" if "frame_enhancer" in selected else "❌ Апскейл"
    builder.add(
        KeyboardButton(text=face_enhance_text),
        KeyboardButton(text=frame_enhance_text),
        KeyboardButton(text="🔄 Сбросить всё"),
        KeyboardButton(text="▶️ Запустить обработку"),
        KeyboardButton(text="Отмена"),
    )
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)


def get_cancel_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.add(KeyboardButton(text="Отмена"))
    return builder.as_markup(resize_keyboard=True)


def get_agreement_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Принимаю", callback_data="accept_agreement")],
        [InlineKeyboardButton(text="Отказаться", callback_data="decline_agreement")],
    ])
