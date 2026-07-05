import os
import logging
import asyncio
import hashlib

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import (
    Message,
    BufferedInputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
    BotCommand,
)

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ["BOT_TOKEN"]

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

user_state = {}

BTN_PHOTO = "Фото + Фото"
BTN_VIDEO = "Фото + Видео"
BTN_RESTART = "Начать заново"


def mode_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_PHOTO), KeyboardButton(text=BTN_VIDEO)],
        ],
        resize_keyboard=True,
    )


def restart_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_RESTART)],
        ],
        resize_keyboard=True,
    )


async def show_mode_choice(target):
    await target.answer("Выбери режим:", reply_markup=mode_keyboard())


def get_photo_file_id(message: Message):
    if message.photo:
        return message.photo[-1].file_id
    if message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        return message.document.file_id
    return None


def get_video_file_id(message: Message):
    if message.video:
        return message.video.file_id
    if message.document and message.document.mime_type and message.document.mime_type.startswith("video/"):
        return message.document.file_id
    return None


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


@dp.message(Command("start"))
async def start_command(message: Message):
    user_state.pop(message.from_user.id, None)
    await show_mode_choice(message)


@dp.message()
async def handle_message(message: Message):
    user_id = message.from_user.id
    state = user_state.get(user_id)

    if message.text == BTN_RESTART:
        user_state.pop(user_id, None)
        await show_mode_choice(message)
        return

    if message.text == BTN_PHOTO:
        user_state[user_id] = {"step": "waiting_source", "target_kind": "photo"}
        await message.answer(
            "Пришли фото человека, чьё лицо нужно перенести (источник). Можно как фото или как файл — файлом будет качественнее.",
            reply_markup=restart_keyboard(),
        )
        return

    if message.text == BTN_VIDEO:
        user_state[user_id] = {"step": "waiting_source", "target_kind": "video"}
        await message.answer(
            "Пришли фото человека, чьё лицо нужно перенести (источник). Можно как фото или как файл — файлом будет качественнее.",
            reply_markup=restart_keyboard(),
        )
        return

    if state is None:
        await show_mode_choice(message)
        return

    step = state["step"]
    target_kind = state["target_kind"]

    if step == "waiting_source":
        source_file_id = get_photo_file_id(message)
        if source_file_id is None:
            await message.answer("Нужно фото (как фото или как файл-картинка). Попробуй ещё раз, либо нажми \"Начать заново\".")
            return

        source_bytes = await download_telegram_file(source_file_id)
        state["source_bytes"] = source_bytes
        state["step"] = "waiting_target"

        if target_kind == "photo":
            await message.answer("Принял. Теперь пришли целевое фото (как фото или как файл).")
        else:
            await message.answer("Принял. Теперь пришли целевое видео (как видео или как файл).")
        return

    if step == "waiting_target":
        if target_kind == "photo":
            target_file_id = get_photo_file_id(message)
        else:
            target_file_id = get_video_file_id(message)

        if target_file_id is None:
            if target_kind == "photo":
                await message.answer("Нужно фото (как фото или как файл-картинка). Попробуй ещё раз, либо нажми \"Начать заново\".")
            else:
                await message.answer("Нужно видео (как видео или как файл). Попробуй ещё раз, либо нажми \"Начать заново\".")
            return

        target_bytes = await download_telegram_file(target_file_id)

        if target_kind == "photo" and file_hash(target_bytes) == file_hash(state["source_bytes"]):
            await message.answer("Это то же самое фото, что и источник. Пришли, пожалуйста, другое целевое фото.")
            return

        status_message = await message.answer("Обрабатываю, подожди немного...")

        if target_kind == "photo":
            result_file = BufferedInputFile(target_bytes, filename="result.jpg")
            await message.answer_photo(result_file, caption="Готово! (бета: тут просто твоё же фото)")
        else:
            result_file = BufferedInputFile(target_bytes, filename="result.mp4")
            await message.answer_video(result_file, caption="Готово! (бета: тут просто твоё же видео)")

        await status_message.delete()
        del user_state[user_id]

        await show_mode_choice(message)
        return


async def download_telegram_file(file_id: str) -> bytes:
    tg_file = await bot.get_file(file_id)
    file_data = await bot.download_file(tg_file.file_path)
    return file_data.read()


async def main():
    await bot.set_my_commands([
        BotCommand(command="start", description="Начать заново / выбрать режим"),
    ])
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
