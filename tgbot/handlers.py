import random
import uuid
from pathlib import Path
from aiogram import F, Router
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from aiogram.enums import ParseMode

from config import (
    ACK_PHRASES,
    AGREEMENT_TEXT,
    CheckStates,
    SwapStates,
    WELCOME_TEXT,
    WORK_DIR,
    add_agreed_user,
    bot,
    is_user_agreed,
    task_queue,
)
from keyboards import (
    get_agreement_keyboard,
    get_cancel_keyboard,
    get_main_keyboard,
    get_swap_options_keyboard,
)

from models import TaskType, make_task, user_str

router = Router()

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    if is_user_agreed(message.from_user.id):
        await message.answer(
            WELCOME_TEXT,
            parse_mode=ParseMode.HTML,
            reply_markup=get_main_keyboard(),
        )
    else:
        await message.answer(WELCOME_TEXT, parse_mode=ParseMode.HTML)
        await message.answer(
            AGREEMENT_TEXT,
            parse_mode=ParseMode.HTML,
            reply_markup=get_agreement_keyboard(),
        )


@router.callback_query(F.data == "accept_agreement")
async def accept_agreement(callback: CallbackQuery):
    add_agreed_user(callback.from_user.id)
    await callback.answer("Согласие принято")
    await callback.message.delete()
    await callback.message.answer(
        WELCOME_TEXT,
        parse_mode=ParseMode.HTML,
        reply_markup=get_main_keyboard(),
    )


@router.callback_query(F.data == "decline_agreement")
async def decline_agreement(callback: CallbackQuery):
    await callback.answer("Согласие отклонено")
    await callback.message.delete()
    await callback.message.answer(
        "Вы отклонили пользовательское соглашение.\n\n"
        "Без согласия вы не можете пользоваться ботом.\n"
        "Если передумаете — нажмите /start"
    )


@router.message(Command("cancel"))
@router.message(F.text == "Отмена")
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    if not is_user_agreed(message.from_user.id):
        await message.answer(
            "Вы не приняли пользовательское соглашение.\n"
            "Нажмите /start чтобы прочитать и принять."
        )
        return
    await message.answer(
        "Отменено. Выберите действие:",
        reply_markup=get_main_keyboard(),
    )


@router.message(F.text == "Face Swap")
@router.message(Command("swap"))
async def cmd_swap(message: Message, state: FSMContext):
    if not is_user_agreed(message.from_user.id):
        await message.answer(
            "Вы не приняли пользовательское соглашение.\n"
            "Нажмите /start чтобы прочитать и принять."
        )
        return

    await state.set_state(SwapStates.waiting_options)
    await state.update_data(selected_processors=set())

    menu_msg = await message.answer(
        "⚙️ Выберите дополнительные опции:",
        reply_markup=get_swap_options_keyboard(set()),
    )
    await state.update_data(options_menu_msg_id=menu_msg.message_id)


@router.message(
    StateFilter(SwapStates.waiting_options),
    F.text.startswith("✅") | F.text.startswith("❌")
    | F.text.in_(["🔄 Сбросить всё", "▶️ Запустить обработку"]),
)
async def on_option_toggle(message: Message, state: FSMContext):
    data = await state.get_data()
    selected = data.get("selected_processors", set())
    menu_msg_id = data.get("options_menu_msg_id")

    processor_map = {
        "Улучшение лица": "face_enhancer",
        "Апскейл": "frame_enhancer",
    }
    text = message.text

    if text == "🔄 Сбросить всё":
        selected = set()
        await state.update_data(selected_processors=selected)
        phrase = random.choice(ACK_PHRASES)
        new_text = f"{phrase}\n\n⚙️ Выберите дополнительные опции:"
        try:
            await message.bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=menu_msg_id,
                text=new_text,
                reply_markup=get_swap_options_keyboard(selected),
            )
        except Exception:
            await message.answer(new_text, reply_markup=get_swap_options_keyboard(selected))
        return

    if text == "▶️ Запустить обработку":
        final_processors = ["face_swapper"] + list(selected)
        await state.update_data(selected_processors=selected)
        await state.set_state(SwapStates.waiting_source)
        processors_text = ", ".join(final_processors)
        await message.answer(
            f"Выбраны процессоры: {processors_text}\n\n"
            "Теперь пришлите SOURCE-фото (чьё лицо перенести):",
            reply_markup=get_cancel_keyboard(),
        )
        return

    for display_name, processor in processor_map.items():
        if display_name in text:
            if processor in selected:
                selected.remove(processor)
            else:
                selected.add(processor)
            break

    await state.update_data(selected_processors=selected)
    phrase = random.choice(ACK_PHRASES)
    new_text = f"{phrase}\n\n️ Выберите дополнительные опции:"

    try:
        await message.bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=menu_msg_id,
            text=new_text,
            reply_markup=get_swap_options_keyboard(selected),
        )
    except Exception:
        await message.answer(new_text, reply_markup=get_swap_options_keyboard(selected))


@router.message(StateFilter(SwapStates.waiting_options))
async def on_options_wrong(message: Message):
    await message.answer(
        "Используйте кнопки для выбора опций:",
        reply_markup=get_swap_options_keyboard(set()),
    )


@router.message(StateFilter(SwapStates.waiting_source), F.photo)
async def on_source_photo(message: Message, state: FSMContext):
    session_id = uuid.uuid4().hex[:8]
    user_dir = WORK_DIR / user_str(message.from_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    source_path = user_dir / f"{session_id}_source.jpg"

    await bot.download(message.photo[-1], destination=source_path)

    await state.update_data(session_id=session_id, source_path=str(source_path))
    await state.set_state(SwapStates.waiting_target)
    await message.answer(
        "Принято. Теперь пришлите TARGET-фото:",
        reply_markup=get_cancel_keyboard(),
    )


@router.message(StateFilter(SwapStates.waiting_source))
async def on_source_wrong(message: Message):
    await message.answer("Отправьте именно фото.")


@router.message(StateFilter(SwapStates.waiting_target), F.photo)
async def on_target_photo(message: Message, state: FSMContext):
    data = await state.get_data()
    session_id = data["session_id"]
    source_path = Path(data["source_path"])
    selected_processors = data.get("selected_processors", set())

    user_dir = WORK_DIR / user_str(message.from_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    target_path = user_dir / f"{session_id}_target.jpg"

    await bot.download(message.photo[-1], destination=target_path)

    processors = ["face_swapper"] + list(selected_processors)
    task = make_task(
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        task_type=TaskType.FACESWAP,
        source_path=source_path,
        target_path=target_path,
        is_video=False,
        processors=processors,
    )
    await task_queue.put(task)
    await state.clear()
    await message.answer(
        "Задача добавлена в очередь. Ожидайте...",
        reply_markup=get_main_keyboard(),
    )


@router.message(StateFilter(SwapStates.waiting_target))
async def on_target_wrong(message: Message):
    await message.answer("Отправьте именно фото.")


@router.message(F.text == "Check Deepfake")
@router.message(Command("check"))
async def cmd_check(message: Message, state: FSMContext):
    if not is_user_agreed(message.from_user.id):
        await message.answer(
            "Вы не приняли пользовательское соглашение.\n"
            "Нажмите /start чтобы прочитать и принять."
        )
        return

    await state.set_state(CheckStates.waiting_photo)
    await message.answer(
        "Пришлите фото для проверки на deepfake:",
        reply_markup=get_cancel_keyboard(),
    )


@router.message(StateFilter(CheckStates.waiting_photo), F.photo)
async def on_check_photo(message: Message, state: FSMContext):
    user_dir = WORK_DIR / user_str(message.from_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    target_path = user_dir / f"check_{uuid.uuid4().hex[:8]}.jpg"

    await bot.download(message.photo[-1], destination=target_path)

    task = make_task(
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        task_type=TaskType.DETECT,
        source_path=None,
        target_path=target_path,
        is_video=False,
    )
    await task_queue.put(task)
    await state.clear()
    await message.answer(
        "Задача добавлена в очередь. Ожидайте...",
        reply_markup=get_main_keyboard(),
    )


@router.message(StateFilter(CheckStates.waiting_photo))
async def on_check_wrong(message: Message):
    await message.answer("Отправьте именно фото.")


@router.message()
async def handle_other(message: Message, state: FSMContext):
    if not is_user_agreed(message.from_user.id):
        await message.answer(
            "Вы не приняли пользовательское соглашение.\n"
            "Нажмите /start чтобы прочитать и принять."
        )
        return

    current_state = await state.get_state()
    if current_state is None:
        await message.answer(
            "Используйте кнопки для выбора действия:",
            reply_markup=get_main_keyboard(),
        )
    else:
        await message.answer(
            "Следуйте инструкциям или нажмите Отмена.",
            reply_markup=get_cancel_keyboard(),
        )
