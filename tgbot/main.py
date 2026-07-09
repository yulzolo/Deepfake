"""
Точка входа: диспетчер, модели задач, воркер очереди,
запуск FaceFusion и Deepfake Detector.
"""
import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from aiogram import Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import FSInputFile

from config import (
    DETECTOR_CHECKPOINT,
    DETECTOR_DIR,
    DETECTOR_PYTHON,
    FACEFUSION_DIR,
    FACEFUSION_PYTHON,
    PROCESS_TIMEOUT,
    WORK_DIR,
    agreed_users,
    bot,
    task_queue,
)
from handlers import router

# ============================================================
# ========================= ЛОГИ =============================
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WORK_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# ========================= МОДЕЛИ ===========================
# ============================================================
class TaskType(Enum):
    FACESWAP = "faceswap"
    DETECT = "detect"


@dataclass
class BotTask:
    user_id: int
    chat_id: int
    task_type: TaskType
    source_path: Optional[Path]
    target_path: Path
    output_path: Path
    is_video: bool
    processors: list = None


@dataclass
class ProcessResult:
    success: bool
    output_path: Optional[Path]
    error_message: Optional[str]
    stdout: str
    stderr: str


def user_str(user_id: int) -> str:
    return f"user_{user_id}"


def make_task(
    user_id: int,
    chat_id: int,
    task_type: TaskType,
    source_path: Optional[Path],
    target_path: Path,
    is_video: bool,
    processors: list = None,
) -> BotTask:
    task_id = uuid.uuid4().hex[:8]
    if task_type == TaskType.FACESWAP:
        ext = ".mp4" if is_video else ".jpg"
        output_path = WORK_DIR / user_str(user_id) / f"faceswap_{task_id}{ext}"
    else:
        output_path = WORK_DIR / user_str(user_id) / f"detect_{task_id}.txt"

    if processors is None:
        processors = ["face_swapper"]
    elif "face_swapper" not in processors:
        processors = ["face_swapper"] + processors

    return BotTask(
        user_id=user_id,
        chat_id=chat_id,
        task_type=task_type,
        source_path=source_path,
        target_path=target_path,
        output_path=output_path,
        is_video=is_video,
        processors=processors,
    )


# ============================================================
# ====================== FACEFUSION ==========================
# ============================================================
async def run_facefusion(
    source_path: Path, target_path: Path, output_path: Path, processors: list
) -> ProcessResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_abs = source_path.absolute()
    target_abs = target_path.absolute()
    output_abs = output_path.absolute()

    if not source_abs.exists():
        return ProcessResult(False, None, f"Source файл не найден: {source_abs}", "", "")
    if not target_abs.exists():
        return ProcessResult(False, None, f"Target файл не найден: {target_abs}", "", "")

    if "face_swapper" not in processors:
        processors = ["face_swapper"] + processors

    cmd = [
        FACEFUSION_PYTHON,
        "facefusion.py",
        "headless-run",
        "-s", str(source_abs),
        "-t", str(target_abs),
        "-o", str(output_abs),
        "--execution-providers", "cuda",
        "--processors",
    ] + processors

    logger.info("Запуск FaceFusion: %s", " ".join(cmd))

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(FACEFUSION_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=PROCESS_TIMEOUT
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return ProcessResult(False, None, f"Таймаут ({PROCESS_TIMEOUT} сек)", "", "")

        stdout_text = stdout.decode(errors="ignore")
        stderr_text = stderr.decode(errors="ignore")

        if output_abs.exists() and output_abs.stat().st_size > 0:
            return ProcessResult(True, output_abs, None, stdout_text, stderr_text)
        else:
            return ProcessResult(
                False, None,
                f"Файл не создан.\nSTDOUT: {stdout_text[:200]}\nSTDERR: {stderr_text[:200]}",
                stdout_text, stderr_text,
            )
    except Exception as e:
        logger.exception("Ошибка FaceFusion")
        return ProcessResult(False, None, str(e), "", "")


# ============================================================
# ====================== DETECTOR ============================
# ============================================================
async def run_deepfake_detector(target_path: Path, output_path: Path) -> ProcessResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_full = str(DETECTOR_DIR / DETECTOR_CHECKPOINT)

    cmd = [
        DETECTOR_PYTHON,
        "predict.py",
        "--checkpoint", checkpoint_full,
        "--image", str(target_path),
    ]
    logger.info("Запуск Deepfake Detector: %s", " ".join(cmd))

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(DETECTOR_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=PROCESS_TIMEOUT
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return ProcessResult(False, None, f"Таймаут ({PROCESS_TIMEOUT} сек)", "", "")

        stdout_text = stdout.decode(errors="ignore")
        stderr_text = stderr.decode(errors="ignore")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"=== STDOUT ===\n{stdout_text}\n\n=== STDERR ===\n{stderr_text}")

        if process.returncode == 0 and stdout_text.strip():
            return ProcessResult(True, output_path, None, stdout_text, stderr_text)
        else:
            return ProcessResult(
                False, None,
                stderr_text if stderr_text else "Пустой вывод от детектора",
                stdout_text, stderr_text,
            )
    except Exception as e:
        logger.exception("Ошибка Detector")
        return ProcessResult(False, None, str(e), "", "")


def parse_detector_output(stdout: str) -> tuple:
    try:
        data = json.loads(stdout)
        return data.get('is_deepfake', False), data.get('confidence', 0.0), data.get('details', '')
    except json.JSONDecodeError:
        pass

    text = stdout.strip()
    is_deepfake = "deepfake" in text.lower() or "fake" in text.lower()
    confidence = 0.0
    numbers = re.findall(r"(\d+\.?\d*)", text)
    if numbers:
        try:
            confidence = float(numbers[0])
            if confidence > 1.0:
                confidence = confidence / 100.0
        except ValueError:
            pass
    return is_deepfake, confidence, text


# ============================================================
# ========================= ВОКЕР ============================
# ============================================================
async def process_task(task: BotTask):
    if task_queue.qsize() > 0:
        await bot.send_message(
            task.chat_id,
            f"Задача в очереди. Перед вами ещё {task_queue.qsize()} задач.",
        )

    status_msg = await bot.send_message(task.chat_id, "Начинаю обработку...")

    if task.task_type == TaskType.FACESWAP:
        result = await run_facefusion(
            task.source_path, task.target_path, task.output_path, task.processors
        )
        if result.success:
            await status_msg.delete()
            await bot.send_photo(
                task.chat_id,
                photo=FSInputFile(result.output_path),
                caption="Готово!",
            )
        else:
            await status_msg.edit_text(f"Ошибка генерации:\n{result.error_message}")
    else:
        result = await run_deepfake_detector(task.target_path, task.output_path)
        if result.success:
            is_deepfake, confidence, details = parse_detector_output(result.stdout)
            label = "FAKE" if is_deepfake else "REAL"
            response = (
                f"Результат: {label}\n"
                f"Уверенность: {confidence * 100:.1f}%\n"
                f"{details}"
            )
            await status_msg.edit_text(response)
        else:
            await status_msg.edit_text(f"Ошибка проверки:\n{result.error_message}")


async def worker_loop():
    logger.info("Воркер очереди запущен")
    while True:
        task = await task_queue.get()
        try:
            await process_task(task)
        except Exception as e:
            logger.exception("Ошибка обработки задачи")
            try:
                await bot.send_message(task.chat_id, f"Непредвиденная ошибка: {e}")
            except Exception:
                pass
        finally:
            task_queue.task_done()


# ============================================================
# ========================= ЗАПУСК ===========================
# ============================================================
async def main():
    logger.info("Загружено %d пользователей, принявших соглашение", len(agreed_users))
    logger.info("Рабочая папка: %s", WORK_DIR)

    if not FACEFUSION_DIR.exists():
        logger.warning("FACEFUSION_DIR не найден: %s", FACEFUSION_DIR)
    if not Path(FACEFUSION_PYTHON).exists():
        logger.warning("FACEFUSION_PYTHON не найден: %s", FACEFUSION_PYTHON)
    if not DETECTOR_DIR.exists():
        logger.warning("DETECTOR_DIR не найден: %s", DETECTOR_DIR)
    if not Path(DETECTOR_PYTHON).exists():
        logger.warning("DETECTOR_PYTHON не найден: %s", DETECTOR_PYTHON)

    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    asyncio.create_task(worker_loop())

    logger.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
