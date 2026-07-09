import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
from config import WORK_DIR 


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
