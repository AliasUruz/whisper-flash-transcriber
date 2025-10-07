"""Download controller for orchestrating ASR model downloads."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from . import model_manager
from . import state_manager as sm
from .logging_utils import get_logger, log_context

LOGGER = get_logger("whisper_flash_transcriber.model_download_controller", component="ModelDownloadController")


@dataclass
class DownloadTask:
    """In-memory representation of a scheduled download."""

    task_id: str
    model_id: str
    backend: str
    cache_dir: str
    quant: Optional[str]
    timeout: Optional[float]
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    stage: str = "queued"
    message: str = ""
    bytes_done: int = 0
    bytes_total: int = 0
    eta_seconds: Optional[float] = None
    throughput_bps: Optional[float] = None
    target_dir: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    pause_requested: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)
    result: Optional[model_manager.ModelDownloadResult] = None
    error: Optional[BaseException] = None

    def snapshot(self) -> dict:
        percent = None
        if self.bytes_total:
            percent = min(100.0, (self.bytes_done / self.bytes_total) * 100.0)
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "backend": self.backend,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "percent": percent,
            "eta_seconds": self.eta_seconds,
            "throughput_bps": self.throughput_bps,
            "target_dir": self.target_dir,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": dict(self.metadata),
        }


class ModelDownloadController:
    """Coordinates model downloads with bounded concurrency and observability."""

    def __init__(
        self,
        *,
        state_manager: sm.StateManager,
        config_manager,
        max_parallel_downloads: int = 1,
        on_task_finished: Optional[Callable[[DownloadTask], None]] = None,
    ) -> None:
        self._state_manager = state_manager
        self._config_manager = config_manager
        self._on_task_finished = on_task_finished
        self._lock = threading.RLock()
        self._tasks: Dict[str, DownloadTask] = {}
        self._task_order: list[str] = []
        self._futures: Dict[str, Future] = {}
        self._max_parallel = max(1, int(max_parallel_downloads))
        # Allow bursting up to schema limits while constraining via semaphore.
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ModelDownload")
        self._semaphore = threading.Semaphore(self._max_parallel)

    def shutdown(self) -> None:
        with self._lock:
            for task in self._tasks.values():
                task.cancel_event.set()
            self._executor.shutdown(wait=False, cancel_futures=True)

    def update_parallel_limit(self, new_limit: int) -> None:
        normalized = max(1, int(new_limit))
        with self._lock:
            if normalized == self._max_parallel:
                return
            delta = normalized - self._max_parallel
            if delta > 0:
                for _ in range(delta):
                    self._semaphore.release()
            else:
                for _ in range(-delta):
                    try:
                        self._semaphore.acquire(blocking=False)
                    except Exception:
                        # Semaphore already at zero available permits; rely on
                        # running tasks to release naturally.
                        break
            self._max_parallel = normalized

    def schedule_download(
        self,
        model_id: str,
        backend: str,
        cache_dir: str,
        quant: Optional[str],
        *,
        timeout: Optional[float] = None,
    ) -> DownloadTask:
        task = DownloadTask(
            task_id=uuid.uuid4().hex,
            model_id=model_id,
            backend=backend,
            cache_dir=cache_dir,
            quant=quant,
            timeout=timeout,
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._task_order.append(task.task_id)
            self._publish(task, message="Queued")
            future = self._executor.submit(self._task_wrapper, task.task_id)
            self._futures[task.task_id] = future
        return task

    def cancel(self, task_id: str, *, by_user: bool = True) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return False
        task.pause_requested = False
        task.cancel_event.set()
        task.message = "Cancellation requested by user" if by_user else "Cancellation requested"
        task.status = "cancelling"
        self._publish(task)
        return True

    def pause(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return False
        task.pause_requested = True
        task.cancel_event.set()
        task.message = "Pause requested"
        task.status = "pausing"
        self._publish(task)
        return True

    def resume(self, task_id: str) -> Optional[DownloadTask]:
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return None
        if task.status not in {"paused", "cancelled"}:
            return task
        task.pause_requested = False
        task.cancel_event = threading.Event()
        task.status = "queued"
        task.stage = "queued"
        task.bytes_done = 0
        task.bytes_total = 0
        task.started_at = None
        task.finished_at = None
        task.message = "Rescheduled"
        self._publish(task)
        future = self._executor.submit(self._task_wrapper, task.task_id)
        with self._lock:
            self._futures[task.task_id] = future
        return task

    def snapshot(self) -> dict:
        with self._lock:
            ordered = [self._tasks[tid].snapshot() for tid in self._task_order if tid in self._tasks]
        return {"tasks": ordered, "max_parallel_downloads": self._max_parallel}

    def _task_wrapper(self, task_id: str) -> None:
        acquired = False
        try:
            acquired = self._semaphore.acquire(timeout=None)
            if not acquired:
                return
            self._run_task(task_id)
        finally:
            if acquired:
                self._semaphore.release()

    def _run_task(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return
        task.status = "running"
        task.stage = "starting"
        task.started_at = time.time()
        task.cancel_event.clear()
        self._publish(task, message="Starting download")

        def _on_progress(bytes_done: int, bytes_total: int) -> None:
            if bytes_total > 0:
                task.bytes_total = max(task.bytes_total, bytes_total)
            task.bytes_done = max(task.bytes_done, bytes_done)
            now = time.time()
            if task.started_at and task.bytes_done:
                elapsed = now - task.started_at
                if elapsed > 0:
                    task.throughput_bps = task.bytes_done / elapsed
                    remaining = max(task.bytes_total - task.bytes_done, 0)
                    task.eta_seconds = remaining / task.throughput_bps if task.throughput_bps else None
            self._publish(task)

        def _on_stage(stage_id: str, metadata: dict) -> None:
            task.stage = stage_id
            task.metadata.update(metadata or {})
            message = metadata.get("message") if isinstance(metadata, dict) else None
            if stage_id == "size_estimated" and isinstance(metadata, dict):
                estimated_bytes = int(metadata.get("estimated_bytes") or 0)
                if estimated_bytes:
                    task.bytes_total = estimated_bytes
            if stage_id == "download_start":
                task.target_dir = metadata.get("path")
            if stage_id == "success":
                if metadata.get("bytes_downloaded"):
                    task.bytes_done = int(metadata.get("bytes_downloaded"))
                    task.bytes_total = max(task.bytes_total, task.bytes_done)
                throughput = metadata.get("throughput_bps")
                if isinstance(throughput, (float, int)):
                    task.throughput_bps = float(throughput)
            if message:
                task.message = str(message)
            self._publish(task)

        try:
            result = model_manager.ensure_download(
                task.model_id,
                task.backend,
                task.cache_dir,
                task.quant,
                timeout=task.timeout,
                cancel_event=task.cancel_event,
                on_progress=_on_progress,
                on_stage_change=_on_stage,
            )
        except model_manager.DownloadCancelledError as exc:
            task.finished_at = time.time()
            task.error = exc
            by_user = getattr(exc, "by_user", False)
            timed_out = getattr(exc, "timed_out", False)
            if task.pause_requested and not timed_out:
                task.status = "paused"
                task.message = "Download paused"
            else:
                task.status = "cancelled" if by_user or not timed_out else "timed_out"
                task.message = str(exc) or ("Timed out" if timed_out else "Cancelled")
            self._publish(task)
            self._finalize(task)
            return
        except Exception as exc:  # pragma: no cover - defensive
            task.finished_at = time.time()
            task.status = "error"
            task.error = exc
            task.message = str(exc)
            self._publish(task)
            self._finalize(task)
            return

        task.finished_at = time.time()
        task.result = result
        task.status = "skipped" if not result.downloaded else "completed"
        task.message = "Model already present" if task.status == "skipped" else "Download finished"
        if result.target_dir:
            task.target_dir = result.target_dir
        if result.bytes_downloaded:
            task.bytes_done = int(result.bytes_downloaded)
            task.bytes_total = max(task.bytes_total, task.bytes_done)
        if result.duration_seconds and result.bytes_downloaded:
            duration = max(result.duration_seconds, 1e-6)
            task.throughput_bps = result.bytes_downloaded / duration
        self._publish(task)
        self._finalize(task)

    def _publish(self, task: DownloadTask, message: Optional[str] = None) -> None:
        if message:
            task.message = message
        with self._lock:
            details = task.snapshot()
            try:
                queue_position = self._task_order.index(task.task_id)
            except ValueError:
                queue_position = None
            details["queue_position"] = queue_position
            tasks_snapshot = [
                self._tasks[tid].snapshot()
                for tid in self._task_order
                if tid in self._tasks
            ]
        self._state_manager.set_state(
            sm.StateEvent.MODEL_DOWNLOAD_PROGRESS,
            details={
                **details,
                "message": task.message,
                "tasks": tasks_snapshot,
            },
            source="model_download",
        )

    def _finalize(self, task: DownloadTask) -> None:
        if self._on_task_finished:
            try:
                self._on_task_finished(task)
            except Exception:  # pragma: no cover - observer safety
                LOGGER.debug("Task finished callback failed for %s", task.task_id, exc_info=True)

        bytes_downloaded = None
        throughput = None
        target_dir = task.target_dir
        duration_seconds = None
        if task.result and task.result.bytes_downloaded is not None:
            bytes_downloaded = int(task.result.bytes_downloaded)
            target_dir = task.result.target_dir or target_dir
            if task.result.duration_seconds:
                duration = max(task.result.duration_seconds, 1e-6)
                throughput = bytes_downloaded / duration
                duration_seconds = float(task.result.duration_seconds)
        elif task.bytes_done:
            bytes_downloaded = task.bytes_done
            throughput = task.throughput_bps
        if duration_seconds is None and task.started_at and task.finished_at:
            duration_seconds = max(task.finished_at - task.started_at, 0.0)
        status_value = task.status
        if status_value == "completed":
            status_value = "success"
        elif status_value == "timed_out":
            status_value = "timeout"
        try:
            self._config_manager.record_model_download_status(
                status=status_value,
                model_id=task.model_id,
                backend=task.backend,
                message=task.message,
                details=target_dir or "",
                target_dir=target_dir,
                bytes_downloaded=bytes_downloaded,
                throughput_bytes_per_sec=throughput,
                duration_seconds=duration_seconds,
                task_id=task.task_id,
            )
        except Exception:  # pragma: no cover - persistence best effort
            LOGGER.debug(
                "Failed to persist download status for task %s", task.task_id, exc_info=True
            )
        LOGGER.info(
            log_context(
                "Model download task finalized.",
                event="model.download_controller.task_finalized",
                task_id=task.task_id,
                status=task.status,
                model=task.model_id,
                backend=task.backend,
            )
        )
