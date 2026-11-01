"""Persistent vocabulary jobs. Runtime data stays under the ignored outputs tree."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import threading
import uuid

from tools.vocabulary_audio import assemble_audio, prepare_audio, read_mono


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_vocabulary(text):
    items = [line.strip() for line in (text or "").splitlines() if line.strip()]
    warnings = []
    for index, item in enumerate(items, 1):
        if re.search(r"\[(?:停顿|pause|silence)[^\]]*\]|^#{1,6}\s|^\d+[.)、]\s*|^[-*]\s", item, re.I):
            warnings.append(f"第 {index} 项可能含停顿标记、标题或编号，请整理后再生成。")
        if re.search(r"\b(?:sth|sb)\b", item, re.I):
            warnings.append(f"第 {index} 项含 sth/sb，请先改成需要朗读的完整表达。")
    return items, warnings


def audio_options(repeats=2, repeat_gap=0.8, item_gap=1.5, trim=False):
    values = (float(repeats), float(repeat_gap), float(item_gap))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("重复次数和间隔必须是有限数字")
    if values[0] != int(values[0]) or not 1 <= values[0] <= 10:
        raise ValueError("重复次数必须为 1–10 的整数")
    if not all(0 <= x <= 30 for x in values[1:]):
        raise ValueError("间隔必须在 0–30 秒之间")
    return dict(repeats=int(values[0]), repeat_gap=values[1], item_gap=values[2], trim=bool(trim))


class VocabularyReader:
    def __init__(self, model, output_root, model_lock=None, version="2.5"):
        self.model = model
        self.root = Path(output_root).resolve()
        self.model_lock = model_lock if model_lock is not None else threading.Lock()
        self.version = version
        self._locks = {}
        self._locks_guard = threading.Lock()

    def directory(self, task_id):
        if not isinstance(task_id, str) or not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("请先生成词表任务")
        return self.root / task_id

    def _lock(self, task_id):
        self.directory(task_id)
        with self._locks_guard:
            return self._locks.setdefault(task_id, threading.RLock())

    def load(self, task_id):
        with self._lock(task_id):
            return json.loads((self.directory(task_id) / "task.json").read_text(encoding="utf-8"))

    def _save(self, task):
        task["updated_at"] = now()
        directory = self.directory(task["id"])
        temp = directory / "task.tmp.json"
        temp.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(directory / "task.json")

    def create(self, text, reference, duration_factor=1.0, **options):
        if self.version != "2.5":
            raise ValueError("词表朗读目前需要 IndexTTS-2.5")
        items, warnings = parse_vocabulary(text)
        if not items or len(items) > 300 or any(len(item) > 500 for item in items):
            raise ValueError("请输入 1–300 项，每项不超过 500 个字符")
        if warnings:
            raise ValueError("\n".join(warnings))
        options = audio_options(**options)
        duration_factor = float(duration_factor)
        if not math.isfinite(duration_factor) or not 0.5 <= duration_factor <= 2:
            raise ValueError("时长系数必须在 0.5–2 之间")
        if not reference or not Path(reference).is_file():
            raise ValueError("请上传参考录音或加载已有音色预设")
        read_mono(reference)  # Validate before creating a job.
        task_id = uuid.uuid4().hex
        directory = self.directory(task_id)
        directory.mkdir(parents=True)
        reference_name = "reference" + Path(reference).suffix.lower()
        shutil.copyfile(reference, directory / reference_name)
        task = dict(id=task_id, created_at=now(), status="pending", reference=reference_name,
                    model_version=self.version, language="EN", duration_factor=duration_factor,
                    options=options, combined=None, assembly_error=None, exports=[], items=[
                        dict(text=item, status="pending", revision=0, raw=None, processed=None,
                             error=None, history=[]) for item in items])
        self._save(task)
        return task_id

    def run(self, task_id, indices=None, progress=None):
        """Yield snapshots before/after each item, preserving failures and old revisions."""
        with self._lock(task_id):
            task = self.load(task_id)
            if indices is None:
                indices = [i for i, item in enumerate(task["items"]) if item["status"] != "ready"]
            indices = list(indices)
            if not indices or any(type(i) is not int or not 0 <= i < len(task["items"]) for i in indices):
                raise ValueError("没有待重试的项目，或项目编号无效")
            task["combined"] = None
            task["assembly_error"] = None
            task["status"] = "running"
            self._save(task)
            try:
                for done, index in enumerate(indices):
                    item = task["items"][index]
                    item["revision"] += 1
                    item.update(status="running", error=None, processed=None)
                    self._save(task)
                    yield deepcopy(task)
                    relative = f"items/{index + 1:04d}/r{item['revision']:03d}.raw.wav"
                    raw = self.directory(task_id) / relative
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    attempt = dict(at=now(), revision=item["revision"], raw=relative)
                    try:
                        with self.model_lock:
                            self.model.gr_progress = progress
                            self.model.infer(
                                spk_audio_prompt=str(self.directory(task_id) / task["reference"]),
                                text=item["text"], output_path=str(raw), lang=task["language"],
                                duration_factor=task["duration_factor"],
                                max_text_tokens_per_segment=120,
                            )
                        read_mono(raw)  # A return value alone is not a successful generation.
                        item.update(status="ready", raw=relative)
                        attempt["status"] = "ready"
                    except Exception as error:
                        item.update(status="failed", error=f"{type(error).__name__}: {error}")
                        attempt.update(status="failed", error=item["error"])
                    item["history"].append(attempt)
                    self._save(task)
                    if progress:
                        progress((done + 1, len(indices)), desc=f"{index + 1}. {item['text']}")
                    yield deepcopy(task)
                if all(item["status"] == "ready" for item in task["items"]):
                    try:
                        task = self.reassemble(task_id, **task["options"])
                    except Exception:
                        task = self.load(task_id)
                else:
                    task["status"] = "incomplete"
                    self._save(task)
                yield deepcopy(task)
            finally:
                if task["status"] == "running":
                    for item in task["items"]:
                        if item["status"] == "running":
                            item.update(status="failed", error="任务中断，请重试此项")
                    task["status"] = "incomplete"
                    self._save(task)

    def reassemble(self, task_id, **options):
        options = audio_options(**options)
        with self._lock(task_id):
            task = self.load(task_id)
            if any(item["status"] != "ready" for item in task["items"]):
                raise ValueError("仍有未完成或失败项目，请重试成功后再拼接")
            task.update(combined=None, status="assembling", options=options, assembly_error=None)
            self._save(task)
            directory = self.directory(task_id)
            export_id = uuid.uuid4().hex
            paths = []
            rate = None
            try:
                for index, item in enumerate(task["items"]):
                    processed = f"items/{index + 1:04d}/r{item['revision']:03d}-{export_id}.wav"
                    rate, _ = prepare_audio(directory / item["raw"], directory / processed,
                                            trim=options["trim"], sample_rate=rate)
                    item["processed"] = processed
                    paths.append(directory / processed)
                combined = f"mix-{export_id}.wav"
                stats = assemble_audio(paths, directory / combined, **{
                    name: options[name] for name in ("repeats", "repeat_gap", "item_gap")})
                task.update(status="complete", combined=combined)
                task["exports"].append(dict(at=now(), file=combined, options=options, **stats))
                self._save(task)
                return task
            except Exception as error:
                task.update(status="assembly_failed", combined=None,
                            assembly_error=f"{type(error).__name__}: {error}")
                self._save(task)
                raise

    def audio_path(self, task, relative):
        path = self.directory(task["id"]) / relative if relative else None
        return str(path) if path is not None and path.is_file() else None
