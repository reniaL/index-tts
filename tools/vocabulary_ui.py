"""Gradio vocabulary tab; shares the existing model and inference queue."""

from pathlib import Path

import gradio as gr

from indextts.utils.presets import list_presets, load_preset
from tools.vocabulary_reader import VocabularyReader, parse_vocabulary


def build_vocabulary_tab(model, model_lock, version="2.5"):
    reader = VocabularyReader(model, Path("outputs/vocabulary"), model_lock, version)

    def inspect(text):
        items, warnings = parse_vocabulary(text)
        return f"共 {len(items)} 项（保留重复项和顺序）。" + ("\n\n" + "\n\n".join(warnings) if warnings else "")

    def view(task, selected=None):
        items = task["items"]
        selected = 0 if selected is None else int(selected)
        choices = [(f"{i + 1}. {item['text']}", str(i)) for i, item in enumerate(items)]
        ready = sum(item["status"] == "ready" for item in items)
        active = next((f"正在生成第 {i + 1} 项：{item['text']}" for i, item in enumerate(items)
                       if item["status"] == "running"), "")
        labels = {"pending": "待生成", "running": "生成中", "ready": "已生成", "failed": "失败"}
        rows = [[i + 1, item["text"], labels[item["status"]], item["revision"], item["error"] or ""]
                for i, item in enumerate(items)]
        combined = reader.audio_path(task, task["combined"])
        item = items[selected]
        processed = reader.audio_path(task, item["processed"]) if item["status"] == "ready" else None
        raw = reader.audio_path(task, item["raw"]) if item["status"] == "ready" else None
        if task["status"] == "complete":
            stats = task["exports"][-1]
            message = f"已完成 {ready}/{len(items)} 项；整份音频 {stats['seconds']:.2f} 秒。"
        elif task["status"] == "running":
            message = f"已生成 {ready}/{len(items)} 项。{active}"
        elif task["status"] == "assembly_failed":
            message = f"音频拼接失败：{task['assembly_error']}。可修正后重新拼接，无需重新推理。"
        else:
            message = f"已生成 {ready}/{len(items)} 项。任务未完成，请查看错误并重试。"
        location = str(reader.directory(task["id"]))
        return (task["id"], message, rows, gr.update(choices=choices, value=str(selected)),
                processed or raw, raw, combined, combined, location)

    def generate(text, reference, speed, repeats, repeat_gap, item_gap, trim, progress=gr.Progress()):
        try:
            task_id = reader.create(text, reference, duration_factor=speed, repeats=repeats,
                                    repeat_gap=repeat_gap, item_gap=item_gap, trim=trim)
            for task in reader.run(task_id, progress=progress):
                yield view(task)
        except (ValueError, OSError) as error:
            raise gr.Error(str(error)) from error

    def regenerate(task_id, selected, progress=gr.Progress()):
        try:
            if selected is None:
                raise ValueError("请先选择一个项目")
            for task in reader.run(task_id, [int(selected)], progress=progress):
                yield view(task, selected)
        except (ValueError, OSError) as error:
            raise gr.Error(str(error)) from error

    def retry(task_id, progress=gr.Progress()):
        try:
            for task in reader.run(task_id, progress=progress):
                yield view(task)
        except (ValueError, OSError) as error:
            raise gr.Error(str(error)) from error

    def reassemble(task_id, repeats, repeat_gap, item_gap, trim, selected):
        try:
            task = reader.reassemble(task_id, repeats=repeats, repeat_gap=repeat_gap,
                                     item_gap=item_gap, trim=trim)
            return view(task, selected)
        except Exception as error:
            if task_id:
                try:
                    task = reader.load(task_id)
                except (ValueError, OSError):
                    raise gr.Error(str(error)) from error
                if task["status"] == "assembly_failed":
                    gr.Warning(str(error))
                    return view(task, selected)
            raise gr.Error(str(error)) from error

    def preview(task_id, selected):
        if not task_id or selected is None:
            return None, None
        task = reader.load(task_id)
        item = task["items"][int(selected)]
        if item["status"] != "ready":
            return None, None
        raw = reader.audio_path(task, item["raw"])
        return reader.audio_path(task, item["processed"]) or raw, raw

    def use_preset(name):
        data = load_preset(name) if name else None
        path = data.get("prompt_audio") if data else None
        if not path or not Path(path).is_file():
            raise gr.Error("此预设没有可用的参考录音，请上传录音或选择其他预设")
        return path

    with gr.Tab("词表朗读"):
        gr.Markdown("每行一项，支持单词、短语和句子，按原顺序朗读。参考录音用于整份词表；默认每项两遍。")
        if version != "2.5":
            gr.Markdown("此功能需要以 IndexTTS-2.5 启动。")
        task_id = gr.State("")
        with gr.Row():
            reference = gr.Audio(label="参考录音", sources=["upload", "microphone"], type="filepath")
            with gr.Column():
                preset = gr.Dropdown(label="复用已有音色预设", choices=list_presets())
                gr.Markdown("预设仅复用参考录音。词表使用英语和参考录音自身的情感。")
                refresh = gr.Button("刷新预设列表")
                apply = gr.Button("加载预设参考录音")
        text = gr.Textbox(label="词表（每行一项）", lines=8, placeholder="ignore\ngap\nindependent\nonly child")
        count = gr.Markdown("共 0 项。")
        with gr.Row():
            repeats = gr.Slider(1, 10, value=2, step=1, label="每项重复次数")
            repeat_gap = gr.Slider(0, 10, value=0.8, step=0.05, label="同项重复间隔（秒）")
            item_gap = gr.Slider(0, 10, value=1.5, step=0.05, label="不同项间隔（秒）")
        speed = gr.Slider(0.5, 2, value=1, step=0.05, label="时长系数（大于 1 放慢）")
        trim = gr.Checkbox(value=False, label="保守裁剪首尾空白（保留 80 毫秒边缘余量，需试听）")
        gr.Markdown("修改词表、参考录音或时长系数后，请生成新任务。修改重复、间隔或裁剪设置后，可直接重新拼接。")
        generate_button = gr.Button("生成新词表任务", variant="primary", interactive=version == "2.5")
        status = gr.Textbox(label="任务状态", interactive=False)
        table = gr.Dataframe(headers=["序号", "词条", "状态", "生成版本", "错误"],
                             datatype=["number", "str", "str", "number", "str"], interactive=False)
        selected = gr.Dropdown(label="选择项目试听或重生成", choices=[], value=None)
        with gr.Row():
            item_audio = gr.Audio(label="单项试听（当前处理版本）", type="filepath")
            raw_audio = gr.Audio(label="原始音频（供裁剪对照）", type="filepath")
        with gr.Row():
            regenerate_button = gr.Button("仅重生成选中项")
            retry_button = gr.Button("重试所有未完成项")
            reassemble_button = gr.Button("按当前设置重新拼接")
        final_audio = gr.Audio(label="整份 WAV 试听", type="filepath")
        download = gr.File(label="下载整份 WAV")
        location = gr.Textbox(label="本机任务保存目录", interactive=False)

        outputs = [task_id, status, table, selected, item_audio, raw_audio, final_audio, download, location]
        text.change(inspect, inputs=[text], outputs=[count], queue=False, api_name=False)
        refresh.click(lambda: gr.update(choices=list_presets(), value=None), outputs=[preset], api_name=False)
        apply.click(use_preset, inputs=[preset], outputs=[reference], api_name=False)
        selected.change(preview, inputs=[task_id, selected], outputs=[item_audio, raw_audio], api_name=False)
        generate_button.click(generate, inputs=[text, reference, speed, repeats, repeat_gap, item_gap, trim],
                              outputs=outputs, concurrency_id="tts-inference", concurrency_limit=1,
                              api_name="vocabulary_generate")
        regenerate_button.click(regenerate, inputs=[task_id, selected], outputs=outputs,
                                concurrency_id="tts-inference", concurrency_limit=1,
                                api_name="vocabulary_regenerate")
        retry_button.click(retry, inputs=[task_id], outputs=outputs,
                           concurrency_id="tts-inference", concurrency_limit=1, api_name="vocabulary_retry")
        reassemble_button.click(reassemble, inputs=[task_id, repeats, repeat_gap, item_gap, trim, selected],
                                outputs=outputs, api_name="vocabulary_reassemble")
    return reader
