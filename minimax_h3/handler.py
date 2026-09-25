import runpod

from processing.generate_video import run_image_to_video


def handler(job):
    """RunPod Serverless entry point. `job["input"]` mirrors the kwargs
    already sent to the `image_to_video` Celery task (see celery_worker.py)
    — same payload shape, different transport, so the aafactory_nsfw
    backend can dispatch to either without changing what it sends.

    aafactory_nsfw's runpod_serverless.py always folds a `task_name` key
    into the RunPod job input (submit_job's `{**payload, "task_name":
    task_name}`), even for a single-task server like this one — it's only
    meaningful for a server with more than one task type (see
    qwen_chat/qwen_image's handler.py, which dispatch on it). Popped and
    discarded here so it isn't forwarded as an unexpected kwarg.
    """
    job_input = dict(job["input"])
    job_input.pop("task_name", None)

    # Surfaced back through RunPod's own /status polling — see
    # aafactory_nsfw's backend/runpod_serverless.py for the reader side.
    def _report_progress(step, total_steps):
        runpod.serverless.progress_update(
            job, {"step": step, "total_steps": total_steps}
        )

    result = run_image_to_video(**job_input, progress_callback=_report_progress)
    return {"video_base64": result.decode("utf-8")}


runpod.serverless.start({"handler": handler})
