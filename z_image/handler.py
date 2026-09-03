import runpod

from processing.generate_image import run_text_to_image

_TASKS = {
    "text_to_image": run_text_to_image,
}


def handler(job):
    """RunPod Serverless entry point. `job["input"]` mirrors the kwargs
    already sent to the matching Celery task in celery_worker.py, plus a
    `task_name` key (added by aafactory_nsfw's runpod_serverless.py). Only
    one task type exists today, but the dict-dispatch keeps this in step
    with qwen_image's server so adding Z-Image-Edit later (once released)
    is a one-line change rather than a rewrite.
    """
    job_input = dict(job["input"])
    task_name = job_input.pop("task_name")
    run_task = _TASKS[task_name]

    def _report_progress(step, total_steps):
        runpod.serverless.progress_update(
            job, {"step": step, "total_steps": total_steps}
        )

    result = run_task(**job_input, progress_callback=_report_progress)
    return {"image_base64": result.decode("utf-8")}


runpod.serverless.start({"handler": handler})
