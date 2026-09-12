import runpod

from processing.compute_text_to_video import run_text_to_video

_TASKS = {
    "text_to_video": run_text_to_video,
}


def handler(job):
    """RunPod Serverless entry point. `job["input"]` mirrors the kwargs
    already sent to the matching Celery task in celery_worker.py, plus a
    `task_name` key (added by aafactory_nsfw's runpod_serverless.py). Only
    one task type exists today, but the dict-dispatch keeps this in step
    with z_image/qwen_image's servers so adding e.g. image-to-video later
    is a one-line change rather than a rewrite.
    """
    job_input = dict(job["input"])
    task_name = job_input.pop("task_name")
    run_task = _TASKS[task_name]

    result = run_task(**job_input)
    return {"video_base64": result.decode("utf-8")}


runpod.serverless.start({"handler": handler})
