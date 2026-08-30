import runpod

from processing.generate_image import run_text_to_image, run_image_to_image_edit

_TASKS = {
    "text_to_image": run_text_to_image,
    "image_to_image_edit": run_image_to_image_edit,
}


def handler(job):
    """RunPod Serverless entry point. `job["input"]` mirrors the kwargs
    already sent to the matching Celery task in celery_worker.py, plus a
    `task_name` key (added by aafactory_nsfw's runpod_serverless.py) that
    tells us which of this server's two task types to run — unlike zonos,
    this server has more than one, so dispatch can't be hardcoded.
    """
    job_input = dict(job["input"])
    task_name = job_input.pop("task_name")
    run_task = _TASKS[task_name]

    result = run_task(**job_input)
    return {"image_base64": result.decode("utf-8")}


runpod.serverless.start({"handler": handler})
