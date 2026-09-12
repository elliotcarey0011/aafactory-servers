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

    # Surfaced back through RunPod's own /status polling — see
    # aafactory_nsfw's backend/runpod_serverless.py for the reader side.
    # Exact field name RunPod exposes this under in the status response is
    # unverified against public docs; that side is written defensively.
    def _report_progress(step, total_steps):
        runpod.serverless.progress_update(
            job, {"step": step, "total_steps": total_steps}
        )

    result = run_task(**job_input, progress_callback=_report_progress)
    return {"image_base64": result.decode("utf-8")}


runpod.serverless.start({"handler": handler})
