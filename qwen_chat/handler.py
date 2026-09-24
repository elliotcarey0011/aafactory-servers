import runpod

from processing.generate_text import run_chat, run_generate_prompt

_TASKS = {
    "chat": run_chat,
    "generate_prompt": run_generate_prompt,
}


def handler(job):
    """RunPod Serverless entry point. `job["input"]` mirrors the kwargs
    already sent to the matching Celery task in celery_worker.py, plus a
    `task_name` key (added by aafactory_nsfw's runpod_serverless.py) that
    tells us which of this server's two task types to run.
    """
    job_input = dict(job["input"])
    task_name = job_input.pop("task_name")
    run_task = _TASKS[task_name]

    # Surfaced back through RunPod's own /status polling — see
    # aafactory_nsfw's backend/runpod_serverless.py for the reader side.
    def _report_progress(step, total_steps):
        runpod.serverless.progress_update(
            job, {"step": step, "total_steps": total_steps}
        )

    result = run_task(**job_input, progress_callback=_report_progress)
    return {"text": result}


runpod.serverless.start({"handler": handler})
