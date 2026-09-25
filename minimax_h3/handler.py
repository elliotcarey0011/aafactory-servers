import runpod

from processing.generate_video import run_image_to_video, run_reference_to_video

_TASKS = {
    "image_to_video": run_image_to_video,
    "reference_to_video": run_reference_to_video,
}


def handler(job):
    """RunPod Serverless entry point. `job["input"]` mirrors the kwargs
    already sent to the matching Celery task in celery_worker.py, plus a
    `task_name` key (added by aafactory_nsfw's runpod_serverless.py) that
    tells us which of this server's two task types to run — see
    qwen_chat/qwen_image's handler.py for the same pattern.
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

    # Fired if generation hits the known transient GPU allocator error (see
    # _call_pipe_with_retry in processing/generate_video.py) and is being
    # retried - surfaced through the same progress channel as
    # _report_progress above so it's visible to RunPod's /status polling
    # instead of only showing up in worker logs.
    def _report_status(message):
        runpod.serverless.progress_update(job, {"status": message})

    result = run_task(
        **job_input, progress_callback=_report_progress, status_callback=_report_status
    )
    return {"video_base64": result.decode("utf-8")}


runpod.serverless.start({"handler": handler})
