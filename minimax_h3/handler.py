import runpod

from processing.generate_video import run_image_to_video


def handler(job):
    """RunPod Serverless entry point. `job["input"]` mirrors the kwargs
    already sent to the `image_to_video` Celery task (see celery_worker.py)
    — same payload shape, different transport, so the aafactory_nsfw
    backend can dispatch to either without changing what it sends.
    """
    job_input = job["input"]

    # Surfaced back through RunPod's own /status polling — see
    # aafactory_nsfw's backend/runpod_serverless.py for the reader side.
    def _report_progress(step, total_steps):
        runpod.serverless.progress_update(
            job, {"step": step, "total_steps": total_steps}
        )

    result = run_image_to_video(**job_input, progress_callback=_report_progress)
    return {"video_base64": result.decode("utf-8")}


runpod.serverless.start({"handler": handler})
