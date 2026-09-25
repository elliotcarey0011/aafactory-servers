from processing.generate_video import (
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    run_image_to_video,
    run_reference_to_video,
)

from celery import Celery
import os


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

app = Celery(
    "minimax_h3_worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)


@app.task(name="image_to_video", queue="minimax_h3", bind=True)
def image_to_video(
    self,
    image_bytes: str,
    prompt: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    seed: int | None = None,
) -> str:
    def _report_progress(step, total_steps):
        self.update_state(state="PROGRESS", meta={"step": step, "total_steps": total_steps})

    # Fired if generation hits the known transient GPU allocator error (see
    # _call_pipe_with_retry in processing/generate_video.py) and is being
    # retried - a custom Celery state, so callers polling AsyncResult.state
    # see "RETRYING" (with the reason in .info) instead of the task just
    # going quiet mid-run.
    def _report_status(message):
        self.update_state(state="RETRYING", meta={"message": message})

    result = run_image_to_video(
        image_bytes=image_bytes,
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        progress_callback=_report_progress,
        status_callback=_report_status,
    )
    return result.decode("utf-8")


@app.task(name="reference_to_video", queue="minimax_h3", bind=True)
def reference_to_video(
    self,
    reference_images: list[str],
    prompt: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    seed: int | None = None,
) -> str:
    def _report_progress(step, total_steps):
        self.update_state(state="PROGRESS", meta={"step": step, "total_steps": total_steps})

    def _report_status(message):
        self.update_state(state="RETRYING", meta={"message": message})

    result = run_reference_to_video(
        reference_images=reference_images,
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        progress_callback=_report_progress,
        status_callback=_report_status,
    )
    return result.decode("utf-8")
