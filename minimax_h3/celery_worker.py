from processing.generate_video import DEFAULT_NUM_FRAMES, run_image_to_video

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
    seed: int | None = None,
) -> str:
    def _report_progress(step, total_steps):
        self.update_state(state="PROGRESS", meta={"step": step, "total_steps": total_steps})

    result = run_image_to_video(
        image_bytes=image_bytes,
        prompt=prompt,
        num_frames=num_frames,
        seed=seed,
        progress_callback=_report_progress,
    )
    return result.decode("utf-8")
