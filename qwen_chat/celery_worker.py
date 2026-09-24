from processing.generate_text import run_chat, run_generate_prompt

from celery import Celery
import os


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

app = Celery(
    "qwen_chat_worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)


@app.task(name="chat", queue="qwen_chat", bind=True)
def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
    def _report_progress(step, total_steps):
        self.update_state(state="PROGRESS", meta={"step": step, "total_steps": total_steps})

    return run_chat(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        progress_callback=_report_progress,
    )


@app.task(name="generate_prompt", queue="qwen_chat", bind=True)
def generate_prompt(self, subject: str, style_hints: str = "", max_tokens: int = 256) -> str:
    def _report_progress(step, total_steps):
        self.update_state(state="PROGRESS", meta={"step": step, "total_steps": total_steps})

    return run_generate_prompt(
        subject=subject,
        style_hints=style_hints,
        max_tokens=max_tokens,
        progress_callback=_report_progress,
    )
