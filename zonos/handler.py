import runpod

from processing.compute_tts import run_text_to_speech


def handler(job):
    """RunPod Serverless entry point. `job["input"]` mirrors the kwargs
    already sent to the `custom_voice_to_audio` Celery task (see
    celery_worker.py) — same payload shape, different transport, so the
    aafactory_nsfw backend can dispatch to either without changing what it
    sends.
    """
    job_input = job["input"]
    result = run_text_to_speech(
        prompt=job_input["prompt"],
        voice_bytes=job_input["voice_bytes"],
        language=job_input["language"],
    )
    return {"audio_base64": result.decode("utf-8")}


runpod.serverless.start({"handler": handler})
