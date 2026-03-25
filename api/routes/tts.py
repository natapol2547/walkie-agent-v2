"""TTS (Text-to-Speech) blueprint."""

from flask import Blueprint, Response, request, stream_with_context

from api.utils import error, extract_provider_config, success
from services.tts import TTS

bp = Blueprint("tts", __name__, url_prefix="/tts")


@bp.get("/providers")
def list_providers():
    return success(TTS.available_providers())


@bp.post("/synthesize")
def synthesize():
    body = request.get_json(silent=True) or {}
    text = body.get("text") or request.form.get("text")
    if not text:
        return error("Missing 'text' field")

    provider, config = extract_provider_config({k: v for k, v in body.items() if k != "text"})

    kwargs = {"provider": provider} if provider else {}
    kwargs.update(config)

    try:
        tts = TTS(**kwargs)
        audio_bytes = tts.synthesize(text)
    except Exception as exc:
        return error(str(exc), 500)

    content_type = _infer_content_type(tts.get_supported_formats())
    return Response(audio_bytes, mimetype=content_type)


@bp.post("/synthesize-stream")
def synthesize_stream():
    body = request.get_json(silent=True) or {}
    text = body.get("text") or request.form.get("text")
    if not text:
        return error("Missing 'text' field")

    provider, config = extract_provider_config({k: v for k, v in body.items() if k != "text"})

    kwargs = {"provider": provider} if provider else {}
    kwargs.update(config)

    try:
        tts = TTS(**kwargs)
        if not tts.supports_streaming():
            return error("Selected provider does not support streaming", 400)
        content_type = _infer_content_type(tts.get_supported_formats())

        def generate():
            for chunk in tts.synthesize_stream(text):
                yield chunk

    except Exception as exc:
        return error(str(exc), 500)

    return Response(stream_with_context(generate()), mimetype=content_type)


def _infer_content_type(formats: list[str]) -> str:
    fmt_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "pcm": "audio/pcm",
    }
    for fmt in formats:
        mime = fmt_map.get(fmt.lower().split("_")[0])
        if mime:
            return mime
    return "application/octet-stream"
