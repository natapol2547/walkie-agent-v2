"""STT (Speech-to-Text) blueprint."""

from flask import Blueprint, request

from api.utils import error, extract_provider_config, success
from services.stt import STT

bp = Blueprint("stt", __name__, url_prefix="/stt")


@bp.get("/providers")
def list_providers():
    return success(STT.available_providers())


@bp.post("/transcribe")
def transcribe():
    if "audio" not in request.files:
        return error("Missing 'audio' file in request")

    audio_bytes = request.files["audio"].read()

    form = request.form.to_dict()
    provider, config = extract_provider_config(form)

    kwargs = {"provider": provider} if provider else {}
    kwargs.update(config)

    try:
        stt = STT(**kwargs)
        transcription = stt.transcribe(audio_bytes)
    except Exception as exc:
        return error(str(exc), 500)

    return success({"transcription": transcription})
