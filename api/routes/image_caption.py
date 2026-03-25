"""Image Caption blueprint."""

from flask import Blueprint, request

from api.utils import error, extract_provider_config, image_from_request_file, success
from services.image_caption import ImageCaption

bp = Blueprint("image_caption", __name__, url_prefix="/image-caption")


@bp.get("/providers")
def list_providers():
    return success(ImageCaption.available_providers())


@bp.post("/caption")
def caption():
    if "image" not in request.files:
        return error("Missing 'image' file in request")

    try:
        image = image_from_request_file(request.files["image"])
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    prompt = request.form.get("prompt")
    form = {k: v for k, v in request.form.to_dict().items() if k not in ("prompt",)}
    provider, config = extract_provider_config(form)

    kwargs = {"provider": provider} if provider else {}
    kwargs.update(config)

    try:
        ic = ImageCaption(**kwargs)
        result = ic.caption(image, prompt=prompt)
    except Exception as exc:
        return error(str(exc), 500)

    return success({"caption": result})


@bp.post("/caption-batch")
def caption_batch():
    files = request.files.getlist("images")
    if not files:
        return error("Missing 'images' files in request")

    try:
        images = [image_from_request_file(f) for f in files]
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    prompts_raw = request.form.getlist("prompts")
    prompts = prompts_raw if prompts_raw else None

    form = {k: v for k, v in request.form.to_dict().items() if k not in ("prompts",)}
    provider, config = extract_provider_config(form)

    kwargs = {"provider": provider} if provider else {}
    kwargs.update(config)

    try:
        ic = ImageCaption(**kwargs)
        results = ic.caption_batch(images, prompts=prompts)
    except Exception as exc:
        return error(str(exc), 500)

    return success({"captions": results})
