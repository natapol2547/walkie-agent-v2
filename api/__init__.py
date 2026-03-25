"""Flask application factory."""

from flask import Flask

from api.routes import image_caption, image_embed, object_detection, pose_estimation, stt, tts


def create_app() -> Flask:
    app = Flask(__name__)

    app.register_blueprint(stt.bp)
    app.register_blueprint(tts.bp)
    app.register_blueprint(object_detection.bp)
    app.register_blueprint(pose_estimation.bp)
    app.register_blueprint(image_caption.bp)
    app.register_blueprint(image_embed.bp)

    @app.get("/")
    def index():
        return {
            "service": "walkie-agent-v2",
            "endpoints": [
                "/stt/providers", "/stt/transcribe",
                "/tts/providers", "/tts/synthesize", "/tts/synthesize-stream",
                "/object-detection/providers", "/object-detection/detect",
                "/pose-estimation/providers", "/pose-estimation/estimate",
                "/image-caption/providers", "/image-caption/caption", "/image-caption/caption-batch",
                "/image-embed/providers", "/image-embed/embed-image", "/image-embed/embed-text", "/image-embed/similarity",
            ],
        }

    return app
