from flask import Flask
from .config import Config
from .errors import register_error_handlers
from .socketio_instance import socketio  # 🔥 import your shared instance

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    from .summary.routes import summary_bp
    from .grammar.routes import grammar_bp
    from .ocr.routes import ocr_bp

    register_error_handlers(app)

    app.register_blueprint(summary_bp)
    app.register_blueprint(grammar_bp)
    app.register_blueprint(ocr_bp)

    socketio.init_app(app)  # 🔌 initialize here

    return app