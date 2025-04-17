from app import create_app
from app.socketio_instance import socketio  # use the shared one, not a new one!

app = create_app()

if __name__ == '__main__':
    socketio.run(app, port=app.config['PORT'], host='0.0.0.0',allow_unsafe_werkzeug=True)