from flask_socketio import SocketIO, disconnect

socketio = SocketIO(cors_allowed_origins="*")


@socketio.on('disconnect')
def handle_disconnect():
    print("⚠️ Client disconnected gracefully")
