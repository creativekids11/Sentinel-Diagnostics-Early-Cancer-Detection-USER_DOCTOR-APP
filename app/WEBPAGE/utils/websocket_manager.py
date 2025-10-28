from flask_socketio import SocketIO, emit, join_room, leave_room
from functools import wraps
from flask import request
from flask_login import current_user

socketio = SocketIO()
reports_clients = set()

def authenticated_only(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            disconnect()
        else:
            return f(*args, **kwargs)
    return wrapped

@socketio.on('connect', namespace='/ws/reports')
@authenticated_only
def handle_reports_connect():
    reports_clients.add(request.sid)
    join_room(f"user_{current_user.id}")
    print(f"Client {request.sid} connected to reports websocket")

@socketio.on('disconnect', namespace='/ws/reports')
def handle_reports_disconnect():
    reports_clients.remove(request.sid)
    print(f"Client {request.sid} disconnected from reports websocket")

def broadcast_report_update(reports_data):
    """Broadcast report updates to all connected clients"""
    socketio.emit('report_update', {
        'type': 'report_update',
        'reports': reports_data
    }, namespace='/ws/reports', broadcast=True)