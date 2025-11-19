from . import api_bp
from flask import request, jsonify
from server.parkingpal.models import ParkingSession
from server.parkingpal.extensions import db
from datetime import datetime, UTC
from server.parkingpal.utils import calculate_duration

@api_bp.route("/register_plate", methods=["POST"])
def handle_plate_event():
    """Handle plate detection events from cameras"""
    data = request.json
    camera_id = data.get("camera_id")
    role = data.get("role")
    hashed_plate = data.get("hashed_plate")
    timestamp = data.get("timestamp")

    if not all([camera_id, role, hashed_plate, timestamp]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        if role == "entrance":
            # Check for active session
            existing = ParkingSession.query.filter_by(car_id_hash=hashed_plate, timestamp_end=None).first()
            if existing:
                return jsonify({"status": "warning", "message": "Vehicle already has active parking session"}), 200

            # Create new parking session
            session_entry = ParkingSession(
                car_id_hash=hashed_plate,
                timestamp_start=datetime.fromisoformat(timestamp),
            )
            db.session.add(session_entry)
            db.session.commit()

            return jsonify({"status": "success", "message": "Vehicle entered - Session started", "session_id": session_entry.car_id_hash}), 200

        elif role == "exit":
            session_entry = ParkingSession.query.filter_by(car_id_hash=hashed_plate, timestamp_end=None).first()
            if not session_entry:
                return jsonify({"status": "warning", "message": "No active parking session found"}), 200

            session_entry.timestamp_end = datetime.fromisoformat(timestamp)
            db.session.commit()

            duration = calculate_duration(session_entry.timestamp_start.isoformat(), timestamp)

            return jsonify({"status": "success", "message": f"Vehicle exited - Duration: {duration} min", "duration_minutes": duration}), 200

        else:
            return jsonify({"error": "Invalid camera role"}), 400

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500



@api_bp.route("/active_sessions", methods=["GET"])
def get_active_sessions():
    sessions = ParkingSession.query.filter_by(timestamp_end=None).order_by(ParkingSession.timestamp_start.desc()).all()
    result = [{
        "car_id_hash": s.car_id_hash,
        "timestamp_start": s.timestamp_start.isoformat(),
    } for s in sessions]
    return jsonify({"active_sessions": result}), 200


@api_bp.route("/session_history", methods=["GET"])
def get_session_history():
    limit = request.args.get("limit", 50, type=int)
    sessions = ParkingSession.query.order_by(ParkingSession.timestamp_start.desc()).limit(limit).all()
    result = [{
        "car_id_hash": s.car_id_hash,
        "timestamp_start": s.timestamp_start.isoformat() if s.timestamp_start else None,
        "timestamp_end": s.timestamp_end.isoformat() if s.timestamp_end else None
    } for s in sessions]
    return jsonify({"sessions": result}), 200


@api_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}), 200