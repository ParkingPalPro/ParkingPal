from . import api_bp
from flask import jsonify, request
from datetime import datetime, UTC
from server.parkingpal.extensions import db
from server.parkingpal.models import ParkingSpace


@api_bp.route('/parking/update', methods=['POST'])
def update_parking_status():
    data = request.json
    camera_id = data.get('camera_id', 'default')
    spaces = data.get('spaces', [])

    if not spaces:
        return jsonify({"error": "No spaces provided"}), 400

    timestamp = datetime.now(UTC)

    try:
        for space in spaces:
            space_number = space["space_number"]
            is_free = space["is_free"]

            # Try to fetch existing row
            existing = ParkingSpace.query.filter_by(
                camera_id=camera_id,
                space_number=space_number
            ).first()

            if existing:
                # Update
                existing.is_free = is_free
                existing.last_updated = timestamp
            else:
                # Insert new
                new_space = ParkingSpace(
                    camera_id=camera_id,
                    space_number=space_number,
                    is_free=is_free,
                    last_updated=timestamp
                )
                db.session.add(new_space)

        db.session.commit()

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

    return jsonify({'status': 'success', 'updated': len(spaces)})


@api_bp.route('/parking/status', methods=['GET'])
def get_parking_status():

    spaces = ParkingSpace.query.all()

    result = [{
        "id": s.space_number,
        "status": "available" if s.is_free else "occupied",
        "last_updated": s.last_updated.isoformat()
    } for s in spaces]

    return jsonify({
        "total": len(spaces),
        "available": sum(1 for s in spaces if s.is_free),
        "occupied": sum(1 for s in spaces if not s.is_free),
        "spaces": result
    })

