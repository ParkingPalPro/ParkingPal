from . import api_bp
from flask import request, jsonify, send_file
from server.parkingpal.models import CameraConfig, ParkingSpace
from server.parkingpal.extensions import db
from datetime import datetime, UTC
import threading
import io
import json

# Store current camera snapshots (one per camera)
camera_snapshots = {}  # {camera_id: {data, timestamp}}
snapshot_lock = threading.Lock()

# Store parking space configurations per camera
parking_configs = {}
config_lock = threading.Lock()


@api_bp.route('/cameras', methods=['GET'])
def get_cameras():
    """Get list of registered cameras"""
    cameras = set()

    # Get cameras from config database
    db_camera_ids = (
        db.session.query(CameraConfig.camera_id)
        .distinct()
        .all()
    )

    for row in db_camera_ids:
        cameras.add(row.camera_id)

    # Get cameras from active snapshots
    with snapshot_lock:
        for camera_id in camera_snapshots.keys():
            cameras.add(camera_id)

    return jsonify({'cameras': sorted(list(cameras))})


@api_bp.route('/camera/<camera_id>/snapshot', methods=['GET', 'POST'])
def camera_snapshot(camera_id):
    """Get or update camera snapshot"""

    if request.method == 'POST':
        # Camera uploads snapshot
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        image_data = file.read()

        with snapshot_lock:
            camera_snapshots[camera_id] = {
                'data': image_data,
                'timestamp': datetime.now().isoformat()
            }

        print(f"✓ Snapshot received from {camera_id}")
        return jsonify({'status': 'success', 'message': 'Snapshot uploaded'})

    else:
        # Admin requests snapshot
        with snapshot_lock:
            if camera_id not in camera_snapshots:
                return jsonify({'error': f'No snapshot available for {camera_id}'}), 404

            snapshot = camera_snapshots[camera_id]

        return send_file(
            io.BytesIO(snapshot['data']),
            mimetype='image/jpeg'
        )


@api_bp.route('/camera/<camera_id>/config', methods=['GET', 'POST'])
def camera_config(camera_id):
    """
    Get or update camera configuration.
    Now includes manual space number assignment with validation.
    """

    if request.method == 'POST':
        # Admin saves configuration
        data = request.json
        spaces = data.get('spaces', [])

        # Validate space numbers
        validation_result = _validate_space_numbers(spaces, camera_id)
        if not validation_result['valid']:
            return jsonify({
                'status': 'error',
                'message': validation_result['message'],
                'conflicts': validation_result.get('conflicts', [])
            }), 400

        config_json = json.dumps({
            'spaces': spaces,
            'image_width': data.get('image_width'),
            'image_height': data.get('image_height')
        })

        # Check if record exists
        existing = CameraConfig.query.get(camera_id)

        if existing:
            # Update existing
            existing.config_json = config_json
            existing.image_width = data.get('image_width')
            existing.image_height = data.get('image_height')
            existing.last_updated = datetime.now(UTC)
        else:
            # Insert new
            new_config = CameraConfig(
                camera_id=camera_id,
                config_json=config_json,
                image_width=data.get('image_width'),
                image_height=data.get('image_height'),
                last_updated=datetime.now(UTC),
            )
            db.session.add(new_config)

        db.session.commit()

        # Register/update parking spaces in database
        _sync_parking_spaces(camera_id, spaces)

        # Store in memory for quick access
        with config_lock:
            parking_configs[camera_id] = spaces

        print(f"✓ Configuration saved for {camera_id}: {len(spaces)} spaces")
        print(f"  Space numbers: {[s['id'] for s in spaces]}")

        return jsonify({
            'status': 'success',
            'message': 'Configuration saved',
            'spaces': spaces
        })

    else:
        # GET – camera or admin requests configuration
        row = CameraConfig.query.get(camera_id)

        if row:
            config = json.loads(row.config_json)
            return jsonify(config)
        else:
            return jsonify({'spaces': []})


def _validate_space_numbers(spaces, current_camera_id):
    """
    Validate space numbers for uniqueness and correctness.
    Returns: {'valid': bool, 'message': str, 'conflicts': list}
    """
    if not spaces:
        return {'valid': True}
    print(spaces)

    # Check for None/null values
    for i, space in enumerate(spaces):
        if space.get('id') is None:
            return {
                'valid': False,
                'message': f'Space at index {i} is missing a space number. Please assign a number to all spaces.'
            }

    # Check for duplicates within the same configuration
    space_numbers = [s['id'] for s in spaces]
    duplicates = [num for num in space_numbers if space_numbers.count(num) > 1]

    if duplicates:
        unique_dupes = list(set(duplicates))
        return {
            'valid': False,
            'message': f'Duplicate space numbers found: {", ".join(map(str, unique_dupes))}. Each space must have a unique number.',
            'conflicts': unique_dupes
        }

    # Check for negative numbers
    negative_numbers = [num for num in space_numbers if num < 0]
    if negative_numbers:
        return {
            'valid': False,
            'message': f'Space numbers must be positive integers (1 or greater). Found: {", ".join(map(str, negative_numbers))}'
        }

    # Check against existing spaces in database (excluding spaces from this camera)
    existing_spaces = ParkingSpace.query.filter(
        ParkingSpace.space_number.in_(space_numbers),
        ParkingSpace.camera_id != current_camera_id
    ).all()

    if existing_spaces:
        conflicts = [
            {
                'space_number': s.space_number,
                'camera_id': s.camera_id
            }
            for s in existing_spaces
        ]
        conflict_numbers = [s.space_number for s in existing_spaces]
        conflict_cameras = list(set([s.camera_id for s in existing_spaces]))

        return {
            'valid': False,
            'message': f'Space number(s) {", ".join(map(str, conflict_numbers))} already assigned to camera(s) {", ".join(conflict_cameras)}. Please choose different numbers.',
            'conflicts': conflicts
        }

    return {'valid': True}


def _sync_parking_spaces(camera_id, spaces):
    """
    Synchronize parking spaces in database.
    Creates or updates ParkingSpace records for each polygon.
    """
    timestamp = datetime.now(UTC)

    # Get current space numbers for this camera
    existing_spaces = ParkingSpace.query.filter_by(camera_id=camera_id).all()
    existing_numbers = {s.space_number for s in existing_spaces}
    new_numbers = {s['id'] for s in spaces}

    # Remove spaces that are no longer in config
    removed_numbers = existing_numbers - new_numbers
    for space_num in removed_numbers:
        space = ParkingSpace.query.filter_by(
            camera_id=camera_id,
            space_number=space_num
        ).first()
        if space:
            db.session.delete(space)
            print(f"  Removed space {space_num}")

    # Add or update spaces
    for space_data in spaces:
        space_number = space_data['id']
        if space_number is None:
            return

        existing = ParkingSpace.query.filter_by(space_number=space_number).first()
        print(f"Is existing: {existing}")

        if existing:
            # Update camera_id in case space moved (shouldn't happen but handle it)
            if existing.camera_id != camera_id:
                print(f"  Warning: Space {space_number} moved from {existing.camera_id} to {camera_id}")
            existing.camera_id = camera_id
            existing.last_updated = timestamp
        else:
            # Create new space
            new_space = ParkingSpace(
                camera_id=camera_id,
                space_number=space_number,
                is_free=True,  # Initially free
                last_updated=timestamp
            )
            db.session.add(new_space)
            print(f"  Created space {space_number}")

    db.session.commit()
