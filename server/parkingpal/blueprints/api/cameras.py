from . import api_bp
from flask import request, jsonify, send_file
from server.parkingpal.models import CameraConfig
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
    """Get or update camera configuration"""

    if request.method == 'POST':
        # Admin saves configuration
        data = request.json

        config_json = json.dumps({
            'spaces': data.get('spaces', []),
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

        # Store in memory for quick access
        with config_lock:
            parking_configs[camera_id] = data.get('spaces', [])

        print(f"✓ Configuration saved for {camera_id}: {len(data.get('spaces', []))} spaces")
        return jsonify({'status': 'success', 'message': 'Configuration saved'})

    else:
        # GET – camera or admin requests configuration
        row = CameraConfig.query.get(camera_id)

        if row:
            return jsonify(json.loads(row.config_json))
        else:
            return jsonify({'spaces': []})