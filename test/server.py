from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
from datetime import datetime
import sqlite3
import json
import io
from PIL import Image
import threading
import time

app = Flask(__name__)
CORS(app)

# Email Configuration - UPDATE THESE!
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"  # Use App Password for Gmail
FROM_EMAIL = "parking@yourdomain.com"

# Store current camera snapshots (one per camera)
camera_snapshots = {}  # {camera_id: {data, timestamp}}
snapshot_lock = threading.Lock()

# Store parking space configurations per camera
parking_configs = {}
config_lock = threading.Lock()


def init_db():
    """Initialize database"""
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()

    # Parking sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS parking_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hashed_plate TEXT NOT NULL,
            original_plate TEXT,
            user_email TEXT,
            entrance_time TEXT NOT NULL,
            exit_time TEXT,
            entrance_camera TEXT,
            exit_camera TEXT,
            status TEXT DEFAULT 'active'
        )
    ''')

    # Parking space status table
    c.execute('''
        CREATE TABLE IF NOT EXISTS parking_spaces (
            camera_id TEXT NOT NULL,
            space_number INTEGER NOT NULL,
            is_free BOOLEAN NOT NULL,
            last_updated TEXT NOT NULL,
            PRIMARY KEY (camera_id, space_number)
        )
    ''')

    # Camera configurations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS camera_configs (
            camera_id TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            image_width INTEGER,
            image_height INTEGER,
            last_updated TEXT NOT NULL
        )
    ''')

    # User plates table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_plates (
            hashed_plate TEXT PRIMARY KEY,
            original_plate TEXT NOT NULL,
            user_email TEXT NOT NULL,
            registered_date TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


# HTML Template for Admin Interface
ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parking Space Manager</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .content {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
            padding: 20px;
        }

        .canvas-section {
            background: #f5f5f5;
            border-radius: 8px;
            padding: 20px;
        }

        .canvas-container {
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #parkingCanvas {
            display: block;
            width: 100%;
            height: auto;
            cursor: crosshair;
        }

        .controls {
            background: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
        }

        .control-group {
            margin-bottom: 20px;
        }

        .control-group h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .btn {
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-success {
            background: #10b981;
            color: white;
        }

        .btn-success:hover {
            background: #059669;
        }

        .btn-warning {
            background: #f59e0b;
            color: white;
        }

        .btn-warning:hover {
            background: #d97706;
        }

        .btn-danger {
            background: #ef4444;
            color: white;
        }

        .btn-danger:hover {
            background: #dc2626;
        }

        .btn-secondary {
            background: #6b7280;
            color: white;
        }

        .btn-secondary:hover {
            background: #4b5563;
        }

        .status-box {
            background: white;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }

        .status-box h4 {
            color: #667eea;
            margin-bottom: 8px;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #e5e7eb;
        }

        .status-item:last-child {
            border-bottom: none;
        }

        .space-list {
            max-height: 300px;
            overflow-y: auto;
            background: white;
            border-radius: 6px;
            padding: 10px;
        }

        .space-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px;
            margin: 4px 0;
            background: #f3f4f6;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .space-item:hover {
            background: #e5e7eb;
            transform: translateX(5px);
        }

        .space-item.selected {
            background: #dbeafe;
            border-left: 3px solid #3b82f6;
        }

        .instructions {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }

        .instructions h4 {
            color: #92400e;
            margin-bottom: 8px;
        }

        .instructions ul {
            margin-left: 20px;
            color: #78350f;
        }

        .instructions li {
            margin: 4px 0;
        }

        .camera-select {
            width: 100%;
            padding: 10px;
            border: 2px solid #e5e7eb;
            border-radius: 6px;
            font-size: 14px;
            margin-bottom: 15px;
            background: white;
        }

        .camera-select option.active {
            background-color: #d1fae5;
            font-weight: bold;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #6b7280;
        }

        .spinner {
            border: 4px solid #f3f4f6;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 1024px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🅿️ Parking Space Manager</h1>
            <p>Define and manage parking spaces with interactive polygon editor</p>
        </div>

        <div class="content">
            <div class="canvas-section">
                <div class="canvas-container" id="canvasContainer">
                    <div class="loading">
                        <p>Select a camera to begin</p>
                    </div>
                </div>
            </div>

            <div class="controls">
                <div class="control-group">
                    <h3>Camera Selection</h3>
                    <select id="cameraSelect" class="camera-select">
                        <option value="">Select Camera...</option>
                    </select>
                    <button class="btn btn-primary" onclick="refreshSnapshot()">📷 Refresh Snapshot</button>
                    <button class="btn btn-secondary" onclick="loadCameraList()">🔄 Refresh Camera List</button>
                </div>

                <div class="instructions">
                    <h4>Instructions</h4>
                    <ul>
                        <li>Click 4 points to create a parking space</li>
                        <li>Click on a space to select it</li>
                        <li>Press Delete or click Remove to delete</li>
                        <li>Right-click to cancel current drawing</li>
                    </ul>
                </div>

                <div class="control-group">
                    <h3>Actions</h3>
                    <button class="btn btn-success" onclick="addNewSpace()">➕ New Space</button>
                    <button class="btn btn-warning" onclick="removeSelectedSpace()">🗑️ Remove Selected</button>
                    <button class="btn btn-danger" onclick="clearAllSpaces()">❌ Clear All</button>
                    <button class="btn btn-primary" onclick="saveConfiguration()">💾 Save & Deploy</button>
                </div>

                <div class="status-box">
                    <h4>Status</h4>
                    <div class="status-item">
                        <span>Camera:</span>
                        <strong id="cameraStatus">Not Selected</strong>
                    </div>
                    <div class="status-item">
                        <span>Total Spaces:</span>
                        <strong id="totalSpaces">0</strong>
                    </div>
                    <div class="status-item">
                        <span>Current Points:</span>
                        <strong id="currentPoints">0/4</strong>
                    </div>
                    <div class="status-item">
                        <span>Selected:</span>
                        <strong id="selectedSpace">None</strong>
                    </div>
                </div>

                <div class="control-group">
                    <h3>Parking Spaces</h3>
                    <div class="space-list" id="spaceList"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let canvas = null;
        let ctx = null;

        let cameraImage = null;
        let parkingSpaces = [];
        let currentPoints = [];
        let selectedSpaceIndex = -1;
        let hoveredSpaceIndex = -1;
        let currentCameraId = '';

        // Initialize
        loadCameraList();

        // Auto-refresh camera list every 5 seconds
        setInterval(loadCameraList, 5000);

        function loadCameraList() {
            fetch('/api/cameras')
                .then(r => r.json())
                .then(data => {
                    const select = document.getElementById('cameraSelect');
                    const currentValue = select.value;

                    select.innerHTML = '<option value="">Select Camera...</option>';

                    if (data.cameras.length === 0) {
                        const option = document.createElement('option');
                        option.disabled = true;
                        option.textContent = 'No cameras detected - waiting...';
                        select.appendChild(option);
                    } else {
                        data.cameras.forEach(cam => {
                            const option = document.createElement('option');
                            option.value = cam;
                            option.textContent = `📹 ${cam}`;
                            option.className = 'active';
                            select.appendChild(option);
                        });

                        // Restore previous selection if still available
                        if (currentValue && data.cameras.includes(currentValue)) {
                            select.value = currentValue;
                        } else if (data.cameras.length === 1 && !currentValue) {
                            // Auto-select if only one camera
                            select.value = data.cameras[0];
                            currentCameraId = data.cameras[0];
                            loadCameraSnapshot();
                            loadConfiguration();
                        }
                    }
                })
                .catch(err => console.error('Failed to load cameras:', err));
        }

        document.getElementById('cameraSelect').addEventListener('change', function() {
            currentCameraId = this.value;
            if (currentCameraId) {
                updateStatus();
                loadCameraSnapshot();
                loadConfiguration();
            }
        });

        function loadCameraSnapshot() {
            if (!currentCameraId) return;

            // Show loading indicator
            const container = document.getElementById('canvasContainer');
            container.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading camera snapshot...</p></div>';

            fetch(`/api/camera/${currentCameraId}/snapshot`)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}`);
                    return r.blob();
                })
                .then(blob => {
                    const img = new Image();
                    img.onload = function() {
                        // Restore canvas
                        container.innerHTML = '<canvas id="parkingCanvas"></canvas>';
                        canvas = document.getElementById('parkingCanvas');
                        ctx = canvas.getContext('2d');

                        canvas.width = img.width;
                        canvas.height = img.height;
                        cameraImage = img;

                        // Re-attach event listeners
                        setupCanvasListeners();

                        redraw();

                        console.log(`✓ Snapshot loaded: ${img.width}x${img.height}`);
                    };
                    img.onerror = function() {
                        container.innerHTML = '<div class="loading"><p style="color: #ef4444;">❌ Failed to load image</p></div>';
                    };
                    img.src = URL.createObjectURL(blob);
                })
                .catch(err => {
                    console.error('Snapshot load error:', err);
                    container.innerHTML = `
                        <div class="loading">
                            <p style="color: #ef4444;">❌ Failed to load snapshot</p>
                            <p style="font-size: 12px; color: #6b7280;">
                                Make sure camera "${currentCameraId}" is running and sending snapshots.
                            </p>
                            <button class="btn btn-primary" onclick="refreshSnapshot()" style="margin-top: 10px;">
                                Try Again
                            </button>
                        </div>
                    `;
                });
        }

        function refreshSnapshot() {
            if (!currentCameraId) {
                alert('Please select a camera first');
                return;
            }
            loadCameraSnapshot();
        }

        function loadConfiguration() {
            if (!currentCameraId) return;

            fetch(`/api/camera/${currentCameraId}/config`)
                .then(r => r.json())
                .then(data => {
                    if (data.spaces) {
                        parkingSpaces = data.spaces;
                        redraw();
                        updateSpaceList();
                        updateStatus();
                    }
                })
                .catch(err => console.error('Failed to load config:', err));
        }

        function saveConfiguration() {
            if (!currentCameraId) {
                alert('Please select a camera first');
                return;
            }

            const config = {
                camera_id: currentCameraId,
                spaces: parkingSpaces,
                image_width: canvas ? canvas.width : 0,
                image_height: canvas ? canvas.height : 0
            };

            fetch(`/api/camera/${currentCameraId}/config`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            })
            .then(r => r.json())
            .then(data => {
                alert('Configuration saved and deployed to camera!');
                console.log(data);
            })
            .catch(err => {
                alert('Failed to save configuration');
                console.error(err);
            });
        }

        function addNewSpace() {
            currentPoints = [];
            selectedSpaceIndex = -1;
            redraw();
            updateStatus();
        }

        function removeSelectedSpace() {
            if (selectedSpaceIndex >= 0) {
                parkingSpaces.splice(selectedSpaceIndex, 1);
                selectedSpaceIndex = -1;
                redraw();
                updateSpaceList();
                updateStatus();
            } else {
                alert('Please select a space first');
            }
        }

        function clearAllSpaces() {
            if (confirm('Are you sure you want to clear all parking spaces?')) {
                parkingSpaces = [];
                currentPoints = [];
                selectedSpaceIndex = -1;
                redraw();
                updateSpaceList();
                updateStatus();
            }
        }

        function setupCanvasListeners() {
            if (!canvas) return;

            canvas.addEventListener('click', function(e) {
                const rect = canvas.getBoundingClientRect();
                const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
                const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));

                // Check if clicking on existing space
                const clickedSpace = findSpaceAtPoint(x, y);
                if (clickedSpace >= 0) {
                    selectedSpaceIndex = clickedSpace;
                    currentPoints = [];
                    redraw();
                    updateSpaceList();
                    updateStatus();
                    return;
                }

                // Add point for new space
                if (currentPoints.length < 4) {
                    currentPoints.push([x, y]);

                    if (currentPoints.length === 4) {
                        parkingSpaces.push({
                            id: parkingSpaces.length,
                            points: currentPoints.slice()
                        });
                        currentPoints = [];
                        selectedSpaceIndex = parkingSpaces.length - 1;
                        updateSpaceList();
                    }

                    redraw();
                    updateStatus();
                }
            });

            canvas.addEventListener('contextmenu', function(e) {
                e.preventDefault();
                currentPoints = [];
                redraw();
                updateStatus();
            });

            canvas.addEventListener('mousemove', function(e) {
                const rect = canvas.getBoundingClientRect();
                const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
                const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));

                hoveredSpaceIndex = findSpaceAtPoint(x, y);
                redraw();
            });
        }

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Delete' && selectedSpaceIndex >= 0) {
                removeSelectedSpace();
            }
        });

        function findSpaceAtPoint(x, y) {
            for (let i = parkingSpaces.length - 1; i >= 0; i--) {
                if (isPointInPolygon([x, y], parkingSpaces[i].points)) {
                    return i;
                }
            }
            return -1;
        }

        function isPointInPolygon(point, polygon) {
            let inside = false;
            for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
                const xi = polygon[i][0], yi = polygon[i][1];
                const xj = polygon[j][0], yj = polygon[j][1];

                const intersect = ((yi > point[1]) !== (yj > point[1]))
                    && (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi);
                if (intersect) inside = !inside;
            }
            return inside;
        }

        function redraw() {
            if (!canvas || !ctx) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw camera image
            if (cameraImage) {
                ctx.drawImage(cameraImage, 0, 0);
            }

            // Draw existing parking spaces
            parkingSpaces.forEach((space, index) => {
                const isSelected = index === selectedSpaceIndex;
                const isHovered = index === hoveredSpaceIndex;

                // Fill
                ctx.fillStyle = isSelected ? 'rgba(59, 130, 246, 0.3)' : 
                               isHovered ? 'rgba(251, 191, 36, 0.3)' :
                               'rgba(168, 85, 247, 0.2)';
                ctx.beginPath();
                ctx.moveTo(space.points[0][0], space.points[0][1]);
                for (let i = 1; i < space.points.length; i++) {
                    ctx.lineTo(space.points[i][0], space.points[i][1]);
                }
                ctx.closePath();
                ctx.fill();

                // Border
                ctx.strokeStyle = isSelected ? '#3b82f6' : 
                                 isHovered ? '#f59e0b' :
                                 '#a855f7';
                ctx.lineWidth = isSelected ? 3 : 2;
                ctx.stroke();

                // Points
                space.points.forEach(pt => {
                    ctx.fillStyle = isSelected ? '#3b82f6' : '#a855f7';
                    ctx.beginPath();
                    ctx.arc(pt[0], pt[1], 5, 0, Math.PI * 2);
                    ctx.fill();
                });

                // Label
                const center = getCentroid(space.points);
                ctx.fillStyle = '#fff';
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 3;
                ctx.font = 'bold 20px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.strokeText(index.toString(), center[0], center[1]);
                ctx.fillText(index.toString(), center[0], center[1]);
            });

            // Draw current points being placed
            if (currentPoints.length > 0) {
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);

                ctx.beginPath();
                ctx.moveTo(currentPoints[0][0], currentPoints[0][1]);
                for (let i = 1; i < currentPoints.length; i++) {
                    ctx.lineTo(currentPoints[i][0], currentPoints[i][1]);
                }
                ctx.stroke();
                ctx.setLineDash([]);

                currentPoints.forEach((pt, i) => {
                    ctx.fillStyle = '#10b981';
                    ctx.beginPath();
                    ctx.arc(pt[0], pt[1], 6, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.fillStyle = '#fff';
                    ctx.font = 'bold 12px Arial';
                    ctx.fillText((i + 1).toString(), pt[0], pt[1] - 12);
                });
            }
        }

        function getCentroid(points) {
            let x = 0, y = 0;
            points.forEach(pt => {
                x += pt[0];
                y += pt[1];
            });
            return [x / points.length, y / points.length];
        }

        function updateSpaceList() {
            const list = document.getElementById('spaceList');
            list.innerHTML = '';

            parkingSpaces.forEach((space, index) => {
                const item = document.createElement('div');
                item.className = 'space-item' + (index === selectedSpaceIndex ? ' selected' : '');
                item.innerHTML = `
                    <span>Space ${index}</span>
                    <span>${space.points.length} points</span>
                `;
                item.onclick = () => {
                    selectedSpaceIndex = index;
                    currentPoints = [];
                    redraw();
                    updateSpaceList();
                    updateStatus();
                };
                list.appendChild(item);
            });
        }

        function updateStatus() {
            document.getElementById('cameraStatus').textContent = currentCameraId || 'Not Selected';
            document.getElementById('totalSpaces').textContent = parkingSpaces.length;
            document.getElementById('currentPoints').textContent = `${currentPoints.length}/4`;
            document.getElementById('selectedSpace').textContent = 
                selectedSpaceIndex >= 0 ? `Space ${selectedSpaceIndex}` : 'None';
        }
    </script>
</body>
</html>
"""


@app.route('/')
def admin_interface():
    """Serve the admin interface"""
    return render_template_string(ADMIN_TEMPLATE)


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get list of registered cameras"""
    cameras = set()

    # Get cameras from config database
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()
    c.execute('SELECT DISTINCT camera_id FROM camera_configs')
    for row in c.fetchall():
        cameras.add(row[0])
    conn.close()

    # Get cameras from active snapshots
    with snapshot_lock:
        for camera_id in camera_snapshots.keys():
            cameras.add(camera_id)

    return jsonify({'cameras': sorted(list(cameras))})


@app.route('/api/camera/<camera_id>/snapshot', methods=['GET', 'POST'])
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


@app.route('/api/camera/<camera_id>/config', methods=['GET', 'POST'])
def camera_config(camera_id):
    """Get or update camera configuration"""
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()

    if request.method == 'POST':
        # Admin saves configuration
        data = request.json

        config_json = json.dumps({
            'spaces': data.get('spaces', []),
            'image_width': data.get('image_width'),
            'image_height': data.get('image_height')
        })

        c.execute('''
            INSERT OR REPLACE INTO camera_configs 
            (camera_id, config_json, image_width, image_height, last_updated)
            VALUES (?, ?, ?, ?, ?)
        ''', (camera_id, config_json, data.get('image_width'),
              data.get('image_height'), datetime.now().isoformat()))

        conn.commit()
        conn.close()

        # Store in memory for quick access
        with config_lock:
            parking_configs[camera_id] = data.get('spaces', [])

        print(f"✓ Configuration saved for {camera_id}: {len(data.get('spaces', []))} spaces")
        return jsonify({'status': 'success', 'message': 'Configuration saved'})

    else:
        # Camera or admin requests configuration
        c.execute('SELECT config_json FROM camera_configs WHERE camera_id = ?', (camera_id,))
        row = c.fetchone()
        conn.close()

        if row:
            config = json.loads(row[0])
            return jsonify(config)
        else:
            return jsonify({'spaces': []})


@app.route('/api/parking/update', methods=['POST'])
def update_parking_status():
    """Receive parking space status updates from detector"""
    data = request.json
    camera_id = data.get('camera_id', 'default')
    spaces = data.get('spaces', [])

    conn = sqlite3.connect('parking.db')
    c = conn.cursor()

    timestamp = datetime.now().isoformat()

    for space in spaces:
        space_number = space['space_number']
        is_free = space['is_free']

        c.execute('''
            INSERT OR REPLACE INTO parking_spaces 
            (camera_id, space_number, is_free, last_updated)
            VALUES (?, ?, ?, ?)
        ''', (camera_id, space_number, is_free, timestamp))

    conn.commit()
    conn.close()

    return jsonify({'status': 'success', 'updated': len(spaces)})


@app.route('/api/parking/status', methods=['GET'])
def get_parking_status():
    """Get current parking status"""
    camera_id = request.args.get('camera_id', 'default')

    conn = sqlite3.connect('parking.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute('''
        SELECT space_number, is_free, last_updated 
        FROM parking_spaces 
        WHERE camera_id = ?
        ORDER BY space_number
    ''', (camera_id,))

    spaces = [dict(row) for row in c.fetchall()]
    conn.close()

    free_count = sum(1 for s in spaces if s['is_free'])
    occupied_count = len(spaces) - free_count

    return jsonify({
        'camera_id': camera_id,
        'total_spaces': len(spaces),
        'free_spaces': free_count,
        'occupied_spaces': occupied_count,
        'spaces': spaces
    })


if __name__ == '__main__':
    print("Initializing Parking Admin Server...")
    init_db()
    print("Database initialized")
    print("\n" + "=" * 60)
    print("  PARKING ADMIN SERVER")
    print("=" * 60)
    print("\nAdmin Interface: http://0.0.0.0:5000")
    print("\nAPI Endpoints:")
    print("  GET  /api/cameras - List cameras")
    print("  GET  /api/camera/<id>/snapshot - Get camera snapshot")
    print("  POST /api/camera/<id>/snapshot - Upload snapshot")
    print("  GET  /api/camera/<id>/config - Get parking config")
    print("  POST /api/camera/<id>/config - Save parking config")
    print("  POST /api/parking/update - Update space status")
    print("  GET  /api/parking/status - Get parking status")
    print("\n" + "=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)