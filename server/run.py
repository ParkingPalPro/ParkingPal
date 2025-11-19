from flask import Flask
from config import Config
from server.parkingpal.extensions import db
from server.parkingpal.utils import hash_password


def create_app():
    app = Flask(__name__, template_folder='parkingpal/templates')
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    from server.parkingpal.blueprints.auth import auth_bp
    from server.parkingpal.blueprints.admin import admin_bp
    from server.parkingpal.blueprints.api import api_bp
    #from app.blueprints.dashboard import dashboard_bp
    from server.parkingpal.blueprints.user import user_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
    #app.register_blueprint(dashboard_bp)
    app.register_blueprint(user_bp)

    # Import models BEFORE create_all()
    with app.app_context():
        from server.parkingpal.models.user import User
        from server.parkingpal.models.camera_config import CameraConfig
        from server.parkingpal.models.parking_space import ParkingSpace
        from server.parkingpal.models.parking_session import ParkingSession

        # Create default admin if not exists
        if not User.query.filter_by(username='admin').first():
            admin_user = User(
                username='admin',
                email='admin@example.com',
                password=hash_password('Password1.'),  # choose a secure password
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created: admin / Password1.")

        db.create_all()

    return app


application = create_app()

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000, debug=True)
