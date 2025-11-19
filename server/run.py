from flask import Flask
from config import Config
from server.parkingpal.extensions import db


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    #from app.blueprints.auth import auth_bp
    #from app.blueprints.admin import admin_bp
    from server.parkingpal.blueprints.api import api_bp
    #from app.blueprints.dashboard import dashboard_bp
    #from app.blueprints.user import user_bp

    #app.register_blueprint(auth_bp)
    #app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
    #app.register_blueprint(dashboard_bp)
    #app.register_blueprint(user_bp)

    # Import models BEFORE create_all()
    with app.app_context():
        from server.parkingpal.models.user import User
        from server.parkingpal.models.parking_space import ParkingSpace
        from server.parkingpal.models.parking_session import ParkingSession
        db.create_all()

    return app


application = create_app()

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000, debug=True)
