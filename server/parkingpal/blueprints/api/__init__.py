from flask import Blueprint

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Import routes
from . import parking_sessions
from . import parking_spaces
#from . import health
