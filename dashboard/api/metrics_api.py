from flask import Blueprint, jsonify

metrics_api = Blueprint("metrics_api", __name__, url_prefix="/api/metrics")


@metrics_api.get("/")
def get_metrics():
	# Placeholder metrics until pipeline integration is ready.
	return jsonify({"status": "ok"})
