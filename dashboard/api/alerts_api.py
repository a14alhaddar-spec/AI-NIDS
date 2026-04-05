from flask import Blueprint, jsonify

alerts_api = Blueprint("alerts_api", __name__, url_prefix="/api/alerts")


@alerts_api.get("/")
def list_alerts():
	# Placeholder data until alert storage is wired in.
	return jsonify([])
