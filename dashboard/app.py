from flask import Flask, jsonify
from api.alerts_api import alerts_api
from api.metrics_api import metrics_api

app = Flask(__name__)
app.register_blueprint(alerts_api)
app.register_blueprint(metrics_api)

@app.get("/")
def home():
    return "AI NIDS + SOAR Dashboard Active"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
