from flask import Flask, jsonify
from model import predict_stock_price
from datetime import datetime
app = Flask(__name__)

@app.route("/")
def home():
    return "API-ul este activ!"

@app.route("/stock/<string:ticker>")
def get_stock_price(ticker):
    price = predict_stock_price(ticker)
    current_time = datetime.now().isoformat()  # Generates timestamp in ISO 8601 format
    
    return jsonify({
        "symbol": ticker,
        "price": price,
        "date": current_time
    })

if __name__ == "__main__":
    app.run(debug=True)
