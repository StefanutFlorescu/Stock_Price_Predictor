from flask import Flask, jsonify
from model import predict_stock_price
app = Flask(__name__)

@app.route("/")
def home():
    return "API-ul este activ!"

@app.route("/stock/<string:ticker>")
def get_stock_price(ticker):
    price = predict_stock_price(ticker)
    return jsonify({"ticker": ticker, "price": price})

if __name__ == "__main__":
    app.run(debug=True)
