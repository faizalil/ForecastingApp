from flask import Flask, render_template, request
import yfinance as yf
import matplotlib.pyplot as plt
import io, base64
import numpy as np

from utils.preprocessing import prepare_data
from models.model_lstm import build_lstm
from models.model_gru import build_gru

from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

def evaluate(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return round(mape, 2), round(mae, 4), round(rmse, 4)

@app.route("/", methods=["GET", "POST"])
def home():
    result, plot_url = None, None

    if request.method == "POST":
        ticker = request.form["ticker"]
        model_name = request.form["model"]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        df = yf.download(ticker, start=start_date, end=end_date)
        prices = df["Close"].values

        X, y, scaler = prepare_data(prices)
        input_shape = (X.shape[1], 1)

        if model_name == "LSTM":
            model = build_lstm(input_shape)
        else:
            model = build_gru(input_shape)

        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        preds = model.predict(X)
        preds_inv = scaler.inverse_transform(preds)
        y_inv = scaler.inverse_transform(y.reshape(-1, 1))

        mape, mae, rmse = evaluate(y_inv, preds_inv)
        result = {"mape": mape, "mae": mae, "rmse": rmse}

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(y_inv, label="Actual")
        plt.plot(preds_inv, label="Predicted")
        plt.legend()
        plt.title(f"Hasil Prediksi {ticker} dengan {model_name}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_url = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

    return render_template("home.html", result=result, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
