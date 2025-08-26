from flask import Flask, render_template, request, redirect, url_for, session
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import io, base64

app = Flask(__name__)
app.secret_key = "secret123"

# --------------------------
# User credentials
# --------------------------
USERS = {"admin": "1234"}  # username: password

# --------------------------
# Predefined stock list (10 stocks)
# --------------------------
STOCKS = {
    "AAPL": {"name": "Apple", "logo": "https://logo.clearbit.com/apple.com"},
    "GOOG": {"name": "Google", "logo": "https://logo.clearbit.com/google.com"},
    "MSFT": {"name": "Microsoft", "logo": "https://logo.clearbit.com/microsoft.com"},
    "AMZN": {"name": "Amazon", "logo": "https://logo.clearbit.com/amazon.com"},
    "TSLA": {"name": "Tesla", "logo": "https://logo.clearbit.com/tesla.com"},
    "META": {"name": "Meta (Facebook)", "logo": "https://logo.clearbit.com/meta.com"},
    "NFLX": {"name": "Netflix", "logo": "https://logo.clearbit.com/netflix.com"},
    "NVDA": {"name": "NVIDIA", "logo": "https://logo.clearbit.com/nvidia.com"},
    "RELIANCE.NS": {"name": "Reliance India", "logo": "https://logo.clearbit.com/reliance.com"},
    "TCS.NS": {"name": "TCS", "logo": "https://logo.clearbit.com/tcs.com"},
}

# --------------------------
# Convert matplotlib fig → base64 img
# --------------------------
def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img

# --------------------------
# Login
# --------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in USERS and USERS[username] == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            error = "❌ Invalid Username or Password"
    return render_template("login.html", error=error)

# --------------------------
# Dashboard
# --------------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", stocks=STOCKS)

# --------------------------
# Stock page
# --------------------------
@app.route("/stock/<ticker>", methods=["GET", "POST"])
def stock(ticker):
    if "user" not in session:
        return redirect(url_for("login"))

    # Download stock data (1 year, disable progress bar)
    df = yf.download(ticker, period="1y", progress=False)
    if df.empty:
        return render_template("stock.html", ticker=ticker, error="⚠ No data found.")

    # Current price
    current_price = df["Close"].tail(1)

    chart_img, prediction_img, future_price, accuracy = None, None, None, None

    # Plot actual last 1-year prices
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df["Close"], label="Actual Price")
    ax.set_title(f"{ticker} Stock Price (1 Year)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    chart_img = plot_to_img(fig)
    plt.close(fig)

    # If user enters prediction date
    if request.method == "POST":
        future_date = request.form["future_date"]

        df["Days"] = np.arange(len(df))
        X, y = df[["Days"]], df["Close"]

        split = int(len(df) * 0.8)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

        # ---------------- Random Forest ----------------
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        rmse_rf = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
        r2_rf = float(r2_score(y_test, y_pred_rf))

        # ---------------- Linear Regression ----------------
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        rmse_lr = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
        r2_lr = float(r2_score(y_test, y_pred_lr))

        # Accuracy text
        accuracy = (
            f"RandomForest → RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f} | \n"
            f"LinearRegression → RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}"
        )

        # Future prediction (from both models)
        days_ahead = (datetime.strptime(future_date, "%Y-%m-%d") - df.index[-1]).days
        if days_ahead > 0:
            future_price_rf = float(rf_model.predict([[len(df) + days_ahead]])[0])
            future_price_lr = float(lr_model.predict([[len(df) + days_ahead]])[0])
            future_price = f"RF: {future_price_rf:.2f} | LR: {future_price_lr:.2f}"

        # Moving averages
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        # Plot with predictions
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index, df["Close"], label="Actual Price")
        ax.plot(df.index, df["MA20"], "--", label="20-Day MA")
        ax.plot(df.index, df["MA50"], "--", label="50-Day MA")
        if days_ahead > 0:
            ax.scatter(datetime.strptime(future_date, "%Y-%m-%d"), future_price_rf,
                       color="red", label=f"RF Prediction ({future_date})")
            ax.scatter(datetime.strptime(future_date, "%Y-%m-%d"), future_price_lr,
                       color="green", label=f"LR Prediction ({future_date})")
        ax.set_title(f"{ticker} Prediction")
        ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
        prediction_img = plot_to_img(fig)
        plt.close(fig)

    return render_template("stock.html",
                           ticker=ticker,
                           current_price=current_price,
                           chart_img=chart_img,
                           prediction_img=prediction_img,
                           future_price=future_price,
                           accuracy=accuracy)

# --------------------------
# Logout
# --------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
