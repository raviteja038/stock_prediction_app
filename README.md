# 📈 Stock Prediction Web App

This is a Flask-based web application for **stock price prediction**.  
It allows users to:  
- 🔑 Login with credentials  
- 🏠 Access a dashboard with multiple stocks  
- 📊 View stock information (charts, moving averages, latest price)  
- 🔮 Predict future stock prices by entering a **date**  
- ⚡ Compare actual vs predicted prices with visualizations  

---

## 🚀 Features  
✅ User authentication (Login Page with bull & bear background)  
✅ Dashboard with clickable stock cards (logos + names)  
✅ Fetches stock data from **Yahoo Finance API (yfinance)**  
✅ Data visualization using **Matplotlib**  
✅ Predicts future prices using **Machine Learning (Random Forest Regressor)**  
✅ Shows current price, moving averages, accuracy score  
✅ Stylish UI with gradient backgrounds  

---

## 🛠️ Tech Stack  
- **Backend:** Python, Flask  
- **Frontend:** HTML, CSS (gradient backgrounds, styled pages)  
- **Data Source:** [yfinance](https://pypi.org/project/yfinance/)  
- **ML Model:** Random Forest Regressor (scikit-learn)  
- **Visualization:** Matplotlib  

---

## 📂 Project Structure  

```
stock_prediction_app/
│── app.py                 # Main Flask application
│── templates/
│   ├── login.html         # Login page
│   ├── dashboard.html     # Dashboard with stock list
│   ├── stock.html         # Stock details, charts, predictions
│── static/
│   └── bullbear.jpg       # Background image for login
│── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/stock_prediction_app.git
cd stock_prediction_app
```

### 2️⃣ Install Dependencies  
Make sure you have **Python 3.10+** installed.  
Install required libraries:  
```bash
pip install flask yfinance scikit-learn matplotlib pandas
```

### 3️⃣ Run the App  
```bash
python app.py
```

You should see:  
```
Server running at http://127.0.0.1:5000
```

### 4️⃣ Open in Browser  
Go to 👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)  

---

## 🔑 Default Login Credentials  
```
Username: admin
Password: admin
```

(You can edit credentials inside `app.py` in the `USERS` dictionary.)

---

## 📊 Example Screenshots  
- **Login Page** (Bull & Bear background)  
- **Dashboard** with multiple stock options  
- **Stock Page** with current price, actual vs predicted chart  

---

## 📌 Future Improvements  
- Add more ML models (LSTM, ARIMA, Linear Regression) for comparison  
- Deploy on **Heroku / AWS / Azure**  
- Add user registration & database storage  
- Improve UI with Bootstrap  

---

## 👨‍💻 Author  
Developed by **P. Naga Ravi Teja**  
(B.Tech AI & ML, KSRM College, AP, India)  
