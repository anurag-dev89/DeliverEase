# 🛵 DeliverEase: Food Delivery Time Prediction

Ever wondered how long your food will take to arrive? This project tries to answer exactly that.

Given the details of an order (who's delivering it, how far they have to go, the weather, the traffic, the time of day, and more), the model predicts how many **minutes** the delivery will take.

I built this from start to finish: from messy raw data all the way to a saved model that can make predictions on new orders.

---

## 📊 How well does it work?

I tried four different models and compared them. The tuned XGBoost model won.

| Model | Test R² | Test RMSE (min) | Train R² | What I noticed |
|---|---|---|---|---|
| Linear Regression | 0.57 | 6.21 | 0.58 | A simple starting point. The data isn't a straight-line pattern. |
| Decision Tree | 0.67 | 5.46 | 1.00 | Memorised the training data, so it struggles on new data. |
| Random Forest | 0.82 | 4.03 | 0.97 | Big jump. Combining many trees fixes a lot of the overfitting. |
| **XGBoost (tuned)** | **0.83** | **3.91** | 0.85 | Best score, and the training and test results are close, which is a good sign. |

**In plain words:** the final model is off by about 4 minutes on average. Not bad for something as unpredictable as food delivery!

---

## 🧭 What I did, step by step

1. **Looked at the data.** Checked what's in it and how it's laid out.
2. **Cleaned it up.** Fixed messy text and filled in missing values. For example, `"conditions Sunny"` became `Sunny`, and `"(min) 24"` became `24`.
3. **Explored it (EDA).** Found out which things actually change the delivery time.
4. **Created new features.**
   - `distance_km`: the real distance between the restaurant and the customer, worked out from their coordinates using the **Haversine formula**.
   - `order_hour`: the hour of the day the order was placed.
5. **Prepared the data for the models.** Turned categories into numbers (ordered for traffic density, one-hot for the rest) and scaled the features.
6. **Trained and compared four models.** Linear Regression, Decision Tree, Random Forest and XGBoost.
7. **Tuned the best one.** Used RandomizedSearchCV to find good settings, checked it with RMSE, R² and Adjusted R², then saved it so it's ready to use.

I explained the maths and the thinking behind each step right inside the notebook, so you can follow along.

---

## 🗂️ About the data

The dataset (`data/Food_delivery.csv`) has past food delivery orders. For each order it includes:

- The delivery person's age and rating
- Restaurant and delivery location coordinates
- Weather and road traffic
- Vehicle type and order type
- Whether it was a festival day
- City type

What we're trying to predict is `Time_taken(min)`, the delivery time in minutes.

---

## 🧰 Tools used

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · XGBoost · joblib

---

## ▶️ Run it yourself

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/DeliverEase.git
cd DeliverEase

# 2. (Optional) Make a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install the libraries
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook notebooks/DeliverEase.ipynb
```

One small thing: in the first cell where the data is loaded, make sure the file path points to `data/Food_delivery.csv`. Since the notebook lives inside the `notebooks/` folder, the path may need to be `../data/Food_delivery.csv`.

---

## 📁 What's in this repo

```
DeliverEase/
│
├── data/
│   └── Food_delivery.csv        # the dataset
│
├── notebooks/
│   └── DeliverEase.ipynb        # the full notebook, with explanations
│
├── models/
│   ├── scaler.pkl               # the saved scaler
│   └── best_model.pkl           # the saved best model (XGBoost)
│
├── README.md                    # you are here
├── requirements.txt             # libraries you need
└── .gitignore
```

---

Thanks for stopping by! If you found this useful, a ⭐ on the repo would make my day.