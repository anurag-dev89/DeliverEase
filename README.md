# 🛵 DeliverEase: Food Delivery Time Prediction

> *"How long will my food take to arrive?"* This project answers that question with machine learning.

Given the details of an order (who is delivering it, how far they have to go, the weather, the traffic, the time of day and more), the model predicts the delivery time in **minutes**.

I built this end to end: raw messy data → cleaning → exploration → feature engineering → four models → tuning → a saved model that can predict on new orders.

---

## 📌 Quick Summary

| | |
|---|---|
| **Problem type** | Supervised learning, Regression |
| **Target** | `Time_taken(min)` (delivery time in minutes) |
| **Dataset size** | 45,593 orders → **45,162** after cleaning |
| **Features used** | 12 original → **21** after encoding |
| **Best model** | Tuned Random Forest / Tuned XGBoost (almost tied) |
| **Final error** | about **3.9 minutes** (RMSE) |
| **Variance explained** | about **83%** (R²) |

---

## 🗂️ About the Data

File: `data/Food_delivery.csv`: **45,593 rows × 20 columns**, one row per past order (orders placed between Feb and Apr 2022).

| Column | What it means |
|---|---|
| `ID`, `Delivery_person_ID` | Order ID and delivery partner ID (IDs were not used for training) |
| `Delivery_person_Age` | Age of the delivery partner |
| `Delivery_person_Ratings` | Rating of the delivery partner |
| `Restaurant_latitude / longitude` | Where the food is picked up |
| `Delivery_location_latitude / longitude` | Where the food is delivered |
| `Order_Date` | Date of the order |
| `Time_Orderd`, `Time_Order_picked` | When the order was placed and picked up |
| `Weatherconditions` | Sunny, Cloudy, Fog, Stormy, Windy, Sandstorms |
| `Road_traffic_density` | Low, Medium, High, Jam |
| `Vehicle_condition` | Condition of the vehicle (0 to 3) |
| `Type_of_order` | Snack, Meal, Drinks, Buffet |
| `Type_of_vehicle` | motorcycle, scooter, electric_scooter, bicycle |
| `multiple_deliveries` | How many other deliveries the rider carried at the same time |
| `Festival` | Was it a festival day? (Yes/No) |
| `City` | Metropolitian, Urban, Semi-Urban |
| **`Time_taken(min)`** | **The target we predict** (10 to 54 min, average about 26 min) |

---

## 🧹 Step 1: Data Cleaning

The raw data was messy. Here is what I fixed:

| Problem | Example | Fix |
|---|---|---|
| Weather had a text prefix | `"conditions Sunny"` | Removed the `"conditions "` prefix, now `Sunny` |
| Fake missing values stored as text | `"NaN"` (with extra spaces) | Stripped spaces, then converted to real `NaN` |
| Target stored as text | `"(min) 24"` | Extracted the number, now `24` (int) |
| Numbers stored as text | Age, Ratings, multiple_deliveries | Converted to numeric |
| Date stored as text | `19-03-2022` | Converted to datetime |
| Invalid ratings | 53 rows had rating **6** (max is 5) | Capped (clipped) at 5 |
| Missing values | See table below | Median for numbers, mode for categories |
| Duplicates | Checked | **0 duplicate rows** found |

**Missing values found:**

| Column | Missing | Filled with |
|---|---:|---|
| Delivery_person_Age | 1,854 | median |
| Delivery_person_Ratings | 1,908 | median |
| multiple_deliveries | 993 | median |
| Weatherconditions | 616 | mode (most frequent) |
| Road_traffic_density | 601 | mode |
| Festival | 228 | mode |
| City | 1,200 | mode |
| Time_Orderd | 1,731 | handled later when making `order_hour` |

> 💡 **Why median?** It is not pulled around by extreme values the way the mean is. **Why mode?** For categories, the most common value is the only sensible "average".

---

## 🔍 Step 2: Exploratory Data Analysis (EDA)

I grouped the data by each feature and compared the average delivery time.

### 🎯 The target
Delivery time ranges from **10 to 54 minutes**: mean **26.3 min**, median **26 min**, standard deviation **9.4 min**. The histogram is slightly right-skewed (a few very long deliveries).

### 🚦 Road traffic (strong effect)

| Traffic | Orders | Avg time (min) |
|---|---:|---:|
| Low | 16,078 | **21.5** |
| Medium | 10,947 | 26.7 |
| High | 4,425 | 27.2 |
| Jam | 14,143 | **31.2** |

More traffic means a longer delivery, which is exactly what you would expect.

### 🌦️ Weather

| Weather | Avg time (min) |
|---|---:|
| Sunny | **21.9** (fastest) |
| Stormy | 25.9 |
| Sandstorms | 25.9 |
| Windy | 26.1 |
| Fog | 28.7 |
| Cloudy | **28.9** (slowest) |

### 🎉 Festival days (very strong effect)

| Festival | Orders | Avg time (min) |
|---|---:|---:|
| No | 44,697 | 25.9 |
| Yes | 896 | **45.5** |

Festival days add about **20 extra minutes** on average, and times are also more consistent (std 4.0 vs 9.05).

### 🏙️ City type

| City | Orders | Avg time (min) |
|---|---:|---:|
| Urban | 10,136 | 23.0 |
| Metropolitian | 35,293 | 27.1 |
| Semi-Urban | 164 | **49.7** |

Semi-Urban deliveries are by far the slowest, but only 164 orders (a tiny group).

### 🏍️ Vehicle type

| Vehicle | Orders | Avg time (min) |
|---|---:|---:|
| electric_scooter | 3,814 | 24.5 |
| scooter | 15,276 | 24.5 |
| bicycle | 68 | 26.4 |
| motorcycle | 26,435 | 27.6 |

### 🍔 Order type
Snack, Meal, Drinks and Buffet all average about **26 minutes**. The order type makes almost no difference.

### 👤 Delivery person
- **Age:** averages about 29.6 years (range 15 to 50). The age trend line is fairly flat.
- **Ratings:** partners rated **4.5 to 5.0** average about **23.6 to 25.5 min**, while partners rated **4.0 to 4.4** average about **34.5 min**. Better rated riders deliver faster. Only about **3%** of orders have a rating below 4.

### 🕒 Time of day

| Order hour | Avg time (min) |
|---|---:|
| 8 to 10 AM | ~19.5 (fastest) |
| 11 AM to 2 PM | ~26 to 27.5 |
| 3 to 4 PM | ~23 |
| 5 to 6 PM | ~27.3 |
| **7 to 9 PM** | **~31** (slowest, dinner rush) |
| 10 PM to 12 AM | ~22 to 23 |

### 📈 Charts in the notebook
The notebook produces these charts (you can export them into an `images/` folder and show them here):
1. Histogram + KDE of delivery time
2. Bar chart: average delivery time by weather
3. Bar chart: festival vs non-festival
4. Histogram of delivery partner age
5. Line plot: delivery time vs partner age
6. Line plot: delivery time vs partner rating
7. Bar chart: average delivery time by order hour
8. Correlation heatmap of the numeric features

To add them, save each plot with `plt.savefig("images/name.png")` and then embed with `![caption](images/name.png)`.

---

## 🛠️ Step 3: Feature Engineering

Two new, more useful features were created from the raw columns:

1. **`distance_km`**: instead of four raw latitude/longitude columns, I computed the real straight-line distance between restaurant and customer with the **Haversine formula**:

   ```
   a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
   d = 2 · R · arcsin(√a)        (R = 6371 km)
   ```

2. **`order_hour`**: the hour (0 to 23) extracted from `Time_Orderd`. A handful of unparseable times were filled with the median hour.

### ⚠️ Removing bad GPS rows
Some rows had wrong coordinates (for example, a negative latitude), which gave distances from **2,216 km up to 19,692 km**. That is impossible for food delivery. I removed these **431 rows** (distance over 100 km).

| | Rows |
|---|---:|
| Before | 45,593 |
| Removed (bad GPS) | 431 |
| **After** | **45,162** |

After cleaning, distance ranges from **1.5 km to 21 km** (average about 9.7 km).

**Columns dropped before modelling:** raw lat/long (replaced by distance), raw timestamps (replaced by hour), and `Delivery_person_ID` (1,320 unique IDs would just cause overfitting).

---

## 🔢 Step 4: Preparing the Data for Models

- **Ordinal encoding** for `Road_traffic_density`: `Low=0, Medium=1, High=2, Jam=3` (it has a natural order).
- **One-hot encoding** (`drop_first=True`) for `Type_of_order`, `Type_of_vehicle`, `Festival`, `City`, `Weatherconditions` (no natural order).
- Result: **21 features**.
- **Train/test split:** 80/20, `random_state=42` → **36,129 train rows** and **9,033 test rows**.
- **Scaling:** `StandardScaler` fitted **only on training data** (to avoid data leakage) and then applied to test data. Saved as `scaler.pkl`.

---

## 🤖 Step 5: Models & Results

I trained four models and compared them using **RMSE** (error in minutes, lower is better) and **R²** (share of variation explained, higher is better).

| Model | Train R² | Test R² | Train RMSE | Test RMSE (min) | What I noticed |
|---|---:|---:|---:|---:|---|
| Linear Regression | 0.583 | 0.569 | 6.04 | 6.21 | A simple baseline. The data is not a straight-line pattern. |
| Decision Tree | 1.000 | 0.664 | 0.00 | 5.49 | Memorised the training data (severe overfitting). |
| Random Forest | 0.974 | 0.819 | 1.51 | 4.03 | Big jump. Combining many trees fixes much of the overfitting. |
| XGBoost (default) | – | 0.822 | – | 4.00 | Strong right out of the box. |
| **Random Forest (tuned)** | – | **0.830** | – | **3.91** | Best test score. |
| **XGBoost (tuned)** | 0.847 | 0.828 | 3.66 | 3.93 | Almost identical, with a small train/test gap (healthy). |

### 🎛️ Hyperparameter tuning (`RandomizedSearchCV`, 25 combinations, 5-fold CV)

**Random Forest best settings:** `n_estimators=400`, `max_depth=20`, `min_samples_split=10`, `min_samples_leaf=1`, `max_features=0.5`

**XGBoost best settings:** `n_estimators=500`, `max_depth=8`, `learning_rate=0.01`, `subsample=0.7`, `colsample_bytree=0.9`, `reg_alpha=1`, `reg_lambda=1`

| | Before tuning | After tuning |
|---|---:|---:|
| Random Forest test RMSE / R² | 4.03 / 0.819 | **3.91 / 0.830** |
| XGBoost test RMSE / R² | 4.00 / 0.822 | **3.93 / 0.828** |

### 🧪 Extra experiment: does an "order period" feature help?
I tried grouping the hour into **Lunch (12 to 14) / Dinner (19 to 21) / Normal** and retrained XGBoost with the same settings.

- Adjusted R² **without** the feature: **0.8276**
- Adjusted R² **with** the feature: **0.8272**

No improvement, so I kept the simpler model. Not every new feature is worth adding.

### 🔮 Sample predictions (tuned XGBoost)

| Predicted (min) | Actual (min) | Error |
|---:|---:|---:|
| 40.6 | 46 | 5.4 |
| 18.7 | 19 | 0.3 |
| 37.6 | 40 | 2.4 |

Average error on these three: about **2.7 minutes**.

---

## ✅ Final Conclusion

- Tuned **Random Forest** and tuned **XGBoost** perform almost the same: about **3.9 minutes error** and about **83% R²**.
- The model is off by roughly **4 minutes** on average. That is good for something as unpredictable as food delivery.
- **Biggest drivers of delivery time:** traffic, festival days, distance, partner rating/age, and number of simultaneous deliveries.
- Linear Regression was too simple, the single Decision Tree overfit badly, and ensembles (Random Forest, XGBoost) worked best.

---

## 🧰 Tools Used

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · XGBoost · joblib

---

## ▶️ Run It Yourself

```bash
# 1. Clone the repo
git clone https://github.com/anurag-dev89/DeliverEase.git
cd DeliverEase

# 2. (Optional) Make a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install the libraries
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook notebooks/DeliverEase.ipynb
```

> The notebook lives in `notebooks/`, so the data path is `../data/Food_delivery.csv`.

### Use the saved model to predict

```python
import joblib

model  = joblib.load("models/xgb_best.pkl")
scaler = joblib.load("models/scaler.pkl")

# new_orders must have the same 21 columns, in the same order, as the training data
new_scaled  = scaler.transform(new_orders)
predictions = model.predict(new_scaled)      # delivery time in minutes
```

---

## 📁 Repository Structure

```
DeliverEase/
│
├── data/
│   └── Food_delivery.csv        # the dataset
│
├── notebooks/
│   └── DeliverEase.ipynb        # full notebook with explanations
│
├── models/
│   ├── scaler.pkl               # saved StandardScaler
│   └── xgb_best.pkl             # saved best XGBoost model
│
├── images/                      # (optional) exported charts
├── README.md                    # you are here
├── requirements.txt             # libraries you need
└── .gitignore
```

---

Thanks for stopping by! If you found this useful, a ⭐ on the repo would make my day.