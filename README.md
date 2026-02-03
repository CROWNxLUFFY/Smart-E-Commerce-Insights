# Smart-E-Commerce-Insights

# 📊 Market Insights – Interactive E-Commerce Analytics Platform

## 📌 Project Overview

Market Insights is a **low-code, interactive analytics web application** designed to help businesses understand customer purchase behavior, generate product recommendations, and forecast future sales trends from transactional e-commerce data.

The platform is built to work with **flexible CSV datasets**, allowing users to upload data with different column names and manually map them through an intuitive interface.

---

## 🎯 Objectives

* Analyze **customer spending behavior**
* Segment customers into **value-based groups**
* Recommend products using **association rule mining**
* Forecast future sales using **time-series modeling**
* Handle **real-world data issues** such as missing or inconsistent columns

---

## ✨ Key Features

### 👥 Customer Segmentation

* Uses **K-Means clustering**
* Dynamically labels customers based on spending levels
* Supports **2 to 8 clusters**
* Automatically ranks customers from low to high value

### 📦 Product Recommendations

* Uses **Apriori algorithm**
* Identifies products frequently purchased together
* Generates personalized recommendations per customer
* Based on **support and confidence metrics**

### 📈 Sales Forecasting

* Uses **ARIMA time-series model**
* Forecasts future sales trends
* Displays results using an **interactive Plotly graph**
* Supports zoom, hover, and comparison of historical vs forecasted sales

### 🔄 Flexible Data Handling

* Automatic column detection
* Manual column mapping via sidebar
* Handles:

  * Missing price values
  * Extra irrelevant columns
  * Inconsistent or dirty datasets (negative test cases)

---

## 📄 Dataset Requirements

### Required Columns (any naming convention supported):

* **Customer Identifier** (e.g., CustomerID, UserID)
* **Product Identifier** (e.g., Product, Item, SKU)
* **Quantity Purchased**
* **Transaction Date**

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit** – Web interface
* **Pandas & NumPy** – Data processing
* **Scikit-learn** – Customer clustering
* **MLxtend** – Association rule mining (Apriori)
* **Statsmodels** – ARIMA forecasting
* **Plotly** – Interactive visualizations

---

## 👤 Author

**ANV Abhijit**
Engineering – Data Science
Major Project

---

## ⭐ Conclusion

Market Insights demonstrates how **machine learning and analytics** can be combined with a **low-code interactive interface** to deliver meaningful business insights from raw transactional data. The system is robust, flexible, and suitable for real-world datasets.
