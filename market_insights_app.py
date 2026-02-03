import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import time

# Auto-detect columns

def detect_column(columns, keywords):
    for col in columns:
        for key in keywords:
            if key in col.lower():
                return col
    return None

# Dynamic cluster labels

def get_dynamic_labels(k):
    label_sets = {
        2: ["Low Value", "High Value"],
        3: ["Low Value", "Medium Value", "High Value"],
        4: ["Very Low", "Low", "High", "Very High"],
        5: ["Very Low", "Low", "Medium", "High", "Very High"],
        6: ["Extremely Low", "Very Low", "Low", "High", "Very High", "Premium"],
        7: ["Extremely Low", "Very Low", "Low", "Medium", "High", "Very High", "Premium"],
        8: ["Extremely Low", "Very Low", "Low", "Lower-Medium", "Upper-Medium", "High", "Very High", "Premium"]
    }
    return label_sets.get(k, [f"Segment {i+1}" for i in range(k)])

# Page Config

st.set_page_config(page_title="Market Insights", layout="wide")
st.title("📊 Market Insights – Interactive E-Commerce Analytics Platform")

# USER INSTRUCTIONS

st.info("""
### 📄 Dataset Requirements

Your CSV should contain **transaction-level data** with:

• **Customer column** (CustomerID, UserID, ClientID)  
• **Product column** (Product, Item, SKU)  
• **Quantity column** (Quantity, Qty)  
• **Date column** (Date, OrderDate)  
• **Price column (optional)** – used for revenue & forecasting  

📌 Additional columns like City, State, Country, Branch are optional and will be ignored.
""")
with st.expander("📘 How to prepare your CSV file"):
    st.write("""
    1. Each row should represent **one purchase transaction**  
    2. The same customer can appear multiple times  
    3. Date should be in a readable format (**YYYY-MM-DD recommended**)  
    4. If price is missing, the system assumes **price = 1**  
    5. Column names do **NOT** have to be exact — you can map them manually
    """)

# Upload Dataset

uploaded_file = st.file_uploader("⬆️ Upload your CSV file", type=["csv"])

if uploaded_file:
    with st.spinner("Reading dataset..."):
        time.sleep(1)
        df = pd.read_csv(uploaded_file)

    columns = df.columns.tolist()

    auto_customer = detect_column(columns, ["customer", "user", "client"])
    auto_product = detect_column(columns, ["product", "item", "sku"])
    auto_quantity = detect_column(columns, ["quantity", "qty", "count"])
    auto_price = detect_column(columns, ["price", "amount", "cost", "value"])
    auto_date = detect_column(columns, ["date", "time", "order"])

    st.sidebar.header("🔗 Column Mapping")

    customer_col = st.sidebar.selectbox(
        "Customer Column",
        columns,
        index=columns.index(auto_customer) if auto_customer in columns else 0
    )

    product_col = st.sidebar.selectbox(
        "Product Column",
        columns,
        index=columns.index(auto_product) if auto_product in columns else 0
    )

    quantity_col = st.sidebar.selectbox(
        "Quantity Column",
        columns,
        index=columns.index(auto_quantity) if auto_quantity in columns else 0
    )

    date_col = st.sidebar.selectbox(
        "Date Column",
        columns,
        index=columns.index(auto_date) if auto_date in columns else 0
    )

    price_col = st.sidebar.selectbox(
        "Price Column (optional)",
        ["None"] + columns,
        index=(columns.index(auto_price) + 1) if auto_price in columns else 0
    )

    with st.spinner("Preparing data..."):
        time.sleep(1)
        df = df.rename(columns={
            customer_col: "CustomerID",
            product_col: "Product",
            quantity_col: "Quantity",
            date_col: "Date"
        })

        if price_col != "None":
            df = df.rename(columns={price_col: "Price"})
        else:
            df["Price"] = 1

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["CustomerID", "Product", "Quantity", "Date"], inplace=True)

    st.success("Dataset processed successfully!")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📁 Dataset Overview", "👥 Segmentation", "📦 Recommendations", "📈 Forecast"]
    )

    # TAB 1: DATASET OVERVIEW

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", df["CustomerID"].nunique())
        col2.metric("Total Products", df["Product"].nunique())
        col3.metric("Total Sales", int((df["Quantity"] * df["Price"]).sum()))
        col4.metric("Transactions", len(df))
        st.dataframe(df.head())

    # TAB 2: SEGMENTATION
    
    with tab2:
        cust_features = df.groupby("CustomerID").agg({
            "Quantity": "sum",
            "Price": "sum"
        }).reset_index()

        max_clusters = min(8, cust_features.shape[0])
        k = st.slider("Number of Customer Segments", 2, max_clusters, min(3, max_clusters))

        if st.button("▶ Run Customer Segmentation"):
            with st.spinner("Clustering customers..."):
                time.sleep(1)

                kmeans = KMeans(n_clusters=k, random_state=42)
                cust_features["Cluster"] = kmeans.fit_predict(
                    cust_features[["Quantity", "Price"]]
                )

                cluster_order = cust_features.groupby("Cluster")["Price"].mean().sort_values().index
                labels = get_dynamic_labels(k)

                cust_features["Customer Segment"] = cust_features["Cluster"].map(
                    {cluster_order[i]: labels[i] for i in range(len(cluster_order))}
                )

            st.success("Customer segmentation completed!")
            st.dataframe(
                cust_features[["CustomerID", "Quantity", "Price", "Customer Segment"]]
            )

    # TAB 3: RECOMMENDATIONS
    
    with tab3:
        basket = (
            df.groupby(["CustomerID", "Product"])["Quantity"]
            .sum()
            .unstack()
            .fillna(0)
        ).applymap(lambda x: 1 if x > 0 else 0)

        frequent_items = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6) \
            if not frequent_items.empty else pd.DataFrame()

        selected_customer = st.selectbox("Select Customer", df["CustomerID"].unique())
        bought = df[df["CustomerID"] == selected_customer]["Product"].unique()

        st.write("🛒 Products Purchased:", list(bought))

        if not rules.empty:
            rec = set()
            for _, row in rules.iterrows():
                if row["antecedents"].issubset(bought):
                    rec.update(row["consequents"])

            st.write("🎯 Recommended Products:", list(rec - set(bought)) or "No recommendations found")

    # TAB 4: FORECAST (INTERACTIVE)
    
    with tab4:
        if st.button("📈 Forecast Sales"):
            with st.spinner("Generating interactive forecast..."):
                time.sleep(1)

                df["Sales"] = df["Quantity"] * df["Price"]
                sales_ts = df.groupby(df["Date"].dt.to_period("M"))["Sales"].sum()
                sales_ts.index = sales_ts.index.to_timestamp()

                if len(sales_ts) >= 6:
                    model = ARIMA(sales_ts, order=(1, 1, 1))
                    forecast = model.fit().forecast(steps=3)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sales_ts.index,
                        y=sales_ts.values,
                        mode="lines+markers",
                        name="Historical Sales"
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast.values,
                        mode="lines+markers",
                        name="Forecasted Sales",
                        line=dict(dash="dash")
                    ))

                    fig.update_layout(
                        title="📈 Sales Forecast (Interactive)",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("At least 6 months of data required.")

else:
    st.warning("Please upload a CSV file to begin analysis.")

