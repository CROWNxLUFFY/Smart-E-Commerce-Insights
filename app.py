from flask import Flask, render_template, request, session
import pandas as pd
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import os

app = Flask(__name__)
app.secret_key = "market_insights_secret_key"

# Auto Detect Columns
def detect_column(columns, keywords):
    for col in columns:
        for key in keywords:
            if key in col.lower():
                return col
    return None

# Dynamic Labels
def get_dynamic_labels(k):
    label_sets = {
        2: ["Low Value", "High Value"],
        3: ["Low Value", "Medium Value", "High Value"],
        4: ["Very Low","Low","High","Very High"],
        5: ["Very Low","Low","Medium","High","Very High"]
    }

    return label_sets.get(
        k,
        [f"Segment {i+1}" for i in range(k)]
    )

@app.route("/", methods=["GET", "POST"])
def home():

    filename = None
    total_customers = None
    total_products = None
    total_sales = None
    transactions = None

    preview_data = None
    segmentation_table = None
    message = None

    customers = []
    selected_customer = None
    purchased_products = []
    recommended_products = []
    forecast_chart = None
    forecast_message = None

    # CUSTOMER RECOMMENDATION BUTTON
    if request.method == "POST" and "selected_customer" in request.form and "uploaded_csv" in session:

        selected_customer = request.form.get("selected_customer")

        if "uploaded_csv" in session:

            df = pd.read_json(session["uploaded_csv"])
            filename = "Uploaded Dataset"

            columns = df.columns.tolist()
            price_col = detect_column(
                columns,
                ["price", "amount", "cost", "value"]
            )

            date_col = detect_column(
                columns,
                ["date", "time", "order"]
            )

            customer_col = detect_column(
                columns,
                ["customer", "user", "client"]
            )

            product_col = detect_column(
                columns,
                ["product", "item", "sku"]
            )

            quantity_col = detect_column(
                columns,
                ["quantity", "qty", "count"]
            )

            customers = sorted(
                df[customer_col]
                .astype(str)
                .unique()
                .tolist()
            )
            total_customers = df[customer_col].nunique()
            total_products = df[product_col].nunique()

            if price_col:
                sales_amount = (
                    df[quantity_col] * df[price_col]
                ).sum()

                if sales_amount >= 10000000:
                    total_sales = f"₹ {sales_amount/10000000:.2f} Cr"

                elif sales_amount >= 100000:
                    total_sales = f"₹ {sales_amount/100000:.2f} L"

                elif sales_amount >= 1000:
                    total_sales = f"₹ {sales_amount/1000:.2f} K"

                else:
                    total_sales = f"₹ {sales_amount:.2f}"

            transactions = len(df)

            preview_data = (
                df.head()
                .to_html(index=False)
            )

            purchased_products = (
                df[
                    df[customer_col]
                    .astype(str)
                    == selected_customer
                ][product_col]
                .unique()
                .tolist()
            )

            basket = (
                df.groupby(
                    [customer_col, product_col]
                )[quantity_col]
                .sum()
                .unstack()
                .fillna(0)
            )

            basket = basket > 0

            frequent_items = apriori(
                basket,
                min_support=0.05,
                use_colnames=True
            )

            if not frequent_items.empty:

                rules = association_rules(
                    frequent_items,
                    metric="confidence",
                    min_threshold=0.6
                )

                rec = set()

                for _, row in rules.iterrows():

                    if row["antecedents"].issubset(
                        purchased_products
                    ):
                        rec.update(
                            row["consequents"]
                        )

                recommended_products = list(
                    rec - set(purchased_products)
                )
                if price_col:

                    cust_features = (
                        df.groupby(customer_col)
                        .agg({
                            quantity_col: "sum",
                            price_col: "sum"
                        })
                        .reset_index()
                    )

                    if len(cust_features) >= 3:

                        kmeans = KMeans(
                            n_clusters=min(5, len(cust_features)),
                            random_state=42,
                            n_init=10
                        )

                        cust_features["Cluster"] = (
                            kmeans.fit_predict(
                                cust_features[
                                    [quantity_col, price_col]
                                ]
                            )
                        )

                        cluster_order = (
                            cust_features
                            .groupby("Cluster")[price_col]
                            .mean()
                            .sort_values()
                            .index
                        )

                        labels = get_dynamic_labels(
                            kmeans.n_clusters
                        )

                        mapping = {
                            cluster_order[i]: labels[i]
                            for i in range(
                                len(cluster_order)
                            )
                        }

                        cust_features["Segment"] = (
                            cust_features["Cluster"]
                            .map(mapping)
                        )

                        segmentation_table = (
                            cust_features[
                                [
                                    customer_col,
                                    quantity_col,
                                    price_col,
                                    "Segment"
                                ]
                            ]
                            .to_html(index=False)
                        )
                        if date_col and price_col:

                            try:

                                forecast_df = df.copy()

                                forecast_df[date_col] = pd.to_datetime(
                                    forecast_df[date_col],
                                    errors="coerce"
                                )

                                forecast_df.dropna(
                                    subset=[date_col],
                                    inplace=True
                                )

                                forecast_df["Sales"] = (
                                    forecast_df[quantity_col]
                                    * forecast_df[price_col]
                                )

                                sales_ts = (
                                    forecast_df.groupby(
                                        forecast_df[date_col]
                                        .dt.to_period("M")
                                    )["Sales"]
                                    .sum()
                                )

                                sales_ts.index = (
                                    sales_ts.index.to_timestamp()
                                )

                                if len(sales_ts) >= 6:

                                    model = ARIMA(
                                        sales_ts,
                                        order=(1,1,1)
                                    )

                                    forecast = (
                                        model.fit()
                                        .forecast(steps=6)
                                    )

                                    fig = go.Figure()

                                    fig.add_trace(
                                        go.Scatter(
                                            x=sales_ts.index,
                                            y=sales_ts.values,
                                            mode="lines+markers",
                                            name="Historical Sales"
                                        )
                                    )   

                                    fig.add_trace(
                                        go.Scatter(
                                            x=forecast.index,
                                            y=forecast.values,
                                            mode="lines+markers",
                                            name="Forecasted Sales"
                                        )
                                    )

                                    forecast_chart = fig.to_html(
                                        full_html=False
                                    )

                            except Exception as e:

                                forecast_message = str(e)

    # FILE UPLOAD
    elif request.method == "POST":

        file = request.files.get("csv_file")

        if file and file.filename.endswith(".csv"):

            from io import BytesIO

            file_bytes = BytesIO(file.read())

            df = pd.read_csv(file_bytes)
            session["uploaded_csv"] = df.to_json()

            filename = file.filename

            columns = df.columns.tolist()

            customer_col = detect_column(
                columns,
                ["customer", "user", "client"]
            )

            product_col = detect_column(
                columns,
                ["product", "item", "sku"]
            )

            quantity_col = detect_column(
                columns,
                ["quantity", "qty", "count"]
            )

            price_col = detect_column(
                columns,
                ["price", "amount", "cost", "value"]
            )
            date_col = detect_column(
                columns,
                ["date", "time", "order"]
            )

            if customer_col and product_col and quantity_col:

                customers = sorted(
                    df[customer_col]
                    .astype(str)
                    .unique()
                    .tolist()
                )

                total_customers = (
                    df[customer_col]
                    .nunique()
                )

                total_products = (
                    df[product_col]
                    .nunique()
                )

                if price_col:

                    sales_amount = (
                        df[quantity_col] * df[price_col]
                    ).sum()

                    if sales_amount >= 10000000:
                        total_sales = f"₹ {sales_amount/10000000:.2f} Cr"

                    elif sales_amount >= 100000:
                        total_sales = f"₹ {sales_amount/100000:.2f} L"

                    elif sales_amount >= 1000:
                        total_sales = f"₹ {sales_amount/1000:.2f} K"

                    else:
                        total_sales = f"₹ {sales_amount:.2f}"

                else:

                    total_sales = (
                        "Price Column Missing"
                    )

                transactions = len(df)

                preview_data = (
                    df.head()
                    .to_html(index=False)
                )

                if price_col:

                    cust_features = (
                        df.groupby(customer_col)
                        .agg({
                            quantity_col: "sum",
                            price_col: "sum"
                        })
                        .reset_index()
                    )

                    if len(cust_features) >= 3:

                        kmeans = KMeans(
                            n_clusters=min(5, len(cust_features)),
                            random_state=42,
                            n_init=10
                        )

                        cust_features["Cluster"] = (
                            kmeans.fit_predict(
                                cust_features[
                                    [
                                        quantity_col,
                                        price_col
                                    ]
                                ]
                            )
                        )

                        cluster_order = (
                            cust_features
                            .groupby("Cluster")
                            [price_col]
                            .mean()
                            .sort_values()
                            .index
                        )

                        labels = get_dynamic_labels(kmeans.n_clusters)

                        mapping = {
                            cluster_order[i]:
                            labels[i]
                            for i in range(
                                len(cluster_order)
                            )
                        }

                        cust_features["Segment"] = (
                            cust_features["Cluster"]
                            .map(mapping)
                        )

                        segmentation_table = (
                            cust_features[
                                [
                                    customer_col,
                                    quantity_col,
                                    price_col,
                                    "Segment"
                                ]
                            ]
                            .to_html(index=False)
                        )
                        if date_col and price_col:
                            try:
                                forecast_df = df.copy()
                                forecast_df[date_col] = pd.to_datetime(
                                    forecast_df[date_col],
                                    errors="coerce"
                                )
                                forecast_df.dropna(
                                    subset=[date_col],
                                    inplace=True
                                )
                                forecast_df["Sales"] = (
                                     forecast_df[quantity_col]
                                    * forecast_df[price_col]
                                )
                                sales_ts = (
                                    forecast_df.groupby(
                                        forecast_df[date_col].dt.to_period("M")
                                    )["Sales"]
                                    .sum()
                                )

                                sales_ts.index = sales_ts.index.to_timestamp()

                                if len(sales_ts) >= 6:

                                    model = ARIMA(
                                        sales_ts,
                                        order=(1, 1, 1)
                                    )

                                    forecast = (
                                        model.fit()
                                        .forecast(steps=6)
                                    )

                                    fig = go.Figure()

                                    fig.add_trace(
                                        go.Scatter(
                                            x=sales_ts.index,
                                            y=sales_ts.values,
                                            mode="lines+markers",
                                            name="Historical Sales",
                                            line=dict(
                                                color="#4F46E5",
                                                width=4,
                                                shape="spline"
                                            ),
                                            marker=dict(
                                                size=8,
                                                color="#4F46E5",
                                                line=dict(
                                                    color="white",
                                                    width=2
                                                )
                                            ),
                                            fill="tozeroy",
                                            fillcolor="rgba(79,70,229,0.08)"
                                        )
                                    )

                                    fig.add_trace(
                                    go.Scatter(
                                        x=forecast.index,
                                        y=forecast.values,
                                        mode="lines+markers",
                                        name="Forecasted Sales",
                                        line=dict(
                                            color="#06B6D4",
                                            width=4,
                                            dash="dash",
                                            shape="spline"
                                        ),
                                        marker=dict(
                                            size=8,
                                            color="#06B6D4",
                                            line=dict(
                                                color="white",
                                                width=2
                                                )
                                            )
                                        )
                                    )

                                    fig.add_shape(
                                        type="line",
                                        x0=forecast.index[0].to_pydatetime(),
                                        x1=forecast.index[0].to_pydatetime(),
                                        y0=min(sales_ts.min(), forecast.min()),
                                        y1=max(sales_ts.max(), forecast.max()),
                                        line=dict(
                                            color="red",
                                            width=2,
                                            dash="dash"
                                        )
                                    )

                                    fig.update_layout(

                                        title=dict(
                                            text="📈 AI Sales Forecast",
                                            x=0.5,
                                            font=dict(
                                                family="Poppins",
                                                size=26,
                                                color="#111827"
                                            )
                                        ),

                                        template="plotly_white",

                                        height=600,

                                        hovermode="x unified",

                                        paper_bgcolor="rgba(0,0,0,0)",

                                        plot_bgcolor="white",

                                        font=dict(
                                            family="Poppins",
                                            size=14
                                        ),

                                        legend=dict(
                                            orientation="h",
                                            y=1.08,
                                            x=1,
                                            xanchor="right",
                                            bgcolor="rgba(255,255,255,0.7)"
                                        ),

                                        margin=dict(
                                            l=30,
                                            r=30,
                                            t=70,
                                            b=30
                                        ),

                                        xaxis=dict(
                                            title="Month",
                                            showgrid=True,
                                            gridcolor="rgba(0,0,0,0.05)",
                                            zeroline=False
                                        ),

                                        yaxis=dict(
                                            title="Sales",
                                            showgrid=True,
                                            gridcolor="rgba(0,0,0,0.05)",
                                            zeroline=False
                                        )
                                    )

                                    forecast_chart = fig.to_html(
                                        full_html=False,
                                        config={
                                            "displayModeBar": False,
                                            "responsive": True
                                        }
                                    )

                                else:

                                    forecast_message = (
                                        "At least 6 months of data required."
                                    )

                            except Exception as e:
                                forecast_message = str(e)

            else:

                message = (
                    "Required columns could not be detected."
                )

    return render_template(
        "index.html",
        filename=filename,
        total_customers=total_customers,
        total_products=total_products,
        total_sales=total_sales,
        transactions=transactions,
        preview_data=preview_data,
        segmentation_table=segmentation_table,
        customers=customers,
        selected_customer=selected_customer,
        purchased_products=purchased_products,
        recommended_products=recommended_products,
        message=message,
        forecast_chart=forecast_chart,
        forecast_message=forecast_message
    )

if __name__ == "__main__":
    app.run(debug=True)
