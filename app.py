import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hackathon Q-Commerce Dashboard", layout="wide")
st.title("📊 Quick Commerce Analytics Dashboard")

st.sidebar.header("Upload Datasets")
sales_file = st.sidebar.file_uploader("sales.csv", type=["csv"])
stock_file = st.sidebar.file_uploader("inventory.csv", type=["csv"])


if sales_file and stock_file:
    sales_df = pd.read_csv(sales_file)
    stock_df = pd.read_csv(stock_file)

    # Ensure 'product_id' columns are of the same type (string) for merging
    sales_df['product_id'] = sales_df['product_id'].astype(str)
    stock_df['product_id'] = stock_df['product_id'].astype(str)

    st.success("✅ Data uploaded successfully!")

    st.subheader("🔹 Scenario 1: Top 5 Selling Products")


    top_products = (
        sales_df.groupby(["product_id", "product_name"])["units_sold"]
        .sum()
        .reset_index()
        .sort_values(by="units_sold", ascending=False)
        .head(5)
    )

    st.write("Here are the **Top 5 Products by Units Sold**:")
    st.dataframe(top_products)

    st.bar_chart(top_products.set_index("product_name")["units_sold"])

    st.subheader("🔹 Scenario 2: High Demand but Low Stock")


    demand_df = (
        sales_df.groupby("product_id")["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"units_sold": "total_demand"})
    )
    merged_df = pd.merge(demand_df, stock_df, on="product_id", how="inner")

    threshold = st.slider("Set Low Stock Threshold", 0, 100, 20)

    high_demand_low_stock = merged_df[
        (merged_df["total_demand"] > merged_df["total_demand"].mean())
        & (merged_df["stock_quantity"] < threshold)
    ][["product_id", "product_name", "total_demand", "stock_quantity", "city_name"]]

    st.write("⚠️ Products with **High Demand but Low Stock**:")
    st.dataframe(high_demand_low_stock)

else:
    st.info("👆 Please upload both datasets to start.")
