import streamlit as st

st.set_page_config(
    page_title="IT5006 E-Commerce Analytics Demo",
    page_icon="📦",
    layout="wide",
)

st.title("📦 IT5006 E-Commerce Analytics")
st.caption("Simple Streamlit deployment test using fake data only")

region = st.sidebar.selectbox(
    "Region",
    ["All", "São Paulo", "Rio de Janeiro", "Minas Gerais"],
)

orders = st.sidebar.slider("Fake number of orders", 100, 1000, 500, 50)

col1, col2, col3 = st.columns(3)
col1.metric("Orders", f"{orders:,}")
col2.metric("Avg. Delivery Time", "12.4 days")
col3.metric("Negative Review Rate", "14.8%")

st.subheader("Fake Monthly Orders")
st.line_chart(
    {
        "Orders": [320, 410, 390, 480, 520, 610, 590, 670, 720, 760, 810, 850]
    }
)

st.subheader("Fake Review Score Distribution")
st.bar_chart(
    {
        "Reviews": [120, 55, 80, 160, 585]
    },
    x_label="Review score group",
    y_label="Number of reviews",
)

st.info(
    f"Demo filter selected: {region}. This app contains no real Olist data yet. "
    "Its purpose is only to verify that GitHub → Streamlit Community Cloud deployment works."
)
