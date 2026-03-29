import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import joblib

st.set_page_config(page_title="Grab Data Dashboard", layout="wide")

engine = create_engine("postgresql://grab:grab123@postgres:5432/grabdb")

@st.cache_data(ttl=60)
def load_data():
    df = pd.read_sql("SELECT * FROM orders_transformed", engine)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6])
    df["orders_per_driver"] = df["orders"] / df["driver_online"]
    return df

df = load_data()

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("Filters")
cities = ["All"] + sorted(df["city"].unique().tolist())
selected_city = st.sidebar.selectbox("City", cities)
selected_weather = st.sidebar.multiselect("Weather", df["weather"].unique().tolist(), default=df["weather"].unique().tolist())

filtered = df.copy()
if selected_city != "All":
    filtered = filtered[filtered["city"] == selected_city]
if selected_weather:
    filtered = filtered[filtered["weather"].isin(selected_weather)]

# ── Header ───────────────────────────────────────────────
st.title("Grab Orders Dashboard")
st.caption("Country & Marketing Analytics — Operations Intelligence")

# ── KPIs ─────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Orders", f"{int(filtered['orders'].sum()):,}")
col2.metric("Avg Orders/Day", f"{int(filtered['orders'].mean()):,}")
col3.metric("Cities", filtered["city"].nunique())
col4.metric("Promo Days", int(filtered["promo"].sum()))
promo_lift = 0
if filtered["promo"].nunique() > 1:
    p1 = filtered[filtered["promo"]==1]["orders"].mean()
    p0 = filtered[filtered["promo"]==0]["orders"].mean()
    promo_lift = round((p1 - p0) / p0 * 100, 1)
col5.metric("Promo Lift", f"{promo_lift}%")

st.divider()

# ── Tab layout ───────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Analytics", "🌦 Segmentation", "🤖 Demand Prediction", "💬 AI Analyst"])

# ── Tab 1: Analytics ─────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orders Trend")
        trend = filtered.groupby("date")["orders"].sum().reset_index()
        st.line_chart(trend.set_index("date"))

    with col2:
        st.subheader("Orders by City")
        by_city = filtered.groupby("city")["orders"].sum().reset_index()
        st.bar_chart(by_city.set_index("city"))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Promo vs No-Promo Orders")
        promo_df = filtered.groupby("promo")["orders"].mean().reset_index()
        promo_df["promo"] = promo_df["promo"].map({0: "No Promo", 1: "Promo"})
        st.bar_chart(promo_df.set_index("promo"))

    with col4:
        st.subheader("Orders per Driver (Efficiency)")
        eff = filtered.groupby("city")["orders_per_driver"].mean().reset_index()
        st.bar_chart(eff.set_index("city"))

# ── Tab 2: Segmentation ───────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Weather Impact on Orders")
        weather_df = filtered.groupby("weather")["orders"].mean().sort_values(ascending=False).reset_index()
        st.bar_chart(weather_df.set_index("weather"))

    with col2:
        st.subheader("Weekend vs Weekday")
        wk = filtered.groupby("is_weekend")["orders"].mean().reset_index()
        wk["is_weekend"] = wk["is_weekend"].map({True: "Weekend", False: "Weekday"})
        st.bar_chart(wk.set_index("is_weekend"))

    st.subheader("Orders by Day of Week")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = filtered.groupby("day_of_week")["orders"].mean().reindex(dow_order).reset_index()
    st.bar_chart(dow.set_index("day_of_week"))

    st.subheader("Raw Data")
    st.dataframe(filtered.sort_values("date", ascending=False).head(50), use_container_width=True)

# ── Tab 3: Demand Prediction ──────────────────────────────
with tab3:
    st.subheader("Demand Forecasting Model")
    try:
        model = joblib.load("analytics/demand_model.pkl")
        le_city = joblib.load("analytics/le_city.pkl")
        le_weather = joblib.load("analytics/le_weather.pkl")
        features = joblib.load("analytics/model_features.pkl")

        col1, col2 = st.columns(2)
        with col1:
            pred_city = st.selectbox("City", sorted(df["city"].unique().tolist()))
            pred_weather = st.selectbox("Weather", sorted(df["weather"].unique().tolist()))
            pred_promo = st.slider("Promo (0 = no promo, 1 = promo)", 0, 1)
        with col2:
            pred_drivers = st.slider("Drivers Online", 1000, 3000, 2000)
            pred_dow = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
            pred_weekend = 1 if pred_dow >= 5 else 0
            pred_month = st.slider("Month", 1, 12, 1)

        city_enc = le_city.transform([pred_city])[0]
        weather_enc = le_weather.transform([pred_weather])[0]

        input_data = pd.DataFrame([[pred_promo, pred_drivers, city_enc, weather_enc, pred_dow, pred_weekend, pred_month]], columns=features)
        prediction = model.predict(input_data)[0]

        st.metric("Predicted Orders", f"{int(prediction):,}")

        # Feature importance
        st.subheader("Feature Importance")
        fi = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
        st.bar_chart(fi.set_index("Feature"))

    except Exception as e:
        st.warning(f"Train the model first: docker compose run analytics python analytics/demand_model.py")
        st.caption(str(e))

# ── Tab 4: AI Analyst ─────────────────────────────────────
with tab4:
    st.subheader("AI Analyst — Ask questions about the data")
    st.caption("Powered by GPT-5 nano (KADA)")

    summary_stats = f"""
Dataset summary:
- Rows: {len(filtered)}, Cities: {filtered['city'].unique().tolist()}
- Date range: {filtered['date'].min().date()} to {filtered['date'].max().date()}
- Total orders: {filtered['orders'].sum()}, Avg/day: {filtered['orders'].mean():.1f}
- Promo lift: {promo_lift}%
- Weather types: {filtered['weather'].unique().tolist()}
- Avg orders by city: {filtered.groupby('city')['orders'].mean().round(1).to_dict()}
- Avg orders by weather: {filtered.groupby('weather')['orders'].mean().round(1).to_dict()}
"""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a business question about Grab orders...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    from openai import OpenAI

                    api_key = os.environ.get("KADA_API_KEY", "")
                    base_url = os.environ.get("KADA_BASE_URL", "")

                    if not api_key or not base_url:
                        st.error("Set KADA_API_KEY and KADA_BASE_URL in your .env file")
                    else:
                        client = OpenAI(api_key=api_key, base_url=base_url)

                        messages = [
                            {
                                "role": "system",
                                "content": f"You are a senior data scientist at Grab. Answer using data analytics perspective. Be concise and actionable.\n\n{summary_stats}"
                            }
                        ]
                        for msg in st.session_state.chat_history:
                            messages.append({"role": msg["role"], "content": msg["content"]})

                        response = client.chat.completions.create(
                            model="openai/gpt-5-nano",
                            messages=messages
                        )
                        answer = response.choices[0].message.content
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"GPT API error: {e}")