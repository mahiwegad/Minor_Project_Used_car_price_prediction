import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("car_price_model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_artifacts()

st.title("🚗 Used Car Price Predictor (India)")
st.caption(
    f"Model: {meta['model_name']} | R² = {meta['r2']:.3f} | "
    f"typical error (MAE) ≈ ₹{meta['mae']:,.0f}"
)
st.write("Enter the car's details to get an estimated resale price.")

med = meta["medians"]

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", meta["brands"], index=meta["brands"].index("Maruti") if "Maruti" in meta["brands"] else 0)
    fuel = st.selectbox("Fuel type", meta["fuels"])
    transmission = st.selectbox("Transmission", meta["transmissions"])
    seller_type = st.selectbox("Seller type", meta["seller_types"])
    owner = st.selectbox("Owner", meta["owners"])
with col2:
    car_age = st.slider("Car age (years)", 0, 25, 5)
    km_driven = st.number_input("Kilometres driven", min_value=0, max_value=500000, value=50000, step=1000)
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=round(med["mileage"], 1))
    engine = st.number_input("Engine (CC)", min_value=500, max_value=6000, value=int(med["engine"]), step=50)
    max_power = st.number_input("Max power (bhp)", min_value=20.0, max_value=700.0, value=round(med["max_power"], 1))
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "car_age": car_age, "km_driven": km_driven, "mileage": mileage,
        "engine": engine, "max_power": max_power, "seats": seats,
        "brand": brand, "fuel": fuel, "seller_type": seller_type,
        "transmission": transmission, "owner": owner,
    }])
    price = float(model.predict(row)[0])
    low, high = price - meta["mae"], price + meta["mae"]
    st.success(f"Estimated price: ₹{price:,.0f}")
    st.write(f"Likely range (± typical error): ₹{max(low, 0):,.0f} to ₹{high:,.0f}")

st.divider()
st.caption("Built with Python, scikit-learn and Streamlit. Estimates are indicative, not a valuation.")
