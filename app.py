import mlflow, mlflow.sklearn, numpy as np, streamlit as st

MODEL_NAME = "CaliforniaHousingPredictor"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 California House Price Predictor")
st.caption("MLflow + Ridge Regression")


@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")


try:
    model = load_model()
    st.success("Model loaded ✓")
except Exception as e:
    st.error(f"Model not available: {e}")
    st.info("Run: python train_ridge.py --sweep → python model.py")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    med_inc = st.number_input("Median Income (× $10k)", 0.0, 15.0, 5.0, 0.1)
    house_age = st.number_input("House Age", 1.0, 60.0, 25.0, 1.0)
    ave_rooms = st.number_input("Avg Rooms", 1.0, 15.0, 5.0, 0.1)
    ave_bedrms = st.number_input("Avg Bedrooms", 0.5, 5.0, 1.0, 0.1)
with c2:
    population = st.number_input("Population", 1.0, 5000.0, 1000.0, 10.0)
    ave_occup = st.number_input("Avg Occupancy", 1.0, 10.0, 3.0, 0.1)
    latitude = st.number_input("Latitude", 32.0, 42.0, 34.0, 0.01)
    longitude = st.number_input("Longitude", -125.0, -114.0, -118.0, 0.01)

if st.button("Predict Price", type="primary", use_container_width=True):
    x = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                   population, ave_occup, latitude, longitude]])
    price = float(model.predict(x)[0]) * 100_000
    st.success(f"### Predicted Price: **${price:,.0f}**")