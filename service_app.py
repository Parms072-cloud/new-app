import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------- PAGE CONFIG ---------
st.set_page_config(
    page_title="Go Auto – Time to Next Service Predictor",
    page_icon="🚗",
    layout="wide",
)

# --------- LOAD MODEL & ARTIFACTS ---------
@st.cache_resource
def load_model():
    artifacts = joblib.load("service_interval_model.joblib")
    model = artifacts["model"]
    label_encoders = artifacts["label_encoders"]
    feature_columns = artifacts["feature_columns"]
    return model, label_encoders, feature_columns

model, label_encoders, feature_columns = load_model()

CATEGORICAL_COLS = list(label_encoders.keys())
DATE_DERIVED_COLS = ["service_year", "service_month"]
# --------- UI HEADER ---------
st.title("Vehicle Service Due Date Predictor")
st.write(
    "Enter the vehicle details and the **last service date** to estimate "
    "the **next service due date**."
)

# ---- Last service date ----
service_date = st.date_input(
    "Last service date",
    value=dt.date.today(),
    help="The date when the vehicle was last serviced."
)

service_year = service_date.year
service_month = service_date.month
# --------- VEHICLE INFORMATION (CATEGORICAL) ---------
st.subheader("Vehicle Information")

cat_inputs = {}
for col in CATEGORICAL_COLS:
    le = label_encoders[col]
    options = list(le.classes_)
    default_idx = 0
    cat_inputs[col] = st.selectbox(
        col.replace("_", " ").capitalize(),
        options=options,
        index=default_idx if default_idx < len(options) else 0,
    )
# --------- NUMERIC INPUTS ---------
st.subheader("Additional Details")

numeric_features = [
    col for col in feature_columns
    if col not in CATEGORICAL_COLS + DATE_DERIVED_COLS
]

numeric_inputs = {}
for col in numeric_features:
    # choose nicer defaults for some fields
    if col == "mileage":
        default_val = 50000.0
    elif col == "distance":
        default_val = 10.0
    elif col in ("service_count_so_far", "prev_interval"):
        default_val = 2.0
    else:
        default_val = 0.0

    numeric_inputs[col] = st.number_input(
        col.replace("_", " ").capitalize(),
        value=default_val,
        step=1.0,
        format="%.2f",
    )
# --------- BUILD SINGLE-ROW DATAFRAME FOR PREDICTION ---------
input_data = pd.DataFrame(columns=feature_columns)
input_data.loc[0] = 0  # initialize all zeros

# date-derived
if "service_year" in feature_columns:
    input_data.loc[0, "service_year"] = service_year
if "service_month" in feature_columns:
    input_data.loc[0, "service_month"] = service_month

# categorical (encode using label_encoders)
for col, raw_val in cat_inputs.items():
    le = label_encoders[col]
    input_data.loc[0, col] = le.transform([raw_val])[0]

# numeric
for col, val in numeric_inputs.items():
    input_data.loc[0, col] = float(val)
# --------- PREDICTION ---------
if st.button("Predict Next Service Due Date"):

    raw_interval = float(model.predict(input_data)[0])

    # guard: avoid negative / zero months for scheduling
    if raw_interval < 0:
        raw_interval = 0.0
    interval_for_date = max(1, int(round(raw_interval)))

    next_service_ts = pd.Timestamp(service_date) + pd.DateOffset(
        months=interval_for_date
    )
    next_service_date = next_service_ts.date()

    st.markdown("---")
    st.subheader("Prediction Result")

    st.write(f"**Estimated interval (model output):** {raw_interval:.2f} months")
    st.write(
        f"**Interval used to compute due date (min 1 month):** "
        f"{interval_for_date} month(s)"
    )
    st.write(
        f"👉 **Next service due date is likely around:** "
        f"🗓️ **{next_service_date.strftime('%Y-%m-%d')}**"
    )

    st.info(
        "This prediction is based on historical Go Auto service patterns "
        "for similar vehicles. Use it together with manufacturer "
        "recommendations and dealer advice."
    )
