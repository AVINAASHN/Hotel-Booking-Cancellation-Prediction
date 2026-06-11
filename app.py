import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
model = joblib.load("hotel_cancellation_logistic.pkl")

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Predict whether a hotel booking is likely to be canceled.")

# ==========================
# User Inputs
# ==========================

col1, col2 = st.columns(2)

with col1:
    hotel = st.selectbox(
        "Hotel Type",
        ["Resort Hotel", "City Hotel"]
    )

    lead_time = st.number_input(
        "Lead Time",
        min_value=0,
        value=50
    )

    arrival_date_year = st.selectbox(
        "Arrival Year",
        [2015, 2016, 2017]
    )

    arrival_date_month = st.selectbox(
        "Arrival Month",
        ["January","February","March","April","May",
         "June","July","August","September",
         "October","November","December"]
    )

    arrival_date_week_number = st.number_input(
        "Arrival Week Number",
        min_value=1,
        max_value=53,
        value=25
    )

    arrival_date_day_of_month = st.number_input(
        "Arrival Day",
        min_value=1,
        max_value=31,
        value=15
    )

    stays_in_weekend_nights = st.number_input(
        "Weekend Nights",
        min_value=0,
        value=1
    )

    stays_in_week_nights = st.number_input(
        "Week Nights",
        min_value=0,
        value=3
    )

    adults = st.number_input(
        "Adults",
        min_value=1,
        value=2
    )

    children = st.number_input(
        "Children",
        min_value=0,
        value=0
    )

    babies = st.number_input(
        "Babies",
        min_value=0,
        value=0
    )

with col2:

    meal = st.selectbox(
        "Meal Plan",
        ["BB", "HB", "FB", "SC"]
    )

    country = st.text_input(
        "Country Code",
        value="PRT"
    )

    market_segment = st.selectbox(
        "Market Segment",
        ["Online TA", "Offline TA/TO", "Direct", "Corporate"]
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        ["TA/TO", "Direct", "Corporate"]
    )

    is_repeated_guest = st.selectbox(
        "Repeated Guest",
        [0, 1]
    )

    previous_cancellations = st.number_input(
        "Previous Cancellations",
        min_value=0,
        value=0
    )

    previous_bookings_not_canceled = st.number_input(
        "Previous Successful Bookings",
        min_value=0,
        value=0
    )

    reserved_room_type = st.text_input(
        "Reserved Room Type",
        value="A"
    )

    assigned_room_type = st.text_input(
        "Assigned Room Type",
        value="A"
    )

    booking_changes = st.number_input(
        "Booking Changes",
        min_value=0,
        value=0
    )

    deposit_type = st.selectbox(
        "Deposit Type",
        ["No Deposit", "Refundable", "Non Refund"]
    )

    days_in_waiting_list = st.number_input(
        "Days in Waiting List",
        min_value=0,
        value=0
    )

    customer_type = st.selectbox(
        "Customer Type",
        ["Transient", "Contract", "Group", "Transient-Party"]
    )

    adr = st.number_input(
        "ADR",
        min_value=0.0,
        value=100.0
    )

    required_car_parking_spaces = st.number_input(
        "Parking Spaces",
        min_value=0,
        value=0
    )

    total_of_special_requests = st.number_input(
        "Special Requests",
        min_value=0,
        value=1
    )

# ==========================
# Prediction
# ==========================

if st.button("Predict Cancellation"):

    input_df = pd.DataFrame({
        'hotel':[hotel],
        'lead_time':[lead_time],
        'arrival_date_year':[arrival_date_year],
        'arrival_date_month':[arrival_date_month],
        'arrival_date_week_number':[arrival_date_week_number],
        'arrival_date_day_of_month':[arrival_date_day_of_month],
        'stays_in_weekend_nights':[stays_in_weekend_nights],
        'stays_in_week_nights':[stays_in_week_nights],
        'adults':[adults],
        'children':[children],
        'babies':[babies],
        'meal':[meal],
        'country':[country],
        'market_segment':[market_segment],
        'distribution_channel':[distribution_channel],
        'is_repeated_guest':[is_repeated_guest],
        'previous_cancellations':[previous_cancellations],
        'previous_bookings_not_canceled':[previous_bookings_not_canceled],
        'reserved_room_type':[reserved_room_type],
        'assigned_room_type':[assigned_room_type],
        'booking_changes':[booking_changes],
        'deposit_type':[deposit_type],
        'days_in_waiting_list':[days_in_waiting_list],
        'customer_type':[customer_type],
        'adr':[adr],
        'required_car_parking_spaces':[required_car_parking_spaces],
        'total_of_special_requests':[total_of_special_requests]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Cancellation Probability",
        f"{probability*100:.2f}%"
    )

    if probability < 0.3:
        st.success("✅ Low Risk of Cancellation")
    elif probability < 0.7:
        st.warning("⚠️ Medium Risk of Cancellation")
    else:
        st.error("🚨 High Risk of Cancellation")

    if prediction == 1:
        st.error("Booking Likely to be Canceled")
    else:
        st.success("Booking Likely to be Honored")