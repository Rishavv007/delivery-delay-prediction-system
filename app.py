import streamlit as st
import requests

st.set_page_config(page_title="Delivery Delay Predictor", page_icon="🚚", layout="centered")

st.title("🚚 Delivery Delay Predictor")
st.markdown("Enter shipment details below to predict the probability of a delivery delay.")

st.subheader("Shipment Information")

col1, col2 = st.columns(2)

with col1:
    days_scheduled = st.number_input("Days for shipment (scheduled)", min_value=0, value=4)
    shipping_mode = st.selectbox(
        "Shipping Mode", 
        options=["Standard Class", "First Class", "Second Class", "Same Day"]
    )
    order_item_qty = st.number_input("Order Item Quantity", min_value=1, value=1)
    
with col2:
    REGION_COUNTRY_MAP = {
        "Southeast Asia": ["Indonesia", "Thailand", "Vietnam", "Malaysia", "Singapore", "Philippines"],
        "North America": ["United States", "Canada", "Mexico"],
        "Western Europe": ["Germany", "United Kingdom", "France", "Netherlands", "Switzerland", "Belgium"],
        "Southern Europe": ["Italy", "Spain"],
        "Eastern Europe": ["Poland", "Russia"],
        "Northern Europe": ["Sweden"],
        "Eastern Asia": ["China", "Japan", "South Korea", "Taiwan"],
        "Southern Asia": ["India"],
        "South America": ["Brazil", "Argentina"],
        "Oceania": ["Australia"],
        "Middle East": ["Turkey"]
    }
    
    # Region Dropdown
    order_region = st.selectbox("Order Region", options=list(REGION_COUNTRY_MAP.keys()))
    
    # Dependent Country Dropdown
    available_countries = REGION_COUNTRY_MAP[order_region]
    order_country = st.selectbox("Order Country", options=sorted(available_countries))

    sales = st.number_input("Sales ($)", min_value=0.0, value=327.75, step=10.0)
    
submit_button = st.button(label="Predict Delay Risk")

if submit_button:
    # Prepare payload
    payload = {
        "Days for shipment (scheduled)": days_scheduled,
        "Shipping Mode": shipping_mode,
        "Order Region": order_region,
        "Order Country": order_country,
        "Order Item Quantity": order_item_qty,
        "Sales": sales
    }
    
    with st.spinner("Analyzing risk..."):
        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prob = result.get("delay_probability", 0)
                risk_level = result.get("risk_level", "UNKNOWN")
                
                # Display Results beautifully
                st.subheader("Prediction Results")
                
                if risk_level == "LOW":
                    st.success(f"**Risk Level: {risk_level}**")
                elif risk_level == "MEDIUM":
                    st.warning(f"**Risk Level: {risk_level}**")
                else:
                    st.error(f"**Risk Level: {risk_level}**")
                    
                st.metric(label="Delay Probability", value=f"{prob * 100:.2f}%")
                
                # Progress bar visualization
                st.progress(prob)
                
            else:
                st.error(f"Error from API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the prediction API. Ensure the FastAPI server is running on http://localhost:8000.")
