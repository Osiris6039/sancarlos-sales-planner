
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_DIR = "data"
MODELS_DIR = "models"
SALES_DATA_PATH = os.path.join(DATA_DIR, "sales_data.csv")
EVENTS_DATA_PATH = os.path.join(DATA_DIR, "events_data.csv")
SALES_MODEL_PATH = os.path.join(MODELS_DIR, "sales_forecast_model.pkl")
CUSTOMERS_MODEL_PATH = os.path.join(MODELS_DIR, "customers_forecast_model.pkl")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Data Loading and Saving Functions ---
def load_sales_data():
    """Loads sales data from CSV, creates a new one if it doesn't exist."""
    if not os.path.exists(SALES_DATA_PATH):
        df = pd.DataFrame(columns=[
            'Date', 'Sales', 'Customers', 'Add_on_Sales', 'Weather'
        ])
        df['Date'] = pd.to_datetime(df['Date'])
        df.to_csv(SALES_DATA_PATH, index=False)
    else:
        df = pd.read_csv(SALES_DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def save_sales_data(df):
    """Saves sales data to CSV."""
    df.to_csv(SALES_DATA_PATH, index=False)

def load_events_data():
    """Loads events data from CSV, creates a new one if it doesn't exist."""
    if not os.path.exists(EVENTS_DATA_PATH):
        df = pd.DataFrame(columns=['Event_Date', 'Event_Name', 'Impact'])
        df['Event_Date'] = pd.to_datetime(df['Event_Date'])
        df.to_csv(EVENTS_DATA_PATH, index=False)
    else:
        df = pd.read_csv(EVENTS_DATA_PATH)
        df['Event_Date'] = pd.to_datetime(df['Event_Date'])
    return df

def save_events_data(df):
    """Saves events data to CSV."""
    df.to_csv(EVENTS_DATA_PATH, index=False)

# --- AI Model Training and Prediction Functions ---
def preprocess_data(df_sales, df_events):
    """
    Preprocesses sales and events data for model training.
    Creates features like day of week, month, year, is_weekend, weather encoding, and event impact.
    """
    if df_sales.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_sales.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Time-based features
    df['day_of_week'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int) # 1 for weekend, 0 for weekday

    # Weather encoding
    df = pd.get_dummies(df, columns=['Weather'], prefix='weather', dummy_na=False)

    # Merge with events data
    if not df_events.empty:
        df_events['Event_Date'] = pd.to_datetime(df_events['Event_Date'])
        # Create an 'is_event' flag and an 'event_impact' numerical value
        df['is_event'] = 0
        df['event_impact_score'] = 0.0 # Numerical representation of impact

        # Map Impact to scores
        impact_map = {'Low': 0.1, 'Medium': 0.5, 'High': 1.0}
        df_events['Impact_Score'] = df_events['Impact'].map(impact_map).fillna(0)

        # Merge based on date
        for index, row in df_events.iterrows():
            event_date = row['Event_Date']
            impact_score = row['Impact_Score']
            df.loc[df['Date'] == event_date, 'is_event'] = 1
            df.loc[df['Date'] == event_date, 'event_impact_score'] = impact_score

    # Lag features for sales and customers (e.g., previous day's sales)
    # Ensure data is sorted by date before creating lags
    df = df.sort_values('Date')
    df['Sales_Lag1'] = df['Sales'].shift(1)
    df['Customers_Lag1'] = df['Customers'].shift(1)
    df['Sales_Lag7'] = df['Sales'].shift(7) # Previous week's same day
    df['Customers_Lag7'] = df['Customers'].shift(7)

    # Fill NaN values created by shifting. For features that are inputs, fill with 0 or mean/median.
    # For simplicity here, fill with 0. In a real scenario, consider more sophisticated imputation.
    df = df.fillna(0)

    # Features to use for training (excluding 'Date', 'Sales', 'Customers', 'Add_on_Sales')
    # and ensuring 'Add_on_Sales' is explicitly NOT used for core sales/customer prediction to prevent overfitting.
    feature_columns = [
        'day_of_week', 'day_of_year', 'month', 'year', 'week_of_year', 'is_weekend',
        'Sales_Lag1', 'Customers_Lag1', 'Sales_Lag7', 'Customers_Lag7',
        'is_event', 'event_impact_score'
    ]
    # Add weather columns dynamically
    weather_cols = [col for col in df.columns if col.startswith('weather_')]
    feature_columns.extend(weather_cols)

    # Ensure all required weather columns exist (for consistent feature set)
    # Define all possible weather conditions for consistency
    all_weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy'] # Extend as needed
    for cond in all_weather_conditions:
        col_name = f'weather_{cond}'
        if col_name not in df.columns:
            df[col_name] = 0

    # Filter feature_columns to only include columns that actually exist in the dataframe
    final_feature_columns = [col for col in feature_columns if col in df.columns]

    X = df[final_feature_columns]
    y_sales = df['Sales']
    y_customers = df['Customers']

    # Store the final feature column names for prediction
    st.session_state['feature_columns'] = final_feature_columns
    st.session_state['all_weather_conditions'] = all_weather_conditions

    return X, y_sales, y_customers

def train_models(X, y_sales, y_customers):
    """Trains Sales and Customers RandomForestRegressor models and saves them."""
    if X.empty:
        st.warning("Not enough data to train the model. Please add more sales records.")
        return None, None

    sales_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    customers_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Train sales model
    sales_model.fit(X, y_sales)
    joblib.dump(sales_model, SALES_MODEL_PATH)

    # Train customers model
    customers_model.fit(X, y_customers)
    joblib.dump(customers_model, CUSTOMERS_MODEL_PATH)

    st.success("AI models trained successfully!")
    return sales_model, customers_model

def load_or_train_models():
    """Loads models if they exist, otherwise trains them."""
    sales_model = None
    customers_model = None
    if os.path.exists(SALES_MODEL_PATH) and os.path.exists(CUSTOMERS_MODEL_PATH):
        try:
            sales_model = joblib.load(SALES_MODEL_PATH)
            customers_model = joblib.load(CUSTOMERS_MODEL_PATH)
            st.info("AI models loaded from disk.")
        except Exception as e:
            st.error(f"Error loading models: {e}. Retraining.")
            sales_df = st.session_state.sales_data
            events_df = st.session_state.events_data
            X, y_sales, y_customers = preprocess_data(sales_df, events_df)
            sales_model, customers_model = train_models(X, y_sales, y_customers)
    else:
        st.info("No models found. Training AI models...")
        sales_df = st.session_state.sales_data
        events_df = st.session_state.events_data
        X, y_sales, y_customers = preprocess_data(sales_df, events_df)
        sales_model, customers_model = train_models(X, y_sales, y_customers)
    return sales_model, customers_model


def generate_forecast(sales_df, events_df, sales_model, customers_model, num_days=10):
    """
    Generates sales and customer forecasts for the next N days.
    Assumes future weather is known or defaulted.
    """
    if sales_model is None or customers_model is None:
        st.warning("Models are not trained. Please add sufficient data and retrain.")
        return pd.DataFrame()

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    forecast_dates = [today + timedelta(days=i) for i in range(1, num_days + 1)]

    # Create a DataFrame for future features
    future_df = pd.DataFrame({'Date': forecast_dates})
    future_df['day_of_week'] = future_df['Date'].dt.dayofweek
    future_df['day_of_year'] = future_df['Date'].dt.dayofyear
    future_df['month'] = future_df['Date'].dt.month
    future_df['year'] = future_df['Date'].dt.year
    future_df['week_of_year'] = future_df['Date'].dt.isocalendar().week.astype(int)
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)

    # For future weather, we'll assume 'Sunny' for simplicity or allow user input.
    # In a real app, you might integrate a weather API.
    # For now, default to 'Sunny'
    future_df['Weather'] = 'Sunny' # Default future weather

    # Add dummy weather columns for consistency
    all_weather_conditions = st.session_state.get('all_weather_conditions', [])
    for cond in all_weather_conditions:
        col_name = f'weather_{cond}'
        future_df[col_name] = (future_df['Weather'] == cond).astype(int)

    # Add event impact for future dates
    future_df['is_event'] = 0
    future_df['event_impact_score'] = 0.0
    if not events_df.empty:
        for index, row in events_df.iterrows():
            event_date = row['Event_Date'].to_datetime64()
            impact_map = {'Low': 0.1, 'Medium': 0.5, 'High': 1.0}
            impact_score = impact_map.get(row['Impact'], 0)
            if event_date in future_df['Date'].dt.to_datetime64().values:
                future_df.loc[future_df['Date'].dt.to_datetime64() == event_date, 'is_event'] = 1
                future_df.loc[future_df['Date'].dt.to_datetime64() == event_date, 'event_impact_score'] = impact_score

    # Prepare lagged features for forecasting. This requires a loop or iterative prediction.
    # The initial lags for the first forecast day will be based on the last known actual data.
    # For subsequent days, they will be based on the *predicted* values of the previous day.
    forecast_results = []
    current_sales_lag1 = sales_df['Sales'].iloc[-1] if not sales_df.empty else 0
    current_customers_lag1 = sales_df['Customers'].iloc[-1] if not sales_df.empty else 0
    
    # Get last 7 days of sales for Sales_Lag7 and Customers_Lag7.
    # If less than 7 days, fill with 0 or average if appropriate.
    last_7_sales = sales_df['Sales'].tail(7).tolist()
    last_7_customers = sales_df['Customers'].tail(7).tolist()
    
    for i, row in future_df.iterrows():
        # Prepare features for the current forecast date
        input_features = pd.DataFrame([row]).drop(columns=['Date', 'Weather']) # Drop original Weather
        
        # Add lag features based on the most recent actual data or previous forecast
        input_features['Sales_Lag1'] = current_sales_lag1
        input_features['Customers_Lag1'] = current_customers_lag1

        # For Sales_Lag7 and Customers_Lag7, get the sales/customer data from 7 days ago if available
        # or use a default if not enough historical data.
        if len(last_7_sales) >= 7 and i < len(last_7_sales):
            input_features['Sales_Lag7'] = last_7_sales[i]
            input_features['Customers_Lag7'] = last_7_customers[i]
        else: # Handle cases where there isn't 7 days of historical data for initial lags
            input_features['Sales_Lag7'] = 0
            input_features['Customers_Lag7'] = 0

        # Ensure all columns expected by the model are present and in the correct order
        # Pad with zeros for any missing weather columns dynamically determined during training
        feature_cols = st.session_state.get('feature_columns', [])
        for col in feature_cols:
            if col not in input_features.columns:
                input_features[col] = 0
        
        input_features = input_features[feature_cols] # Ensure order consistency

        # Predict
        predicted_sales = sales_model.predict(input_features)[0]
        predicted_customers = customers_model.predict(input_features)[0]

        forecast_results.append({
            'Date': row['Date'].strftime('%Y-%m-%d'),
            'Forecasted Sales': max(0, round(predicted_sales, 2)), # Ensure non-negative
            'Forecasted Customers': max(0, round(predicted_customers)), # Ensure non-negative
            'Weather': row['Weather'] # Display assumed weather
        })

        # Update lags for the next iteration (chained forecasting)
        current_sales_lag1 = predicted_sales
        current_customers_lag1 = predicted_customers
        
        # Update last_7_sales and last_7_customers for the next iteration
        # This is a simplification; for a robust implementation, you might need to
        # manage a rolling window of actual and forecasted data for lags.
        # For this example, we simply shift the initial 7 days list.
        if len(last_7_sales) > 0:
            last_7_sales.pop(0)
            last_7_sales.append(predicted_sales)
        if len(last_7_customers) > 0:
            last_7_customers.pop(0)
            last_7_customers.append(predicted_customers)

    return pd.DataFrame(forecast_results)

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Sales & Customer Forecast App")

st.title("🎯 AI Sales & Customer Forecast Analyst")
st.markdown("Your 200 IQ analyst for daily sales and customer volume forecasting!")

# Initialize session state for data
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = load_sales_data()
if 'events_data' not in st.session_state:
    st.session_state.events_data = load_events_data()
if 'sales_model' not in st.session_state:
    st.session_state.sales_model = None
if 'customers_model' not in st.session_state:
    st.session_state.customers_model = None

# Sidebar for Event Logger
st.sidebar.header("🗓️ Event Logger (Past Years)")
with st.sidebar.form("event_input_form"):
    st.subheader("Add Historical Event")
    event_date = st.date_input("Event Date", datetime.now() - timedelta(days=365))
    event_name = st.text_input("Event Name (e.g., Charter Day, Fiesta)")
    event_impact = st.selectbox("Impact", ['Low', 'Medium', 'High'])
    add_event_button = st.form_submit_button("Add Event")

    if add_event_button:
        new_event_df = pd.DataFrame([{
            'Event_Date': pd.to_datetime(event_date),
            'Event_Name': event_name,
            'Impact': event_impact
        }])
        st.session_state.events_data = pd.concat([st.session_state.events_data, new_event_df], ignore_index=True)
        st.session_state.events_data = st.session_state.events_data.drop_duplicates(subset=['Event_Date'], keep='last')
        save_events_data(st.session_state.events_data)
        st.sidebar.success(f"Event '{event_name}' added!")
        # Retrain models automatically after adding events
        with st.spinner("Retraining AI models with new event data..."):
            X, y_sales, y_customers = preprocess_data(st.session_state.sales_data, st.session_state.events_data)
            st.session_state.sales_model, st.session_state.customers_model = train_models(X, y_sales, y_customers)


st.sidebar.subheader("Logged Events")
if not st.session_state.events_data.empty:
    st.sidebar.dataframe(st.session_state.events_data.sort_values('Event_Date', ascending=False).style.format({'Event_Date': lambda x: x.strftime('%Y-%m-%d')}))

    # Delete Event functionality
    event_dates_to_delete = st.sidebar.multiselect(
        "Select events to delete:",
        st.session_state.events_data['Event_Date'].dt.strftime('%Y-%m-%d').tolist()
    )
    if st.sidebar.button("Delete Selected Events"):
        dates_to_delete_dt = [datetime.strptime(d, '%Y-%m-%d') for d in event_dates_to_delete]
        st.session_state.events_data = st.session_state.events_data[
            ~st.session_state.events_data['Event_Date'].isin(dates_to_delete_dt)
        ].reset_index(drop=True)
        save_events_data(st.session_state.events_data)
        st.sidebar.success("Selected events deleted!")
        st.experimental_rerun() # Rerun to update the dataframe and multiselect
else:
    st.sidebar.info("No events logged yet.")


# Main tabs for navigation
tab1, tab2, tab3 = st.tabs(["📊 Daily Sales Input", "📈 10-Day Forecast", "📊 Forecast Accuracy Tracking"])

with tab1:
    st.header("Smart Data Input System")
    st.markdown("Enter daily sales and customer data. The AI will learn from these inputs.")

    with st.form("daily_input_form"):
        st.subheader("Add New Daily Record")
        col1, col2, col3 = st.columns(3)
        with col1:
            input_date = st.date_input("Date", datetime.now())
        with col2:
            sales = st.number_input("Sales", min_value=0.0, format="%.2f")
        with col3:
            customers = st.number_input("Number of Customers", min_value=0, step=1)

        col4, col5 = st.columns(2)
        with col4:
            add_on_sales = st.number_input("Add-on Sales (e.g., birthdays, bulk)", min_value=0.0, format="%.2f")
        with col5:
            weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Rainy', 'Snowy']) # Added Snowy for more variety

        add_record_button = st.form_submit_button("Add Record")

        if add_record_button:
            # Check for duplicate date
            if pd.to_datetime(input_date) in st.session_state.sales_data['Date'].values:
                st.warning(f"Data for {input_date} already exists. Please edit the existing record or choose a different date.")
            else:
                new_record = pd.DataFrame([{
                    'Date': pd.to_datetime(input_date),
                    'Sales': sales,
                    'Customers': customers,
                    'Add_on_Sales': add_on_sales,
                    'Weather': weather
                }])
                st.session_state.sales_data = pd.concat([st.session_state.sales_data, new_record], ignore_index=True)
                st.session_state.sales_data = st.session_state.sales_data.sort_values('Date').reset_index(drop=True)
                save_sales_data(st.session_state.sales_data)
                st.success("Record added successfully! AI will retrain automatically.")
                # Retrain models automatically
                with st.spinner("Retraining AI models with new data..."):
                    X, y_sales, y_customers = preprocess_data(st.session_state.sales_data, st.session_state.events_data)
                    st.session_state.sales_model, st.session_state.customers_model = train_models(X, y_sales, y_customers)


    st.subheader("Last 7 Days of Inputs")
    if not st.session_state.sales_data.empty:
        last_7_days = st.session_state.sales_data.tail(7).sort_values('Date', ascending=False).copy()
        last_7_days['Date'] = last_7_days['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(last_7_days)

        st.subheader("Edit/Delete Records")
        if not st.session_state.sales_data.empty:
            # Create a unique identifier for each row for selection
            editable_df = st.session_state.sales_data.copy()
            editable_df['Display_Date'] = editable_df['Date'].dt.strftime('%Y-%m-%d')
            
            selected_date_for_edit_delete = st.selectbox(
                "Select a record by Date for editing or deleting:",
                editable_df['Display_Date'].tolist(),
                key='edit_delete_selector'
            )

            if selected_date_for_edit_delete:
                selected_row = st.session_state.sales_data[
                    st.session_state.sales_data['Date'] == pd.to_datetime(selected_date_for_edit_delete)
                ].iloc[0]

                st.markdown(f"**Selected Record for {selected_date_for_edit_delete}:**")
                
                with st.form("edit_delete_form"):
                    edit_sales = st.number_input("Edit Sales", value=float(selected_row['Sales']), format="%.2f")
                    edit_customers = st.number_input("Edit Customers", value=int(selected_row['Customers']), step=1)
                    edit_add_on_sales = st.number_input("Edit Add-on Sales", value=float(selected_row['Add_on_Sales']), format="%.2f")
                    edit_weather = st.selectbox("Edit Weather", ['Sunny', 'Cloudy', 'Rainy', 'Snowy'], index=['Sunny', 'Cloudy', 'Rainy', 'Snowy'].index(selected_row['Weather']))

                    col_edit_del_btns1, col_edit_del_btns2 = st.columns(2)
                    with col_edit_del_btns1:
                        update_button = st.form_submit_button("Update Record")
                    with col_edit_del_btns2:
                        delete_button = st.form_submit_button("Delete Record")

                    if update_button:
                        # Update the specific row
                        st.session_state.sales_data.loc[
                            st.session_state.sales_data['Date'] == pd.to_datetime(selected_date_for_edit_delete),
                            ['Sales', 'Customers', 'Add_on_Sales', 'Weather']
                        ] = [edit_sales, edit_customers, edit_add_on_sales, edit_weather]
                        save_sales_data(st.session_state.sales_data)
                        st.success("Record updated successfully! AI will retrain.")
                        with st.spinner("Retraining AI models with updated data..."):
                            X, y_sales, y_customers = preprocess_data(st.session_state.sales_data, st.session_state.events_data)
                            st.session_state.sales_model, st.session_state.customers_model = train_models(X, y_sales, y_customers)
                        st.experimental_rerun() # Rerun to update display
                    elif delete_button:
                        # Delete the specific row
                        st.session_state.sales_data = st.session_state.sales_data[
                            st.session_state.sales_data['Date'] != pd.to_datetime(selected_date_for_edit_delete)
                        ].reset_index(drop=True)
                        save_sales_data(st.session_state.sales_data)
                        st.success("Record deleted successfully! AI will retrain.")
                        with st.spinner("Retraining AI models with deleted data..."):
                            X, y_sales, y_customers = preprocess_data(st.session_state.sales_data, st.session_state.events_data)
                            st.session_state.sales_model, st.session_state.customers_model = train_models(X, y_sales, y_customers)
                        st.experimental_rerun() # Rerun to update display
        else:
            st.info("No sales data to edit or delete yet.")

    else:
        st.info("No sales data entered yet.")


with tab2:
    st.header("10-Day Sales & Customer Forecast")
    st.markdown("View the AI's predictions for the next 10 days.")

    # Load or train models before forecasting
    if st.session_state.sales_model is None or st.session_state.customers_model is None:
        with st.spinner("Preparing AI models..."):
            X, y_sales, y_customers = preprocess_data(st.session_state.sales_data, st.session_state.events_data)
            st.session_state.sales_model, st.session_state.customers_model = train_models(X, y_sales, y_customers)

    if st.button("Generate 10-Day Forecast"):
        if st.session_state.sales_data.empty or st.session_state.sales_data.shape[0] < 10:
            st.warning("Please enter at least 10 days of sales data to generate a meaningful forecast.")
        elif st.session_state.sales_model and st.session_state.customers_model:
            with st.spinner("Generating forecast... This might take a moment as the AI thinks ahead!"):
                forecast_df = generate_forecast(
                    st.session_state.sales_data,
                    st.session_state.events_data,
                    st.session_state.sales_model,
                    st.session_state.customers_model
                )
                st.session_state.forecast_df = forecast_df
                st.success("Forecast generated!")
        else:
            st.error("AI models are not ready. Please ensure you have sufficient data and try again.")
    
    if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
        st.subheader("Forecasted Data")
        st.dataframe(st.session_state.forecast_df)

        # Download as CSV
        csv = st.session_state.forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name="sales_customer_forecast.csv",
            mime="text/csv",
        )

        st.subheader("Forecast Visualization")
        # Combine historical and forecasted for plotting
        historical_df_plot = st.session_state.sales_data.copy()
        historical_df_plot['Type'] = 'Actual'
        historical_df_plot = historical_df_plot.rename(columns={'Sales': 'Value', 'Customers': 'Value_Customers'})
        historical_df_plot['Date'] = historical_df_plot['Date'].dt.strftime('%Y-%m-%d')

        forecast_df_plot = st.session_state.forecast_df.copy()
        forecast_df_plot['Type'] = 'Forecast'
        forecast_df_plot = forecast_df_plot.rename(columns={'Forecasted Sales': 'Value', 'Forecasted Customers': 'Value_Customers'})
        
        combined_df_sales = pd.concat([
            historical_df_plot[['Date', 'Value', 'Type']],
            forecast_df_plot[['Date', 'Value', 'Type']]
        ])
        
        combined_df_customers = pd.concat([
            historical_df_plot[['Date', 'Value_Customers', 'Type']],
            forecast_df_plot[['Date', 'Value_Customers', 'Type']]
        ])
        
        combined_df_sales['Date'] = pd.to_datetime(combined_df_sales['Date'])
        combined_df_customers['Date'] = pd.to_datetime(combined_df_customers['Date'])

        fig_sales, ax_sales = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=combined_df_sales, x='Date', y='Value', hue='Type', marker='o', ax=ax_sales)
        ax_sales.set_title('Sales: Actual vs. Forecast')
        ax_sales.set_xlabel('Date')
        ax_sales.set_ylabel('Sales')
        ax_sales.grid(True)
        st.pyplot(fig_sales)

        fig_customers, ax_customers = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=combined_df_customers, x='Date', y='Value_Customers', hue='Type', marker='o', ax=ax_customers)
        ax_customers.set_title('Customers: Actual vs. Forecast')
        ax_customers.set_xlabel('Date')
        ax_customers.set_ylabel('Customers')
        ax_customers.grid(True)
        st.pyplot(fig_customers)
    else:
        st.info("Click 'Generate 10-Day Forecast' to see predictions.")


with tab3:
    st.header("Forecast Accuracy Tracking")
    st.markdown("Compare past forecasts with actual data to track AI performance.")

    if st.button("Calculate Accuracy"):
        if st.session_state.sales_data.empty:
            st.warning("No sales data available to calculate accuracy.")
        elif st.session_state.sales_data.shape[0] < 2:
             st.warning("Please enter at least 2 days of sales data to calculate accuracy.")
        elif st.session_state.sales_model and st.session_state.customers_model:
            with st.spinner("Calculating accuracy..."):
                # Use historical data to predict and compare
                X_hist, y_sales_hist, y_customers_hist = preprocess_data(st.session_state.sales_data, st.session_state.events_data)

                if not X_hist.empty and X_hist.shape[0] > 1: # Need at least 2 data points for meaningful lag features
                    # Predict on the historical data (excluding the very first row due to lags)
                    # For accuracy, we want to predict what *would have been* predicted given past data.
                    # This means training on up to day T-1 to predict T.
                    # For simplicity, here we'll just predict on the whole preprocessed dataset
                    # and compare. A more rigorous approach would involve backtesting.

                    # Ensure the models are trained before attempting to predict
                    if st.session_state.sales_model is None or st.session_state.customers_model is None:
                         st.error("Models are not trained. Cannot calculate accuracy.")
                    else:
                        predicted_sales_hist = st.session_state.sales_model.predict(X_hist)
                        predicted_customers_hist = st.session_state.customers_model.predict(X_hist)

                        # Align predictions with actuals based on the preprocessed data's index
                        actual_sales_for_accuracy = y_sales_hist
                        actual_customers_for_accuracy = y_customers_hist

                        # Calculate metrics
                        mae_sales = mean_absolute_error(actual_sales_for_accuracy, predicted_sales_hist)
                        r2_sales = r2_score(actual_sales_for_accuracy, predicted_sales_hist)

                        mae_customers = mean_absolute_error(actual_customers_for_accuracy, predicted_customers_hist)
                        r2_customers = r2_score(actual_customers_for_accuracy, predicted_customers_hist)

                        st.subheader("Overall Model Accuracy (on historical data)")
                        st.write(f"**Sales MAE (Mean Absolute Error):** {mae_sales:.2f}")
                        st.write(f"**Sales R² Score:** {r2_sales:.2f}")
                        st.write(f"**Customers MAE (Mean Absolute Error):** {mae_customers:.2f}")
                        st.write(f"**Customers R² Score:** {r2_customers:.2f}")
                        st.info("An R² score closer to 1 indicates a better fit. MAE shows average error in units.")

                        # --- Visualization of Actual vs. Predicted (Historical) ---
                        accuracy_plot_df = pd.DataFrame({
                            'Date': st.session_state.sales_data['Date'].iloc[X_hist.index], # Align dates
                            'Actual Sales': actual_sales_for_accuracy,
                            'Predicted Sales': predicted_sales_hist,
                            'Actual Customers': actual_customers_for_accuracy,
                            'Predicted Customers': predicted_customers_hist
                        })
                        accuracy_plot_df['Date'] = pd.to_datetime(accuracy_plot_df['Date'])

                        fig_acc_sales, ax_acc_sales = plt.subplots(figsize=(12, 6))
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Actual Sales', label='Actual Sales', marker='o', ax=ax_acc_sales)
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Predicted Sales', label='Predicted Sales', marker='x', linestyle='--', ax=ax_acc_sales)
                        ax_acc_sales.set_title('Historical Sales: Actual vs. Predicted')
                        ax_acc_sales.set_xlabel('Date')
                        ax_acc_sales.set_ylabel('Sales')
                        ax_acc_sales.legend()
                        ax_acc_sales.grid(True)
                        st.pyplot(fig_acc_sales)

                        fig_acc_customers, ax_acc_customers = plt.subplots(figsize=(12, 6))
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Actual Customers', label='Actual Customers', marker='o', ax=ax_acc_customers)
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Predicted Customers', label='Predicted Customers', marker='x', linestyle='--', ax=ax_acc_customers)
                        ax_acc_customers.set_title('Historical Customers: Actual vs. Predicted')
                        ax_acc_customers.set_xlabel('Date')
                        ax_acc_customers.set_ylabel('Customers')
                        ax_acc_customers.legend()
                        ax_acc_customers.grid(True)
                        st.pyplot(fig_acc_customers)
                else:
                    st.warning("Not enough data points after preprocessing for accuracy calculation. Please add more sales records.")
        else:
            st.error("AI models are not ready. Please ensure you have sufficient data and train the models first.")
    else:
        st.info("Click 'Calculate Accuracy' to see how well the AI performs on past data.")


# --- Initial Sample Data Creation (Run once if files don't exist) ---
def create_sample_data():
    """Creates sample sales and event data if files are empty or don't exist."""
    if os.path.exists(SALES_DATA_PATH) and os.path.getsize(SALES_DATA_PATH) > 0:
        sales_df_check = pd.read_csv(SALES_DATA_PATH)
        if not sales_df_check.empty:
            return # Data already exists

    st.info("Creating sample sales and event data for a quick start...")
    # Sample Sales Data
    dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=30, freq='D'))
    np.random.seed(42)
    sales = np.random.randint(500, 1500, size=len(dates)) + np.random.randn(len(dates)) * 50
    customers = np.random.randint(50, 200, size=len(dates)) + np.random.randn(len(dates)) * 10
    add_on_sales = np.random.randint(0, 100, size=len(dates))
    weather_choices = ['Sunny', 'Cloudy', 'Rainy']
    weather = np.random.choice(weather_choices, size=len(dates), p=[0.6, 0.3, 0.1])

    # Introduce some patterns
    # Higher sales on weekends
    for i, date in enumerate(dates):
        if date.dayofweek >= 5: # Saturday or Sunday
            sales[i] = sales[i] * 1.2
            customers[i] = customers[i] * 1.2
        if weather[i] == 'Rainy':
            sales[i] = sales[i] * 0.8
            customers[i] = customers[i] * 0.8

    sample_sales_df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.round(2),
        'Customers': customers.round().astype(int),
        'Add_on_Sales': add_on_sales.round(2),
        'Weather': weather
    })
    save_sales_data(sample_sales_df)

    # Sample Events Data
    sample_events_df = pd.DataFrame([
        {'Event_Date': pd.to_datetime('2024-06-20'), 'Event_Name': 'Annual Fair', 'Impact': 'High'},
        {'Event_Date': pd.to_datetime('2023-12-25'), 'Event_Name': 'Christmas Day', 'Impact': 'High'},
        {'Event_Date': pd.to_datetime('2024-03-15'), 'Event_Name': 'Spring Festival', 'Impact': 'Medium'},
        {'Event_Date': pd.to_datetime('2025-06-27'), 'Event_Name': 'Charter Day 2025 (Future)', 'Impact': 'High'} # Example future event
    ])
    save_events_data(sample_events_df)
    st.success("Sample data created! You can now start using the app.")
    st.experimental_rerun() # Rerun to load initial data into session state


# Run initial data creation on app startup
if 'app_initialized' not in st.session_state:
    create_sample_data()
    st.session_state['app_initialized'] = True

# --- Auto-load/train models on initial app load ---
if st.session_state.sales_model is None or st.session_state.customers_model is None:
    # Only attempt to train if there's enough data
    if st.session_state.sales_data.shape[0] > 0:
        with st.spinner("Loading/Training AI models on startup..."):
            X, y_sales, y_customers = preprocess_data(st.session_state.sales_data, st.session_state.events_data)
            if not X.empty:
                st.session_state.sales_model, st.session_state.customers_model = load_or_train_models()
            else:
                st.warning("Not enough data to train models on startup. Please add more daily sales records.")
    else:
        st.info("Add sales records to enable AI model training and forecasting.")
