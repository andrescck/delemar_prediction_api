import pandas as pd
import numpy as np
import streamlit as st
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to calculate AIC
def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic

# Load the data
data = pd.read_excel('data_ready_3.xlsx')

# Drop the first 5 columns
data_cluster = data.drop(data.columns[[0, 1, 2, 3, 4, 5]], axis=1)

# Define features and target variable
features = ['Number of previous performances', 'Show length (minutes']
categorical_features = ['Category', 'Show_status', 'Time of the day', 'Capacity level']

# Define categorical indices for KPrototypes before one-hot encoding
categorical_indices = [data_cluster.columns.get_loc(col) for col in categorical_features]
target = 'Total seats sold'

# Fit KPrototypes
kproto = KPrototypes(n_clusters=4, init='Cao', verbose=0, random_state=42)
kproto.fit_predict(data_cluster, categorical=categorical_indices)

# Assign the labels to the DataFrame
data['Cluster'] = kproto.labels_

# One-hot encode categorical features
data = pd.get_dummies(data, columns=categorical_features)
one_hot_features = [col for col in data.columns if col.startswith(tuple(categorical_features))]
features.extend(one_hot_features)

# Train models and select the best one for each cluster based on AIC
model_candidates = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree Regressor", DecisionTreeRegressor(random_state=42)),
    ("Random Forest Regressor", RandomForestRegressor(random_state=42)),
    ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42)),
    ("Support Vector Regressor", SVR())
]

best_models = {}

for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]

    X = cluster_data[features]
    y = cluster_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = None
    best_aic = np.inf

    for model_name, model in model_candidates:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        aic = calculate_aic(len(y_test), mse, X_train.shape[1] + 1)
        print(f"{model_name} - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, AIC: {aic:.2f}, R-squared: {r2:.2f}")

        if aic < best_aic:
            best_model = model
            best_aic = aic

    best_models[cluster] = best_model
    print(f"Best model for Cluster {cluster}: {best_model.__class__.__name__} with AIC: {best_aic:.2f}\n")

# Function to predict ticket sales for a new show
def predict_ticket_sales(new_data, kproto, best_models, features, categorical_indices):
    # new_data_df = pd.DataFrame([new_data])

    # # Ensure new_data_df has all categorical features and convert them to 'category' dtype
    # for col in categorical_features:
    #     new_data_df[col] = new_data_df[col].astype('category')
    #     new_data_df[col].cat.set_categories(data_cluster[col].cat.categories, inplace=True)

    new_data_df = pd.DataFrame([new_data])

    # Ensure new_data_df has all categorical features and convert them to 'category' dtype
    for col in categorical_features:
        if col in new_data_df.columns:
            new_data_df[col] = new_data_df[col].astype('category')
            if data_cluster[col].dtype.name == 'category':
                new_data_df[col].cat.set_categories(data_cluster[col].cat.categories, inplace=True)

    # Align the new show DataFrame with the training DataFrame
    new_data_df = new_data_df.reindex(columns=data_cluster.columns, fill_value=0)

    cluster = kproto.predict(new_data_df.to_numpy(), categorical=categorical_indices)[0]

    model = best_models[cluster]
    new_data_df = pd.get_dummies(new_data_df, columns=categorical_features)
    new_data_df = new_data_df.reindex(columns=features, fill_value=0)

    predicted_tickets = model.predict(new_data_df)

    return int(np.round(predicted_tickets[0])), cluster

# Streamlit interface
st.title("Theater Show Ticket Sales Predictor")

# Input form
with st.form(key='show_form'):
    category = st.selectbox('Category', ['Musical', 'Cabaret', 'Concert', 'Dans', 'Jeugd', 'Muziektheater', 'Specials', 'Toneel'])
    show_length = st.number_input('Show length (minutes)', min_value=0, value=90)
    show_status = st.selectbox('Show status', ['New', 'Returning'])
    time_of_day = st.selectbox('Time of the day', ['Afternoon', 'Evening'])
    num_prev_performances = st.number_input('Number of previous performances', min_value=0, value=0)
    capacity_level = st.selectbox('Capacity level', ['small', 'medium', 'large'])

    submit_button = st.form_submit_button(label='Predict Ticket Sales')

# Prediction
if submit_button:
    new_show = {
        'Category': category,
        'Show length (minutes': show_length,
        'Show_status': show_status,
        'Time of the day': time_of_day,
        'Number of previous performances': num_prev_performances,
        'Capacity level': capacity_level
    }

    predicted_tickets, cluster = predict_ticket_sales(new_show, kproto, best_models, features, categorical_indices)
    st.write(f"Predicted tickets sold: {predicted_tickets}")