import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from dataidea.logger import event_log

# Define the ScalableModel class
class ScalableModel:
    def __init__(self, model, feature_scaler=None, target_scaler=None):
        self.model = model
        self.feature_scaler = feature_scaler if feature_scaler else MinMaxScaler()
        self.target_scaler = target_scaler if target_scaler else MinMaxScaler()
    
    def predict(self, X_test):
        X_test_scaled = self.feature_scaler.transform(X_test)
        predictions_scaled = self.model.predict(X_test_scaled).reshape(-1, 1)
        return self.target_scaler.inverse_transform(predictions_scaled).flatten()

# Load the trained model and scalers
xgboost_model = joblib.load("XGBoost_Model_v1.pkl")  # Ensure the model is saved


# Feature names (MUST match your training data exactly)
feature_names = [
    'garlic_kgs_organic_pesticides', #
    'ginger_kgs_organic_pesticides', #
    'plastic_tanks_120_ltrs_liquid_manure', #
    'sacks_liquid_manure', #
    'hoes_tools', #
    'spades_tools', #
    'pick_axes_tools', #
    'forked_hoes_tools', #
    'pangas_tools', #
    'wheelbarrows_tools', #
    'trowels_tools', #
    'watering_cans_tools', #
    'spray_pumps_tools', #
    'Beans_seeds', #
    'Maize_seeds', #
    'Soyabean_seeds', #
    'Gnuts_seeds', #
    'Irish Potatoes_seeds', #
    # 'Cassava (bags)',
    # 'Mugavu tree seedlings',
    # 'Onions (Kg)_seeds',
    # 'Millet.1_seeds',
    'GPS-Altitude',
    'Land_size_agriculture',
    'Time_to_collect_Water_for_Household_use_Minutes',
    'Beans_total_yield', #
    'Cassava_total_yield',
    'Maize_total_yield',
    'Sweet_potatoes_total_yield',
    'Food_banana_total_yield',
    'Coffee_total_yield'
]


# Define prediction function
def predict_agriculture_value(*features):
    input_data = np.array(features).reshape(1, -1)
    # Check if all input values are zero
    if np.all(input_data == 0):
        return 0.0  # Return zero directly
    
    df_input = pd.DataFrame(input_data, columns=feature_names)
    
    # Use the model's built-in predict function
    prediction = xgboost_model.predict(df_input)

    event_log({
        'api_key': '1968c15b-ed45-4a2d-a7dc-90ce623324b8',
        'project_name': 'IDS',
        'user_id': 'Emmanuel Nsubuga', # optional
        'message': 'Prediction Log',
        'level': 'info',
        'metadata': {
            'input_data': input_data.tolist(),
            'prediction': prediction.tolist(),
            'source': 'gradio'
        } # optional
    })
    
    return float(prediction[0])

# Create Gradio interface
input_components = [gr.Number(label=feature) for feature in feature_names]

iface = gr.Interface(
    fn=predict_agriculture_value,
    inputs=input_components,
    outputs=gr.Number(label="Predicted Agriculture Value (USD)"),
    title="Agriculture Value Prediction",
    description="Enter input values to predict Agriculture Value (USD) using the trained XGBoost model."
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()


# # XGBoost Model

# import gradio as gr
# import joblib
# import numpy as np
# import pandas as pd
# from xgboost import XGBRegressor
# from sklearn.preprocessing import MinMaxScaler

# # Define ScalableModel class
# class ScalableModel:
#     def __init__(self, model, feature_scaler=None, target_scaler=None):
#         self.model = model
#         self.feature_scaler = feature_scaler if feature_scaler else MinMaxScaler()
#         self.target_scaler = target_scaler if target_scaler else MinMaxScaler()

#     def fit(self, X_train, y_train):
#         self.feature_scaler.fit(X_train)
#         self.target_scaler.fit(y_train.values.reshape(-1, 1))
#         self.model.fit(self.feature_scaler.transform(X_train), self.target_scaler.transform(y_train.values.reshape(-1, 1)).flatten())

#     def predict(self, X_test):
#         X_test_scaled = self.feature_scaler.transform(X_test)
#         predictions_scaled = self.model.predict(X_test_scaled)
#         return self.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

# # Load the trained XGBoost model
# xgboost_model = joblib.load("XGBoost_Model_v1.pkl")

# # Load the scalers
# scalers = joblib.load("Scalers_XGBoost.pkl")
# feature_scaler = scalers["feature_scaler"]
# target_scaler = scalers["target_scaler"]

# # Define feature names (must match training data)
# feature_names = [
#     'garlic_kgs_organic_pesticides', 'ginger_kgs_organic_pesticides',
#     'plastic_tanks_120_ltrs_liquid_manure', 'sacks_liquid_manure',
#     'hoes_tools', 'spades_tools', 'pick_axes_tools', 'forked_hoes_tools',
#     'pangas_tools', 'wheelbarrows_tools', 'trowels_tools', 'watering_cans_tools',
#     'spray_pumps_tools', 'Beans_seeds', 'Maize_seeds', 'Soyabean_seeds',
#     'Gnuts_seeds', 'Irish Potatoes_seeds', 'GPS-Altitude', 'Land_size_agriculture',
#     'Time_to_collect_Water_for_Household_use_Minutes', 'Beans_total_yield',
#     'Cassava_total_yield', 'Maize_total_yield', 'Sweet_potatoes_total_yield',
#     'Food_banana_total_yield', 'Coffee_total_yield'
# ]

# # Define prediction function
# def predict_agriculture_value(*features):
#     input_data = np.array(features).reshape(1, -1)

#     # Scale the input features
#     input_data_scaled = feature_scaler.transform(input_data)

#     # Get scaled predictions
#     prediction_scaled = xgboost_model.predict(input_data_scaled).reshape(-1, 1)

#     # Inverse transform to get predictions in original scale
#     prediction = target_scaler.inverse_transform(prediction_scaled).flatten()

#     return float(prediction[0])

# # Create Gradio interface
# input_components = [gr.Number(label=feature) for feature in feature_names]

# iface = gr.Interface(
#     fn=predict_agriculture_value,
#     inputs=input_components,
#     outputs=gr.Number(label="Predicted Agriculture Value (USD)"),
#     title="Agriculture Value Prediction",
#     description="Enter input values to predict Agriculture Value (USD) using the trained XGBoost model."
# )

# # Run the Gradio app
# if __name__ == "__main__":
#     iface.launch()

# print("Expected feature order:", feature_names)