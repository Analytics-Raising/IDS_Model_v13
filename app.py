
import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Define Custom KNN Class 
class CustomKNN(KNeighborsRegressor):
    def predict(self, X):
        X = np.array(X)
        zero_mask = np.all(X == 0, axis=1)
        predictions = super().predict(X)
        predictions[zero_mask] = 0  # Override zero-input predictions with 0
        return predictions
# Load the trained model
model = joblib.load("knn_model_8.pkl")  # Change filename if needed

# Define the feature names (same order as during training)
feature_names = [
 'beans_kgs_hh_seed',
 'ground_nuts_kgs_hh_seed',
 'maize_kgs_hh_seed',
 'soya_bean_kgs_hh_seed',
 'garlic_kgs_organic_pesticides',
 'ginger_kgs_organic_pesticides',
 'plastic_tanks_120_ltrs_liquid_manure',
 'sacks_liquid_manure',
 'hoes_tools',
 'spades_tools',
 'pick_axes_tools',
 'forked_hoes_tools',
 'pangas_tools',
 'wheelbarrows_tools',
 'trowels_tools',
 'watering_cans_tools',
 'spray_pumps_tools',
 'GPS-Altitude',
 'hhh_age',
 'Land_size_agriculture',
 'Time_to_collect_Water_for_Household_use_Minutes',
 'Beans_total_yield',
 'Cassava_total_yield',
 'Maize_total_yield',
 'Sweet_potatoes_total_yield',
 'Food_banana_total_yield',
 'Coffee_total_yield'
 ]

# Define the prediction function
def predict(*features):
    # Convert input to a DataFrame
    input_data = pd.DataFrame([features], columns=feature_names)

    # Apply log transformation to match training
    input_data_log = np.log1p(input_data)

    # Make prediction
    prediction_log = model.predict(input_data_log)

    # Convert prediction back using inverse log transformation
    prediction = np.expm1(prediction_log)

    return float(prediction[0])  # Convert to float for display

# Create Gradio interface
input_components = [gr.Number(label=feature) for feature in feature_names]

iface = gr.Interface(
    fn=predict,
    inputs=input_components,
    outputs=gr.Number(label="Predicted Agriculture Value (USD)"),
    title="Agriculture Value Prediction",
    description="Enter input values to predict Agriculture Value (USD) using the trained model."
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()
