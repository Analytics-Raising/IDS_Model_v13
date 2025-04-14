import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and scaler
def load_model():
    try:
        model = joblib.load('rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return None, None

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

def preprocess_input(input_data):
    """Preprocess the input data similar to training data"""
    # Convert to DataFrame
    df = pd.DataFrame([input_data], columns=feature_names)
    
    # Handle missing values (if any)
    df = df.fillna(df.mean())
    
    # Remove outliers using IQR method
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)
    
    return df

def predict(*features):
    """Make predictions using the Random Forest model"""
    # Load model and scaler
    model, scaler = load_model()
    if model is None or scaler is None:
        return "Error: Model not found. Please train the model first."
    
    # Preprocess input data
    input_data = preprocess_input(features)
    
    # Scale the features
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Create feature importance plot
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('feature_importance.png')
    plt.close()
    
    return f"Predicted Agriculture Value (USD): {prediction:.2f}", 'feature_importance.png'

# Create Gradio interface
input_components = [gr.Number(label=feature) for feature in feature_names]

iface = gr.Interface(
    fn=predict,
    inputs=input_components,
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(label="Feature Importance Plot")
    ],
    title="Agriculture Value Prediction (Random Forest)",
    description="Enter input values to predict Agriculture Value (USD) using the trained Random Forest model.",
    theme="huggingface"
)

if __name__ == "__main__":
    iface.launch() 