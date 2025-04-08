import gradio as gr
import joblib
import pandas as pd

# Load pre-trained model
model = joblib.load("model.pkl")

# Define a function to make predictions
def predict_price(area, bedrooms, bathrooms):
    # Prepare the input data
    input_data = pd.DataFrame({"area": [area], "bedrooms": [bedrooms], "bathrooms": [bathrooms]})
    # Make prediction
    prediction = model.predict(input_data)
    return f"Predicted price: ${prediction[0]:,.2f}"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Area (sq ft)", value=1000),
        gr.Number(label="Bedrooms", value=3),
        gr.Number(label="Bathrooms", value=2)
    ],
    outputs="text",
    live=True
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
