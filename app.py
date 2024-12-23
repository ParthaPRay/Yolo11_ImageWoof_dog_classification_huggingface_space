# Partha Pratim Ray

import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import gradio as gr
import pandas as pd

# Paths
model_path = "best.pt"  # Ensure the best.pt is in the local directory or provide full path

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}.")

# Load the YOLO model
model = YOLO(model_path)

##################################
# Hardcoded metrics for classification
overall_top1_accuracy = 0.9142  # Replace with your Top-1 accuracy
overall_top5_accuracy = 0.9926  # Replace with your Top-5 accuracy

# Metrics DataFrame
metrics_data = [
    ["Overall Top-1 Accuracy", f"{overall_top1_accuracy * 100:.2f}%"],
    ["Overall Top-5 Accuracy", f"{overall_top5_accuracy * 100:.2f}%"]
]
metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
##################################

def run_inference(img: np.ndarray, model):
    """
    Runs inference on the input image using the YOLO model.
    Returns the annotated image, Top-1 prediction, and Top-5 predictions.
    """
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run prediction
    results = model.predict(img_rgb)
    
    # Extract probabilities (if available)
    result_probs = results[0].probs
    
    # Get top-1 and top-5 predictions
    top1_class = result_probs.top1
    top5_classes = result_probs.top5
    top1_conf = result_probs.top1conf.item()
    top5_conf = result_probs.top5conf
    
    # Generate annotated image (RGB format)
    annotated_img = results[0].plot()  # Assuming this returns RGB
    
    # Format results
    top1_result = f"Class: {model.names[top1_class]}, Confidence: {top1_conf:.2f}"
    top5_results = [
        f"{model.names[c]}: {conf:.2f}" for c, conf in zip(top5_classes, top5_conf)
    ]
    
    return annotated_img, top1_result, top5_results

def process_image(image):
    """
    Processes the input image, runs inference, and prepares the outputs.
    """
    # Convert PIL Image to NumPy array
    img = np.array(image)
    
    # Convert from RGB to BGR for OpenCV (if needed by YOLO)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Run classification inference
    annotated_img, top1_result, top5_results = run_inference(img_bgr, model)
    
    # Convert annotated image back to PIL format without altering color channels
    annotated_img_pil = Image.fromarray(annotated_img)  # Assuming annotated_img is in RGB
    
    # Return the annotated image, Top-1, and Top-5 predictions, along with metrics
    return annotated_img_pil, f"Top-1: {top1_result}", "\n".join(top5_results), metrics_df

with gr.Blocks() as demo:
    gr.Markdown("# YOLO Dog ImageWoof Classification Web App")
    gr.Markdown("Upload an image, and the model will classify it and show precomputed validation metrics.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Run Inference")
        with gr.Column():
            annotated_image = gr.Image(type="pil", label="Annotated Image")  # Shows annotated image in RGB
            top1_output = gr.Textbox(label="Top-1 Prediction")
            top5_output = gr.Textbox(label="Top-5 Predictions")
            metrics_table = gr.DataFrame(value=metrics_df, label="Validation Metrics")

    submit_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[annotated_image, top1_output, top5_output, metrics_table]
    )

demo.launch()

