# YOLO Dog ImageWoof Classification Web App

![YOLO Dog ImageWoof](https://img.icons8.com/color/96/000000/dog.png)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **YOLO Dog ImageWoof Classification Web App** is a user-friendly web application that leverages the power of the YOLO (You Only Look Once) object detection model to classify dog images. Users can upload an image of a dog, and the model will accurately classify it, providing both Top-1 and Top-5 predictions along with precomputed validation metrics.

Built with Python and utilizing the [Gradio](https://gradio.app/) library for the web interface, this application offers a seamless experience for users to interact with machine learning models without requiring any technical expertise.

## Features

- **Image Upload**: Users can easily upload images in common formats (JPEG, PNG, etc.).
- **Real-time Classification**: The YOLO model processes the uploaded image and provides classification results instantly.
- **Annotated Images**: The app displays annotated images highlighting the detected objects.
- **Top-1 and Top-5 Predictions**: Users receive detailed classification results, including the most probable class and the top five predictions with confidence scores.
- **Validation Metrics**: Precomputed metrics such as Overall Top-1 and Top-5 Accuracy are displayed to provide insights into the model's performance.
- **User-Friendly Interface**: Intuitive design ensures a smooth user experience.

## Demo

![Web App Screenshot](https://via.placeholder.com/800x400.png?text=YOLO+Dog+ImageWoof+Web+App)

*Note: Replace the placeholder image link with an actual screenshot of your web app for a better demonstration.*

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure Python is installed on your system. You can download it from the [official website](https://www.python.org/downloads/).
- **Git**: Optional, for cloning the repository. Download from [here](https://git-scm.com/downloads).

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/yolo-dog-imagewoof.git
   cd yolo-dog-imagewoof
   ```

   *Alternatively, you can download the repository as a ZIP file and extract it.*

2. **Create a Virtual Environment (Recommended)**

   It's best practice to use a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   - **Windows:**

     ```bash
     venv\Scripts\activate
     ```

   - **macOS and Linux:**

     ```bash
     source venv/bin/activate
     ```

4. **Install Required Packages**

   Ensure you have `pip` updated:

   ```bash
   pip install --upgrade pip
   ```

   Install dependencies using the provided `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

5. **Download or Place the YOLO Model**

   Ensure the YOLO model file (`best.pt`) is placed in the project directory or update the `model_path` variable in the script to point to its location.

   *You can train your own YOLO model or download a pre-trained one compatible with your classification task.*

## Usage

1. **Run the Application**

   Execute the Python script to launch the Gradio web interface.

   ```bash
   python your_script_name.py
   ```

   Replace `your_script_name.py` with the actual name of your Python script file.

2. **Access the Web Interface**

   After running the script, Gradio will provide a local URL (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser to access the application.

3. **Classify an Image**

   - **Upload Image**: Click on the "Upload Image" button and select a dog image from your device.
   - **Run Inference**: Click on the "Run Inference" button to process the image.
   - **View Results**: The annotated image, Top-1 and Top-5 predictions, along with validation metrics, will be displayed.

## Configuration

### Model Path

Ensure that the `model_path` variable in the script correctly points to your YOLO model file (`best.pt`).

```python
model_path = "best.pt"  # Update this path if your model is located elsewhere
```

### Validation Metrics

The script includes hardcoded validation metrics. Update these values as per your model's performance.

```python
overall_top1_accuracy = 0.9142  # Replace with your Top-1 accuracy
overall_top5_accuracy = 0.9926  # Replace with your Top-5 accuracy
```

## Project Structure

```
yolo-dog-imagewoof/
│
├── best.pt                  # YOLO model file
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── app.py      # Main Python script
```

*Ensure that `best.pt` is placed in the project directory or update the script accordingly.*

## Dependencies

All required Python packages are listed in the `requirements.txt` file. Below is a summary:

```plaintext
numpy
opencv-python
Pillow
gradio
pandas
ultralytics
```

### Descriptions:

- **numpy**: Fundamental package for scientific computing with Python.
- **opencv-python**: Open Source Computer Vision Library.
- **Pillow**: Python Imaging Library (PIL) fork for image processing.
- **gradio**: Library for creating user interfaces for machine learning models.
- **pandas**: Data manipulation and analysis library.
- **ultralytics**: YOLO (You Only Look Once) object detection and classification models.

*Refer to the `requirements.txt` file for exact versions if specified.*

## Troubleshooting

- **Model Not Found Error**

  ```plaintext
  FileNotFoundError: Model file not found at best.pt.
  ```

  - **Solution**: Ensure that `best.pt` is placed in the correct directory as specified by the `model_path` variable. Update the path if necessary.

- **Import Errors**

  If you encounter errors related to missing packages, ensure all dependencies are installed correctly:

  ```bash
  pip install -r requirements.txt
  ```

- **Gradio Not Launching**

  - **Possible Issues**: Port conflicts or firewall restrictions.
  - **Solution**: Specify a different port when launching Gradio or check your firewall settings.

- **Incorrect Annotated Image Colors**

  If the annotated image does not display in the correct RGB format:

  - **Solution**: Ensure that color space conversions are handled correctly in the script. The provided revised script addresses this issue by maintaining the correct color format.


## License

This project is licensed under the [MIT License](LICENSE).

---

## Hugging Face Space

https://huggingface.co/spaces/csepartha/yolo11n_imagewoof_dog_classification

*Developed by Partha Pratim Ray*

For any queries or support, please contact [parthapratimray1986@gmail.com](mailto:parthapratimray1986@gmail.com).
