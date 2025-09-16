import gradio as gr
import torch
from PIL import Image
import json
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os

# Custom CSS for enhanced UI/UX
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
}

.header-text {
    text-align: center;
    color: #d62300;
    font-weight: bold;
    margin-bottom: 20px;
}

.description-text {
    text-align: center;
    color: #666;
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 30px;
    padding: 0 20px;
}

.upload-area {
    border: 2px dashed #d62300;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    background-color: #fff8f6;
}

.results-container {
    margin-top: 20px;
    padding: 20px;
    border-radius: 10px;
    background-color: #f8f9fa;
}

.confidence-slider {
    margin: 20px 0;
}

.footer-text {
    text-align: center;
    color: #888;
    font-size: 14px;
    margin-top: 30px;
    font-style: italic;
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 10px;
    }
    
    .description-text {
        font-size: 14px;
        padding: 0 10px;
    }
}
"""

# Download and load model
def load_model():
    try:
        print("🔄 Downloading TimHortons detection model...")
        model_path = hf_hub_download(
            repo_id="Eviekiwi/timhortons-cup-yolo-detection",
            filename="best_model_v6.pt",
            repo_type="space"
        )
        print("✅ Model downloaded successfully!")
        
        print("🔄 Loading YOLO model...")
        model = YOLO(model_path)
        print("✅ Model loaded and ready for detection!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("🔄 Falling back to pretrained YOLOv8 model...")
        return YOLO('yolov8n.pt')

# Initialize model
model = load_model()

def predict_image(image, confidence_threshold=0.5):
    """
    Perform TimHortons object detection on uploaded image
    
    Args:
        image: PIL Image object
        confidence_threshold: Float between 0.1 and 0.9
    
    Returns:
        tuple: (annotated_image, detection_results_json)
    """
    if image is None:
        return None, json.dumps({"error": "No image uploaded"}, indent=2)
    
    try:
        # Set confidence threshold for detection
        model.conf = confidence_threshold
        
        # Run inference on the image
        results = model(image)
        
        # Generate annotated image with bounding boxes
        annotated_img = results[0].plot()
        
        # Extract detection information
        detections = {
            "total_detections": 0,
            "tim_hortons_items": [],
            "confidence_threshold": confidence_threshold,
            "model_info": "Custom-trained YOLOv8 for TimHortons detection"
        }
        
        if results[0].boxes is not None:
            detections["total_detections"] = len(results[0].boxes)
            
            for i, box in enumerate(results[0].boxes):
                detection = {
                    "detection_id": i + 1,
                    "class_name": results[0].names[int(box.cls)],
                    "confidence_score": round(float(box.conf), 3),
                    "bounding_box": {
                        "x1": round(float(box.xyxy[0][0]), 1),
                        "y1": round(float(box.xyxy[0][1]), 1),
                        "x2": round(float(box.xyxy[0][2]), 1),
                        "y2": round(float(box.xyxy[0][3]), 1)
                    }
                }
                detections["tim_hortons_items"].append(detection)
        
        # Add summary message
        if detections["total_detections"] > 0:
            detections["summary"] = f"🎯 Found {detections['total_detections']} TimHortons item(s) in your image!"
        else:
            detections["summary"] = "🔍 No TimHortons items detected. Try adjusting the confidence threshold or upload a different image."
        
        return annotated_img, json.dumps(detections, indent=2)
    
    except Exception as e:
        error_response = {
            "error": f"Detection failed: {str(e)}",
            "suggestion": "Please try uploading a different image or check your internet connection."
        }
        return None, json.dumps(error_response, indent=2)

def create_examples():
    """Create example images for users to try"""
    # Note: In a real deployment, you'd want to include actual example images
    return [
        ["example_tim_cup.jpg", 0.5],
        ["example_tim_donut.jpg", 0.3],
        ["example_tim_coffee.jpg", 0.7]
    ]

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="TimHortons Detection System") as demo:
    # Header
    gr.HTML("""
        <div class="header-text">
            <h1>🎯 TimHortons YOLO Detection System</h1>
        </div>
    """)
    
    # Description
    gr.HTML("""
        <div class="description-text">
            <p><strong>Discover the TIMS in your world!</strong></p>
            <p>This AI-powered detection system uses a custom-trained YOLOv8 model to identify TimHortons items in your images. 
            Whether it's the iconic double-double coffee cup, delicious donuts, or other TIMS merchandise, 
            our model will help you spot those familiar red and white traces of Canadian coffee culture.</p>
            <p>Simply upload an image and watch as our AI finds every TimHortons item with precision!</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image upload section
            gr.HTML('<div class="upload-area">')
            image_input = gr.Image(
                type="pil", 
                label="📷 Upload Your Image",
                elem_id="image-upload"
            )
            gr.HTML('</div>')
            
            # Confidence threshold slider
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="🎚️ Detection Confidence Threshold",
                info="Lower values detect more objects (but may include false positives)",
                elem_classes=["confidence-slider"]
            )
            
            # Detection button
            detect_btn = gr.Button(
                "🔍 Detect TimHortons Items", 
                variant="primary",
                size="lg"
            )
            
            # Clear button
            clear_btn = gr.Button(
                "🗑️ Clear All", 
                variant="secondary"
            )
    
        with gr.Column(scale=1):
            # Results section
            gr.HTML('<div class="results-container">')
            
            # Annotated image output
            image_output = gr.Image(
                type="pil", 
                label="🎯 Detection Results",
                elem_id="results-image"
            )
            
            # JSON results output
            json_output = gr.JSON(
                label="📊 Detailed Detection Data",
                elem_id="results-json"
            )
            
            gr.HTML('</div>')
    
    # Examples section
    gr.HTML("""
        <div style="margin-top: 30px;">
            <h3 style="text-align: center; color: #d62300;">Try These Examples:</h3>
            <p style="text-align: center; color: #666;">
                Click on any example below to see the detection system in action!
            </p>
        </div>
    """)
    
    # Event handlers
    detect_btn.click(
        fn=predict_image,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, json_output]
    )
    
    # Auto-detect when image is uploaded
    image_input.change(
        fn=predict_image,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, json_output]
    )
    
    # Clear functionality
    def clear_all():
        return None, None, None
    
    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, image_output, json_output]
    )
    
    # Footer
    gr.HTML("""
        <div class="footer-text">
            <p>🍁 Proudly detecting Canadian coffee culture, one image at a time 🍁</p>
            <p>Built with ❤️ using YOLOv8 and Gradio | Model trained specifically for TimHortons recognition</p>
        </div>
    """)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        show_tips=True
    )