from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import base64
import re
import io
from vgg16 import VGG16

app = Flask(__name__)

# Load the model
model = VGG16()
model.load_state_dict(torch.load('./models/vgg_16_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image transformation without normalization
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def test_single_image(model, image, device):
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move image and model to the specified device
    image = image.to(device)
    model = model.to(device)
    
    # Add batch dimension if necessary (model expects batches)
    if len(image.shape) == 3:  # Assuming the image is [C, H, W]
        image = image.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
    
    # Perform inference
    with torch.no_grad():
        logits = model(image)  # Forward pass
        confidences = torch.softmax(logits, dim=1).squeeze().tolist()  # Confidence scores
        pred_label = torch.argmax(logits, dim=1).item()  # Predicted label
    
    return pred_label, confidences

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Transform the image without normalization for visualization and prediction
    transformed_image = transform(image).squeeze(0).unsqueeze(0)  # Remove extra channel dimension and add batch dimension

    # Print statistics about the transformed image tensor
    # print(f'Transformed Image Tensor: {transformed_image}')
    # print(f'Shape: {transformed_image.shape}')
    # print(f'Data Type: {transformed_image.dtype}')
    # print(f'Min Value: {transformed_image.min()}')
    # print(f'Max Value: {transformed_image.max()}')

    # Convert the transformed image to PIL and encode it to base64
    transformed_image_pil = transforms.ToPILImage()(transformed_image.squeeze(0))
    buffered_transformed = io.BytesIO()
    transformed_image_pil.save(buffered_transformed, format="PNG")
    encoded_transformed_image = base64.b64encode(buffered_transformed.getvalue()).decode('utf-8')

    # Predict the digit and get confidence scores
    device = torch.device('cpu')
    prediction, confidences = test_single_image(model, transformed_image, device)

    # Print the prediction result in the terminal
    print(f'Prediction: {prediction}')
    print(f'Confidences: {confidences}')

    return jsonify({
        'prediction': prediction,
        'transformed_image': f'data:image/png;base64,{encoded_transformed_image}',
        'confidences': confidences
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3788)