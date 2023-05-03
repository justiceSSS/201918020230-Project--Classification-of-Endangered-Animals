import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import torch
from torchvision import datasets, models, transforms
from tkinter import filedialog

# Define the image transforms
image_transforms = {
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Define the label mapping
label_map = {
    0: 'panda',
    1: 'pangolin',
    2: 'sea turtle'
}

# Load the model
model = torch.load('./models/data_model_1.pt', map_location=torch.device('cpu'))

# Define the function to classify the image
def classify_image():
    # Get the image path
    image_path = filedialog.askopenfilename(initialdir='./', title='Select Image',
                                            filetypes=(('Image files', '*.jpg;*.jpeg;*.png'),
                                                       ('All files', '*.*')))
    # Open and transform the image
    image = Image.open(image_path)
    image = image_transforms['valid'](image).unsqueeze(0)

    # Predict the class probabilities and label
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        class_idx = np.argmax(probabilities)
        label = label_map[class_idx]
        accuracy = probabilities[class_idx] * 100

    # Display the image and predicted label
    image = ImageTk.PhotoImage(Image.open(image_path).resize((224, 224)))
    image_label.config(image=image)
    image_label.image = image
    result_label.config(text=f'Predicted species: {label}\nAccuracy: {accuracy:.2f}%')

# Create the main window
root = tk.Tk()
root.title('Animal Classifier')

# Create the button to select an image
select_button = tk.Button(root, text='Select Image', command=classify_image)
select_button.pack(padx=10, pady=10)

# Create the image label
image_label = tk.Label(root)
image_label.pack(padx=10, pady=10)

# Create the label to display the predicted species and accuracy
result_label = tk.Label(root, text='')
result_label.pack(padx=10, pady=10)

# Start the main loop
root.mainloop()





