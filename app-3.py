import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing import image
import customtkinter as ctk  # Import customtkinter

# Define constants
image_size = 128  # Image size used during training
batch_size = 32

# Load InceptionV3 model without top layers
inception_v3_model = InceptionV3(include_top=False, input_shape=(image_size, image_size, 3), pooling='avg')
inception_v3_model.trainable = False

# Build your sequential model
inception_cnn_model = Sequential([
    inception_v3_model,
    Flatten(),
    Dense(1024, activation='relu', name='Hidden-Layer-1'),
    Dense(4, activation='softmax', name='Output-Layer')
])

# Compile the model
inception_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load weights from local checkpoint (TensorFlow format)
checkpoint_path = './Model-Checkpoints/InceptionV3'  # Update this to your local path
inception_cnn_model.load_weights(checkpoint_path)

# Static information about each tumor type
tumor_info = {
    "Glioma": "Gliomas are tumors that arise from glial cells, which support the nerve cells of the brain. They can occur in the brain or spinal cord and are categorized into different types (e.g., astrocytomas, oligodendrogliomas). Gliomas are graded based on their aggressiveness (grades I to IV), with higher grades indicating more aggressive behavior. Treatment options include surgery, radiation therapy, and chemotherapy, depending on the tumor location and grade.",
    
    "Meningioma": "Meningiomas are typically slow-growing tumors that originate from the meninges, the layers of tissue that cover the brain and spinal cord. These tumors are usually benign (noncancerous), but they can cause symptoms if they grow large enough to press on nearby structures. Meningiomas are more common in women and often occur in adults. Treatment options include observation, surgery, and in some cases, radiation therapy.",
    
    "No Tumor": "No tumor detected. The MRI scan does not show any abnormal growths or lesions in the brain. This result is reassuring, indicating the absence of pathological findings that could cause symptoms or require further investigation.",
    
    "Pituitary tumor": "Pituitary tumors are growths that develop in the pituitary gland, a small gland located at the base of the brain. These tumors can be noncancerous (benign) or cancerous (rare). Depending on their size and type, pituitary tumors can cause hormone imbalances, leading to symptoms such as headaches, vision problems, and hormonal disturbances. Treatment options include observation, medication, surgery, and radiation therapy, tailored to the specific characteristics of the tumor."
}


# Function to process and predict
def process_and_predict(img_path):
    img = image.load_img(img_path, target_size=(image_size, image_size))  # Resize to 128x128
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    
    predictions = inception_cnn_model.predict(img_array)
    return predictions

# Function to get image size
def get_image_size(img_path):
    with Image.open(img_path) as img:
        return img.size

# Function to handle file selection and display
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize for display in Tkinter window
        img = ImageTk.PhotoImage(img)
        
        panel.configure(image=img)
        panel.image = img
        
        predictions = process_and_predict(file_path)[0]
        
        # Clear the table before adding new predictions
        for row in tree.get_children():
            tree.delete(row)
        
        # Add predictions to the table
        for i, (label, prob) in enumerate(zip(labels, predictions)):
            if prob > 0.8:  # Check if probability is greater than 80%
                tree.insert("", "end", values=(label, f"{prob:.2%}", 'red'))  # Add red color tag for high probability
            else:
                tree.insert("", "end", values=(label, f"{prob:.2%}"))


        # Display image size
        width, height = get_image_size(file_path)
        size_label.configure(text=f"Image Size: {width} x {height}")
        
        # Display detailed information based on the highest prediction
        max_prob = max(predictions)
        if max_prob > 0.8:
            max_index = np.argmax(predictions)
            detail_label.configure(text=tumor_info[labels[max_index]], font=('Arial', 14), wraplength=400, justify='left')
        else:
            detail_label.configure(text="No high-confidence prediction.")

# Setup customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Setup Tkinter
root = ctk.CTk()
root.title("Brain Tumor Detection")

# Set the window size and center it
window_width = 1200
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

# Main frame
main_frame = ctk.CTkFrame(root, fg_color="#2E2E2E", corner_radius=10)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Image display panel
panel_frame = ctk.CTkFrame(main_frame, fg_color="#1E1E1E", corner_radius=10)
panel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=10)

panel_title = ctk.CTkLabel(panel_frame, text="MRI Image", font=('Helvetica', 18, 'bold'), fg_color="#1E1E1E")
panel_title.pack(pady=10)

panel = ctk.CTkLabel(panel_frame)
panel.configure(text="")
panel.pack(pady=10)

# Upload button
btn = ctk.CTkButton(panel_frame, text="Upload MRI Image", command=open_file, font=('Helvetica', 14, 'bold'))
btn.pack(pady=10)

# Result display
result_frame = ctk.CTkFrame(main_frame, fg_color="#1E1E1E", corner_radius=10)
result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=10)

result_title = ctk.CTkLabel(result_frame, text="Prediction Results", font=('Helvetica', 18, 'bold'), fg_color="#1E1E1E")
result_title.pack(pady=10)

# Define the labels for each class
labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary tumor"]

# Style for the Treeview
style = ttk.Style()
style.configure("Treeview.Heading", font=('Helvetica', 16, 'bold'))  # Increase font size of headings

# Increase font size for Treeview cells
style.configure("Treeview", font=('Arial', 16))

# Table to display predictions
tree = ttk.Treeview(result_frame, columns=("Class", "Probability"), show='headings', height=4, style="Treeview")
tree.heading("Class", text="Class")
tree.heading("Probability", text="Probability")
tree.column("Class", anchor=tk.CENTER, width=200)
tree.column("Probability", anchor=tk.CENTER, width=200)
tree.pack(fill=tk.BOTH, expand=True)

size_label = ctk.CTkLabel(result_frame, text="Image Size: ", font=('Helvetica', 14))
size_label.pack(pady=10)

# Detailed information frame
detail_frame = ctk.CTkFrame(main_frame, fg_color="#1E1E1E", corner_radius=10)
detail_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=10)

detail_title = ctk.CTkLabel(detail_frame, text="Detailed Information", font=('Helvetica', 18, 'bold'), fg_color="#1E1E1E")
detail_title.pack(pady=10)

detail_label = ctk.CTkLabel(detail_frame, text="", wraplength=400, font=('Arial', 14), justify='left')
detail_label.pack(pady=10)

root.mainloop()
