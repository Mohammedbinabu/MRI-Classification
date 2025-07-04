from tensorflow.keras.models import load_model

model = load_model("my_model.h5")

image_path = r"/home/gpu-linux/Desktop/MRI-Classification/data/test/MIldDemented/0aded37d-6f68-4fc0-8031-ebf4714d3436.jpg"  # Replace with the path to your image

def predict_image(image_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np

    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)[0]  # Return the predicted class index


if __name__ == "__main__":
    alzheimer_classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    result = predict_image(image_path)
    print(f"Prediction for the image: {alzheimer_classes[result]}")