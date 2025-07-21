import os
import pandas as pd
from inference import predict_image
from sklearn.metrics import confusion_matrix,f1_score,classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

MildDemented = os.listdir("/home/gpu-linux/Desktop/MRI-Classification/data/test/MIldDemented")
ModerateDemented = os.listdir("/home/gpu-linux/Desktop/MRI-Classification/data/test/ModerateDemented")
NonDemented = os.listdir("/home/gpu-linux/Desktop/MRI-Classification/data/test/NonDemented")
VeryMildDemented = os.listdir("/home/gpu-linux/Desktop/MRI-Classification/data/test/VeryMildDemented")

name = []
class_n = []
for i in MildDemented:
    name.append(f'/home/gpu-linux/Desktop/MRI-Classification/data/test/MIldDemented/{i}')
    class_n.append(0)
for i in ModerateDemented:
    name.append(f'/home/gpu-linux/Desktop/MRI-Classification/data/test/ModerateDemented/{i}')
    class_n.append(1)
for i in NonDemented:
    name.append(f'/home/gpu-linux/Desktop/MRI-Classification/data/test/NonDemented/{i}')
    class_n.append(2)
for i in VeryMildDemented:
    name.append(f'/home/gpu-linux/Desktop/MRI-Classification/data/test/VeryMildDemented/{i}')
    class_n.append(3)
df = pd.DataFrame({"name": name, "class": class_n})
MildDemented = df[df["class"] == 0]
ModerateDemented = df[df["class"] == 1]
NonDemented = df[df["class"] == 2]
VeryMildDemented = df[df["class"] == 3]
MildDemented.to_csv("/home/gpu-linux/Desktop/MRI-Classification/data/test/MildDemented.csv", index=False)
ModerateDemented.to_csv("/home/gpu-linux/Desktop/MRI-Classification/data/test/ModerateDemented.csv", index=False)
NonDemented.to_csv("/home/gpu-linux/Desktop/MRI-Classification/data/test/NonDemented.csv", index=False)
VeryMildDemented.to_csv("/home/gpu-linux/Desktop/MRI-Classification/data/test/VeryMildDemented.csv", index=False)
df.to_csv("/home/gpu-linux/Desktop/MRI-Classification/data/test/test_dataset.csv", index=False)
print("CSV files created successfully for each class in the test dataset.")



df = pd.read_csv("/home/gpu-linux/Desktop/MRI-Classification/data/test/test_dataset.csv")
pred =[]
for i in range(len(df)):
    image_path = df["name"][i]
    result = predict_image(image_path)
    print(f"Prediction for the image {image_path}: {result}")
    pred.append(result)

y = df["class"].tolist()
cm = confusion_matrix(y, pred)
classification_report = classification_report(y, pred, target_names=["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"])

print('Classification Report:\n', classification_report)

alzheimer_classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=alzheimer_classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
plt.savefig('/home/gpu-linux/Desktop/MRI-Classification/data/test/confusion_matrix.jpeg')
print("Confusion matrix saved as 'confusion_matrix.png'.")