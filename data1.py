from datasets import load_dataset
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load the dataset
ds = load_dataset("SilpaCS/Alzheimer", split="train")

# Define the root directories for train and test data
output_base_dir = "alzheimer_dataset"
train_output_root_dir = os.path.join(output_base_dir, "train")
test_output_root_dir = os.path.join(output_base_dir, "test")

# Create base directories if they don't exist
os.makedirs(train_output_root_dir, exist_ok=True)
os.makedirs(test_output_root_dir, exist_ok=True)

print(f"Starting to process {len(ds)} images...")

# Get the label names
label_names = None
if 'label' in ds.features and hasattr(ds.features['label'], 'names'):
    label_names = ds.features['label'].names
    print(f"Detected label names: {label_names}")
else:
    print("Label names not found in dataset features. Using numeric labels for folder names.")
    # If label names aren't available, we'll create a mapping for numeric labels
    unique_labels = sorted(list(set(item['label'] for item in ds)))
    label_names = {label: f"label_{label}" for label in unique_labels}


# Group data by label
data_by_label = defaultdict(list)
for i, item in enumerate(ds):
    data_by_label[item['label']].append(item)

# Perform stratified train-test split for each label
train_data = []
test_data = []

for label, items_for_label in data_by_label.items():
    if len(items_for_label) < 5: # Small classes might not split well with 20%
        print(f"Warning: Class '{label_names[label]}' has only {len(items_for_label)} samples. All will go to train.")
        train_data.extend(items_for_label)
        continue

    # Split current label's data
    # random_state for reproducibility
    train_split, test_split = train_test_split(
        items_for_label,
        test_size=0.2, # 20% for test
        random_state=42 # Ensure reproducibility of the split
    )
    train_data.extend(train_split)
    test_data.extend(test_split)

print(f"Split complete. Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# Function to save images
def save_images(data_list, base_dir, label_names_map):
    for i, item in enumerate(data_list):
        image = item['image']
        label = item['label']

        folder_name = label_names_map[label]
        output_dir = os.path.join(base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        image_filename = f"image_{i:05d}.jpg"
        image_path = os.path.join(output_dir, image_filename)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path)
        if (i + 1) % 500 == 0:
            print(f"Saved {i + 1} images to {base_dir}...")

print("\nSaving training images...")
save_images(train_data, train_output_root_dir, label_names)

print("\nSaving testing images...")
save_images(test_data, test_output_root_dir, label_names)

print(f"\nDataset successfully split and saved to:\n- {train_output_root_dir}\n- {test_output_root_dir}")
print("Directory structure will be:")
print(f"{output_base_dir}/")
print(f"├── train/")
print(f"│   ├── {label_names[0] if isinstance(label_names, list) else label_names[list(label_names.keys())[0]]}/") # Example label folder
print(f"│   └── ...")
print(f"└── test/")
print(f"    ├── {label_names[0] if isinstance(label_names, list) else label_names[list(label_names.keys())[0]]}/") # Example label folder
print(f"    └── ...")