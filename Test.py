import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score 
import pandas as pd
import numpy as np
import os
import feature_extractor as fe
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a=1
class StudentResNet(nn.Module):
    def __init__(self):
        super(StudentResNet, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        # Replace the final fully connected layer
        # ResNet18's last layer has 512 output features
        self.resnet.fc = nn.Linear(512, 128)

        # Layers to generate features for distillation
        self.fc2_features = nn.Linear(128, 768)
        self.fc3_features = nn.Linear(768, 229 * 768)

        # Final layer for the actual task (e.g., classification)
        self.final_layer = nn.Linear(128, 1)

    def forward(self, x):
        # Use the ResNet18 model to extract features
        x = self.resnet(x)

        # Feature extraction for distillation
        features = F.relu(self.fc2_features(x))
        features = self.fc3_features(features)
        features = features.view(x.size(0),229, 768)  

        # Final output for the actual task
        output = self.final_layer(x)  # Assuming a single value output (e.g., for binary classification)

        return output, features

class MyDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data = pd.read_csv(data_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.loc[idx]
        
        image = Image.open(item['image']).convert('RGB')
        label = item['label']
        text = item['text']
        if self.transform:
            image = self.transform(image)
        # Assuming 'image_tensor' is your normalized tensor
            # unnormalized_tensor = unnormalize(image)
            # pil_image = transforms.ToPILImage()(unnormalized_tensor)

        return image, text, label
    
final_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#Test Part 
# Load test data directly
test_dataset = MyDataset(data_file='DataPath.csv', transform=final_transform)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

student_model = StudentResNet().to(device)

student_model.load_state_dict(torch.load('Trained_Model_Path', map_location=torch.device('cpu')))

# Set the model to evaluation mode

student_model.eval()
# Initialize lists to store true labels and predictions
true_labels = []
pred_labels = []

# Disable gradient calculation for testing
with torch.no_grad():
    for batch_images, batch_text, batch_labels in test_dataloader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device).float()

        # Forward pass through the student model
        student_logits, _ = student_model(batch_images)
        student_logits = student_logits.squeeze(1)

        # Convert logits to predictions
        predictions = torch.sigmoid(student_logits).round()

        # Collect the true labels and predictions
        true_labels.extend(batch_labels.cpu().numpy())
        pred_labels.extend(predictions.cpu().numpy())

# Calculate accuracy and F1 score
test_acc = accuracy_score(true_labels, pred_labels)
test_f1 = f1_score(true_labels, pred_labels, average='binary')

print(f'Test Accuracy: {test_acc * 100:.2f}%')
print(f'Test F1 Score: {test_f1 * 100:.2f}%')

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Set larger font sizes
plt.rcParams.update({'font.size': 16})  # Updates the default font size for all elements
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['real', 'synthetic'], yticklabels=['real', 'synthetic'],
            annot_kws={"size": 14})  # Increase annotation text size

plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
