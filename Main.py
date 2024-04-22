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
        features = features.view(x.size(0), 229, 768)  # Reshape to [batch_size, 32, 768]

        # Final output for the actual task
        output = self.final_layer(x)  # Assuming a single value output (e.g., for binary classification)

        return output, features
    

# Define the Feature Distillation Loss
class FeatureDistillationLoss(nn.Module):
    def __init__(self):
        super(FeatureDistillationLoss, self).__init__()

    def forward(self, student_features, teacher_features):
        return F.mse_loss(student_features, teacher_features)

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Unnormalize an image tensor.
    
    Args:
    tensor (torch.Tensor): The normalized image tensor.
    mean (list): The mean used for normalization.
    std (list): The standard deviation used for normalization.

    Returns:
    torch.Tensor: The unnormalized image tensor.
    """
    # Clone the tensor to not do changes in-place
    tensor = tensor.clone()

    # Convert mean and std to tensors
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    # Unnormalize the image
    tensor.mul_(std).add_(mean)

    # Clip the values to be between 0 and 1
    tensor = torch.clamp(tensor, 0, 1)

    return tensor

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
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create a dataset instancecon with final_transform
dataset = MyDataset(data_file='TestMIX.csv', transform=final_transform)

# Create the final DataLoader
dataloader = DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
student_model = StudentResNet().to(device)
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
classification_criterion = nn.BCEWithLogitsLoss()
distillation_criterion = FeatureDistillationLoss()

# Define a function to extract and combine features
def extract_and_combine_features(model_blip2, processor_blip2, model_vitgpt2, feature_extractor, tokenizer, images, text):
    
    pixel_values = feature_extractor(images, return_tensors="pt", do_rescale=False).pixel_values 
    # labels = map(str, labels.tolist())
    labels = tokenizer(text, return_tensors="pt").input_ids
    
    inputs = processor_blip2(images=images, text=text, return_tensors="pt")

     
    blip2_out = model_blip2(**inputs)
    outputs = model_vitgpt2(pixel_values=pixel_values, labels=labels)


    combined_tensor = torch.cat([outputs.encoder_last_hidden_state, blip2_out.qformer_outputs.last_hidden_state], dim=1).to(device)
    return combined_tensor
   

def binary_accuracy(logits, labels):

    """Calculate accuracy for binary classification."""
    # Apply sigmoid to logits and round to get predictions
    preds = torch.sigmoid(logits).round()
    correct = torch.eq(preds, labels).float()  # Convert boolean to float and compare
    acc = correct.sum() / len(correct)
    return acc


print('Device ===== '+ str(device))
# Training loop
student_model.train()
num_epochs = 20
for epoch in range(num_epochs):
   
    epoch_loss = 0
    epoch_acc = []
    dis_loss = 0
    cls_loss = 0
    scores = []

    for batch_images, batch_text, batch_labels in dataloader:
    
        batch_images = batch_images.to(device)  # Ensure images are on the same device as the model
        batch_labels = batch_labels.to(device).float()  # Ensure labels are on the same device as the model view(-1, 1).float()

        # Forward pass through the student modelBatch Loss
        student_logits, student_features = student_model(batch_images)


        # Get combined features from teacher models for the current batch
        model_vitgpt2 = fe.model
        teacher_features_batch = extract_and_combine_features(
            fe.model_blip2, fe.processor_blip2, model_vitgpt2, fe.feature_extractor, fe.tokenizer, batch_images, list(batch_text)
        )

        student_logits = student_logits.squeeze(1)
        # Calculate classification loss<
        classification_loss = classification_criterion(student_logits, batch_labels)


        # Calculate distillation loss
        distillation_loss = distillation_criterion(student_features, teacher_features_batch)

        # Combine losses
        final_loss = classification_loss + distillation_loss

        # Backward pass and optimize
        final_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
          # Convert logits to predictions for accuracy calculation
        preds = torch.sigmoid(student_logits).round()

        
        # batch_acc = preds.eq(batch_labels.view_as(preds)).sum().item() / len(batch_labels)

        print(f'Epoch {epoch+1}, Batch Loss: {final_loss.item()}')
     
 
        # student_logits_cpu = student_logits.detach().cpu()
        batch_labels_cpu = batch_labels.detach().cpu()
        # preds = torch.sigmoid(student_logits_cpu).round()

        # Convert logits to predictions for f1_score calculation
        # Calculate f1 score
        f1_sco = f1_score(preds.detach().cpu().numpy(), batch_labels_cpu.numpy(), average='binary')
        acc_score = accuracy_score(preds.detach().cpu().numpy(),batch_labels_cpu.numpy())
  
        epoch_loss += final_loss.item()
        epoch_acc.append(acc_score.item())
        dis_loss += distillation_loss.item()
        cls_loss += classification_loss.item()
        scores.append(f1_sco.item())
        # scores += f1_sco.item()

    # Calculate average loss and accuracy for the epoch
    epoch_loss /= len(dataloader)
    # epoch_acc /= len(dataloader)
    dis_loss /= len(dataloader)
    cls_loss /= len(dataloader)
    # scores /= f1_sco.item()

    print(f'Epoch {epoch+1}: | Loss: {epoch_loss:.5f} |  Dis Loss: {dis_loss:.5f} |  Cls Loss: {cls_loss:.5f} | Acc: {np.mean(epoch_acc)*100:.2f} | F1_score: {np.mean(scores)*100:.2f}%')
    print(f'Epoch {epoch+1}: | Avg Loss: {epoch_loss:.5f} | Avg Acc: {np.mean(epoch_acc)*100:.2f}%')
# Save the model after training
torch.save(student_model.state_dict(), 'student_modelMIX.pth')
