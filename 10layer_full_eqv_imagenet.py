import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from e2cnn import gspaces, nn as enn
import os

class GEquivariantCNN10(nn.Module):
    def __init__(self, input_channels=3, num_classes=100):
        super(GEquivariantCNN10, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)

        def make_layer(in_type, out_channels, kernel_size=3, padding=1, pool=False):
            out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
            layers = [
                enn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding),
                enn.InnerBatchNorm(out_type),
                enn.ReLU(out_type)
            ]
            if pool:
                layers.append(enn.PointwiseMaxPool(out_type, kernel_size=2))
            return enn.SequentialModule(*layers), out_type

        # Input Field - Store this as an instance variable
        self.in_type = enn.FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])

        self.block1, out_type1 = make_layer(self.in_type, 16)
        self.block2, out_type2 = make_layer(out_type1, 32, pool=True)
        self.block3, out_type3 = make_layer(out_type2, 32)
        self.block4, out_type4 = make_layer(out_type3, 64, pool=True)
        self.block5, out_type5 = make_layer(out_type4, 64)
        self.block6, out_type6 = make_layer(out_type5, 64, pool=True)
        self.block7, out_type7 = make_layer(out_type6, 128)
        self.block8, out_type8 = make_layer(out_type7, 128)
        self.block9, out_type9 = make_layer(out_type8, 128)
        self.block10, out_type10 = make_layer(out_type9, 256)

        self.global_pool = enn.PointwiseAdaptiveAvgPool(out_type10, output_size=1)
        self.fc = nn.Linear(256 * self.r2_act.fibergroup.order(), num_classes)

    def forward(self, x):
        # Use the stored input type instead of trying to access block1[0]
        x = enn.GeometricTensor(x, self.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.global_pool(x)
        x = x.tensor.view(x.tensor.size(0), -1)
        return self.fc(x)



# Dataset paths
train_path = "./ImageNet100/train"
test_path = "./ImageNet100/val"

# Define transforms for ImageNet
train_transform = transforms.Compose([
    transforms.Resize((224,224)),                    # Resize shorter side to 256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),                    # Resize shorter side to 256  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load datasets directly
print("=" * 80)
print("LOADING IMAGENET-100 DATASET")
print("=" * 80)

print("Loading datasets...")
train_dataset = ImageFolder(train_path, transform=train_transform)
test_dataset = ImageFolder(test_path, transform=test_transform)

print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")
print(f"✓ Number of classes: {len(train_dataset.classes)}")

# Verify both datasets have same classes
assert train_dataset.classes == test_dataset.classes, "Train and test datasets have different classes!"

class_names = train_dataset.classes

# Create data loaders with optimal settings for ImageNet-1000
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

print(f"\nDataset loading complete!")
print(f"Training batches: {len(train_loader)}")
print(f"Testing batches: {len(test_loader)}")
print(f"Classes range from: {class_names[0]} to {class_names[-1]}")

# Define device, model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = GEquivariantCNN10(input_channels=3,num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

# Training Loop with learning rate scheduling
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

num_epochs = 50  # Increased for ImageNet-100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        # Print progress every 100 batches (adjusted for larger dataset)
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")
    
    # Step the scheduler
    scheduler.step()
    
    train_accuracy = 100 * correct_train / total_train
    avg_loss = running_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch + 1}/{num_epochs}] Complete - "
          f"Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, LR: {current_lr:.6f}")
    
    # Test every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        print(f"Test Accuracy after epoch {epoch + 1}: {test_accuracy:.2f}%")
    

# Save the final trained model
model_save_path = "10layerfulleqv_imagenet100.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\nFinal FullyEquivariantCNN model saved to '{model_save_path}'")

# Final Testing Loop
print("\nFinal evaluation...")
model.eval()
correct = 0
total = 0
class_correct = [0] * 100
class_total = [0] * 100

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

final_accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")

# Print per-class accuracy for first 10 classes as sample
print(f"\nPer-class accuracy (first 10 classes):")
for i in range(10):
    if class_total[i] > 0:
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f"Class {i} ({class_names[i]}): {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

print("\nTraining completed!")
