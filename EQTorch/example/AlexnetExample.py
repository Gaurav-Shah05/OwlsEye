import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from qtorch import Posit, FixedPosit
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Hyperparameters
num_classes = 10
num_epochs = 30
batch_size = 64
learning_rate = 0.005


# Define transformations for the training and test sets
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,))
])


# Load datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)


# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define the AlexNet model (with quantization support)
class AlexNet(nn.Module):
   def __init__(self, quant, num_classes=10):
       super(AlexNet, self).__init__()
       self.features = nn.Sequential(
           nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),
           quant(),
           nn.Conv2d(64, 192, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),
           quant(),
           nn.Conv2d(192, 384, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           nn.Conv2d(384, 256, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           nn.Conv2d(256, 256, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),
           quant(),
       )
       self.classifier = nn.Sequential(
           nn.Dropout(),
           nn.Linear(256 * 4 * 4, 4096),
           nn.ReLU(inplace=True),
           nn.Dropout(),
           nn.Linear(4096, 4096),
           nn.ReLU(inplace=True),
           nn.Linear(4096, num_classes),
       )


   def forward(self, x):
       x = self.features(x)
       x = torch.flatten(x, 1)
       x = self.classifier(x)
       return x


# Quantization setup
# bit_8 = Posit(nsize=7, es=2)
# bit_8_scale = Posit(nsize=7, es=2, scale=16)
# bit_16 = Posit(nsize=16, es=2)
bit_8 = FixedPosit(nsize=30, es=6, rf=2)
# bit_8_scale = FixedPosit(nsize=30, es=6, rf=2, scale=16)
bit_16 = FixedPosit(nsize=30, es=6, rf=2)
weight_quant = quantizer(bit_8)
grad_quant = quantizer(bit_8)
momentum_quant = quantizer(bit_16)
acc_quant = quantizer(bit_16)
act_error_quant = lambda: Quantizer(forward_number=bit_8, backward_number=bit_8)


# Initialize model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexNet(act_error_quant, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = OptimLP(optimizer,
#                     weight_quant=weight_quant,
#                     grad_quant=grad_quant,
#                     momentum_quant=momentum_quant)
optimizer = OptimLP(optimizer,
                   weight_quant=weight_quant,
                   grad_quant=grad_quant,
                   momentum_quant=momentum_quant,
                   acc_quant=acc_quant)
                   # grad_scaling=2**5)








# Validation loop
def validate(model, test_loader):
   model.eval()
   with torch.no_grad():
       correct = 0
       total = 0
       for images, labels in test_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()


       accuracy = 100 * correct / total
       print(f'Validation Accuracy: {accuracy:.2f}%')


# Testing loop
def test(model, test_loader):
   model.eval()
   with torch.no_grad():
       correct = 0
       total = 0
       for images, labels in test_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()


       accuracy = 100 * correct / total
       print(f'Test Accuracy: {accuracy:.2f}%')


# Perform validation and testing


total_step = len(train_loader)
train_losses = []
valid_accuracies = []


for epoch in range(num_epochs):
   start_time = time.time()
  
   running_loss = 0.0
   for i, (images, labels) in enumerate(train_loader): 
       # Move tensors to the configured device
       images = images.to(device)
       labels = labels.to(device)
      
       # Forward pass
       outputs = model(images)
       loss = criterion(outputs, labels)
      
       # Backward and optimize
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
      
       running_loss += loss.item()
      
       if (i + 1) % 100 == 0:
           print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
  
   # Calculate and store average training loss
   epoch_loss = running_loss / total_step
   train_losses.append(epoch_loss)
  
   # Validation
   model.eval()
   correct = 0
   total = 0
   with torch.no_grad():
       for images, labels in test_loader:
           images = images.to(device)
           labels = labels.to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
  
   accuracy = 100 * correct / total
   valid_accuracies.append(accuracy)
  
   elapsed_time = time.time() - start_time
   remaining_time = elapsed_time * (num_epochs - (epoch + 1))
  
   print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Accuracy: {:.2f}%, Time Elapsed: {:.2f}s, Estimated Time Left: {:.2f}s'
         .format(epoch + 1, num_epochs, epoch_loss, accuracy, elapsed_time, remaining_time))


   model.train()


# Plot Loss and Accuracy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))


ax1.plot(range(1, num_epochs + 1), train_losses, 'b', label='Training loss')
ax1.set_title('Training loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()


ax2.plot(range(1, num_epochs + 1), valid_accuracies, 'r', label='Validation accuracy')
ax2.set_title('Validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()


plt.tight_layout()
# plt.show()
plt.savefig('Loss_Accuracy.png')


