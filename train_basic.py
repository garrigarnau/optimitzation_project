import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os

# Assegurar que el directori per guardar els resultats existeix
if not os.path.exists('results'):
    os.makedirs('results')

# 1. Definir el model
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 1600)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Configurar les dades
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Descarregar i preparar el dataset MNIST
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# 3. Funció d'entrenament
def train_model(model, train_loader, test_loader, num_epochs=5):  # Afegit test_loader com a paràmetre
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    losses = []
    times = []
    accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                avg_loss = running_loss / 100
                losses.append(avg_loss)
                print(f'[Època {epoch + 1}, Batch {i + 1}] loss: {avg_loss:.3f}')
                running_loss = 0.0
                
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        
        # Calcular precisió
        accuracy = evaluate_model(model, test_loader)
        accuracies.append(accuracy)
        
        print(f'Època {epoch + 1} completada en {epoch_time:.2f} segons')
        print(f'Precisió actual: {accuracy:.2f}%\n')
    
    return losses, times, accuracies

# 4. Funció d'avaluació
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# 5. Funció per visualitzar resultats
def plot_results(losses, times, accuracies):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Pèrdua durant entrenament')
    plt.xlabel('Iteracions (x100)')
    plt.ylabel('Pèrdua')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Precisió per època')
    plt.xlabel('Època')
    plt.ylabel('Precisió (%)')
    
    plt.tight_layout()
    plt.savefig('results/training_results.png')
    plt.close()

# 6. Funció principal
def main():
    print("Iniciant entrenament bàsic...")
    
    # Preparar dades
    train_loader, test_loader = load_data()
    print("Dades carregades correctament")
    
    # Crear i entrenar el model
    model = SimpleConvNet()
    print("Model creat. Començant entrenament...")
    
    # Entrenar i obtenir resultats
    losses, times, accuracies = train_model(model, train_loader, test_loader)  # Afegit test_loader aquí
    
    # Guardar resultats
    print("\nResultats finals:")
    print(f"Temps total d'entrenament: {sum(times):.2f} segons")
    print(f"Precisió final: {accuracies[-1]:.2f}%")
    
    # Visualitzar resultats
    plot_results(losses, times, accuracies)
    print("Gràfiques guardades a 'results/training_results.png'")
    
    # Guardar el model
    torch.save(model.state_dict(), 'results/model_basic.pth')
    print("Model guardat a 'results/model_basic.pth'")

if __name__ == "__main__":
    main()