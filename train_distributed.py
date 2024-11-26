import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import time
import os
from multiprocessing import Process, Queue
import numpy as np


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

def train_subset(process_id, dataset_indices, queue):
    """Entrena el model amb un subconjunt de dades"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    subset = Subset(full_dataset, dataset_indices)
    train_loader = DataLoader(subset, batch_size=64, shuffle=True)
    
    model = SimpleConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    start_time = time.time()
    losses = []
    times = []
    accuracies = []
    
    # Afegim test_loader per calcular precisió
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    for epoch in range(5):
        epoch_start = time.time()
        running_loss = 0.0
        batch_losses = []
        
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                avg_loss = running_loss / 100
                batch_losses.append(avg_loss)
                print(f'Procés {process_id} - [Època {epoch + 1}, Batch {i + 1}] loss: {avg_loss:.3f}')
                running_loss = 0.0
        
        # Guardar mètriques per època
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        losses.extend(batch_losses)
        
        # Calcular precisió
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        
    training_time = time.time() - start_time
    
    # Enviar més informació al procés principal
    queue.put({
        'process_id': process_id,
        'training_time': training_time,
        'losses': losses,
        'times': times,
        'accuracies': accuracies,
        'final_loss': losses[-1] if losses else 0,
        'state_dict': model.state_dict()
    })

def plot_distributed_results(results):
    """Crear gràfiques dels resultats distribuïts"""
    plt.figure(figsize=(15, 5))
    
    # Gràfica de pèrdues
    plt.subplot(1, 3, 1)
    for result in results:
        plt.plot(result['losses'], 
                label=f'Procés {result["process_id"]}')
    plt.title('Pèrdua durant entrenament')
    plt.xlabel('Iteracions (x100)')
    plt.ylabel('Pèrdua')
    plt.legend()
    
    # Gràfica de temps
    plt.subplot(1, 3, 2)
    for result in results:
        plt.plot(result['times'], 
                label=f'Procés {result["process_id"]}')
    plt.title('Temps per època')
    plt.xlabel('Època')
    plt.ylabel('Temps (s)')
    plt.legend()
    
    # Gràfica de precisió
    plt.subplot(1, 3, 3)
    for result in results:
        plt.plot(result['accuracies'], 
                label=f'Procés {result["process_id"]}')
    plt.title('Precisió per època')
    plt.xlabel('Època')
    plt.ylabel('Precisió (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/distributed_training_results.png')
    plt.close()

def main():
    print("Iniciant entrenament distribuït simplificat...")
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    num_samples = len(full_dataset)
    
    num_processes = 2
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_processes)
    
    queue = Queue()
    processes = []
    start_time = time.time()
    
    for i in range(num_processes):
        p = Process(target=train_subset, args=(i, split_indices[i], queue))
        processes.append(p)
        p.start()
    
    # Recollir resultats
    results = []
    for _ in range(num_processes):
        results.append(queue.get())
    
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # Mostrar resultats
    print("\nResultats de l'entrenament distribuït:")
    print(f"Temps total: {total_time:.2f} segons")
    for result in results:
        print(f"Procés {result['process_id']}:")
        print(f"- Temps d'entrenament: {result['training_time']:.2f} segons")
        print(f"- Pèrdua final: {result['final_loss']:.3f}")
        print(f"- Precisió final: {result['accuracies'][-1]:.2f}%")
    
    # Crear gràfiques
    plot_distributed_results(results)
    print("\nGràfiques guardades a 'results/distributed_training_results.png'")
    
    # Guardar resultats complets
    torch.save({
        'time': total_time,
        'process_results': results
    }, 'results/distributed_results.pt')
    
    print("Resultats complets guardats a 'results/distributed_results.pt'")

if __name__ == "__main__":
    main()