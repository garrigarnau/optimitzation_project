import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
import json

class Trainer:
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum']
        )
        self.history = {
            'loss': [],
            'accuracy': [],
            'time': []
        }
        
    def train_epoch(self):
        self.model.train()
        start_time = time.time()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_time = time.time() - start_time
        
        self.history['loss'].append(epoch_loss)
        self.history['time'].append(epoch_time)
        
        return epoch_loss
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        self.history['accuracy'].append(accuracy)
        return accuracy
    
    def save_results(self):
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Guardar model
        torch.save(self.model.state_dict(), results_dir / 'model.pth')
        
        # Guardar històric
        with open(results_dir / 'history.json', 'w') as f:
            json.dump(self.history, f)