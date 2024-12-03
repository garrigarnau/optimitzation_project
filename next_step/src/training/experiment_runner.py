import itertools
import json
import matplotlib.pyplot as plt
from src.data.data_loader import get_mnist_loaders
from src.models.conv_net import SimpleConvNet
from src.training.trainer import Trainer
from src.utils.config import load_config
import torch
import torch.nn as nn

class ExperimentRunner:
    def __init__(self):
        self.config = load_config()
        self.results = {}
        
    def run_experiments(self):
        # Paràmetres a provar
        params = {
            'learning_rate': [0.001, 0.1],     # 2 valors extrems
            'batch_size': [32, 128],           # petit vs gran
            'epochs': [5],                     # fix
            'dropout': [0.0],                  # sense dropout
            'l2_reg': [0.0]                    # sense regularització
        }
        
        # Generar combinacions
        param_combinations = list(itertools.product(
            params['learning_rate'],
            params['batch_size'],
            params['epochs'],
            params['dropout'],
            params['l2_reg']
        ))
        
        for lr, bs, epochs, dropout, l2_reg in param_combinations:
            # Configurar experiment
            self.config['training']['learning_rate'] = lr
            self.config['training']['batch_size'] = bs
            self.config['training']['epochs'] = epochs
            
            # Preparar model i dades
            train_loader, test_loader = get_mnist_loaders(bs)
            model = SimpleConvNet(self.config)
            
            # Afegir dropout si necessari
            if dropout > 0:
                model.dropout = nn.Dropout(dropout)
            
            # Configurar trainer amb L2 regularització si necessari
            trainer = Trainer(model, train_loader, test_loader, self.config)
            if l2_reg > 0:
                trainer.optimizer = torch.optim.SGD(
                    model.parameters(), 
                    lr=lr, 
                    momentum=0.9,
                    weight_decay=l2_reg
                )
            
            # Entrenar
            print(f"\nExperiment amb lr={lr}, bs={bs}, epochs={epochs}, dropout={dropout}, l2_reg={l2_reg}")
            for epoch in range(epochs):
                loss = trainer.train_epoch()
                accuracy = trainer.evaluate()
                print(f'Època {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%')
            
            # Guardar resultats
            exp_name = f"lr{lr}_bs{bs}_ep{epochs}_dp{dropout}_l2{l2_reg}"
            self.results[exp_name] = trainer.history
            
        self.save_results()
        self.plot_results()
    
    def save_results(self):
        with open('results/experiment_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
            
    def plot_results(self):
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        for exp_name, history in self.results.items():
            plt.plot(history['accuracy'], label=exp_name)
        plt.title('Accuracy per Configuration')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot loss
        plt.subplot(2, 2, 2)
        for exp_name, history in self.results.items():
            plt.plot(history['loss'], label=exp_name)
        plt.title('Loss per Configuration')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot training time
        plt.subplot(2, 2, 3)
        avg_times = {exp: sum(hist['time'])/len(hist['time']) 
                    for exp, hist in self.results.items()}
        plt.bar(avg_times.keys(), avg_times.values())
        plt.title('Average Training Time per Configuration')
        plt.xticks(rotation=45)
        plt.ylabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig('results/experiment_results.png')
        plt.close()

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_experiments()