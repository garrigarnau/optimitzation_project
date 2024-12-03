import torch
from pathlib import Path
from src.data.data_loader import get_mnist_loaders
from src.models.conv_net import SimpleConvNet
from src.training.trainer import Trainer
from src.utils.config import load_config

def main():
    # Crear directori de resultats
    Path('results').mkdir(exist_ok=True)
    
    # Carregar configuració
    config = load_config()
    
    # Preparar dades
    train_loader, test_loader = get_mnist_loaders(config['training']['batch_size'])
    
    # Crear model
    model = SimpleConvNet(config)
    
    # Iniciar entrenament
    trainer = Trainer(model, train_loader, test_loader, config)
    
    # Entrenar model
    for epoch in range(config['training']['epochs']):
        loss = trainer.train_epoch()
        accuracy = trainer.evaluate()
        print(f'Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%')
    
    # Guardar resultats
    trainer.save_results()
    print("Training finished. Results saved in 'results' directory.")

if __name__ == "__main__":
    main()