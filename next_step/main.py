from pathlib import Path
from src.training.experiment_runner import ExperimentRunner
from src.data.data_loader import get_mnist_loaders
from src.models.conv_net import SimpleConvNet
from src.training.trainer import Trainer
from src.utils.config import load_config

def train_basic():
   config = load_config()
   train_loader, test_loader = get_mnist_loaders(config['training']['batch_size'])
   model = SimpleConvNet(config)
   trainer = Trainer(model, train_loader, test_loader, config)
   
   for epoch in range(config['training']['epochs']):
       loss = trainer.train_epoch()
       accuracy = trainer.evaluate()
       print(f'Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%')
   
   trainer.save_results()
   print("Basic training finished. Results saved in 'results' directory.")

def run_experiments():
   runner = ExperimentRunner()
   runner.run_experiments()
   print("Experiments finished. Results saved in 'results' directory.")

def main():
   Path('results').mkdir(exist_ok=True)
   
   # Descomentar la funció que es vulgui executar
   # train_basic()
   run_experiments()

if __name__ == "__main__":
   main()