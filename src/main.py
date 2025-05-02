"""SimGNN runner."""
from src.utils import tab_printer
from src.simgnn import SimGNNTrainer
from src.param_parser import parameter_parser
import torch
import os

def main():
    """
    Parse command line parameters,
    load data, train and evaluate a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    
    # Check CUDA availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create necessary directories if they don't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    if args.similarities_file:
        os.makedirs(os.path.dirname(args.similarities_file), exist_ok=True)
    
    # Initialize trainer
    trainer = SimGNNTrainer(args)
    
    # Train or load model
    if args.load_path and os.path.exists(args.load_path):
        trainer.load()
    else:
        trainer.fit()
    
    # Evaluate model
    trainer.score()
    
    # Save model if specified
    if args.save_path:
        trainer.save()

if __name__ == "__main__":
    main()
