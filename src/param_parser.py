"""Getting params from the command line."""
import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Graph Similarity Learning with SimGNN and Caption Generation.")
    
    # Data parameters
    parser.add_argument("--base-path",
                        type=str,
                        default="/home/akshay/com_full/instruments18_caption",
                        help="Base path to prepend to id_path values.")
    parser.add_argument("--similarities-file",
                        type=str,
                        default="/home/akshay/com_full/modified_captions/scores.json",
                        help="Path to the similarity scores JSON file.")
    parser.add_argument("--template-file", 
                        type=str, 
                        default="/home/akshay/com_full/modified_captions/template.json",
                        help="Path to template graphs and captions JSON file.")
    parser.add_argument("--train-captions-file", 
                        type=str, 
                        default="/home/akshay/com_full/annotations_resnet/captions_train.json",
                        help="Path to training captions JSON file.")
    parser.add_argument("--val-captions-file", 
                        type=str, 
                        default="/home/akshay/com_full/annotations_resnet/captions_val.json",
                        help="Path to validation captions JSON file.")
    parser.add_argument("--training-split",
                        type=float,
                        default=0.6,
                        help="Training data ratio (default: 0.6).")
    parser.add_argument("--validation-split",
                        type=float,
                        default=0.2,
                        help="Validation data ratio (default: 0.2).")
    parser.add_argument("--testing-split",
                        type=float,
                        default=0.2,
                        help="Testing data ratio (default: 0.2).")
    parser.add_argument("--max-nodes",
                        type=int,
                        default=6,
                        help="Maximum number of nodes in graphs (default: 6).")
                        
    # Model parameters
    parser.add_argument("--filters-1",
                        type=int,
                        default=128,
                        help="Filters (neurons) in 1st convolution (default: 128).")
    parser.add_argument("--filters-2",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 2nd convolution (default: 64).")
    parser.add_argument("--filters-3",
                        type=int,
                        default=32,
                        help="Filters (neurons) in 3rd convolution (default: 32).")
    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
                        help="Neurons in tensor network layer (default: 16).")
    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
                        help="Bottle neck layer neurons (default: 16).")
                        
    # Training parameters
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Number of graph pairs per batch (default: 32).")
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="Number of training epochs (default: 20).")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate (default: 0.001).")
    parser.add_argument("--caption-learning-rate",
                        type=float,
                        default=1e-5,
                        help="Learning rate for caption model (default: 1e-5).")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout probability (default: 0.5).")
    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-5,
                        help="Adam weight decay (default: 10^-5).")
                        
    # Caption generation parameters
    parser.add_argument("--use-geometric",
                        action="store_true",
                        help="Use PyTorch Geometric GNN implementation.")
    parser.add_argument("--beam-width",
                        type=int,
                        default=3,
                        help="Beam width for beam search decoding.")
    parser.add_argument("--max-length",
                        type=int,
                        default=50,
                        help="Maximum caption length.")
    parser.add_argument("--primary-metric",
                        type=str,
                        default="ROUGE_L",
                        help="Primary metric for model selection.")
    parser.add_argument("--joint-training",
                        action="store_true",
                        help="Train SimGNN and caption models jointly.")
                        
    # Model saving/loading
    parser.add_argument("--save-path",
                        type=str,
                        default="/home/akshay/com_full/joint_model.pt",
                        help="Where to save the trained model (default: ./models/joint_model.pt).")
    parser.add_argument("--load-path",
                        type=str,
                        default="",
                        help="Load a pretrained model from path (default: none).")
                        
    return parser.parse_args()
