"""Combined model for graph matching and caption generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np
import random  # Added import for random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from torch.optim import Adam
from torch_geometric.nn import GATConv
from nltk.translate.bleu_score import sentence_bleu
import sys
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add src folder to path to import SimGNN modules
sys.path.append('./src')
# Add evaluation folder to path
sys.path.append('./evaluation')
from src.simgnn import SimGNN
from src.utils import process_pair, load_similarity_data
from src.utils import load_template_data, load_caption_data
# Added imports from utils
from src.utils import generate_similarity_scores, split_similarity_data, tab_printer
from src.param_parser import parameter_parser
# Import evaluation metrics directly
from evaluation.bleu.bleu import Bleu
from evaluation.cider.cider import Cider
from evaluation.rouge.rouge import Rouge

# Set up device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Function to compute all evaluation metrics
def compute_scores(gts, gen):
    """
    Compute the full set of NLG evaluation metrics
    Args:
        gts: Dictionary of reference captions {id: [caption1, caption2, ...]}
        gen: Dictionary of generated captions {id: [caption]}
    Returns:
        scores_dict: Dictionary of metric scores
        scores_list: List of scores for each id
    """
    metrics = {}
    scores_list = {}

    # BLEU scores
    scorer = Bleu(n=4)
    score, scores = scorer.compute_score(gts, gen)
    metrics['Bleu_1'] = score[0]
    metrics['Bleu_2'] = score[1]
    metrics['Bleu_3'] = score[2]
    metrics['Bleu_4'] = score[3]
    scores_list['Bleu_1'] = scores[0]
    scores_list['Bleu_2'] = scores[1]
    scores_list['Bleu_3'] = scores[2]
    scores_list['Bleu_4'] = scores[3]

    # CIDEr score
    scorer = Cider()
    score, scores = scorer.compute_score(gts, gen)
    metrics['CIDEr'] = score
    scores_list['CIDEr'] = scores

    # ROUGE_L score
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, gen)
    metrics['ROUGE_L'] = score
    scores_list['ROUGE_L'] = scores

    return metrics, scores_list

# Template Dataset - stores all template graphs and their captions
class TemplateGraphDataset:
    def __init__(self, template_json_file, similarity_args, tokenizer):
        """
        Initialize the template dataset
        :param template_json_file: JSON file containing template graphs and captions
        :param similarity_args: Arguments for the similarity model
        :param tokenizer: BERT tokenizer for processing captions
        """
        self.data = json.load(open(template_json_file, 'r'))
        self.args = similarity_args
        self.tokenizer = tokenizer
        self.template_graphs = []
        self.template_captions = []
        
        print(f"Loading {len(self.data)} template graphs...")
        
        # Process all template graphs and captions
        for item in self.data:
            # Process graph features
            graph_data = self._process_graph(item["id_path"])
            self.template_graphs.append(graph_data)
            
            # Process caption
            caption = item["caption"]
            caption_tokens = self.tokenizer(caption, 
                                           return_tensors="pt", 
                                           padding='max_length', 
                                           truncation=True, 
                                           max_length=50)
            
            self.template_captions.append({
                'caption': caption,
                'input_ids': caption_tokens.input_ids.squeeze(0).to(device),
                'attention_mask': caption_tokens.attention_mask.squeeze(0).to(device)
            })
        
        print(f"Loaded {len(self.template_graphs)} template graphs and captions.")
    
    def _process_graph(self, id_path):
        """
        Process a single graph
        :param id_path: Path to the graph file
        :return: Dictionary with graph features
        """
        # Create dummy pair data for process_pair function
        pair_data = {
            "id_path_1": id_path,
            "id_path_2": id_path,  # Not used but required by process_pair
            "score": 0.0,  # Not used but required by process_pair
            "train_caption": "",
            "template_caption": ""
        }
        
        # Process the graph using SimGNN's process_pair function
        data = process_pair(pair_data, self.args.max_nodes, self.args.base_path)
        
        # Convert NumPy arrays to PyTorch tensors
        graph_data = {
            'features': torch.FloatTensor(data["features_1"]).to(device),
            'edge_features': torch.FloatTensor(data["edge_features_1"]).to(device),
            'id_path': id_path
        }
        
        return graph_data
    
    def get_most_similar_caption(self, query_graph, similarity_model):
        """
        Find the most similar template graph and return its caption
        :param query_graph: Query graph features
        :param similarity_model: SimGNN model for computing similarity
        :return: Caption of the most similar template graph and its embeddings
        """
        max_similarity = -1
        most_similar_idx = -1
        
        similarity_model.eval()
        with torch.no_grad():
            for i, template_graph in enumerate(self.template_graphs):
                # Prepare data for similarity model
                data = {
                    "features_1": query_graph['features'].unsqueeze(0),
                    "features_2": template_graph['features'].unsqueeze(0),
                    "edge_features_1": query_graph['edge_features'].unsqueeze(0),
                    "edge_features_2": template_graph['edge_features'].unsqueeze(0)
                }
                
                # Compute similarity score
                similarity_score = similarity_model(data).item()
                
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    most_similar_idx = i
        
        return self.template_captions[most_similar_idx], max_similarity


# Graph Caption Dataset - for training and evaluation
class GraphCaptionDataset(Dataset):
    def __init__(self, json_file, base_path, max_nodes=6):
        """
        Initialize the dataset
        :param json_file: JSON file containing graph-caption pairs
        :param base_path: Base path for graph files
        :param max_nodes: Maximum number of nodes in graphs
        """
        self.data = json.load(open(json_file, 'r'))
        self.base_path = base_path
        self.max_nodes = max_nodes
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.data)
    
    def pad_node_features(self, node_features, max_nodes=6, feature_dim=512):
        """
        Pad node features to fixed size
        :param node_features: Node features
        :param max_nodes: Maximum number of nodes
        :param feature_dim: Feature dimension
        :return: Padded features
        """
        padded_nodes = np.zeros((max_nodes, feature_dim))
        num_nodes = min(node_features.shape[0], max_nodes)
        padded_nodes[:num_nodes, :] = node_features[:num_nodes, :]
        return padded_nodes
    
    def pad_edge_features(self, edge_features, max_nodes=6, feature_dim=512):
        """
        Pad edge features to fixed size
        :param edge_features: Edge features
        :param max_nodes: Maximum number of nodes
        :param feature_dim: Feature dimension
        :return: Padded features
        """
        padded_edges = np.zeros((max_nodes, max_nodes, feature_dim))
        
        # Handle different possible shapes of edge features
        if len(edge_features.shape) == 3:  # Already in shape (nodes, nodes, features)
            num_nodes = min(edge_features.shape[0], max_nodes)
            padded_edges[:num_nodes, :num_nodes, :] = edge_features[:num_nodes, :num_nodes, :]
        elif len(edge_features.shape) == 2:  # Shape (edges, features)
            # For this case, we need to know how to map to adjacency tensor
            # This is a placeholder - actual implementation depends on your data format
            print("Warning: Edge features in (edges, features) format not properly handled.")
        
        return padded_edges
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset
        :param idx: Index
        :return: Dictionary with graph features and caption
        """
        item = self.data[idx]
        node_path = os.path.join(self.base_path, item["id_path"])
        edge_path = node_path.replace("node", "edge")
        
        # Load node and edge features
        try:
            node_features = np.load(node_path)
            edge_features = np.load(edge_path)
            
            # Pad node and edge features
            node_features = self.pad_node_features(node_features, self.max_nodes)
            edge_features = self.pad_edge_features(edge_features, self.max_nodes)
            
            # Convert caption to BERT input tokens
            caption = item["caption"]
            tokens = self.tokenizer(caption, 
                                   return_tensors="pt", 
                                   padding='max_length', 
                                   truncation=True, 
                                   max_length=50)
            
            return {
                'node_features': torch.tensor(node_features, dtype=torch.float).to(device),
                'edge_features': torch.tensor(edge_features, dtype=torch.float).to(device),
                'caption': tokens.input_ids.squeeze(0).to(device),
                'attention_mask': tokens.attention_mask.squeeze(0).to(device),
                'id_path': item["id_path"]
            }
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # Return a dummy item in case of error
            return self.__getitem__((idx + 1) % len(self))


class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, heads=1):
        super(GATLayer, self).__init__()
        self.gat = GATConv(input_dim, output_dim, heads=heads, concat=True)
    
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return x


# Alternative GNN implementation that doesn't require PyTorch Geometric
class SimpleGNNWithoutGeometric(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim):
        super(SimpleGNNWithoutGeometric, self).__init__()
        self.node_fc1 = nn.Linear(node_input_dim, 256)
        self.node_fc2 = nn.Linear(256, 384)
        self.edge_fc = nn.Linear(edge_input_dim, 128)
        self.combined_fc = nn.Linear(512, output_dim)
    
    def forward(self, node_features, edge_features):
        batch_size, num_nodes, feature_dim = node_features.size()
        
        # Process nodes with simple FC layers
        node_emb = F.relu(self.node_fc1(node_features))  # [batch, nodes, 256]
        node_emb = F.relu(self.node_fc2(node_emb))  # [batch, nodes, 384]
        
        # Process edge features
        edge_emb = F.relu(self.edge_fc(edge_features.view(batch_size, -1)))
        edge_emb = edge_emb.unsqueeze(1).expand(-1, num_nodes, -1)
        
        # Combine node and edge features
        combined_emb = torch.cat((node_emb, edge_emb), dim=-1)
        
        # Create graph-level representation with mean pooling
        graph_emb = self.combined_fc(combined_emb.mean(dim=1))
        return graph_emb


# GNN Model with PyTorch Geometric
class SimpleGNN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim):
        super(SimpleGNN, self).__init__()
        # Reduced model size to avoid memory issues
        self.node_gat1 = GATLayer(node_input_dim, 64, heads=2)  # Smaller than original
        self.node_gat2 = GATLayer(64 * 2, 128, heads=1)  # Smaller than original
        self.edge_fc = nn.Linear(edge_input_dim, 128)
        self.combined_fc = nn.Linear(256, output_dim)  # Adjusted for reduced dimensions
    
    def forward(self, node_features, edge_features):
        batch_size, num_nodes, feature_dim = node_features.size()
        node_features = node_features.view(-1, feature_dim)  # Flatten for GAT
        edge_index = self.get_edge_index(num_nodes).to(device)
        
        node_emb = F.elu(self.node_gat1(node_features, edge_index))
        node_emb = F.elu(self.node_gat2(node_emb, edge_index))
        node_emb = node_emb.view(batch_size, num_nodes, -1)  # Reshape back to batch
        
        # Process edge features - flattened approach
        edge_emb = F.elu(self.edge_fc(edge_features.view(batch_size, -1)))
        edge_emb = edge_emb.unsqueeze(1).repeat(1, num_nodes, 1)
        
        combined_emb = torch.cat((node_emb, edge_emb), dim=-1)
        
        # Graph-level representation
        graph_emb = self.combined_fc(combined_emb.mean(dim=1))
        return graph_emb
    
    def get_edge_index(self, num_nodes):
        """
        Create fully connected edge index
        :param num_nodes: Number of nodes
        :return: Edge indices
        """
        row = torch.arange(num_nodes).repeat(num_nodes)
        col = torch.arange(num_nodes).repeat_interleave(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        
        valid_edge_mask = (row < num_nodes) & (col < num_nodes)
        edge_index = edge_index[:, valid_edge_mask]
        return edge_index


# Combined Model for joint training
class JointGraphCaptionModel(nn.Module):
    def __init__(self, args, template_dataset):
        """
        Initialize the combined model for joint training
        :param args: Arguments object
        :param template_dataset: Template dataset for similarity comparison
        """
        super(JointGraphCaptionModel, self).__init__()
        
        # Initialize SimGNN model
        self.sim_gnn = SimGNN(args)
        
        # Initialize graph GNN model
        if args.use_geometric:
            print("Using PyTorch Geometric GNN implementation")
            self.graph_gnn = SimpleGNN(
                node_input_dim=512,  # Node feature dimension
                edge_input_dim=args.max_nodes * args.max_nodes * 512,  # Flattened edge features
                output_dim=768  # BERT embedding dimension
            )
        else:
            print("Using standard PyTorch GNN implementation (no PyTorch Geometric)")
            self.graph_gnn = SimpleGNNWithoutGeometric(
                node_input_dim=512,  # Node feature dimension
                edge_input_dim=args.max_nodes * args.max_nodes * 512,  # Flattened edge features
                output_dim=768  # BERT embedding dimension
            )
        
        # Initialize BERT model
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        
        # FCN to combine graph embedding and caption embedding
        self.combine_fc1 = nn.Linear(768 * 2, 768)
        self.combine_fc2 = nn.Linear(768, 768)
        
        # Store template dataset
        self.template_dataset = template_dataset
    
    def forward_similarity(self, data):
        """
        Forward pass for similarity model
        :param data: Dict containing features_1, features_2, edge_features_1, edge_features_2
        :return: Similarity score
        """
        return self.sim_gnn(data)
    
    def forward_caption(self, node_features, edge_features, captions=None, attention_mask=None):
        """
        Forward pass for caption generation
        :param node_features: Node features
        :param edge_features: Edge features
        :param captions: Target captions (for training)
        :param attention_mask: Attention mask for captions
        :return: Loss and logits
        """
        batch_size = node_features.size(0)
        
        all_caption_embeds = []
        all_graph_embeds = []
        all_combined_embeds = []
        all_attn_masks = []
        all_target_captions = []
        
        # Process each graph in the batch
        for b in range(batch_size):
            # Extract single graph from batch
            single_graph = {
                'features': node_features[b],
                'edge_features': edge_features[b]
            }
            
            # Find most similar template caption
            template_caption, similarity_score = self.template_dataset.get_most_similar_caption(
                single_graph, self.sim_gnn)
            
            # Get graph embedding from graph GNN
            graph_emb = self.graph_gnn(
                node_features[b:b+1], 
                edge_features[b:b+1]
            )
            
            # Use template caption input_ids as intermediate representation
            caption_embed = self.bert.bert.embeddings.word_embeddings(template_caption['input_ids'])
            
            # Store embeddings and masks for processing
            all_graph_embeds.append(graph_emb)
            all_caption_embeds.append(caption_embed)
            all_attn_masks.append(template_caption['attention_mask'])
            
            if captions is not None:
                all_target_captions.append(captions[b])
        
        # Combine embeddings for all items in batch
        for b in range(batch_size):
            graph_emb = all_graph_embeds[b]  # Shape: [1, 768]
            caption_embed = all_caption_embeds[b]  # Shape: [seq_len, 768]
            
            # Check dimensions and reshape if needed
            seq_len = caption_embed.size(0)
            
            # Ensure graph_emb has the right shape: [1, 768] -> [1, seq_len, 768]
            expanded_graph_emb = graph_emb.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Now ensure both tensors are 3D (batch_size=1, seq_len, hidden_dim)
            # caption_embed needs to be: [seq_len, 768] -> [1, seq_len, 768]
            if len(caption_embed.shape) == 2:
                caption_embed = caption_embed.unsqueeze(0)
            
            # Now both should be [1, seq_len, 768]
            # Concatenate on the last dimension to get [1, seq_len, 768*2]
            concat_emb = torch.cat([expanded_graph_emb, caption_embed], dim=-1)
            
            # Apply the fully connected layers
            combined_emb = F.relu(self.combine_fc1(concat_emb))
            combined_emb = self.combine_fc2(combined_emb)
            
            all_combined_embeds.append(combined_emb)
        
        # Stack all embeddings and masks
        combined_embeds = torch.cat(all_combined_embeds, dim=0)  # Concatenate along batch dimension
        attention_masks = torch.stack(all_attn_masks)
        
        # For training phase
        if captions is not None:
            target_captions = torch.stack(all_target_captions)
            outputs = self.bert(
                inputs_embeds=combined_embeds,
                labels=target_captions,
                attention_mask=attention_masks
            )
            return outputs.loss, outputs.logits
        
        # For inference phase
        else:
            outputs = self.bert(
                inputs_embeds=combined_embeds,
                attention_mask=attention_masks
            )
            return None, outputs.logits


# Debug tensor helper function
def debug_tensor(tensor, name="tensor"):
    """Print tensor shape, type, device, and a few values for debugging"""
    print(f"\n--- DEBUG: {name} ---")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    try:
        print(f"Min/Max/Mean: {tensor.min().item():.5f} / {tensor.max().item():.5f} / {tensor.mean().item():.5f}")
    except:
        print("Could not calculate min/max/mean (possibly contains non-numeric values)")
    try:
        print(f"Sample: {tensor.flatten()[:3].tolist()}")
    except:
        print("Could not convert sample to list")
    print("-" * 50)


# Beam Search Decoding for caption generation
def beam_search_decode(logits, tokenizer, beam_width=3, max_length=50):
    """
    Beam search decoding for caption generation
    :param logits: Logits from the model
    :param tokenizer: BERT tokenizer
    :param beam_width: Beam width
    :param max_length: Maximum caption length
    :return: Generated captions
    """
    batch_size, seq_len, vocab_size = logits.size()
    all_best_sequences = []
    
    for b in range(batch_size):
        beams = [(torch.tensor([], dtype=torch.long).to(device), 0)]

        for t in range(seq_len):
            new_beams = []
            for seq, score in beams:
                probs = torch.log_softmax(logits[b, t, :], dim=-1)
                topk_probs, topk_indices = probs.topk(beam_width)

                for i in range(beam_width):
                    new_seq = torch.cat((seq, topk_indices[i].unsqueeze(0)), dim=0)
                    new_score = score + topk_probs[i].item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        best_seq = beams[0][0]
        all_best_sequences.append(best_seq)
    
    return all_best_sequences


# Evaluation function with comprehensive metrics
def evaluate(model, val_loader, tokenizer, beam_width=3, max_length=50):
    """
    Evaluate the model using all available evaluation metrics
    :param model: Model to evaluate
    :param val_loader: Validation data loader
    :param tokenizer: BERT tokenizer
    :param beam_width: Beam width for beam search
    :param max_length: Maximum caption length
    :return: Dictionary of evaluation metrics
    """
    model.eval()
    
    gen = {}  # Generated captions
    gts = {}  # Ground truth captions
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(val_loader, desc="Evaluation")):
            node_features = batch['node_features']
            edge_features = batch['edge_features']
            captions = batch['caption']
            attention_mask = batch['attention_mask']
            
            _, logits = model.forward_caption(
                node_features, 
                edge_features, 
                attention_mask=attention_mask
            )
            predicted_ids_list = beam_search_decode(
                logits, 
                tokenizer, 
                beam_width=beam_width, 
                max_length=max_length
            )
            
            for i, predicted_ids in enumerate(predicted_ids_list):
                predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                true_text = tokenizer.decode(captions[i], skip_special_tokens=True)
                
                # Store for comprehensive evaluation
                sample_id = f'{batch_id}_{i}'
                gen[sample_id] = [predicted_text]
                gts[sample_id] = [true_text]
                
                # Print a few examples
                if i < 2 and batch_id < 2:  # Print just a couple of examples
                    print(f"\nExample {sample_id}:")
                    print(f"True: {true_text}")
                    print(f"Pred: {predicted_text}")
    
    # Compute all evaluation metrics using our compute_scores function
    print("\nComputing comprehensive evaluation metrics...")
    scores, _ = compute_scores(gts, gen)
    
    # Print all metrics
    print("\nEvaluation Metrics:")
    for metric_name, score in scores.items():
        print(f"{metric_name}: {score:.4f}")
    
    return scores


# Joint training function
def train_joint_model(model, sim_data, train_loader, val_loader, args, tokenizer):
    """
    Train the combined model with both similarity and caption generation tasks
    :param model: Joint model
    :param sim_data: Similarity data from SimGNN
    :param train_loader: Training data loader for caption generation
    :param val_loader: Validation data loader for caption generation
    :param args: Command line arguments
    :param tokenizer: BERT tokenizer
    :return: Trained model
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create optimizer with different learning rates for different components
    optimizer = Adam([
        {'params': model.sim_gnn.parameters(), 'lr': args.learning_rate},
        {'params': model.graph_gnn.parameters(), 'lr': args.caption_learning_rate},
        {'params': model.combine_fc1.parameters(), 'lr': args.caption_learning_rate},
        {'params': model.combine_fc2.parameters(), 'lr': args.caption_learning_rate},
        {'params': model.bert.parameters(), 'lr': args.caption_learning_rate / 10}  # Lower LR for BERT
    ], weight_decay=args.weight_decay)
    
    # Define metrics to track
    sim_metrics = {'mse': [], 'mae': []}
    caption_metrics = {
        'Bleu_1': [], 'Bleu_2': [], 'Bleu_3': [], 'Bleu_4': [], 
        'CIDEr': [], 'ROUGE_L': [], 'loss': []
    }
    
    # Setup for early stopping and model saving
    best_rouge = 0  # Use ROUGE as primary metric for caption generation
    best_sim_mse = float('inf')  # Use MSE as primary metric for similarity
    
    # Create a directory for metrics plots
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    plots_dir = os.path.join('./plots', timestamp)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Training plots will be saved to {plots_dir}")
    
    # For saving epoch-wise metrics
    metrics_log_path = args.save_path.replace('.pt', '_metrics.json')
    
    # Create batches for similarity data
    def create_batches(train_data, batch_size):
        random.shuffle(train_data)
        batches = []
        for graph in range(0, len(train_data), batch_size):
            batches.append(train_data[graph:graph+batch_size])
        return batches
    
    # Transfer similarity data to PyTorch tensors
    def transfer_to_torch(data):
        new_data = dict()
        new_data["features_1"] = torch.FloatTensor(data["features_1"]).to(device)
        new_data["features_2"] = torch.FloatTensor(data["features_2"]).to(device)
        new_data["edge_features_1"] = torch.FloatTensor(data["edge_features_1"]).to(device)
        new_data["edge_features_2"] = torch.FloatTensor(data["edge_features_2"]).to(device)
        new_data["target"] = torch.FloatTensor([data["target"]]).to(device)
        return new_data
    
    print(f"\nStarting joint training for {args.epochs} epochs")
    print(f"{'='*80}")
    
    for epoch in range(args.epochs):
        model.train()
        sim_loss_sum = 0
        caption_loss_sum = 0
        sim_batch_count = 0
        caption_batch_count = 0
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"{'-'*50}")
        
        # Create batches for similarity data
        sim_batches = create_batches(sim_data, args.batch_size)
        
        # First phase: Train similarity model
        print(f"Training similarity model (Phase 1)...")
        for index, batch in tqdm(enumerate(sim_batches), total=len(sim_batches), desc=f"SimGNN Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch_loss = 0
            
            for graph_pair in batch:
                data = process_pair(graph_pair, args.max_nodes, args.base_path)
                data = transfer_to_torch(data)
                
                # Forward pass through similarity model
                prediction = model.forward_similarity(data)
                
                # Calculate loss
                loss = (prediction - data["target"]) ** 2
                batch_loss += loss.item()
                
                # Backward pass
                loss.backward(retain_graph=True)
            
            # Average loss and optimize
            batch_loss = batch_loss / len(batch)
            sim_loss_sum += batch_loss * len(batch)
            sim_batch_count += len(batch)
            
            # Step optimizer
            optimizer.step()
        
        # Calculate average similarity loss
        sim_avg_loss = sim_loss_sum / sim_batch_count if sim_batch_count > 0 else float('inf')
        print(f"Similarity Training Loss: {sim_avg_loss:.6f}")
        
        # Second phase: Train caption model
        print(f"Training caption model (Phase 2)...")
        for batch_id, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Caption Training Epoch {epoch+1}"):
            node_features = batch['node_features']
            edge_features = batch['edge_features']
            captions = batch['caption']
            attention_mask = batch['attention_mask']
            
            # Forward pass through caption model
            optimizer.zero_grad()
            loss, _ = model.forward_caption(node_features, edge_features, captions, attention_mask)
            
            if loss is not None:
                # Backward pass
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                caption_loss_sum += loss.item()
                caption_batch_count += 1
        
        # Calculate average caption loss
        caption_avg_loss = caption_loss_sum / caption_batch_count if caption_batch_count > 0 else float('inf')
        print(f"Caption Training Loss: {caption_avg_loss:.6f}")
        
        # Store training losses
        sim_metrics['mse'].append(sim_avg_loss)
        caption_metrics['loss'].append(caption_avg_loss)
        
        # Evaluate similarity model
        print(f"Evaluating similarity model...")
        model.eval()
        val_sim_scores = []
        val_ground_truth = []
        
        # Use a subset of validation data for faster evaluation
        val_sim_data = random.sample(sim_data, min(500, len(sim_data)))
        
        with torch.no_grad():
            for graph_pair in tqdm(val_sim_data, desc="Similarity Evaluation"):
                data = process_pair(graph_pair, args.max_nodes, args.base_path)
                data = transfer_to_torch(data)
                prediction = model.forward_similarity(data)
                
                val_sim_scores.append(prediction.item())
                val_ground_truth.append(data["target"].item())
        
        # Calculate validation similarity metrics
        val_mse = np.mean([(s - g)**2 for s, g in zip(val_sim_scores, val_ground_truth)])
        val_mae = np.mean([abs(s - g) for s, g in zip(val_sim_scores, val_ground_truth)])
        
        print(f"Similarity Validation MSE: {val_mse:.6f}")
        print(f"Similarity Validation MAE: {val_mae:.6f}")
        
        # Store validation similarity metrics
        sim_metrics['mse'].append(val_mse)
        sim_metrics['mae'].append(val_mae)
        
        # Evaluate caption model
        print(f"Evaluating caption model...")
        caption_scores = evaluate(model, val_loader, tokenizer, beam_width=args.beam_width, max_length=args.max_length)
        
        # Store validation caption metrics
        for metric in caption_scores:
            if metric in caption_metrics:
                caption_metrics[metric].append(caption_scores[metric])
        
        # Early stopping and model saving logic
        improved_sim = val_mse < best_sim_mse
        improved_caption = caption_scores.get(args.primary_metric, 0) > best_rouge
        
        if improved_sim:
            best_sim_mse = val_mse
            print(f"New best similarity MSE: {best_sim_mse:.6f}")
        
        if improved_caption:
            best_rouge = caption_scores.get(args.primary_metric, 0)
            print(f"New best caption {args.primary_metric}: {best_rouge:.6f}")
        
        if improved_caption:
            # Save model state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sim_metrics': sim_metrics,
                'caption_metrics': caption_metrics,
                'best_sim_mse': best_sim_mse,
                'best_caption_score': best_rouge
            }
            torch.save(checkpoint, args.save_path)
            print(f"Model saved to {args.save_path}")
            
            # Save metrics
            all_metrics = {
                'similarity': sim_metrics,
                'caption': caption_metrics
            }
            with open(metrics_log_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
        
        # # Plot training metrics
        # if epoch > 0:  # Only plot when we have at least two points
        #     # Plot similarity metrics
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(range(1, epoch + 2), sim_metrics['mse'], marker='o', label='MSE')
        #     plt.plot(range(1, epoch + 2), sim_metrics['mae'], marker='o', label='MAE')
        #     plt.title('Similarity Metrics Over Epochs')
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Error')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(os.path.join(plots_dir, 'similarity_metrics.png'))
        #     plt.close()
            
        #     # Plot caption loss
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(range(1, epoch + 2), caption_metrics['loss'], marker='o')
        #     plt.title('Caption Loss Over Epochs')
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Loss')
        #     plt.grid(True)
        #     plt.savefig(os.path.join(plots_dir, 'caption_loss.png'))
        #     plt.close()
            
        #     # Plot caption evaluation metrics
        #     for metric in caption_metrics:
        #         if metric != 'loss' and len(caption_metrics[metric]) > 0:
        #             plt.figure(figsize=(10, 6))
        #             plt.plot(range(1, epoch + 2), caption_metrics[metric], marker='o')
        #             plt.title(f'{metric} Over Epochs')
        #             plt.xlabel('Epoch')
        #             plt.ylabel(metric)
        #             plt.grid(True)
        #             plt.savefig(os.path.join(plots_dir, f'{metric}_score.png'))
        #             plt.close()
    
    # Load best model at the end
    if os.path.exists(args.save_path):
        checkpoint = torch.load(args.save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', '?')}")
        print(f"Best Similarity MSE: {checkpoint.get('best_sim_mse', '?'):.6f}")
        print(f"Best Caption {args.primary_metric}: {checkpoint.get('best_caption_score', '?'):.6f}")
    
    return model


# Main function for the combined model
def main():
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train the Combined Graph Matching and Caption Generation Model")
    parser.add_argument('--template_file', type=str, default='/home/akshay/com_full/modified_captions/template.json')
    parser.add_argument('--train_file', type=str, default='/home/akshay/com_full/annotations_resnet/captions_train.json')
    parser.add_argument('--val_file', type=str, default='/home/akshay/com_full/annotations_resnet/captions_val.json')
    parser.add_argument('--save_path', type=str, default='/home/akshay/com_full/combined_model_2.pt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--caption_learning_rate', type=float, default=1e-5)
    parser.add_argument('--use_geometric', action='store_true', help="Use PyTorch Geometric GNN implementation")
    parser.add_argument('--beam_width', type=int, default=3, help="Beam width for beam search decoding")
    parser.add_argument('--max_length', type=int, default=50, help="Maximum caption length")
    parser.add_argument('--eval_only', action='store_true', help="Only evaluate a trained model, no training")
    parser.add_argument('--primary_metric', type=str, default='ROUGE_L', help="Primary metric for model selection")
    parser.add_argument('--joint_training', action='store_true', help="Train SimGNN and caption models jointly")
    # In the main function, add this parameter to the argument parser:
    parser.add_argument('--weight_decay', 
                    type=float, 
                    default=1e-5, 
                    help="Weight decay for Adam optimizer (default: 1e-5)")
    parser.add_argument('--max_nodes', 
                    type=int, 
                    default=6, 
                    help="Maximum number of nodes in graphs (default: 6)")
    parser.add_argument('--base_path', 
                    type=str, 
                    default='/home/akshay/com_full/instruments18_caption', 
                    help="Base path for graph files")
    combined_args = parser.parse_args()

    # Also load SimGNN arguments
    simgnn_args = parameter_parser()
    simgnn_args.base_path = '/home/akshay/com_full/instruments18_caption'  # Set the correct base path
    simgnn_args.template_file = combined_args.template_file
    simgnn_args.train_captions_file = combined_args.train_file
    simgnn_args.val_captions_file = combined_args.val_file
    
    # Create necessary directories
    os.makedirs(os.path.dirname(combined_args.save_path), exist_ok=True)
    
    # Display training settings
    print("\n===== Training Settings =====")
    print("SimGNN Parameters:")
    tab_printer(simgnn_args)
    print("\nCombined Model Parameters:")
    for arg in vars(combined_args):
        print(f"{arg}: {getattr(combined_args, arg)}")
    print("===========================\n")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load template, train, and validation data
    template_data = load_template_data(combined_args.template_file)
    train_caption_data = load_caption_data(combined_args.train_file)
    val_caption_data = load_caption_data(combined_args.val_file)
    
    # Generate similarity data using cosine similarity between template and training graphs
    similarity_data = generate_similarity_scores(
        template_data,
        train_caption_data,
        simgnn_args.base_path,
        simgnn_args.max_nodes
    )
    
    # Split similarity data for training
    train_sim_data, val_sim_data, test_sim_data = split_similarity_data(
        similarity_data,
        train_split=simgnn_args.training_split,
        val_split=simgnn_args.validation_split
    )
    
    # Initialize template dataset
    print(f"Initializing template dataset from {combined_args.template_file}...")
    template_dataset = TemplateGraphDataset(combined_args.template_file, simgnn_args, tokenizer)
    
    # Initialize joint model
    print("Initializing joint model...")
    joint_model = JointGraphCaptionModel(simgnn_args, template_dataset).to(device)
    
    # Load datasets for caption training
    print(f"Loading validation dataset from {combined_args.val_file}...")
    val_dataset = GraphCaptionDataset(combined_args.val_file, simgnn_args.base_path, simgnn_args.max_nodes)
    val_loader = DataLoader(val_dataset, batch_size=combined_args.batch_size)
    
    # Check if we should load a pre-trained model
    if combined_args.eval_only and os.path.exists(combined_args.save_path):
        print(f"Loading pre-trained model from {combined_args.save_path} for evaluation...")
        checkpoint = torch.load(combined_args.save_path, map_location=device)
        joint_model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'sim_metrics' in checkpoint and 'caption_metrics' in checkpoint:
            print("\nPrevious best metrics:")
            print(f"Similarity MSE: {checkpoint.get('best_sim_mse', 'N/A')}")
            print(f"Caption {combined_args.primary_metric}: {checkpoint.get('best_caption_score', 'N/A')}")
    
    # If eval_only, just evaluate and exit
    if combined_args.eval_only:
        print("\nRunning evaluation only mode...")
        
        # Evaluate similarity model
        print("Evaluating similarity model...")
        with torch.no_grad():
            val_sim_scores = []
            val_ground_truth = []
            
            for graph_pair in tqdm(test_sim_data, desc="Similarity Evaluation"):
                data = process_pair(graph_pair, simgnn_args.max_nodes, simgnn_args.base_path)
                data_tensor = {
                    "features_1": torch.FloatTensor(data["features_1"]).to(device),
                    "features_2": torch.FloatTensor(data["features_2"]).to(device),
                    "edge_features_1": torch.FloatTensor(data["edge_features_1"]).to(device),
                    "edge_features_2": torch.FloatTensor(data["edge_features_2"]).to(device),
                    "target": torch.FloatTensor([data["target"]]).to(device)
                }
                
                prediction = joint_model.forward_similarity(data_tensor)
                
                val_sim_scores.append(prediction.item())
                val_ground_truth.append(data_tensor["target"].item())
            
            # Calculate validation similarity metrics
            val_mse = np.mean([(s - g)**2 for s, g in zip(val_sim_scores, val_ground_truth)])
            val_mae = np.mean([abs(s - g) for s, g in zip(val_sim_scores, val_ground_truth)])
            
            print(f"Similarity Test MSE: {val_mse:.6f}")
            print(f"Similarity Test MAE: {val_mae:.6f}")
        
        # Evaluate caption model
        print("Evaluating caption model...")
        caption_scores = evaluate(
            joint_model,
            val_loader,
            tokenizer,
            beam_width=combined_args.beam_width,
            max_length=combined_args.max_length
        )
        
        print("\nEvaluation complete.")
        return
    
    # Otherwise, proceed with training
    print(f"Loading training dataset from {combined_args.train_file}...")
    train_dataset = GraphCaptionDataset(combined_args.train_file, simgnn_args.base_path, simgnn_args.max_nodes)
    train_loader = DataLoader(train_dataset, batch_size=combined_args.batch_size, shuffle=True)
    
    # Train the model
    print("\nStarting model training...")
    joint_model = train_joint_model(
        model=joint_model,
        sim_data=train_sim_data,
        train_loader=train_loader,
        val_loader=val_loader,
        args=combined_args,
        tokenizer=tokenizer
    )
    
    # Final evaluation
    print("\nFinal evaluation on test set:")
    
    # Evaluate similarity model
    print("Evaluating similarity model...")
    with torch.no_grad():
        test_sim_scores = []
        test_ground_truth = []
        
        for graph_pair in tqdm(test_sim_data, desc="Similarity Evaluation"):
            data = process_pair(graph_pair, simgnn_args.max_nodes, simgnn_args.base_path)
            data_tensor = {
                "features_1": torch.FloatTensor(data["features_1"]).to(device),
                "features_2": torch.FloatTensor(data["features_2"]).to(device),
                "edge_features_1": torch.FloatTensor(data["edge_features_1"]).to(device),
                "edge_features_2": torch.FloatTensor(data["edge_features_2"]).to(device),
                "target": torch.FloatTensor([data["target"]]).to(device)
            }
            
            prediction = joint_model.forward_similarity(data_tensor)
            
            test_sim_scores.append(prediction.item())
            test_ground_truth.append(data_tensor["target"].item())
        
        # Calculate test similarity metrics
        test_mse = np.mean([(s - g)**2 for s, g in zip(test_sim_scores, test_ground_truth)])
        test_mae = np.mean([abs(s - g) for s, g in zip(test_sim_scores, test_ground_truth)])
        
        print(f"Similarity Test MSE: {test_mse:.6f}")
        print(f"Similarity Test MAE: {test_mae:.6f}")
    
    # Evaluate caption model
    print("Evaluating caption model...")
    caption_scores = evaluate(
        joint_model,
        val_loader,
        tokenizer,
        beam_width=combined_args.beam_width,
        max_length=combined_args.max_length
    )
    
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
