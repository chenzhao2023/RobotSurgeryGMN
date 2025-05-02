"""Data processing utilities."""
import json
import math
import random
import numpy as np
from texttable import Texttable
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def compute_cosine_similarity(features1, features2):
    """
    Compute cosine similarity between two feature vectors.
    :param features1: First feature vector.
    :param features2: Second feature vector.
    :return: Cosine similarity score.
    """
    # If they are multi-dimensional, flatten them
    if len(features1.shape) > 1:
        features1 = features1.reshape(1, -1)
    if len(features2.shape) > 1:
        features2 = features2.reshape(1, -1)
        
    # Calculate cosine similarity
    similarity = cosine_similarity(features1, features2)[0][0]
    
    # Normalize to [0, 1] range
    similarity = (similarity + 1) / 2
    
    return similarity

def generate_similarity_scores(template_dataset, train_dataset, base_path="", max_nodes=6):
    """
    Generate similarity scores between template graphs and training graphs.
    :param template_dataset: List of template graphs.
    :param train_dataset: List of training graphs.
    :param base_path: Base path for graph files.
    :param max_nodes: Maximum number of nodes in graphs.
    :return: List of graph pairs with similarity scores.
    """
    similarity_data = []
    
    print("Computing similarity scores between template and training graphs...")
    for train_idx, train_item in enumerate(train_dataset):
        train_path = train_item["id_path"]
        train_features = load_graph_features(train_path, base_path, max_nodes)
        train_caption = train_item["caption"]
        
        for template_idx, template_item in enumerate(template_dataset):
            template_path = template_item["id_path"]
            template_features = load_graph_features(template_path, base_path, max_nodes)
            template_caption = template_item["caption"]
            
            # Compute similarity score based on graph features
            similarity_score = compute_cosine_similarity(train_features, template_features)
            
            # Create a pair entry
            pair_data = {
                "id_path_1": train_path,
                "id_path_2": template_path,
                "score": float(similarity_score),
                "train_caption": train_caption,
                "template_caption": template_caption
            }
            
            similarity_data.append(pair_data)
    
    print(f"Generated {len(similarity_data)} graph pairs with similarity scores.")
    return similarity_data

def load_similarity_data(path):
    """
    Read similarity data from JSON file.
    :param path: Path to the similarity JSON file.
    :return: List of graph pairs with similarity scores.
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} graph pairs with similarity scores.")
    return data

def split_similarity_data(data, train_split=0.6, val_split=0.2):
    """
    Split the data into training, validation, and testing sets.
    :param data: List of graph pairs with similarity scores.
    :param train_split: Ratio of training data (default: 0.6).
    :param val_split: Ratio of validation data (default: 0.2).
    :return: Training, validation, and testing data lists.
    """
    random.shuffle(data)
    train_size = int(len(data) * train_split)
    val_size = int(len(data) * val_split)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"Split data into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} testing pairs.")
    return train_data, val_data, test_data

def pad_features(features, max_nodes, feature_dim=512):
    """
    Pad node features to fixed size.
    :param features: Node features.
    :param max_nodes: Maximum number of nodes.
    :param feature_dim: Feature dimension.
    :return: Padded features.
    """
    padded = np.zeros((max_nodes, feature_dim))
    num_nodes = min(features.shape[0], max_nodes)
    padded[:num_nodes, :] = features[:num_nodes, :]
    return padded

def pad_edge_features(edge_features, max_nodes, feature_dim=512):
    """
    Pad edge features to fixed size.
    :param edge_features: Edge features.
    :param max_nodes: Maximum number of nodes.
    :param feature_dim: Feature dimension.
    :return: Padded edge features.
    """
    padded = np.zeros((max_nodes, max_nodes, feature_dim))
    
    # Handle different possible shapes of edge features
    if len(edge_features.shape) == 3:  # Already in shape (nodes, nodes, features)
        num_nodes = min(edge_features.shape[0], max_nodes)
        padded[:num_nodes, :num_nodes, :] = edge_features[:num_nodes, :num_nodes, :]
    elif len(edge_features.shape) == 2:  # Shape (edges, features)
        # For this case, we need to know how to map to adjacency tensor
        # This is a placeholder - actual implementation depends on your data format
        print("Warning: Edge features in (edges, features) format not properly handled.")
    
    return padded

def get_edge_path_from_node_path(node_path):
    """
    Convert node features path to edge features path.
    :param node_path: Path to node features file.
    :return: Path to edge features file.
    """
    return node_path.replace("node", "edge")

def load_graph_features(id_path, base_path="", max_nodes=6, feature_dim=512):
    """
    Load graph node features from file path.
    :param id_path: Path to the node features file.
    :param base_path: Base directory path to prepend.
    :param max_nodes: Maximum number of nodes.
    :param feature_dim: Feature dimension.
    :return: Padded node features.
    """
    # Combine base path and id_path
    full_path = os.path.join(base_path, id_path)
    
    try:
        features = np.load(full_path)
        return pad_features(features, max_nodes, feature_dim)
    except FileNotFoundError:
        print(f"Warning: File not found: {full_path}")
        return np.zeros((max_nodes, feature_dim))

def load_edge_features(id_path, base_path="", max_nodes=6, feature_dim=512):
    """
    Load graph edge features from file path.
    :param id_path: Path to the node features file (will be converted to edge path).
    :param base_path: Base directory path to prepend.
    :param max_nodes: Maximum number of nodes.
    :param feature_dim: Feature dimension.
    :return: Padded edge features.
    """
    # Convert node path to edge path
    edge_path = get_edge_path_from_node_path(id_path)
    
    # Combine base path and edge path
    full_path = os.path.join(base_path, edge_path)
    
    try:
        features = np.load(full_path)
        return pad_edge_features(features, max_nodes, feature_dim)
    except FileNotFoundError:
        print(f"Warning: Edge file not found: {full_path}")
        return np.zeros((max_nodes, max_nodes, feature_dim))

def process_pair(pair_data, max_nodes=6, base_path=""):
    """
    Process a pair of graphs from similarity data.
    :param pair_data: Dictionary containing graph pair information.
    :param max_nodes: Maximum number of nodes.
    :param base_path: Base directory path to prepend to id_path values.
    :return data: Dictionary with processed graph data.
    """
    # Load node features with base path
    features_1 = load_graph_features(pair_data["id_path_1"], base_path, max_nodes)
    features_2 = load_graph_features(pair_data["id_path_2"], base_path, max_nodes)
    
    # Load edge features with base path
    edge_features_1 = load_edge_features(pair_data["id_path_1"], base_path, max_nodes)
    edge_features_2 = load_edge_features(pair_data["id_path_2"], base_path, max_nodes)
    
    # Similarity score is already normalized between 0 and 1
    target = pair_data["score"]
    
    data = {
        "features_1": features_1,
        "features_2": features_2,
        "edge_features_1": edge_features_1,
        "edge_features_2": edge_features_2,
        "target": target,
        "id_path_1": pair_data["id_path_1"],
        "id_path_2": pair_data["id_path_2"],
        "train_caption": pair_data.get("train_caption", ""),
        "template_caption": pair_data.get("template_caption", "")
    }
    
    return data

def calculate_loss(prediction, target):
    """
    Calculate mean squared error between prediction and target.
    :param prediction: Predicted similarity score.
    :param target: Ground truth similarity score.
    :return: MSE loss.
    """
    return (prediction - target) ** 2

def load_template_data(template_file, base_path=""):
    """
    Load template graphs and captions data.
    :param template_file: Path to template graphs JSON file.
    :param base_path: Base path for graph files.
    :return: List of template graphs with captions.
    """
    with open(template_file, "r") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} template graphs with captions.")
    return data

def load_caption_data(caption_file, base_path=""):
    """
    Load caption data for graphs.
    :param caption_file: Path to caption JSON file.
    :param base_path: Base path for graph files.
    :return: List of graphs with captions.
    """
    with open(caption_file, "r") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} graphs with captions.")
    return data

def save_similarity_data(data, path):
    """
    Save similarity data to a JSON file.
    :param data: List of graph pairs with similarity scores.
    :param path: Path to save the JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved {len(data)} graph pairs with similarity scores to {path}.")
