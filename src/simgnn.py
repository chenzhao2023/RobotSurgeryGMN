"""SimGNN class and trainer."""

import torch
import random
import numpy as np
import os
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from src.layers import AttentionModule, TensorNetworkModule
from src.utils import process_pair, calculate_loss, load_similarity_data, split_similarity_data
from src.utils import load_template_data, load_caption_data, generate_similarity_scores, save_similarity_data

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        # Input feature dimension from your data
        self.input_dim = 512  # Your node feature dimension
        
        # GCN layers
        self.convolution_1 = GCNConv(self.input_dim, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        
        # Edge feature processing
        self.edge_feature_transform = torch.nn.Linear(self.input_dim, self.args.filters_3)
        
        # Feature combination layer
        self.feature_combine = torch.nn.Linear(self.args.filters_3 * 2, self.args.filters_3)
        
        # Attention and tensor network modules
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        
        # Final layers
        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons, 
                                                    self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Node features.
        :return features: Node embeddings.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, 
                                            p=self.args.dropout,
                                            training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                            p=self.args.dropout,
                                            training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def process_edge_features(self, edge_features):
        """
        Process edge features.
        :param edge_features: Edge features tensor [nodes x nodes x features].
        :return: Processed edge features.
        """
        batch_size, num_nodes, num_nodes, feat_dim = edge_features.size()
        
        # Reshape to process all edges at once
        edge_features_reshaped = edge_features.view(-1, feat_dim)
        
        # Transform edge features
        transformed_edges = self.edge_feature_transform(edge_features_reshaped)
        transformed_edges = torch.nn.functional.relu(transformed_edges)
        
        # Reshape back
        transformed_edges = transformed_edges.view(batch_size, num_nodes, num_nodes, -1)
        
        # Average edge features for each node
        edge_embeddings = transformed_edges.mean(dim=2)
        
        return edge_embeddings

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Dict containing features_1, features_2, edge_features_1, edge_features_2.
        :return score: Similarity score.
        """
        # Check if we have batched data or not
        if len(data["features_1"].shape) == 2:  # No batch dimension
            data["features_1"] = data["features_1"].unsqueeze(0)
            data["features_2"] = data["features_2"].unsqueeze(0)
            data["edge_features_1"] = data["edge_features_1"].unsqueeze(0)
            data["edge_features_2"] = data["edge_features_2"].unsqueeze(0)
        
        batch_size = data["features_1"].size(0)
        
        # Create edge indices for fully connected graphs
        edge_index_1 = self.get_edge_index(data["features_1"].size(1)).to(data["features_1"].device)
        edge_index_2 = self.get_edge_index(data["features_2"].size(1)).to(data["features_2"].device)
        
        # Process node features for each graph in batch
        all_abstract_features_1 = []
        all_abstract_features_2 = []
        
        for b in range(batch_size):
            # Process node features through GCN layers
            node_features_1 = data["features_1"][b]
            node_features_2 = data["features_2"][b]
            
            abstract_features_1 = self.convolutional_pass(edge_index_1, node_features_1)
            abstract_features_2 = self.convolutional_pass(edge_index_2, node_features_2)
            
            all_abstract_features_1.append(abstract_features_1)
            all_abstract_features_2.append(abstract_features_2)
        
        # Stack batched features
        abstract_features_1 = torch.stack(all_abstract_features_1)
        abstract_features_2 = torch.stack(all_abstract_features_2)
        
        # Process edge features
        edge_embeddings_1 = self.process_edge_features(data["edge_features_1"])
        edge_embeddings_2 = self.process_edge_features(data["edge_features_2"])
        
        # Combine node and edge features
        combined_features_1 = []
        combined_features_2 = []
        
        for b in range(batch_size):
            # Combine node and edge embeddings
            combined_1 = torch.cat([abstract_features_1[b], edge_embeddings_1[b]], dim=1)
            combined_2 = torch.cat([abstract_features_2[b], edge_embeddings_2[b]], dim=1)
            
            combined_features_1.append(self.feature_combine(combined_1))
            combined_features_2.append(self.feature_combine(combined_2))
        
        # Generate graph level representations for batch
        all_pooled_features_1 = []
        all_pooled_features_2 = []
        
        for b in range(batch_size):
            pooled_1 = self.attention(combined_features_1[b])
            pooled_2 = self.attention(combined_features_2[b])
            
            all_pooled_features_1.append(pooled_1)
            all_pooled_features_2.append(pooled_2)
        
        # Process each pair in batch
        all_scores = []
        
        for b in range(batch_size):
            # Calculate similarity scores
            scores = self.tensor_network(all_pooled_features_1[b], all_pooled_features_2[b])
            scores = torch.t(scores)
            
            # Final prediction
            scores = torch.nn.functional.relu(self.fully_connected_first(scores))
            score = torch.sigmoid(self.scoring_layer(scores))
            
            all_scores.append(score)
        
        # Stack scores and return
        if batch_size > 1:
            return torch.cat(all_scores)
        else:
            return all_scores[0]

    def get_edge_index(self, num_nodes):
        """
        Create fully connected edge index.
        :param num_nodes: Number of nodes.
        :return edge_index: Edge indices.
        """
        row = torch.arange(num_nodes).repeat(num_nodes)
        col = torch.arange(num_nodes).repeat_interleave(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def get_graph_embedding(self, node_features, edge_features):
        """
        Get a graph embedding for a single graph.
        :param node_features: Node features.
        :param edge_features: Edge features.
        :return: Graph embedding.
        """
        # Add batch dimension if needed
        if len(node_features.shape) == 2:
            node_features = node_features.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
        
        # Get edge index
        edge_index = self.get_edge_index(node_features.size(1)).to(node_features.device)
        
        # Process node features
        abstract_features = self.convolutional_pass(edge_index, node_features[0])
        
        # Process edge features
        edge_embeddings = self.process_edge_features(edge_features)[0]
        
        # Combine features
        combined_features = torch.cat([abstract_features, edge_embeddings], dim=1)
        combined_features = self.feature_combine(combined_features)
        
        # Generate graph embedding
        graph_embedding = self.attention(combined_features)
        
        return graph_embedding


class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load and process data
        if args.similarities_file and os.path.exists(args.similarities_file):
            # Use existing similarity data if available
            self.setup_data_from_file()
        else:
            # Generate similarity data if not available
            self.setup_data_from_scratch()
            
        self.setup_model()

    def setup_data_from_file(self):
        """
        Load data from existing similarity file.
        """
        self.data = load_similarity_data(self.args.similarities_file)
        self.train_data, self.val_data, self.test_data = split_similarity_data(
            self.data, 
            train_split=self.args.training_split,
            val_split=self.args.validation_split
        )
        
        # Also load template and caption data for reference
        self.template_data = load_template_data(self.args.template_file, self.args.base_path)
        self.train_caption_data = load_caption_data(self.args.train_captions_file, self.args.base_path)
        self.val_caption_data = load_caption_data(self.args.val_captions_file, self.args.base_path)

    def setup_data_from_scratch(self):
        """
        Generate similarity data from template and training datasets.
        """
        # Load template and caption data
        self.template_data = load_template_data(self.args.template_file, self.args.base_path)
        self.train_caption_data = load_caption_data(self.args.train_captions_file, self.args.base_path)
        self.val_caption_data = load_caption_data(self.args.val_captions_file, self.args.base_path)
        
        # Generate similarity scores
        self.data = generate_similarity_scores(
            self.template_data, 
            self.train_caption_data, 
            self.args.base_path,
            self.args.max_nodes
        )
        
        # Save similarity data if a path is provided
        if self.args.similarities_file:
            save_similarity_data(self.data, self.args.similarities_file)
        
        # Split data
        self.train_data, self.val_data, self.test_data = split_similarity_data(
            self.data, 
            train_split=self.args.training_split,
            val_split=self.args.validation_split
        )

    def setup_model(self):
        """
        Creating a SimGNN model.
        """
        self.model = SimGNN(self.args).to(self.device)

    def create_batches(self):
        """
        Creating batches from the training data.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.train_data)
        batches = []
        for graph in range(0, len(self.train_data), self.args.batch_size):
            batches.append(self.train_data[graph:graph+self.args.batch_size])
        return batches

    def transfer_to_torch(self, data):
        """
        Transfer the data to PyTorch tensors.
        :param data: Data dictionary.
        :return new_data: Dictionary of PyTorch tensors.
        """
        new_data = dict()
        
        # Transfer features to torch and normalize
        new_data["features_1"] = torch.FloatTensor(data["features_1"]).to(self.device)
        new_data["features_2"] = torch.FloatTensor(data["features_2"]).to(self.device)
        new_data["edge_features_1"] = torch.FloatTensor(data["edge_features_1"]).to(self.device)
        new_data["edge_features_2"] = torch.FloatTensor(data["edge_features_2"]).to(self.device)
        
        # Transfer target similarity score
        new_data["target"] = torch.FloatTensor([data["target"]]).to(self.device)
        
        return new_data

    def process_batch(self, batch):
        """
        Process a batch of graph pairs.
        :param batch: Batch of graph pairs.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        for graph_pair in batch:
            data = process_pair(graph_pair, self.args.max_nodes, self.args.base_path)
            data = self.transfer_to_torch(data)
            prediction = self.model(data)
            losses = losses + calculate_loss(prediction, data["target"])
        
        losses = losses / len(batch)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        """
        Train the model.
        """
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr=self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)

        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 5  # Stop if validation doesn't improve for 5 epochs
        
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            
            train_loss = self.loss_sum/main_index
            print(f"\nEpoch {epoch+1}/{self.args.epochs} - Training Loss: {train_loss:.6f}")
            
            # Switch to evaluation mode for validation
            self.model.eval()
            
            # Calculate metrics on validation data
            val_scores = []
            val_ground_truth = []
            
            for graph_pair in self.val_data:
                data = process_pair(graph_pair, self.args.max_nodes, self.args.base_path)
                data = self.transfer_to_torch(data)
                with torch.no_grad():
                    prediction = self.model(data)
                
                score = prediction.item()
                val_scores.append(score)
                val_ground_truth.append(data["target"].item())
            
            # Calculate validation metrics
            val_mse = np.mean([(s - g)**2 for s, g in zip(val_scores, val_ground_truth)])
            val_mae = np.mean([abs(s - g) for s, g in zip(val_scores, val_ground_truth)])
            
            print(f"Validation MSE: {val_mse:.6f}")
            print(f"Validation MAE: {val_mae:.6f}")
            
            # Early stopping logic
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
                print("New best validation loss achieved. Saving model state.")
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break
            
            # Switch back to training mode
            self.model.train()
        
        # If training completed without early stopping, ensure we use the best model
        if patience_counter < early_stopping_patience and 'best_model_state' in locals():
            print("Training completed. Loading best model state.")
            self.model.load_state_dict(best_model_state)

    def score(self):
        """
        Score on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        scores = []
        ground_truth = []
        
        for graph_pair in tqdm(self.test_data):
            data = process_pair(graph_pair, self.args.max_nodes, self.args.base_path)
            data = self.transfer_to_torch(data)
            prediction = self.model(data)
            
            score = prediction.item()
            scores.append(score)
            ground_truth.append(data["target"].item())
        
        mse = np.mean([(s - g)**2 for s, g in zip(scores, ground_truth)])
        mae = np.mean([abs(s - g) for s, g in zip(scores, ground_truth)])
        
        print(f"Test MSE: {mse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        
        return mse, mae

    def save(self):
        """
        Save the model.
        """
        torch.save(self.model.state_dict(), self.args.save_path)
        print(f"\nModel saved to: {self.args.save_path}")

    def load(self):
        """
        Load a saved model.
        """
        self.model.load_state_dict(torch.load(self.args.load_path))
        print(f"\nModel loaded from: {self.args.load_path}")
