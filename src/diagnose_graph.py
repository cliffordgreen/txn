import sys
import os
import pandas as pd
import torch

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.transaction_graph import TransactionGraphBuilder
from src.train_hyper_temporal_model import AdvancedTransactionSequenceBuilder

def main():
    """
    Diagnose issues with the transaction graph and sequences.
    """
    # Load transaction data from CSV
    csv_path = 'data/synthetic_transactions.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    transactions_df = pd.read_csv(csv_path)
    print(f"Loaded {len(transactions_df)} transactions")
    
    # Initialize graph builder
    graph_builder = TransactionGraphBuilder()
    
    # Build graph from transaction data
    graph = graph_builder.build_graph(transactions_df)
    
    # Print graph information
    print("\nGraph Information:")
    print(f"Node types: {graph.node_types}")
    for node_type in graph.node_types:
        print(f"- {node_type}: {graph[node_type].x.size()}")
    
    print("\nEdge types:")
    for edge_type in graph.edge_types:
        print(f"- {edge_type}: {graph[edge_type].edge_index.size()}")
    
    # Initialize sequence builder
    sequence_builder = AdvancedTransactionSequenceBuilder(max_seq_length=20)
    
    # Build sequences
    print("\nBuilding sequences...")
    sequences, seq_lengths, seq_timestamps, user_ids, time_features = sequence_builder.build_sequences(transactions_df)
    
    # Print sequence information
    print("\nSequence Information:")
    print(f"Sequences shape: {sequences.size()}")
    print(f"Sequence lengths shape: {seq_lengths.size()}")
    print(f"Timestamps shape: {seq_timestamps.size()}")
    print(f"Time features shape: {time_features.size()}")
    
    print("\nDiagnostic complete.")
    
if __name__ == "__main__":
    main()