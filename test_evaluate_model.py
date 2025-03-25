#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to test the evaluation of a trained streamlined graph model
on new transaction data.
"""

import os
import sys
import argparse
import subprocess
import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test evaluation of streamlined graph model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, 
                       default='./models/enhanced_model_output/best_model.pt',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                       default='./data/parquet_files',
                       help='Directory containing transaction data parquet files')
    parser.add_argument('--results_dir', type=str, 
                       default=None,
                       help='Directory to save evaluation results (defaults to timestamped directory)')
    parser.add_argument('--extract_features', action='store_true',
                       help='Extract embeddings for further analysis')
    parser.add_argument('--cpu_only', action='store_true',
                       help='Force CPU usage even if GPU is available')
    
    return parser.parse_args()

def main():
    """Main function to run the evaluation script"""
    print("Starting evaluation test script...")
    
    # Parse arguments
    args = parse_args()
    
    # Create timestamped results directory if not specified
    if args.results_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = f"./evaluation_results_{timestamp}"
    
    # Verify paths exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return 1
    
    # Build command
    cmd = [
        "python",
        "./src/evaluate_streamlined_graph_model.py",
        f"--model_path={args.model_path}",
        f"--data_dir={args.data_dir}",
        f"--results_dir={args.results_dir}",
        "--verbose",
    ]
    
    # Add optional arguments
    if args.extract_features:
        cmd.append("--extract_features")
    
    if args.cpu_only:
        cmd.append("--cpu_only")
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the evaluation script
    try:
        process = subprocess.run(cmd, check=True)
        print(f"Evaluation completed with return code: {process.returncode}")
        
        # Print summary of results
        if os.path.exists(args.results_dir):
            print(f"\nEvaluation results saved to: {args.results_dir}")
            report_path = os.path.join(args.results_dir, 'evaluation_report.json')
            if os.path.exists(report_path):
                print(f"See {report_path} for detailed metrics")
            
            predictions_path = os.path.join(args.results_dir, 'predictions.csv')
            if os.path.exists(predictions_path):
                print(f"Model predictions saved to: {predictions_path}")
        
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation script: {e}")
        return e.returncode
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())