import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def configure_for_hardware(config):
    """Configure settings based on available hardware with enhanced detection and performance tuning"""
    # Initialize variables to track hardware capabilities
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available()
    
    # Determine compute device with priorities: CUDA -> MPS -> CPU
    if has_cuda:
        device_type = 'cuda'
        device = torch.device('cuda')
        print(f"🚀 Using CUDA GPU acceleration")
    elif has_mps:
        device_type = 'mps'
        device = torch.device('mps')
        print(f"🚀 Using MPS (Apple Silicon) acceleration")
    else:
        device_type = 'cpu'
        device = torch.device('cpu')
        print(f"⚠️ Using CPU (no GPU acceleration available)")
    
    # Configure settings based on detected hardware
    if device_type == 'cuda':
        # Get detailed GPU information
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        
        # Get GPU memory information in GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        current_allocated = torch.cuda.memory_allocated(0) / 1e9
        current_reserved = torch.cuda.memory_reserved(0) / 1e9
        available_memory = total_memory - current_allocated
        
        print(f"GPU: {gpu_name} ({gpu_count} available)")
        print(f"CUDA Version: {cuda_version}")
        print(f"Memory: {current_allocated:.2f} GB allocated, {current_reserved:.2f} GB reserved")
        print(f"Total GPU Memory: {total_memory:.2f} GB, Available: {available_memory:.2f} GB")
        
        # Define GPU tiers based on card type and memory
        is_v100 = 'V100' in gpu_name
        is_a100 = 'A100' in gpu_name
        is_h100 = 'H100' in gpu_name
        is_t4 = 'T4' in gpu_name
        is_ampere = any(x in gpu_name for x in ['A10', 'A40', 'A100', '3090', '4090'])
        is_high_end = is_v100 or is_a100 or is_h100 or ('TITAN' in gpu_name) or ('RTX' in gpu_name and ('2080' in gpu_name or '3080' in gpu_name))
        
        # Configure for specific GPU types
        if is_h100:
            # H100 configuration - maximum performance
            print("🔥 Using optimized configuration for H100 GPU")
            config.batch_size = min(256, max(128, int(available_memory * 30)))  # Dynamic batch size
            config.hidden_dim = 1024
            config.num_heads = 16
            config.use_amp = True
            config.use_cuda_graphs = True
            config.num_workers = min(8, os.cpu_count() or 4)
            config.use_neural_ode = True  # H100 can handle this complexity
            
        elif is_a100:
            # A100 configuration - very high performance
            print("🔥 Using optimized configuration for A100 GPU")
            config.batch_size = min(192, max(96, int(available_memory * 25)))
            config.hidden_dim = 768
            config.num_heads = 12
            config.use_amp = True
            config.use_cuda_graphs = True
            config.num_workers = min(6, os.cpu_count() or 4)
            
        elif is_v100:
            # V100 configuration - high performance
            print("🔥 Using optimized configuration for V100 GPU")
            config.batch_size = min(128, max(64, int(available_memory * 20)))
            config.hidden_dim = 512
            config.num_heads = 8
            config.use_amp = True
            config.use_cuda_graphs = True
            config.num_workers = min(4, os.cpu_count() or 2)
            
        elif is_t4 or is_ampere:
            # T4 or Ampere (A10, A40, RTX 30 series) configuration - good performance
            print("🔥 Using optimized configuration for T4/Ampere GPU")
            config.batch_size = min(96, max(32, int(available_memory * 15)))
            config.hidden_dim = 384
            config.num_heads = 6
            config.use_amp = True
            config.use_cuda_graphs = True
            config.num_workers = min(4, os.cpu_count() or 2)
            
        elif is_high_end:
            # High-end GPU configuration
            print("🔥 Using optimized configuration for high-end GPU")
            config.batch_size = min(64, max(32, int(available_memory * 15)))
            config.hidden_dim = 256
            config.num_heads = 4
            config.use_amp = True
            config.use_cuda_graphs = False  # Not all high-end GPUs support CUDA graphs well
            config.num_workers = min(2, os.cpu_count() or 1)
            
        else:
            # Default GPU configuration - balanced for most NVIDIA GPUs
            print("⚡ Using standard GPU configuration")
            config.batch_size = min(32, max(16, int(available_memory * 10)))
            config.hidden_dim = 192
            config.num_heads = 3
            config.use_amp = True
            config.use_cuda_graphs = False
            config.num_workers = min(2, os.cpu_count() or 1)
            
    elif device_type == 'mps':
        # MPS (Apple Silicon) configuration - iPhone-style SoC
        print("🍎 Using optimized configuration for Apple Silicon")
        config.batch_size = 32  # M1/M2 unified memory can handle decent batch sizes
        config.hidden_dim = 128
        config.num_heads = 4
        config.use_amp = False  # MPS doesn't support mixed precision yet
        config.use_cuda_graphs = False
        config.num_workers = min(2, os.cpu_count() or 1)
        config.use_neural_ode = False  # Simplified model for MPS
        
    else:
        # CPU configuration - conservative settings for memory efficiency
        print("💻 Using optimized configuration for CPU training")
        
        # Smaller model configuration for CPU
        config.batch_size = 16
        config.hidden_dim = 64
        config.num_heads = 2
        config.use_amp = False
        config.use_cuda_graphs = False
        config.use_neural_ode = False
        config.num_workers = 0  # Safer default for CPU
        config.prefetch_factor = None
        
        # Reduce complexity for CPU training
        config.num_graph_layers = min(config.num_graph_layers, 2)
        config.num_temporal_layers = min(config.num_temporal_layers, 2)
        config.use_hyperbolic = False
    
    # Always set cuda_graph_batch_size for consistency
    config.cuda_graph_batch_size = config.batch_size
    
    # Print final configuration summary
    print("\n📊 Hardware-Optimized Configuration:")
    print(f"Batch Size: {config.batch_size}")
    print(f"Hidden Dimension: {config.hidden_dim}")
    print(f"Attention Heads: {config.num_heads}")
    print(f"Mixed Precision: {'Enabled' if config.use_amp else 'Disabled'}")
    print(f"CUDA Graphs: {'Enabled' if config.use_cuda_graphs else 'Disabled'}")
    print(f"Data Workers: {config.num_workers}")
    print(f"Graph Layers: {config.num_graph_layers}")
    print(f"Temporal Layers: {config.num_temporal_layers}")
    print(f"Hyperbolic Encoding: {'Enabled' if config.use_hyperbolic else 'Disabled'}")
    print(f"Neural ODE: {'Enabled' if config.use_neural_ode else 'Disabled'}")
    
    return device, config


def plot_training_curves(metrics, config, current_epoch=None):
    """Plot training curves with enhanced diagnostic analysis"""
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss curves
    axs[0].plot(metrics['train_losses'], label='Train Loss', color='blue', marker='o')
    axs[0].plot(metrics['val_losses'], label='Validation Loss', color='red', marker='s')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    # Add learning rate as secondary y-axis
    if metrics['learning_rates']:
        ax2 = axs[0].twinx()
        ax2.plot(metrics['learning_rates'], label='Learning Rate', color='green', linestyle='--')
        ax2.set_ylabel('Learning Rate', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_yscale('log')  # Log scale for learning rate
        
        # Show both legends
        lines1, labels1 = axs[0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot accuracy curves with variance bands if we have enough data
    axs[1].plot(metrics['train_accs'], label='Train Accuracy', color='blue', marker='o')
    axs[1].plot(metrics['val_accs'], label='Validation Accuracy', color='red', marker='s')
    
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_ylim(0, 1.05)  # Accuracy range 0-1 with small margin
    axs[1].legend()
    
    # Add early stopping indicator and best epoch
    if len(metrics['val_losses']) > 0:
        best_epoch = np.argmin(metrics['val_losses'])
        best_val_loss = metrics['val_losses'][best_epoch]
        axs[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
        axs[0].text(best_epoch + 0.1, best_val_loss, f'Best: {best_val_loss:.4f}', 
                    verticalalignment='center')
    
    # Add analysis: Check for overfitting by examining gap between train and val loss
    if len(metrics['train_losses']) > 2 and len(metrics['val_losses']) > 2:
        # Compute gaps for loss
        train_val_loss_gaps = [val - train for train, val in zip(metrics['train_losses'], metrics['val_losses'])]
        
        # Calculate mean gap in the last epochs (overfitting monitor)
        last_n_epochs = min(3, len(train_val_loss_gaps))
        mean_last_gaps = sum(train_val_loss_gaps[-last_n_epochs:]) / last_n_epochs
        
        # Add text analysis
        axs[0].text(0.02, 0.02, 
                   f"Mean Train-Val Gap (last {last_n_epochs} epochs): {mean_last_gaps:.2f}",
                   transform=axs[0].transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add overfitting/underfitting annotation
        plt.figtext(0.5, 0.94, 
                   f"Training-Validation Gap: {mean_last_gaps:.2f} / Latest gap: {train_val_loss_gaps[-1] if train_val_loss_gaps else 0}")
        
        if mean_last_gaps > 20:
            plt.text(0.5, 0.9, 'Potential Overfitting', ha='center', transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='red', alpha=0.2))
        elif mean_last_gaps < -20:  # Underfitting case
            plt.text(0.5, 0.9, 'Potential Underfitting', ha='center', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='blue', alpha=0.2))
        else:
            plt.text(0.5, 0.9, 'Good Fit', ha='center', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='green', alpha=0.2))
            
        plt.grid(True, alpha=0.3)
    
    # Add configuration summary as text
    plt.figtext(0.5, 0.01, 
               f"Model: {config.hidden_dim}d, {'Hyperbolic' if config.use_hyperbolic else 'Euclidean'}, "
               f"BS={config.batch_size}, LR={config.learning_rate:g}", 
               ha="center", fontsize=10)
    
    # Ensure proper spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure with epoch info if provided
    filename = 'training_curves.png' if current_epoch is None else f'training_curves_epoch_{current_epoch}.png'
    plt.savefig(os.path.join(config.output_dir, filename), dpi=120)
    plt.close()