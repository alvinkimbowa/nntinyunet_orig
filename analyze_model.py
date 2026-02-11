import argparse
import json
import os
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.mono.mono_layer import Mono2D, Mono2DV2
import torchprofile
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze model parameters and FLOPs")
    parser.add_argument("--train_dataset_id", type=int, required=True)
    parser.add_argument("--plans", type=str, required=True)
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--cfg", type=str, default="2d")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_metrics", type=str2bool, default=True,
                       help="Save metrics to JSON file")
    args = parser.parse_args()
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_dataset_name(nnUNet_raw, dataset_id):
    dataset_id = str(dataset_id).zfill(3)
    dataset_name = [dataset_name for dataset_name in os.listdir(nnUNet_raw) 
                   if dataset_name.startswith(f"Dataset{dataset_id}") and 
                   os.path.isdir(join(nnUNet_raw, dataset_name))]
    assert len(dataset_name) == 1, f"Found {len(dataset_name)} datasets with id {dataset_id}, expected 1"
    return dataset_name[0]


def get_mono2d_macs(n, x):
    """
    Estimate the total MACs used by Mono2D_v2 on input x (shape: [C, H, W])

    Approximate breakdown:
        Rough MACs estimate:
        Total MACs ≈ [ (2.5 * C) + (5 * C * n) ] * H * W * log2(H*W)
                     + 18 * C * n * H * W
        Where:
            B = batch size,
            C = in_channels,
            n = nscale,
            H, W = spatial dims.
    """
    _, C, H, W = x.shape
    macs = (2.5 * C + 5 * C * n) * H * W * np.log2(H * W) + 18 * C * n * H * W
    return macs.item()


def load_nnunet_model(train_dataset_id, plans, trainer, cfg, fold, device):
    dataset_name = get_dataset_name(nnUNet_raw, train_dataset_id)
    nnunet_trainer = get_trainer_from_args(dataset_name, cfg, fold, trainer, plans, device=device)
    nnunet_trainer.initialize()
    model = nnunet_trainer.network.to(device)
    in_channels = nnunet_trainer.num_input_channels
    patch_size = nnunet_trainer.configuration_manager.patch_size
    return model, dataset_name, in_channels, patch_size


def analyze_model(model, dataset_name, input_size, in_channels, model_dir, gpu, save_metrics=True):
    """Analyze model parameters and FLOPs"""
    print(f"Using input size from plans: {input_size[0]}x{input_size[1]}")
    print(f"Using input channels from dataset: {in_channels}")

    if gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    
    # Create input tensor
    input_tensor = torch.randn(1, in_channels, input_size[0], input_size[1]).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compute FLOPs using flopth
    macs = torchprofile.profile_macs(model, input_tensor)
    input_tensor = input_tensor.to(device)
    # print("modules: ", list(model.modules()))
    for layer in model.modules():
        if isinstance(layer, Mono2D) or isinstance(layer, Mono2DV2):
            print(f"Layer: {layer}")
            nscale = layer.nscale
            print(f"MACs: {get_mono2d_macs(nscale, input_tensor)}")
            print(f"Nscale: {nscale}")
            macs += get_mono2d_macs(nscale, input_tensor)

    # Print results
    print(f"\n{'='*60}")
    print(f"MODEL ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Model directory: {model_dir}")
    print(f"Input size: {input_size[0]}x{input_size[1]}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    print(f"PARAMETERS:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e3:.2f}K)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e3:.2f}K)")
    print(f"  torchprofile parameters: {total_params:,} ({total_params/1e3:.2f}K)")
    print(f"{'='*60}")
    print(f"MACS:")
    print(f"  torchprofile MACS: {macs:,} ({macs/1e9:.2f}G)")
    print(f"{'='*60}")
    
    # Prepare metrics dictionary
    metrics = {
        "model_dir": model_dir,
        "dataset_name": dataset_name,
        "input_size": input_size,
        "device": str(device),
        "parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "total_thousands": round(total_params/1e3, 2),
            "trainable_thousands": round(trainable_params/1e3, 2)
        },
        "macs": {'total': int(macs), 'giga': round(macs/1e9, 2)},
        "flops": {
            "total": int(macs * 2),
            "giga": round((macs * 2)/1e9, 2)
            }
    }
       
    # Save metrics to JSON
    if save_metrics:
        metrics_path = join(model_dir, "model_analysis.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
    
    return metrics


def main():
    args = parse_args()
    
    device = torch.device('cpu' if args.gpu < 0 else 'cuda')
    model, dataset_name, in_channels, patch_size = load_nnunet_model(
        args.train_dataset_id,
        args.plans,
        args.trainer,
        args.cfg,
        args.fold,
        device,
    )
    model_dir = join(nnUNet_results, dataset_name, f'{args.trainer}__{args.plans}__{args.cfg}')
    if not os.path.exists(model_dir) and args.save_metrics:
        os.makedirs(model_dir, exist_ok=True)

    print(f"Analyzing model...")
    print(f"Dataset: {dataset_name}")
    print(f"Model directory: {model_dir}")

    metrics = analyze_model(
        model=model,
        dataset_name=dataset_name,
        input_size=patch_size,
        in_channels=in_channels,
        model_dir=model_dir,
        gpu=args.gpu,
        save_metrics=args.save_metrics,
    )
    
    return metrics


if __name__ == "__main__":
    main()
