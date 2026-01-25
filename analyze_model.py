import argparse
import json
import os
import torch
import copy
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.variants.UNeXtPredictor import UNeXtPredictor
from nnunetv2.inference.variants.MonaiPredictors import UNetPlusPlusPredictor, UNETRPredictor
from nnunetv2.inference.variants.LightMUNetPredictor import LightMUNetPredictor
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.mono.mono_layer import Mono2D, Mono2DV2
import torchprofile
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze model parameters and FLOPs")
    parser.add_argument("--train_dataset_id", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--plans", type=str, required=True)
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--cfg", type=str, default="2d")
    parser.add_argument("--chk", type=str, default="checkpoint_final.pth")
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


def get_predictor_class(trainer):
    """Get the appropriate predictor class based on trainer name"""
    if "UNeXtTrainer" in trainer:
        return UNeXtPredictor
    elif trainer == "UNetPlusPlusTrainer":
        return UNetPlusPlusPredictor
    elif trainer == "UNETRTrainer":
        return UNETRPredictor
    elif trainer == "nnUNetTrainerLightMUNet":
        return LightMUNetPredictor
    else:
        return nnUNetPredictor


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


def analyze_model(model_dir, trainer, fold, chk, gpu, save_metrics=True):
    """Analyze model parameters and FLOPs"""
    
    # Load plans to get input size
    plans_path = join(model_dir, "plans.json")
    with open(plans_path, 'r') as f:
        plans = json.load(f)
    # Get patch size from plans (assuming 2d configuration)
    input_size = plans["configurations"]["2d"]["patch_size"]
    print(f"Using input size from plans: {input_size[0]}x{input_size[1]}")

    with open(join(model_dir, "dataset.json"), "r") as f:
        dataset = json.load(f)
    in_channels = len(dataset["channel_names"])
    print(f"Using input channels from dataset: {in_channels}")
    
    # Set device
    if gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    # Get predictor class
    predictor_class = get_predictor_class(trainer)
    
    # Initialize predictor
    predictor = predictor_class(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    
    # Initialize from trained model
    # use_folds = (0 if fold == "all" else int(fold),)
    use_folds = fold
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=use_folds,
        checkpoint_name=chk,
    )
    
    # Get the network
    model = copy.deepcopy(predictor.network)
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
    print(f"Trainer: {trainer}")
    print(f"Fold: {fold}")
    print(f"Checkpoint: {chk}")
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
        "trainer": trainer,
        "fold": fold,
        "checkpoint": chk,
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
    
    # Get dataset name
    train_dataset_name = get_dataset_name(nnUNet_raw, args.train_dataset_id)
    
    # Construct model directory
    model_dir = join(nnUNet_results, train_dataset_name, f'{args.trainer}__{args.plans}__{args.cfg}')
    
    print(f"Analyzing model...")
    print(f"Dataset: {train_dataset_name}")
    print(f"Model directory: {model_dir}")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist!")
        return
    
    # Analyze the model
    metrics = analyze_model(
        model_dir=model_dir,
        trainer=args.trainer,
        fold=args.fold,
        chk=args.chk,
        gpu=args.gpu,
        save_metrics=args.save_metrics
    )
    
    return metrics


if __name__ == "__main__":
    main()
