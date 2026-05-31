"""
Extract and save point_cloud from a training checkpoint without running any training.
Usage: python save_from_checkpoint.py --checkpoint <path>.pth --model_path <output_dir>
"""
import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from scene.gaussian_model import GaussianModel
from arguments import ModelParams, OptimizationParams, PipelineParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args()

    dataset = lp.extract(args)
    opt = op.extract(args)

    print(f"Loading checkpoint: {args.checkpoint}")
    (model_params, iteration) = torch.load(args.checkpoint, weights_only=False)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.restore(model_params, opt)

    point_cloud_path = os.path.join(dataset.model_path, f"point_cloud/iteration_{iteration}")
    os.makedirs(point_cloud_path, exist_ok=True)

    print(f"Saving point cloud to: {point_cloud_path}")
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    gaussians.save_texture(point_cloud_path)

    print(f"Done. Saved {gaussians._xyz.shape[0]} Gaussians at iteration {iteration} to {point_cloud_path}.")


if __name__ == "__main__":
    main()
