import math, torch, sys
sys.path.insert(0, ".")
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from argparse import ArgumentParser

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=10000, type=int)
args = get_combined_args(parser)
args.iteration = 10000

dataset = model.extract(args)
pipe = pipeline.extract(args)
bg = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float32, device="cuda")

# Test 1: preproc=False (train-style loading)
print("=== preproc=False (train-style) ===")
gaussians = GaussianModel(dataset.sh_degree, texture_preproc=False)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
psnrs = []
with torch.no_grad():
    for cam in scene.getTestCameras()[:8]:
        pkg = render(cam, gaussians, pipe, bg)
        img = pkg["render"].clamp(0,1)
        gt = cam.original_image.cuda().clamp(0,1)
        gt = gt[:, :img.shape[1], :img.shape[2]]
        mse = ((img - gt)**2).mean().item()
        p = 10 * math.log10(1.0 / (mse + 1e-10))
        psnrs.append(p)
print(f"Mean PSNR: {sum(psnrs)/len(psnrs):.2f} dB")

# Test 2: preproc=True (render.py-style loading)
print("=== preproc=True (render.py-style) ===")
gaussians2 = GaussianModel(dataset.sh_degree, texture_preproc=True)
scene2 = Scene(dataset, gaussians2, load_iteration=args.iteration, shuffle=False)
psnrs2 = []
with torch.no_grad():
    for cam in scene2.getTestCameras()[:8]:
        pkg = render(cam, gaussians2, pipe, bg)
        img = pkg["render"].clamp(0,1)
        gt = cam.original_image.cuda().clamp(0,1)
        gt = gt[:, :img.shape[1], :img.shape[2]]
        mse = ((img - gt)**2).mean().item()
        p = 10 * math.log10(1.0 / (mse + 1e-10))
        psnrs2.append(p)
print(f"Mean PSNR: {sum(psnrs2)/len(psnrs2):.2f} dB")
