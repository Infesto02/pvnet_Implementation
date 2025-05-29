import torch
from models.pvnet import PVNet
from datasets.linemod_dataset import LineMODDataset
from utils.voting import ransac_voting
from utils.pnp import solve_pnp

dataset = LineMODDataset('./bop_datasets/linemod')
model = PVNet(num_keypoints=8, num_classes=1).cuda()
model.load_state_dict(torch.load("model.pth"))
model.eval()

for image, vector_gt, mask in dataset:
    with torch.no_grad():
        image = image.unsqueeze(0).cuda()
        vectors, seg = model(image)
        vectors = vectors[0].cpu().numpy()
        mask = mask.numpy()

        keypoints_2d = ransac_voting(vectors, mask)
        keypoints_3d = [...]  # Load from object model
        camera_matrix = [...]  # Intrinsics

        R, t = solve_pnp(keypoints_2d, keypoints_3d, camera_matrix)
        print("Pose R:\n", R)
        print("Pose t:\n", t)
