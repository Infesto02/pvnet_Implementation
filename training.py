import torch
from torch.utils.data import DataLoader
from models.pvnet import PVNet
from datasets.linemod_dataset import LineMODDataset
import torch.nn.functional as F

dataset = LineMODDataset('./bop_datasets/linemod')
loader = DataLoader(dataset, batch_size=4, shuffle=True)
model = PVNet(num_keypoints=8, num_classes=1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    model.train()
    for image, vector_gt, mask in loader:
        image, vector_gt, mask = image.cuda(), vector_gt.cuda(), mask.cuda()
        vector_pred, mask_pred = model(image)
        loss_vec = F.smooth_l1_loss(vector_pred, vector_gt)
        loss_mask = F.cross_entropy(mask_pred, mask)
        loss = loss_vec + 0.5 * loss_mask

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
