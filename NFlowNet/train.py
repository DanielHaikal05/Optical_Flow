import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import convert_image_dtype
import numpy as np
from pathlib import Path
import json
import random

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class TransposeResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))
    

class Conv_Encoder_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=7, stride=2, padding=3), nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2), nn.ReLU(True))

        self.enc2 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), nn.ReLU(True))

        self.enc3 = nn.Sequential(
            ResidualBlock(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), nn.ReLU(True))

        self.dec3 = nn.Sequential(
            TransposeResidualBlock(256),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True))     

        self.dec2 = nn.Sequential(
            TransposeResidualBlock(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True))   

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))   

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        return x

    
    def project_flow_to_normal_scalar(self, flow_prediction, normal_flow_direction):
        if flow_prediction.ndim != 4 or flow_prediction.shape[1] != 2:
            raise ValueError(
                f"Expected flow_prediction with shape [B, 2, H, W], "
                f"got {tuple(flow_prediction.shape)}")

        if normal_flow_direction.shape[0] == 1 and flow_prediction.shape[0] > 1:
            normal_flow_direction = normal_flow_direction.expand(flow_prediction.shape[0], -1, -1, -1)

        scalar_prediction = (flow_prediction * normal_flow_direction).sum(dim=1)
        return scalar_prediction

    
    def fit(self, optimizer, path_tuple):
        super().train(True)

        loss = 0.0
        optimizer.zero_grad()
        cumulative_loss = 0.0
        batch_px_nb = 0; epoch_px_nb = 0
        for i in range(BATCHES_PER_EPOCH):
            image_paths = path_tuple[i][0]
            gt_paths = path_tuple[i][1]
            mask_paths = path_tuple[i][2]
            gradient_paths = path_tuple[i][3]
            dataset = path_tuple[i][4]
            batch_size = len(gt_paths)

            batch = torch.zeros(batch_size, 6, 480, 640).to(device=self.device)
            batch_gt_scalars = torch.zeros(batch_size, 480, 640).to(device=self.device)
            batch_valid_masks = torch.zeros(batch_size, 480, 640).to(device=self.device)
            batch_gradient_dir = torch.zeros(batch_size, 2, 480, 640).to(device=self.device)

            for j in range(batch_size):
                image1 = read_image(image_paths[j],mode=ImageReadMode.RGB)
                image2 = read_image(image_paths[j+1],mode=ImageReadMode.RGB)

                image1 = convert_image_dtype(image1, torch.float32)
                image2 = convert_image_dtype(image2, torch.float32)
                batch[j,:3,:,:] = image1.to(device=self.device)
                batch[j,3:,:,:] = image2.to(device=self.device)

                gt_scalars = torch.from_numpy(np.load(gt_paths[j])).to(device=self.device, dtype=torch.float32)
                valid_mask = read_image(str(mask_paths[j]), mode=ImageReadMode.GRAY)[0].to(device=self.device)
                valid_mask = (valid_mask > 0).to(torch.float32)
                gt_gradients = torch.from_numpy(np.load(gradient_paths[j])).to(device=self.device, dtype=torch.float32)
                gt_gradients = gt_gradients.permute(2,0,1)

                batch_gt_scalars[j,:,:] = gt_scalars
                batch_valid_masks[j,:,:] = valid_mask
                batch_gradient_dir[j,:,:,:] = gt_gradients

            batch_output = self.forward(batch)
            normal_flow_predictions = self.project_flow_to_normal_scalar(batch_output, batch_gradient_dir)

            squared_error = (normal_flow_predictions-batch_gt_scalars)**2
            loss = (squared_error * batch_valid_masks).sum() / ZERO_MSE[dataset]
            batch_px_nb = batch_valid_masks.sum()

            (loss / max(batch_px_nb,1)).backward()
            optimizer.step()
            loss = loss.item() * 100
            print(f"        Completed: {i+1}/{BATCHES_PER_EPOCH},  Loss: {loss/max(batch_px_nb.item(), 1) :.4f} %")
            cumulative_loss += loss
            epoch_px_nb += batch_px_nb.detach().item()

            optimizer.zero_grad()

        cumulative_loss = cumulative_loss / max(epoch_px_nb, 1)
        print(f"        Sequence loss: {cumulative_loss};  Learning rate: {optimizer.param_groups[0]['lr']}")
        return cumulative_loss
        
    @torch.no_grad()
    def evaluate(self, img_path, gt_path, mask_path, gradient_path, dataset):
        super().eval()

        image_paths = [str(path) for path in sorted(img_path.glob("*.png"))]
        gt_paths = [str(path) for path in sorted(gt_path.glob("*.npy"))]
        mask_paths = [str(path) for path in sorted(mask_path.glob("*.png"))]
        gradient_paths = [str(path) for path in sorted(gradient_path.glob("*.npy"))]
        assert len(image_paths) == len(gt_paths)+1, f'Input ({len(image_paths)})/ GT Scalars ({len(gt_paths)+1}) mismatch in number of files'
        assert len(image_paths) == len(mask_paths)+1, f'Input ({len(image_paths)})/ GT Masks ({len(mask_paths)+1}) mismatch in number of files'
        assert len(image_paths) == len(gradient_paths)+1, f'Input ({len(image_paths)})/ GT Gradients ({len(gradient_paths)+1}) mismatch in number of files'

        loss = 0.0
        cumulative_loss = 0.0
        batch_px_nb = 0; epoch_px_nb = 0
        for start in range(0, len(gt_paths), MINI_BATCH):
            end = min(start+MINI_BATCH, len(gt_paths))
            if start == end: continue
            batch_size = end - start

            batch = torch.zeros(batch_size, 6, 480, 640).to(device=self.device)
            batch_gt_scalars = torch.zeros(batch_size, 480, 640).to(device=self.device)
            batch_valid_masks = torch.zeros(batch_size, 480, 640).to(device=self.device)
            batch_gradient_dir = torch.zeros(batch_size, 2, 480, 640).to(device=self.device)

            for j in range(batch_size):
                image1 = read_image(image_paths[start+j],mode=ImageReadMode.RGB)
                image2 = read_image(image_paths[start+j+1],mode=ImageReadMode.RGB)

                image1 = convert_image_dtype(image1, torch.float32)
                image2 = convert_image_dtype(image2, torch.float32)
                batch[j,:3,:,:] = image1.to(device=self.device)
                batch[j,3:,:,:] = image2.to(device=self.device)

                gt_scalars = torch.from_numpy(np.load(gt_paths[start+j])).to(device=self.device, dtype=torch.float32)
                valid_mask = read_image(str(mask_paths[start+j]), mode=ImageReadMode.GRAY)[0].to(device=self.device)
                valid_mask = (valid_mask > 0).to(torch.float32)
                gt_gradients = torch.from_numpy(np.load(gradient_paths[start+j])).to(device=self.device, dtype=torch.float32)
                gt_gradients = gt_gradients.permute(2,0,1)

                batch_gt_scalars[j,:,:] = gt_scalars
                batch_valid_masks[j,:,:] = valid_mask
                batch_gradient_dir[j,:,:,:] = gt_gradients

            batch_output = self.forward(batch)
            normal_flow_predictions = self.project_flow_to_normal_scalar(batch_output, batch_gradient_dir)

            squared_error = (normal_flow_predictions-batch_gt_scalars)**2
            loss = (squared_error * batch_valid_masks).sum() / ZERO_MSE[dataset]
            batch_px_nb = batch_valid_masks.sum()

            loss = loss.item() * 100
            print(f"        Completed: {end}/{BATCHES_PER_EPOCH},  Loss: {loss/max(batch_px_nb.item(), 1) :.4f} %")
            cumulative_loss += loss
            epoch_px_nb += batch_px_nb.detach().item()

        cumulative_loss = cumulative_loss / max(epoch_px_nb, 1)
        print(f"        Sequence loss: {cumulative_loss};  Learning rate: {optimizer.param_groups[0]['lr']}")
        return cumulative_loss


def generate_training_batches():
    path_tuple = []

    for dataset in training_sets:
        img_path = Path(f"{ROOT}/{dataset}/image_left")
        gt_path = Path(f"{ROOT}/{dataset}/normal_flow_gt/scalar")
        mask_path = Path(f"{ROOT}/{dataset}/normal_flow_gt/valid_mask")
        gradient_path = Path(f"{ROOT}/{dataset}/normal_flow_gt/gradient_dir")

        image_paths = [str(path) for path in sorted(img_path.glob("*.png"))]
        gt_paths = [str(path) for path in sorted(gt_path.glob("*.npy"))]
        mask_paths = [str(path) for path in sorted(mask_path.glob("*.png"))]
        gradient_paths = [str(path) for path in sorted(gradient_path.glob("*.npy"))]
        assert len(image_paths) == len(gt_paths)+1, f'Input ({len(image_paths)})/ GT Scalars ({len(gt_paths)+1}) mismatch in number of files for dataset: {dataset}'
        assert len(image_paths) == len(mask_paths)+1, f'Input ({len(image_paths)})/ GT Masks ({len(mask_paths)+1}) mismatch in number of files for dataset: {dataset}' 
        assert len(image_paths) == len(gradient_paths)+1, f'Input ({len(image_paths)})/ GT Gradients ({len(gradient_paths)+1}) mismatch in number of files for dataset: {dataset}'

        for start in range(0, len(gt_paths), MINI_BATCH):
            end = min(start+MINI_BATCH, len(gt_paths))
            if start == end: continue

            path_tuple.append((image_paths[start:end+1], gt_paths[start:end], mask_paths[start:end], gradient_paths[start:end], dataset))

    return path_tuple



LR = 3e-5
MINI_BATCH = 8
EPOCHS = 1000
BATCHES_PER_EPOCH = 500
ROOT = "/home/daniel/Optical_Flow/Datasets/TartanAir"
ZERO_MSE = {"Hospital": 109.9318, "Office": 164.2847, "Seaside_town": 417.5715, "Soulcity": 253.7524, "Western_desert": 249.8860, "Amusement": 67.7512, "Carwelding": 173.2149, "Neighborhood": 73.6183, "Japanese_alley": 65.8193}



model = Conv_Encoder_Decoder().to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6)

training_sets = ["Amusement", "Carwelding", "Neighborhood", "Seaside_town", "Soulcity", "Western_desert", "Japanese_alley"]
validation_sets = ["Hospital", "Office",]

training_metrics = {"loss":[], "lr":[]}
eval_metrics = {"loss":{}}

path_tuple = generate_training_batches()
BATCHES_PER_EPOCH = min(BATCHES_PER_EPOCH, len(path_tuple))

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}:")
    random.shuffle(path_tuple)

    training_loss = model.fit(optimizer, path_tuple)
    training_metrics["loss"].append(training_loss)
    training_metrics["lr"].append(optimizer.param_groups[0]['lr'])
    validation_loss = 0

    for dataset in validation_sets:
        print(f"    Evaluating on {dataset}")

        img_path = Path(f"{ROOT}/{dataset}/image_left")
        gt_path = Path(f"{ROOT}/{dataset}/normal_flow_gt/scalar")
        mask_path = Path(f"{ROOT}/{dataset}/normal_flow_gt/valid_mask")
        gradient_path = Path(f"{ROOT}/{dataset}/normal_flow_gt/gradient_dir")

        set_loss = model.evaluate(img_path, gt_path, mask_path, gradient_path, dataset)
        validation_loss += set_loss / len(validation_sets)

        if dataset in eval_metrics["loss"]:
            eval_metrics["loss"][dataset] = eval_metrics["loss"][dataset] + [set_loss]
        else:
            eval_metrics["loss"][dataset] = [set_loss]

    scheduler.step(validation_loss)

    with open("training_metrics.json", "w") as file:
        json.dump(training_metrics, file, indent=4)

    with open("validation_metrics.json", "w") as file:
        json.dump(eval_metrics, file, indent=4)

    torch.save(model.state_dict(), "model_weights.pth")