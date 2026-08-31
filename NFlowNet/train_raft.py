import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import convert_image_dtype
import numpy as np
from pathlib import Path
import json
import random
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class Raft_Small_Class(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.device = torch.device("cuda")

        weights = (Raft_Small_Weights.DEFAULT if pretrained else None)
        self.raft = raft_small(weights=weights)
        self.transforms = (weights.transforms() if weights is not None else None)


    def forward(self, image1, image2, num_flow_updates=12):
        return self.raft(image1, image2, num_flow_updates=num_flow_updates)


    def prepare_images(self, image1, image2):
        if self.transforms is not None:
            return self.transforms(image1, image2)
        return (image1 * 2.0 - 1.0, image2 * 2.0 - 1.0)


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

            batch1 = torch.zeros(batch_size, 3, 480, 640).to(device=self.device)
            batch2 = torch.zeros(batch_size, 3, 480, 640).to(device=self.device)
            batch_gt_scalars = torch.zeros(batch_size, 480, 640).to(device=self.device)
            batch_valid_masks = torch.zeros(batch_size, 480, 640).to(device=self.device)
            batch_gradient_dir = torch.zeros(batch_size, 2, 480, 640).to(device=self.device)

            for j in range(batch_size):
                image1 = read_image(image_paths[j],mode=ImageReadMode.RGB)
                image2 = read_image(image_paths[j+1],mode=ImageReadMode.RGB)

                image1 = convert_image_dtype(image1, torch.float32)
                image2 = convert_image_dtype(image2, torch.float32)
                batch1[j,:,:,:] = image1.to(device=self.device)
                batch2[j,:,:,:] = image2.to(device=self.device)

                gt_scalars = torch.from_numpy(np.load(gt_paths[j])).to(device=self.device, dtype=torch.float32)
                valid_mask = read_image(str(mask_paths[j]), mode=ImageReadMode.GRAY)[0].to(device=self.device)
                valid_mask = (valid_mask > 0).to(torch.float32)
                gt_gradients = torch.from_numpy(np.load(gradient_paths[j])).to(device=self.device, dtype=torch.float32)
                gt_gradients = gt_gradients.permute(2,0,1)

                batch_gt_scalars[j,:,:] = gt_scalars
                batch_valid_masks[j,:,:] = valid_mask
                batch_gradient_dir[j,:,:,:] = gt_gradients

            batch1, batch2 = self.prepare_images(batch1, batch2)

            flow_predictions = self.forward(batch1, batch2)[-1]
            normal_flow_predictions = self.project_flow_to_normal_scalar(flow_predictions, batch_gradient_dir)

            squared_error = (normal_flow_predictions-batch_gt_scalars)**2
            loss = (squared_error * batch_valid_masks).sum() / ZERO_MSE[dataset]
            batch_px_nb = batch_valid_masks.sum()

            (loss/batch_px_nb.clamp(1)).backward()
            optimizer.step()
            loss = loss.item() * 100
            print(f"        Completed: {i+1}/{BATCHES_PER_EPOCH},  Loss: {loss/max(batch_px_nb.item(), 1) :.4f} %")
            cumulative_loss += loss
            epoch_px_nb += batch_px_nb.detach().item()
            optimizer.zero_grad()

        cumulative_loss = cumulative_loss / max(epoch_px_nb, 1)
        print(f"        Epoch loss: {cumulative_loss};  Learning rate: {optimizer.param_groups[0]['lr']}")
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
        batch_px_nb = 0; seq_px_nb = 0
        for start in range(0, len(gt_paths), MINI_BATCH):
            end = min(start+MINI_BATCH, len(gt_paths))
            if start == end: continue
            batch_size = end - start

            batch1 = torch.zeros(batch_size, 3, 480, 640).to(device=self.device)
            batch2 = torch.zeros(batch_size, 3, 480, 640).to(device=self.device)
            batch_gt_scalars = torch.zeros(batch_size, 480, 640).to(device=self.device)
            batch_valid_masks = torch.zeros(batch_size, 480, 640).to(device=self.device)
            batch_gradient_dir = torch.zeros(batch_size, 2, 480, 640).to(device=self.device)

            for j in range(batch_size):
                image1 = read_image(image_paths[start+j],mode=ImageReadMode.RGB)
                image2 = read_image(image_paths[start+j+1],mode=ImageReadMode.RGB)

                image1 = convert_image_dtype(image1, torch.float32)
                image2 = convert_image_dtype(image2, torch.float32)
                batch1[j,:,:,:] = image1.to(device=self.device)
                batch2[j,:,:,:] = image2.to(device=self.device)

                gt_scalars = torch.from_numpy(np.load(gt_paths[start+j])).to(device=self.device, dtype=torch.float32)
                valid_mask = read_image(str(mask_paths[start+j]), mode=ImageReadMode.GRAY)[0].to(device=self.device)
                valid_mask = (valid_mask > 0).to(torch.float32)
                gt_gradients = torch.from_numpy(np.load(gradient_paths[start+j])).to(device=self.device, dtype=torch.float32)
                gt_gradients = gt_gradients.permute(2,0,1)

                batch_gt_scalars[j,:,:] = gt_scalars
                batch_valid_masks[j,:,:] = valid_mask
                batch_gradient_dir[j,:,:,:] = gt_gradients

            batch1, batch2 = self.prepare_images(batch1, batch2)

            flow_predictions = self.forward(batch1, batch2)[-1]
            normal_flow_predictions = self.project_flow_to_normal_scalar(flow_predictions, batch_gradient_dir)

            squared_error = (normal_flow_predictions-batch_gt_scalars)**2
            batch_px_nb = batch_valid_masks.sum().detach().item()
            loss = 100 * (squared_error * batch_valid_masks).sum() / ZERO_MSE[dataset]

            print(f"        Completed: {end}/{len(gt_paths)},  Loss: {loss / max(batch_px_nb, 1) :.4f} %")
            cumulative_loss += loss.detach().item()
            seq_px_nb += batch_px_nb

        cumulative_loss = cumulative_loss / max(seq_px_nb, 1)
        print(f"        Epoch loss: {cumulative_loss}")
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
        assert len(image_paths) == len(gt_paths)+1, f'Input ({len(image_paths)})/ GT Scalars ({len(gt_paths)+1}) mismatch in number of files'
        assert len(image_paths) == len(mask_paths)+1, f'Input ({len(image_paths)})/ GT Masks ({len(mask_paths)+1}) mismatch in number of files' 
        assert len(image_paths) == len(gradient_paths)+1, f'Input ({len(image_paths)})/ GT Gradients ({len(gradient_paths)+1}) mismatch in number of files'

        for start in range(0, len(gt_paths), MINI_BATCH):
            end = min(start+MINI_BATCH, len(gt_paths))
            if start == end: continue

            path_tuple.append((image_paths[start:end+1], gt_paths[start:end], mask_paths[start:end], gradient_paths[start:end], dataset))

    return path_tuple



LR = 3e-5
MINI_BATCH = 4
EPOCHS = 200
BATCHES_PER_EPOCH = 512
ROOT = "/home/daniel/Optical_Flow/Datasets/TartanAir"
ZERO_MSE = {"Hospital": 109.9318, "Office": 164.2847, "Seaside_town": 417.5715, "Soulcity": 253.7524, "Western_desert": 249.8860, "Amusement": 67.7512, "Carwelding": 173.2149,}

model = Raft_Small_Class(pretrained=True).to("cuda")
optimizer = torch.optim.Adam(
    [{"params": model.raft.update_block.parameters(), "lr": LR},
     {"params": model.raft.feature_encoder.parameters(), "lr": LR/100},
     {"params": model.raft.context_encoder.parameters(), "lr": LR/100}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

training_sets = ["Hospital", "Office", "Seaside_town", "Soulcity", "Western_desert"]
validation_sets = ["Amusement", "Carwelding"]

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