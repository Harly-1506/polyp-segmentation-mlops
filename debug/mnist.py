import os
import tempfile

import ray.train.torch
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


def train_func():
    rank = ray.train.get_context().get_world_rank()

    # Model, Loss, Optimizer
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model = ray.train.torch.prepare_model(model)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Data
    transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    data_dir = os.path.join(tempfile.gettempdir(), "data")

    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    if rank == 0:
        val_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # Training
    for epoch in range(1):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, disable=rank != 0, desc=f"Epoch {epoch+1} [Train]"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        metrics = {"train_loss": train_loss / len(train_loader), "epoch": epoch}

        # Validation chỉ khi rank 0
        if rank == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            metrics["val_loss"] = val_loss / len(val_loader)
            metrics["val_acc"] = correct / total

        # Save checkpoint
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )

        if rank == 0:
            print(metrics)


if __name__ == "__main__":
    scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=False)

    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
    )
    result = trainer.fit()

    with result.checkpoint.as_directory() as checkpoint_dir:
        model_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
        model = resnet18(num_classes=10)
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        model.load_state_dict(model_state_dict)



# import os
# import tempfile

# import torch
# from torch.nn import CrossEntropyLoss
# from torch.optim import Adam
# from torch.utils.data import DataLoader
# from torchvision.models import resnet18
# from torchvision.datasets import FashionMNIST
# from torchvision.transforms import ToTensor, Normalize, Compose
# from tqdm import tqdm

# import ray.train.torch

# def train_func():
#     # Model, Loss, Optimizer
#     model = resnet18(num_classes=10)
#     model.conv1 = torch.nn.Conv2d(
#         1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#     )
#     # [1] Prepare model.
#     model = ray.train.torch.prepare_model(model)
#     # model.to("cuda")  # This is done by `prepare_model`
#     criterion = CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=0.001)

#     # Data
#     transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
#     data_dir = os.path.join(tempfile.gettempdir(), "data")
#     train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
#     train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
#     # [2] Prepare dataloader.
#     train_loader = ray.train.torch.prepare_data_loader(train_loader)

#     # Training
#     for epoch in range(1):
#         if ray.train.get_context().get_world_size() > 1:
#             train_loader.sampler.set_epoch(epoch)

#         for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
#             # This is done by `prepare_data_loader`!
#             # images, labels = images.to("cuda"), labels.to("cuda")
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # [3] Report metrics and checkpoint.
#         metrics = {"loss": loss.item(), "epoch": epoch}
#         with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
#             torch.save(
#                 model.module.state_dict(),
#                 os.path.join(temp_checkpoint_dir, "model.pt")
#             )
#             ray.train.report(
#                 metrics,
#                 checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
#             )
#         if ray.train.get_context().get_world_rank() == 0:
#             print(metrics)

# if __name__ == "__main__":
#     # [4] Configure scaling and resource requirements.
#     scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=False)

#     # [5] Launch distributed training job.
#     trainer = ray.train.torch.TorchTrainer(
#         train_func,
#         scaling_config=scaling_config,
#         # [5a] If running in a multi-node cluster, this is where you
#         # should configure the run's persistent storage that is accessible
#         # across all worker nodes.
#         # run_config=ray.train.RunConfig(storage_path="s3://..."),
#     )
#     result = trainer.fit()

#     # [6] Load the trained model.
#     with result.checkpoint.as_directory() as checkpoint_dir:
#         model_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
#         model = resnet18(num_classes=10)
#         model.conv1 = torch.nn.Conv2d(
#             1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#         )
#         model.load_state_dict(model_state_dict)