# logger_setup.py
import os
from datetime import datetime

from loguru import logger
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

import ray
from ray import train


def setup_logger_ray(world_rank, model_name="model", log_type="train", level="DEBUG"):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{model_name}_{date_str}_{log_type}_worker{world_rank}.log"
    log_dir = "./logs/training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger.remove()
    logger.add(log_path,
               rotation="10 MB",
               level=level,
               enqueue=True,
               backtrace=True,
               diagnose=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                      "<level>{message}</level>")
    logger.add(lambda msg: print(msg, end=""), level=level)
    return logger
# train_ray.py


def train_loop_per_worker(config):
    rank = train.get_context().get_world_rank()
    logger = setup_logger_ray(rank)
    logger.info(f"Worker {rank} bắt đầu training...")

    # Ví dụ giả lập huấn luyện
    for i in range(3):
        logger.info(f"Worker {rank} epoch {i}")
    logger.info(f"Worker {rank} kết thúc training.")

if __name__ == "__main__":
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    )

    result = trainer.fit()
    print("Training kết thúc.")
