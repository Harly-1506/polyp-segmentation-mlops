import os
from datetime import datetime
from pathlib import Path
from time import process_time

from loguru import logger


def setup_logger(
    model_name: str = "model", log_type: str = "train", level: str = "DEBUG"
):
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = f"{model_name}_{date_str}_{log_type}.log"
    print(f"Log file: {log_file}")
    log_dir = "./logs/training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger.remove()
    logger.add(
        log_path,
        rotation="10 MB",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    return logger


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start = process_time()
        result = func(*args, **kwargs)
        end = process_time()
        logger.info(f"Execution time for `{func.__name__}`: {end - start:.4f}s")
        return result

    return wrapper


def setup_logger_ray(
    world_rank,
    model_name: str = "model",
    log_type: str = "train",
    level: str = "DEBUG",
):

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{model_name}_{date_str}_{log_type}_worker{world_rank}.log"
    log_dir = "./logs/training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger.remove()

    logger.add(
        log_path,
        rotation="10 MB",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    logger.add(lambda msg: print(msg, end=""), level=level)

    return logger


if __name__ == "__main__":
    # Example usage
    logger = setup_logger(model_name="example_model", log_type="train")
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.error("This is an error message.")

    @measure_execution_time
    def example_function():
        for _ in range(1000000):
            pass

    example_function()  # This will log the execution time of the function
