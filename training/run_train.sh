#!/bin/bash

# -------------------------
# Configuration
# -------------------------

EPOCH=100
LR=1e-4
BATCHSIZE=16
TRAINSIZE=352
MODEL="UNet"

TRAIN_PATH="./dataset1/TrainDataset/"
TEST_PATH="./dataset1/TestDataset/"
SAVE_PATH="./model_pth/"
# LOG_PATH="./logs/train_$(date +%Y%m%d_%H%M%S).log"

# -------------------------
# Start training
# -------------------------

echo "🚀 Starting training..."
echo "Model: $MODEL | Epoch: $EPOCH | LR: $LR | BatchSize: $BATCHSIZE | TrainSize: $TRAINSIZE"
echo "Train Path: $TRAIN_PATH"
echo "Test Path: $TEST_PATH"
echo "Saving to: $SAVE_PATH"
echo "Logging to: $LOG_PATH"

# Run Python script and log output
python training/main.py \
    --epoch "$EPOCH" \
    --lr "$LR" \
    --batchsize "$BATCHSIZE" \
    --trainsize "$TRAINSIZE" \
    --model "$MODEL" \
    --train_path "$TRAIN_PATH" \
    --test_path "$TEST_PATH" \
    --train_save "$SAVE_PATH" | tee "$LOG_PATH"

echo "✅ Training complete!"
