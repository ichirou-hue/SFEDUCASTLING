import os

MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_NAME = "khoilamalphaai/chess-coach-move-review"

OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
ADAPTER_DIR = OUTPUT_DIR
MERGED_DIR = os.path.join(OUTPUT_DIR, "merged")

MAX_SEQ_LENGTH = 4096

# Quantization Config
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = True

# LoRA Config
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Config
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3
OPTIMIZER = "paged_adamw_8bit"
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STRATEGY = "steps"
SAVE_STEPS = 100
EVAL_STRATEGY = "steps"
EVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 3
FP16 = False
BF16 = True
