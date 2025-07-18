BASE_MODEL="Qwen/Qwen3-4B"
OUTPUT_PATH="/home/rwkv/model/qwen3-4b-miss-mini-d"
DATA_PATH=/home/rwkv/JL/data/MetaMathQA

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
HF_ENDPOINT="https://hf-mirror.com" deepspeed --include=localhost:0 finetune.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --use_miss True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --miss_r 64 \
    --miss_mini_r 64 \
    --init_miss_weight 'mini' \
    --data_path $DATA_PATH \
    --dataset_field query response \
    --dataset_split "train[:100000]"\
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --merge True \
    --report_to "none" \
    --run_name "miss"
