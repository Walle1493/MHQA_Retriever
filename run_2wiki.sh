export DATA_DIR=/home/mxdong/Data/HotpotQA/format_data
export TASK_NAME=Retriever1
export MODEL_NAME=bert-base-uncased

CUDA_VISIBLE_DEVICES=2 python run_hotpotqa.py \
    --model_type bert \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 8   \
    --per_gpu_train_batch_size 8   \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 4.0 \
    --output_dir Checkpoints/$TASK_NAME/${MODEL_NAME} \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --k_sent 2 \
    --overwrite_output_dir \
    # --fp16 \