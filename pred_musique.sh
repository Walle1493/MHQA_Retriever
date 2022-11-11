export DATA_DIR=/home/mxdong/Data/MuSiQue/format_data
export TASK_NAME=Retriever3
export MODEL_NAME=/home/mxdong/Codes/Checkpoints/Retriever_MuSiQue/checkpoint-18400

CUDA_VISIBLE_DEVICES=0 python run_musique.py \
    --model_type bert \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_test \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 4   \
    --per_gpu_train_batch_size 4   \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 4.0 \
    --output_dir Checkpoints/$TASK_NAME/Test_Preds \
    --logging_steps 400 \
    --save_steps 400 \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --k_sent 4 \
    --overwrite_output_dir \
    # --fp16 \