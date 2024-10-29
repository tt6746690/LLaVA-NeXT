
cmd = f"""
{launcher} \
    ChatUniVi/train/train_mem.py \
    {'--deepspeed '+str(deepspeed) if deepspeed else ''} \
    --model_name_or_path {model_name_or_path} \
    --version {get_default_chat_template_version(model_name_or_path)} \
    --model_use {model_use} \
    --dataset_use {dataset_use} \
    --vision_tower {vision_tower} \
    {"--pretrain_mm_mlp_adapter " + '"' + str(pretrain_mm_mlp_adapter) + '"' if pretrain_mm_mlp_adapter is not None else ""} \
    --mm_projector_type {mm_projector_type} \
    {"--tune_mm_mlp_adapter True" if finetune_type == 'pt' else ""} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    {"--train_size " + str(train_size) if train_size else ""} \
    --num_train_epochs {num_train_epochs} \
    --per_device_train_batch_size {per_device_train_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps {str(50000) if finetune_type == 'sft' else str(24000)} \
    --save_total_limit 1 \
    --learning_rate {'2e-5' if finetune_type == 'sft' else '2e-3'} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing {gradient_checkpointing} \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to {'none' if test_run else report_to} \
    {model_config_overwrites_args if model_config_overwrites_args!='' else ''} \
    --output_dir "{output_dir}"
"""