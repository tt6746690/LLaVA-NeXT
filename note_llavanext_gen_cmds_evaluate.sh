#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llms-eval-run.sh "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2" "videochatgpt" "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/vcgbench" "qwen_1_5" "llava_qwen" "64" "llava_vid" 2>&1 | tee "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/vcgbench/bash_script_log.txt" ; [ ${PIPESTATUS[0]} -ne 0 ] && { exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llms-eval-run.sh "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2" "activitynetqa" "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/activitynetqa" "qwen_1_5" "llava_qwen" "64" "llava_vid" 2>&1 | tee "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/activitynetqa/bash_script_log.txt" ; [ ${PIPESTATUS[0]} -ne 0 ] && { exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llms-eval-run.sh "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2" "mlvu" "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/mlvu" "qwen_1_5" "llava_qwen" "64" "llava_vid" 2>&1 | tee "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/mlvu/bash_script_log.txt" ; [ ${PIPESTATUS[0]} -ne 0 ] && { exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llms-eval-run.sh "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2" "mvbench" "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/mvbench" "qwen_1_5" "llava_qwen" "64" "llava_vid" 2>&1 | tee "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/mvbench/bash_script_log.txt" ; [ ${PIPESTATUS[0]} -ne 0 ] && { exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llms-eval-run.sh "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2" "nextqa_mc_test" "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/nextqa" "qwen_1_5" "llava_qwen" "64" "llava_vid" 2>&1 | tee "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/nextqa/bash_script_log.txt" ; [ ${PIPESTATUS[0]} -ne 0 ] && { exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llms-eval-run.sh "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2" "perceptiontest_val_mc" "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/perceptiontest" "qwen_1_5" "llava_qwen" "64" "llava_vid" 2>&1 | tee "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/perceptiontest/bash_script_log.txt" ; [ ${PIPESTATUS[0]} -ne 0 ] && { exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llms-eval-run.sh "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2" "longvideobench_val_v" "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/longvideobench" "qwen_1_5" "llava_qwen" "64" "llava_vid" 2>&1 | tee "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/longvideobench/bash_script_log.txt" ; [ ${PIPESTATUS[0]} -ne 0 ] && { exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llms-eval-run.sh "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2" "videomme" "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/videomme" "qwen_1_5" "llava_qwen" "64" "llava_vid" 2>&1 | tee "/fsx/wpq/.results/baselines/lmms-lab/LLaVA-Video-7B-Qwen2/llms-eval/videomme/bash_script_log.txt" ; [ ${PIPESTATUS[0]} -ne 0 ] && { exit 1; }
