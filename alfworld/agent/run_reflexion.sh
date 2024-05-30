export ALFWORLD_DATA="/data/fengxueyang/reflexion/alfworld_runs/data"
export CUDA_VISIBLE_DEVICES="0"
python main.py \
        --num_trials 10 \
        --num_envs 134 \
        --run_name "reflexion_run_logs_llama13b" \
        --use_memory \
        --model "gpt-3.5-turbo" 