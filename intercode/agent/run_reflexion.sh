CUDA_VISIBLE_DEVICES=0 /data/fengxueyang/anaconda3/envs/intercode/bin/python -m main \
        --data_path /data/fengxueyang/intercode/data/sql/spider/ic_spider_dev.json \
        --env sql \
        --image_name docker-env-sql \
        --num_trials 10 \
        --num_envs 100 \
        --base_model "/data/pretrain_dir/Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496" \
        --run_name "reflexion_run_logs_llama_13b" \
        --use_memory \
