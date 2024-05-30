from transformers import AutoTokenizer

# 创建分词器实例，这里以BERT的分词器为例
tokenizer = AutoTokenizer.from_pretrained("/data/pretrain_dir/Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93")

# 分词
tokens = tokenizer.tokenize("~ Hello, how are you?")

# 将分词后的结果转换为词汇标识符
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# 现在我们来找出特定词汇标识符对应的原始词汇
specific_token_id = token_ids[0]  # 假设我们想找第一个词汇标识符对应的原始词汇
specific_token = tokenizer.convert_ids_to_tokens([specific_token_id])[0]

print(f"Token ID: {specific_token_id}, Original Token: {specific_token}")
