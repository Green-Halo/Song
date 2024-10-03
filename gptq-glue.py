import time, os, gc, torch
from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM
from datasets import load_dataset, DownloadConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
torch.cuda.empty_cache()

# 设置模型名称
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

# 检查是否有pad_token，如果没有则添加
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 定义量化位数
selected_bits = [#8, 
                 4]

download_config = DownloadConfig()
download_config.num_proc = 8  # 根据您的网络情况调整
download_config.max_retries = 5

# 设置保存路径
save_dir = "/home/syd/Code/Quant_model"
os.makedirs(save_dir, exist_ok=True)

datasets_dir = "/home/syd/Code/Datasets"
os.makedirs(datasets_dir, exist_ok=True)

# 加载GLUE数据集
glue_tasks = ["sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]

# 创建合并样本列表
examples = []

# 遍历GLUE任务，加载和处理数据集
for glue_task in glue_tasks:
    print(f"Processing GLUE task: {glue_task}")
    
    # 加载数据集
    try:
        dataset = load_dataset("glue", glue_task, split="train[:1%]", cache_dir=datasets_dir, download_config=download_config)
        print(f"Columns for {glue_task}: {dataset.column_names}")
    except KeyError:
        print(f"Task {glue_task} not found or does not have a 'train' split.")
        continue
    
    # 提取文本数据
    if glue_task == "mrpc":
        examples += [f"{example['sentence1']} {example['sentence2']}" for example in dataset]
    elif glue_task == "qqp":
        examples += [f"{example['question1']} {example['question2']}" for example in dataset]
    elif glue_task == "mnli":
        examples += [f"{example['premise']} {example['hypothesis']}" for example in dataset]
    elif glue_task == "qnli":
        examples += [f"{example['question']} {example['sentence']}" for example in dataset]
    elif glue_task in ["rte", "wnli"]:
        examples += [f"{example['sentence1']} {example['sentence2']}" for example in dataset]
    elif glue_task == "sst2":
        examples += [example["sentence"] for example in dataset]

batch_size = 1024 # 设定批处理大小为
tokenized_length = 64

# 对合并的数据进行Tokenization
tokenized_examples = [
    tokenizer(example, return_tensors="pt", truncation=True, 
              max_length=tokenized_length, padding=True) 
    for example in examples[:batch_size]  
]
print("数据集编码完成.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 执行量化并保存模型
for bits in selected_bits:
    start_time = time.time()
    
    quantized_model_name = f"Llama3-8B-GPTQ-{bits}bit-all_glue_tasks"
    
    gptq_config = GPTQConfig(
        bits=bits,
        dataset=tokenized_examples,
        tokenizer=tokenizer
    )


    # 加载模型并进行量化
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=gptq_config,
        )
    model.gradient_checkpointing_enable()
    # 保存量化后的模型
    # 移除 dataset 避免序列化问题
    model.quantize_config.dataset = None
    
    
    model.save_pretrained(os.path.join(save_dir, quantized_model_name))
    tokenizer.save_pretrained(os.path.join(save_dir, quantized_model_name))

    time_taken = time.time() - start_time
    print(f"{bits}-bit quantization: {time_taken:.2f} seconds.")
    
    # 清理模型并释放CUDA内存
    del model
    torch.cuda.empty_cache()
    gc.collect()
