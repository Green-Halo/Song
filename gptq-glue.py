import time, os, gc, torch
from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM
from datasets import load_dataset, DownloadConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
torch.cuda.empty_cache()

# 设置模型名称
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)

# 检查是否有pad_token，如果没有则添加
tokenizer.pad_token = tokenizer.eos_token

# 定义量化位数
selected_bits = [4]

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
#glue_tasks = ["sst2"]
# 创建合并样本列表
examples = []

# 加载并处理每个GLUE任务的数据集
datasets = {}
for glue_task in glue_tasks:
    print(f"Processing GLUE task: {glue_task}")
    try:
        dataset = load_dataset("glue", glue_task, split="train[:1%]", cache_dir=datasets_dir, download_config=download_config)
        datasets[glue_task] = dataset
        print(f"Columns for {glue_task}: {dataset.column_names}")
    except KeyError:
        print(f"Task {glue_task} not found or does not have a 'train' split.")
        continue

# 获取每个数据集的最大长度
max_length = max(len(dataset) for dataset in datasets.values())

# 遍历并处理每个数据集
for i in range(max_length):
    for glue_task, dataset in datasets.items():
        if i < len(dataset):
            example = dataset[i]
            if glue_task == "mrpc":
                examples.append(f"{example['sentence1']} {example['sentence2']}")
            elif glue_task == "qqp":
                examples.append(f"{example['question1']} {example['question2']}")
            elif glue_task == "mnli":
                examples.append(f"{example['premise']} {example['hypothesis']}")
            elif glue_task == "qnli":
                examples.append(f"{example['question']} {example['sentence']}")
            elif glue_task in ["rte", "wnli"]:
                examples.append(f"{example['sentence1']} {example['sentence2']}")
            elif glue_task == "sst2":
                examples.append(example["sentence"])

batch_size = 4096 # 设定批处理大小为
tokenized_length = 256

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
    
    quantized_model_name = f"Llama3-8B-GPTQ-{bits}bit-all_glue_tasks-remove-tokenizer"
    
    gptq_config = GPTQConfig(
        bits=bits,
        dataset=tokenized_examples,
       # tokenizer=tokenizer
    )

    # 加载模型并进行量化
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=gptq_config,
        )
    print("序列化问题")
    model.config.quantization_config.dataset = None
    print("量化结束 保存模型")
    model.save_pretrained(os.path.join(save_dir, quantized_model_name))
    print('保存模型完成') 
    print("保存tokenizer")
    tokenizer.save_pretrained(os.path.join(save_dir, quantized_model_name))
    print('保存tokenizer完成')
    time_taken = time.time() - start_time
    print(f"{bits}-bit quantization: {time_taken:.2f} seconds.")
    
    # 清理模型并释放CUDA内存
    del model
    torch.cuda.empty_cache()
    gc.collect()