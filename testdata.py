import subprocess
from datasets import load_dataset
import time# 打印数据集的样例以检查其结构


# 加载所有需要的 GLUE 数据集，去掉 mnli
glue_datasets = ["sst2", "mrpc", "qqp", "qnli", "rte", "wnli"]
glue_data = {task: load_dataset("glue", task) for task in glue_datasets}

# 对于 mnli，加载两个不同的验证集
glue_data["mnli_matched"] = load_dataset("glue", "mnli", split="validation_matched")
glue_data["mnli_mismatched"] = load_dataset("glue", "mnli", split="validation_mismatched")
for task in glue_datasets:
    print(f"Task: {task}")
    print(glue_data[task]['validation'][0])
    print("-" * 50)
    
for example in data_split:
    prompt = format_glue_input(example, task)
    print(f"Formatted prompt for task {task}: {prompt}")  # 打印格式化后的 prompt
    output = run_llama_cpp(prompt)
    print(f"Output for task {task}: {output}")  # 打印模型输出
