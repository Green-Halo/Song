import subprocess
from datasets import load_dataset
import time

# llama.cpp 的路径
llama_cpp_executable = "/home/syd/Code/Llama/llama.cpp.gpu/llama.cpp/llama-cli"
# 量化模型路径
model_path = "/home/syd/Code/Llama/llama.cpp/models/Quantized_models/Llama-3.1-8B-Instruct_llamacpp_Q4_K_M.gguf"

# 设置 GPU 层数
n_gpu_layers = 20  # 你可以根据你的 GPU 选择合适的层数

# 调用 llama.cpp 的推理函数，支持 GPU
def run_llama_cpp(prompt, retry_limit=3):
    for attempt in range(retry_limit):
        try:
            # 添加 --n-gpu-layers 选项以启用 GPU 推理
            print(f"Attempt {attempt + 1}: Running llama.cpp with prompt: {prompt[:50]}...")  # 打印部分提示
            result = subprocess.run([llama_cpp_executable, '-m', model_path, '--prompt', prompt, '--n-gpu-layers', str(n_gpu_layers)],
                                    capture_output=True, text=True, timeout=300)
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()[:100]}...")  # 打印部分输出用于检查
                return result.stdout
            else:
                print(f"Warning: No output on attempt {attempt + 1}. Retrying...")
                time.sleep(2)  # 延迟几秒后重试
        except subprocess.CalledProcessError as e:
            print(f"Error in llama.cpp execution: {e}")
        except subprocess.TimeoutExpired:
            print(f"Warning: Timed out on attempt {attempt + 1}. Retrying...")
            time.sleep(2)
    
    return ""  # 如果超出重试次数仍然失败，返回空字符串

# 加载所有需要的 GLUE 数据集，去掉 mnli
#glue_datasets = ["sst2", "mrpc", "qqp", "qnli", "rte", "wnli"]
glue_datasets = ["mrpc", "qqp", "qnli", "rte", "wnli"]

# 加载数据集并打印加载状态
glue_data = {}
for task in glue_datasets:
    print(f"Loading {task} dataset...")
    glue_data[task] = load_dataset("glue", task)
    print(f"{task} dataset loaded successfully.")

# 对于 mnli，加载两个不同的验证集
print("Loading mnli_matched dataset...")
glue_data["mnli_matched"] = load_dataset("glue", "mnli", split="validation_matched")
print("mnli_matched dataset loaded successfully.")

print("Loading mnli_mismatched dataset...")
glue_data["mnli_mismatched"] = load_dataset("glue", "mnli", split="validation_mismatched")
print("mnli_mismatched dataset loaded successfully.")

# 定义格式化输入的函数
def format_glue_input(example, task):
    print(f"Formatting input for {task} task.")
    if task == "sst2":
        return f"Please classify the sentiment of the following sentence as 'positive' or 'negative': \"{example['sentence']}\""
    elif task == "mrpc":
        return f"Please determine if the following two sentences are paraphrases of each other: Sentence 1: \"{example['sentence1']}\" Sentence 2: \"{example['sentence2']}\""
    elif task == "qqp":
        return f"Please determine if the following two questions are paraphrases of each other: Question 1: \"{example['question1']}\" Question 2: \"{example['question2']}\""
    elif task == "mnli":
        return f"Premise: \"{example['premise']}\" Hypothesis: \"{example['hypothesis']}\". Is the hypothesis correct? Answer with 'yes' or 'no'."
    elif task == "qnli":
        return f"Question: \"{example['question']}\" Sentence: \"{example['sentence']}\". Does the sentence answer the question? Answer with 'yes' or 'no'."
    elif task in ["rte", "wnli"]:
        return f"Sentence 1: \"{example['sentence1']}\" Sentence 2: \"{example['sentence2']}\". Are these sentences similar in meaning? Answer with 'yes' or 'no'."

# 提取模型生成的预测结果
def extract_prediction(output, task):
    print(f"Extracting prediction for {task} task.")
    if task == "sst2":
        return "positive" if "positive" in output.lower() else "negative"
    elif task in ["mrpc", "qqp"]:
        return "paraphrase" if "yes" in output.lower() else "not paraphrase"
    elif task in ["mnli", "qnli", "rte", "wnli"]:
        return "yes" if "yes" in output.lower() else "no"

# 计算任务的准确度
def evaluate_glue_task(dataset, task, is_single_split=False):
    correct = 0
    total = 0
    
    # 如果是单独的数据集分割（如 mnli_matched 或 mnli_mismatched），直接使用 dataset
    if is_single_split:
        data_split = dataset
    else:
        # 获取可用的分割
        available_splits = dataset.keys()
        if 'validation' in available_splits:
            split = 'validation'
        elif 'test' in available_splits:
            split = 'test'
        else:
            print(f"No validation or test split found for {task}.")
            return 0
        data_split = dataset[split]

    for example in data_split:
        # 打印每个样本的内容以便检查格式
        print(f"Processing example from {task}: {example}")
        
        prompt = format_glue_input(example, task)
        output = run_llama_cpp(prompt)
        prediction = extract_prediction(output, task)
        
        # 检查是否成功生成了预测结果
        if not output:
            print(f"Warning: No output for example in {task}. Skipping this example.")
            continue

        # 对比预测结果和真实标签
        if task == "sst2":
            label = "positive" if example["label"] == 1 else "negative"
        elif task in ["mrpc", "qqp"]:
            label = "paraphrase" if example["label"] == 1 else "not paraphrase"
        elif task in ["mnli", "qnli", "rte", "wnli"]:
            label = "yes" if example["label"] == 1 else "no"
        
        # 比较预测结果与标签
        if prediction == label:
            correct += 1
        total += 1

    # 如果数据集没有样本
    if total == 0:
        print(f"No examples were processed for {task}")
        return 0

    # 计算准确率
    accuracy = correct / total * 100
    return accuracy

# 评估所有任务（去掉 mnli，因为我们会分别评估 mnli_matched 和 mnli_mismatched）
for task in glue_datasets:
    accuracy = evaluate_glue_task(glue_data[task], task)
    print(f"Accuracy on {task}: {accuracy:.2f}%")

# 评估 mnli 的 matched 和 mismatched 数据集
accuracy_matched = evaluate_glue_task(glue_data["mnli_matched"], "mnli", is_single_split=True)
print(f"Accuracy on mnli_matched: {accuracy_matched:.2f}%")

accuracy_mismatched = evaluate_glue_task(glue_data["mnli_mismatched"], "mnli", is_single_split=True)
print(f"Accuracy on mnli_mismatched: {accuracy_mismatched:.2f}%")
