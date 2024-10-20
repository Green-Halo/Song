import subprocess
from datasets import load_dataset

# llama.cpp 的路径
llama_cpp_executable = "/home/syd/Code/Llama/llama.cpp/llama-cli"
# 量化模型路径
model_path = "/home/syd/Code/Llama/llama.cpp/models/Quantized_models/Llama-3.1-8B-Instruct_llamacpp_Q4_K_M.gguf"

print(f"Using model: {model_path}")

# 调用 llama.cpp 的推理函数
def run_llama_cpp(prompt):
    try:
        print(f"Running llama.cpp with model: {model_path} and prompt: {prompt}")
        result = subprocess.run([llama_cpp_executable, '-m', model_path, '--prompt', prompt, '-t', '12'], capture_output=True, text=True)
        print(f"Llama.cpp output: {result.stdout[:200]}...")  # 只打印前200个字符作为示例
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error in llama.cpp execution: {e}")
        return ""

# 加载所有需要的 GLUE 数据集，去掉 mnli
glue_datasets = ["sst2", "mrpc", "qqp", "qnli", "rte", "wnli"]
glue_data = {task: load_dataset("glue", task) for task in glue_datasets}

# 对于 mnli，加载两个不同的验证集
glue_data["mnli_matched"] = load_dataset("glue", "mnli", split="validation_matched")
glue_data["mnli_mismatched"] = load_dataset("glue", "mnli", split="validation_mismatched")

# 定义格式化输入的函数
def format_glue_input(example, task):
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
