import subprocess
from datasets import load_dataset
import time

# llama.cpp 的路径
llama_cpp_executable = "/home/syd/Code/Llama/llama.cpp.gpu/llama.cpp/llama-cli"
# 量化模型路径
model_path = "/home/syd/Code/Llama/llama.cpp/models/Quantized_models/Llama-3.1-8B-Instruct_llamacpp_Q4_K_M.gguf"

# 设置 GPU 层数
n_gpu_layers = 10  # 你可以根据你的 GPU 选择合适的层数

# 调用 llama.cpp 的推理函数，支持 GPU
def run_llama_cpp(prompt, retry_limit=3):
    for attempt in range(retry_limit):
        try:
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

# 加载 sst2 数据集
print("Loading sst2 dataset...")
sst2_data = load_dataset("glue", "sst2")
print("sst2 dataset loaded successfully.")

# 定义格式化输入的函数
def format_sst2_input(example):
    return f"Please classify the sentiment of the following sentence as 'positive' or 'negative': \"{example['sentence']}\""

# 提取模型生成的预测结果
def extract_prediction(output):
    return "positive" if "positive" in output.lower() else "negative"

# 评估 sst2 的预测结果
def evaluate_sst2_task(dataset):
    correct = 0
    total = 0
    data_split = dataset['validation']  # 使用验证集
    
    for example in data_split:
        prompt = format_sst2_input(example)
        output = run_llama_cpp(prompt)
        prediction = extract_prediction(output)
        
        # 检查是否成功生成了预测结果
        if not output:
            print(f"Warning: No output for example in sst2. Skipping this example.")
            continue

        # 对比预测结果和真实标签
        label = "positive" if example["label"] == 1 else "negative"
        
        # 比较预测结果与标签
        if prediction == label:
            correct += 1
        total += 1

        # 仅测试前 10 个样本
        if total >= 10:
            break

    # 如果数据集没有样本
    if total == 0:
        print(f"No examples were processed for sst2")
        return 0

    # 计算准确率
    accuracy = correct / total * 100
    return accuracy

# 评估 sst2 任务
accuracy = evaluate_sst2_task(sst2_data)
print(f"Accuracy on sst2: {accuracy:.2f}%")
