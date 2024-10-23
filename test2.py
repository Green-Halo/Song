import subprocess
import select
import time
from datasets import load_dataset

# 定义最大等待时间
TIMEOUT = 10  # 单位秒

glue_tasks = ["sst2"]
datasets = {}

# 加载 GLUE 任务数据集
for task in glue_tasks:
    datasets[task] = load_dataset("glue", task, split="train")

prompt_dic = {
    "sst2": """In the following conversations, predict the sentiment of the given sentence and output 0 if it is negative and 1 if it is positive. No analyses or explanations. Only respond with 0, 1."""
}

model_name = "Llama-3.1-8B-Instruct_llamacpp_Q4_K_M.gguf"
model_path = f"/home/syd/Code/Llama/llama.cpp/models/Quantized_models/Llama-3.1-8B-Instruct-llamacpp/{model_name}"
llama_path = "/home/syd/Code/Llama/llama.cpp.gpu/llama.cpp/llama-cli"

with open(f"{model_name}.txt", "w") as f:
    print("Running Llama model...")
    process = subprocess.Popen(
        [llama_path, '-m', model_path, '-cnv'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # 设置为行缓冲
    )
    
    print("send prompt")
    process.stdin.write(prompt_dic["sst2"] + '\n')
    process.stdin.flush()
    print("prompt has been sent")
    
    start_time = time.time()
    
    # 循环读取输出，但设置超时时间
    while True:
        ready, _, _ = select.select([process.stdout], [], [], 1)  # 超时1秒检查一次
        if ready:
            output = process.stdout.readline().strip()
            print(f"Ignored output: {output}")
        if time.time() - start_time > TIMEOUT:
            print("Timeout reached, proceeding to input sentences.")
            break
    
    # 发送句子并获取响应
    for i in range(3):
        sentence = datasets["sst2"][i]["sentence"]
        process.stdin.write(sentence + '\n')
        process.stdin.flush()
        print(f"Sentence '{sentence}' sent")
        
        # 逐行读取子进程的输出
        output = process.stdout.readline().strip()
        print(f"Received output: {output}")
        
        f.write(sentence + ' ' + output + '\n')
    
    process.stdin.close()  # 关闭输入
    
    # 在最后读取并处理子进程的错误输出
    error = process.stderr.read().strip()
    if error:
        print(f"Error for input {error}")
    
    process.wait()  # 等待子进程完全完成
