import subprocess

# 构建命令
command = [
    "/home/syd/Code/Llama/llama.cpp.gpu/llama.cpp/llama-cli",
    "-m", "/home/syd/Code/Llama/llama.cpp/models/Quantized_models/Llama-3.1-8B-Instruct_llamacpp_Q4_K_M.gguf",
    "--prompt", "What is the capital of France?",
    "--n-gpu-layers", "10",
    "--n-predict", "50"  # 添加此行限制输出为50个token
]

# 捕获输出
result = subprocess.run(command, capture_output=True, text=True)

# 将输出保存到文件
output_file = "llama_cpp_output.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(result.stdout)

print(f"输出已保存到文件: {output_file}")
