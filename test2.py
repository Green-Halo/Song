import subprocess

# 定义命令和参数
command = [
    "/home/syd/Code/Llama/llama.cpp.gpu/llama.cpp/llama-cli",
    "-m", "/home/syd/Code/Llama/llama.cpp/models/Quantized_models/Llama-3.1-8B-Instruct_llamacpp_Q4_K_M.gguf",
    "-cnv"
]

# 打开输出文件
with open("output.txt", "w") as output_file:
    # 使用 subprocess.Popen 运行命令，并捕获标准输出和标准错误
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # 逐行读取输出并写入文件
    while True:
        line = process.stdout.readline()
        if not line:
            break
        output_file.write(line)
        print(line, end='')  # 同时在终端打印输出（可选）

# 等待进程结束
process.wait()

print("命令输出已保存到 output.txt 文件中。")
