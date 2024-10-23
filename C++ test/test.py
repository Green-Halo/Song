import subprocess

# 定义可执行文件路径
executable_path = "/home/syd/Code/Song/C++ test/main"

# 启动 C++ 可执行文件进程
process = subprocess.Popen([executable_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

# 持续输入和交互
try:
    while True:
        # 用户输入
        user_input = input("Enter a number (or 'q' to quit): ")

        # 退出条件
        if user_input == 'q':
            process.stdin.write(user_input + '\n')
            process.stdin.flush()
            print("Exiting interaction.")
            break

        # 向 C++ 程序传递输入
        process.stdin.write(user_input + '\n')
        process.stdin.flush()

        # 读取并打印 C++ 程序的输出
        output = process.stdout.readline().strip()
        print(f"Output from C++ program: {output}")

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # 确保进程关闭
    process.stdin.close()
    process.stdout.close()
    process.stderr.close()
    process.wait()
