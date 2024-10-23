#include <iostream>
#include <string>

int main() {
    int input;
    std::string user_input;

    // 无限循环，持续接受输入
    while (true) {
        std::cin >> user_input;

        // 判断是否退出
        if (user_input == "q") {
            std::cout << "Exiting..." << std::endl;
            break;
        }

        // 尝试将输入转换为整数
        try {
            input = std::stoi(user_input);
        } catch (std::invalid_argument& e) {
            std::cout << "Invalid input, please enter a number or 'q' to quit." << std::endl;
            continue;
        }

        // 判断输入值是否为 1
        if (input == 1) {
            std::cout << "A" << std::endl;
        } else {
            std::cout << "B" << std::endl;
        }

        // 刷新输出，确保 Python 能立刻读取到
        std::cout.flush();
    }

    return 0;
}
