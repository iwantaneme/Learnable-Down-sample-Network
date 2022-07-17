import math
import numpy as np
import matplotlib.pyplot as plt


def normaldist():
    u = 0  # 均值μ
    u2 = 5
    sig = math.sqrt(1)  # 标准差δ
    x = np.linspace(u - 2 * sig, u + 2 * sig, 50)  # 定义域
    t = np.linspace(u2 - 2 * sig, u2 + 2 * sig, 50)
    y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)  # 定义曲线函数
    z = np.exp(-(t - u2) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)  # 定义曲线函数
    plt.plot(x, y, "g", linewidth=2, label="Real-World")  # 加载曲线
    plt.plot(t, z, "b", linewidth=2, label="bicubit")
    plt.legend()

    plt.yticks([])  # 去掉纵坐标值
    # plt.grid(True)  # 网格线
    plt.show()  # 显示


def logistic():
    a = np.linspace(-10, 10, 1000)
    b = 1.0 / (1.0 + np.exp(-a))
    plt.title('logitstic function')
    plt.plot(a, b)
    plt.grid(True)
    plt.show()

def ReLU():
    def ReLU(value):
        if value < 0:
            return 0
        else:
            return value

    plt.figure(figsize=(6, 4))
    x = np.linspace(-4, 4, 100)
    y = np.array([])
    for v in x:
        y = np.append(y, np.linspace(ReLU(v), ReLU(v), 1))
    plt.plot(x, y, 'b', label='ReLU function')
    plt.title('ReLU function')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    Str1 = "Canon_001_HR.png"
    print(Str1[10:12])