import numpy as np
from matplotlib import pyplot as plt

def least_squares_method(v_x, v_y):
    """最小二乘法
    该方法仅支持二元一次方程 y = mx + c
    
    Arguments:
        v_x {ndarray 1D} -- 样本数据中的特征值
        v_y {ndarray 1d} -- 样本数据中的真实值
    
    Returns:
        float -- 斜率 m
        float -- 截距 c
    """
    # compute mean
    mean_x = v_x.mean()
    mean_y = v_y.mean()
    # compute slop
    temp1 = ((v_x - mean_x) * (v_y - mean_y)).sum()
    temp2 = ((v_x - mean_x) ** 2).sum()
    slop = temp1 / temp2
    # compute intercept
    intercept = mean_y - slop * mean_x
    # return
    return slop, intercept

def plot_line(x, y, slop, intercept, **kwargs):
    x2 = np.linspace(x.min(), x.max(), 100)
    y2 = x2 * slop + intercept

    plt.plot(x2, y2, **kwargs)   




if __name__ == "__main__":
    x = np.array([8, 2, 11, 6, 5, 4, 12, 9, 6, 1])
    y = np.array([3, 10, 3, 6, 8, 12, 1, 4, 9, 14])
    plt.plot(x, y, 'o')

    # 原始数据
    m, c = least_squares_method(x, y)
    plot_line(x, y, m, c, label='Origin')
    # 增加平均分布的噪音
    m, c = least_squares_method(x + np.random.rand(x.shape[0]), y)
    plot_line(x, y, m, c, label='Uniform noise')
    # 增加正态分布的噪音
    m, c = least_squares_method(x + np.random.randn(x.shape[0]), y)
    plot_line(x, y, m, c, label='Normal noise')

    plt.legend()
    plt.show()

    print(f'slop is {m}, intercept is {c}')