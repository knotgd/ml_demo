# @Time : 2020/6/16 19:42 
# @Author : 大太阳小白
# @Software: PyCharm
"""
画个旋转抛物面
z是高度，画出等高线
关于x和y的二维平面可以看成一个椭圆
求关于z的曲线偏导，每次沿梯度方向移动lambda不长，直至找到圆心附近
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
# 步长
delta = 0.025
x = np.arange(-2.0, 2.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = -((X - 1) ** 2)
Z2 = -((Y / 2) ** 2)
# 目标函数，想要学得的完美模型
Z = 1.0 * (Z2 + Z1) + 5.0
plt.figure()
CS = plt.contour(X, Y, Z)
# 开始迭代开始点
start_x = -1.8
start_y = -1.9
lam = 0.4
z_current = -(start_x - 1) ** 2 - (start_y / 2)**2 + 5
z_max = -(1 - 1) ** 2 - (0 / 2) + 5
print("开始z值", z_current, "z_max目标值", z_max)
w = np.ones((1, 2))
for index in range(10):
    plt.text(start_x, start_y, 'P{}'.format(index))
    end_x = -2 * start_x + 2
    end_y = -0.5 * start_y
    dist = (end_x ** 2 + end_y ** 2) ** 0.5
    end_x = start_x + lam * end_x / dist
    end_y = start_y + lam * end_y / dist
    w -=np.mat([lam * end_x / dist,lam * end_y / dist])
    plt.annotate('', xy=(start_x, start_y), xycoords='data', va="center", ha="center",
                 xytext=(end_x, end_y), textcoords='data', bbox=leafNode, arrowprops=arrow_args)
    start_x = end_x
    start_y = end_y
    z_current = -(start_x - 1) ** 2 - (start_y / 2)**2 + 5
    print("开始z值", z_current, "z_max目标值", z_max)
z = (np.mat([x,y]).T * w.T).T.tolist()
plt.plot(x,z[0])
plt.show()
