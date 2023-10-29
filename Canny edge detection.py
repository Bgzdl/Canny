import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Canny:
    def __init__(self, kernel_size=3, sigma=0.8, stride=1, threshold: tuple = (10, 20)):
        """
        :param kernel_size: int
        :param sigma: float
        :param stride: int
        :param threshold: (low_threshold: int, high_threshold: int)
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = stride
        self.threshold = threshold

    @staticmethod
    def _gray(image):
        """
        灰度化
        :param image: numpy array [h, w, 3]
        :return: gray image: numpy [h, w]
        """
        b = image[:, :, 0].copy()
        g = image[:, :, 1].copy()
        r = image[:, :, 2].copy()
        # 计算灰度图的像素值
        out = 0.299 * r + 0.578 * g + 0.114 * b
        out = out.astype(np.uint8)
        return out

    @staticmethod
    def _pad(image, pad_size: int):
        """
        边缘填充
        :param image: numpy array([h, w])
        :param pad_size: int
        :return: padded_image: numpy array([h+pad_size*2, w+pad_size*2])
        """
        padded_image = np.pad(image, pad_width=((pad_size, pad_size), (pad_size, pad_size)))
        return padded_image

    @staticmethod
    def _first_derivative_of_gaussian_kernel(kernel_size, sigma):
        """
        获得高斯滤波器的x方向和y方向的卷积核
        :param kernel_size: int
        :param sigma: float
        :return: kernel_x: numpy array([kernel_size, kernel_size])
        :return: kernel_y: numpy array([kernel_size, kernel_size])
        """
        kernel_x = np.zeros(shape=(kernel_size, kernel_size))
        kernel_y = np.zeros(shape=(kernel_size, kernel_size))
        # 计算核矩阵的中心位置
        mid = (kernel_size - 1) / 2
        # 计算高斯核函数的x和y导数
        for i in range(kernel_size):
            for j in range(kernel_size):
                # 获得中心偏移后的坐标
                y = i - mid
                x = j - mid
                kernel_y[i][j] = - y / (2 * np.pi * sigma ** 4) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
                kernel_x[i][j] = - x / (2 * np.pi * sigma ** 4) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
        return kernel_y, kernel_x

    @staticmethod
    def _first_derivative_of_gaussian_filter(image, kernel_x, kernel_y, stride):
        """
        对图片应用高斯一阶滤波器
        :param image: numpy array((h, w))
        :param kernel_x: numpy array([kernel_size, kernel_size])
        :param kernel_y: numpy array([kernel_size, kernel_size])
        :param stride: int
        :return: numpy array([h, w]), numpy array([h, w])
        """
        h, w = image.shape
        kernel_size = kernel_x.shape[0]
        gradient = np.zeros((h, w, 2))
        # 对输入图像进行边缘填充，使得图片在卷积后保留原大小
        padded_image = Canny._pad(image, int((kernel_size - 1) / 2))
        # 计算梯度图像中每一个像素点的梯度值
        for i in range(h):
            for j in range(w):
                gradient[i][j][0] = np.sum(kernel_x * padded_image[i:i + kernel_size, j:j + kernel_size])
                gradient[i][j][1] = np.sum(kernel_y * padded_image[i:i + kernel_size, j:j + kernel_size])
                # 根据步长调整下一个位置
                j += stride - 1
            i += stride - 1
        # 计算梯度图像的像素值并将像素值限制在[0,255]之间
        gradient_image = np.sqrt(gradient[:, :, 0] ** 2 + gradient[:, :, 1] ** 2)
        gradient_image = np.clip(gradient_image, 0, 255)
        # 计算梯度图像中每一个像素点的角度值并规范为[0,45,90,135]
        angle = np.arctan2(gradient[:, :, 1], gradient[:, :, 0])
        angle[angle < 0] = angle[angle < 0] + np.pi
        angle = np.rad2deg(angle)
        angle[angle < 22.5] = 0
        angle[angle > 157.5] = 0
        angle[(22.5 <= angle) & (angle < 67.5)] = 45
        angle[(67.5 <= angle) & (angle < 112.5)] = 90
        angle[(112.5 <= angle) & (angle < 157.5)] = 135
        return angle, gradient_image

    @staticmethod
    def _NMS(angle, gradient_image):
        """
        非极大值抑制
        :param angle: numpy array([h, w])
        :param gradient_image: numpy array([h, w])
        :return: numpy array([h, w])
        """
        h, w = angle.shape
        nms_image = gradient_image.copy()
        # 遍历梯度图像中的每一个像素点
        for y in range(h):
            for x in range(w):
                # 计算梯度方向的相对坐标
                if angle[y, x] == 0:
                    dx_1, dy_1, dx_2, dy_2 = -1, 0, 1, 0
                elif angle[y, x] == 45:
                    dx_1, dy_1, dx_2, dy_2 = -1, 1, 1, -1
                elif angle[y, x] == 90:
                    dx_1, dy_1, dx_2, dy_2 = 0, -1, 0, 1
                elif angle[y, x] == 135:
                    dx_1, dy_1, dx_2, dy_2 = -1, -1, 1, 1
                else:
                    raise Exception("Angle Error")
                # 边界修正
                if x == 0:
                    dx_1 = max(dx_1, 0)
                    dx_2 = max(dx_2, 0)
                elif x == w - 1:
                    dx_1 = min(dx_1, 0)
                    dx_2 = min(dx_2, 0)
                else:
                    pass
                if y == 0:
                    dy_1 = max(dy_1, 0)
                    dy_2 = max(dy_2, 0)
                elif y == h - 1:
                    dy_1 = min(dy_1, 0)
                    dy_2 = min(dy_2, 0)
                else:
                    pass
                # 如果最大值不是当前点，则将当前点置为0
                max_value = np.max(
                    [gradient_image[y][x], gradient_image[y + dy_1][x + dx_1], gradient_image[y + dy_2][x + dx_2]])
                if max_value != gradient_image[y][x]:
                    nms_image[y][x] = 0
        return nms_image

    @staticmethod
    def _hysteresis(nms_image, threshold: tuple):
        """
        双阈值法，将图像分为前景和背景两部分，前景点的灰度值大于或等于阈值，背景点的灰度值小于等于阈值
        然后用深度优先算法搜寻所有的弱边缘点
        :param nms_image: numpy array([H, W])
        :param threshold: (low_threshold: int, high_threshold: int)
        :return: numpy array([H, W])
        """
        visited = np.zeros_like(nms_image)
        HT_image = nms_image.copy()
        H, W = HT_image.shape

        # 深度优先算法访问周围的8个点
        def dfs(i, j):
            if i >= H or i < 0 or j >= W or j < 0 or visited[i, j] == 1:
                return
            visited[i, j] = 1
            if HT_image[i, j] > threshold[0]:
                HT_image[i, j] = 255
                dfs(i - 1, j - 1)
                dfs(i - 1, j)
                dfs(i - 1, j + 1)
                dfs(i, j - 1)
                dfs(i, j + 1)
                dfs(i + 1, j - 1)
                dfs(i + 1, j)
                dfs(i + 1, j + 1)
            else:
                HT_image[i, j] = 0

        # 强边缘点以及其周边弱边缘点的检测
        for h in range(H):
            for w in range(W):
                if visited[h, w] == 1:
                    continue
                if HT_image[h, w] >= threshold[1]:
                    dfs(h, w)
                elif HT_image[h, w] <= threshold[0]:
                    HT_image[h, w] = 0
                    visited[h, w] = 1
        # 未访问的的点设为0
        for h in range(H):
            for w in range(W):
                if visited[h, w] == 0:
                    HT_image[h, w] = 0
        return HT_image

    def Canny(self, image):
        """
        Canny边缘检测
        :param image: numpy array([h, w, channels])
        :param kernel_size: int
        :param sigma: float
        :param stride: int
        :param threshold: (low_threshold: int, high_threshold: int)
        :return: canny_image: numpy array([h, w])
        """
        # 灰度化图像
        gray_image = Canny._gray(image)
        # 获得一阶高斯滤波器
        kernel_y, kernel_x = Canny._first_derivative_of_gaussian_kernel(self.kernel_size, self.sigma)
        # 获得图片的梯度信息
        angle, gradient_image = Canny._first_derivative_of_gaussian_filter(
            gray_image, kernel_x, kernel_y, self.stride)
        # 对梯度图片应用非极大值抑制
        nms_image = Canny._NMS(angle, gradient_image)
        # 阈值分割和滞后补偿
        canny_image = Canny._hysteresis(nms_image, self.threshold)
        return canny_image


if __name__ == '__main__':
    img = np.array(Image.open('./1.jpg'))
    canny = Canny(kernel_size=3, sigma=0.8, stride=1, threshold=(10, 20))
    canny_image = canny.Canny(img)
    plt.imshow(canny_image, cmap='gray')
    plt.axis('off')
    plt.show()
