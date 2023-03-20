import cv2
import numpy as np
# depth_image_path = './2011_10_03_drive_0047_sync_velodyne_raw_0000000785_image_02.png'
def depth2color(depth_image):
    # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    print(depth_image.max())
    # depth_image_scaled = depth_image / 256
    # print(depth_image_scaled.max())

    # 缩放深度图像到0-255的范围，并转换为8位无符号整数类型
    scaled_depth = cv2.convertScaleAbs(depth_image, alpha=0.03)
    print(scaled_depth.max())
    # 应用颜色映射，例如cv2.COLORMAP_JET
    color_image = cv2.applyColorMap(scaled_depth, cv2.COLORMAP_JET)

    # scaled_depth_scaled = cv2.convertScaleAbs(depth_image_scaled, alpha=0.03)
    # 应用颜色映射，例如cv2.COLORMAP_JET
    # color_image_scaled = cv2.applyColorMap(scaled_depth_scaled, cv2.COLORMAP_JET)

    # 显示或保存彩色图像
    cv2.imwrite("Color Image_998.png", color_image)
    # cv2.imwrite("Color Image_2.png", color_image_scaled)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
if __name__=="__main__":
    # npy_path = '0.npy'
    depth_image_path = './0000000998.png'
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    # depth_image = np.load(npy_path) * 1000
    depth2color(depth_image)
