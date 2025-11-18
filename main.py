import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# 导入 PIL (Pillow) 库
# OpenCV (cv2) 原生不支持绘制中文，所以我们需要 PIL 来把图片转为 PIL 格式绘制中文，再转回 OpenCV 格式
from PIL import Image, ImageDraw, ImageFont


# ---------------------- 核心显示组件类 ----------------------
class ImageLabel(QLabel):
    """
    继承自 QLabel，用于显示图片并处理鼠标交互。
    这是程序的核心逻辑部分，包含了标定、测量、绘图的所有功能。
    """

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)  # 图片居中显示
        self.setMinimumSize(800, 600)  # 设置最小窗口大小
        self.setStyleSheet("background-color: #f0f0f0;")  # 设置灰色背景

        # --- 图像数据 ---
        self.origin_img = None  # 原始加载的图片（用于撤销或重置）
        self.current_img = None  # 当前处理中的图片（可能经过透视变换）
        self.pix_per_cm = None  # 核心比例尺：1厘米代表多少个像素

        # --- 状态标志 ---
        self.auto_detect_enabled = True  # 是否开启自动轮廓检测
        self.selecting_mode = None  # 当前鼠标模式："calib"(标定), "perspective"(透视), "manual"(手动)

        # --- 交互点存储 ---
        self.calib_pts = []  # 存储标定的两个点
        self.perspective_pts = []  # 存储透视矫正的四个角点
        self.manual_points = []  # 存储手动测量的多边形顶点

    def mousePressEvent(self, event):
        """
        鼠标点击事件处理函数。
        主要负责将点击的 UI 坐标转换为 图片的实际像素坐标。
        """
        # 如果没有图片或不是左/右键点击，直接忽略
        if self.current_img is None or event.button() not in (Qt.LeftButton, Qt.RightButton):
            return

        # 为空则忽略
        if not self.pixmap():
            return

        # --- 坐标映射逻辑 (关键) ---
        # QLabel 显示图片时可能会缩放 (Scaled)，所以必须计算缩放比例和偏移量
        pw = self.pixmap().width()  # 显示在屏幕上的图片宽度
        ph = self.pixmap().height()  # 显示在屏幕上的图片高度
        img_w, img_h = self.current_img.shape[1], self.current_img.shape[0]  # 图片实际分辨率

        scale_x = img_w / pw  # X轴缩放比例
        scale_y = img_h / ph  # Y轴缩放比例

        # 计算图片在 Label 中居中显示时的偏移量
        offset_x = (self.width() - pw) / 2
        offset_y = (self.height() - ph) / 2

        # 将鼠标点击坐标 (event.pos) 转换回 图片像素坐标 (x, y)
        x = int((event.pos().x() - offset_x) * scale_x)
        y = int((event.pos().y() - offset_y) * scale_y)

        # 限制坐标范围，防止点击到图片外区域导致越界
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))

        # --- 右键逻辑：结束手动测量 ---
        if event.button() == Qt.RightButton and self.selecting_mode == "manual":
            if len(self.manual_points) >= 3:
                self.finish_manual_measurement()
            return

        # --- 根据当前模式处理左键点击 ---
        if self.selecting_mode == "calib":
            self.calib_pts.append((x, y))  # 记录标定点
            self.update_display()  # 刷新显示（画点）
            if len(self.calib_pts) == 2:  # 如果选够了2个点，结束标定
                self.finish_calibration()

        elif self.selecting_mode == "perspective":
            self.perspective_pts.append((x, y))
            self.update_display()
            if len(self.perspective_pts) == 4:  # 如果选够了4个点，执行透视变换
                self.apply_perspective_correction()

        elif self.selecting_mode == "manual":
            self.manual_points.append((x, y))
            self.update_display()
            # 如果双击（其实是通过判断最后两点距离极近模拟双击或闭合），则结束
            if len(self.manual_points) > 1 and np.linalg.norm(
                    np.array(self.manual_points[-1]) - np.array(self.manual_points[-2])) < 20:
                self.manual_points.pop()  # 移除重复的最后一点
                self.finish_manual_measurement()

    # ---------------------- 标定功能 ----------------------
    def start_calibration(self):
        """进入标定模式：用户点击两个点来定义真实距离"""
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        self.selecting_mode = "calib"  # 更改鼠标模式
        self.calib_pts = []
        self.update_display()
        QMessageBox.information(self, "两点标定", "请点击已知长度的两个端点（如A4纸短边21cm）")

    def finish_calibration(self):
        """计算像素与现实长度的比例关系"""
        # 计算两点在图片上的像素距离
        dist_px = np.linalg.norm(np.array(self.calib_pts[0]) - np.array(self.calib_pts[1]))
        if dist_px < 20:
            QMessageBox.warning(self, "错误", "两点太近，请重新选择")
            self.calib_pts = []
            self.update_display()
            return

        # 弹出输入框让用户输入真实距离
        real, ok = QInputDialog.getDouble(self, "输入真实长度", "实际长度（厘米）:", 21.0, 0.1, 1000, 2)
        if ok:
            self.pix_per_cm = dist_px / real  # 计算比例尺：1cm = 多少像素
            QMessageBox.information(self, "成功", f"标定完成！1cm ≈ {self.pix_per_cm:.2f}像素")

        # 重置状态
        self.calib_pts = []
        self.selecting_mode = None
        self.update_display()

    # ---------------------- 透视矫正功能 ----------------------
    def start_perspective(self):
        """进入透视矫正模式：用于把倾斜拍摄的图片拉正"""
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        self.selecting_mode = "perspective"
        self.perspective_pts = []
        self.update_display()
        QMessageBox.information(self, "透视矫正", "请按顺序点击四个角：左上→右上→右下→左下")

    def apply_perspective_correction(self):
        """执行透视变换算法"""
        src = np.float32(self.perspective_pts)

        # 计算变换后目标图片的宽和高（取最大边长，减少变形）
        # 宽度 = max(上边长, 下边长)
        w = max(int(np.linalg.norm(src[0] - src[1])), int(np.linalg.norm(src[2] - src[3])))
        # 高度 = max(左边长, 右边长)
        h = max(int(np.linalg.norm(src[0] - src[3])), int(np.linalg.norm(src[1] - src[2])))

        if w < 100 or h < 100:
            QMessageBox.warning(self, "错误", "区域太小，请重新选择")
            self.perspective_pts = []
            self.update_display()
            return

        # 定义目标图片的四个角点坐标（矩形）
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # 获取变换矩阵 M
        M = cv2.getPerspectiveTransform(src, dst)
        # 执行变换
        result = cv2.warpPerspective(self.current_img, M, (w, h))

        self.current_img = result  # 更新当前显示的图片
        self.pix_per_cm = None  # 图片变了，之前的标定失效，必须重置

        QMessageBox.information(self, "完成", "透视矫正成功！请重新标定距离")
        self.perspective_pts = []
        self.selecting_mode = None
        self.update_display()

    # ---------------------- 手动测量功能 ----------------------
    def start_manual(self):
        """进入手动测量模式：用户可以手动绘制任意形状进行测量"""
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        self.selecting_mode = "manual"
        self.manual_points = []
        self.update_display()
        QMessageBox.information(self, "手动测量", "沿边缘点击，右键或双击结束")

    def finish_manual_measurement(self):
        """计算手动绘制多边形的周长和面积"""
        if len(self.manual_points) < 3:
            self.manual_points = []
            self.selecting_mode = None
            self.update_display()
            return

        cnt = np.array(self.manual_points).reshape((-1, 1, 2))  # 转为 OpenCV 轮廓格式
        peri = cv2.arcLength(cnt, True)  # 计算像素周长
        area = cv2.contourArea(cnt)  # 计算像素面积

        scale = self.pix_per_cm if self.pix_per_cm else 1
        unit = "cm" if self.pix_per_cm else "px"

        # 判断形状是否接近矩形 (公差 2%)
        shape = "矩形" if len(cv2.approxPolyDP(cnt, 0.02 * peri, True)) == 4 else "不规则"

        # 弹窗显示结果（面积需要除以比例尺的平方）
        QMessageBox.information(self, "结果",
                                f"形状：{shape}\n周长：{peri / scale:.2f} {unit}\n面积：{area / (scale ** 2):.2f} {unit}²")

        self.manual_points = []
        self.selecting_mode = None
        self.update_display()

    # ---------------------- 绘图与刷新核心 ----------------------
    def update_display(self):
        """
        将当前图片 + 所有的标注/点/线 绘制出来并显示在界面上。
        每次鼠标点击或状态改变都会调用此函数。
        """
        # 如果当前没有加载图片则返回
        if self.current_img is None:
            self.setPixmap(QPixmap())
            return

        img = self.current_img.copy()  # 拷贝一份，避免直接修改原始数据

        # 1. 自动检测与标注逻辑
        if self.auto_detect_enabled:
            contours = detect_contours(img)
            for cnt in contours:
                if cv2.contourArea(cnt) < 1200: continue  # 忽略太小的噪点

                # 计算质心（用于放文字位置）
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 计算最小外接矩形
                rect = cv2.minAreaRect(cnt)
                w_px, h_px = rect[1]

                # 判断形状
                shape = "矩形" if len(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) == 4 else "其他"
                peri = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)

                # 生成显示文字
                if self.pix_per_cm:
                    text = f"{shape}\n{w_px / self.pix_per_cm:.1f}×{h_px / self.pix_per_cm:.1f}cm\n周长:{peri / self.pix_per_cm:.1f}cm\n面积:{area / (self.pix_per_cm ** 2):.1f}cm²"
                else:
                    text = f"{shape}\n未标定"

                # 绘制轮廓和文字
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
                self.draw_text(img, text, (cx - 80, cy - 60), color=(255, 0, 0), size=2)

        # 2. 绘制用户交互的点和线 (标定点、透视点、手动点)
        # 列表格式: (点集, 颜色BGR, 半径, 线宽)
        pts_list = [
            (self.calib_pts, (0, 255, 255), 12, 4),  # 黄色：标定
            (self.perspective_pts, (255, 150, 0), 14, 4),  # 橙色：透视
            (self.manual_points, (0, 0, 255), 10, -1)  # 红色：手动
        ]
        for pts, color, radius, thick in pts_list:
            for i, pt in enumerate(pts):
                cv2.circle(img, pt, radius, color, thick)  # 画圆点
                # 画数字序号
                cv2.putText(img, str(i + 1), (pt[0] + 15, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            if len(pts) > 1:
                # 画连线
                cv2.polylines(img, [np.array(pts, np.int32)], False, color, 4)

        # 3. 左上角显示当前标定状态，可用size参数调整大小
        if self.pix_per_cm:
            self.draw_text(img, f"已标定 1cm≈{self.pix_per_cm:.1f}px", (10, 40), color=(0, 255, 0), size=2)

        # 4. OpenCV(BGR) 转 Qt(RGB) 并显示
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        # 缩放到控件大小，保持比例，平滑缩放
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def draw_text(self, img, text, org, color=(255, 255, 255), size=0.8):
        """
        使用 PIL 绘制中文文本的辅助函数。
        OpenCV 的 putText 不支持中文。
        """
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font_size = int(30 * size)

        # 尝试加载中文字体
        try:
            font = ImageFont.truetype("simhei.ttf", font_size)  # Windows 默认黑体
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)  # macOS 默认
            except:
                font = ImageFont.load_default()  # 回退默认字体（可能无法显示中文）

        # 支持多行文本绘制
        lines = text.split('\n')
        for i, line in enumerate(lines):
            draw.text((org[0], org[1] + i * (font_size + 5)), line, font=font, fill=color)

        # 转回 OpenCV 格式
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------- 图像算法函数 ----------------------
def detect_contours(img):
    """
    图像分割与边缘检测流程：
    1. 转灰度
    2. 高斯模糊 (去噪)
    3. Canny (边缘提取)
    4. 闭运算 (连接断开的边缘)
    5. 寻找轮廓
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # 高斯模糊
    edges = cv2.Canny(blurred, 30, 120)  # 提取边缘
    kernel = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 提取轮廓
    # 过滤掉面积太小的噪点
    return [c for c in contours if cv2.contourArea(c) > 1200]


# ---------------------- 主窗口类 ----------------------
class MainWindow(QMainWindow):
    """
    主窗口类，负责界面布局和功能按钮的组织
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("物件测量工具")  # 窗口标题
        central = QWidget()
        self.setCentralWidget(central)

        # 实例化核心组件
        self.image_label = ImageLabel()

        # 创建按钮并绑定事件
        btn_open = QPushButton("打开图片")
        btn_open.clicked.connect(self.open_image)

        btn_pers = QPushButton("【可选】透视矫正")
        btn_pers.clicked.connect(self.image_label.start_perspective)

        btn_calib = QPushButton("两点标定距离")
        btn_calib.clicked.connect(self.image_label.start_calibration)

        btn_manual = QPushButton("手动测量任意形状")
        btn_manual.clicked.connect(self.image_label.start_manual)

        self.btn_auto = QPushButton("自动标注：开启")
        self.btn_auto.clicked.connect(self.toggle_auto)

        btn_clear = QPushButton("清空所有点")
        btn_clear.clicked.connect(lambda: self.clear_points())

        # 布局管理
        left = QVBoxLayout()  # 左侧按钮栏
        left.addWidget(btn_open)
        left.addWidget(btn_pers)
        left.addWidget(btn_calib)
        left.addWidget(btn_manual)
        left.addWidget(self.btn_auto)
        left.addWidget(btn_clear)
        left.addStretch()  # 底部弹簧，把按钮顶上去

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, 1)  # 图片占大部分空间
        main_layout.addLayout(left)
        central.setLayout(main_layout)
        self.resize(1400, 900)

    def open_image(self):
        """打开文件对话框加载图片"""
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                # 初始化新图片
                self.image_label.current_img = img.copy()
                self.image_label.origin_img = img.copy()
                self.image_label.pix_per_cm = None
                self.image_label.update_display()

    def toggle_auto(self):
        """切换自动检测开关"""
        self.image_label.auto_detect_enabled = not self.image_label.auto_detect_enabled
        self.btn_auto.setText("自动标注：" + ("开启" if self.image_label.auto_detect_enabled else "关闭"))
        self.image_label.update_display()

    def clear_points(self):
        """清空界面上的所有标定点和测量线"""
        self.image_label.calib_pts = []
        self.image_label.perspective_pts = []
        self.image_label.manual_points = []
        self.image_label.update_display()


if __name__ == "__main__":
    # 创建Qt应用和主窗口
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())