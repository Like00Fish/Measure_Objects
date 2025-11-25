import sys
import cv2
import numpy as np
import math
import json
from datetime import datetime

# ---------------------- 导入 UI 库 ----------------------
try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtGui import *
    from PyQt6.QtCore import *
except ImportError:
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *

from PIL import Image, ImageDraw, ImageFont


# ---------------------- 核心显示组件类 ----------------------
class ImageLabel(QLabel):
    measurementAdded = pyqtSignal()
    calibrationFinished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #cccccc;
                border-radius: 5px;
            }
        """)

        # --- 图像数据 ---
        self.origin_img = None
        self.current_img = None
        self.processed_img = None
        self.pix_per_cm = None

        # --- 状态标志 ---
        self.auto_detect_enabled = True
        self.selecting_mode = None
        self.measure_mode = "manual"

        # --- 交互点存储 ---
        self.calib_pts = []
        self.perspective_pts = []
        self.manual_points = []

        # --- 测量结果存储 ---
        self.measurement_results = []
        self.measurement_count = 0

        # --- 画笔模式相关 ---
        self.brush_mode_enabled = False
        self.brush_points = []  # 当前笔画点列表
        self.brush_is_drawing = False
        self.brush_strokes = []  # 已完成的笔画，每个为点列表
        self.brush_mode_type = "curve"  # "curve" 或 "line"
        self.brush_longpress_timer = QTimer(self)
        self.brush_longpress_timer.setSingleShot(True)
        self.brush_longpress_timer.timeout.connect(self._on_brush_longpress)
        self._press_pos = None

    def set_image(self, img):
        """设置并显示图像（外部调用）"""
        self.origin_img = img.copy()
        self.current_img = img.copy()
        self.pix_per_cm = None
        self.clear_points()
        self.update_display()

    def mousePressEvent(self, event):
        if self.current_img is None or event.button() not in (Qt.LeftButton, Qt.RightButton):
            return

        if not self.pixmap():
            return

        # --- 坐标映射逻辑 ---
        pw = self.pixmap().width()
        ph = self.pixmap().height()
        img_w, img_h = self.current_img.shape[1], self.current_img.shape[0]

        scale_x = img_w / pw
        scale_y = img_h / ph

        offset_x = (self.width() - pw) / 2
        offset_y = (self.height() - ph) / 2

        x = int((event.pos().x() - offset_x) * scale_x)
        y = int((event.pos().y() - offset_y) * scale_y)

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))

        # --- 右键逻辑：结束手动测量或结束画笔绘制 ---
        if event.button() == Qt.RightButton:
            if self.selecting_mode == "manual":
                if len(self.manual_points) >= 3:
                    self.finish_manual_measurement()
                return
            if self.brush_mode_enabled and self.brush_is_drawing:
                # 右键结束当前笔画
                self._finish_brush_stroke()
                return

        # --- 画笔模式长按检测 ---
        if self.brush_mode_enabled and event.button() == Qt.LeftButton:
            # 记录按下位置并启动长按计时器
            self._press_pos = (x, y)
            self.brush_longpress_timer.start(250)  # 250ms 视为长按开始绘制
            return

        # 如果画笔模式未启用或未触发长按，则继续原有交互
        # --- 根据当前模式处理左键点击 ---
        if self.selecting_mode == "calib":
            self.calib_pts.append((x, y))
            self.update_display()
            if len(self.calib_pts) == 2:
                self.finish_calibration()

        elif self.selecting_mode == "perspective":
            self.perspective_pts.append((x, y))
            self.update_display()
            if len(self.perspective_pts) == 4:
                self.apply_perspective_correction()

        elif self.selecting_mode == "manual":
            self.manual_points.append((x, y))
            self.update_display()
            if len(self.manual_points) > 1 and np.linalg.norm(
                    np.array(self.manual_points[-1]) - np.array(self.manual_points[-2])) < 20:
                self.manual_points.pop()
                self.finish_manual_measurement()

    def mouseMoveEvent(self, event):
        if not self.brush_mode_enabled:
            return

        if not self.pixmap():
            return

        if not self.brush_is_drawing:
            return

        # 坐标映射
        pw = self.pixmap().width()
        ph = self.pixmap().height()
        img_w, img_h = self.current_img.shape[1], self.current_img.shape[0]

        scale_x = img_w / pw
        scale_y = img_h / ph

        offset_x = (self.width() - pw) / 2
        offset_y = (self.height() - ph) / 2

        x = int((event.pos().x() - offset_x) * scale_x)
        y = int((event.pos().y() - offset_y) * scale_y)

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))

        # 添加点到当前笔画
        last = self.brush_points[-1] if self.brush_points else None
        if last is None or np.linalg.norm(np.array((x, y)) - np.array(last)) > 2:
            self.brush_points.append((x, y))
            self.update_display()

    def mouseReleaseEvent(self, event):
        if self.brush_mode_enabled and event.button() == Qt.LeftButton:
            # 如果长按计时器还在，说明未触发长按，取消计时器
            if self.brush_longpress_timer.isActive():
                self.brush_longpress_timer.stop()
                self._press_pos = None
                return

            # 如果正在绘制，结束当前笔画
            if self.brush_is_drawing:
                self._finish_brush_stroke()
                return

    # ---------------------- 图像处理功能 ----------------------
    def apply_denoise(self, method, params):
        """应用去噪处理"""
        if self.current_img is None:
            return False

        try:
            if method == "高斯滤波":
                ksize = params.get("ksize", 5)
                if ksize % 2 == 0:
                    ksize += 1
                self.current_img = cv2.GaussianBlur(self.current_img, (ksize, ksize), 0)

            elif method == "中值滤波":
                ksize = params.get("ksize", 5)
                self.current_img = cv2.medianBlur(self.current_img, ksize)

            elif method == "双边滤波":
                d = params.get("d", 9)
                sigma_color = params.get("sigma_color", 75)
                sigma_space = params.get("sigma_space", 75)
                self.current_img = cv2.bilateralFilter(self.current_img, d, sigma_color, sigma_space)

            self.update_display()
            return True
        except Exception as e:
            QMessageBox.warning(self, "错误", f"去噪处理失败: {str(e)}")
            return False

    def apply_enhancement(self, method, params):
        """应用图像增强"""
        if self.current_img is None:
            return False

        try:
            if method == "直方图均衡化":
                img_yuv = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                self.current_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            elif method == "对比度拉伸":
                alpha = params.get("alpha", 1.5)
                beta = params.get("beta", 0)
                self.current_img = cv2.convertScaleAbs(self.current_img, alpha=alpha, beta=beta)

            elif method == "伽马校正":
                gamma = params.get("gamma", 1.2)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                self.current_img = cv2.LUT(self.current_img, table)

            self.update_display()
            return True
        except Exception as e:
            QMessageBox.warning(self, "错误", f"图像增强失败: {str(e)}")
            return False

    def reset_image(self):
        """重置图像到原始状态"""
        if self.origin_img is not None:
            self.current_img = self.origin_img.copy()
            self.clear_points()
            self.update_display()
            QMessageBox.information(self, "提示", "图像已重置为原始状态")

    def save_image(self):
        """保存处理后的图像"""
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "没有图像可保存")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "图片 (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                cv2.imwrite(path, self.current_img)
                QMessageBox.information(self, "成功", f"图像已保存到: {path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")

    def save_measurements(self):
        """保存测量结果"""
        if not self.measurement_results:
            QMessageBox.warning(self, "提示", "没有测量结果可保存")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存测量结果", "", "文本文件 (*.txt);;JSON文件 (*.json)")
        if path:
            try:
                if path.endswith('.json'):
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(self.measurement_results, f, ensure_ascii=False, indent=2)
                else:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write("工件测量结果报告\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"标定比例: {self.pix_per_cm:.2f} px/cm\n" if self.pix_per_cm else "未标定\n")
                        f.write("-" * 50 + "\n")

                        for i, result in enumerate(self.measurement_results, 1):
                            f.write(f"测量 #{i}:\n")
                            for key, value in result.items():
                                f.write(f"  {key}: {value}\n")
                            f.write("\n")

                QMessageBox.information(self, "成功", f"测量结果已保存到: {path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")

    # ---------------------- 标定功能 ----------------------
    def start_calibration(self):
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        self.selecting_mode = "calib"
        self.calib_pts = []
        self.update_display()
        QMessageBox.information(self, "两点标定", "请点击已知长度的两个端点（如A4纸短边21cm）")

    def finish_calibration(self):
        dist_px = np.linalg.norm(np.array(self.calib_pts[0]) - np.array(self.calib_pts[1]))
        if dist_px < 20:
            QMessageBox.warning(self, "错误", "两点太近，请重新选择")
            self.calib_pts = []
            self.update_display()
            return

        real, ok = QInputDialog.getDouble(self, "输入真实长度", "实际长度（厘米）:", 21.0, 0.1, 1000, 2)
        if ok:
            self.pix_per_cm = dist_px / real
            QMessageBox.information(self, "成功", f"标定完成！1cm ≈ {self.pix_per_cm:.2f}像素")

        self.calib_pts = []
        self.selecting_mode = None
        self.update_display()
        self.calibrationFinished.emit()

    # ---------------------- 透视矫正功能 ----------------------
    def start_perspective(self):
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        self.selecting_mode = "perspective"
        self.perspective_pts = []
        self.update_display()
        QMessageBox.information(self, "透视矫正", "请按顺序点击四个角：左上→右上→右下→左下")

    def apply_perspective_correction(self):
        src = np.float32(self.perspective_pts)
        w = max(int(np.linalg.norm(src[0] - src[1])), int(np.linalg.norm(src[2] - src[3])))
        h = max(int(np.linalg.norm(src[0] - src[3])), int(np.linalg.norm(src[1] - src[2])))

        if w < 100 or h < 100:
            QMessageBox.warning(self, "错误", "区域太小，请重新选择")
            self.perspective_pts = []
            self.update_display()
            return

        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(self.current_img, M, (w, h))

        self.current_img = result
        self.pix_per_cm = None

        QMessageBox.information(self, "完成", "透视矫正成功！请重新标定距离")
        self.perspective_pts = []
        self.selecting_mode = None
        self.update_display()

    # ---------------------- 测量功能 ----------------------
    def start_manual(self):
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        self.selecting_mode = "manual"
        self.manual_points = []
        self.update_display()
        QMessageBox.information(self, "手动测量", "沿边缘点击，右键或双击结束")

    def finish_manual_measurement(self):
        if len(self.manual_points) < 3:
            self.manual_points = []
            self.selecting_mode = None
            self.update_display()
            return

        cnt = np.array(self.manual_points).reshape((-1, 1, 2))
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)

        scale = self.pix_per_cm if self.pix_per_cm else 1
        unit = "cm" if self.pix_per_cm else "px"

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            shape = "矩形"
        elif len(approx) >= 8:
            shape = "圆形"
        else:
            shape = "不规则形状"

        self.measurement_count += 1
        result = {
            "序号": self.measurement_count,
            "形状": shape,
            "周长": f"{peri / scale:.2f} {unit}",
            "面积": f"{area / (scale ** 2):.2f} {unit}²",
            "测量时间": datetime.now().strftime("%H:%M:%S")
        }
        self.measurement_results.append(result)
        # 发出信号通知主窗口刷新
        try:
            self.measurementAdded.emit()
        except:
            pass

        QMessageBox.information(self, "结果",
                                f"形状：{shape}\n周长：{peri / scale:.2f} {unit}\n面积：{area / (scale ** 2):.2f} {unit}²")

        self.manual_points = []
        self.selecting_mode = None
        self.update_display()

    def auto_measure_rectangles(self):
        """自动测量矩形"""
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return

        if not self.pix_per_cm:
            QMessageBox.warning(self, "提示", "请先进行标定")
            return

        contours = detect_contours(self.current_img)
        rectangles = []

        for cnt in contours:
            if cv2.contourArea(cnt) < 1200:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                width_px = min(rect[1])
                height_px = max(rect[1])
                area_px = cv2.contourArea(cnt)

                width_cm = width_px / self.pix_per_cm
                height_cm = height_px / self.pix_per_cm
                area_cm = area_px / (self.pix_per_cm ** 2)

                rectangles.append({
                    "contour": cnt,
                    "box": box,
                    "width": width_cm,
                    "height": height_cm,
                    "area": area_cm
                })

                self.measurement_count += 1
                result = {
                    "序号": self.measurement_count,
                    "形状": "矩形",
                    "宽度": f"{width_cm:.2f} cm",
                    "高度": f"{height_cm:.2f} cm",
                    "面积": f"{area_cm:.2f} cm²",
                    "测量时间": datetime.now().strftime("%H:%M:%S")
                }
                self.measurement_results.append(result)

        # 发出信号通知主窗口刷新（如果有新增）
        try:
            if rectangles:
                self.measurementAdded.emit()
        except:
            pass

        img = self.current_img.copy()
        for rect in rectangles:
            cv2.drawContours(img, [rect['box']], 0, (0, 255, 0), 3)

            M = cv2.moments(rect['contour'])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                text = f"矩形\n{rect['width']:.1f}×{rect['height']:.1f}cm\n面积:{rect['area']:.1f}cm²"
                self.draw_text(img, text, (cx - 80, cy - 60), color=(255, 0, 0), size=1.5)

        self.current_img = img
        self.update_display()

        QMessageBox.information(self, "完成", f"检测到 {len(rectangles)} 个矩形")

    def auto_measure_circles(self):
        """自动测量圆形"""
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return

        if not self.pix_per_cm:
            QMessageBox.warning(self, "提示", "请先进行标定")
            return

        gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                   param1=100, param2=30, minRadius=20, maxRadius=200)

        img = self.current_img.copy()
        detected_circles = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius_px = i[2]

                radius_cm = radius_px / self.pix_per_cm
                diameter_cm = 2 * radius_cm
                area_cm = math.pi * (radius_cm ** 2)

                cv2.circle(img, center, radius_px, (0, 255, 0), 3)
                cv2.circle(img, center, 2, (0, 0, 255), 3)

                text = f"圆形\n直径:{diameter_cm:.1f}cm\n面积:{area_cm:.1f}cm²"
                self.draw_text(img, text, (center[0] - 60, center[1] - radius_px - 40), color=(255, 0, 0), size=1.5)

                self.measurement_count += 1
                result = {
                    "序号": self.measurement_count,
                    "形状": "圆形",
                    "直径": f"{diameter_cm:.2f} cm",
                    "半径": f"{radius_cm:.2f} cm",
                    "面积": f"{area_cm:.2f} cm²",
                    "测量时间": datetime.now().strftime("%H:%M:%S")
                }
                self.measurement_results.append(result)

                detected_circles += 1

        # 发出信号通知主窗口刷新（如果有新增）
        try:
            if detected_circles > 0:
                self.measurementAdded.emit()
        except:
            pass

        self.current_img = img
        self.update_display()

        QMessageBox.information(self, "完成", f"检测到 {detected_circles} 个圆形")

    def clear_measurements(self):
        """清除测量结果"""
        self.measurement_results = []
        self.measurement_count = 0
        if self.origin_img is not None:
            self.current_img = self.origin_img.copy()
        self.brush_strokes = []
        self.update_display()
        QMessageBox.information(self, "提示", "测量结果已清除")
        try:
            self.measurementAdded.emit()
        except:
            pass

    def clear_points(self):
        """清空所有点"""
        self.calib_pts = []
        self.perspective_pts = []
        self.manual_points = []
        # 清除画笔数据
        self.brush_points = []
        self.brush_strokes = []
        self.brush_is_drawing = False
        self.update_display()

    # ---------------------- 画笔相关方法 ----------------------
    def _on_brush_longpress(self):
        # 长按触发，开始绘制
        if self._press_pos is None:
            return
        self.brush_is_drawing = True
        self.brush_points = [self._press_pos]
        self._press_pos = None
        self.update_display()

    def _finish_brush_stroke(self):
        if not self.brush_is_drawing:
            return
        # 如果是直线模式，将点简化为两点直线
        if self.brush_mode_type == "line" and len(self.brush_points) >= 2:
            self.brush_points = [self.brush_points[0], self.brush_points[-1]]

        # 保存笔画
        if len(self.brush_points) >= 2:
            self.brush_strokes.append(self.brush_points.copy())

            # 计算长度
            length_px = 0.0
            for i in range(1, len(self.brush_points)):
                length_px += np.linalg.norm(np.array(self.brush_points[i]) - np.array(self.brush_points[i - 1]))
            length = length_px / (self.pix_per_cm if self.pix_per_cm else 1)
            unit = "cm" if self.pix_per_cm else "px"

            # 检查闭合并计算面积
            if len(self.brush_points) >= 3 and np.linalg.norm(
                    np.array(self.brush_points[0]) - np.array(self.brush_points[-1])) < 10:
                # 计算多边形面积（像素）
                poly = np.array(self.brush_points, dtype=np.int32)
                area_px = cv2.contourArea(poly)
                area = area_px / ((self.pix_per_cm if self.pix_per_cm else 1) ** 2)

                # 记录测量结果（闭合）
                self.measurement_count += 1
                result = {
                    "序号": self.measurement_count,
                    "形状": "手绘闭合",
                    "周长": f"{length:.2f} {unit}",
                    "面积": f"{area:.2f} {unit}²",
                    "测量时间": datetime.now().strftime("%H:%M:%S")
                }
                self.measurement_results.append(result)
                try:
                    self.measurementAdded.emit()
                except:
                    pass
            else:
                # 仅记录长度（曲线或直线）
                self.measurement_count += 1
                result = {
                    "序号": self.measurement_count,
                    "形状": "手绘曲线" if self.brush_mode_type == "curve" else "手绘直线",
                    "长度": f"{length:.2f} {unit}",
                    "测量时间": datetime.now().strftime("%H:%M:%S")
                }
                self.measurement_results.append(result)
                try:
                    self.measurementAdded.emit()
                except:
                    pass

        # 清理当前笔画状态
        self.brush_points = []
        self.brush_is_drawing = False
        self.update_display()

    # ---------------------- 绘图与刷新核心 ----------------------
    def update_display(self):
        if self.current_img is None:
            self.setPixmap(QPixmap())
            return

        img = self.current_img.copy()

        # 1. 自动检测与标注逻辑
        if self.auto_detect_enabled:
            contours = detect_contours(img)
            for cnt in contours:
                if cv2.contourArea(cnt) < 1200:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                rect = cv2.minAreaRect(cnt)
                w_px, h_px = rect[1]

                shape = "矩形" if len(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) == 4 else "其他"
                peri = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)

                if self.pix_per_cm:
                    text = f"{shape}\n{w_px / self.pix_per_cm:.1f}×{h_px / self.pix_per_cm:.1f}cm\n周长:{peri / self.pix_per_cm:.1f}cm\n面积:{area / (self.pix_per_cm ** 2):.1f}cm²"
                else:
                    text = f"{shape}\n未标定"

                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
                self.draw_text(img, text, (cx - 80, cy - 60), color=(255, 0, 0), size=2)

        # 2. 绘制用户交互的点和线
        pts_list = [
            (self.calib_pts, (0, 255, 255), 12, 4),
            (self.perspective_pts, (255, 150, 0), 14, 4),
            (self.manual_points, (0, 0, 255), 10, -1)
        ]
        for pts, color, radius, thick in pts_list:
            for i, pt in enumerate(pts):
                cv2.circle(img, pt, radius, color, thick)
                cv2.putText(img, str(i + 1), (pt[0] + 15, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, color, 4)

        # 画笔笔画绘制
        # 绘制已完成的笔画
        for stroke in self.brush_strokes:
            pts = np.array(stroke, np.int32)
            if len(pts) >= 2:
                closed = True if (
                        len(pts) >= 3 and np.linalg.norm(np.array(pts[0]) - np.array(pts[-1])) < 10) else False
                cv2.polylines(img, [pts], closed, (255, 0, 255), 3)
                # 在笔画中心显示长度
                mid_idx = len(pts) // 2
                length_px = sum(np.linalg.norm(np.array(pts[i]) - np.array(pts[i - 1])) for i in range(1, len(pts)))
                length = length_px / (self.pix_per_cm if self.pix_per_cm else 1)
                unit = "cm" if self.pix_per_cm else "px"
                self.draw_text(img, f"Len:{length:.1f}{unit}", (pts[mid_idx][0] - 40, pts[mid_idx][1] - 20),
                               color=(255, 0, 255), size=0.9)

        # 绘制当前正在绘制的笔画
        if self.brush_points:
            pts = np.array(self.brush_points, np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, (200, 0, 200), 2)

        # 3. 显示标定状态和测量次数
        status_text = []
        if self.pix_per_cm:
            status_text.append(f"已标定 1cm≈{self.pix_per_cm:.1f}px")
        status_text.append(f"测量次数: {self.measurement_count}")

        for i, text in enumerate(status_text):
            self.draw_text(img, text, (10, 40 + i * 40), color=(0, 255, 0), size=1.2)

        # 4. OpenCV转Qt显示
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def draw_text(self, img, text, org, color=(255, 255, 255), size=0.8):
        """使用PIL绘制中文文本"""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font_size = int(30 * size)

        try:
            font = ImageFont.truetype("simhei.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
            except:
                font = ImageFont.load_default()

        lines = text.split('\n')
        for i, line in enumerate(lines):
            draw.text((org[0], org[1] + i * (font_size + 5)), line, font=font, fill=tuple(color))

        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------- 图像算法函数 ----------------------
def detect_contours(img):
    """
    图像分割与边缘检测流程
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > 1200]


# ---------------------- 主窗口类 ----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工件测量系统")
        self.setWindowIcon(QIcon(self.create_icon()))
        central = QWidget()
        self.setCentralWidget(central)

        # 实例化核心组件
        self.image_label = ImageLabel()
        # 连接测量结果信号到刷新表格
        try:
            self.image_label.measurementAdded.connect(self.refresh_results_table)
            self.image_label.calibrationFinished.connect(self.refresh_calibration_status)
        except:
            pass

        # 创建控件
        self.create_controls()

        # 布局管理 - 使用更紧凑的布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, 4)  # 图像区域占4份空间

        # 右侧控制面板 - 使用滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(400)  # 限制最大宽度
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(10)  # 减少间距

        # 添加控件组
        control_layout.addWidget(self.file_group)
        control_layout.addWidget(self.process_group)
        control_layout.addWidget(self.calib_group)
        control_layout.addWidget(self.measure_group)
        control_layout.addWidget(self.results_group)
        control_layout.addStretch()

        scroll_area.setWidget(control_widget)
        main_layout.addWidget(scroll_area, 1)  # 控制面板占1份空间

        central.setLayout(main_layout)

        # 设置合适的初始大小
        self.resize(1200, 800)
        self.apply_styles()

    def create_icon(self):
        """创建应用程序图标"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(70, 130, 180)))
        painter.setPen(QPen(QColor(30, 60, 120), 2))
        painter.drawRect(4, 4, 24, 24)
        painter.drawLine(4, 4, 28, 28)
        painter.end()
        return pixmap

    def create_controls(self):
        """创建所有控制控件"""
        self.create_file_operations()
        self.create_image_processing()
        self.create_calibration_section()
        self.create_measurement_section()
        self.create_results_section()

    def create_file_operations(self):
        """创建文件操作区域"""
        self.file_group = QGroupBox("文件操作")
        layout = QGridLayout()

        btn_open = QPushButton("打开图像")
        btn_save = QPushButton("保存图像")
        btn_camera = QPushButton("相机采集")
        btn_reset = QPushButton("重置图像")

        btn_open.clicked.connect(self.open_image)
        btn_save.clicked.connect(self.image_label.save_image)
        btn_camera.clicked.connect(self.camera_capture)
        btn_reset.clicked.connect(self.image_label.reset_image)

        layout.addWidget(btn_open, 0, 0)
        layout.addWidget(btn_save, 0, 1)
        layout.addWidget(btn_camera, 1, 0)
        layout.addWidget(btn_reset, 1, 1)

        self.file_group.setLayout(layout)
        self.file_group.setMaximumHeight(120)

    def create_image_processing(self):
        """创建图像处理区域"""
        self.process_group = QGroupBox("图像预处理")
        layout = QGridLayout()

        # 去噪方法
        layout.addWidget(QLabel("去噪方法:"), 0, 0)
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["高斯滤波", "中值滤波", "双边滤波"])
        layout.addWidget(self.denoise_combo, 0, 1)

        layout.addWidget(QLabel("参数:"), 1, 0)
        self.denoise_param = QLineEdit("5")
        self.denoise_param.setMaximumWidth(60)
        layout.addWidget(self.denoise_param, 1, 1)

        # 增强方法
        layout.addWidget(QLabel("增强方法:"), 2, 0)
        self.enhance_combo = QComboBox()
        self.enhance_combo.addItems(["直方图均衡化", "对比度拉伸", "伽马校正"])
        layout.addWidget(self.enhance_combo, 2, 1)

        layout.addWidget(QLabel("参数:"), 3, 0)
        self.enhance_param = QLineEdit("1.5")
        self.enhance_param.setMaximumWidth(60)
        layout.addWidget(self.enhance_param, 3, 1)

        # 处理按钮
        btn_denoise = QPushButton("去噪处理")
        btn_enhance = QPushButton("对比度修改")
        btn_denoise.clicked.connect(self.apply_denoise)
        btn_enhance.clicked.connect(self.apply_enhancement)
        layout.addWidget(btn_denoise, 4, 0)
        layout.addWidget(btn_enhance, 4, 1)

        # 透视矫正按钮
        btn_perspective = QPushButton("透视矫正（选4个角）")
        btn_perspective.setToolTip("点击后依次点击图像的左上→右上→右下→左下四个角")
        btn_perspective.clicked.connect(self.image_label.start_perspective)
        layout.addWidget(btn_perspective, 5, 0, 1, 2)  # 占两列，让按钮更宽

        self.process_group.setLayout(layout)
        self.process_group.setMaximumHeight(250)

    def create_calibration_section(self):
        """创建标定区域"""
        self.calib_group = QGroupBox("系统标定")
        layout = QGridLayout()

        # 标定状态
        layout.addWidget(QLabel("标定状态:"), 0, 0)
        self.calib_status = QLabel("未标定")
        self.calib_status.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.calib_status, 0, 1)

        # 标定方法
        layout.addWidget(QLabel("标定方法:"), 1, 0)
        self.calib_combo = QComboBox()
        self.calib_combo.addItems(["两点快速标定"])
        layout.addWidget(self.calib_combo, 1, 1)

        # 标定按钮 - 改为水平布局节省空间
        btn_exec_calib = QPushButton("执行标定")

        btn_exec_calib.clicked.connect(self.image_label.start_calibration)

        layout.addWidget(btn_exec_calib, 2, 0, 1, 2)

        self.calib_group.setLayout(layout)
        self.calib_group.setMaximumHeight(120)

    def create_measurement_section(self):
        """创建测量区域"""
        self.measure_group = QGroupBox("尺寸测量")
        layout = QGridLayout()

        # 测量模式
        layout.addWidget(QLabel("测量模式:"), 0, 0)
        self.measure_combo = QComboBox()
        self.measure_combo.addItems(["手动测量", "自动测量圆形"])
        layout.addWidget(self.measure_combo, 0, 1)

        # 测量按钮
        btn_start_measure = QPushButton("开始测量")
        btn_clear_measure = QPushButton("清除测量")
        btn_export_results = QPushButton("导出结果")

        btn_start_measure.clicked.connect(self.start_measurement)
        btn_clear_measure.clicked.connect(self.image_label.clear_measurements)
        btn_export_results.clicked.connect(self.image_label.save_measurements)

        layout.addWidget(btn_start_measure, 1, 0)
        layout.addWidget(btn_clear_measure, 1, 1)
        layout.addWidget(btn_export_results, 2, 0, 1, 2)

        # 实时轮廓检测开关
        self.auto_detect_check = QCheckBox("开关自动检测物品轮廓功能")
        self.auto_detect_check.setChecked(True)  # 默认开启，和原来行为一致
        self.auto_detect_check.toggled.connect(self.toggle_auto_detect)
        layout.addWidget(self.auto_detect_check, 3, 0, 1, 2)

        # 检测灵敏度
        layout.addWidget(QLabel("检测灵敏度:"), 4, 0, 1, 2)
        self.sensitivity_input = QLineEdit("5")

        layout.addWidget(self.sensitivity_input, 5, 0, 1, 2)
        # 画笔模式开关
        self.brush_toggle_btn = QPushButton("画笔模式")
        self.brush_toggle_btn.setCheckable(True)
        self.brush_mode_combo = QComboBox()
        self.brush_mode_combo.addItems(["曲线", "直线"])
        layout.addWidget(self.brush_toggle_btn, 6, 0)
        layout.addWidget(self.brush_mode_combo, 6, 1)

        self.brush_toggle_btn.toggled.connect(self.toggle_brush_mode)
        self.brush_mode_combo.currentIndexChanged.connect(self.change_brush_mode)

        self.measure_group.setLayout(layout)
        self.measure_group.setMaximumHeight(300)

    def create_results_section(self):
        """创建结果显示区域"""
        self.results_group = QGroupBox("测量结果")
        layout = QVBoxLayout()

        # 测量统计
        stats_layout = QGridLayout()
        stats_layout.addWidget(QLabel("测量次数:"), 0, 0)
        self.measure_count_label = QLabel("0")
        stats_layout.addWidget(self.measure_count_label, 0, 1)

        stats_layout.addWidget(QLabel("标定比例:"), 1, 0)
        self.scale_label = QLabel("未标定")
        stats_layout.addWidget(self.scale_label, 1, 1)

        layout.addLayout(stats_layout)

        # 结果列表
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["序号", "形状", "尺寸", "面积"])
        self.results_table.setMaximumHeight(200)
        layout.addWidget(self.results_table)

        self.results_group.setLayout(layout)
        self.results_group.setMaximumHeight(300)

    def apply_styles(self):
        """应用样式表"""
        self.setStyleSheet("""
            QMainWindow { background: #ffffff; }
            QGroupBox { font-weight: bold; }
            QPushButton { padding: 6px; }
        """)

    def refresh_calibration_status(self):
        if self.image_label.pix_per_cm:
            self.calib_status.setText("已标定")
            self.calib_status.setStyleSheet("color: green; font-weight: bold;")
            self.scale_label.setText(f"{self.image_label.pix_per_cm:.2f} px/cm")
        else:
            self.calib_status.setText("未标定")
            self.calib_status.setStyleSheet("color: red; font-weight: bold;")
            self.scale_label.setText("未标定")

    # ---------------------- 交互槽与工具函数 ----------------------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", "图片 (*.png *.jpg *.jpeg *.bmp)")
        if path:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                QMessageBox.warning(self, "错误", "无法打开图像")
                return
            self.image_label.set_image(img)
            self.refresh_results_table()
            self.refresh_calibration_status()

    def camera_capture(self):
        QMessageBox.information(self, "提示", "相机采集功能未实现（占位）")

    def apply_denoise(self):
        method = self.denoise_combo.currentText()
        try:
            k = int(self.denoise_param.text())
        except:
            k = 5
        params = {"ksize": k}
        self.image_label.apply_denoise(method, params)

    def apply_enhancement(self):
        method = self.enhance_combo.currentText()
        try:
            val = float(self.enhance_param.text())
        except:
            val = 1.5
        params = {}
        if method == "对比度拉伸":
            params["alpha"] = val
            params["beta"] = 0
        elif method == "伽马校正":
            params["gamma"] = val
        self.image_label.apply_enhancement(method, params)

    def start_measurement(self):
        mode = self.measure_combo.currentText()
        if mode == "手动测量":
            self.image_label.start_manual()
        elif mode == "自动测量圆形":
            self.image_label.auto_measure_circles()
        self.refresh_results_table()

    def toggle_brush_mode(self, checked):
        self.image_label.brush_mode_enabled = checked
        if checked:
            self.image_label.selecting_mode = None
            QMessageBox.information(self, "画笔模式", "已进入画笔模式。长按并拖动开始绘制，右键结束笔画。")
        else:
            # 退出画笔模式时结束任何正在绘制的笔画
            if self.image_label.brush_is_drawing:
                self.image_label._finish_brush_stroke()
            self.image_label.brush_mode_enabled = False
        self.image_label.update_display()
        self.refresh_results_table()

    def toggle_auto_detect(self, checked):
        """开关实时轮廓检测"""
        self.image_label.auto_detect_enabled = checked
        self.image_label.update_display()  # 立即刷新显示

    def change_brush_mode(self, idx):
        self.image_label.brush_mode_type = "curve" if idx == 0 else "line"

    def refresh_results_table(self):
        results = self.image_label.measurement_results
        self.results_table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(r.get("序号", ""))))
            self.results_table.setItem(i, 1, QTableWidgetItem(r.get("形状", "")))
            # 合并尺寸字段
            size_text = r.get("长度", r.get("周长", r.get("宽度", "")))
            area_text = r.get("面积", "")
            self.results_table.setItem(i, 2, QTableWidgetItem(size_text))
            self.results_table.setItem(i, 3, QTableWidgetItem(area_text))
        self.measure_count_label.setText(str(self.image_label.measurement_count))
        self.scale_label.setText(
            f"{self.image_label.pix_per_cm:.2f} px/cm" if self.image_label.pix_per_cm else "未标定")


# ---------------------- 启动应用 ----------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
