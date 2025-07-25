import cv2
import torch
import numpy as np
import os
from collections import deque
import math
from pathlib import Path

# 引入YOLO
class YOLODetector:
    def __init__(self, model_path='best.pt', device='cpu'):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.to(device)
        self.model.eval()

    def detect(self, frame, conf_thres=0.3):
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()

        boxes = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf > conf_thres:
                boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))  # 转为 (x, y, w, h)
        return boxes

class DroneTracker:
    def __init__(self):
        # 初始化参数
        self.target_selected = False
        self.target_bbox = None
        self.positive_samples = []
        self.negative_samples = []
        self.feature_detector = cv2.SIFT_create()

        # 光流参数
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # 跟踪状态
        self.tracking_state = "INIT"  # INIT, TRACKING, LOST
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.target_features = []
        self.background_features = []

        # 匹配阈值
        self.global_match_threshold = 0.3
        self.local_match_threshold = 0.3
        self.lost_threshold = 0.2

        # 特征数据库
        self.positive_database = {"keypoints": [], "descriptors": []}
        self.negative_database = {"keypoints": [], "descriptors": []}

        # 光流点
        self.optical_flow_points = []

    def detect_initial_target(self, frame):
        """检测初始目标（简化版本 - 使用鼠标选择）"""

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                param['start_point'] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                param['end_point'] = (x, y)
                param['selecting'] = False

        # 创建窗口并设置鼠标回调
        cv2.namedWindow('Select Target', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Target', 800, 600)
        mouse_params = {'start_point': None, 'end_point': None, 'selecting': True}
        cv2.setMouseCallback('Select Target', mouse_callback, mouse_params)

        temp_frame = frame.copy()
        while mouse_params['selecting']:
            cv2.imshow('Select Target', temp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow('Select Target')

        if mouse_params['start_point'] and mouse_params['end_point']:
            x1, y1 = mouse_params['start_point']
            x2, y2 = mouse_params['end_point']
            self.target_bbox = (min(x1, x2), min(y1, y2),
                                abs(x2 - x1), abs(y2 - y1))
            self.target_selected = True
            return True
        return False

    def extract_features_with_descriptors(self, frame, region=None):
        """提取SIFT特征和描述子"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if region:
            x, y, w, h = region
            roi = gray[y:y + h, x:x + w]
            keypoints, descriptors = self.feature_detector.detectAndCompute(roi, None)
            # 调整关键点坐标到全图坐标系
            for kp in keypoints:
                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
        else:
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def initialize_tracker(self, frame):
        """初始化跟踪器"""
        if not self.target_selected:
            return False

        x, y, w, h = self.target_bbox

        # 提取目标区域特征（正样本）
        target_keypoints, target_descriptors = self.extract_features_with_descriptors(
            frame, (x, y, w, h))

        # 提取背景区域特征（负样本）- 目标尺寸2倍范围
        bg_x = max(0, x - w)
        bg_y = max(0, y - h)
        bg_w = min(frame.shape[1] - bg_x, w * 3)
        bg_h = min(frame.shape[0] - bg_y, h * 3)

        bg_keypoints, bg_descriptors = self.extract_features_with_descriptors(
            frame, (bg_x, bg_y, bg_w, bg_h))

        # 过滤背景特征，排除目标区域内的特征
        filtered_bg_keypoints = []
        filtered_bg_descriptors = []

        if bg_descriptors is not None:
            for i, kp in enumerate(bg_keypoints):
                if not (x <= kp.pt[0] <= x + w and y <= kp.pt[1] <= y + h):
                    filtered_bg_keypoints.append(kp)
                    filtered_bg_descriptors.append(bg_descriptors[i])

        # 建立正负样本数据库
        if target_descriptors is not None:
            self.positive_database["keypoints"] = target_keypoints
            self.positive_database["descriptors"] = target_descriptors

        if filtered_bg_descriptors:
            self.negative_database["keypoints"] = filtered_bg_keypoints
            self.negative_database["descriptors"] = np.array(filtered_bg_descriptors)

        # 初始化光流点
        if target_keypoints:
            self.optical_flow_points = [kp.pt for kp in target_keypoints]

        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tracking_state = "TRACKING"

        return True

    def compute_optical_flow(self, current_frame):
        """计算光流"""
        if self.prev_frame is None or not self.optical_flow_points:
            return []

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # 转换为numpy数组
        p0 = np.float32(self.optical_flow_points).reshape(-1, 1, 2)

        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, current_gray, p0, None, **self.lk_params)

        # 筛选有效点
        valid_points = []
        if p1 is not None:
            for i, (point, status) in enumerate(zip(p1, st)):
                if status == 1:
                    valid_points.append(tuple(point.ravel()))

        return valid_points

    def match_features(self, current_keypoints, current_descriptors, optical_flow_points):
        """特征匹配"""
        global_matches = []
        local_matches = []

        if current_descriptors is None:
            return global_matches, local_matches

        # 全局匹配 - 与正样本数据库匹配
        if len(self.positive_database["descriptors"]) > 0:
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(
                self.positive_database["descriptors"], current_descriptors, k=2)

            # 应用比值测试
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        global_matches.append(m)

        # 局部匹配 - 光流特征与检测特征匹配
        if optical_flow_points and current_keypoints:
            for i, flow_point in enumerate(optical_flow_points):
                min_dist = float('inf')
                best_match_idx = -1

                for j, kp in enumerate(current_keypoints):
                    dist = np.linalg.norm(np.array(flow_point) - np.array(kp.pt))
                    if dist < min_dist and dist < 20:  # 距离阈值
                        min_dist = dist
                        best_match_idx = j

                if best_match_idx != -1:
                    local_matches.append((i, best_match_idx))

        return global_matches, local_matches

    def fuse_matches(self, global_matches, local_matches, current_keypoints):
        """融合匹配点"""
        effective_matches = []

        # 处理全局匹配
        for match in global_matches:
            if match.distance < 100:  # 距离阈值
                effective_matches.append(current_keypoints[match.trainIdx])

        # 处理局部匹配
        for flow_idx, kp_idx in local_matches:
            effective_matches.append(current_keypoints[kp_idx])

        return effective_matches

    def update_target_position(self, effective_matches):
        """更新目标位置"""
        if not effective_matches:
            return None

        # 计算匹配点的中心和边界
        points = np.array([kp.pt for kp in effective_matches])

        if len(points) < 3:
            return None

        # 计算边界框
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # 添加边界扩展
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        w = x_max - x_min + 2 * margin
        h = y_max - y_min + 2 * margin

        return (int(x_min), int(y_min), int(w), int(h))

    def update_feature_database(self, effective_matches, current_descriptors):
        """更新特征数据库"""
        if self.tracking_state != "TRACKING":
            return

        # 更新正样本数据库
        if effective_matches and current_descriptors is not None:
            # 简化版本 - 只保留最近的特征
            max_features = 100
            if len(effective_matches) > max_features:
                effective_matches = effective_matches[:max_features]

            self.positive_database["keypoints"] = effective_matches
            # 这里简化处理描述子更新

    def track_frame(self, frame):
        """跟踪单帧"""
        if self.tracking_state == "INIT":
            if not self.initialize_tracker(frame):
                return frame, False

        # 提取当前帧特征
        current_keypoints, current_descriptors = self.extract_features_with_descriptors(frame)

        # 计算光流
        optical_flow_points = self.compute_optical_flow(frame)

        # 特征匹配
        global_matches, local_matches = self.match_features(
            current_keypoints, current_descriptors, optical_flow_points)

        # 融合匹配点
        effective_matches = self.fuse_matches(global_matches, local_matches, current_keypoints)

        # 计算匹配率
        global_match_rate = len(global_matches) / max(len(self.positive_database["keypoints"]), 1)
        local_match_rate = len(local_matches) / max(len(self.optical_flow_points), 1)

        # 判断跟踪状态
        if (global_match_rate < self.lost_threshold and
                local_match_rate < self.lost_threshold):
            self.tracking_state = "LOST"
            is_tracking = False
        else:
            if self.tracking_state == "LOST":
                self.tracking_state = "TRACKING"

            # 更新目标位置
            new_bbox = self.update_target_position(effective_matches)
            if new_bbox:
                self.target_bbox = new_bbox

            # 更新特征数据库
            self.update_feature_database(effective_matches, current_descriptors)

            # 更新光流点
            if effective_matches:
                self.optical_flow_points = [kp.pt for kp in effective_matches]

            is_tracking = True

        # 更新前一帧
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 绘制结果
        result_frame = self.draw_results(frame, is_tracking, global_match_rate, local_match_rate)

        return result_frame, is_tracking

    def draw_results(self, frame, is_tracking, global_rate, local_rate):
        """绘制跟踪结果"""
        result = frame.copy()

        if self.target_bbox:
            x, y, w, h = self.target_bbox

            # 绘制边界框
            if is_tracking:
                color = (0, 255, 0)  # 绿色 - 正在跟踪
                status_text = "TRACKING"
            else:
                color = (0, 0, 255)  # 红色 - 跟踪丢失
                status_text = "LOST"

            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # 绘制状态信息
            cv2.putText(result, f"Status: {status_text}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 绘制匹配率信息
            cv2.putText(result, f"Global: {global_rate:.2f}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result, f"Local: {local_rate:.2f}", (x, y + h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result


# 修改 process_video 函数
def process_video(input_path, output_path, yolo_model_path):
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 {input_path} 不存在")
        return

    if not input_path.lower().endswith(".mp4"):
        print("错误: 输入文件必须是 .mp4 格式")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = YOLODetector(model_path=yolo_model_path)
    tracker = DroneTracker()

    frame_count = 0
    tracking_success_count = 0

    print("开始处理视频...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count == 1:
            # 第1帧用YOLO检测
            boxes = detector.detect(frame)

            if not boxes:
                print("未检测到目标，退出处理")
                break

            temp_frame = frame.copy()
            for idx, (x, y, w, h) in enumerate(boxes):
                cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(temp_frame, f"ID {idx}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Detected Targets", temp_frame)
            cv2.waitKey(1)
            selected_id = input(f"检测到 {len(boxes)} 个目标，请输入要跟踪的目标ID (0 ~ {len(boxes) - 1}): ")

            try:
                selected_id = int(selected_id)
                tracker.target_bbox = boxes[selected_id]
                tracker.target_selected = True
                print(f"已选择目标 ID {selected_id}，初始化跟踪器")
            except Exception as e:
                print("输入无效，退出。")
                break

            cv2.destroyWindow("Detected Targets")

        result_frame, is_tracking = tracker.track_frame(frame)

        if is_tracking:
            tracking_success_count += 1

        out.write(result_frame)

        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    tracking_rate = (tracking_success_count / frame_count) * 100
    print(f"\n处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"成功跟踪帧数: {tracking_success_count}")
    print(f"跟踪成功率: {tracking_rate:.1f}%")
    print(f"输出视频已保存到: {output_path}")


if __name__ == "__main__":
    input_video = r"C:\Users\Administrator\Desktop\konggongda\konggongda2\给的视频.mp4"
    output_video = "output_drone_tracking.mp4"
    yolov5_model_path = "yolov5s.pt"  # 确保该路径有效

    print("无人机目标检测+跟踪系统")
    print("=" * 50)

    input_path = input("请输入视频文件路径 (或按Enter使用默认): ").strip()
    if not input_path:
        input_path = input_video

    output_path = input("请输入输出视频路径 (或按Enter使用默认): ").strip()
    if not output_path:
        output_path = output_video
    elif not output_path.lower().endswith(".mp4"):
        output_path += ".mp4"

    yolo_path = input("请输入YOLOv5模型路径 (或按Enter使用默认 yolov5s.pt): ").strip()
    if yolo_path:
        yolov5_model_path = yolo_path

    process_video(input_path, output_path, yolov5_model_path)

