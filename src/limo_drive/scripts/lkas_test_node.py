#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import Bool


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


class LKAS:
    def __init__(self):
        rospy.init_node("lkas_node")

        # -----------------------------
        # (FSM_MUX 호환) 토픽 파라미터
        # -----------------------------
        self.image_topic  = rospy.get_param("~image_topic",  "/camera/rgb/image_raw/compressed")
        self.cmd_topic    = rospy.get_param("~cmd_topic",    "/cmd_vel_lkas")     # FSM_MUX가 구독 :contentReference[oaicite:3]{index=3}
        self.enable_topic = rospy.get_param("~enable_topic", "/lkas_enable")      # FSM_MUX가 publish :contentReference[oaicite:4]{index=4}
        self.use_fsm      = rospy.get_param("~use_fsm", True)

        # CvBridge
        self.bridge = CvBridge()

        # Publishers/Subscribers
        self.ctrl_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(self.image_topic, CompressedImage, self.img_CB, queue_size=1)
        rospy.Subscriber(self.enable_topic, Bool, self.enable_cb, queue_size=1)

        # -----------------------------
        # 주행 파라미터
        # -----------------------------
        self.speed       = float(rospy.get_param("~speed", 0.18))
        self.steer_gain  = float(rospy.get_param("~steer_gain", 0.9))   # (권장) 픽셀 정규화 기반 gain
        self.max_yawrate = float(rospy.get_param("~max_yawrate", 1.2))

        # 프레임 스킵
        self.frame_skip = int(rospy.get_param("~frame_skip", 2))
        self._frame_count = 0

        # warp 관련
        self.img_x = 0
        self.img_y = 0
        self.offset_x = int(rospy.get_param("~offset_x", 40))

        # sliding window
        self.nwindows = int(rospy.get_param("~nwindows", 10))
        self.window_height = 0
        self.nothing_flag = False

        # publish rate limit
        self.last_pub_time = rospy.get_time()
        self.pub_period = float(rospy.get_param("~pub_period", 0.05))  # 20Hz 기본

        # enable 상태
        self.enabled = (not self.use_fsm)
        self.prev_enabled = self.enabled
        self.have_published_cmd = False

        # -----------------------------
        # 동적 "차선 사이" 제거 + 폭/추종 모드 파라미터
        # -----------------------------
        self.y_start_ratio     = float(rospy.get_param("~y_start_ratio", 0.45))
        self.hist_roi_ratio    = float(rospy.get_param("~hist_roi_ratio", 0.60))   # 아래쪽 몇 %로 히스토그램
        self.left_search_ratio = float(rospy.get_param("~left_search_ratio", 0.45))
        self.right_search_ratio= float(rospy.get_param("~right_search_ratio", 0.55))

        self.keep_edge_px      = int(rospy.get_param("~keep_edge_px", 12))         # 차선 라인 보존 여유
        self.min_peak_value    = int(rospy.get_param("~min_peak_value", 30))       # 히스토그램 peak 최소
        self.min_lane_width_px = int(rospy.get_param("~min_lane_width_px", 80))    # 너무 좁으면 무시

        self.widen_factor      = float(rospy.get_param("~widen_factor", 1.55))     # 폭이 EMA 대비 이 이상이면 "한쪽만"
        self.width_ema_alpha   = float(rospy.get_param("~width_ema_alpha", 0.15))  # lane width EMA 업데이트율

        self.lane_width_ema = None
        self.prev_left_base = None
        self.prev_right_base = None

        # -----------------------------
        # imshow 디버그
        # -----------------------------
        self.debug_view  = bool(rospy.get_param("~debug_view", False))
        self.debug_wait  = int(rospy.get_param("~debug_wait", 1))
        self.debug_every = int(rospy.get_param("~debug_every", 1))

        if self.debug_view and not os.environ.get("DISPLAY"):
            rospy.logwarn("[LKAS] DISPLAY not set. Disable imshow to avoid crash.")
            self.debug_view = False

        rospy.loginfo("[LKAS] image_topic=%s, cmd_topic=%s, enable_topic=%s, use_fsm=%s",
                      self.image_topic, self.cmd_topic, self.enable_topic, str(self.use_fsm))

    # -------------------------------------------------------
    # FSM enable callback
    # -------------------------------------------------------
    def enable_cb(self, msg: Bool):
        self.enabled = bool(msg.data)

        # True -> False 전환 시 stop 1회 publish (잔류 cmd 방지)
        if (self.prev_enabled is True) and (self.enabled is False) and self.have_published_cmd:
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.ctrl_pub.publish(stop)

        self.prev_enabled = self.enabled

    # -------------------------------------------------------
    # 색 기반 차선 검출
    # -------------------------------------------------------
    def detect_color(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([15, 80, 0], dtype=np.uint8)
        yellow_upper = np.array([45, 255, 255], dtype=np.uint8)

        white_lower = np.array([0, 0, 230], dtype=np.uint8)
        white_upper = np.array([179, 40, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask  = cv2.inRange(hsv, white_lower,  white_upper)

        blend_mask = yellow_mask | white_mask
        return cv2.bitwise_and(img_bgr, img_bgr, mask=blend_mask)

    # -------------------------------------------------------
    # BEV warp
    # -------------------------------------------------------
    def img_warp(self, img):
        self.img_x, self.img_y = img.shape[1], img.shape[0]

        # NOTE: 이 src는 카메라 세팅에 민감함(필요시 파라미터화 권장)
        src_center_offset = [100, 158]
        src = np.array(
            [
                [0, self.img_y - 1],
                [src_center_offset[0], src_center_offset[1]],
                [self.img_x - src_center_offset[0], src_center_offset[1]],
                [self.img_x - 1, self.img_y - 1],
            ],
            dtype=np.float32,
        )

        dst = np.array(
            [
                [self.offset_x, self.img_y],
                [self.offset_x, 0],
                [self.img_x - self.offset_x, 0],
                [self.img_x - self.offset_x, self.img_y],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, matrix, (self.img_x, self.img_y))

    # -------------------------------------------------------
    # (1) 이진화
    # (2) 하단 히스토그램으로 좌/우 차선 base 추정
    # (3) "차선 사이"만 동적으로 제거
    # -------------------------------------------------------
    def img_binary_dynamic(self, blend_line_bgr):
        gray = cv2.cvtColor(blend_line_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)

        h, w = binary.shape
        y_hist = int(h * self.hist_roi_ratio)
        y_start = int(h * self.y_start_ratio)

        # 히스토그램(하단 ROI)
        hist = np.sum(binary[y_hist:h, :], axis=0)

        left_end = int(w * self.left_search_ratio)
        right_start = int(w * self.right_search_ratio)

        left_hist = hist[:left_end]
        right_hist = hist[right_start:]

        left_base = None
        right_base = None

        if left_hist.size > 0:
            li = int(np.argmax(left_hist))
            if left_hist[li] >= self.min_peak_value:
                left_base = li

        if right_hist.size > 0:
            ri = int(np.argmax(right_hist))
            if right_hist[ri] >= self.min_peak_value:
                right_base = right_start + ri

        # 이전 값 fallback
        if left_base is None:
            left_base = self.prev_left_base
        if right_base is None:
            right_base = self.prev_right_base

        # 저장
        self.prev_left_base = left_base
        self.prev_right_base = right_base

        # 차선 사이(내부) 제거: 두 base가 있고 폭이 충분하면 내부만 제거
        if (left_base is not None) and (right_base is not None):
            width = right_base - left_base
            if width >= self.min_lane_width_px:
                x0 = int(left_base + self.keep_edge_px)
                x1 = int(right_base - self.keep_edge_px)
                x0 = clamp(x0, 0, w)
                x1 = clamp(x1, 0, w)
                if x1 > x0:
                    binary[y_start:h, x0:x1] = 0

        return binary, left_base, right_base

    # -------------------------------------------------------
    # nothing fallback
    # -------------------------------------------------------
    def detect_nothing(self):
        offset = int(self.img_x * 0.140625)
        self.nothing_left_x_base = offset
        self.nothing_right_x_base = self.img_x - offset

    # -------------------------------------------------------
    # sliding window (base 입력 가능)
    # -------------------------------------------------------
    def window_search(self, binary_line, left_base=None, right_base=None):
        h, w = binary_line.shape

        if left_base is None:
            left_base = self.nothing_left_x_base
        if right_base is None:
            right_base = self.nothing_right_x_base

        left_x_current = int(left_base)
        right_x_current = int(right_base)

        out_img = (np.dstack((binary_line, binary_line, binary_line)).astype(np.uint8) * 255)

        margin = 80
        min_pix = int((margin * 2 * self.window_height) * 0.005)

        lane_y, lane_x = binary_line.nonzero()
        lane_y = lane_y.astype(np.int32)
        lane_x = lane_x.astype(np.int32)

        left_lane_idx_list = []
        right_lane_idx_list = []

        for window in range(self.nwindows):
            win_y_low  = h - (window + 1) * self.window_height
            win_y_high = h - window * self.window_height

            left_low  = left_x_current - margin
            left_high = left_x_current + margin
            right_low  = right_x_current - margin
            right_high = right_x_current + margin

            in_window = (lane_y >= win_y_low) & (lane_y < win_y_high)

            good_left_idx = np.where(in_window & (lane_x >= left_low) & (lane_x < left_high))[0]
            good_right_idx = np.where(in_window & (lane_x >= right_low) & (lane_x < right_high))[0]

            left_lane_idx_list.append(good_left_idx)
            right_lane_idx_list.append(good_right_idx)

            if len(good_left_idx) > min_pix:
                left_x_current = int(np.mean(lane_x[good_left_idx]))
            if len(good_right_idx) > min_pix:
                right_x_current = int(np.mean(lane_x[good_right_idx]))

        left_lane_idx = np.concatenate(left_lane_idx_list) if left_lane_idx_list else np.array([], dtype=int)
        right_lane_idx = np.concatenate(right_lane_idx_list) if right_lane_idx_list else np.array([], dtype=int)

        left_x = lane_x[left_lane_idx] if left_lane_idx.size else np.array([], dtype=np.int32)
        left_y = lane_y[left_lane_idx] if left_lane_idx.size else np.array([], dtype=np.int32)
        right_x = lane_x[right_lane_idx] if right_lane_idx.size else np.array([], dtype=np.int32)
        right_y = lane_y[right_lane_idx] if right_lane_idx.size else np.array([], dtype=np.int32)

        left_cnt = int(left_x.size)
        right_cnt = int(right_x.size)

        # 둘 다 없으면 실패
        if left_cnt == 0 and right_cnt == 0:
            return out_img, None

        # 2차 곡선 fit (있는 쪽만)
        left_fit = None
        right_fit = None

        if left_cnt > 30:
            left_fit = np.polyfit(left_y, left_x, 2)
        if right_cnt > 30:
            right_fit = np.polyfit(right_y, right_x, 2)

        return out_img, {
            "left_fit": left_fit,
            "right_fit": right_fit,
            "left_cnt": left_cnt,
            "right_cnt": right_cnt,
            "h": h,
            "w": w,
        }

    # -------------------------------------------------------
    # 목표 lane center 계산: (폭 급증 시 한쪽 차선 추종)
    # -------------------------------------------------------
    def compute_lane_center(self, fit_info):
        h = fit_info["h"]
        w = fit_info["w"]
        yb = h - 1

        left_fit = fit_info["left_fit"]
        right_fit = fit_info["right_fit"]
        left_cnt = fit_info["left_cnt"]
        right_cnt = fit_info["right_cnt"]

        # bottom x 계산
        bottom_x_left = None
        bottom_x_right = None

        if left_fit is not None:
            a, b, c = left_fit
            bottom_x_left = a * (yb**2) + b * yb + c

        if right_fit is not None:
            a, b, c = right_fit
            bottom_x_right = a * (yb**2) + b * yb + c

        # lane width EMA 업데이트(둘 다 있을 때만)
        width = None
        if (bottom_x_left is not None) and (bottom_x_right is not None):
            width = float(bottom_x_right - bottom_x_left)
            if width > self.min_lane_width_px and width < w * 0.95:
                if self.lane_width_ema is None:
                    self.lane_width_ema = width
                else:
                    a = self.width_ema_alpha
                    self.lane_width_ema = (1 - a) * self.lane_width_ema + a * width

        # EMA가 아직 없으면 보수적으로 w*0.5 사용
        nominal_width = float(self.lane_width_ema) if self.lane_width_ema is not None else float(w * 0.5)

        # 기본: 둘 다 있으면 가운데
        if (bottom_x_left is not None) and (bottom_x_right is not None):
            # 폭이 갑자기 커지면 -> 한쪽 차선 추종 모드
            if (self.lane_width_ema is not None) and (width is not None) and (width > self.widen_factor * self.lane_width_ema):
                # 픽셀 수 많은 쪽을 우선
                if left_cnt >= right_cnt:
                    lane_center_x = bottom_x_left + 0.5 * nominal_width
                else:
                    lane_center_x = bottom_x_right - 0.5 * nominal_width
            else:
                lane_center_x = 0.5 * (bottom_x_left + bottom_x_right)

            return float(clamp(lane_center_x, 0.0, float(w - 1)))

        # 한쪽만 있는 경우: 그쪽 기준으로 중심 합성
        if bottom_x_left is not None:
            lane_center_x = bottom_x_left + 0.5 * nominal_width
            return float(clamp(lane_center_x, 0.0, float(w - 1)))

        if bottom_x_right is not None:
            lane_center_x = bottom_x_right - 0.5 * nominal_width
            return float(clamp(lane_center_x, 0.0, float(w - 1)))

        return None

    # -------------------------------------------------------
    # 제어: 픽셀 오차 정규화 기반 (단순/안정)
    # -------------------------------------------------------
    def ctrl_cmd_from_center(self, lane_center_x, img_w):
        img_center_x = img_w / 2.0
        err_px = lane_center_x - img_center_x

        # [-1, 1] 정규화
        err_norm = err_px / (img_w / 2.0)

        msg = Twist()
        msg.linear.x = float(self.speed)

        yaw = -self.steer_gain * err_norm
        msg.angular.z = float(clamp(yaw, -self.max_yawrate, self.max_yawrate))
        return msg

    # -------------------------------------------------------
    # image callback
    # -------------------------------------------------------
    def img_CB(self, data: CompressedImage):
        if not self.enabled:
            return

        self._frame_count += 1
        if self.frame_skip > 1 and (self._frame_count % self.frame_skip != 0):
            return

        try:
            img = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")

            # downscale (연산량 감소)
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (w0 // 2, h0 // 2))

            # window params
            self.window_height = img.shape[0] // self.nwindows

            warp_img = self.img_warp(img)
            blend_img = self.detect_color(warp_img)

            # (핵심) 동적 "차선 사이" 제거
            binary_img, left_base, right_base = self.img_binary_dynamic(blend_img)

            if not self.nothing_flag:
                self.detect_nothing()
                self.nothing_flag = True

            sliding_window_img, fit_info = self.window_search(binary_img, left_base, right_base)

            # 차선 중심 계산 실패 시: 안전하게 정지
            if fit_info is None:
                cmd = Twist()
            else:
                lane_center_x = self.compute_lane_center(fit_info)
                if lane_center_x is None:
                    cmd = Twist()
                else:
                    cmd = self.ctrl_cmd_from_center(lane_center_x, fit_info["w"])

            # debug view
            if self.debug_view and (self._frame_count % self.debug_every == 0):
                binary_vis = (binary_img * 255).astype(np.uint8)
                cv2.imshow("01_raw_resized", img)
                cv2.imshow("02_warp", warp_img)
                cv2.imshow("03_color_blend", blend_img)
                cv2.imshow("04_binary_dynamic", binary_vis)
                cv2.imshow("05_sliding_window", sliding_window_img)
                key = cv2.waitKey(self.debug_wait) & 0xFF
                if key == ord("q") or key == 27:
                    rospy.signal_shutdown("User requested shutdown via OpenCV window")
                    return

            # rate limit publish
            now = rospy.get_time()
            if now - self.last_pub_time >= self.pub_period:
                self.ctrl_pub.publish(cmd)
                self.have_published_cmd = True
                self.last_pub_time = now

        except Exception as e:
            rospy.logerr("[LKAS] exception in img_CB: %s", str(e))
            # 안전 정지
            stop = Twist()
            self.ctrl_pub.publish(stop)


if __name__ == "__main__":
    node = LKAS()
    try:
        rospy.spin()
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass
