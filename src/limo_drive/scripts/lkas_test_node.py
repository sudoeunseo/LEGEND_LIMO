#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import Bool

import numpy as np
import cv2
from math import atan


class LKAS:
    def __init__(self):
        rospy.init_node("lkas_node")

        # -----------------------------
        # (FSM 호환) 토픽 파라미터
        # -----------------------------
        self.image_topic  = rospy.get_param("~image_topic",  "/camera/rgb/image_raw/compressed")
        self.cmd_topic    = rospy.get_param("~cmd_topic",    "/cmd_vel_lkas")     # FSM이 받는 토픽 :contentReference[oaicite:4]{index=4}
        self.enable_topic = rospy.get_param("~enable_topic", "/lkas_enable")      # FSM이 publish :contentReference[oaicite:5]{index=5}

        # FSM 붙여서 쓸 때: enable 신호가 들어올 때만 동작하도록
        self.use_fsm = rospy.get_param("~use_fsm", True)

        # CvBridge
        self.bridge = CvBridge()

        # Publishers/Subscribers
        self.ctrl_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        rospy.Subscriber(self.image_topic, CompressedImage, self.img_CB, queue_size=1)
        rospy.Subscriber(self.enable_topic, Bool, self.enable_cb, queue_size=1)

        # -----------------------------
        # 주행 파라미터
        # -----------------------------
        self.speed = rospy.get_param("~speed", 0.18)
        self.trun_mutip = rospy.get_param("~turn_mult", 0.14)

        # 프레임 스킵
        self.frame_skip = int(rospy.get_param("~frame_skip", 3))
        self._frame_count = 0

        # warp 관련
        self.img_x = 0
        self.img_y = 0
        self.offset_x = int(rospy.get_param("~offset_x", 40))

        # 슬라이딩 윈도우
        self.nwindows = int(rospy.get_param("~nwindows", 10))
        self.window_height = 0
        self.nothing_flag = False

        # publish rate limit
        self.last_pub_time = rospy.get_time()
        self.pub_period = float(rospy.get_param("~pub_period", 0.1))

        # enable 상태
        if self.use_fsm:
            self.enabled = False  # FSM enable 들어오기 전에는 동작 X
        else:
            self.enabled = True   # 단독 실행 시 바로 동작

        self.prev_enabled = self.enabled
        self.have_published_cmd = False

        # -----------------------------
        # imshow 디버그(안전)
        # -----------------------------
        self.debug_view  = bool(rospy.get_param("~debug_view", False))
        self.debug_wait  = int(rospy.get_param("~debug_wait", 1))
        self.debug_every = int(rospy.get_param("~debug_every", 1))

        if self.debug_view and not os.environ.get("DISPLAY"):
            rospy.logwarn("[LKAS] DISPLAY not set. Disable imshow to avoid crash.")
            self.debug_view = False

        rospy.loginfo("[LKAS] image_topic=%s, cmd_topic=%s, enable_topic=%s, use_fsm=%s",
                      self.image_topic, self.cmd_topic, self.enable_topic, str(self.use_fsm))

    def enable_cb(self, msg):
        self.enabled = bool(msg.data)

        # enable이 True -> False로 바뀌는 순간: 한 번 정지 명령 publish (잔류 cmd 방지)
        if (self.prev_enabled is True) and (self.enabled is False) and self.have_published_cmd:
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.ctrl_pub.publish(stop)
        self.prev_enabled = self.enabled

    # -----------------------------
    # 색 기반 차선 검출
    # -----------------------------
    def detect_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([15, 80, 0], dtype=np.uint8)
        yellow_upper = np.array([45, 255, 255], dtype=np.uint8)

        white_lower = np.array([0, 0, 230], dtype=np.uint8)
        white_upper = np.array([179, 40, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask  = cv2.inRange(hsv, white_lower,  white_upper)

        blend_mask = yellow_mask | white_mask
        return cv2.bitwise_and(img, img, mask=blend_mask)

    # -----------------------------
    # BEV warp
    # -----------------------------
    def img_warp(self, img):
        self.img_x, self.img_y = img.shape[1], img.shape[0]

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

    # -----------------------------
    # 이진화 + 숫자 억제(중앙 밴드)
    # -----------------------------
    def img_binary(self, blend_line):
        gray = cv2.cvtColor(blend_line, cv2.COLOR_BGR2GRAY)
        _, binary_line = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)

        h, w = binary_line.shape
        cx = w // 2

        center_band_half = int(w * 0.23)     # 0.18~0.28 조절
        y_start = int(h * 0.45)              # 0.35~0.60 조절

        x0 = max(0, cx - center_band_half)
        x1 = min(w, cx + center_band_half)
        binary_line[y_start:h, x0:x1] = 0

        return binary_line

    # -----------------------------
    # nothing fallback
    # -----------------------------
    def detect_nothing(self):
        offset = int(self.img_x * 0.140625)
        self.nothing_left_x_base = offset
        self.nothing_right_x_base = self.img_x - offset

        self.nothing_pixel_left_x = np.full(self.nwindows, self.nothing_left_x_base, dtype=np.int32)
        self.nothing_pixel_right_x = np.full(self.nwindows, self.nothing_right_x_base, dtype=np.int32)

        base_y = int(self.window_height / 2)
        self.nothing_pixel_y = np.arange(0, self.nwindows * base_y, base_y, dtype=np.int32)

    # -----------------------------
    # sliding window
    # -----------------------------
    def window_search(self, binary_line):
        h, w = binary_line.shape

        bottom_half = binary_line[h // 2 :, :]
        histogram = np.sum(bottom_half, axis=0)

        midpoint = w // 2
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

        left_x_current = left_x_base if left_x_base != 0 else self.nothing_left_x_base
        right_x_current = right_x_base if right_x_base != midpoint else self.nothing_right_x_base

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

        left_x = lane_x[left_lane_idx]
        left_y = lane_y[left_lane_idx]
        right_x = lane_x[right_lane_idx]
        right_y = lane_y[right_lane_idx]

        if len(left_x) == 0 and len(right_x) == 0:
            left_x = self.nothing_pixel_left_x
            left_y = self.nothing_pixel_y
            right_x = self.nothing_pixel_right_x
            right_y = self.nothing_pixel_y
        else:
            if len(left_x) == 0:
                left_x = right_x - self.img_x // 2
                left_y = right_y
            elif len(right_x) == 0:
                right_x = left_x + self.img_x // 2
                right_y = left_y

        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        plot_y = np.linspace(0, h - 1, 5)
        left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        center_fit_x = (right_fit_x + left_fit_x) / 2.0

        left_pts = np.int32(np.column_stack((left_fit_x, plot_y)))
        right_pts = np.int32(np.column_stack((right_fit_x, plot_y)))
        center_pts = np.int32(np.column_stack((center_fit_x, plot_y)))

        cv2.polylines(out_img, [left_pts], False, (0, 0, 255), 5)
        cv2.polylines(out_img, [right_pts], False, (0, 255, 0), 5)
        cv2.polylines(out_img, [center_pts], False, (255, 0, 0), 3)

        return out_img, left_x, left_y, right_x, right_y

    # -----------------------------
    # offset 계산
    # -----------------------------
    def meter_per_pixel(self):
        world_warp = np.array([[97, 1610], [109, 1610], [109, 1606], [97, 1606]], dtype=np.float32)

        dx_x = world_warp[0, 0] - world_warp[3, 0]
        dy_x = world_warp[0, 1] - world_warp[3, 1]
        meter_x = dx_x * dx_x + dy_x * dy_x

        dx_y = world_warp[0, 0] - world_warp[1, 0]
        dy_y = world_warp[0, 1] - world_warp[1, 1]
        meter_y = dx_y * dx_y + dy_y * dy_y

        meter_per_pix_x = meter_x / float(self.img_x)
        meter_per_pix_y = meter_y / float(self.img_y)
        return meter_per_pix_x, meter_per_pix_y

    def calc_vehicle_offset(self, img, left_x, left_y, right_x, right_y):
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        h = img.shape[0]
        bottom_y = h - 1
        y2 = bottom_y * bottom_y

        a_l, b_l, c_l = left_fit
        a_r, b_r, c_r = right_fit

        bottom_x_left = a_l * y2 + b_l * bottom_y + c_l
        bottom_x_right = a_r * y2 + b_r * bottom_y + c_r

        img_center_x = img.shape[1] / 2.0
        lane_center_x = (bottom_x_left + bottom_x_right) / 2.0
        pixel_offset = img_center_x - lane_center_x

        meter_per_pix_x, _ = self.meter_per_pixel()
        vehicle_offset = pixel_offset * (2 * meter_per_pix_x)
        return vehicle_offset

    def ctrl_cmd(self, vehicle_offset):
        msg = Twist()
        msg.linear.x = self.speed
        msg.angular.z = -vehicle_offset * self.trun_mutip
        return msg

    # -----------------------------
    # image callback
    # -----------------------------
    def img_CB(self, data):
        if not self.enabled:
            return

        self._frame_count += 1
        if self.frame_skip > 1 and (self._frame_count % self.frame_skip != 0):
            return

        try:
            img = self.bridge.compressed_imgmsg_to_cv2(data)

            # downscale
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // 2, h // 2))

            # window params
            self.window_height = img.shape[0] // self.nwindows

            warp_img = self.img_warp(img)
            blend_img = self.detect_color(warp_img)
            binary_img = self.img_binary(blend_img)

            if not self.nothing_flag:
                self.detect_nothing()
                self.nothing_flag = True

            sliding_window_img, left_x, left_y, right_x, right_y = self.window_search(binary_img)

            vehicle_offset = self.calc_vehicle_offset(sliding_window_img, left_x, left_y, right_x, right_y)
            ctrl_cmd_msg = self.ctrl_cmd(vehicle_offset)

            # imshow (optional)
            if self.debug_view and (self._frame_count % self.debug_every == 0):
                binary_vis = (binary_img * 255).astype(np.uint8)
                cv2.imshow("01_raw_resized", img)
                cv2.imshow("02_warp", warp_img)
                cv2.imshow("03_color_blend", blend_img)
                cv2.imshow("04_binary", binary_vis)
                cv2.imshow("05_sliding_window", sliding_window_img)
                key = cv2.waitKey(self.debug_wait) & 0xFF
                if key == ord("q") or key == 27:
                    rospy.signal_shutdown("User requested shutdown via OpenCV window")
                    return

            now = rospy.get_time()
            if now - self.last_pub_time >= self.pub_period:
                self.ctrl_pub.publish(ctrl_cmd_msg)
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