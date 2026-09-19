"""渠道服（4399）启动悬浮球处理。

4399 等渠道服客户端启动后，屏幕顶部会出现 SDK 悬浮球（半透明圆盘，
直径约 70px，截图里几乎不可见），顶部带绿色「○○○」标志（三个小圆点）。
悬浮球每次启动停靠位置不固定，通过绿标动态定位球中心后拖拽到屏幕
中下触发「隐藏悬浮球」对话框，并点击「隐藏」按钮将其彻底关闭。
"""
import time

import numpy as np

from module.base.base import ModuleBase
from module.base.button import Button
from module.base.utils import crop
from module.base.timer import Timer
from module.config.deep import deep_get
from module.handler.assets import LOGIN_CHECK
from module.logger import logger
from module.ui.page import page_main_white

# 悬浮球识别区域：主页面左上角范围（1280x720）。
# SDK 悬浮球停靠在屏幕顶部一带，顶部带绿色「○○○」标志（三个小圆点横排）。
# 悬浮球每次启动停靠位置不固定（实测出现过绿标位于 (299~349,0~9) 与
# (220,45) 附近两种），因此采用动态定位而非硬编码坐标。
CHANNEL_FLOAT_AREA = (0, 0, 640, 100)
# 排除区域：游戏头像框旁的绿色箭头（原生UI，约 (85~120, 60~90)），
# 颜色特征与悬浮球绿标相同，需显式排除避免误定位
CHANNEL_FLOAT_EXCLUDE_AREA = (80, 55, 125, 95)
# 绿色像素计数阈值：实测有球约330、无球0，取 20 作为安全阈值
CHANNEL_FLOAT_GREEN_THRESHOLD = 20
# 绿标中心到球中心的垂直偏移：球直径约70px，绿标位于球顶部，
# 实测绿标中心(324,4)对应球心(324,35)，偏移约+31px
CHANNEL_FLOAT_BALL_CENTER_OFFSET_Y = 30
# 悬浮球拖拽终点（屏幕中下偏下）：
# 「隐藏悬浮球」对话框的触发区域位于屏幕下方，实测终点需压到
# y=680 附近才能稳定触发（660 仍偏浅，松手后悬浮球弹回原位）
CHANNEL_FLOAT_SWIPE_END = (640, 680)
# 拖到终点后按住停留时长：悬浮球需停留片刻再松手才会触发「隐藏悬浮球」
# 对话框，立即松手会被判定为甩动；drag 后端另有约 0.28s 的内置停顿
CHANNEL_FLOAT_HOLD_DURATION = 0.2
CHANNEL_FLOAT_MAX_ATTEMPTS = 4
# 「隐藏悬浮球」对话框中的「隐藏」按钮：
# 实测标定(2026-09-17 截图)：对话框白色主体在游戏 y=120~599，「隐藏」
# 绿色文字位于 (734~778, 552~573) 中心 (756,562)，按钮热区四周放宽；
# 此前定义 (728,604,848,664) 中心偏下约 72px 落在对话框外，点击无效
CHANNEL_FLOAT_HIDE_BUTTON = Button(
    area=(700, 535, 800, 590),
    color=(),
    button=(700, 535, 800, 590),
    name='CHANNEL_FLOAT_HIDE_BUTTON',
)
# 「隐藏」按钮绿色文字像素数阈值：实测有按钮 334、主界面同位置干扰 0
CHANNEL_FLOAT_HIDE_GREEN_THRESHOLD = 30


def channel_float_position(image):
    """定位悬浮球：返回球中心坐标，未识别到返回 None。

    悬浮球半透明难以直接识别，但其顶部带有绿色「○○○」标志。
    在左上识别区内统计绿色像素并排除头像框旁绿色箭头（原生UI），
    绿色质心即绿标中心，向下偏移约 30px 为球中心。悬浮球每次启动
    停靠位置不固定，动态定位取代硬编码拖拽起点。

    Args:
        image: 当前截图（1280x720）。

    Returns:
        tuple: 球中心坐标 (x, y)；未识别到时 None。
    """
    area = CHANNEL_FLOAT_AREA
    area_img = crop(image, area, copy=False)
    r = area_img[:, :, 0].astype(np.int16)
    g = area_img[:, :, 1].astype(np.int16)
    b = area_img[:, :, 2].astype(np.int16)
    green = (g > r + 15) & (g > b + 15) & (g > 100)
    ex = CHANNEL_FLOAT_EXCLUDE_AREA
    x0 = max(ex[0] - area[0], 0)
    y0 = max(ex[1] - area[1], 0)
    x1 = min(ex[2] - area[0], green.shape[1])
    y1 = min(ex[3] - area[1], green.shape[0])
    green[y0:y1, x0:x1] = False
    count = int(green.sum())
    if count < CHANNEL_FLOAT_GREEN_THRESHOLD:
        logger.info(f'[渠道悬浮球] 绿色标志像素 {count}，未识别到悬浮球')
        return None
    ys, xs = np.where(green)
    cx = int(xs.mean()) + area[0]
    cy = int(ys.mean()) + area[1] + CHANNEL_FLOAT_BALL_CENTER_OFFSET_Y
    logger.info(f'[渠道悬浮球] 绿色标志像素 {count}，定位球中心 ({cx}, {cy})')
    return (cx, cy)


def hide_button_visible(image) -> bool:
    """「隐藏悬浮球」对话框的「隐藏」按钮是否可见。

    「隐藏」按钮文字为绿色，对话框白底上极易检出；此前用按钮区域
    平均亮度判断对话框弹出，但该区域落在对话框外，主界面背景亮度
    不稳定导致误判。改为统计按钮区域绿色文字像素数。

    Args:
        image: 当前截图（1280x720）。

    Returns:
        bool: True 表示「隐藏」按钮可见（可安全点击）。
    """
    area_img = crop(image, CHANNEL_FLOAT_HIDE_BUTTON.area, copy=False)
    r = area_img[:, :, 0].astype(np.int16)
    g = area_img[:, :, 1].astype(np.int16)
    b = area_img[:, :, 2].astype(np.int16)
    green = int(np.sum((g > r + 40) & (g > b + 40) & (g > 100)))
    return green >= CHANNEL_FLOAT_HIDE_GREEN_THRESHOLD


class ChannelFloatHandler(ModuleBase):
    """检测并处理渠道服启动悬浮球。"""

    def _enabled(self) -> bool:
        """渠道服悬浮球处理是否启用。

        仅当游戏为 4399 渠道服（包名 com.bilibili.blhx.m4399 且服务器为
        cn_channel-*）时，开关 Restart.MoveChannelFloat 才生效；
        其他服务器即使开启开关也不会生效。

        Returns:
            bool: True 表示启用。
        """
        if not bool(deep_get(self.config.data, 'Restart.Restart.MoveChannelFloat', default=False)):
            logger.info('[渠道悬浮球] 未启用：开关 Restart.MoveChannelFloat 未开启')
            return False
        package = str(deep_get(self.config.data, 'Alas.Emulator.PackageName', default=''))
        server_name = str(deep_get(self.config.data, 'Alas.Emulator.ServerName', default=''))
        if package == 'com.bilibili.blhx.m4399' and server_name.startswith('cn_channel-'):
            return True
        logger.info(f'[渠道悬浮球] 未启用：非 4399 渠道服（server={server_name}, package={package}）')
        return False

    def handle_channel_float(self, ball_pos) -> bool:
        """拖拽悬浮球到屏幕中下，并在「隐藏」对话框弹出后点击「隐藏」。

        Args:
            ball_pos: 悬浮球中心坐标 (x, y)，由 channel_float_position 动态定位。

        Returns:
            bool: 固定返回 True，表示已执行处理。
        """
        logger.info(
            f'[渠道悬浮球] 拖拽 {ball_pos} -> {CHANNEL_FLOAT_SWIPE_END}, '
            f'终点停留 {CHANNEL_FLOAT_HOLD_DURATION}s')
        start = time.monotonic()
        self.device.drag(
            ball_pos, CHANNEL_FLOAT_SWIPE_END,
            point_random=(0, 0, 0, 0), hold_duration=CHANNEL_FLOAT_HOLD_DURATION,
            name='CHANNEL_FLOAT_DRAG')
        logger.info(f'[渠道悬浮球] 拖拽完成，耗时 {time.monotonic() - start:.2f}s')
        # 等待「隐藏」按钮出现（截图循环，最多等 4 秒）
        dialog_timer = Timer(4).start()
        while 1:
            self.device.screenshot()
            if hide_button_visible(self.device.image):
                logger.info('[渠道悬浮球] 检测到「隐藏」按钮，点击')
                self.device.click(CHANNEL_FLOAT_HIDE_BUTTON)
                break
            if dialog_timer.reached():
                logger.info('[渠道悬浮球] 未见「隐藏」按钮，跳过点击')
                break
        return True

    def run(self) -> bool:
        """任务前的悬浮球检查入口（每个会话仅调用一次）。

        通过绿色标志检测悬浮球；识别到时自动拖拽并点击「隐藏」确认，
        最多尝试 CHANNEL_FLOAT_MAX_ATTEMPTS 次；未识别到悬浮球时不产生
        任何输入操作。

        Returns:
            bool: True 表示本回合检查已消费（无悬浮球或已处理）。
        """
        if not self._enabled():
            return True
        # 游戏进程未运行时直接跳过，避免阻塞调度器的 GameNotRunning -> Restart 流程
        if not self.device.app_is_running():
            logger.info('[渠道悬浮球] 游戏进程未运行，跳过检查')
            return True
        logger.hr('渠道悬浮球检查', level=2)
        # 等待进入主界面：游戏重启后可能停在服务器选择页，需要点击确认，
        # 未到主界面时自动点击 LOGIN_CHECK 进入（上限 60 秒，避免阻塞任务）
        wait_timer = Timer(60).start()
        while 1:
            self.device.screenshot()
            if self.appear(page_main_white.check_button, offset=(30, 30)):
                break
            if self.appear(LOGIN_CHECK, offset=(30, 30), interval=2):
                logger.info('[渠道悬浮球] 点击进入主界面（LOGIN_CHECK）')
                self.device.click(LOGIN_CHECK)
                continue
            if wait_timer.reached():
                logger.info('[渠道悬浮球] 等待主界面超时，跳过本会话')
                return True
        logger.attr('检测区域', CHANNEL_FLOAT_AREA)
        for attempt in range(CHANNEL_FLOAT_MAX_ATTEMPTS):
            self.device.screenshot()
            ball_pos = channel_float_position(self.device.image)
            if ball_pos is None:
                logger.info(
                    f'[渠道悬浮球] 第 {attempt + 1}/{CHANNEL_FLOAT_MAX_ATTEMPTS} 次：'
                    '未识别到悬浮球，跳过')
                return True
            logger.info(
                f'[渠道悬浮球] 第 {attempt + 1}/{CHANNEL_FLOAT_MAX_ATTEMPTS} 次：'
                '识别到悬浮球，开始处理')
            self.handle_channel_float(ball_pos)
        logger.info('[渠道悬浮球] 多次处理仍未消失，跳过本回合')
        return True
