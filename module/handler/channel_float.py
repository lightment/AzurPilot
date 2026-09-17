"""渠道服（4399）启动悬浮球处理。

4399 等渠道服客户端启动后，屏幕左上角（角色名右上方）会出现 SDK 悬浮球。
悬浮球是半透明圆盘，截图里几乎不可见，但顶部带有一圈绿色「○○○」标志，
而黑色标题栏背景不含绿色，因此通过统计绿标像素即可可靠检出。
检出后自动将悬浮球拖拽到屏幕中下，并点击「隐藏悬浮球」对话框的「隐藏」按钮。
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

# 悬浮球识别区域：主页面左上角上边沿左半范围（1280x720），
# 绿色「○○○」标志为识别特征；悬浮球需位于等级与名字中间（见 GUI 说明）
CHANNEL_FLOAT_AREA = (0, 0, 640, 50)
# 绿色像素计数阈值：实测有球 443、无球 0，取 20 作为安全阈值
CHANNEL_FLOAT_GREEN_THRESHOLD = 20
# 悬浮球拖拽起点（悬浮球中心）与终点（屏幕中下偏下）：
# 「隐藏悬浮球」对话框的触发区域位于屏幕下方，实测终点需压到
# y=660 附近才能稳定触发（620 偏高，松手后悬浮球弹回原位）
CHANNEL_FLOAT_SWIPE_START = (220, 45)
CHANNEL_FLOAT_SWIPE_END = (640, 660)
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


def detect_channel_float(image) -> bool:
    """检测悬浮球：统计绿色「○○○」标志像素。

    Args:
        image: 当前截图（1280x720）。

    Returns:
        bool: True 表示识别到悬浮球。
    """
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)
    green = int(np.sum((g > r + 15) & (g > b + 15) & (g > 100)))
    logger.info(f'[渠道悬浮球] 绿色标志像素 {green}')
    return green >= CHANNEL_FLOAT_GREEN_THRESHOLD


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

    def detected(self) -> bool:
        """悬浮球是否出现在屏幕左上角黑条区域。

        悬浮球半透明难以直接模板识别，但其顶部带有绿色「○○○」标志，
        黑色标题栏背景不含绿色像素，通过统计绿色像素数量即可可靠检出。

        Returns:
            bool: True 表示识别到悬浮球。
        """
        image = crop(self.device.image, CHANNEL_FLOAT_AREA, copy=False)
        return detect_channel_float(image)

    def handle_channel_float(self) -> bool:
        """拖拽悬浮球到屏幕中下，并在「隐藏」对话框弹出后点击「隐藏」。

        Returns:
            bool: 固定返回 True，表示已执行处理。
        """
        logger.info(
            f'[渠道悬浮球] 拖拽 {CHANNEL_FLOAT_SWIPE_START} -> {CHANNEL_FLOAT_SWIPE_END}, '
            f'终点停留 {CHANNEL_FLOAT_HOLD_DURATION}s')
        start = time.monotonic()
        self.device.drag(
            CHANNEL_FLOAT_SWIPE_START, CHANNEL_FLOAT_SWIPE_END,
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
        logger.attr('绿色阈值', CHANNEL_FLOAT_GREEN_THRESHOLD)
        for attempt in range(CHANNEL_FLOAT_MAX_ATTEMPTS):
            self.device.screenshot()
            if not self.detected():
                logger.info(
                    f'[渠道悬浮球] 第 {attempt + 1}/{CHANNEL_FLOAT_MAX_ATTEMPTS} 次：'
                    '未识别到悬浮球，跳过')
                return True
            logger.info(
                f'[渠道悬浮球] 第 {attempt + 1}/{CHANNEL_FLOAT_MAX_ATTEMPTS} 次：'
                '识别到悬浮球，开始处理')
            self.handle_channel_float()
        logger.info('[渠道悬浮球] 多次处理仍未消失，跳过本回合')
        return True
