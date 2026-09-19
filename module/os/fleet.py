"""
大世界舰队控制模块。

负责大世界（Operation Siren）模式下的舰队基础控制，涵盖大世界特有的
移动逻辑、血量检测、港口定位以及根据战斗状态切换编队的底层指令。

主要类:
    OSFleet: 大世界舰队主控类，整合摄像机、战斗、地图舰队和余烬系统。
    BossFleet: Boss 舰队数据类，记录舰队索引和待命位置。
    PercentageOcr: 百分比 OCR 识别器，用于塞壬要塞清理进度。

术语:
    大世界 (Operation Siren / OS): 碧蓝航线的高难度 PVE 模式。
    塞壬 (Siren): 大世界中的敌对势力。
    余烬 (Ash/Ember): 大世界中的特殊系统，可通过信标触发战斗。
    行动力 (Action Point / AP): 进入海域需要消耗的资源。
    港口 (Port): 碧蓝航线的补给/修理据点。
    适应性 (Adaptability): 大世界中的特殊属性加成。
    净化装置 (Purification Device): 大世界中的特殊装置。
    安全区 (Safe Zone): 已完全清理的海域。
    危险区 (Danger Zone): 未完全清理的海域。
"""
import re

import inflection
import numpy as np

from module.base.button import Button, ButtonGrid
from module.base.filter import Filter
from module.base.timer import Timer
from module.base.utils import area_cross_area, point_limit
from module.config.utils import dict_to_kv
from module.exception import MapWalkError
from module.handler.assets import MAINTENANCE_ANNOUNCE
from module.logger import logger
from module.map.fleet import Fleet
from module.map.map_grids import SelectedGrids
from module.map.utils import location_ensure
from module.map_detection.utils import area2corner, corner2inner
from module.ocr.ocr import Ocr
from module.os.assets import FLEET_EMP_DEBUFF, MAP_EXIT, MAP_GOTO_GLOBE, MAP_GOTO_GLOBE_FOG, STRONGHOLD_PERCENTAGE, TEMPLATE_EMPTY_HP
from module.os.camera import OSCamera
from module.os.map_base import OSCampaignMap
from module.os_ash.ash import OSAsh
from module.os_combat.combat import Combat
from module.os_combat.assets import SIREN_PREPARATION
from module.combat.assets import BATTLE_PREPARATION
from module.os_handler.assets import AUTO_SEARCH_REWARD, CLICK_SAFE_AREA, IN_MAP, PORT_ENTER, TEMPLATE_STORAGE_SHIP_EMPTY
from module.os_shop.assets import PORT_SUPPLY_CHECK
from module.ui.assets import BACK_ARROW

FLEET_FILTER = Filter(regex=re.compile(r'fleet-?(\d)'), attr=('fleet',), preset=('callsubmarine',))


def limit_walk(location, step=3):
    """限制舰队单次移动步数，防止越界。

    将移动向量限制在曼哈顿距离不超过 step 的范围内，
    优先保留 y 方向的移动量。

    Args:
        location (tuple[int, int]): 目标移动向量 (x, y)。
        step (int): 最大步数。默认为 3。

    Returns:
        tuple[int, int]: 限制后的移动向量。
    """
    x, y = location
    if abs(x) > 0:
        x = min(abs(x), step - abs(y)) * x // abs(x)
    return x, y


class BossFleet:
    """Boss 战斗用舰队数据类。

    用于舰队筛选器中代表一支可参与 Boss 战的编队。
    每支 BossFleet 会分配一个待命位置 (standby_loca)，
    在 Boss 战间歇时舰队会移动到该位置等待。

    Attributes:
        fleet_index (int): 舰队编号（1~4）。
        fleet (str): 舰队编号的字符串形式。
        standby_loca (tuple[int, int]): 待命位置坐标，
            用于 Boss 战间的站位安排。
    """
    def __init__(self, fleet_index):
        """
        Args:
            fleet_index (int): 舰队编号（1~4）。
        """
        self.fleet_index = fleet_index
        self.fleet = str(fleet_index)
        self.standby_loca = (0, 0)

    def __str__(self):
        return f'Fleet-{self.fleet}'

    __repr__ = __str__

    def __eq__(self, other):
        return str(self) == str(other)


class PercentageOcr(Ocr):
    """百分比 OCR 识别器。

    用于识别塞壬要塞清理进度等百分比数值。
    对输入图像进行上下白色填充以提高识别准确率。

    Attributes:
        继承自 Ocr 的所有属性。
    """
    def __init__(self, *args, **kwargs):
        """初始化百分比 OCR，强制使用 azur_lane 语言模型。"""
        kwargs['lang'] = 'azur_lane'
        super().__init__(*args, **kwargs)

    def pre_process(self, image):
        """对 OCR 输入图像进行预处理。

        调用父类预处理后，在图像上下各填充 2 像素的白色边框，
        以提高百分比数字的识别率。

        Args:
            image (np.ndarray): 原始裁剪图像。

        Returns:
            np.ndarray: 预处理后的图像。
        """
        image = super().pre_process(image)
        image = np.pad(image, ((2, 2), (0, 0)), mode='constant', constant_values=255)
        return image


FLEET_LOW_RESOLVE = Button(
    area=(294, 76, 339, 121), color=(255, 44, 33), button=(294, 76, 339, 121),
    name='FLEET_LOW_RESOLVE')


class OSFleet(OSCamera, Combat, Fleet, OSAsh):
    """大世界舰队主控类。

    整合摄像机控制 (OSCamera)、战斗系统 (Combat)、地图舰队 (Fleet)
    和余烬系统 (OSAsh)，提供大世界模式下完整的舰队操控能力。

    负责：
    - 舰队在地图上的移动和导航
    - 血量检测和修理判断
    - 港口定位和前往
    - Boss 战斗编队切换
    - 深渊海域 (Abyssal) 处理
    - 塞壬要塞进度检测

    Attributes:
        need_repair (list[bool]): 每个舰位是否需要修理（显示扳手图标）。
        _os_map_event_handled (bool): 地图事件处理标志，用于伏击/神秘格子判断。
    """
    def _goto(self, location, expected=''):
        """移动舰队到指定位置，同时更新雷达和处理余烬信标攻击。

        Args:
            location (tuple[int, int]): 目标格子坐标。
            expected (str): 预期到达后的状态，如 'combat'、'mystery'。
        """
        super()._goto(location, expected)
        self.predict_radar()
        self.map.show()

        if self.handle_ash_beacon_attack():
            # 余烬攻击后，摄像机重新聚焦到当前舰队。
            self.camera = location
            self.update()

    def map_data_init(self, map_=None):
        """
        创建新的地图对象，使用当前海域的形状。
        """
        map_ = OSCampaignMap()
        map_.shape = self.zone.shape
        super().map_data_init(map_)

    def map_control_init(self):
        """
        移除不存在的元素（如策略、回合等）。
        """
        # self.handle_strategy(index=1 if not self.fleets_reversed() else 2)
        self.update()
        # if self.handle_fleet_reverse():
        #     self.handle_strategy(index=1)
        self.hp_reset()
        self.hp_get()
        self.lv_reset()
        self.lv_get()
        self.ensure_edge_insight(preset=self.map.in_map_swipe_preset_data, swipe_limit=(6, 5))
        # self.full_scan(must_scan=self.map.camera_data_spawn_point)
        # self.find_current_fleet()
        # self.find_path_initial()
        # self.map.show_cost()
        # self.round_reset()
        # self.round_battle()

    def find_current_fleet(self):
        self.fleet_1 = self.camera

    @property
    def _walk_sight(self):
        """获取大世界行走视野范围。

        Returns:
            tuple[int, int, int, int]: 视野边界 (x1, y1, x2, y2)。
        """
        sight = (-4, -1, 3, 2)
        return sight

    _os_map_event_handled = False

    def ambush_color_initial(self):
        """重置大世界地图事件处理标志。"""
        self._os_map_event_handled = False

    def handle_ambush(self):
        """
        将地图事件视为伏击，触发行走重试。
        """
        if self.handle_map_get_items():
            self._os_map_event_handled = True
            self.device.sleep(0.3)
            self.device.screenshot()
            return True
        elif self.handle_map_event():
            self.ensure_no_map_event()
            self._os_map_event_handled = True
            return True
        else:
            return False

    def handle_mystery(self, button=None):
        """
        处理伏击后，如果舰队已到达则视为神秘格子，否则仅视为伏击。
        """
        if self._os_map_event_handled and button.predict_fleet() and button.predict_current_fleet():
            return 'get_item'
        else:
            return False

    @staticmethod
    def _get_goto_expected(grid):
        """
        获取 _goto() 中使用的 `expected` 参数值。
        """
        if grid.is_enemy:
            return 'combat'
        elif grid.is_resource or grid.is_meowfficer or grid.is_exclamation:
            return 'mystery'
        else:
            return ''

    def _hp_grid(self):
        """获取舰队血条位置网格。

        根据不同服务器的 OS 布局调整血条位置。

        Returns:
            ButtonGrid: 血条按钮网格。
        """
        hp_grid = super()._hp_grid()

        # 六个血条的位置，根据各服务器的 OS 布局
        if self.config.SERVER == 'en':
            hp_grid = ButtonGrid(origin=(35, 205), delta=(0, 100), button_shape=(66, 3), grid_shape=(1, 6))
        elif self.config.SERVER == 'jp':
            pass
        else:
            pass

        return hp_grid

    def _storage_hp_grid(self):
        """获取仓库界面中的血条位置网格。

        Returns:
            ButtonGrid: 仓库界面的血条按钮网格。
        """
        return ButtonGrid(origin=(185, 553), delta=(166, 0), button_shape=(99, 4), grid_shape=(6, 1))

    def hp_retreat_triggered(self):
        return False

    need_repair = [False, False, False, False, False, False]

    def hp_get(self):
        """
        计算当前血量，同时检测扳手图标（舰船已阵亡，需要修理）。
        """
        super().hp_get()
        if self.config.OpsiHazard1Leveling_SkipHpCheck:
            self.need_repair = [False, False, False, False, False, False]
            return

        ship_icon = self._hp_grid().crop((0, -67, 67, 0))
        need_repair = [TEMPLATE_EMPTY_HP.match(self.image_crop(button, copy=False)) for button in ship_icon.buttons]
        self.need_repair = need_repair
        logger.attr('维修图标', need_repair)

        if any(need_repair):
            for index, repair in enumerate(need_repair):
                if repair:
                    self._hp_has_ship[self.fleet_current_index][index] = True
                    self._hp[self.fleet_current_index][index] = 0

            logger.attr('血量', ' '.join(
                [str(int(data * 100)).rjust(3) + '%' if use else '____'
                 for data, use in zip(self.hp, self.hp_has_ship, strict=False)]))

        return self.hp

    def _storage_hp_get(self):
        super().hp_get()
        ship_icon = self._hp_grid().crop((-29, -165, 106, -30))
        has_ship = [not TEMPLATE_STORAGE_SHIP_EMPTY.match(
                    self.image_crop(button, copy=False), similarity=0.5) for button in ship_icon.buttons]
        need_repair = [not repair for repair in self.hp_has_ship]
        for index, repair in enumerate(need_repair):
            if repair:
                self._hp[self.fleet_current_index][index] = 0
        for index, ship in enumerate(has_ship):
            self._hp_has_ship[self.fleet_current_index][index] = ship
        self.need_repair = [all(repair) for repair in zip(need_repair, has_ship, strict=False)]
        logger.attr('维修图标', self.need_repair)
        logger.attr('血量', ' '.join(
            [str(int(data * 100)).rjust(3) + '%' if use else '____'
            for data, use in zip(self.hp, self.hp_has_ship, strict=False)]))

    def storage_hp_get(self):
        """
        在 STORAGE_CHECK 页面计算当前血量，同时检测扳手图标（舰船已阵亡，需要修理）。
        """
        origin = (self._hp_grid, self.COLOR_HP_RED)
        self._hp_grid = self._storage_hp_grid
        self.COLOR_HP_RED = (236, 0, 0)
        try:
            self._storage_hp_get()
        finally:
            self._hp_grid = origin[0]
            self.COLOR_HP_RED = origin[1]
        return self.hp

    def lv_get(self, after_battle=False):
        """获取舰队等级（大世界中为空实现）。"""
        pass

    def fleet_low_resolve_appear(self):
        """
        检测当前舰队是否有低士气减益效果。
        """
        return self.image_color_count(
            FLEET_LOW_RESOLVE, color=FLEET_LOW_RESOLVE.color, threshold=30, count=250)

    def get_sea_grids(self):
        """
        获取当前视野中的海洋格子。

        Returns:
            SelectedGrids: 海洋格子集合，按与摄像机距离排序。
        """
        sea = []
        for local in self.view:
            if not local.predict_sea() or local.predict_current_fleet():
                continue
            # local = np.array(location) - self.camera + self.view.center_loca
            location = np.array(local.location) + self.camera - self.view.center_loca
            location = tuple(location.tolist())
            if location == self.fleet_current or location not in self.map:
                continue
            sea.append(self.map[location])

        if len(self.fleet_current):
            center = self.fleet_current
        else:
            center = self.camera
        return SelectedGrids(sea).sort_by_camera_distance(center)

    def wait_until_camera_stable(self, skip_first_screenshot=True):
        """
        等待 homo_loca 稳定。DETECTION_BACKEND 必须为 'homography'。
        """
        logger.hr('等待摄像机稳定')
        record = None
        # 卡死兜底：正常镜头恢复 3~5 秒内完成；深渊海域等场景瓦片匹配
        # 周期性失败，homo_loca 在正确值与错误值间交替振荡，稳定确认窗
        # 永远被打断，循环会卡死至 GameStuckError。连续 150 轮(~50 秒)
        # 未确认稳定视为画面检测异常，跳出由外层循环重新截图决策（点击
        # 会促使镜头重新对齐恢复，与手动干预同原理）。
        stall_count = 0
        confirm_timer = Timer(0.6, count=2).start()
        for _ in self.loop(skip_first=skip_first_screenshot):
            self.update_os()
            current = self.view.backend.homo_loca
            logger.attr('单应位置', current)
            stall_count += 1
            if record is None or (current is not None and np.linalg.norm(np.subtract(current, record)) < 3):
                if confirm_timer.reached():
                    break
            else:
                confirm_timer.reset()
            if stall_count >= 150:
                logger.warning('[大世界-摄像机] 长时间未确认稳定，画面检测异常，跳出稳定等待')
                break

            record = current

        logger.info('[大世界-摄像机] 摄像机已稳定')

    def wait_until_walk_stable(self, confirm_timer=None, skip_first_screenshot=False, walk_out_of_step=True, drop=None):
        """
        等待 homo_loca 稳定。DETECTION_BACKEND 必须为 'homography'。

        Args:
            confirm_timer (Timer): 确认计时器。
            skip_first_screenshot (bool): 是否跳过第一次截图。
            walk_out_of_step (bool): 是否捕获 walk_out_of_step 错误。
                默认为 True，深渊海域中使用 False。
            drop (DropImage): 掉落记录对象。

        Returns:
            str: 舰队途中遇到的事件，如 'event'、'search'、'akashi'、'combat'，
                或其组合如 'event_akashi'、'event_combat'，无事件时返回空字符串 ''。

        Raises:
            MapWalkError: 无法到达目标格子时抛出。
        """
        logger.hr('等待移动稳定')
        record = None
        enemy_searching_appear = False
        self.device.screenshot_interval_set(0.35)
        if confirm_timer is None:
            confirm_timer = Timer(0.8, count=2)
        result = set()
        # 记录剧情历史以清除点击记录
        clicked_story = False
        clicked_story_count = 0

        confirm_timer.reset()
        # 卡死兜底：正常移动确认最长实测 54 秒/144 帧（长距离移动+检测
        # 坏帧混杂，稳定窗靠振荡中偶然凑齐）；深渊海域死锁段实测 61 秒/
        # 166 帧后 GameStuckError 崩任务（敏感任务禁止自动重启）。
        # stall_count 统计"在地图内、无事件、未确认稳定"的连续轮次：
        # 事件分支(战斗/剧情/明石等)continue 不计数，IN_MAP 不匹配
        # (战斗/弹窗画面)清零，仅画面在地图内且确认不通过时累积。
        # 上限 200 轮(~70 秒)：正常段(最大 144 帧)留有余量，死锁段
        # (166 帧)可截住；即使正常段超限误跳出，代价仅是外层重新点击
        # 重试一轮(点击促使镜头重新对齐恢复，与手动干预同原理)，远好
        # 于 GameStuckError 停机。
        stall_count = 0

        def abyssal_expected_end():
            # 添加 handle_map_event() 因为 OSCombat.combat_status() 会移除 get_items
            if self.handle_map_event(drop=drop):
                return False
            return self.is_in_map()

        for _ in self.loop(skip_first=skip_first_screenshot):
            # 地图事件
            event = self.handle_map_event(drop=drop)
            if event:
                confirm_timer.reset()
                result.add('event')
                if event == 'story_skip':
                    clicked_story = True
                    clicked_story_count += 1
                    # 清除点击记录，避免塞壬扫描装置中超过 6 个选项导致的 GameTooManyClickError
                    # 塞壬扫描装置中提交物品的流程为
                    # STORY_OPTION_2_OF_3 -> POPUP_CONFIRM_STORY_SKIP
                    # 两个操作都返回 'story_skip' 事件
                    # 连续 2 次 story_skip 表示提交了塞壬扫描装置
                    if clicked_story_count >= 11:
                        logger.info('[大世界-剧情] 剧情中连续选项')
                        self.device.click_record_clear()
                        clicked_story_count = 0
                elif event == 'map_get_items':
                    # story_skip -> map_get_items 表示收到了深渊进度奖励
                    if clicked_story:
                        logger.info('[大世界-剧情] 从剧情获得物品')
                        self.device.click_record_clear()
                        clicked_story = False
                    clicked_story_count = 0
                else:
                    # 处理了其他事件，清除历史记录
                    clicked_story = False
                    clicked_story_count = 0
                continue
            if self.handle_retirement():
                confirm_timer.reset()
                continue
            if self.handle_walk_out_of_step():
                if walk_out_of_step:
                    raise MapWalkError('walk_out_of_step')
                else:
                    continue
            if self.handle_popup_confirm('WALK_UNTIL_STABLE'):
                confirm_timer.reset()
                continue
            if self.handle_manjuu():
                confirm_timer.reset()
                continue

            # 意外点击
            if self.is_in_globe():
                self.os_globe_goto_map()
                confirm_timer.reset()
                continue
            if self.is_in_storage():
                self.storage_quit()
                confirm_timer.reset()
                continue
            if self.is_in_os_mission():
                self.os_mission_quit()
                confirm_timer.reset()
                continue
            if self.handle_os_game_tips():
                confirm_timer.reset()
                continue
            if self.is_in_map_order():
                self.order_quit()
                confirm_timer.reset()
                continue

            # 战斗
            if self.combat_appear():
                # 使用 ui_back() 进行测试，因为每月深渊日志太少。
                # self.ui_back(check_button=self.is_in_map)
                self.combat(expected_end=abyssal_expected_end, fleet_index=self.fleet_show_index, save_get_items=drop)
                confirm_timer.reset()
                result.add('event')
                continue

            # 明石商店
            if self.appear(PORT_SUPPLY_CHECK, offset=(20, 20)):
                self.interval_clear(PORT_SUPPLY_CHECK)
                self.handle_akashi_supply_buy(CLICK_SAFE_AREA)
                confirm_timer.reset()
                result.add('akashi')
                continue

            # 游戏 bug：上一个已清理海域的 AUTO_SEARCH_REWARD 弹窗
            if self.appear_then_click(AUTO_SEARCH_REWARD, offset=(50, 50), interval=3):
                confirm_timer.reset()
                continue

            # 敌人搜索
            if not enemy_searching_appear and self.enemy_searching_appear():
                enemy_searching_appear = True
                confirm_timer.reset()
                continue
            else:
                if enemy_searching_appear:
                    self.handle_enemy_flashing()
                    self.device.sleep(0.3)
                    logger.info('[大世界-战斗] 敌舰搜索出现')
                    enemy_searching_appear = False
                    confirm_timer.reset()
                    result.add('search')
                if self.is_in_map():
                    self.enemy_searching_color_initial()

            # 到达检测
            # 检查颜色，因为解锁时屏幕会变黑。
            # 直接使用 IN_MAP，本质上是 `self.is_in_map() and IN_MAP.match_template_color()`
            if self.match_template_color(IN_MAP, offset=(200, 5), threshold=50):
                self.update_os()
                current = self.view.backend.homo_loca
                logger.attr('单应位置', current)
                # 已知最大距离为 4.48px，homo_loca 在 (56, 60) 和 (52, 58) 之间
                if record is None or (current is not None and np.linalg.norm(np.subtract(current, record)) < 5.5):
                    if confirm_timer.reached():
                        break
                else:
                    confirm_timer.reset()
                record = current
                stall_count += 1
                if stall_count >= 200:
                    logger.warning('[大世界-移动] 长时间未确认稳定，画面检测异常，跳出稳定等待')
                    break
            else:
                confirm_timer.reset()
                stall_count = 0

        result = '_'.join(result)
        logger.info(f'[大世界-移动] 移动已稳定, 结果: {result}')
        self.device.screenshot_interval_set()
        return result

    def port_goto(self, allow_port_arrive=True):
        """
        简单的港口导航实现，通过雷达搜索港口。

        在大世界中，舰队移动时摄像机始终跟随，会干扰 `self.goto()`。
        多数情况下使用自律寻敌清理地图，经典方法已弃用。
        但仍需要将舰队移向港口，此方法用于该场景。

        Raises:
            MapWalkError: 无法到达目标格子时抛出。
                可能点击了陆地、港口中心或舰队自身。
        """
        confirm_timer = Timer(3, count=6).start()
        while 1:
            # 计算目的地
            grid = self.radar.port_predict(self.device.image)
            logger.info(f'[大世界-港口] 港口路径在 {grid}')
            if grid is None:
                self.device.screenshot()
                continue

            radar_arrive = np.linalg.norm(grid) == 0
            port_arrive = self.appear(PORT_ENTER, offset=(20, 20))
            if allow_port_arrive and port_arrive:
                logger.info('[大世界-港口] 到达港口 (port_arrive)')
                break
            elif allow_port_arrive and (not port_arrive and radar_arrive):
                if confirm_timer.reached():
                    logger.warning('[大世界-港口] 雷达上到达港口但港口入口未出现')
                    raise MapWalkError
                else:
                    logger.info('[大世界-港口] 雷达上到达港口但港口入口未出现，确认中')
                    self.device.screenshot()
                    continue
            elif not allow_port_arrive and radar_arrive:
                logger.info('[大世界-港口] 到达港口 (radar_arrive)')
                break
            else:
                confirm_timer.reset()

            # 更新本地视野
            self.update_os()
            self.predict()

            # 点击路径点
            grid = point_limit(grid, area=(-4, -2, 3, 2))
            grid = self.convert_radar_to_local(grid)
            self.device.click(grid)

            # 等待到达
            self.wait_until_walk_stable()

    def fleet_set(self, index=1, skip_first_screenshot=True):
        """
        Args:
            index (int): Target fleet_current_index
            skip_first_screenshot (bool):

        Returns:
            bool: If switched.
        """
        logger.hr(f'舰队设置为 {index}')
        if self.fleet_selector.ensure_to_be(index):
            self.wait_until_camera_stable()
            return True
        else:
            return False

    def storage_fleet_set(self, index=1, skip_first_screenshot=True):
        """
        Args:
            index (int): Target fleet_current_index
            skip_first_screenshot (bool):

        Returns:
            bool: If switched.
        """
        logger.hr(f'舰队设置为 {index}')
        return self.storage_fleet_selector.ensure_to_be(index)

    def parse_fleet_filter(self):
        """
        Returns:
            list: List of BossFleet or str. Such as [Fleet-4, 'CallSubmarine', Fleet-2, Fleet-3, Fleet-1].
        """
        FLEET_FILTER.load(self.config.OpsiFleetFilter_Filter)
        fleets = FLEET_FILTER.apply([BossFleet(f) for f in [1, 2, 3, 4]])

        # 设置待命位置
        standby_list = [(-1, -1), (0, -1), (1, -1)]
        index = 0
        for fleet in fleets:
            if isinstance(fleet, BossFleet) and index < len(standby_list):
                fleet.standby_loca = standby_list[index]
                index += 1

        return fleets

    def relative_goto(self, has_fleet_step=False, near_by=False, relative_position=(0, 0), index=0, **kwargs):
        """根据雷达选择的相对位置导航舰队。

        更新本地视野后，通过雷达选择目标格子并点击移动。

        Args:
            has_fleet_step (bool): 是否限制舰队移动步数。
            near_by (bool): 是否按距离排序选择最近的格子。
            relative_position (tuple[int, int]): 相对偏移量。
            index (int): 选择第几个匹配的格子。
            **kwargs: 传递给 radar.select() 的筛选条件。
        """
        logger.hr('相对移动')
        logger.info(f'[大世界-移动] 相对移动, {dict_to_kv(kwargs)}')

        # 更新本地视野
        # 不截图，复用旧截图
        self.update_os()
        self.predict()
        self.predict_radar()

        # 计算目的地
        grids = self.radar.select(**kwargs)
        if near_by:
            grids = grids.sort_by_camera_distance((0, 0))
        if grids:
            # 点击路径点
            grid = np.add(location_ensure(grids[index]), relative_position)

            grid = point_limit(grid, area=(-4, -2, 3, 2))
            if has_fleet_step:
                grid = limit_walk(grid)
            grid = self.convert_radar_to_local(grid)
            self.device.click(grid)
        else:
            logger.info('[大世界-移动] 无目标位置，停止')

        # 等待到达
        # 使用新截图
        self.wait_until_walk_stable(confirm_timer=Timer(1.5, count=4), walk_out_of_step=False)

    def go_month_boss_room(self, is_normal=True):
        """导航到月度 Boss 房间入口并进入。

        先移动到入口下方 2 格处，再进入 Boss 房间。
        普通模式和困难模式使用不同的交互方式。

        Args:
            is_normal (bool): 是否为普通模式。困难模式使用问号导航。
        """
        logger.hr('前往房间入口')
        logger.info(f'[大世界-移动] 前往房间入口, 是否普通={is_normal}')
        while 1:
            if self.appear(MAP_EXIT, offset=(20, 20)):
                break

            # 入口下方 2 格
            self.relative_goto(has_fleet_step=True, near_by=True, relative_position=(3, -2), is_port=True)

            self.update_os()
            self.predict()
            self.predict_radar()
            grid = self.radar.select(is_port=True).first_or_none()
            if grid is not None and grid.location == (-3, 2):
                logger.info('[大世界-移动] 在房间入口')
                break

        logger.hr('进入房间入口')
        while 1:
            if self.appear(MAP_EXIT, offset=(20, 20)):
                logger.info('[大世界-移动] 已进入Boss房间')
                break

            if is_normal:
                self.relative_goto(has_fleet_step=True, near_by=True, is_exclamation=True)
            else:
                if self.radar.select(is_exclamation=True).count:
                    logger.warning('[大世界-移动] 尝试进入月度Boss困难模式但存在感叹号')
                    self.relative_goto(has_fleet_step=True, near_by=True, is_exclamation=True)
                else:
                    self.relative_goto(has_fleet_step=True, near_by=True, is_question=True)

    def question_goto(self, has_fleet_step=False):
        """导航到最近的问号标记。

        通过雷达查找问号位置并移动舰队前往。
        适用于深渊海域中寻找 Boss 入口等场景。

        Args:
            has_fleet_step (bool): 是否限制舰队移动步数。
        """
        logger.hr('前往问号')
        while 1:
            # 游戏 bug：上一个已清理海域的 AUTO_SEARCH_REWARD 弹窗
            if self.appear_then_click(AUTO_SEARCH_REWARD, offset=(50, 50), interval=3):
                self.device.screenshot()
                continue

            # 更新本地视野
            # 不截图，复用旧截图
            self.update_os()
            self.predict()
            self.predict_radar()

            # 计算目的地
            grids = self.radar.select(is_question=True)
            if grids:
                # 点击路径点
                grid = location_ensure(grids[0])
                grid = point_limit(grid, area=(-4, -2, 3, 2))
                if has_fleet_step:
                    grid = limit_walk(grid)
                grid = self.convert_radar_to_local(grid)
                self.device.click(grid)
            else:
                logger.info('[大世界-移动] 无问号可前往，停止')
                break

            # 等待到达
            # 使用新截图
            self.wait_until_walk_stable(confirm_timer=Timer(1.5, count=4), walk_out_of_step=False)

    def month_boss_goto_additional(self, location=(0, 0), has_fleet_step=False, drop=None):
        """导航到月度 Boss 区域入口。

        通过问号的相对位置定位 Boss 区域入口并前往。

        Args:
            location (tuple[int, int]): 位置偏移量。
            has_fleet_step (bool): 是否限制舰队移动步数。
            drop: 掉落记录对象。
        """
        self.update_os()
        self.predict()
        self.predict_radar()

        # 计算目的地
        grids = self.radar.select(is_question=True)
        if grids:
            # 点击路径点
            grid = np.add(location_ensure(grids[0]), location)
            # 使用问号的相对位置来定位 Boss 区域入口
            grid = np.add(grid, (1, -6))
            grid = point_limit(grid, area=(-4, -2, 3, 2))
            if has_fleet_step:
                grid = limit_walk(grid)
            if grid == (0, 0):
                logger.info(f'[大世界-移动] 到达目的地: Boss {location}')
            grid = self.convert_radar_to_local(grid)
            self.device.click(grid)
        else:
            logger.info('[大世界-移动] 无Boss可前往，停止')
        self.wait_until_walk_stable(confirm_timer=Timer(1.5, count=4), walk_out_of_step=False, drop=drop)

    def boss_goto(self, location=(0, 0), has_fleet_step=False, drop=None, is_month=False):
        """导航到 Boss 位置并准备战斗。

        Args:
            location (tuple[int, int]): 位置偏移量。
            has_fleet_step (bool): 是否限制舰队移动步数。
            drop: 掉落记录对象。
            is_month (bool): 是否为月度 Boss。
        """
        logger.hr('前往Boss')

        if is_month:
            self.month_boss_goto_additional(location=location, has_fleet_step=has_fleet_step, drop=drop)

        while 1:
            # 更新本地视野
            # 不截图，复用旧截图
            self.update_os()
            self.predict()
            self.predict_radar()

            # 计算目的地
            grids = self.radar.select(is_enemy=True)
            if grids:
                # 点击路径点
                grid = np.add(location_ensure(grids[0]), location)
                grid = point_limit(grid, area=(-4, -2, 3, 2))
                if has_fleet_step:
                    grid = limit_walk(grid)
                if grid == (0, 0):
                    logger.info(f'[大世界-移动] 到达目的地: Boss {location}')
                    break
                grid = self.convert_radar_to_local(grid)
                self.device.click(grid)
            else:
                logger.info('[大世界-移动] 无Boss可前往，停止')
                break

            # 等待到达
            # 使用新截图
            self.wait_until_walk_stable(confirm_timer=Timer(1.5, count=4), walk_out_of_step=False, drop=drop)

    def get_boss_leave_button(self):
        """获取 Boss 区域的离开按钮位置。

        通过检测当前舰队或被塞壬捕获的舰队位置，
        计算离开按钮的点击区域。

        Returns:
            Button: 离开按钮，如果当前舰队可见则返回 None。
        """
        for grid in self.view:
            if grid.predict_current_fleet():
                return None

        grids = [grid for grid in self.view if grid.predict_caught_by_siren()]
        if len(grids) == 1:
            center = grids[0]
        elif len(grids) > 1:
            logger.warning(f'[大世界-舰队] 在Boss处发现多个舰队 ({grids})，使用中心的一个')
            center = SelectedGrids(grids).sort_by_camera_distance(self.view.center_loca)[0]
        else:
            logger.warning('[大世界-舰队] Boss处无舰队，使用摄像机中心代替')
            center = self.view[self.view.center_loca]

        logger.info(f'[大世界-舰队] Boss处的舰队: {center}')
        # 中心格子左侧半个格子。
        area = corner2inner(center.grid2screen(area2corner((1, 0.25, 1.5, 0.75))))
        button = Button(area=area, color=(), button=area, name='BOSS_LEAVE')
        return button

    def boss_leave(self):
        """
        离开 Boss 区域。

        Pages:
            in: is_in_map() 或 combat_appear()
            out: is_in_map(), 舰队不在 Boss 区域中。
        """
        logger.hr('离开Boss')
        # 更新本地视野
        self.update_os()
        self.predict()

        click_timer = Timer(3)
        pause_interval = Timer(0.5, count=1)
        for _ in self.loop():
            # 结束条件
            if self.is_in_map():
                self.predict_radar()
                if self.radar.select(is_enemy=True):
                    logger.info('[大世界-舰队] 舰队离开Boss，找到Boss')
                    break

            # 意外重新进入 Boss
            if pause_interval.reached():
                if self.appear(BATTLE_PREPARATION):
                    logger.info(f'{BATTLE_PREPARATION} -> {BACK_ARROW}')
                    self.device.click(BACK_ARROW)
                    pause_interval.reset()
                    continue
                if self.appear(SIREN_PREPARATION, offset=(20, 20)):
                    logger.info(f'{SIREN_PREPARATION} -> {BACK_ARROW}')
                    self.device.click(BACK_ARROW)
                    pause_interval.reset()
                    continue
                pause = self.is_combat_executing()
                if pause:
                    self.device.click(pause)
                    self.interval_reset(MAINTENANCE_ANNOUNCE)
                    pause_interval.reset()
                    continue
            if self.handle_combat_quit():
                self.interval_reset(MAINTENANCE_ANNOUNCE)
                pause_interval.reset()
                continue
            if self.handle_combat_quit_reconfirm():
                self.interval_reset(MAINTENANCE_ANNOUNCE)
                pause_interval.reset()
                continue

            # 点击离开按钮
            if self.is_in_map() and click_timer.reached():
                button = self.get_boss_leave_button()
                if button is not None:
                    self.device.click(button)
                    click_timer.reset()
                    continue
                else:
                    logger.info('[大世界-舰队] 舰队离开Boss，找到当前舰队')
                    break

    def boss_clear(self, has_fleet_step=True, is_month=False, allow_submarine_call=True):
        """
        所有舰队轮流攻击 Boss。

        Args:
            has_fleet_step (bool): 是否限制舰队移动步数。
            is_month (bool): 是否为月度Boss。
            allow_submarine_call (bool): 是否允许呼叫潜艇。

        Returns:
            bool: 是否成功击败 Boss。

        Pages:
            in: 塞壬日志（深渊），Boss 已出现。
            out: 成功时为危险或安全海域；失败时仍在深渊中。
        """
        logger.hr(f'清除Boss', level=1)

        fleets = self.parse_fleet_filter()
        with self.stat.new(
                genre=inflection.underscore(self.config.task.command),
                method=self.config.DropRecord_OpsiRecord
        ) as drop:
            for fleet in fleets:
                logger.hr(f'回合: {fleet}', level=2)
                if not isinstance(fleet, BossFleet):
                    if allow_submarine_call:
                        self.os_order_execute(recon_scan=False, submarine_call=True)
                    else:
                        logger.info(f'[大世界-舰队] 在深渊中跳过舰队筛选顺序 `{fleet}`')
                    continue

                # 切换舰队
                if self.fleet_set(fleet.fleet_index):
                    pass
                else:
                    # 如果舰队不存在则重新聚焦摄像机
                    others = [f for f in fleets if isinstance(f, BossFleet) and f != fleet]
                    if len(others):
                        other: BossFleet = others[0]
                        self.fleet_set(other.fleet_index)
                        self.fleet_set(fleet.fleet_index)
                    else:
                        logger.warning(f'[大世界-舰队] 从 {fleets} 无其他舰队，跳过重新聚焦')
                        pass

                # 检查舰队
                self.handle_os_map_fleet_lock(enable=False)
                if self.fleet_low_resolve_appear():
                    logger.warning('[大世界-舰队] 因低决心debuff跳过使用当前舰队')
                    self.boss_goto(location=fleet.standby_loca, has_fleet_step=has_fleet_step, drop=drop,
                                   is_month=is_month)
                    continue

                # 确保 Boss 出现
                if is_month:
                    while not self.radar.select(is_enemy=True):
                        self.relative_goto(has_fleet_step=True, is_question=True, relative_position=(1, -6), index=0)
                        try:
                            self.relative_goto(has_fleet_step=True, is_question=True, index=1)
                        except IndexError:
                            self.relative_goto(has_fleet_step=True, is_question=True, relative_position=(1, -7),
                                               index=0)

                # 攻击
                self.boss_goto(location=(0, 0), has_fleet_step=has_fleet_step, drop=drop, is_month=is_month)

                # 结束条件
                self.predict_radar()
                if self.radar.select(is_question=True):
                    logger.info('[大世界-战斗] Boss已清除')
                    if drop.count:
                        drop.add(self.device.image)
                    self.map_exit()
                    return True

                # 待命
                self.boss_leave()
                if fleet.standby_loca != (0, 0):
                    self.boss_goto(location=fleet.standby_loca, has_fleet_step=has_fleet_step, drop=drop)
                else:
                    if drop.count:
                        drop.add(self.device.image)
                    break

        logger.critical('[大世界] 无法击败boss，舰队已耗尽')
        return False

    def run_abyssal(self):
        """
        处理双重确认并攻击深渊（塞壬日志）Boss。
        即使舰队筛选器包含 `CallSubmarine` 条目，深渊也不会使用潜艇命令。

        Returns:
            bool: 是否成功击败 Boss。

        Pages:
            in: 塞壬日志（深渊）。
            out: 成功时为危险或安全海域；失败时仍在深渊中。
        """
        self.handle_os_map_fleet_lock(enable=False)

        def is_at_front(grid):
            # 格子位置通常为 (0, -2)
            x, y = grid.location
            return (abs(x) <= abs(y)) and (y < 0)

        while 1:
            self.device.screenshot()
            self.question_goto(has_fleet_step=True)

            if self.radar.select(is_enemy=True).filter(is_at_front):
                logger.info('[大世界-搜索] 在前方找到Boss')
                break
            else:
                logger.info('[大世界-搜索] 前方无Boss，重试问号前往')
                continue

        result = self.boss_clear(has_fleet_step=True, allow_submarine_call=False)
        return result

    def get_stronghold_percentage(self):
        """
        获取塞壬要塞的清理进度。

        Returns:
            str: 通常为 ['100', '80', '60', '40', '20', '0'] 之一。
        """
        ocr = PercentageOcr(STRONGHOLD_PERCENTAGE, letter=(255, 255, 255), threshold=128, name='STRONGHOLD_PERCENTAGE')
        result = ocr.ocr(self.device.image)
        result = result.rstrip('7Kk')
        for starter in ['100', '80', '60', '40', '20', '0']:
            if result.startswith(starter):
                result = starter
                logger.attr('要塞百分比', result)
                return result

        logger.warning(f'[大世界-要塞] 异常的要塞百分比: {result}')
        return result

    def get_second_fleet(self):
        """
        获取第二支舰队，用于解锁需要 2 支舰队的机关。

        Returns:
            int: 第二支舰队的索引。
        """
        current = self.fleet_selector.get()
        if current == 1:
            second = 2
        else:
            second = 1
        logger.attr('第二舰队', second)
        return second

    @staticmethod
    def fleet_walk_limit(outside, step=3):
        if np.linalg.norm(outside) <= 3:
            return outside
        if step == 1:
            grids = np.array([
                (0, -1), (0, 1), (-1, 0), (1, 0),
            ])
        else:
            grids = np.array([
                (0, -3), (0, 3), (-3, 0), (3, 0),
                (2, -2), (2, 2), (-2, 2), (-2, -2),
            ])
        degree = np.sum(grids * outside, axis=1) / np.linalg.norm(grids, axis=1) / np.linalg.norm(outside)
        return grids[np.argmax(degree)]

    _nearest_object_click_timer = Timer(2)
    # 卡死自愈状态。设计原则：点击决策保持作者原版"直接点最近目标"，
    # 附加机制只做"点击后舰队没动"的检测与自愈，不干预正常路径。
    _nearest_object_last_homo = None  # 上次点击时刻的镜头全局位置(单应位置)
    _nearest_object_stuck_count = 0   # 连续点击后舰队未移动的次数
    _nearest_object_last_focus_stuck = 0  # 上次镜头恢复时的 stuck 计数
    _nearest_object_abandoned = []    # 已放弃目标的截断格列表
    _nearest_object_last_click = None  # 上次点击的雷达格(舰队原点坐标系)
    _nearest_object_obstacles = []     # 障碍格列表(岛屿/被挡)。舰队没动期间
    # 雷达坐标系不变、标记有效；舰队一旦移动即整体清空(坐标系已漂移)。
    _nearest_object_detour_steps = 0  # 绕行移动累计步数(正常推进成功即清零)
    _nearest_object_last_signature = None  # 上轮雷达实体布局指纹
    # 雷达目标选择的视野截断范围，必须与 radar.nearest_object 的默认
    # camera_sight 保持一致：放弃目标时用同一范围算截断格，拉黑才不会失配
    NEAREST_OBJECT_CAMERA_SIGHT = (-4, -3, 3, 3)
    # 连续点击未移动的次数阈值：达 2 次且舰队偏离镜头中心 → 恢复镜头；
    # 达 3 次且舰队在镜头中心 → 标记点击格为障碍改走侧翼；达 10 次 →
    # 兜底放弃目标(防状态机卡死)
    NEAREST_OBJECT_FOCUS_STUCK = 2
    NEAREST_OBJECT_OBSTACLE_STUCK = 3
    NEAREST_OBJECT_ABANDON_STUCK = 10
    # 绕行步数上限：超过说明地形封死通往该目标的路径，放弃换下一个。
    # 地图边缘沿边绕行需要较多小步，放宽到 12
    NEAREST_OBJECT_DETOUR_LIMIT = 12

    def click_nearest_object(self):
        if not self._nearest_object_click_timer.reached():
            return False
        if not self.appear(MAP_GOTO_GLOBE, offset=(200, 20)):
            return False
        if self.appear(PORT_ENTER, offset=(20, 20)):
            return False

        self.update_os()
        self.view.predict()
        self.radar.predict(self.device.image)
        self.radar.show()

        # 只认配置的主舰队：非主舰队时暂停寻路。防止误点其他舰队(2/3/4)
        # 导致镜头切换后坐标系错乱，以及手动切舰队做任务时脚本乱点。
        # get() 读地图左侧舰队编号标签(需先截图)，识别失败(0)时跳过检测。
        primary_fleet = self.config.OpsiFleet_Fleet
        current_fleet_no = self.fleet_selector.get()
        if current_fleet_no > 0 and current_fleet_no != primary_fleet:
            logger.info(f'[大世界-雷达] 当前处于舰队 {current_fleet_no}，与配置主舰队 {primary_fleet} 不符，'
                        '暂停寻路（切回主舰队后继续）')
            return False

        nearest = self.radar.nearest_object(
            camera_sight=self.NEAREST_OBJECT_CAMERA_SIGHT,
            exclude=self._nearest_object_abandoned)
        if nearest is None:
            # 可见目标全部放弃：清空重来，避免永久卡死
            if self._nearest_object_abandoned:
                logger.info('[大世界-雷达] 可见目标均已放弃，清空放弃列表重新寻路')
                self._nearest_object_abandoned = []
                self._nearest_object_stuck_count = 0
                self._nearest_object_last_focus_stuck = 0
            self._nearest_object_click_timer.reset()
            return False

        # 目标截断格：拉黑与侧翼方向计算使用
        homo = self.view.backend.homo_loca
        target = tuple(int(x) for x in point_limit(
            nearest.location, area=self.NEAREST_OBJECT_CAMERA_SIGHT))

        # 卡死检测：判定"点击后舰队是否移动"用双信号。
        # 信号1 homo_loca：镜头相对瓦片网格的偏移，模 HOMO_TILE 周期值，
        #   整格移动同余不变(实测部分移动仅 1px 抖动)，单用会漏判；
        # 信号2 雷达实体指纹：雷达以舰队为原点，舰队一动所有目标坐标整体
        #   平移、指纹必变；真卡死时目标静止、指纹不变。
        # 两者任一变化即判移动——宁可误判移动(仅重置计数，代价小)，
        # 不可漏判(会误标障碍格、误放弃目标)。
        signature = self._nearest_object_radar_signature()
        if self._nearest_object_last_homo is not None and homo is not None:
            homo_moved = np.linalg.norm(
                np.array(homo, dtype=float)
                - np.array(self._nearest_object_last_homo, dtype=float)) > 8
            sig_moved = self._nearest_object_last_signature is not None \
                and signature != self._nearest_object_last_signature
            moved = homo_moved or sig_moved
            if moved:
                if self._nearest_object_obstacles:
                    # 绕行移动成功：累计步数，超限说明地形封死该目标。
                    # 障碍列表保留——舰队挪一格后旧标记坐标虽有漂移，但
                    # 地图边缘连续绕行时地形基本不变，保留可避免把同样的
                    # 坑逐个重踩一遍(每格 2 秒)；误拦的格由换方向兜底
                    self._nearest_object_detour_steps += 1
                    logger.info(f'[大世界-雷达] 绕行推进 ({self._nearest_object_detour_steps})')
                    if self._nearest_object_detour_steps >= self.NEAREST_OBJECT_DETOUR_LIMIT:
                        logger.info('[大世界-雷达] 绕行步数达上限，放弃该目标')
                        self._abandon_nearest_object(target)
                        self._nearest_object_click_timer.reset()
                        return False
                else:
                    # 正常推进成功(无绕行状态)：绕行计数与旧障碍全部作废
                    self._nearest_object_detour_steps = 0
                    self._nearest_object_obstacles = []
                self._nearest_object_stuck_count = 0
                self._nearest_object_last_focus_stuck = 0
            else:
                self._nearest_object_stuck_count += 1
                logger.info(f'[大世界-雷达] 点击后舰队未移动 '
                            f'({self._nearest_object_stuck_count})')
                # 区分镜头问题与地形问题：镜头跟丢时舰队不在视野或偏离中心
                fleets = self.view.select(is_current_fleet=True)
                if fleets.count > 0:
                    offset = np.array(fleets[0].location, dtype=float) \
                        - np.array(self.view.center_loca, dtype=float)
                    camera_lost = np.linalg.norm(offset) > 2.5
                else:
                    camera_lost = True
                if camera_lost:
                    # 每 3 次未移动重试一次镜头恢复(focus 自身有失败可能，
                    # 单次标志位会卡死在"已恢复过但没成功"的状态)
                    if self._nearest_object_stuck_count \
                            >= self._nearest_object_last_focus_stuck + self.NEAREST_OBJECT_FOCUS_STUCK:
                        logger.info('[大世界-雷达] 舰队不在镜头中心，呼出菜单恢复镜头到主舰队')
                        if self.fleet_selector.focus(primary_fleet):
                            self.wait_until_camera_stable()
                        self._nearest_object_last_focus_stuck = self._nearest_object_stuck_count
                        # 镜头已恢复，本轮 view/radar 数据已过期，下轮重新预测寻路
                        self._nearest_object_click_timer.reset()
                        return False
                elif self._nearest_object_stuck_count >= self.NEAREST_OBJECT_OBSTACLE_STUCK \
                        and self._nearest_object_last_click is not None:
                    # 舰队在镜头中心但没动：上次点击的格不可达(岛屿地形/
                    # 路径被挡)。标记为障碍格，本轮点击前会自动改走侧翼。
                    # 舰队没动雷达坐标系不变，标记跨轮有效。
                    click = tuple(int(x) for x in self._nearest_object_last_click)
                    if click not in self._nearest_object_obstacles:
                        if len(self._nearest_object_obstacles) >= 16:
                            self._nearest_object_obstacles = []
                        self._nearest_object_obstacles.append(click)
                        logger.info(f'[大世界-雷达] 点击 {click} 舰队未移动，'
                                    f'标记为障碍格，改走侧翼')
                if self._nearest_object_stuck_count >= self.NEAREST_OBJECT_ABANDON_STUCK:
                    # 兜底：镜头恢复过、障碍也标了仍持续卡死，放弃目标换下一个
                    self._abandon_nearest_object(target)
                    self._nearest_object_click_timer.reset()
                    return False
        self._nearest_object_last_homo = tuple(homo) if homo is not None else None
        self._nearest_object_last_signature = signature

        step = 1 if self.appear(FLEET_EMP_DEBUFF, offset=(50, 20)) else 3
        nearest = self.fleet_walk_limit(nearest.location, step=step)

        # 障碍过滤：方向格/目标格已被标记为障碍(岛屿/路径被挡)时，
        # 改走侧翼格——从 8 个方向格中取与目标方向最接近且非障碍的一个，
        # 绕过地形后再重新朝目标走。8 个方向全被堵则放弃目标。
        click = tuple(int(x) for x in nearest)
        if click in self._nearest_object_obstacles:
            sidestep = self._sidestep_toward(target, step=step)
            if sidestep is None:
                logger.info(f'[大世界-雷达] 目标 {target} 各方向均被障碍封死，放弃该目标')
                self._abandon_nearest_object(target)
                self._nearest_object_click_timer.reset()
                return False
            logger.info(f'[大世界-雷达] {click} 为障碍格，侧翼绕行 {sidestep}')
            click = sidestep
        self._nearest_object_last_click = click

        # 不可安全点击拦截：
        # 1. 与「返回大地图」按钮区域相交的格
        # 2. 本地视野外的格——本地视野的格子对象覆盖屏幕全部位置
        #    (含没有瓦片的地图外深色区)，点击那里会退出海域
        local = self._radar_to_local_clickable(click)
        if local is None:
            if click not in self._nearest_object_obstacles:
                self._nearest_object_obstacles.append(click)
            logger.info(f'[大世界-雷达] {click} 不可安全点击(视野外/按钮区)，标记为障碍')
            self._nearest_object_click_timer.reset()
            return False
        self.device.click(local)
        self._nearest_object_click_timer.reset()
        return True

    def _radar_to_local_clickable(self, radar_grid):
        """将雷达格转为可安全点击的本地格。

        先做雷达→本地坐标转换，再排除两类不可点击位置——本地视野外
        的格，以及与「返回大地图」按钮区域相交的格(点击会退出海域)。

        注：地图边界在雷达上以虚线标示，但雷达十字线与虚线视觉上难以
        区分，检测不可靠，不再用于拦截；地图外误点由按钮区拦截和
        "点击后未移动标记障碍"机制兜底。

        Args:
            radar_grid: 雷达坐标 (x, y)。

        Returns:
            OSGrid or None: None 表示该格不可安全点击。
        """
        try:
            local = self.convert_radar_to_local(radar_grid)
        except KeyError:
            return None
        goto_globe_areas = (MAP_GOTO_GLOBE.area, MAP_GOTO_GLOBE_FOG.area)
        if any(area_cross_area(local.button, globe_area) for globe_area in goto_globe_areas):
            return None
        return local

    def _nearest_object_radar_signature(self):
        """雷达实体布局指纹：所有目标格坐标的集合。

        雷达以舰队为原点，舰队移动后所有目标坐标整体平移、指纹必变；
        舰队不动且目标静止时指纹不变。用于"点击后舰队是否移动"判定。
        """
        return frozenset(
            tuple(int(v) for v in grid.location)
            for grid in self.radar
            if grid.is_enemy or grid.is_resource or grid.is_meowfficer
            or grid.is_exclamation or grid.is_question or grid.is_archive
            or grid.is_port
        )

    def _abandon_nearest_object(self, target):
        """放弃当前目标：拉黑其截断格让 nearest_object 跳过，重置绕行状态。

        Args:
            target (tuple): 目标的视野截断格(雷达坐标系)。
        """
        logger.info(f'[大世界-雷达] 放弃目标 {target}，切换下一个')
        if target not in self._nearest_object_abandoned:
            if len(self._nearest_object_abandoned) >= 4:
                self._nearest_object_abandoned = []
            self._nearest_object_abandoned.append(target)
        self._nearest_object_stuck_count = 0
        self._nearest_object_last_focus_stuck = 0
        self._nearest_object_detour_steps = 0
        self._nearest_object_obstacles = []

    def _sidestep_toward(self, target, step):
        """目标方向被障碍堵住时，选一个绕行格。

        候选分两梯队，按与目标方向的点积降序统一排序(优先最贴近目标
        方向)：
        梯队1 作者 fleet_walk_limit 同款预设方向格(满移动力步长)；
        梯队2 单步 8 邻域——地图边缘/峡角常只剩一个相邻格可走，预设
        方向格里没有它(22:18 实测 7 个预设格全试完才碰到可走格)。
        跳过已知障碍格与超出本地视野的格(本地视野 10x7、舰队位于
        (5,4)，雷达格 x∈[-5,4]、y∈[-4,2] 才能转换点击)。
        全部不可用返回 None(调用方放弃目标)。

        Args:
            target (tuple): 目标的视野截断格(雷达坐标系)。
            step (int): 移动力步长，1(EMP 状态)或 3。

        Returns:
            tuple or None: 侧翼绕行的雷达格。
        """
        if step == 1:
            # EMP 状态一步只能走 4 个正方向，对角格点了也不会动
            sidesteps = [
                (0, -1), (0, 1), (-1, 0), (1, 0),
            ]
        else:
            sidesteps = [
                (0, -3), (0, 3), (-3, 0), (3, 0),
                (2, -2), (2, 2), (-2, 2), (-2, -2),
                (0, -1), (0, 1), (-1, 0), (1, 0),
                (-1, -1), (1, -1), (-1, 1), (1, 1),
            ]
        target_vec = np.array(target, dtype=float)
        norm = np.linalg.norm(target_vec)
        if norm < 1e-6:
            target_vec = np.array([1.0, 0.0])
            norm = 1.0

        def direction_score(grid):
            vec = np.array(grid, dtype=float)
            return np.dot(vec, target_vec) / (np.linalg.norm(vec) * norm)

        for grid in sorted(set(sidesteps), key=direction_score, reverse=True):
            if grid in self._nearest_object_obstacles:
                continue
            # 超出本地视野的格无法点击(舰队(5,4)、视野 x0-9/y0-6)
            if not (-5 <= grid[0] <= 4 and -4 <= grid[1] <= 2):
                continue
            return grid
        return None
