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
        confirm_timer = Timer(0.6, count=2).start()
        for _ in self.loop(skip_first=skip_first_screenshot):
            self.update_os()
            current = self.view.backend.homo_loca
            logger.attr('单应位置', current)
            if record is None or (current is not None and np.linalg.norm(np.subtract(current, record)) < 3):
                if confirm_timer.reached():
                    break
            else:
                confirm_timer.reset()

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
            else:
                confirm_timer.reset()

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
    # 目标级放弃：被封锁的方向(截断格+判别为不可达)与已放弃的目标
    _nearest_object_blocked = []
    _nearest_object_blocked_limit = 12  # 封锁方向数量上限，防止攒满后无路可走卡死
    _nearest_object_blocked_objects = []
    _nearest_object_blocked_objects_limit = 6  # 放弃目标数量上限，防止无限累积导致无可寻目标卡死
    _nearest_object_last_click = None
    _nearest_object_last_fleet = None
    _nearest_object_stuck_count = 0  # 连续未移动次数，用于侧翼绕行打破死胡同
    # 镜头恢复重试
    _nearest_object_camera_lost_count = 0
    _nearest_object_camera_recover_timer = Timer(30)

    def click_nearest_object(self):
        if not self._nearest_object_click_timer.reached():
            return False
        if not self.appear(MAP_GOTO_GLOBE, offset=(200, 20)):
            logger.info('[大世界-雷达] 未检测到 MAP_GOTO_GLOBE(作战总览)按钮，跳过寻路')
            return False
        if self.appear(PORT_ENTER, offset=(20, 20)):
            logger.info('[大世界-雷达] 检测到港口入口，跳过寻路')
            return False

        self.update_os()
        self.view.predict()
        self.radar.predict(self.device.image)
        self.radar.show()

        # ⑥ 只认配置的主舰队：检测当前实际舰队，非主舰队则暂停寻路，避免手动切到其他舰队
        # 做任务时脚本仍为第一舰队寻路。get() 读地图舰队编号标签，识别失败(0)时跳过检测。
        current_fleet_no = self.fleet_selector.get()
        primary_fleet = self.config.OpsiFleet_Fleet
        if current_fleet_no > 0 and current_fleet_no != primary_fleet:
            logger.info(f'[大世界-雷达] 当前处于舰队 {current_fleet_no}，与配置主舰队 {primary_fleet} 不符，暂停寻路（切回主舰队后继续）')
            self._nearest_object_click_timer.reset()
            return False

        # ⑥ 镜头恢复：当前舰队不在镜头内时，呼出菜单强制聚焦主舰队
        fleets = self.view.select(is_current_fleet=True)
        if fleets.count == 0:
            if self._nearest_object_camera_lost_count < 3 \
                    or self._nearest_object_camera_recover_timer.reached():
                logger.info(f'[大世界-雷达] 镜头未跟随舰队，呼出菜单重新聚焦到主舰队 {primary_fleet}')
                if self.fleet_selector.focus(primary_fleet):
                    self.wait_until_camera_stable()
                    self._nearest_object_camera_lost_count = 0
                    self._nearest_object_camera_recover_timer.reset()
                else:
                    self._nearest_object_camera_lost_count += 1
            else:
                self._nearest_object_camera_lost_count += 1
            self._nearest_object_click_timer.reset()
            return False

        current_fleet = tuple(int(x) for x in fleets[0].location) if len(fleets) > 0 else None
        # 监控点击是否让舰队移动，未移动则临时封锁该方向
        if self._nearest_object_last_click is not None:
            if current_fleet == self._nearest_object_last_fleet:
                self._nearest_object_stuck_count += 1
                blocked = self._normalize_coord(self._nearest_object_last_click)
                if blocked not in self._nearest_object_blocked:
                    # 封锁数达到上限则整体清空重来，避免攒满无路可走卡死
                    if len(self._nearest_object_blocked) >= self._nearest_object_blocked_limit:
                        self._nearest_object_blocked = []
                        self._nearest_object_stuck_count = 0
                    self._nearest_object_blocked.append(blocked)
                    logger.info(f'[大世界-雷达] 点击偏移 {blocked} 舰队未移动，封锁该方向')
            else:
                # 舰队确实移动了，说明路是通的，重置卡住状态与封锁
                self._nearest_object_blocked = []
                self._nearest_object_blocked_objects = []
                self._nearest_object_stuck_count = 0

        nearest = self.radar.nearest_object(exclude=self._nearest_object_blocked_objects)
        if nearest is None:
            # 无可寻目标：通常是因为放弃列表把可见目标全排除了。
            # 清空放弃列表重试，避免无限静默卡住
            if self._nearest_object_blocked_objects:
                self._nearest_object_blocked_objects = []
                logger.info('[大世界-雷达] 无可寻目标，清空放弃列表重试')
            self._nearest_object_click_timer.reset()
            return False

        step = 1 if self.appear(FLEET_EMP_DEBUFF, offset=(50, 20)) else 3

        # ⑤⑦ 候选收缩 + 分步接近：优先目标格本身，再纳入其 8 邻域，按距目标远近排序尝试
        intended = self._normalize_coord(self.fleet_walk_limit(nearest.location, step=step))
        candidates = {
            intended,
            self._normalize_coord(nearest.location),  # 敌人格本身优先，避免被方向格顶替
        }
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.add((intended[0] + dx, intended[1] + dy))
        # 过滤已在雷达上的格子，并按与目标的距离排序
        candidates = [
            self._normalize_coord(c) for c in candidates
            if c in self.radar and self._normalize_coord(c) not in self._nearest_object_blocked
        ]
        candidates.sort(key=lambda c: np.linalg.norm(
            np.array(c) - np.array(nearest.location)))

        clicked = self._click_first_available(candidates, current_fleet)
        if clicked is not None:
            return clicked

        # 候选(9 格)都已不可达/不在视野，通常目标在镜头外(雷达能看到但镜内够不着)。
        # 先朝目标方向(原始坐标)在镜内走 1 步，镜头刷新后再循环逐步靠近；
        # 若这一步也走不出去，再落回目标级放弃/侧翼绕行，避免静默不行动。
        if self._step_toward_object():
            return True

        # 所有候选(9 格)都不可达(已封锁或不在本地视野)：目标级放弃，切换下一个
        limited = self._normalize_coord(point_limit(nearest.location, area=(-4, -3, 3, 3)))
        if tuple(limited) not in self._nearest_object_blocked_objects:
            # 放弃目标达到上限则清空重来，避免全部目标被放弃后无可寻
            if len(self._nearest_object_blocked_objects) >= self._nearest_object_blocked_objects_limit:
                self._nearest_object_blocked_objects = []
            self._nearest_object_blocked_objects.append(tuple(limited))
            logger.info(f'[大世界-雷达] 目标 {tuple(limited)} 不可达，放弃并切换下一个')

        # 连续卡住超阈值：尝试侧翼/后方绕行一格，打破死胡同，避免原地打转卡死
        if self._nearest_object_stuck_count >= 4:
            logger.info(f'[大世界-雷达] 连续未移动 {self._nearest_object_stuck_count} 次，尝试侧翼绕行')
            sidesteps = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]
            for dx, dy in sidesteps:
                candidate = (current_fleet[0] + dx, current_fleet[1] + dy)
                if candidate in self.radar and candidate not in self._nearest_object_blocked:
                    local = self._radar_to_local_clickable(candidate)
                    if local is None:
                        continue
                    self._nearest_object_stuck_count = 0
                    self._nearest_object_last_click = candidate
                    self._nearest_object_last_fleet = current_fleet
                    self.device.click(local)
                    self._nearest_object_click_timer.reset()
                    return True

        self._nearest_object_click_timer.reset()
        return False

    @staticmethod
    def _normalize_coord(loc):
        """将 numpy/int 元组统一转为纯 int 元组，保证封锁/放弃匹配稳定。"""
        return tuple(int(x) for x in loc)

    # ── 地图边界虚线检测 ──────────────────────────────────────────
    # 雷达上的虚线矩形对应大世界地图的可操作矩形边界，点击边界外(地图外)
    # 会触发返回大地图。检测虚线矩形位置用于拦截"单次寻路失败后遍历候选格
    # 时点到地图外"。检测结果按雷达画面缓存，同一轮寻路只检测一次。
    _dashed_boundary_cache = None
    _dashed_boundary_cache_ts = 0.0

    def _detect_dashed_boundary(self):
        """检测雷达上标示地图可操作区域边界的虚线矩形。

        大世界地图是矩形，其边界在右上角雷达小地图上以白色分段虚线显示
        (矩形轮廓，随舰队移动而整体平移，但始终标示地图边界)。本方法在
        当前雷达图中检测这一虚线矩形，得到四条边在雷达坐标系的位置。

        结果按时间短缓存：2 秒内复用同一份检测结果，避免同一轮寻路遍历
        候选格时反复做图像检测；超过 2 秒则重新检测，以跟随舰队移动后
        虚线位置的变化（虚线随舰队整体平移，能看到的边数因位置而异）。

        Returns:
            dict | None: {'left': x, 'right': x, 'top': y, 'bottom': y}，
                均为雷达坐标(舰队=原点)的浮点值；某侧不在视野内则为 None。
                未检测到任何可信虚线时返回 None。
        """
        import time
        if not hasattr(self, '_dashed_boundary_cache'):
            self._dashed_boundary_cache = None
            self._dashed_boundary_cache_ts = 0.0
        now = time.time()
        if self._dashed_boundary_cache is not None and (now - self._dashed_boundary_cache_ts) < 2.0:
            return self._dashed_boundary_cache
        result = None
        try:
            result = self._detect_dashed_boundary_once()
        except Exception as e:
            logger.warning(f'[大世界-雷达] 地图边界虚线检测异常: {e}')
        self._dashed_boundary_cache = result
        self._dashed_boundary_cache_ts = now
        return result

    def _detect_dashed_boundary_once(self):
        """虚线矩形检测的实际实现(单次，可抛出异常)。"""
        import numpy as _np
        try:
            image = self.device.image
        except Exception:
            return None
        if image is None or getattr(image, 'size', 0) == 0:
            return None

        radar = self.radar
        cx, cy = radar.center
        # 雷达半径(像素)：格距 * 半径格数。注意本项目的 Radar 并不保存 radius
        # 属性(构造时仅用局部变量 radius 生成 shape)，半径格数从 shape 推导。
        radius_int = abs(radar.shape[0][0])
        radius_px = int(radar.delta[0] * radius_int)
        # 裁剪略大于雷达圆盘的区域
        pad = int(radius_px * 0.25)
        x0, x1 = int(cx - radius_px - pad), int(cx + radius_px + pad)
        y0, y1 = int(cy - radius_px - pad), int(cy + radius_px + pad)
        h_img, w_img = image.shape[:2] if image.ndim == 3 else image.shape
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w_img, x1), min(h_img, y1)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        # 转灰度并提取高亮度(白色)像素。BGR -> 亮度加权。
        gray = 0.114 * crop[..., 0] + 0.587 * crop[..., 1] + 0.299 * crop[..., 2]
        white = gray > 185  # 白色虚线

        # ── 逐行/逐列找"虚线"边界 ──
        # 地图边界虚线的特点：由短白色线段沿水平/垂直方向等距排列。
        # 我们先做水平/垂直方向投影，虚线段会使对应行/列出现较小但非零的
        # 白色计数；实线十字/刻度则计数很大。通过"行内白色像素数落在
        # 一个适中区间"来筛选虚线段所在行/列。
        hh, ww = white.shape

        # 行(水平虚线)：某一行内白色像素数适中(>某阈值且远小于整行)
        row_count = white.sum(axis=1)
        col_count = white.sum(axis=0)

        # 筛选"虚线段所在行"：白色像素数在 (line_min, line_max) 之间，
        # 这一行有不少白色但不至于铺满(实线整行全白)。
        # 虚线短线段约占行宽 30-70%，实线约 100%；圆刻度环扫到也偏满。
        def _dash_rows(count, seg_fn):
            # seg_fn(row)->bool 判断该行是否为虚线候选
            pass

        # 直接做法：把 white 按行考虑，检测"一行内是否由若干等距短白段组成"
        # 实现：对该行作水平方向一维形态学开运算(消除孤立单点)后，统计段数
        # 为避免复杂，简化为：对每行统计"连续白段"数量，虚线段行通常有
        # 多个不相连的白段(3+段)，实线行只有1段。
        def _segments_count(seq):
            # 统计一维二值序列中连续 True 的段数
            seq = _np.asarray(seq, dtype=bool)
            if seq.size == 0:
                return 0
            # 转换点到段：diff
            d = _np.diff(seq.astype(_np.int8))
            return int(_np.sum(d == 1))

        h_seg = _np.array([_segments_count(white[r]) for r in range(hh)], dtype=_np.int32)
        v_seg = _np.array([_segments_count(white[:, c]) for c in range(ww)], dtype=_np.int32)

        # 虚线段行：段数 >= 3（多个短白段）；实线行段数为1
        h_dash = _np.where(h_seg >= 3)[0]
        v_dash = _np.where(v_seg >= 3)[0]

        # 垂直方向(左右边)是垂直线，对应"列"上白色段数>=3 → v_dash
        # 水平方向(上下边)是水平线，对应"行"上白色段数>=3 → h_dash

        # 舰队位于地图内部，任何一条地图边界虚线只会落在舰队的一侧：
        # 中心左侧的垂直虚线只可能是左边界(left)，中心右侧的只可能右边界(right)；
        # 中心上方的水平虚线只可能上边界(top)，下方的只可能下边界(bottom)。
        # 因此按"舰队中心"把虚线段拆到两侧、每侧取最靠外侧的那条；缺边方向
        # 保持 None(交由 _radar_to_local_clickable 跳过该方向的拦截，不误拦)。
        boundary = self._assign_boundaries(
            v_dash, h_dash,
            int(cx - x0), int(cy - y0),
            float(radar.delta[0]), float(radar.delta[1]),
        )
        # 至少检测到一条边即可用于拦截；全部为空(无可信虚线)才返回 None
        if all(v is None for v in boundary.values()):
            return None
        logger.debug(f'[大世界-雷达] 地图边界虚线: '
                     f'left={boundary["left"]} right={boundary["right"]} '
                     f'top={boundary["top"]} bottom={boundary["bottom"]}')
        return boundary

    @staticmethod
    def _assign_boundaries(v_dash, h_dash, center_col, center_row, delta_x, delta_y):
        """把检测到的虚线段行列按舰队中心分侧，分配上下左右四条边界。

        单个方向上若有多条虚线段(理论上仅矩形两边，但可能混入误检)，取最靠
        外侧的那条(最左/最右/最上/最下)；缺边方向保持 None。

        Args:
            v_dash: 垂直虚线段所在像素列(升序 numpy 数组)。
            h_dash: 水平虚线段所在像素行(升序 numpy 数组)。
            center_col: 雷达中心(舰队)在裁剪图中的像素列。
            center_row: 雷达中心(舰队)在裁剪图中的像素行。
            delta_x/delta_y: 每雷达格对应的像素数。

        Returns:
            dict: {'left','right','top','bottom'}，雷达坐标浮点值，缺边为 None。
        """
        boundary = {'left': None, 'right': None, 'top': None, 'bottom': None}
        left_cols = v_dash[v_dash < center_col]
        right_cols = v_dash[v_dash > center_col]
        if len(left_cols):
            boundary['left'] = float((left_cols[0] - center_col) / delta_x)
        if len(right_cols):
            boundary['right'] = float((right_cols[-1] - center_col) / delta_x)
        top_rows = h_dash[h_dash < center_row]
        bottom_rows = h_dash[h_dash > center_row]
        if len(top_rows):
            boundary['top'] = float((top_rows[0] - center_row) / delta_y)
        if len(bottom_rows):
            boundary['bottom'] = float((bottom_rows[-1] - center_row) / delta_y)
        return boundary

    def _radar_to_local_clickable(self, radar_grid):
        """将雷达格转为可安全点击的本地格。

        与 _click_first_available 共用：先做雷达→本地坐标转换，再排除两类
        不可点击位置——与「返回大地图」按钮区域相交的、以及位于地图网格
        边界之外(点击会退出大世界地图)的格子。

        Args:
            radar_grid: 雷达坐标 (x, y)。

        Returns:
            OSGrid or None: None 表示该格不可安全点击。
        """
        try:
            local = self.convert_radar_to_local(radar_grid)
        except KeyError:
            # 目标格不在当前镜头视野内（含地图边界外的格子）→ 不可安全点击
            return None
        goto_globe_areas = (MAP_GOTO_GLOBE.area, MAP_GOTO_GLOBE_FOG.area)
        if any(area_cross_area(local.button, globe_area) for globe_area in goto_globe_areas):
            # 该格与「返回大地图」按钮区域相交，点击会退出大世界地图
            return None
        # 地图边界虚线拦截：候选格落在雷达虚线矩形(地图可操作边界)外时不点，
        # 避免「单次寻路失败后遍历候选格时点到地图外→返回大地图」。
        boundary = self._detect_dashed_boundary()
        if boundary is not None:
            rg = self._normalize_coord(radar_grid)
            if (boundary['left'] is not None and rg[0] < boundary['left']) \
                    or (boundary['right'] is not None and rg[0] > boundary['right']) \
                    or (boundary['top'] is not None and rg[1] < boundary['top']) \
                    or (boundary['bottom'] is not None and rg[1] > boundary['bottom']):
                return None
        return local

    def _step_toward_object(self):
        """朝雷达上最近的远距离目标方向，在镜头内走一步。

        用于目标在镜头外、候选(9格)全部不可达时的分步接近：
        取目标的原始坐标(不受 camera_sight 截断)算方向，在舰队周围的
        可点击邻格中选最朝目标方向的一格走 1 步，镜头跟随舰队刷新后再
        循环靠近。失败(无可走格)返回 False，交由上层落回放弃/绕行。

        方向计算与候选格均使用雷达坐标系(舰队位于雷达中心 (0,0))。
        """
        target = self.radar.nearest_object(exclude=self._nearest_object_blocked_objects, raw=True)
        if target is None:
            return False
        target = self._normalize_coord(target.location)
        if target == (0, 0):
            return False

        direction = np.array(target, dtype=float)
        # 围绕舰队(雷达中心)的邻近格(1 步可到)，按相对舰队方向与目标方向最一致者优先
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        candidates = []
        for neighbor in neighbors:
            if neighbor not in self.radar or self._normalize_coord(neighbor) in self._nearest_object_blocked:
                continue
            vec = np.array(neighbor, dtype=float)
            # 朝目标方向(点积)优先
            score = float(np.dot(vec, direction) / (np.linalg.norm(direction) or 1.0))
            candidates.append((score, neighbor))
        if not candidates:
            return False
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, candidate in candidates:
            local = self._radar_to_local_clickable(candidate)
            if local is None:
                continue
            logger.info(f'[大世界-雷达] 目标 {target} 在镜头外，朝其方向推进一格 {candidate}')
            self._nearest_object_last_click = candidate
            self._nearest_object_last_fleet = (0, 0)
            self.device.click(local)
            self._nearest_object_click_timer.reset()
            return True
        return False

    def _click_first_available(self, candidates, current_fleet):
        """依次点击候选格中可转换为本地坐标的最近一格。返回是否执行点击。"""
        for candidate in candidates:
            local = self._radar_to_local_clickable(candidate)
            if local is None:
                # 该格不可安全点击(不在视野/落在返回大地图按钮/地图边界外)，尝试下一个候选
                continue
            self._nearest_object_last_click = candidate
            self._nearest_object_last_fleet = current_fleet
            self.device.click(local)
            self._nearest_object_click_timer.reset()
            return True
        return None
