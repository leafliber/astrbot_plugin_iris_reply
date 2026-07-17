from __future__ import annotations

from typing import Any

from astrbot.api import AstrBotConfig


_DEFAULTS = {
    "enabled": True,
    "stats_enabled": False,
    "mute_period": {
        "start_hour": 1,
        "start_minute": 0,
        "end_hour": 7,
        "end_minute": 0,
    },
    "window_size": 15,
    "default_n": 15,
    "default_t": 30,
    "max_token": 3000,
    "follow_up_ttl": 10,
    "follow_up_aggregate_window": 6,
    "quality_threshold": 0.2,
    "provider_id": "",
    "trigger_min_interval": 30,
    "boost_factor": 0.6,
    "boost_duration": 15,
    "max_boosted_replies": 5,
}

# Keys managed exclusively through the pages UI (stored in KV overrides).
# _conf_schema.json only keeps enabled / stats_enabled / provider_id.
_PAGE_MANAGED_KEYS = {
    k for k in _DEFAULTS if k not in {"enabled", "stats_enabled", "provider_id"}
}

# Metadata for the settings page UI.
_CONFIG_META = {
    "mute_period": {
        "label": "静音时段",
        "type": "object",
        "hint": "在静音时段内，Iris 不会主动触发任何回复",
        "items": {
            "start_hour": {"label": "开始小时（0-23）", "type": "int", "min": 0, "max": 23},
            "start_minute": {"label": "开始分钟（0-59）", "type": "int", "min": 0, "max": 59},
            "end_hour": {"label": "结束小时（0-23）", "type": "int", "min": 0, "max": 23},
            "end_minute": {"label": "结束分钟（0-59）", "type": "int", "min": 0, "max": 59},
        },
    },
    "window_size": {
        "label": "滑动记忆窗口大小（条数）",
        "type": "int", "min": 5, "max": 30,
        "hint": "保留最近 N 条有效发言",
    },
    "default_n": {
        "label": "默认消息计数阈值 N",
        "type": "int", "min": 5, "max": 120,
        "hint": "每收到 N 条有效消息触发一次采样",
    },
    "default_t": {
        "label": "默认时间间隔阈值 T（分钟）",
        "type": "int", "min": 5, "max": 180,
        "hint": "距上次采样超过 T 分钟且有新消息时触发",
    },
    "max_token": {
        "label": "上下文 token 上限",
        "type": "int", "min": 1000, "max": 8000,
        "hint": "提交给 LLM 的上下文最大 token 数",
    },
    "follow_up_ttl": {
        "label": "跟进注册表默认 TTL（分钟）",
        "type": "int", "min": 5, "max": 120,
        "hint": "动态跟进记录的存活时长",
    },
    "follow_up_aggregate_window": {
        "label": "follow-up 消息聚合等待窗口（秒）",
        "type": "int", "min": 3, "max": 30,
        "hint": "被关注用户发言后等待此时间再触发 LLM 评估",
    },
    "quality_threshold": {
        "label": "消息质量评分阈值",
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
        "hint": "低于此阈值的消息不进入滑动窗口",
    },
    "trigger_min_interval": {
        "label": "触发阶段最小间隔（秒）",
        "type": "int", "min": 10, "max": 120,
        "hint": "同一群两次触发 LLM 调用之间的最小时间间隔",
    },
    "boost_factor": {
        "label": "回复后频率提升系数",
        "type": "float", "min": 0.3, "max": 0.95, "step": 0.05,
        "hint": "Iris 回复后有效阈值临时乘以此系数（<1.0 降低阈值）",
    },
    "boost_duration": {
        "label": "频率提升持续时间（分钟）",
        "type": "int", "min": 1, "max": 60,
        "hint": "回复后 boost 效果的持续时间，之后线性衰减回正常",
    },
    "max_boosted_replies": {
        "label": "最大连续回复 boost 次数",
        "type": "int", "min": 2, "max": 10,
        "hint": "连续回复不超过此次数时享受完整 boost，超出后逐渐减弱",
    },
}


class ConfigManager:
    def __init__(self, config: AstrBotConfig) -> None:
        self._cfg = config
        self._overrides: dict[str, Any] = {}

    def load_overrides(self, data: dict | None) -> None:
        if isinstance(data, dict):
            self._overrides = {k: v for k, v in data.items() if k in _PAGE_MANAGED_KEYS}

    def get_overrides(self) -> dict[str, Any]:
        return dict(self._overrides)

    def set_override(self, key: str, value: Any) -> None:
        if key not in _PAGE_MANAGED_KEYS:
            return
        if key == "mute_period":
            if not isinstance(value, dict):
                return
            dmp = _DEFAULTS["mute_period"]
            self._overrides[key] = {
                "start_hour": int(value.get("start_hour", dmp["start_hour"])),
                "start_minute": int(value.get("start_minute", dmp["start_minute"])),
                "end_hour": int(value.get("end_hour", dmp["end_hour"])),
                "end_minute": int(value.get("end_minute", dmp["end_minute"])),
            }
        else:
            meta = _CONFIG_META.get(key, {})
            v = value
            if meta.get("type") == "int":
                v = int(v)
                lo, hi = meta.get("min"), meta.get("max")
                if lo is not None: v = max(lo, v)
                if hi is not None: v = min(hi, v)
            elif meta.get("type") == "float":
                v = float(v)
                lo, hi = meta.get("min"), meta.get("max")
                if lo is not None: v = max(lo, v)
                if hi is not None: v = min(hi, v)
            self._overrides[key] = v

    def get_all_page_config(self) -> dict[str, Any]:
        result = {}
        for key in _PAGE_MANAGED_KEYS:
            result[key] = self._get(key)
        return result

    @staticmethod
    def get_page_config_meta() -> dict[str, Any]:
        return {k: dict(v) for k, v in _CONFIG_META.items()}

    def _get(self, key: str, default=None):
        if key in self._overrides:
            return self._overrides[key]
        val = self._cfg.get(key)
        if val is not None:
            return val
        return _DEFAULTS.get(key, default)

    @property
    def enabled(self) -> bool:
        return bool(self._get("enabled"))

    @property
    def stats_enabled(self) -> bool:
        return bool(self._get("stats_enabled"))

    @property
    def mute_period(self) -> tuple[int, int, int, int]:
        mp = self._get("mute_period", {})
        dmp = _DEFAULTS["mute_period"]
        return (
            int(mp.get("start_hour", dmp["start_hour"])),
            int(mp.get("start_minute", dmp["start_minute"])),
            int(mp.get("end_hour", dmp["end_hour"])),
            int(mp.get("end_minute", dmp["end_minute"])),
        )

    @property
    def mute_start_hour(self) -> int:
        return self.mute_period[0]

    @property
    def mute_start_minute(self) -> int:
        return self.mute_period[1]

    @property
    def mute_end_hour(self) -> int:
        return self.mute_period[2]

    @property
    def mute_end_minute(self) -> int:
        return self.mute_period[3]

    @property
    def window_size(self) -> int:
        return max(5, min(30, int(self._get("window_size"))))

    @property
    def default_n(self) -> int:
        return max(5, min(120, int(self._get("default_n"))))

    @property
    def default_t(self) -> int:
        return max(5, min(180, int(self._get("default_t"))))

    @property
    def max_token(self) -> int:
        return max(1000, min(8000, int(self._get("max_token"))))

    @property
    def follow_up_ttl(self) -> int:
        return max(5, min(120, int(self._get("follow_up_ttl"))))

    @property
    def follow_up_aggregate_window(self) -> int:
        return max(3, min(30, int(self._get("follow_up_aggregate_window"))))

    @property
    def quality_threshold(self) -> float:
        return max(0.0, min(1.0, float(self._get("quality_threshold"))))

    @property
    def provider_id(self) -> str:
        return str(self._get("provider_id", ""))

    @property
    def trigger_min_interval(self) -> int:
        return max(10, min(120, int(self._get("trigger_min_interval"))))

    @property
    def boost_factor(self) -> float:
        return max(0.3, min(0.95, float(self._get("boost_factor"))))

    @property
    def boost_duration(self) -> int:
        return max(1, min(60, int(self._get("boost_duration"))))

    @property
    def max_boosted_replies(self) -> int:
        return max(2, min(10, int(self._get("max_boosted_replies"))))
