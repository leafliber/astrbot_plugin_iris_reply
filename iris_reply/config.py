from __future__ import annotations

from astrbot.api import AstrBotConfig


_DEFAULTS = {
    "enabled": True,
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


class ConfigManager:
    def __init__(self, config: AstrBotConfig) -> None:
        self._cfg = config

    def _get(self, key: str, default=None):
        val = self._cfg.get(key)
        if val is not None:
            return val
        return _DEFAULTS.get(key, default)

    @property
    def enabled(self) -> bool:
        return bool(self._get("enabled"))

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
