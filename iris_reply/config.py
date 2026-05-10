from __future__ import annotations

from astrbot.api import AstrBotConfig


_DEFAULTS = {
    "enabled": True,
    "mute_period": {
        "start_hour": 23,
        "start_minute": 0,
        "end_hour": 7,
        "end_minute": 0,
    },
    "window_size": 10,
    "system_prompt_template": (
        "你正在观察一个群聊。不介入是常态，介入才是例外。"
        "若不需回复请调用 skip_reply，若需回复则按需选用以下工具。"
    ),
    "default_n": 15,
    "default_t": 30,
    "max_token": 3000,
    "follow_up_ttl": 30,
    "quality_threshold": 0.3,
    "provider_id": "",
    "whitelist": [],
}


class ConfigManager:
    def __init__(self, config: AstrBotConfig) -> None:
        self._cfg = config

    def _get(self, key: str, default=None):
        val = self._cfg.get(key, default)
        if val is None:
            return _DEFAULTS.get(key, default)
        return val

    @property
    def enabled(self) -> bool:
        return bool(self._get("enabled", True))

    @property
    def mute_start_hour(self) -> int:
        mp = self._get("mute_period", {})
        return int(mp.get("start_hour", _DEFAULTS["mute_period"]["start_hour"]))

    @property
    def mute_start_minute(self) -> int:
        mp = self._get("mute_period", {})
        return int(mp.get("start_minute", _DEFAULTS["mute_period"]["start_minute"]))

    @property
    def mute_end_hour(self) -> int:
        mp = self._get("mute_period", {})
        return int(mp.get("end_hour", _DEFAULTS["mute_period"]["end_hour"]))

    @property
    def mute_end_minute(self) -> int:
        mp = self._get("mute_period", {})
        return int(mp.get("end_minute", _DEFAULTS["mute_period"]["end_minute"]))

    @property
    def window_size(self) -> int:
        return max(5, min(30, int(self._get("window_size", 10))))

    @property
    def system_prompt_template(self) -> str:
        return str(self._get("system_prompt_template", _DEFAULTS["system_prompt_template"]))

    @property
    def default_n(self) -> int:
        return max(5, min(120, int(self._get("default_n", 15))))

    @property
    def default_t(self) -> int:
        return max(5, min(180, int(self._get("default_t", 30))))

    @property
    def max_token(self) -> int:
        return max(1000, min(8000, int(self._get("max_token", 3000))))

    @property
    def follow_up_ttl(self) -> int:
        return max(5, min(120, int(self._get("follow_up_ttl", 30))))

    @property
    def quality_threshold(self) -> float:
        return max(0.0, min(1.0, float(self._get("quality_threshold", 0.3))))

    @property
    def provider_id(self) -> str:
        return str(self._get("provider_id", ""))

    @property
    def whitelist(self) -> list[str]:
        wl = self._get("whitelist", [])
        if not isinstance(wl, list):
            return []
        return [str(g) for g in wl]
