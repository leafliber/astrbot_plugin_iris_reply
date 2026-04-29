from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, TypeVar

KT = TypeVar("KT")
VT = TypeVar("VT")


class BoundedDict(OrderedDict[KT, VT]):
    def __init__(
        self,
        max_size: int = 1000,
        on_evict: Optional[Callable[[KT, VT], None]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_size = max(1, max_size)
        self._on_evict = on_evict

    @property
    def max_size(self) -> int:
        return self._max_size

    def __setitem__(self, key: KT, value: VT) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        self._evict()

    def __getitem__(self, key: KT) -> VT:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key: KT, default: Optional[VT] = None) -> Optional[VT]:
        if key in self:
            return self[key]
        return default

    def _evict(self) -> None:
        while len(self) > self._max_size:
            key, value = self.popitem(last=False)
            if self._on_evict is not None:
                try:
                    self._on_evict(key, value)
                except Exception:
                    pass
