from __future__ import annotations

from typing import Callable, Mapping, Sequence


class PtyProcess:
    @classmethod
    def spawn(
        cls,
        argv: Sequence[str],
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        echo: bool = True,
        preexec_fn: Callable[[], None] | None = None,
        dimensions: tuple[int, int] = (24, 80),
        pass_fds: Sequence[int] = (),
    ) -> PtyProcess: ...

    def read(self, size: int) -> bytes: ...
    def write(self, s: bytes, flush: bool = True) -> int: ...
    def terminate(self, force: bool = False) -> None: ...
