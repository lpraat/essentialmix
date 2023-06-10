import asyncio
from typing import Any, Coroutine, Optional

from essentialmix.core.log import Logger

logger = Logger.from_name(__name__)

try:
    import uvloop
except ImportError:
    logger.warn("uvloop is not installed. To speed up the event loop run 'pip install uvloop'")
    uvloop_available = False
else:
    uvloop_available = True


def is_iterable(x: Any) -> bool:
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


def asyncio_run(coro: Coroutine, *, debug: Optional[bool] = None) -> None:
    if uvloop_available:
        uvloop.install()
        asyncio.run(coro, debug=debug)
    else:
        asyncio.run(coro, debug=debug)
