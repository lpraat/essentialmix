import logging

from essentialmix.core.log import Logger


def register_n_loggers(n: int) -> None:
    logger_names = [f"logger_{i}" for i in range(n)]
    for name in logger_names:
        Logger.from_name(name)


def test_loggers_are_correctly_registered() -> None:
    register_n_loggers(5)
    assert len(Logger.instances) == 5


def test_set_level() -> None:
    register_n_loggers(3)
    Logger.set_global_level(logging.INFO)
    assert all(logger.level == logging.INFO for logger in Logger.instances)

    Logger.instances[1].set_level(logging.DEBUG)
    assert Logger.instances[0].level == logging.INFO
    assert Logger.instances[1].level == logging.DEBUG
    assert Logger.instances[2].level == logging.INFO
