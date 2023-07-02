import pytest

from essentialmix.core.registry import Registry


def test_registry() -> None:
    class AbstractA:
        pass

    a_registry = Registry(AbstractA)
    assert a_registry in Registry.instances, "Class should be in Registry instances"

    @a_registry.register
    class A(AbstractA):
        pass

    assert A in a_registry, "Registered class should be in registry"

    a_registry.__del__()
    assert a_registry not in Registry.instances, "Instance should be removed"


def test_cannot_register_different_types() -> None:
    class AbstractA:
        pass

    class AbstractB:
        pass

    a_registry = Registry(AbstractA)

    @a_registry.register
    class A(AbstractA):
        pass

    with pytest.raises(ValueError):

        @a_registry.register  # type: ignore
        class B(AbstractB):
            pass
