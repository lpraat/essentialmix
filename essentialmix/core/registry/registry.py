from typing import ClassVar, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    instances: ClassVar[set] = set()

    def __init__(self, clz: Type[T]):
        self.clz = clz
        # Maps class name to class
        self.registry: dict[str, Type[T]] = dict()
        Registry.instances.add(self)

    def register(self, clz: Type[T]) -> Type[T]:
        if not issubclass(clz, self.clz):
            raise ValueError(f"Cannot register {clz.__name__} in {self.clz} registry")

        if clz.__name__ in self.registry:
            raise RuntimeError(f"{clz} already registered")

        self.registry[clz.__name__] = clz
        return clz

    def __str__(self) -> str:
        return f"Registry(clz={self.clz.__name__},registry={self.registry})"

    def __repr__(self) -> str:
        return f"Registry(clz={self.clz},registry={self.registry})"

    def __contains__(self, item: Type[T]) -> bool:
        return item.__name__ in self.registry

    def __getitem__(self, item: str) -> Type[T]:
        return self.registry[item]

    def __del__(self) -> None:
        Registry.instances.remove(self)
