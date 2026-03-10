import pytest
import infrastructure.container


def test_container_get():
    """
    Tests DependencyContainer register and get functions
    """
    container = infrastructure.container.DependencyContainer()

    assert hasattr(container, "_dependencies")

    with pytest.raises(KeyError):
        container.get("test")

    container.register("test", "test_contents")
    assert container.get("test") == "test_contents"

    with pytest.raises(KeyError):
        container.get("nonexistant")


def test_container_has():
    """
    Tests DependencyContainer register and has functions
    """
    container = infrastructure.container.DependencyContainer()

    assert hasattr(container, "_dependencies")
    assert container.has("test") is False

    container.register("test", "test_contents")
    assert container.has("test") is True
    assert container.has("nonexistant") is False