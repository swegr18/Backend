import infrastucture.container

# Tests for DependencyContainer

def test_container_get():
    """
    Tests DependencyContainer register and get functions
    """
    container = infrastructure.container.DependencyContainer()

    assert hasattr(container, "_dependencies")
    with pytest.raises(KeyError):
        assert container.get("test") == "test_contents"
    
    container.register("test", "test_contents")

    assert container.get("test") == "test_contents"

    with pytest.raises(KeyError):
        assert container.get("nonexistant") != "test_contents"

def test_container_has():
    """
    Tests DependencyContainer register and get functions
    """
    assert hasattr(container, "_dependencies")
    
    container = infrastructure.container.DependencyContainer();
    assert container.has("test") == False

    container.register("test", "test_contents")
    assert container.has("test") == True
    assert container.has("nonexistant") == False