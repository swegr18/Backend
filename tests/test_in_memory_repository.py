import asyncio
from infrastructure.persistence.in_memory_repository import InMemoryRepository

class DummyEntity:
    def __init__(self, id=None, name="test"):
        self.id = id
        self.name = name

def test_in_memory_repository_sync():
    """Test generic InMemoryRepository saving and finding."""
    async def run_tests():
        repo = InMemoryRepository()
        
        # Test save
        entity1 = DummyEntity(name="first")
        saved1 = await repo.save(entity1)
        assert saved1.id == "1"
        
        # Test find_by_id
        found = await repo.find_by_id("1")
        assert found.id == "1"
        
        # Test find_all
        all_entities = await repo.find_all()
        assert len(all_entities) == 1
        
        # Test delete
        assert await repo.delete("1") is True
        assert await repo.delete("1") is False
        
    asyncio.run(run_tests())
import asyncio
from infrastructure.persistence.in_memory_repository import InMemoryRepository

class DummyEntity:
    def __init__(self, id=None, name="test"):
        self.id = id
        self.name = name

def test_in_memory_repository_sync():
    """Test generic InMemoryRepository saving and finding."""
    async def run_tests():
        repo = InMemoryRepository()
        
        # Test save
        entity1 = DummyEntity(name="first")
        saved1 = await repo.save(entity1)
        assert saved1.id == "1"
        
        # Test find_by_id
        found = await repo.find_by_id("1")
        assert found.id == "1"
        
        # Test find_all
        all_entities = await repo.find_all()
        assert len(all_entities) == 1
        
        # Test delete
        assert await repo.delete("1") is True
        assert await repo.delete("1") is False
        
    asyncio.run(run_tests())