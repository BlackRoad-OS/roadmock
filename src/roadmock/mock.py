"""
RoadMock - Mocking & Stubbing for BlackRoad
Mock objects, stubs, spies, and test doubles.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
import asyncio
import functools
import inspect
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class Call:
    """Record of a method call."""
    args: tuple
    kwargs: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    return_value: Any = None
    exception: Optional[Exception] = None


class Mock:
    """A mock object."""

    def __init__(self, spec: Type = None, name: str = "mock"):
        self._spec = spec
        self._name = name
        self._calls: List[Call] = []
        self._return_value: Any = None
        self._side_effect: Optional[Union[Callable, Exception, List]] = None
        self._side_effect_index = 0
        self._children: Dict[str, "Mock"] = {}

    def __call__(self, *args, **kwargs) -> Any:
        call = Call(args=args, kwargs=kwargs)
        self._calls.append(call)

        if self._side_effect is not None:
            if isinstance(self._side_effect, Exception):
                call.exception = self._side_effect
                raise self._side_effect
            elif isinstance(self._side_effect, list):
                if self._side_effect_index < len(self._side_effect):
                    effect = self._side_effect[self._side_effect_index]
                    self._side_effect_index += 1
                    if isinstance(effect, Exception):
                        call.exception = effect
                        raise effect
                    call.return_value = effect
                    return effect
            elif callable(self._side_effect):
                result = self._side_effect(*args, **kwargs)
                call.return_value = result
                return result

        call.return_value = self._return_value
        return self._return_value

    def __getattr__(self, name: str) -> "Mock":
        if name.startswith("_"):
            raise AttributeError(name)
        
        if name not in self._children:
            child = Mock(name=f"{self._name}.{name}")
            self._children[name] = child
        
        return self._children[name]

    def return_value(self, value: Any) -> "Mock":
        """Set return value."""
        self._return_value = value
        return self

    def side_effect(self, effect: Union[Callable, Exception, List]) -> "Mock":
        """Set side effect."""
        self._side_effect = effect
        self._side_effect_index = 0
        return self

    def assert_called(self) -> None:
        """Assert the mock was called."""
        if not self._calls:
            raise AssertionError(f"{self._name} was not called")

    def assert_called_once(self) -> None:
        """Assert the mock was called exactly once."""
        if len(self._calls) != 1:
            raise AssertionError(
                f"{self._name} was called {len(self._calls)} times, expected 1"
            )

    def assert_called_with(self, *args, **kwargs) -> None:
        """Assert the mock was last called with specified arguments."""
        if not self._calls:
            raise AssertionError(f"{self._name} was not called")
        
        last_call = self._calls[-1]
        if last_call.args != args or last_call.kwargs != kwargs:
            raise AssertionError(
                f"{self._name} was called with {last_call.args}, {last_call.kwargs}, "
                f"expected {args}, {kwargs}"
            )

    def assert_any_call(self, *args, **kwargs) -> None:
        """Assert the mock was ever called with specified arguments."""
        for call in self._calls:
            if call.args == args and call.kwargs == kwargs:
                return
        raise AssertionError(
            f"{self._name} was never called with {args}, {kwargs}"
        )

    def assert_not_called(self) -> None:
        """Assert the mock was not called."""
        if self._calls:
            raise AssertionError(f"{self._name} was called {len(self._calls)} times")

    @property
    def call_count(self) -> int:
        """Get number of calls."""
        return len(self._calls)

    @property
    def call_args(self) -> Optional[Call]:
        """Get last call arguments."""
        return self._calls[-1] if self._calls else None

    @property
    def call_args_list(self) -> List[Call]:
        """Get all call arguments."""
        return self._calls.copy()

    def reset_mock(self) -> None:
        """Reset all call history."""
        self._calls.clear()
        self._side_effect_index = 0
        for child in self._children.values():
            child.reset_mock()


class Spy(Mock):
    """A spy that wraps a real object."""

    def __init__(self, target: Any):
        super().__init__(name=type(target).__name__)
        self._target = target

    def __call__(self, *args, **kwargs) -> Any:
        call = Call(args=args, kwargs=kwargs)
        self._calls.append(call)

        try:
            if callable(self._target):
                result = self._target(*args, **kwargs)
            else:
                result = self._target
            call.return_value = result
            return result
        except Exception as e:
            call.exception = e
            raise

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        
        attr = getattr(self._target, name)
        if callable(attr):
            if name not in self._children:
                self._children[name] = Spy(attr)
            return self._children[name]
        return attr


class Stub:
    """A stub with predefined responses."""

    def __init__(self):
        self._responses: Dict[str, Any] = {}
        self._patterns: List[tuple] = []

    def when(self, method: str) -> "StubConfig":
        """Configure a stub response."""
        return StubConfig(self, method)

    def __getattr__(self, name: str) -> Callable:
        def stub_method(*args, **kwargs):
            # Check exact match
            key = (name, args, tuple(sorted(kwargs.items())))
            if key in self._responses:
                return self._responses[key]

            # Check patterns
            for pattern, response in self._patterns:
                if pattern(name, args, kwargs):
                    return response

            # Default response
            return self._responses.get((name, (), ()), None)

        return stub_method


class StubConfig:
    """Configure stub responses."""

    def __init__(self, stub: Stub, method: str):
        self._stub = stub
        self._method = method
        self._args: tuple = ()
        self._kwargs: Dict[str, Any] = {}

    def with_args(self, *args, **kwargs) -> "StubConfig":
        """Specify expected arguments."""
        self._args = args
        self._kwargs = kwargs
        return self

    def then_return(self, value: Any) -> Stub:
        """Set return value."""
        key = (self._method, self._args, tuple(sorted(self._kwargs.items())))
        self._stub._responses[key] = value
        return self._stub

    def then_raise(self, exception: Exception) -> Stub:
        """Set exception to raise."""
        def raiser(*args, **kwargs):
            raise exception
        
        key = (self._method, self._args, tuple(sorted(self._kwargs.items())))
        self._stub._responses[key] = raiser
        return self._stub


class Patch:
    """Context manager for patching objects."""

    def __init__(
        self,
        target: str,
        new: Any = None,
        autospec: bool = False
    ):
        self._target = target
        self._new = new
        self._autospec = autospec
        self._original = None
        self._patcher_active = False

    def _get_target(self) -> tuple:
        """Parse target string into object and attribute."""
        parts = self._target.rsplit(".", 1)
        if len(parts) == 1:
            raise ValueError(f"Invalid target: {self._target}")
        
        module_path, attr_name = parts
        
        # Import module
        module = __import__(module_path)
        for part in module_path.split(".")[1:]:
            module = getattr(module, part)
        
        return module, attr_name

    def __enter__(self) -> Mock:
        module, attr_name = self._get_target()
        self._original = getattr(module, attr_name)
        
        if self._new is None:
            self._new = Mock(name=self._target)
        
        setattr(module, attr_name, self._new)
        self._patcher_active = True
        return self._new

    def __exit__(self, *args) -> None:
        if self._patcher_active:
            module, attr_name = self._get_target()
            setattr(module, attr_name, self._original)
            self._patcher_active = False


def patch(target: str, new: Any = None, autospec: bool = False) -> Patch:
    """Create a patch context manager."""
    return Patch(target, new, autospec)


def create_autospec(spec: Type, instance: bool = True) -> Mock:
    """Create a mock that matches the spec's signature."""
    mock = Mock(spec=spec, name=spec.__name__)
    
    for name in dir(spec):
        if name.startswith("_"):
            continue
        
        attr = getattr(spec, name)
        if callable(attr):
            child_mock = Mock(name=f"{spec.__name__}.{name}")
            mock._children[name] = child_mock
    
    return mock


class MockServer:
    """Mock HTTP server for testing."""

    def __init__(self):
        self.routes: Dict[tuple, Dict[str, Any]] = {}
        self.requests: List[Dict[str, Any]] = []

    def add_route(
        self,
        method: str,
        path: str,
        response: Any,
        status: int = 200,
        headers: Dict[str, str] = None
    ) -> None:
        """Add a mock route."""
        self.routes[(method.upper(), path)] = {
            "response": response,
            "status": status,
            "headers": headers or {}
        }

    def handle_request(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Handle a mock request."""
        self.requests.append({
            "method": method,
            "path": path,
            "body": body,
            "headers": headers
        })

        route = self.routes.get((method.upper(), path))
        if route:
            response = route["response"]
            if callable(response):
                response = response(body, headers)
            return {
                "status": route["status"],
                "body": response,
                "headers": route["headers"]
            }
        
        return {"status": 404, "body": {"error": "Not found"}, "headers": {}}

    def reset(self) -> None:
        """Reset server state."""
        self.requests.clear()


class MockDatabase:
    """Mock database for testing."""

    def __init__(self):
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.queries: List[Dict[str, Any]] = []

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert record."""
        if table not in self.tables:
            self.tables[table] = []
        
        record_id = len(self.tables[table]) + 1
        data["id"] = record_id
        self.tables[table].append(data)
        
        self.queries.append({"op": "insert", "table": table, "data": data})
        return record_id

    def select(self, table: str, **where) -> List[Dict[str, Any]]:
        """Select records."""
        self.queries.append({"op": "select", "table": table, "where": where})
        
        if table not in self.tables:
            return []
        
        results = self.tables[table]
        for key, value in where.items():
            results = [r for r in results if r.get(key) == value]
        
        return results

    def update(self, table: str, data: Dict[str, Any], **where) -> int:
        """Update records."""
        self.queries.append({"op": "update", "table": table, "data": data, "where": where})
        
        if table not in self.tables:
            return 0
        
        updated = 0
        for record in self.tables[table]:
            match = all(record.get(k) == v for k, v in where.items())
            if match:
                record.update(data)
                updated += 1
        
        return updated

    def delete(self, table: str, **where) -> int:
        """Delete records."""
        self.queries.append({"op": "delete", "table": table, "where": where})
        
        if table not in self.tables:
            return 0
        
        original_count = len(self.tables[table])
        self.tables[table] = [
            r for r in self.tables[table]
            if not all(r.get(k) == v for k, v in where.items())
        ]
        
        return original_count - len(self.tables[table])

    def reset(self) -> None:
        """Reset database."""
        self.tables.clear()
        self.queries.clear()


# Example usage
def example_usage():
    """Example mocking usage."""
    # Basic mock
    mock = Mock(name="my_mock")
    mock.return_value(42)
    
    result = mock(1, 2, 3)
    print(f"Mock returned: {result}")
    
    mock.assert_called_with(1, 2, 3)
    print(f"Call count: {mock.call_count}")

    # Side effects
    mock.side_effect([1, 2, 3])
    print(f"Side effect sequence: {mock()}, {mock()}, {mock()}")

    # Stub
    stub = Stub()
    stub.when("get_user").with_args(1).then_return({"id": 1, "name": "Alice"})
    stub.when("get_user").with_args(2).then_return({"id": 2, "name": "Bob"})
    
    print(f"Stub user 1: {stub.get_user(1)}")
    print(f"Stub user 2: {stub.get_user(2)}")

    # Mock server
    server = MockServer()
    server.add_route("GET", "/api/users", [{"id": 1}])
    
    response = server.handle_request("GET", "/api/users")
    print(f"Server response: {response}")

    # Mock database
    db = MockDatabase()
    db.insert("users", {"name": "Alice", "email": "alice@example.com"})
    db.insert("users", {"name": "Bob", "email": "bob@example.com"})
    
    users = db.select("users", name="Alice")
    print(f"Found users: {users}")
