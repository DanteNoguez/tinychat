from dataclasses import dataclass, field
from typing import List, Optional
from tinychat.asynchronous.manager import TaskManager, TaskManagerParams

from tinychat.observers.observer import BaseObserver


@dataclass
class SetupConfig:
    task_manager_params: TaskManagerParams
    task_manager: TaskManager = TaskManager()
    observers: List["BaseObserver"] = field(default_factory=list)


@dataclass(frozen=True)
class ProcessorNode:
    """Information about a processor in the topology graph."""

    name: str
    handles: list[str]
    produces: list[str]


@dataclass(frozen=True)
class Topology:
    """Topology graph of a CompositeProcessor."""

    nodes: list[ProcessorNode]

    def get_node(self, name: str) -> Optional[ProcessorNode]:
        """Get node by processor name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None
