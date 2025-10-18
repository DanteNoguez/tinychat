"""Exceptions for message processor operations."""


class ProcessorException(Exception):
    """Base exception for processor errors."""

    pass


class CycleDetectedError(ProcessorException):
    """Raised when a routing cycle is detected."""

    def __init__(self, processor_name: str, processor_path: list):
        self.processor_name = processor_name
        self.processor_path = processor_path
        path_str = " → ".join(processor_path + [processor_name])
        super().__init__(f"Routing cycle detected: {path_str}")


class MaxHopsExceededError(ProcessorException):
    """Raised when a message exceeds maximum routing depth."""

    def __init__(self, max_hops: int, current_hops: int):
        self.max_hops = max_hops
        self.current_hops = current_hops
        super().__init__(f"Message exceeded maximum hops: {current_hops}/{max_hops}")
