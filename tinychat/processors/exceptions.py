class ProcessorException(Exception):
    pass


class MaxHopsExceededError(ProcessorException):
    def __init__(self, max_hops: int, current_hops: int):
        self.max_hops = max_hops
        self.current_hops = current_hops
        super().__init__(f"Message exceeded maximum hops: {current_hops}/{max_hops}")
