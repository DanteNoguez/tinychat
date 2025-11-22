import inspect
import re
import types

from enum import Enum
from typing import (
    Any,
    Literal,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
    Union,
    Dict,
    List,
    Set,
    Tuple,
)
from abc import abstractmethod
from dataclasses import dataclass

from tinychat.utils.base_object import BaseObject


DataType = Literal["string", "number", "integer", "boolean", "array", "object", "null"]


@dataclass
class TypeSchema:
    data_type: DataType
    description: Optional[str] = None
    enum: Optional[list[str]] = None
    # For Arrays
    items: Optional["TypeSchema"] = None
    # For Objects (dicts)
    properties: Optional[Dict[str, "TypeSchema"]] = None
    required: Optional[List[str]] = None


@dataclass
class ToolParameter:
    name: str
    description: str
    schema: TypeSchema
    required: bool = True


class Tool(BaseObject):
    """
    Base class for LLM tools.

    Subclasses must:
    1. Implement the `run` method with standard Python type hints.
    2. Add a docstring to the `run` method containing:
        - A general description of the tool.
        - The parameters of the tool with their descriptions, Sphinx-style.

    Example:

    async def run(self, operation: Literal["add", "subtract", "multiply", "divide"], a: float, b: float) -> float:
        \"""
        Perform basic arithmetic operations.

        :param operation: The operation to perform.
        :param a: The first operand.
        :param b: The second operand.
        \"""
        return a + b
    """

    def __init__(self, *, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.description, doc_params = self._parse_docstring(self.run.__doc__)
        self.parameters = self._generate_parameters(doc_params)

    def _parse_docstring(self, docstring: str) -> tuple[str, dict[str, str]]:
        """
        Parses the tool description and argument descriptions from Sphinx-style docstrings.
        Returns: (tool_description, {param_name: param_description})
        """
        if not docstring:
            raise ValueError(
                f"Tool {self.name} docstring is empty. Please provide the tool description and parameter descriptions in the docstring."
            )

        # Split into description (everything before first :param) and params
        split_idx = docstring.find(":param")
        if split_idx == -1:
            tool_description = docstring.strip()
            param_section = ""
        else:
            tool_description = docstring[:split_idx].strip()
            param_section = docstring[split_idx:]

        # Clean up description newlines
        tool_description = " ".join(
            line.strip() for line in tool_description.splitlines() if line.strip()
        )

        if not tool_description:
            raise ValueError(
                f"Tool {self.name} is missing a general description in the docstring."
            )

        params = {}
        # Regex pattern:
        # :param\s+        -> Match ":param" followed by whitespace
        # (\w+)            -> Capture the parameter name
        # :\s+             -> Match colon and whitespace
        # (.*?)            -> Capture description non-greedily
        # (?=\n\s*:param|\Z) -> Lookahead: stop before next :param or end of string
        # flags=re.DOTALL  -> Allow . to match newlines (multiline descriptions)
        pattern = r":param\s+(\w+):\s+(.*?)(?=\n\s*:param|\Z)"

        for match in re.finditer(pattern, param_section, flags=re.DOTALL):
            name = match.group(1)
            description = match.group(2).strip()
            # Clean up newlines and extra spaces in multiline descriptions
            description = " ".join(line.strip() for line in description.splitlines())
            params[name] = description

        return tool_description, params

    def _generate_parameters(self, doc_params: dict[str, str]) -> list[ToolParameter]:
        parameters = []
        sig = inspect.signature(self.run)
        type_hints = get_type_hints(self.run)

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "kwargs", "args"):
                continue

            # 1. Get type (code is truth)
            if param_name not in type_hints:
                raise ValueError(
                    f"Parameter '{param_name}' in tool {self.name} is missing a type annotation."
                )

            actual_type = type_hints[param_name]

            # 2. Get description (docstring is documentation)
            description = doc_params.get(param_name)

            if not description:
                raise ValueError(
                    f"Parameter '{param_name}' in tool {self.name} is missing a description in the docstring. "
                    f"Please add ':param {param_name}: ...' to the docstring."
                )

            is_nullable = self._is_nullable(actual_type)
            schema = self._map_python_type(actual_type)

            # If the parameter has a default value, it's not required
            has_default = param.default != inspect.Parameter.empty
            is_required = not has_default and not is_nullable

            parameters.append(
                ToolParameter(
                    name=param_name,
                    description=description,
                    schema=schema,
                    required=is_required,
                )
            )

        return parameters

    def _is_nullable(self, py_type: Any) -> bool:
        origin = get_origin(py_type)
        if origin is Union or (
            hasattr(types, "UnionType") and origin is types.UnionType
        ):
            return type(None) in get_args(py_type)
        return False

    def _map_python_type(self, py_type: Any) -> TypeSchema:
        origin = get_origin(py_type) or py_type
        args = get_args(py_type)

        # Unwrap Optional/Union (take the first non-None type)
        if self._is_nullable(py_type):
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                return self._map_python_type(non_none[0])

        # Handle Literal (legacy enum support)
        if origin is Literal:
            return TypeSchema(data_type="string", enum=list(args))

        # Handle Python Enums
        # check if it's a class and subclass of Enum
        if isinstance(py_type, type) and issubclass(py_type, Enum):
            # Get all enum values
            enum_values = [e.value for e in py_type]

            # Determine type from the first value (assume homogeneous enums)
            if enum_values:
                first_val = enum_values[0]
                if isinstance(first_val, int):
                    data_type = "integer"
                elif isinstance(first_val, float):
                    data_type = "number"
                else:
                    data_type = "string"
            else:
                data_type = "string"

            return TypeSchema(data_type=data_type, enum=enum_values)

        # Primitive mappings
        if origin is str:
            return TypeSchema(data_type="string")
        if origin is int:
            return TypeSchema(data_type="integer")
        if origin is float:
            return TypeSchema(data_type="number")
        if origin is bool:
            return TypeSchema(data_type="boolean")

        # List / Array / Set / Tuple
        # We treat Set and Tuple (homogeneous) as Arrays for JSON Schema purposes
        if origin in (list, List, set, Set, tuple, Tuple):
            items_schema = None
            if args:
                # For Tuple, we might have (int, ...) or (int, str).
                # Current simple schema only supports homogeneous arrays (single 'items' type).
                # We'll take the first arg as the type for the array items.
                # This correctly handles List[T], Set[T], and Tuple[T, ...]
                items_schema = self._map_python_type(args[0])
            return TypeSchema(data_type="array", items=items_schema)

        # Dict / Object (Simplistic handling for now, prevents errors)
        if origin is dict or origin is Dict:
            return TypeSchema(data_type="object")

        # Fallback for unknown types
        raise ValueError(
            f"Tool {self.name} uses unsupported type {py_type}. "
            f"Supported types are: str, int, float, bool, list, set, tuple, dict, Literal, Enum."
        )

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any: ...
