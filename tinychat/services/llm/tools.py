import inspect
import re
import types
import dataclasses

from typing import (
    Any,
    Literal,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
    Union,
    Dict,
    List,
    Set,
    Tuple,
)
from abc import abstractmethod
from dataclasses import dataclass

try:
    from pydantic import BaseModel

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

    class BaseModel:
        pass  # Dummy for type checking


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
    nullable: bool = False
    # Raw JSON Schema override (for Pydantic models)
    json_schema: Optional[dict] = None


@dataclass
class ToolParameter:
    name: str
    description: str
    schema: TypeSchema
    required: bool = True


class Tool(BaseObject):
    """
    Base class for LLM tools in tinychat.

    Subclasses must:
    - Implement the `run` method with standard Python type hints.
    - Add a docstring to the `run` method containing:
        1. A general description of the tool.
        2. The parameters of the tool with their descriptions, Sphinx-style.

    Example:

    async def run(self, a: float, b: float) -> float:
        '''
        Add two numbers.

        :param a: The first operand.
        :param b: The second operand.
        '''

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

        # 1. CRITICAL: Block generic Dicts (Strict Mode Incompatibility)
        # OpenAI Strict Mode requires additionalProperties=False, which forbids dynamic keys.
        if origin is dict or origin is Dict:
            raise ValueError(
                f"Tool parameter in '{self.name}' uses 'dict'. "
                "Generic dictionaries with dynamic keys are NOT supported in OpenAI Strict Mode. "
                "Please use a Pydantic model or TypedDict with defined fields instead."
            )

        # 2. Handle Pydantic Models
        if HAS_PYDANTIC and isinstance(origin, type) and issubclass(origin, BaseModel):
            model_schema = origin.model_json_schema()
            return TypeSchema(data_type="object", json_schema=model_schema)

        # 3. Handle TypedDict
        if is_typeddict(origin):
            return self._map_typed_dict(origin)

        # 4. Unwrap Optional/Union (take the first non-None type)
        if self._is_nullable(py_type):
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                schema = self._map_python_type(non_none[0])
                schema.nullable = True
                return schema

        # Handle Literal (legacy enum support)
        if origin is Literal:
            return TypeSchema(data_type="string", enum=list(args))

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

        # Fallback for unknown types
        raise ValueError(
            f"Tool {self.name} uses unsupported type {py_type}. "
            f"Supported types are: str, int, float, bool, list, set, tuple, Pydantic Models."
        )

    def _map_typed_dict(self, typed_dict_cls: Any) -> TypeSchema:
        """
        Recursively converts a TypedDict to a TypeSchema.
        """
        properties = {}
        required_keys = []

        # Resolve type hints (handles string forward refs automatically)
        type_hints = get_type_hints(typed_dict_cls)

        # TypedDicts have a __required_keys__ frozenset (Python 3.9+)
        required_set = getattr(typed_dict_cls, "__required_keys__", frozenset())

        for name, type_hint in type_hints.items():
            # Recursive call allows nested TypedDicts
            properties[name] = self._map_python_type(type_hint)

            if name in required_set:
                required_keys.append(name)

        return TypeSchema(
            data_type="object",
            properties=properties,
            required=required_keys,
            nullable=False,
        )

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any: ...


class GenerateTypedMessageTool(Tool):
    """
    A specialized Tool that represents a target Message type.
    Used for structured output / routing.
    """

    def __init__(self, message_type: type):
        self.message_type = message_type
        # Use the class name as the tool name
        super().__init__(name=message_type.__name__)
        self.description = (
            message_type.__doc__ or f"Emit a {message_type.__name__} message."
        )
        self.parameters = self._generate_message_parameters(message_type)

    def _generate_message_parameters(self, message_type: type) -> list[ToolParameter]:
        parameters = []
        # Inspect dataclass fields
        for field in dataclasses.fields(message_type):
            # Skip internal fields that are not init-able (id, name, timestamp)
            if not field.init:
                continue

            # Get type and description
            field_type = field.type
            # TODO: We can support field metadata for descriptions later if needed
            description = f"Value for {field.name}"

            schema = self._map_python_type(field_type)

            is_required = (
                field.default == dataclasses.MISSING
                and field.default_factory == dataclasses.MISSING
            )

            parameters.append(
                ToolParameter(
                    name=field.name,
                    description=description,
                    schema=schema,
                    required=is_required,
                )
            )
        return parameters

    async def run(self, **kwargs):
        """
        Emit a specific message type.

        :param kwargs: The keyword arguments to instantiate the message.
        """
        # This won't actually be executed in the standard loop
        # But if it were, it would just instantiate the message
        return self.message_type(**kwargs)
