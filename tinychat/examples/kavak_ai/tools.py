from loguru import logger

from tinychat.agents import PandasAgent
from tinychat.agents.models import PandasAgentConfig, Tool, ToolParameter
from tinychat.examples.kavak_ai.prompts import PANDAS_TOOL_PROMPT
from tinychat.examples.kavak_ai.utils.rag import df


class FinanciamientoTool(Tool):
    async def run(self, enganche: float, presupuesto: float, tasa: float, plazo: int):
        # Input validations
        if enganche <= 0 or presupuesto <= 0:
            raise ValueError(
                "Enganche o presupuesto inválidos. Verifica los valores ingresados."
            )

        if tasa != 10:
            raise ValueError("La tasa de interés debe ser del 10 %.")

        if plazo < 3 or plazo > 6:
            raise ValueError("El plazo debe ser mayor a 3 años y menor a 6.")

        if enganche < 0.2 * presupuesto:
            raise ValueError("El enganche debe ser al menos el 20 % del presupuesto.")

        if enganche >= presupuesto:
            raise ValueError("El enganche no puede ser mayor o igual al presupuesto.")

        # Calculations
        monto_financiar = presupuesto - enganche
        tasa_mensual = tasa / 12 / 100
        num_pagos = plazo * 12

        # Monthly payment calculation using the formula for fixed-rate loans
        pago_mensual = (
            monto_financiar * tasa_mensual * (1 + tasa_mensual) ** num_pagos
        ) / ((1 + tasa_mensual) ** num_pagos - 1)

        # Total amount paid
        total_pagado = pago_mensual * num_pagos + enganche

        # Maximum car price affordable with this plan
        precio_maximo_auto = monto_financiar + enganche

        return {
            "monto_financiar": round(monto_financiar, 2),
            "pago_mensual": round(pago_mensual, 2),
            "total_pagado": round(total_pagado, 2),
            "precio_maximo_auto": round(precio_maximo_auto, 2),
            "num_pagos": num_pagos,
        }


FINANCIAMIENTO_TOOL = FinanciamientoTool(
    name="financiamiento",
    description="""Utiliza esta función para generar planes de financiamiento.
    La consulta debe incluir: monto del enganche, presupuesto, tasa de interés del 10 % y un plazo de financiamiento (de 3 a 6 años).
    Deberás preguntar al usuario por las opciones de su interés antes de poder usar esta función.""",
    parameters=[
        ToolParameter(
            name="enganche",
            description="Monto del enganche (arriba del 20 %).",
            data_type="string",
        ),
        ToolParameter(
            name="presupuesto",
            description="Presupuesto del cliente.",
            data_type="string",
        ),
        ToolParameter(
            name="tasa",
            description="Tasa de interés (fijo, del 10 %).",
            data_type="string",
        ),
        ToolParameter(
            name="plazo",
            description="Plazo de financiamiento en años (solo ofrecemos de 3 a 6 años).",  # should be enum
            data_type="string",
        ),
    ],
)

KAVAK_PANDAS_AGENT_CONFIG = PandasAgentConfig(
    prompt=PANDAS_TOOL_PROMPT, df_dict=df.to_dict()
)

KAVAK_PANDAS_AGENT = PandasAgent(config=KAVAK_PANDAS_AGENT_CONFIG)


class BuscarAutoTool(Tool):
    async def run(self, query: str):
        try:
            return await KAVAK_PANDAS_AGENT.handle_generate_response(query)
        except Exception as e:
            logger.exception(f"Exception using pandas tool: {e}")
            return f"There was an error generating a response: {e}, please try again."


BUSCAR_AUTO_TOOL = BuscarAutoTool(
    name="buscar_auto",
    description="""Utiliza esta función para buscar automóviles en nuestro catálogo.
    Tu búsqueda debe mencionar al menos tres atributos del automóvil deseado, tales como marca, modelo, versión, año, precio, kilometraje, bluetooth, carplay, o dimensiones.
    Deberás preguntar al usuario por los atributos de su interés antes de poder usar esta función.""",
    parameters=[
        ToolParameter(
            name="query",
            description="Frase de búsqueda detallada donde se especifiquen los atributos deseados del automóvil.",
            data_type="string",
        ),
    ],
)
