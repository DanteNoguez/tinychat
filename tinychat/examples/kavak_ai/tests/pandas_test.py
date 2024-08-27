import pandas as pd
import pytest

from tinychat.agents import PandasAgent
from tinychat.agents.models import Tool, ToolParameter


@pytest.fixture(scope="module")
def buscar_auto_tool():
    df = pd.DataFrame(
        {
            "make": ["Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "BMW"],
            "model": ["Corolla", "Civic", "Mustang", "Camaro", "Altima", "3 Series"],
            "year": [2018, 2019, 2020, 2021, 2022, 2023],
            "price": [20000, 22000, 26000, 30000, 25000, 35000],
            "mileage": [15000, 12000, 10000, 5000, 8000, 3000],
            "bluetooth": [True, True, False, True, False, True],
            "carplay": [True, False, True, True, False, True],
        }
    )

    kavak_pandas_agent = PandasAgent(
        prompt="You're an assistant at a car dealership.", df=df
    )

    class BuscarAutoTool(Tool):
        async def run(self, query: str):
            try:
                code = await kavak_pandas_agent.generate_code_async(query)
                print(code)
                return kavak_pandas_agent.execute_code(code)
            except Exception as e:
                print(f"Exception using pandas tool: {e}")
                return (
                    f"There was an error generating a response: {e}, please try again."
                )

    return BuscarAutoTool(
        name="buscar_auto",
        description="Función para buscar automóviles en nuestro catálogo.",
        parameters=[
            ToolParameter(
                name="query",
                description="Frase de búsqueda detallada donde se especifiquen los atributos deseados del automóvil.",
                data_type="string",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_catalog_count(buscar_auto_tool):
    result = await buscar_auto_tool.run("¿Cuántos automóviles tienes en tu catálogo?")
    assert "6" in str(result), "The catalog should contain 6 cars"


@pytest.mark.asyncio
async def test_price_filter(buscar_auto_tool):
    result = await buscar_auto_tool.run(
        "¿Cuántos automóviles en tu catálogo tienen un precio mayor a 25000?"
    )
    assert "3" in str(result), "There should be 3 cars with price > 25000"


@pytest.mark.asyncio
async def test_specific_search(buscar_auto_tool):
    result = await buscar_auto_tool.run(
        "Me interesa un auto Ford del año 2000 en adelante y con carplay."
    )
    assert "Ford" in str(result), "The result should include Ford"
    assert "Mustang" in str(result), "The result should include Mustang model"
    assert "2020" in str(result), "The result should include the year 2020"
