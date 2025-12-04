from typing import Any
from mcp.server.fastmcp import FastMCP
import http.client
import json
import requests

mindverse = FastMCP("mindverse_public")
# External communication disabled
url = "localhost"

messages =[]

@mindverse.tool()
async def get_response(query:str, instance_id:str) -> str | None:
    """
    Received a response based on public secondme model.
    (Disabled: Returns mock response)

    Args:
        query (str): Questions raised by users regarding the secondme model.
        instance_id (str): ID used to identify the secondme model, or url used to identify the secondme model.

    """
    return "Communication with external Second Me servers is disabled."

@mindverse.tool()
async def get_online_instances():
    """
    Check which secondme models are available for chatting online.
    (Disabled: Returns empty list)
    """
    return json.dumps([], ensure_ascii=False, indent=2)

if __name__ == "__main__":

    mindverse.run(transport='stdio')
