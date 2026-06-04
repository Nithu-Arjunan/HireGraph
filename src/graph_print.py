import os, operator
from pathlib import Path
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from IPython.display import Image, Markdown, display

OUT = Path("graph_out")
OUT.mkdir(exist_ok=True)

def save_graph_png(app, name: str) -> None:
    (OUT / f"{name}.mmd").write_text(app.get_graph().draw_mermaid())
    try:
        (OUT / f"{name}.png").write_bytes(app.get_graph().draw_mermaid_png())
        print(f"saved graph_out/{name}.png")
    except Exception as exc:
        print(f"PNG skipped ({exc.__class__.__name__}); see graph_out/{name}.mmd")

def show_graph(name: str) -> None:
    from IPython.display import Image, Markdown, display
    p = OUT / f"{name}.png"
    if p.exists():
        display(Image(filename=str(p)))
    else:
        display(Markdown(f"PNG not on disk. Open `graph_out/{name}.mmd` in https://mermaid.live"))