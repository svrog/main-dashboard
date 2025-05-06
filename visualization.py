# dashboard_langgraph_groq.py

import base64
import io
from loguru import logger
import os
from typing import List, Literal, Dict, Union
from typing_extensions import TypedDict
import matplotlib.pyplot as plt
from PIL import Image

from langgraph.graph import StateGraph

from groq import Groq

gapi = os.getenv("GROQ_API_KEY")

# 🧠 Groq Client Setup
groq_client = Groq(api_key=gapi)  # Replace with your key

# ---- Define State Schema ----

class VizInput(TypedDict):
    type: Literal["bar", "pie", "scatter"]
    data: Dict[str, List[Union[int, float, str]]]

class VizState(TypedDict):
    visualizations: List[VizInput]
    images: List[str]  # base64-encoded charts

# ---- Node 1: Render Visualizations ----

def render_chart(viz: VizInput) -> str:
    logger.debug(f"Rendering {viz['type']} chart")
    fig, ax = plt.subplots()

    if viz["type"] == "bar":
        ax.bar(viz["data"]["x"], viz["data"]["y"])
    elif viz["type"] == "pie":
        ax.pie(viz["data"]["values"], labels=viz["data"].get("labels"))
    elif viz["type"] == "scatter":
        ax.scatter(viz["data"]["x"], viz["data"]["y"])

    ax.set_title(f"{viz['type'].capitalize()} Chart")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def visualization_node(state: VizState) -> VizState:
    logger.debug("Rendering visualizations")
    rendered = []
    for viz in state["visualizations"]:
        img = render_chart(viz)
        rendered.append(img)

    return {"visualizations": state["visualizations"], "images": rendered}

# ---- Node 2: Dashboard Creation ----

def dashboard_node(state: VizState) -> Dict[str, str]:
    logger.debug("Creating dashboard")
    decoded_images = []
    for img_b64 in state["images"]:
        img_data = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        decoded_images.append(img)
    logger.debug(f"Decoded {len(decoded_images)} images")

    widths, heights = zip(*(i.size for i in decoded_images))
    max_width = max(widths)
    total_height = sum(heights)

    dashboard = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in decoded_images:
        dashboard.paste(img, (0, y_offset))
        y_offset += img.height

    output = io.BytesIO()
    dashboard.save(output, format="PNG")
    output.seek(0)
    logger.debug("Dashboard created")
    dashboard_b64 = base64.b64encode(output.read()).decode()
    logger.critical(dashboard_b64)
    
    return {"dashboard_image": dashboard_b64}

# ---- LangGraph Setup ----

def build_graph():
    logger.critical("Gra")
    builder = StateGraph(VizState)
    builder.add_node("render_visualizations", visualization_node)
    builder.add_node("create_dashboard", dashboard_node)
    builder.set_entry_point("render_visualizations")
    builder.add_edge("render_visualizations", "create_dashboard")
    builder.set_finish_point("create_dashboard")

    graph = builder.compile()
    return graph

# ---- Main Execution ----

if __name__ == "__main__":
    # Example input
    example_input = {
        "visualizations": [
            {"type": "bar", "data": {"x": ["Q1", "Q2", "Q3"], "y": [100, 150, 120]}},
            {"type": "pie", "data": {"values": [40, 30, 30], "labels": ["A", "B", "C"]}},
            {"type": "scatter", "data": {"x": [1, 2, 3, 4], "y": [10, 20, 15, 30]}}
        ],
        "images": []
    }

    graph = build_graph()
    logger.critical(graph)
    result1 = graph.invoke(example_input)
    logger.success(result1.keys())

    # Save dashboard image to file
    with open("dashboard_output.png", "wb") as f:
        f.write(base64.b64decode(result1["dashboard_image"]))

    print("✅ Dashboard saved as 'dashboard_output.png'")
