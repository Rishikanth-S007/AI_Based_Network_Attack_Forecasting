import os
import streamlit.components.v1 as components

def render_network_topology(nodes_data, edges_data, height=500):
    """
    Renders an interactive, air-gapped Vis.js network graph.
    Loads vis-network.min.js locally from assets directory to guarantee 100% offline functionality.
    """
    # Path to local vis-network JS script
    js_path = os.path.join(os.path.dirname(__file__), "..", "assets", "vis-network.min.js")
    
    vis_js_code = ""
    if os.path.exists(js_path):
        try:
            with open(js_path, "r", encoding="utf-8") as f:
                vis_js_code = f.read()
        except Exception:
            vis_js_code = ""

    # Style nodes & edges according to cybersecurity severity
    nodes_js = []
    for node in nodes_data:
        is_flagged = node.get("flagged", False)
        color_bg = "#EF4444" if is_flagged else ("#0EA5E9" if node["group"] != "critical" else "#F59E0B")
        color_border = "#F8FAFC" if is_flagged else "#38BDF8"
        size = 32 if is_flagged else (26 if node["group"] == "critical" else 20)

        nodes_js.append({
            "id": node["id"],
            "label": node["label"],
            "shape": "dot",
            "size": size,
            "color": {
                "background": color_bg,
                "border": color_border,
                "highlight": {"background": "#F43F5E", "border": "#FFFFFF"}
            },
            "font": {"color": "#F8FAFC", "size": 13, "face": "-apple-system, system-ui, sans-serif"},
            "shadow": {"enabled": is_flagged, "color": "rgba(239, 68, 68, 0.8)", "size": 15}
        })

    edges_js = []
    for edge in edges_data:
        is_malicious = edge["type"] == "malicious"
        edge_color = "#EF4444" if is_malicious else "#38BDF8"
        dashes = True if is_malicious else False
        width = 3 if is_malicious else 1.5

        edges_js.append({
            "from": edge["source"],
            "to": edge["target"],
            "label": edge.get("label", ""),
            "color": {"color": edge_color, "highlight": "#F43F5E"},
            "dashes": dashes,
            "width": width,
            "font": {"color": "#FCA5A5" if is_malicious else "#94A3B8", "size": 11, "align": "middle", "background": "#0E1424"},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}}
        })

    import json
    nodes_json = json.dumps(nodes_js)
    edges_json = json.dumps(edges_js)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          background-color: #0E1424;
          margin: 0;
          padding: 0;
          overflow: hidden;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        #mynetwork {{
          width: 100%;
          height: {height - 20}px;
          border: 1px solid #24344D;
          border-radius: 8px;
          background: radial-gradient(circle at 50% 50%, #161F30 0%, #0E1424 100%);
        }}
      </style>
      <script type="text/javascript">
        {vis_js_code}
      </script>
    </head>
    <body>
      <div id="mynetwork"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        var container = document.getElementById('mynetwork');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          physics: {{
            enabled: true,
            barnesHut: {{
              gravitationalConstant: -2500,
              centralGravity: 0.3,
              springLength: 110
            }}
          }},
          interaction: {{
            hover: true,
            tooltipDelay: 100,
            zoomView: true
          }}
        }};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """

    components.html(html_code, height=height)
