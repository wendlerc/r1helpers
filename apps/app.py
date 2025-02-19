import streamlit as st
import json
import os
from colorsys import hsv_to_rgb

def get_distinct_colors(n):
    """Generate n visually distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = hsv_to_rgb(hue, 0.8, 1.0)
        rgb_255 = tuple(int(x * 255) for x in rgb)
        colors.append(f"rgb{rgb_255}")
    return colors

def get_color(value, min_val, max_val, base_color):
    """
    Returns an RGB color by adjusting the opacity of the base_color based on the value.
    """
    if max_val == min_val:
        normalized = 0.0
    else:
        normalized = (value - min_val) / (max_val - min_val)
    return f"color-mix(in srgb, {base_color} {normalized * 100}%, white)"

def main():
    st.title("LLM Response Viewer")
    
    st.markdown("""
        <style>
        .token-container {
            position: relative;
            display: inline-block;
            margin: 0;
            padding: 0;
        }
        .custom-tooltip {
            visibility: hidden;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 14px;
            white-space: nowrap;
            z-index: 1000;
            margin-bottom: 4px;
        }
        .custom-tooltip::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: rgba(0, 0, 0, 0.9) transparent transparent transparent;
        }
        .token-container:hover .custom-tooltip {
            visibility: visible;
        }
        .token {
            display: inline-block;
            padding: 2px 4px;
            margin: 0 1px;
            border-radius: 3px;
            cursor: default;
            min-height: 2em;
        }
        .tokens-wrapper {
            line-height: 2;
            white-space: pre-wrap;
            word-spacing: 0;
        }
        .metric-legend {
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
            background: #f0f0f0;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .color-sample {
            width: 20px;
            height: 20px;
            border-radius: 3px;
            flex-shrink: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    jsonl_path = st.text_input("Path to .jsonl file", "example.jsonl")

    if not os.path.exists(jsonl_path):
        st.error(f"File '{jsonl_path}' not found.")
        return

    # Load data
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not data:
        st.error("No data found in the JSONL file.")
        return

    # Datapoint selection
    max_index = len(data) - 1
    index = st.slider("Select datapoint index", 0, max_index, 0)
    datapoint = data[index]

    token_strs = datapoint["token_strs"]
    metrics_dict = datapoint.get("metrics", {})

    if not metrics_dict:
        st.warning("No metrics found in this datapoint.")
        st.write(" ".join(token_strs))
        return

    available_metrics = list(metrics_dict.keys())

    # Metrics selection
    color_metrics = st.multiselect(
        "Metrics to visualize with colors",
        available_metrics,
        default=[available_metrics[0]]
    )

    hover_metrics = st.multiselect(
        "Additional metrics to show on hover",
        [m for m in available_metrics if m not in color_metrics],
        default=[]
    )

    if not color_metrics:
        st.warning("Please select at least one metric to visualize.")
        return

    # Generate colors and create legend
    metric_colors = get_distinct_colors(len(color_metrics))
    metric_color_map = dict(zip(color_metrics, metric_colors))

    # Create legend
    st.markdown('<div class="metric-legend">' + 
                ''.join([f'<div class="legend-item"><div class="color-sample" style="background-color: {color}"></div>{metric}</div>'
                        for metric, color in metric_color_map.items()]) +
                '</div>', unsafe_allow_html=True)

    # Calculate ranges
    metric_ranges = {metric: (min(metrics_dict[metric]), max(metrics_dict[metric]))
                    for metric in color_metrics}

    # Build tokens
    token_spans = []
    for i, token in enumerate(token_strs):
        # Create gradient
        token = token.strip()
        token = token.replace('<', '&lt;').replace('>', '&gt;')
        # Escape special characters
        token = token.replace('\\', '\\\\')
        # Make token HTML compatible by replacing whitespace
        token = token.replace(' ', '&nbsp;').replace('\n', '<br>')
        gradient_stops = []
        n_metrics = len(color_metrics)
        for j, metric in enumerate(color_metrics):
            base_color = metric_color_map[metric]
            val = metrics_dict[metric][i]
            min_val, max_val = metric_ranges[metric]
            color = get_color(val, min_val, max_val, base_color)
            start_pct = (j / n_metrics) * 100
            end_pct = ((j + 1) / n_metrics) * 100
            gradient_stops.append(f"{color} {start_pct}% {end_pct}%")

        gradient = f"linear-gradient(to bottom, {', '.join(gradient_stops)})"

        # Create tooltip
        tooltip_lines = []
        for metric in color_metrics:
            mval = metrics_dict[metric][i]
            tooltip_lines.append(f"{metric}: {mval:.4f}")
        for metric in hover_metrics:
            mval = metrics_dict[metric][i]
            tooltip_lines.append(f"{metric}: {mval:.4f}")
        
        tooltip_text = "<br>".join(tooltip_lines)

        # Build token HTML
        span_html = (f'<div class="token-container">'
                    f'<span class="token" style="background: {gradient};">{token}</span>'
                    f'<div class="custom-tooltip">{tooltip_text}</div>'
                    f'</div>')
        token_spans.append(span_html)

    # Render tokens
    st.markdown(f'<div class="tokens-wrapper">{"".join(token_spans)}</div>', 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
