import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="WSN Sentinel", page_icon="📡", layout="wide")

st.title("📡 WSN Sentinel")
st.caption("Real-Time Energy-Aware Wireless Sensor Network Digital Twin")

if "nodes" not in st.session_state:
    st.session_state.nodes = None
    st.session_state.energy = {}
    st.session_state.history = []
    st.session_state.events = []
    st.session_state.running = False
    st.session_state.cycle = 0
    st.session_state.sent = 0
    st.session_state.delivered = 0
    st.session_state.dropped = 0
    st.session_state.target = None
    st.session_state.faults = set()

with st.sidebar:
    st.header("⚙️ Network Setup")
    node_count = st.slider("Sensor Nodes", 10, 60, 30)
    field_size = st.slider("Field Size", 100, 500, 250)
    radio_range = st.slider("Radio Range", 25, 120, 65)
    initial_energy = st.slider("Battery (J)", 10, 100, 50)
    packets_cycle = st.slider("Packets / Cycle", 1, 15, 5)
    seed = st.number_input("Simulation Seed", 1, 9999, 42)

    st.divider()
    st.header("🧭 Routing")
    routing = st.selectbox(
        "Routing Algorithm",
        ["Energy-Aware", "Nearest Neighbor", "Geographic"]
    )

    st.divider()
    deploy = st.button("🚀 Deploy Network", use_container_width=True)
    start = st.button("▶️ Start / Pause", use_container_width=True)
    target_button = st.button("🎯 Generate Target", use_container_width=True)
    fault_button = st.button("💥 Simulate Node Fault", use_container_width=True)
    reset = st.button("🔄 Reset", use_container_width=True)

if reset:
    for key in [
        "nodes", "energy", "history", "events",
        "target", "faults"
    ]:
        if key in st.session_state:
            st.session_state[key] = None if key in ["nodes", "target"] else []
    st.session_state.energy = {}
    st.session_state.history = []
    st.session_state.events = []
    st.session_state.faults = set()
    st.session_state.running = False
    st.session_state.cycle = 0
    st.session_state.sent = 0
    st.session_state.delivered = 0
    st.session_state.dropped = 0
    st.rerun()

if deploy:
    rng = np.random.default_rng(seed)

    st.session_state.nodes = {
        i: (
            float(rng.uniform(10, field_size - 10)),
            float(rng.uniform(10, field_size - 10))
        )
        for i in range(node_count)
    }

    st.session_state.energy = {
        i: float(initial_energy) for i in range(node_count)
    }

    st.session_state.history = []
    st.session_state.events = []
    st.session_state.faults = set()
    st.session_state.target = None
    st.session_state.cycle = 0
    st.session_state.sent = 0
    st.session_state.delivered = 0
    st.session_state.dropped = 0
    st.session_state.running = False

if st.session_state.nodes is None:
    st.info("Deploy the sensor network from the sidebar.")
    st.stop()

nodes = st.session_state.nodes
energy = st.session_state.energy

sink = max(nodes, key=lambda n: nodes[n][1])

def distance(a, b):
    return np.sqrt(
        (nodes[a][0] - nodes[b][0]) ** 2 +
        (nodes[a][1] - nodes[b][1]) ** 2
    )

def neighbors(node):
    return [
        n for n in nodes
        if n != node
        and n not in st.session_state.faults
        and energy[n] > 0
        and distance(node, n) <= radio_range
    ]

def route(source):
    current = source
    path = [source]
    visited = {source}

    for _ in range(node_count):
        if current == sink:
            return path

        candidates = [
            n for n in neighbors(current)
            if n not in visited
        ]

        if not candidates:
            return []

        if routing == "Nearest Neighbor":
            nxt = min(candidates, key=lambda n: distance(n, sink))

        elif routing == "Geographic":
            nxt = min(
                candidates,
                key=lambda n: distance(n, sink) +
                0.002 * distance(current, n)
            )

        else:
            nxt = max(
                candidates,
                key=lambda n:
                (energy[n] / initial_energy) /
                (distance(n, sink) + 1)
            )

        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return []

if target_button:
    rng = np.random.default_rng()
    st.session_state.target = (
        float(rng.uniform(20, field_size - 20)),
        float(rng.uniform(20, field_size - 20))
    )

    st.session_state.events.append(
        f"🎯 Target detected at "
        f"({st.session_state.target[0]:.1f}, "
        f"{st.session_state.target[1]:.1f})"
    )

if fault_button:
    alive = [
        n for n in nodes
        if energy[n] > 0 and n != sink
        and n not in st.session_state.faults
    ]

    if alive:
        failed = int(np.random.choice(alive))
        st.session_state.faults.add(failed)
        energy[failed] = 0
        st.session_state.events.append(
            f"💥 Node {failed} failed unexpectedly"
        )

if start:
    st.session_state.running = not st.session_state.running

if st.session_state.running:
    alive = [
        n for n in nodes
        if energy[n] > 0 and n not in st.session_state.faults
    ]

    for _ in range(packets_cycle):
        if len(alive) < 2:
            break

        source = int(np.random.choice(alive))

        if source == sink:
            continue

        st.session_state.sent += 1
        path = route(source)

        if path:
            transmission_cost = 0.05 + 0.01 * len(path)

            for node in path:
                energy[node] = max(
                    0,
                    energy[node] - transmission_cost
                )

            st.session_state.delivered += 1
        else:
            st.session_state.dropped += 1

    for node in alive:
        if node != sink:
            energy[node] = max(
                0,
                energy[node] - np.random.uniform(0.005, 0.02)
            )

    st.session_state.cycle += 1

    alive_now = sum(
        energy[n] > 0 and n not in st.session_state.faults
        for n in nodes
    )

    avg_energy = np.mean(list(energy.values()))

    delivery = (
        st.session_state.delivered /
        max(st.session_state.sent, 1) * 100
    )

    st.session_state.history.append({
        "Cycle": st.session_state.cycle,
        "Average Energy": avg_energy,
        "Delivery Rate": delivery,
        "Alive Nodes": alive_now,
        "Dead Nodes": node_count - alive_now
    })

    low = [
        n for n in nodes
        if 0 < energy[n] < initial_energy * 0.2
    ]

    if low:
        st.session_state.events.append(
            f"⚠️ Low battery warning: {len(low)} node(s)"
        )

col1, col2, col3, col4, col5 = st.columns(5)

alive_count = sum(
    energy[n] > 0 and n not in st.session_state.faults
    for n in nodes
)

avg_energy = np.mean(list(energy.values()))

delivery_rate = (
    st.session_state.delivered /
    max(st.session_state.sent, 1) * 100
)

col1.metric("🟢 Alive", alive_count)
col2.metric("⚡ Avg Energy", f"{avg_energy:.2f} J")
col3.metric("📦 Sent", st.session_state.sent)
col4.metric("✅ Delivery", f"{delivery_rate:.1f}%")
col5.metric("❌ Dropped", st.session_state.dropped)

left, right = st.columns([1.6, 1])

with left:
    st.subheader("🗺️ Live Sensor Field")

    edge_x = []
    edge_y = []

    for n in nodes:
        for nb in neighbors(n):
            if n < nb:
                edge_x += [nodes[n][0], nodes[nb][0], None]
                edge_y += [nodes[n][1], nodes[nb][1], None]

    edges = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="#334155", width=1),
        hoverinfo="none"
    )

    x = [nodes[n][0] for n in nodes]
    y = [nodes[n][1] for n in nodes]

    colors = []

    for n in nodes:
        if n == sink:
            colors.append("#00e5ff")
        elif n in st.session_state.faults or energy[n] <= 0:
            colors.append("#ff1744")
        elif energy[n] < initial_energy * 0.2:
            colors.append("#ffb300")
        else:
            colors.append("#00ff88")

    sensor_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=[str(n) for n in nodes],
        textposition="top center",
        marker=dict(
            size=15,
            color=colors,
            line=dict(color="white", width=1)
        ),
        customdata=[
            round(energy[n], 2) for n in nodes
        ],
        hovertemplate=(
            "Node %{text}<br>"
            "Energy: %{customdata} J<extra></extra>"
        )
    )

    traces = [edges, sensor_trace]

    if st.session_state.target:
        tx, ty = st.session_state.target

        target_trace = go.Scatter(
            x=[tx],
            y=[ty],
            mode="markers+text",
            text=["🎯 TARGET"],
            textposition="top center",
            marker=dict(
                size=24,
                color="#ff3366",
                symbol="star"
            )
        )

        traces.append(target_trace)

    fig = go.Figure(traces)

    fig.update_layout(
        height=580,
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        font_color="white",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, field_size], visible=False),
        yaxis=dict(
            range=[0, field_size],
            visible=False,
            scaleanchor="x"
        ),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📈 Network Performance")

    history = pd.DataFrame(st.session_state.history)

    if not history.empty:
        chart = go.Figure()

        chart.add_trace(go.Scatter(
            x=history["Cycle"],
            y=history["Average Energy"],
            name="Energy",
            line=dict(color="#00e5ff", width=3)
        ))

        chart.add_trace(go.Scatter(
            x=history["Cycle"],
            y=history["Delivery Rate"],
            name="Delivery %",
            line=dict(color="#00ff88", width=3),
            yaxis="y2"
        ))

        chart.update_layout(
            height=330,
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            font_color="white",
            margin=dict(l=5, r=5, t=20, b=5),
            xaxis_title="Cycle",
            yaxis=dict(title="Energy (J)"),
            yaxis2=dict(
                title="Delivery %",
                overlaying="y",
                side="right"
            )
        )

        st.plotly_chart(chart, use_container_width=True)

    st.subheader("🚨 System Events")

    if st.session_state.events:
        for event in reversed(st.session_state.events[-8:]):
            st.write(event)
    else:
        st.success("System operating normally.")

st.subheader("📊 Sensor Node Status")

rows = []

for n in nodes:
    status = (
        "SINK" if n == sink else
        "FAILED" if n in st.session_state.faults else
        "DEAD" if energy[n] <= 0 else
        "LOW" if energy[n] < initial_energy * 0.2 else
        "ACTIVE"
    )

    rows.append({
        "Node": n,
        "Energy (J)": round(energy[n], 3),
        "Neighbors": len(neighbors(n)),
        "Status": status
    })

st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
    hide_index=True
)

st.subheader("🎯 Target Detection")

if st.session_state.target:
    tx, ty = st.session_state.target

    detected = []

    for n in nodes:
        if energy[n] > 0:
            d = np.sqrt(
                (nodes[n][0] - tx) ** 2 +
                (nodes[n][1] - ty) ** 2
            )

            if d <= radio_range:
                detected.append(n)

    if detected:
        st.success(
            f"Target detected by sensor nodes: "
            f"{', '.join(map(str, detected))}"
        )
    else:
        st.warning("Target is currently outside sensor coverage.")
else:
    st.info("Press 'Generate Target' to simulate an event.")

if not history.empty:
    st.download_button(
        "⬇️ Download Experiment Log",
        data=history.to_csv(index=False),
        file_name="wsn_experiment_log.csv",
        mime="text/csv"
    )
