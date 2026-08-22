import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import time

st.set_page_config(page_title="WSN Live Digital Twin", page_icon="📡", layout="wide")

st.markdown("""
<style>
.main {background:#07111f;}
.metric-card {padding:12px;border-radius:12px;background:#101d30;}
h1,h2,h3 {color:#42d9ff;}
</style>
""", unsafe_allow_html=True)

st.title("📡 WSN Live Digital Twin")
st.caption("Real-time Wireless Sensor Network • Energy • Routing • Packet Delivery • Fault Detection")

if "nodes" not in st.session_state:
    st.session_state.nodes = None
    st.session_state.graph = None
    st.session_state.energy = {}
    st.session_state.sent = 0
    st.session_state.delivered = 0
    st.session_state.dropped = 0
    st.session_state.tick = 0
    st.session_state.running = False
    st.session_state.history = []
    st.session_state.alerts = []

with st.sidebar:
    st.header("⚙️ Network Configuration")
    n_nodes = st.slider("Sensor nodes", 10, 80, 30)
    area = st.slider("Deployment area", 50, 500, 200)
    radius = st.slider("Radio range", 20, 120, 65)
    battery = st.slider("Initial energy (J)", 10.0, 100.0, 50.0)
    packet_rate = st.slider("Packets / cycle", 1, 20, 5)
    routing = st.selectbox(
        "Routing protocol",
        ["Energy-Aware", "Shortest Path", "Minimum Hop", "Gossiping"]
    )
    seed = st.number_input("Network seed", 1, 9999, 42)
    interval = st.slider("Refresh interval (sec)", 1, 5, 2)

    c1, c2 = st.columns(2)
    with c1:
        create = st.button("🚀 Deploy")
    with c2:
        reset = st.button("🔄 Reset")

    if create:
        np.random.seed(seed)
        positions = {
            i: (np.random.uniform(0, area), np.random.uniform(0, area))
            for i in range(n_nodes)
        }
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                d = np.linalg.norm(
                    np.array(positions[i]) - np.array(positions[j])
                )
                if d <= radius:
                    G.add_edge(i, j, distance=d)

        st.session_state.nodes = positions
        st.session_state.graph = G
        st.session_state.energy = {i: battery for i in range(n_nodes)}
        st.session_state.sent = 0
        st.session_state.delivered = 0
        st.session_state.dropped = 0
        st.session_state.tick = 0
        st.session_state.history = []
        st.session_state.alerts = []
        st.session_state.running = True

    if reset:
        st.session_state.nodes = None
        st.session_state.graph = None
        st.session_state.running = False
        st.session_state.history = []
        st.session_state.alerts = []

    st.divider()
    st.subheader("🎯 Live Controls")
    if st.button("▶️ Start / Pause"):
        st.session_state.running = not st.session_state.running

if st.session_state.graph is None:
    st.info("Deploy the sensor network from the sidebar to start the digital twin.")
    st.stop()

G = st.session_state.graph
positions = st.session_state.nodes
energy = st.session_state.energy

if st.session_state.running:
    st_autorefresh(
        interval=interval * 1000,
        limit=None,
        key="wsn_refresh"
    )

    alive = [n for n, e in energy.items() if e > 0]
    if len(alive) >= 2:
        source = np.random.choice(alive)
        sink = max(alive, key=lambda x: positions[x][1])

        for _ in range(packet_rate):
            st.session_state.sent += 1
            alive = [n for n, e in energy.items() if e > 0]

            if source not in alive or sink not in alive:
                st.session_state.dropped += 1
                continue

            try:
                if routing == "Shortest Path":
                    path = nx.shortest_path(G, source, sink)
                elif routing == "Minimum Hop":
                    path = nx.shortest_path(G, source, sink)
                elif routing == "Energy-Aware":
                    H = G.copy()
                    for u, v in H.edges():
                        eu = max(energy[u], 0.1)
                        ev = max(energy[v], 0.1)
                        H[u][v]["weight"] = 1 / min(eu, ev)
                    path = nx.shortest_path(H, source, sink, weight="weight")
                else:
                    current = source
                    path = [source]
                    for _ in range(8):
                        choices = [
                            x for x in G.neighbors(current)
                            if energy[x] > 0 and x not in path
                        ]
                        if not choices:
                            break
                        current = np.random.choice(choices)
                        path.append(current)
                        if current == sink:
                            break

                if len(path) >= 2 and path[-1] == sink:
                    cost = 0.08 * len(path)
                    for node in path:
                        energy[node] = max(0, energy[node] - cost)
                    st.session_state.delivered += 1
                else:
                    st.session_state.dropped += 1

            except nx.NetworkXNoPath:
                st.session_state.dropped += 1

        st.session_state.tick += 1

        for node in alive:
            if np.random.random() < 0.12:
                energy[node] = max(0, energy[node] - 0.015)

        low_nodes = [n for n, e in energy.items() if 0 < e < battery * 0.2]
        dead_nodes = [n for n, e in energy.items() if e <= 0]

        if low_nodes:
            message = f"⚠️ Low energy nodes: {len(low_nodes)}"
            if message not in st.session_state.alerts[-3:]:
                st.session_state.alerts.append(message)

        if dead_nodes:
            message = f"🔴 Dead nodes detected: {len(dead_nodes)}"
            if message not in st.session_state.alerts[-3:]:
                st.session_state.alerts.append(message)

        avg_energy = np.mean(list(energy.values()))
        delivery = (
            100 * st.session_state.delivered / st.session_state.sent
            if st.session_state.sent else 0
        )

        st.session_state.history.append({
            "Cycle": st.session_state.tick,
            "Average Energy": avg_energy,
            "Delivery Rate": delivery,
            "Alive Nodes": len(alive),
            "Dead Nodes": len(dead_nodes)
        })

col1, col2, col3, col4, col5 = st.columns(5)

alive_count = sum(e > 0 for e in energy.values())
avg_energy = np.mean(list(energy.values()))
delivery_rate = (
    100 * st.session_state.delivered / st.session_state.sent
    if st.session_state.sent else 0
)

col1.metric("🟢 Alive", alive_count)
col2.metric("⚡ Avg Energy", f"{avg_energy:.2f} J")
col3.metric("📦 Packets", st.session_state.sent)
col4.metric("✅ Delivery", f"{delivery_rate:.1f}%")
col5.metric("❌ Dropped", st.session_state.dropped)

left, right = st.columns([1.7, 1])

with left:
    st.subheader("🗺️ Live Sensor Network")

    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [positions[u][0], positions[v][0], None]
        edge_y += [positions[u][1], positions[v][1], None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="#33485f", width=1),
        hoverinfo="none"
    )

    node_x = [positions[n][0] for n in G.nodes()]
    node_y = [positions[n][1] for n in G.nodes()]
    node_energy = [energy[n] for n in G.nodes()]

    colors = [
        "#ff3131" if e <= 0 else
        "#ffb000" if e < battery * 0.2 else
        "#00ff99"
        for e in node_energy
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        marker=dict(
            size=14,
            color=colors,
            line=dict(color="white", width=1)
        ),
        customdata=np.round(node_energy, 2),
        hovertemplate="Node %{text}<br>Energy: %{customdata} J<extra></extra>"
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        height=560,
        margin=dict(l=5, r=5, t=10, b=5),
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        font_color="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📈 Network Health")

    history = pd.DataFrame(st.session_state.history)

    if not history.empty:
        chart = go.Figure()
        chart.add_trace(go.Scatter(
            x=history["Cycle"],
            y=history["Average Energy"],
            name="Avg Energy",
            line=dict(color="#00e5ff", width=3)
        ))
        chart.add_trace(go.Scatter(
            x=history["Cycle"],
            y=history["Delivery Rate"],
            name="Delivery %",
            line=dict(color="#00ff99", width=3),
            yaxis="y2"
        ))
        chart.update_layout(
            height=300,
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            font_color="white",
            xaxis_title="Cycle",
            yaxis=dict(title="Energy"),
            yaxis2=dict(
                title="Delivery %",
                overlaying="y",
                side="right"
            ),
            margin=dict(l=5, r=5, t=20, b=5)
        )
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("🚨 Alerts")
    if st.session_state.alerts:
        for alert in reversed(st.session_state.alerts[-5:]):
            st.warning(alert)
    else:
        st.success("No critical events detected.")

st.subheader("📊 Node Energy Table")

table = pd.DataFrame({
    "Node": list(energy.keys()),
    "Energy (J)": [round(v, 3) for v in energy.values()],
    "Status": [
        "DEAD" if v <= 0 else
        "LOW" if v < battery * 0.2 else
        "ACTIVE"
        for v in energy.values()
    ],
    "Neighbors": [G.degree(n) for n in G.nodes()]
})

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Energy (J)": st.column_config.NumberColumn(format="%.3f")
    }
)

st.subheader("🧪 Protocol Experiment")

if len(history := pd.DataFrame(st.session_state.history)) > 2:
    latest = history.iloc[-1]
    first = history.iloc[0]
    energy_used = max(0, first["Average Energy"] - latest["Average Energy"])

    a, b, c = st.columns(3)
    a.metric("Routing", routing)
    b.metric("Energy Consumed", f"{energy_used:.2f} J")
    c.metric("Current Cycle", int(latest["Cycle"]))

st.download_button(
    "⬇️ Download Experiment Log",
    data=pd.DataFrame(st.session_state.history).to_csv(index=False),
    file_name="wsn_experiment_log.csv",
    mime="text/csv"
)

st.caption(
    "This project uses algorithmic sensor generation only — no dataset and no ML model. "
    "Suitable for demonstrating WSN routing, MAC/energy concepts, topology changes, "
    "fault detection and performance analysis."
)
