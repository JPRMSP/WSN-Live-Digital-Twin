import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="WSN Sentinel",
    page_icon="📡",
    layout="wide"
)

st.title("📡 WSN Sentinel")
st.caption("Real-Time Energy-Aware Wireless Sensor Network Digital Twin")

# ---------- SESSION STATE ----------
defaults = {
    "nodes": None,
    "energy": {},
    "faults": set(),
    "target": None,
    "running": False,
    "cycle": 0,
    "sent": 0,
    "delivered": 0,
    "dropped": 0,
    "history": [],
    "events": [],
    "last_update": 0.0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Network Configuration")

    node_count = st.slider(
        "Sensor Nodes",
        min_value=10,
        max_value=60,
        value=30
    )

    field_size = st.slider(
        "Field Size",
        min_value=100,
        max_value=500,
        value=250
    )

    radio_range = st.slider(
        "Radio Range",
        min_value=25,
        max_value=120,
        value=65
    )

    initial_energy = st.slider(
        "Initial Battery (J)",
        min_value=10,
        max_value=100,
        value=50
    )

    packets_cycle = st.slider(
        "Packets / Cycle",
        min_value=1,
        max_value=15,
        value=5
    )

    routing = st.selectbox(
        "Routing Algorithm",
        [
            "Energy-Aware",
            "Nearest Neighbor",
            "Geographic"
        ]
    )

    seed = st.number_input(
        "Simulation Seed",
        min_value=1,
        max_value=9999,
        value=42
    )

    refresh_time = st.slider(
        "Refresh Time (seconds)",
        1,
        5,
        2
    )

    st.divider()

    deploy = st.button(
        "🚀 Deploy Network",
        use_container_width=True
    )

    start = st.button(
        "▶️ Start / Pause",
        use_container_width=True
    )

    target_button = st.button(
        "🎯 Generate Target",
        use_container_width=True
    )

    fault_button = st.button(
        "💥 Simulate Node Failure",
        use_container_width=True
    )

    reset = st.button(
        "🔄 Reset",
        use_container_width=True
    )

# ---------- RESET ----------
if reset:
    st.session_state.nodes = None
    st.session_state.energy = {}
    st.session_state.faults = set()
    st.session_state.target = None
    st.session_state.running = False
    st.session_state.cycle = 0
    st.session_state.sent = 0
    st.session_state.delivered = 0
    st.session_state.dropped = 0
    st.session_state.history = []
    st.session_state.events = []
    st.rerun()

# ---------- DEPLOY ----------
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
        i: float(initial_energy)
        for i in range(node_count)
    }

    st.session_state.faults = set()
    st.session_state.target = None
    st.session_state.running = False
    st.session_state.cycle = 0
    st.session_state.sent = 0
    st.session_state.delivered = 0
    st.session_state.dropped = 0
    st.session_state.history = []
    st.session_state.events = []

# ---------- NO NETWORK ----------
if st.session_state.nodes is None:
    st.info("Deploy the sensor network from the sidebar to begin.")
    st.stop()

nodes = st.session_state.nodes
energy = st.session_state.energy

# ---------- BASIC FUNCTIONS ----------
def distance(a, b):
    ax, ay = nodes[a]
    bx, by = nodes[b]

    return float(
        np.sqrt(
            (ax - bx) ** 2 +
            (ay - by) ** 2
        )
    )


def distance_to_point(node, point):
    x, y = nodes[node]

    return float(
        np.sqrt(
            (x - point[0]) ** 2 +
            (y - point[1]) ** 2
        )
    )


def get_neighbors(node):
    result = []

    for other in nodes:
        if other == node:
            continue

        if other in st.session_state.faults:
            continue

        if energy[other] <= 0:
            continue

        if distance(node, other) <= radio_range:
            result.append(other)

    return result


def alive_nodes():
    return [
        n for n in nodes
        if energy[n] > 0
        and n not in st.session_state.faults
    ]


def find_route(source, sink):
    """
    Manual routing algorithm.
    No NetworkX is required.
    """

    if source == sink:
        return [source]

    current = source
    path = [source]
    visited = {source}

    for _ in range(node_count):

        if current == sink:
            return path

        candidates = [
            n for n in get_neighbors(current)
            if n not in visited
        ]

        if not candidates:
            return []

        if routing == "Nearest Neighbor":
            next_node = min(
                candidates,
                key=lambda n: distance(n, sink)
            )

        elif routing == "Geographic":
            next_node = min(
                candidates,
                key=lambda n: (
                    distance(n, sink) +
                    0.05 * distance(current, n)
                )
            )

        else:
            # Energy-aware routing
            next_node = max(
                candidates,
                key=lambda n: (
                    energy[n] /
                    max(distance(n, sink), 1)
                )
            )

        current = next_node
        visited.add(current)
        path.append(current)

    return []


def add_event(message):
    st.session_state.events.append(message)

    if len(st.session_state.events) > 20:
        st.session_state.events = (
            st.session_state.events[-20:]
        )


# ---------- TARGET ----------
if target_button:
    rng = np.random.default_rng()

    st.session_state.target = (
        float(rng.uniform(20, field_size - 20)),
        float(rng.uniform(20, field_size - 20))
    )

    tx, ty = st.session_state.target

    detecting_nodes = []

    for node in nodes:
        if energy[node] > 0:
            if distance_to_point(
                node,
                st.session_state.target
            ) <= radio_range:
                detecting_nodes.append(node)

    if detecting_nodes:
        add_event(
            f"🎯 Target detected by "
            f"{len(detecting_nodes)} sensor(s)"
        )
    else:
        add_event(
            f"🎯 Target appeared at "
            f"({tx:.1f}, {ty:.1f})"
        )

# ---------- FAULT ----------
if fault_button:

    available = [
        n for n in alive_nodes()
        if n != max(nodes, key=lambda n: nodes[n][1])
    ]

    if available:
        failed = int(
            np.random.choice(available)
        )

        st.session_state.faults.add(failed)
        energy[failed] = 0

        add_event(
            f"💥 Node {failed} failed"
        )

# ---------- START / PAUSE ----------
if start:
    st.session_state.running = (
        not st.session_state.running
    )

# ---------- SIMULATION ----------
if st.session_state.running:

    current_time = time.time()

    if (
        current_time -
        st.session_state.last_update
        >= refresh_time
    ):

        st.session_state.last_update = current_time

        alive = alive_nodes()

        if len(alive) >= 2:

            sink = max(
                alive,
                key=lambda n: nodes[n][1]
            )

            for _ in range(packets_cycle):

                if len(alive) <= 1:
                    break

                source_candidates = [
                    n for n in alive
                    if n != sink
                ]

                if not source_candidates:
                    break

                source = int(
                    np.random.choice(
                        source_candidates
                    )
                )

                st.session_state.sent += 1

                route = find_route(
                    source,
                    sink
                )

                if route:

                    hop_cost = (
                        0.04 +
                        0.008 * len(route)
                    )

                    for node in route:
                        energy[node] = max(
                            0,
                            energy[node] - hop_cost
                        )

                    st.session_state.delivered += 1

                else:
                    st.session_state.dropped += 1

            # Idle energy consumption
            for node in alive:
                if node != sink:
                    energy[node] = max(
                        0,
                        energy[node] -
                        np.random.uniform(
                            0.003,
                            0.015
                        )
                    )

            st.session_state.cycle += 1

            active_count = len(
                alive_nodes()
            )

            average_energy = float(
                np.mean(
                    list(energy.values())
                )
            )

            delivery_rate = (
                st.session_state.delivered /
                max(
                    st.session_state.sent,
                    1
                ) *
                100
            )

            st.session_state.history.append({
                "Cycle": st.session_state.cycle,
                "Average Energy":
                    round(
                        average_energy,
                        3
                    ),
                "Delivery Rate":
                    round(
                        delivery_rate,
                        2
                    ),
                "Alive Nodes":
                    active_count,
                "Dead Nodes":
                    node_count -
                    active_count
            })

            if average_energy < (
                initial_energy * 0.2
            ):
                add_event(
                    "⚠️ Network energy below 20%"
                )

            if active_count <= (
                node_count * 0.5
            ):
                add_event(
                    "🚨 More than 50% nodes inactive"
                )

            st.rerun()

# ---------- METRICS ----------
alive_count = len(alive_nodes())

average_energy = float(
    np.mean(
        list(energy.values())
    )
)

delivery_rate = (
    st.session_state.delivered /
    max(
        st.session_state.sent,
        1
    ) *
    100
)

dead_count = (
    node_count -
    alive_count
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "🟢 Alive",
    alive_count
)

col2.metric(
    "⚡ Avg Energy",
    f"{average_energy:.2f} J"
)

col3.metric(
    "📦 Packets",
    st.session_state.sent
)

col4.metric(
    "✅ Delivery",
    f"{delivery_rate:.1f}%"
)

col5.metric(
    "🔴 Dead",
    dead_count
)

# ---------- NETWORK MAP ----------
st.subheader("🗺️ Live Sensor Network")

map_rows = []

for node in nodes:

    if node in st.session_state.faults:
        status = "FAILED"

    elif energy[node] <= 0:
        status = "DEAD"

    elif node == max(
        nodes,
        key=lambda n: nodes[n][1]
    ):
        status = "SINK"

    elif energy[node] < (
        initial_energy * 0.2
    ):
        status = "LOW"

    else:
        status = "ACTIVE"

    map_rows.append({
        "Node": node,
        "X": nodes[node][0],
        "Y": nodes[node][1],
        "Energy": round(
            energy[node],
            2
        ),
        "Status": status
    })

map_df = pd.DataFrame(map_rows)

st.scatter_chart(
    map_df,
    x="X",
    y="Y",
    color="Status",
    size="Energy",
    height=500
)

# ---------- PERFORMANCE ----------
st.subheader("📈 Network Performance")

history_df = pd.DataFrame(
    st.session_state.history
)

if not history_df.empty:

    performance_df = history_df[
        [
            "Cycle",
            "Average Energy",
            "Delivery Rate"
        ]
    ].set_index("Cycle")

    st.line_chart(
        performance_df,
        height=320
    )

else:
    st.info(
        "Start the simulation to generate "
        "performance data."
    )

# ---------- TARGET ----------
st.subheader("🎯 Target Detection")

if st.session_state.target:

    tx, ty = st.session_state.target

    detected = []

    for node in nodes:

        if energy[node] <= 0:
            continue

        if distance_to_point(
            node,
            st.session_state.target
        ) <= radio_range:

            detected.append(node)

    st.write(
        f"Target location: "
        f"**({tx:.1f}, {ty:.1f})**"
    )

    if detected:

        st.success(
            "Target detected by nodes: "
            +
            ", ".join(
                map(str, detected)
            )
        )

    else:

        st.warning(
            "Target is outside current "
            "sensor coverage."
        )

else:

    st.info(
        "Generate a target to demonstrate "
        "event detection."
    )

# ---------- NODE TABLE ----------
st.subheader("📊 Sensor Node Status")

table_rows = []

for node in nodes:

    neighbors_count = len(
        get_neighbors(node)
    )

    if node in st.session_state.faults:
        status = "FAILED"

    elif energy[node] <= 0:
        status = "DEAD"

    elif energy[node] < (
        initial_energy * 0.2
    ):
        status = "LOW"

    else:
        status = "ACTIVE"

    table_rows.append({
        "Node": node,
        "Energy (J)": round(
            energy[node],
            3
        ),
        "Neighbors":
            neighbors_count,
        "Status": status
    })

node_df = pd.DataFrame(
    table_rows
)

st.dataframe(
    node_df,
    use_container_width=True,
    hide_index=True
)

# ---------- EVENTS ----------
st.subheader("🚨 System Events")

if st.session_state.events:

    for event in reversed(
        st.session_state.events[-8:]
    ):
        st.write(event)

else:

    st.success(
        "No critical events detected."
    )

# ---------- EXPERIMENT LOG ----------
if not history_df.empty:

    st.subheader(
        "🧪 Experiment Results"
    )

    first_energy = history_df[
        "Average Energy"
    ].iloc[0]

    final_energy = history_df[
        "Average Energy"
    ].iloc[-1]

    consumed = max(
        0,
        first_energy -
        final_energy
    )

    a, b, c = st.columns(3)

    a.metric(
        "Routing",
        routing
    )

    b.metric(
        "Energy Consumed",
        f"{consumed:.2f} J"
    )

    c.metric(
        "Simulation Cycles",
        st.session_state.cycle
    )

    st.download_button(
        "⬇️ Download Experiment Log",
        data=history_df.to_csv(
            index=False
        ),
        file_name=(
            "wsn_experiment_log.csv"
        ),
        mime="text/csv"
    )

st.divider()
