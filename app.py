import re
import joblib
import numpy as np
from pathlib import Path
from datetime import date, timedelta, datetime
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CSPACE | Campus Space Intelligence",
    page_icon="🏫",
    layout="wide"
)

DATA_DIR = Path(__file__).parent

MODEL_PATH = DATA_DIR / "models" / "demand_model.joblib"


@st.cache_resource
def load_demand_model():
    """Load the previously trained DemandCast model."""
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        print(f"DemandCast model loading failed: {error}")
        return None


def predict_room_demand(selected_date, selected_hour):
    """Predict hourly request volume using the saved model."""
    artifact = load_demand_model()

    if artifact is None:
        return None

    model = artifact["model"]
    features = artifact["features"]

    weekday = selected_date.weekday()

    row = {
        "hour": selected_hour,
        "weekday": weekday,
        "month": selected_date.month,
        "day_of_month": selected_date.day,
        "hour_sin": np.sin(2 * np.pi * selected_hour / 24),
        "hour_cos": np.cos(2 * np.pi * selected_hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
    }

    prediction_input = pd.DataFrame(
        [[row[feature] for feature in features]],
        columns=features
    )

    prediction = float(model.predict(prediction_input)[0])

    return max(0, prediction)

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

@st.cache_data
def load_data():
    files = {
        "rooms": "rooms.csv",
        "schedule": "class_schedule.csv",
        "requests": "room_requests.csv",
        "events": "events.csv",
    }

    data = {}

    for key, filename in files.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {filename}. Place it beside app.py."
            )

        data[key] = pd.read_csv(path)

    data["events"]["start_time"] = pd.to_datetime(
        data["events"]["start_time"],
        errors="coerce"
    )

    data["requests"]["request_date"] = pd.to_datetime(
        data["requests"]["request_date"],
        errors="coerce"
    ).dt.normalize()

    return data


try:
    data = load_data()

    rooms = data["rooms"]
    schedule = data["schedule"]
    requests = data["requests"]
    events = data["events"]

except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def has_conflict(
    room_id,
    selected_date,
    start_hour,
    duration
):
    """Check recurring classes and dated events."""

    requested_start = float(start_hour)
    requested_end = requested_start + float(duration)

    weekday = selected_date.strftime("%a")

    # Check recurring class schedules.
    class_rows = schedule[
        (schedule["room_id"] == room_id)
        & (schedule["weekday"].astype(str).str[:3] == weekday)
    ]

    for _, row in class_rows.iterrows():
        class_start = float(row["start_hour"])
        class_end = class_start + float(row["duration_hours"])

        overlaps = (
            requested_start < class_end
            and class_start < requested_end
        )

        if overlaps:
            return True, "Recurring class conflict"

    # Check events on the selected calendar date.
    event_rows = events[
        (events["room_id"] == room_id)
        & (events["start_time"].dt.date == selected_date)
    ]

    for _, row in event_rows.iterrows():
        event_start = (
            row["start_time"].hour
            + row["start_time"].minute / 60
        )

        event_end = event_start + float(row["duration_hours"])

        overlaps = (
            requested_start < event_end
            and event_start < requested_end
        )

        if overlaps:
            return True, "Scheduled event conflict"

    return False, "No conflict found in supplied schedules"


def recommend_rooms(
    selected_date,
    start_hour,
    duration,
    group_size,
    projector_required,
    computers_required
):
    """Filter and rank rooms against user requirements."""

    candidates = rooms.copy()

    # Required capacity.
    candidates = candidates[
        pd.to_numeric(
            candidates["capacity"], errors="coerce"
        ) >= group_size
    ]

    # Required facilities.
    if projector_required:
        candidates = candidates[
            pd.to_numeric(
                candidates["projector"], errors="coerce"
            ).fillna(0) == 1
        ]

    if computers_required:
        candidates = candidates[
            pd.to_numeric(
                candidates["computers"], errors="coerce"
            ).fillna(0) == 1
        ]

    results = []

    for _, room in candidates.iterrows():

        conflict, reason = has_conflict(
            room["room_id"],
            selected_date,
            start_hour,
            duration
        )

        if conflict:
            continue

        capacity = float(room["capacity"])

        # Prefer rooms that fit the group without excessive
        # unused capacity. This is a ranking heuristic.
        capacity_score = (
            100 * group_size / capacity
        )

        results.append({
            "Room ID": room["room_id"],
            "Building": room["building"],
            "Floor": room["floor"],
            "Room Type": room["room_type"],
            "Capacity": int(capacity),
            "Projector": int(room["projector"]),
            "Computers": int(room["computers"]),
            "Accessibility": int(room["accessible"]),
            "Suitability Score": round(
                capacity_score, 1
            ),
            "Availability Check": reason
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values(
            "Suitability Score",
            ascending=False
        )

    return result_df
def find_alternative_slots(
    selected_date,
    start_hour,
    duration,
    group_size,
    projector_required,
    computers_required,
    limit=3
):
    """Find feasible alternatives near the requested time."""

    alternatives = []

    # Search the same date first, then the following day.
    candidate_dates = [
        selected_date,
        selected_date + timedelta(days=1)
    ]

    for candidate_date in candidate_dates:

        for candidate_hour in range(8, 21):

            if candidate_hour == start_hour and candidate_date == selected_date:
                continue

            if candidate_hour + duration > 24:
                continue

            rooms_available = recommend_rooms(
                candidate_date,
                candidate_hour,
                duration,
                group_size,
                projector_required,
                computers_required
            )

            if rooms_available.empty:
                continue

            predicted_demand = predict_room_demand(
                candidate_date,
                candidate_hour
            )

            alternatives.append({
                "Date": candidate_date.isoformat(),
                "Start Time": f"{candidate_hour:02d}:00",
                "Available Rooms": len(rooms_available),
                "Best Room": rooms_available.iloc[0]["Room ID"],
                "Predicted Requests": (
                    round(predicted_demand, 2)
                    if predicted_demand is not None
                    else None
                )
            })

    alternatives.sort(
        key=lambda item: (
            item["Predicted Requests"]
            if item["Predicted Requests"] is not None
            else float("inf"),
            -item["Available Rooms"]
        )
    )

    return pd.DataFrame(alternatives[:limit])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🏫 CSPACE")

st.sidebar.caption(
    "Campus Space Intelligence"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Smart Room Finder",
        "CSPACE Copilot",
        "Campus Analytics",
        "Dataset Explorer"
    ]
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏫 CSPACE")

st.caption(
    "Find Your Space. Predict Demand. Optimize Campus."
)

st.info(
    "Prototype using synthetic historical data. "
    "Availability results are based on the supplied schedules "
    "and events, not live campus booking information."
)


# --------------------------------------------------
# PAGE 1: SMART ROOM FINDER
# --------------------------------------------------

if page == "Smart Room Finder":

    st.header("🔎 Find Your Perfect Space")

    st.write(
        "Specify your requirements to find rooms that match "
        "your capacity, facilities and time constraints."
    )

    valid_dates = pd.concat([
        requests["request_date"],
        events["start_time"].dt.normalize()
    ]).dropna()

    if valid_dates.empty:
        st.error("No usable dates found in the datasets.")
        st.stop()

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    selected_date = st.date_input(
        "Select date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        group_size = st.number_input(
            "Number of people",
            min_value=1,
            max_value=500,
            value=20
        )

    with col2:
        start_hour = st.selectbox(
            "Start time",
            list(range(8, 21)),
            index=6,
            format_func=lambda x: f"{x:02d}:00"
        )

    with col3:
        duration = st.selectbox(
            "Duration",
            [1, 2, 3, 4],
            index=1,
            format_func=lambda x: f"{x} hour(s)"
        )

    col4, col5 = st.columns(2)

    with col4:
        projector_required = st.checkbox(
            "Projector required"
        )

    with col5:
        computers_required = st.checkbox(
            "Computers required"
        )

    if st.button(
        "Find Suitable Rooms",
        type="primary",
        use_container_width=True
    ):

        if start_hour + duration > 24:
            st.error(
                "This prototype supports requests that end "
                "by midnight. Select an earlier start time."
            )

        else:
            results = recommend_rooms(
                selected_date,
                start_hour,
                duration,
                group_size,
                projector_required,
                computers_required
            )

            st.subheader("Your Recommendations")

            if results.empty:

                st.warning(
                    "No rooms matched all your requirements "
                    "for the selected time. Try another time, "
                    "a smaller group, or fewer required facilities."
                )

            else:

                st.success(
                    f"Found {len(results)} suitable room(s)."
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

                best = results.iloc[0]

                st.subheader("⭐ Best Match")

                st.markdown(
                    f"""
                    **Room:** {best['Room ID']}

                    **Building:** {best['Building']}

                    **Capacity:** {best['Capacity']} people

                    **Room type:** {best['Room Type']}

                    **Suitability score:** {best['Suitability Score']}/100

                    **Why recommended:** This room satisfies the
                    requested capacity and selected facility
                    requirements, and no overlapping class or event
                    was found in the supplied schedule records.
                    """
                )

                st.caption(
                    "The score is a capacity-fit ranking, not a "
                    "probability of availability or a learned ML prediction."
                )


# --------------------------------------------------
# PAGE 2: CSPACE COPILOT
# --------------------------------------------------


elif page == "CSPACE Copilot":

    st.header("🤖 CSPACE Copilot")

    st.write(
        "Describe your requirements and CSPACE will find suitable rooms."
    )

    user_text = st.text_input(
        "Your request",
        placeholder=(
            "Find a room for 20 people from 2 PM to 4 PM "
            "on 6 August 2025 with a projector"
        )
    )

    # Use the date range represented in the historical datasets.
    valid_dates = pd.concat([
        requests["request_date"],
        events["start_time"].dt.normalize()
    ]).dropna()

    if valid_dates.empty:
        st.error("No valid dates are available in the datasets.")
        st.stop()

    default_date = valid_dates.min().date()

    selected_date = st.date_input(
        "Meeting date",
        value=default_date,
        min_value=valid_dates.min().date(),
        max_value=valid_dates.max().date()
    )

    if st.button("🔍 Find My Room", type="primary"):

        if not user_text.strip():
            st.warning("Please describe the room you need.")

        else:
            text = user_text.lower()

            # Extract group size.
            people_match = re.search(
                r"(\d+)\s*(?:people|students|persons|members)",
                text
            )

            group_size = (
                int(people_match.group(1))
                if people_match
                else 20
            )

            # Extract start and end times.
            time_match = re.search(
                r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
                r"\s*(?:to|-)\s*"
                r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
                text
            )

            start_hour = 14
            duration = 2

            if time_match:

                h1 = int(time_match.group(1))
                m1 = int(time_match.group(2) or 0)
                ap1 = time_match.group(3)

                h2 = int(time_match.group(4))
                m2 = int(time_match.group(5) or 0)
                ap2 = time_match.group(6)

                # Infer missing AM/PM when possible.
                if ap1 is None and ap2:
                    ap1 = ap2

                if ap2 is None and ap1:
                    ap2 = ap1

                if ap1 == "pm" and h1 < 12:
                    h1 += 12
                elif ap1 == "am" and h1 == 12:
                    h1 = 0

                if ap2 == "pm" and h2 < 12:
                    h2 += 12
                elif ap2 == "am" and h2 == 12:
                    h2 = 0

                start_hour = h1 + m1 / 60
                end_hour = h2 + m2 / 60

                # Handle intervals such as 10 PM to 1 AM.
                if end_hour <= start_hour:
                    end_hour += 24

                duration = end_hour - start_hour

            if start_hour < 0 or start_hour >= 24:
                st.error("Please provide a valid start time.")
                st.stop()

            if start_hour + duration > 24:
                st.error(
                    "Overnight bookings are not supported by this prototype."
                )
                st.stop()

            if duration <= 0:
                st.error("Please provide a valid time interval.")
                st.stop()

            projector_required = "projector" in text

            computers_required = (
                "computer" in text
                or "computers" in text
            )

        

            results = recommend_rooms(
                selected_date,
                start_hour,
                duration,
                group_size,
                projector_required,
                computers_required
            )

            st.write("### Recommended Rooms")

            if results.empty:

                st.warning(
                    "No suitable rooms were found for these "
                    "requirements and this date. Try another date "
                    "or relax a nonessential requirement."
                )

            else:

                st.success(
                    f"Found {len(results)} matching rooms."
                )

                st.dataframe(
                    results.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                best = results.iloc[0]

                st.subheader("⭐ Best Match")

                st.write(
                    f"**Room:** {best['Room ID']}"
                )

                st.write(
                    f"**Building:** {best['Building']}"
                )

                st.write(
                    f"**Capacity:** {best['Capacity']}"
                )

                st.write(
                    f"**Suitability score:** "
                    f"{best['Suitability Score']}/100"
                )

                st.success(
                    "This room meets the selected capacity and "
                    "facility requirements, with no overlapping "
                    "class or event found in the supplied data."
                )


# --------------------------------------------------
# PAGE 3: CAMPUS ANALYTICS
# --------------------------------------------------

elif page == "Campus Analytics":

    st.header("📊 Campus Space Intelligence")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Rooms", len(rooms))
    c2.metric("Class Records", len(schedule))
    c3.metric("Room Requests", len(requests))
    c4.metric("Events", len(events))

    st.subheader("Room Capacity Distribution")

    fig = px.histogram(
        rooms,
        x="capacity",
        color="room_type",
        title="Room Capacity by Room Type",
        labels={
            "capacity": "Room Capacity",
            "room_type": "Room Type"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Room Requests by Purpose")

    purpose_counts = (
        requests["purpose"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Purpose")
        .reset_index(name="Requests")
    )

    fig = px.bar(
        purpose_counts,
        x="Purpose",
        y="Requests",
        title="Historical Requests by Purpose"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Historical Request Status")

    status_counts = (
        requests["status"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Requests")
    )

    fig = px.pie(
        status_counts,
        names="Status",
        values="Requests",
        title="Room Request Status Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "These visualizations describe historical synthetic data. "
        "They do not constitute a validated demand forecast."
    )
    # ------------------------------------------
    # DEMANDCAST: CAMPUS ROOM DEMAND INTELLIGENCE
    # ------------------------------------------

    st.divider()
    st.header("🔮 DemandCast — Predictive Room Demand")

    demand = requests.copy()

    demand["request_date"] = pd.to_datetime(
        demand["request_date"], errors="coerce"
    )

    demand["start_hour"] = pd.to_numeric(
        demand["start_hour"], errors="coerce"
    )

    demand = demand.dropna(
        subset=["request_date", "start_hour"]
    )

    demand["weekday"] = demand["request_date"].dt.dayofweek
    demand["hour"] = demand["start_hour"].astype(int)

    weekday_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

        # ------------------------------------------
    # HISTORICAL ROOM DEMAND HEATMAP
    # ------------------------------------------

    st.subheader("🔥 Historical Room Demand Heatmap")

    demand = requests.copy()

    demand["request_date"] = pd.to_datetime(
        demand["request_date"],
        errors="coerce"
    )

    demand["start_hour"] = pd.to_numeric(
        demand["start_hour"],
        errors="coerce"
    )

    demand = demand.dropna(
        subset=["request_date", "start_hour"]
    )

    demand["weekday"] = demand["request_date"].dt.dayofweek
    demand["hour"] = demand["start_hour"].astype(int)

    weekday_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    heatmap_data = (
        demand.groupby(["weekday", "hour"])
        .size()
        .reset_index(name="requests")
    )

    heatmap_data["day"] = heatmap_data["weekday"].map(
        dict(enumerate(weekday_names))
    )

    pivot = heatmap_data.pivot(
        index="day",
        columns="hour",
        values="requests"
    ).fillna(0)

    pivot = pivot.reindex(weekday_names)

    fig = px.imshow(
        pivot,
        labels={
            "x": "Hour of Day",
            "y": "Weekday",
            "color": "Requests"
        },
        title="Historical Room Demand by Day and Hour",
        aspect="auto",
        text_auto=True,
        color_continuous_scale=[
            [0.0, "#DBEAFE"],
            [0.35, "#60A5FA"],
            [0.60, "#FACC15"],
            [0.80, "#F97316"],
            [1.0, "#DC2626"]
        ]
    )

    fig.update_layout(
        height=450,
        coloraxis_colorbar=dict(
            title="Requests"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------
    # DEMAND PREDICTION
    # ------------------------------------------

    st.subheader("Predict Room Demand")

    # Count requests per observed date and hour.
    daily_demand = (
        demand.groupby(["request_date", "weekday", "hour"])
        .size()
        .reset_index(name="request_count")
    )

    if len(daily_demand) >= 20:

        X = daily_demand[["weekday", "hour"]]
        y = daily_demand["request_count"]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X, y)

        col1, col2 = st.columns(2)

        with col1:
            selected_day = st.selectbox(
                "Select weekday",
                weekday_names
            )

        with col2:
            selected_hour = st.selectbox(
                "Select hour",
                sorted(demand["hour"].unique())
            )

        day_number = weekday_names.index(selected_day)

        prediction_input = pd.DataFrame({
            "weekday": [day_number],
            "hour": [selected_hour]
        })

        predicted_demand = model.predict(
            prediction_input
        )[0]

        predicted_demand = max(
            0, round(float(predicted_demand))
        )

        st.metric(
            "Estimated Room Requests",
            predicted_demand
        )

        # Compare the prediction against observed demand.
        high_demand_threshold = daily_demand[
            "request_count"
        ].quantile(0.75)

        if predicted_demand >= high_demand_threshold:
            st.warning(
                "⚠️ High-demand period: Consider planning "
                "room allocations in advance."
            )
        else:
            st.success(
                "✅ Demand is relatively low compared with "
                "the historical observations."
            )

        st.caption(
            "Experimental estimate trained on synthetic "
            "historical data. Not a validated real-world forecast."
        )

    else:
        st.info(
            "Insufficient observations for this experimental "
            "demand prediction."
        )
    # ------------------------------------------
    # SUPPLY VS DEMAND ANALYSIS
    # ------------------------------------------

    st.divider()

    st.header("🚦 Campus Space Shortage Detector")

    st.write(
        "Compare historical room requests with the supply "
        "of rooms that meet capacity and scheduling constraints."
    )

    # Historical request demand by weekday and hour.
    demand_counts = requests.copy()

    demand_counts["request_date"] = pd.to_datetime(
        demand_counts["request_date"],
        errors="coerce"
    )

    demand_counts["start_hour"] = pd.to_numeric(
        demand_counts["start_hour"],
        errors="coerce"
    )

    demand_counts = demand_counts.dropna(
        subset=["request_date", "start_hour"]
    )

    demand_counts["weekday"] = (
        demand_counts["request_date"].dt.dayofweek
    )

    demand_counts["hour"] = (
        demand_counts["start_hour"].astype(int)
    )

    demand_summary = (
        demand_counts.groupby(["weekday", "hour"])
        .size()
        .reset_index(name="Historical Requests")
    )

    # Estimate how many rooms are free at each recurring
    # weekday/hour using the supplied class schedule.
    supply_records = []

    for weekday_num in range(7):

        weekday_name = [
            "Mon", "Tue", "Wed", "Thu",
            "Fri", "Sat", "Sun"
        ][weekday_num]

        weekday_classes = schedule[
            schedule["weekday"].astype(str).str[:3]
            == weekday_name
        ]

        for hour in range(8, 21):

            occupied_rooms = set()

            for _, row in weekday_classes.iterrows():

                try:
                    class_start = float(row["start_hour"])
                    class_end = (
                        class_start + float(row["duration_hours"])
                    )

                    if class_start < hour + 1 and hour < class_end:
                        occupied_rooms.add(row["room_id"])

                except (ValueError, TypeError):
                    continue

            total_rooms = rooms["room_id"].nunique()

            available_rooms = max(
                0,
                total_rooms - len(occupied_rooms)
            )

            supply_records.append({
                "weekday": weekday_num,
                "hour": hour,
                "Available Rooms": available_rooms
            })

    supply_summary = pd.DataFrame(supply_records)

    # Join demand and supply.
    comparison = demand_summary.merge(
        supply_summary,
        on=["weekday", "hour"],
        how="outer"
    )

    comparison["Historical Requests"] = (
        comparison["Historical Requests"].fillna(0)
    )

    comparison["Available Rooms"] = (
        comparison["Available Rooms"].fillna(0)
    )

    comparison["Demand Gap"] = (
        comparison["Historical Requests"]
        - comparison["Available Rooms"]
    )

    comparison["Status"] = comparison["Demand Gap"].apply(
        lambda gap: (
            "Potential Shortage"
            if gap > 0
            else "Supply Meets Count"
        )
    )

    weekday_labels = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    comparison["Weekday"] = comparison["weekday"].map(
        dict(enumerate(weekday_labels))
    )

    st.subheader("🚨 Top 10 Campus Space Shortages")

    comparison["Time"] = comparison["hour"].apply(
        lambda hour: f"{int(hour):02d}:00"
    )

    comparison["Time Slot"] = (
        comparison["Weekday"] + " " + comparison["Time"]
    )

    # Keep only periods where historical request count
    # exceeds the estimated number of available rooms.
    shortages = comparison[
        comparison["Demand Gap"] > 0
    ].copy()

    shortages = shortages.sort_values(
        "Demand Gap",
        ascending=False
    ).head(10)

    if shortages.empty:

        st.success("No potential shortages were identified.")

    else:

        shortages = shortages.sort_values(
            "Demand Gap",
            ascending=True
        )

        fig = px.bar(
            shortages,
            x="Demand Gap",
            y="Time Slot",
            orientation="h",
            text="Demand Gap",
            hover_data=[
                "Historical Requests",
                "Available Rooms"
            ],
            title="Top 10 Potential Campus Space Shortages",
            labels={
                "Demand Gap": "Request Count − Estimated Available Rooms",
                "Time Slot": "Weekday and Time"
            },
            color="Demand Gap",
            color_continuous_scale=[
                [0, "#FDBA74"],
                [0.5, "#F97316"],
                [1, "#DC2626"]
            ]
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=480,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Potential Shortage Indicator",
            yaxis_title="Time Slot"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        st.plotly_chart(
        fig,
        use_container_width=True,
        key="shortage_bar_chart"
        )
        st.info(
    "💡 CSPACE identifies periods where historical room-request "
    "volume exceeds estimated available room counts. These are "
    "potential shortage indicators, not confirmed booking shortages."
    )
    

    st.subheader("⚠️ Potential Shortage Periods")

    shortages = comparison[
        comparison["Status"] == "Potential Shortage"
    ].sort_values(
        "Demand Gap",
        ascending=False
    )

    if shortages.empty:

        st.success(
            "No potential shortages were identified "
            "by this preliminary count-based comparison."
        )

    else:

        shortages["Time"] = shortages["hour"].apply(
            lambda hour: f"{int(hour):02d}:00"
        )

        st.dataframe(
            shortages[
                [
                    "Weekday",
                    "Time",
                    "Historical Requests",
                    "Available Rooms",
                    "Demand Gap"
                ]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

    st.warning(
        "This is a preliminary screening indicator, not a confirmed "
        "shortage. It compares historical request counts with rooms "
        "estimated to be free under recurring class schedules. "
        "It does not yet account for each request's capacity, facilities, "
        "duration, dated events, or whether requests were approved "
        "or cancelled."
    )

# --------------------------------------------------
# PAGE 4: DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.header("🗂️ Dataset Explorer")

    dataset_name = st.selectbox(
        "Choose dataset",
        [
            "rooms",
            "schedule",
            "requests",
            "events"
        ]
    )

    df = data[dataset_name]

    st.write("### Dataset Dimensions")

    st.write(
        f"{len(df):,} rows × {len(df.columns)} columns"
    )

   

    st.write("### Sample Records")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.write("### Missing Values")

    missing = (
        df.isna()
        .sum()
        .rename("Missing Values")
        .to_frame()
    )

    st.dataframe(
        missing,
        use_container_width=True
    )
st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:15px 0;">
        <h4 style="color:#64748B;"> CSPACE</h4>
        <p>Campus Space Intelligence</p>
        <p style="font-size:13px; color:#64748B;">
            Find Your Space. Predict Demand. Optimize Campus.
        </p>
        <p style="font-size:12px; color:#64748B;">
            Datathon 2026 
        </p>
    </div>
    """,
    unsafe_allow_html=True
)