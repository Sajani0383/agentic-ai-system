def styled_state_frame(state_frame):
    if state_frame.empty:
        return state_frame
    return state_frame.style.format({
        "Capacity": "{:.0f}",
        "Occupied": "{:.0f}",
        "Free": "{:.0f}",
        "Entries": "{:.0f}",
        "Exits": "{:.0f}",
        "Utilisation %": "{:.1f}",
    }).background_gradient(subset=["Utilisation %"], cmap="RdYlGn_r")

def styled_transition_frame(frame):
    if frame.empty:
        return frame
    return frame.style.format({
        "Before": "{:.0f}", "After": "{:.0f}", "Entries": "{:.0f}", "Exits": "{:.0f}", "Net Change": "{:+.0f}"
    }).background_gradient(subset=["Net Change"], cmap="RdYlGn")

def styled_cycle_frame(frame):
    if frame.empty:
        return frame
    return frame.style.format({"Reward": "{:+.2f}"}).background_gradient(subset=["Reward"], cmap="RdYlGn")
