# theme.py


PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="DM Sans", color="#F8FAFC"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    # Increased the bottom margin (b=50) to make room for the legend
    margin=dict(l=10, r=10, t=50, b=50), 
    hovermode="x unified",
    # Moved legend to the bottom center (y=-0.2)
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
)


COLORS = {
    # FIX 1: Changed "primary" from Dark Navy to a bright, glowing Light Blue
    "primary":   "#60A5FA", 
    "teal":      "#17C3B2",
    "coral":     "#E63946",
    "amber":     "#F4A261",
    "green":     "#2A9D8F",
    "grey1":     "#CBD5E1",
    "grey2":     "#94A3B8",
    "grey3":     "#64748B",
    "grey4":     "#475569",
}

SEG_COLORS = {
    "Business":    "#60A5FA", # Updated to match the new primary blue
    "Leisure":     "#17C3B2",
    "Group":       "#F4A261",
    "OTA / Online":"#E63946",
}