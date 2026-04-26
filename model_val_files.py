import pandas as pd
import numpy as np
import engine as engine_1  # Points to engine.py

def create_val_file(property_name, actual_path, output_path, exceeds_limit):
    print(f"\n🚀 Generating validation for {property_name} directly from engine.py...")

    # 1. LOAD ACTUAL DATA
    try:
        actual = pd.read_csv(actual_path)
        actual["Dates"] = pd.to_datetime(actual["Dates"])
        actual = actual.rename(columns={
            "Occperc": "Actual_Occperc",
            "TotalOccupancy": "Actual_occ_rooms"
        })
    except FileNotFoundError:
        print(f"❌ Error: Could not find actuals file at {actual_path}")
        return

    # 2. ASK ENGINE.PY TO GENERATE THE FORECAST
    horizon_days = len(actual) 
    print(f"🧠 Asking engine to forecast {horizon_days} days for {property_name}...")
    
    forecast = engine_1.run_csv_forecast(property_name, horizon=horizon_days)
    
    if forecast.empty:
        print(f"❌ Error: engine.py returned an empty forecast for {property_name}.")
        return

    # Engine outputs 'Date', we need it to match 'Dates'
    forecast["Date"] = pd.to_datetime(forecast["Date"])
    forecast = forecast.rename(columns={"Date": "Dates"})

    # 🚀 THE FIX: Drop redundant columns from forecast so they don't clash with actuals
    cols_to_drop = [c for c in ["DayOfWeek", "Inventory"] if c in forecast.columns]
    if cols_to_drop:
        forecast = forecast.drop(columns=cols_to_drop)

    # 3. MERGE ACTUALS AND ENGINE FORECAST
    merged = pd.merge(actual, forecast, on="Dates", how="inner")
    
    if len(merged) == 0:
        print("❌ Error: Dates from engine forecast do not overlap with actual data dates!")
        return

    # 4. COMPUTE THE ERRORS
    merged["Difference_%"] = (merged["Predicted_Occperc"] - merged["Actual_Occperc"]).round(4)
    merged["Absolute_Error_%"] = merged["Difference_%"].abs().round(4)

    merged["Difference_Rooms"] = (merged["Predicted_Rooms"] - merged["Actual_occ_rooms"]).round(2)
    merged["Absolute_Error_Rooms"] = merged["Difference_Rooms"].abs().round(2)

    merged[f"Exceeds_{exceeds_limit}_Rooms"] = merged["Absolute_Error_Rooms"] > exceeds_limit

    # 5. REORDER COLUMNS
    final_df = merged[[
        "Dates", "DayOfWeek", "Inventory", 
        "Actual_Occperc", "Predicted_Occperc", "Difference_%", "Absolute_Error_%",
        "Actual_occ_rooms", "Predicted_Rooms", "Difference_Rooms", "Absolute_Error_Rooms",
        f"Exceeds_{exceeds_limit}_Rooms"
    ]]

    # 6. EXPORT TO DATA FOLDER
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Matched {len(final_df)} days.")
    print(f"📊 Mean Absolute Error (MAE): {final_df['Absolute_Error_%'].mean():.2f}%")
    print(f"🏨 Mean Room Error: {final_df['Absolute_Error_Rooms'].mean():.1f} rooms")
    print(f"📁 Saved validation file to: {output_path}")


if __name__ == "__main__":
    # --- Generate Velaris ---
    create_val_file(
        property_name="Velaris",
        actual_path=r"G:/visualization/prediction_data/actual_data2.csv",
        output_path=r"data/val_2.csv",
        exceeds_limit=10
    )

    # --- Generate Zenvyra ---
    create_val_file(
        property_name="Zenvyra",
        actual_path=r"G:/visualization/prediction_data/actual_data3.csv",
        output_path=r"data/val_3.csv",
        exceeds_limit=12
    )