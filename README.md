# Home Energy Consumption Forecast

A Streamlit dashboard for the UCI Individual Household Electric Power
Consumption dataset. It aggregates minute-level `Global_active_power` readings
into 15-minute mean power (kW), uses the previous 24 hours as the LSTM input,
and forecasts the next four 15-minute intervals.

## Run locally

1. Use Python 3.10–3.12 and install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

3. Upload `household_power_consumption.txt` in the sidebar. The app also checks
   the standard UCI download location under your Downloads folder automatically.
4. Select **Train / refresh LSTM** to train on up to the most recent 180 days.
   The last 20% of that period is held out for validation.

The source file is not copied into this project. It can be downloaded from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

## Dashboard metrics

- Current 15-minute mean active power
- Today's measured energy so far and peak interval in the dataset
- LSTM estimates at 15, 30, 45, and 60 minutes ahead
- Recent power history and daily energy charts
- Configurable high-consumption warning and energy-saving suggestions
- Sidebar page navigation for Overview, Forecast, Usage analytics, and About the model
- A single 15-minute forecast metric on the Overview page; forecast horizons are shown on the Forecast page
- Distinct color-coded metric cards with subtle entrance and hover animations; motion is reduced for users who prefer reduced motion

Missing active-power values are treated as missing. Only short gaps of up to
one hour are interpolated; long outages remain excluded from training windows
and prevent forecasts when they occur in the latest 24 hours. Daily energy is
estimated as the sum of 15-minute mean power multiplied by 0.25 hours.

The dashboard uses a teal visual theme with distinct mint, blue, amber, and
violet metric cards, subtle entrance/hover animations, and reduced-motion
support. The main dashboard and sidebar scroll independently when their content
exceeds the available viewport. Forecast and analytics charts use a readable
height rather than being compressed to fit. Page navigation, data upload, and
warning settings are grouped in the independently scrollable sidebar. The app
displays historical dataset readings, not live smart-meter data.

## Tests

```powershell
python -m unittest discover -s tests
```
