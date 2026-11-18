# Optional QuantConnect / LEAN replay

The dashboard's own backtest is a transparent research approximation (close-to-close
execution, a flat per-dollar cost). LEAN adds fills, brokerage models and order rules.

1. Install the LEAN CLI (`pip install lean`) and Docker, then `lean login` (QuantConnect account).
2. `lean project-create dashboard_replay` and copy `main.py` into it.
3. Export `weights_history.csv` from **Data & exports** and place it in the project folder.
4. `lean backtest dashboard_replay --parameter strategy min_vol`.

Unverified here: LEAN was not installed in the build environment, and historical TSX
security and corporate-action coverage depends on your data subscription. Confirm
symbol resolution, currency (CAD) and dividend handling before comparing results with
the dashboard.
