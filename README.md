# Allocation Planner (demo)

A Streamlit demo of a monthly warehouse allocation planner, built from the
"Set up & run → Plan → Warnings" wireframe.

* **Set up & run** – pick a month and scenario, set min/max share per warehouse,
  see pre-run checks, press **Run allocation**.
* **Plan** – total cost, units placed, space used per warehouse and storage type,
  share vs limits, cost chart, CSV export, approve.
* **Warnings** – space overflow, missing shipping costs, share limits that bind,
  products placed by business rules.

All data is synthetic (`demo_data.py`). The engine (`engine.py`) follows the
same steps as a production allocation pipeline and solves the linear program
with HiGHS (via SciPy) instead of Gurobi.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch and `app.py`.
3. Click **Deploy** and share the app URL.
