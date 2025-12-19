from flask import Flask, render_template, request
import pandas as pd
import os
import json

app = Flask(__name__)

DATA_PATH = os.path.join("outputs", "college_ranking.csv")


@app.route("/")
def index():
    df_all = pd.read_csv(DATA_PATH)

    df_all["ROI"] = df_all["ROI"].round(2)
    df_all["predicted_income"] = df_all["predicted_income"].round(0)
    df_all["COSTT4_A"] = df_all["COSTT4_A"].round(0)

    states = sorted(df_all["STABBR"].dropna().unique())

    df = df_all.copy()

    state = request.args.get("state", "ALL")
    if state != "ALL":
        df = df[df["STABBR"] == state]

    search = request.args.get("search", "")
    if search:
        df = df[df["INSTNM"].str.contains(search, case=False, na=False)]

    sort_by = request.args.get("sort", "ROI")
    order = request.args.get("order", "desc")
    ascending = order == "asc"

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    top_n = int(request.args.get("top", 50))
    df = df.head(top_n)

    chart_data = {
        "x": df["COSTT4_A"].tolist(),
        "y": df["predicted_income"].tolist(),
        "labels": df["INSTNM"].tolist()
    }

    return render_template(
        "index.html",
        tables=df.to_dict(orient="records"),
        states=states,
        selected_state=state,
        search=search,
        top_n=top_n,
        chart_data=json.dumps(chart_data)
    )



if __name__ == "__main__":
    app.run(debug=True)
