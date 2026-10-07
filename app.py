"""Streamlit demo: upload a match, see the excitement curve, download the reel.

Run with:  streamlit run app.py
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from matchreel import features
from matchreel.reel import render, select
from matchreel.scoring import SCORERS, FusionModel

st.set_page_config(page_title="matchreel", layout="wide")
st.title("matchreel: automatic match highlights")

upload = st.file_uploader("Match video", type=["mp4", "mkv", "mov", "webm"])
col1, col2, col3 = st.columns(3)
model_path = Path("models/fusion.joblib")
options = ["fusion", "events", "loudness"] if model_path.exists() else ["events", "loudness"]
scorer = col1.selectbox("Scorer", options)
budget = col2.slider("Reel length (s)", 60, 900, 300, step=30)
pre = col3.slider("Seconds before each peak", 4, 20, 12)

if upload and st.button("Find highlights"):
    tmp = Path(tempfile.mkdtemp())
    media = tmp / upload.name
    media.write_bytes(upload.getbuffer())
    with st.spinner("Analysing audio (sound events and commentary take a few minutes)..."):
        df = features.build(str(media), str(tmp / "cache"), use_events=scorer != "loudness",
                            use_commentary=scorer == "fusion", verbose=False)
        t = df["time"].to_numpy()
        score = FusionModel.load(str(model_path)).score(df) if scorer == "fusion" else SCORERS[scorer](df)
        clips = select(t, score, budget, pre_s=pre)

    chart = pd.DataFrame({"minute": t / 60, "excitement": score}).set_index("minute")
    st.line_chart(chart)
    st.dataframe(pd.DataFrame([{"start (min)": round(c.start / 60, 2), "end (min)": round(c.end / 60, 2),
                                "score": round(c.score, 3)} for c in clips]))
    with st.spinner("Rendering reel..."):
        out = tmp / "highlights.mp4"
        render(str(media), clips, str(out))
    st.video(str(out))
    st.download_button("Download reel", out.read_bytes(), file_name="highlights.mp4")
