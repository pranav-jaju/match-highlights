"""Train the embedding GBM on every labelled match and save it for `matchreel run`."""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from benchmark_v2 import load  # noqa: E402
from matchreel.learned import EmbeddingScorer  # noqa: E402

data = load()
m = EmbeddingScorer("gbm").fit([d["df"] for d in data], [d["emb"] for d in data], [d["y"] for d in data])
Path("models").mkdir(exist_ok=True)
joblib.dump(m, "models/embedding_gbm.joblib")
print(f"trained on {len(data)} matches -> models/embedding_gbm.joblib")
