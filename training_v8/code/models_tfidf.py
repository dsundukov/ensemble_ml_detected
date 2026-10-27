from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_tfidf_model(seed: int = 42) -> Pipeline:
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 max_features=300_000, sublinear_tf=True, lowercase=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 max_features=100_000, sublinear_tf=True, lowercase=True,
                                 token_pattern=r"(?u)\b\w+\b")),
    ])
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000,
                             solver="liblinear", random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])
