import numpy as np

from hypothesaes import embedding


def test_get_local_embeddings_return_stacked(monkeypatch):
    cached_embeddings = {
        "first": np.array([1.0, 2.0]),
        "second": np.array([3.0, 4.0]),
    }
    monkeypatch.setattr(
        embedding,
        "load_embedding_cache",
        lambda cache_name: cached_embeddings,
    )

    result = embedding.get_local_embeddings(
        ["second", "first"],
        cache_name="test",
        return_stacked=True,
    )

    np.testing.assert_array_equal(result, np.array([[3.0, 4.0], [1.0, 2.0]]))
