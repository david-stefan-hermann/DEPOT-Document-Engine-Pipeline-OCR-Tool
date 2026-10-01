import pytest

from depot import embeddings
from depot.embeddings import Embedder, cosine


class FakeOllama:
    """Stands in for ollama.Client: deterministic vectors, counts calls."""

    calls: list[list[str]] = []
    fail = False

    def __init__(self, host, timeout):
        self.host = host

    def embed(self, model, input, options, keep_alive):
        assert options == {"num_gpu": 0}  # the GPU belongs to the chat model
        if FakeOllama.fail:
            raise ConnectionError("ollama down")
        texts = [input] if isinstance(input, str) else list(input)
        FakeOllama.calls.append(texts)
        return {"embeddings": [[float(len(t)), float(t.count("a")), 1.0] for t in texts]}


@pytest.fixture(autouse=True)
def _fake_ollama(monkeypatch):
    FakeOllama.calls = []
    FakeOllama.fail = False
    monkeypatch.setattr(embeddings.ollama, "Client", FakeOllama)


def test_embed_returns_one_vector_per_text_in_order(tmp_path):
    e = Embedder("http://fake:11434", "fake-embed", tmp_path / "emb.sqlite3")
    vectors = e.embed(["banana", "kiwi", "banana"])
    assert vectors == [[6.0, 3.0, 1.0], [4.0, 0.0, 1.0], [6.0, 3.0, 1.0]]
    assert FakeOllama.calls == [["banana", "kiwi"]]  # duplicates sent once
    e.close()


def test_cached_vectors_are_not_requested_again_even_after_reopening(tmp_path):
    path = tmp_path / "emb.sqlite3"
    e1 = Embedder("http://fake:11434", "fake-embed", path)
    e1.embed(["Ordner: Gesundheit", "Ordner: Finanzen"])
    e1.close()

    e2 = Embedder("http://fake:11434", "fake-embed", path)
    vectors = e2.embed(["Ordner: Finanzen", "Ordner: Neu", "Ordner: Gesundheit"])

    assert FakeOllama.calls[-1] == ["Ordner: Neu"]  # only the unknown text
    assert vectors[0] == [16.0, 1.0, 1.0] and vectors[2] == [18.0, 0.0, 1.0]
    e2.close()


def test_cache_is_per_model(tmp_path):
    path = tmp_path / "emb.sqlite3"
    Embedder("http://fake:11434", "model-a", path).embed(["x"])
    Embedder("http://fake:11434", "model-b", path).embed(["x"])
    assert len(FakeOllama.calls) == 2


def test_without_cache_path_every_call_goes_to_ollama():
    e = Embedder("http://fake:11434", "fake-embed")
    e.embed(["x"])
    e.embed(["x"])
    assert len(FakeOllama.calls) == 2


def test_large_inputs_are_sent_in_batches(tmp_path):
    e = Embedder("http://fake:11434", "fake-embed", tmp_path / "emb.sqlite3")
    e.embed([f"text {i}" for i in range(70)])
    assert [len(c) for c in FakeOllama.calls] == [32, 32, 6]


def test_failure_propagates_and_preload_swallows_it(tmp_path):
    e = Embedder("http://fake:11434", "fake-embed", tmp_path / "emb.sqlite3")
    FakeOllama.fail = True
    with pytest.raises(ConnectionError):
        e.embed(["x"])
    e.preload()  # must not raise


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
