import time

from factory.utils import split_sentences, tfidf_vectors


def test_split_sentences_perf():
    text = "他推开了门。里面一片漆黑！你敢进去吗？" * 2000
    start = time.perf_counter()
    out = split_sentences(text)
    elapsed = time.perf_counter() - start
    assert len(out) > 1000
    assert elapsed < 2.0, "split_sentences too slow: %.3fs" % elapsed


def test_tfidf_perf():
    docs = ["测试文本 %d 包含 监控 尸体 真相 反转" % i for i in range(2000)]
    start = time.perf_counter()
    vectors = tfidf_vectors(docs)
    elapsed = time.perf_counter() - start
    assert len(vectors) == 2000
    assert elapsed < 2.0, "tfidf_vectors too slow: %.3fs" % elapsed
