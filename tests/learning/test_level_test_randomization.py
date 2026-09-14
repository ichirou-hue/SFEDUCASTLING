from backend.api_gateway.routes import learning


def _make_puzzles(per_bucket: int = 10) -> list[dict]:
    puzzles: list[dict] = []

    for bucket in learning.LEVEL_BUCKETS:
        lo = bucket["min_puzzle_rating"]
        for index in range(per_bucket):
            puzzles.append(
                {
                    "id": f"level-{bucket['level']}-{index}",
                    "rating": lo + index,
                    "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                }
            )

    return puzzles


def test_level_test_uses_random_sample_and_shuffles(monkeypatch):
    puzzles = _make_puzzles(per_bucket=10)
    monkeypatch.setattr(learning.state, "puzzle_base", {"puzzles": puzzles})

    sample_calls: list[tuple[list[str], int]] = []
    shuffle_calls: list[list[str]] = []

    def fake_sample(pool, k):
        sample_calls.append(([p["id"] for p in pool], k))
        # Специально выбираем последние элементы, а не первые пять.
        return list(pool[-k:])

    def fake_shuffle(items):
        shuffle_calls.append([p["id"] for p in items])
        items.reverse()

    monkeypatch.setattr(learning.random, "sample", fake_sample)
    monkeypatch.setattr(learning.random, "shuffle", fake_shuffle)

    result = learning._level_test_puzzles()

    assert len(sample_calls) == len(learning.LEVEL_BUCKETS)
    assert all(k == 5 for _, k in sample_calls)
    assert len(result) == 5 * len(learning.LEVEL_BUCKETS)
    assert len({p["id"] for p in result}) == len(result)
    assert len(shuffle_calls) == 1

    # Проверяем, что функция использовала результат random.sample,
    # а не старый pool[:5].
    for bucket in learning.LEVEL_BUCKETS:
        level = bucket["level"]
        selected_ids = {p["id"] for p in result if p["id"].startswith(f"level-{level}-")}
        assert selected_ids == {f"level-{level}-{i}" for i in range(5, 10)}


def test_level_test_handles_bucket_with_less_than_five_puzzles(monkeypatch):
    first_bucket = learning.LEVEL_BUCKETS[0]
    lo = first_bucket["min_puzzle_rating"]
    puzzles = [
        {
            "id": f"short-{i}",
            "rating": lo + i,
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        }
        for i in range(3)
    ]
    monkeypatch.setattr(learning.state, "puzzle_base", {"puzzles": puzzles})

    result = learning._level_test_puzzles()

    assert len(result) == 3
    assert {p["id"] for p in result} == {"short-0", "short-1", "short-2"}
