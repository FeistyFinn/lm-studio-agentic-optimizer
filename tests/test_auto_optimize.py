from auto_optimize import find_max_loadable_ratio, optimize


def make_runner(max_ratio_by_context, tps=lambda ctx, r: r * 100):
    """Simulates an OOM boundary: a config loads iff its ratio is at or below
    the limit for its context (missing context => always OOM)."""
    calls = []

    def runner(ctx, ratio):
        calls.append((ctx, ratio))
        limit = max_ratio_by_context.get(ctx, -1)
        if ratio <= limit:
            return {
                "status": "Success",
                "context_length": ctx,
                "gpu_ratio": ratio,
                "tps": tps(ctx, ratio),
            }
        return {
            "status": "OOM/Load Fail",
            "context_length": ctx,
            "gpu_ratio": ratio,
            "tps": 0.0,
        }

    runner.calls = calls
    return runner


def quiet(_msg):
    pass


def test_full_offload_fast_path():
    runner = make_runner({2048: 1.0, 4096: 1.0, 8192: 1.0, 16384: 1.0})

    best, trials = optimize(runner, log=quiet)

    assert best["gpu_ratio"] == 1.0
    # Phase 1 needs exactly one trial when full offload fits
    phase1 = [c for c in runner.calls if c[0] == 2048]
    assert phase1 == [(2048, 1.0)]
    # Constant ratio => constant tps; ties break toward the largest context
    assert best["context_length"] == 16384
    assert {t["context_length"] for t in trials if t["status"] == "Success"} == {
        2048,
        4096,
        8192,
        16384,
    }


def test_binary_search_converges_to_boundary():
    runner = make_runner({2048: 0.7})

    best, trials = optimize(runner, max_context=2048, log=quiet)

    assert best["gpu_ratio"] == 0.7
    # 1.0 fast path + binary search over the remaining grid, not a linear scan
    assert len(runner.calls) <= 6


def test_nothing_loads():
    runner = make_runner({})

    best, trials = optimize(runner, log=quiet)

    assert best is None
    assert all(t["status"] != "Success" for t in trials)


def test_ratio_backoff_on_context_doubling():
    # 4096 requires stepping down one ratio notch; 8192 is impossible
    runner = make_runner({2048: 0.7, 4096: 0.6})

    best, trials = optimize(runner, max_context=16384, log=quiet)

    successes = {(t["context_length"], t["gpu_ratio"]) for t in trials
                 if t["status"] == "Success"}
    assert (4096, 0.6) in successes
    # best by tps: ratio 0.7 at 2048 beats 0.6 at 4096 with default tps fn
    assert best["context_length"] == 2048
    assert best["gpu_ratio"] == 0.7


def test_ties_break_toward_larger_context():
    runner = make_runner(
        {2048: 1.0, 4096: 1.0}, tps=lambda ctx, r: 50.0
    )

    best, _ = optimize(runner, max_context=4096, log=quiet)

    assert best["context_length"] == 4096


def test_skip_ratio_search_probes_once_and_doubles_context():
    runner = make_runner({2048: 1.0, 4096: 1.0, 8192: 1.0})

    best, trials = optimize(
        runner, max_context=8192, skip_ratio_search=True, log=quiet
    )

    # No binary search: one probe at start, then pure context doubling
    assert runner.calls == [(2048, 1.0), (4096, 1.0), (8192, 1.0)]
    assert best["context_length"] == 8192


def test_skip_ratio_search_no_backoff_on_oom():
    runner = make_runner({2048: 1.0})

    best, trials = optimize(
        runner, max_context=8192, skip_ratio_search=True, log=quiet
    )

    # 4096 OOMs; without ratio control there is nothing to back off to
    assert runner.calls == [(2048, 1.0), (4096, 1.0)]
    assert best["context_length"] == 2048


def test_skip_ratio_search_start_context_fails():
    runner = make_runner({})

    best, trials = optimize(runner, skip_ratio_search=True, log=quiet)

    assert best is None
    assert runner.calls == [(2048, 1.0)]


def test_find_max_loadable_ratio_returns_none_when_all_fail():
    def runner(ratio):
        return {"status": "OOM/Load Fail"}

    ratio, result = find_max_loadable_ratio(runner, [0.0, 0.5, 1.0])

    assert ratio is None
    assert result is None
