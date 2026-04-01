from types import SimpleNamespace

import app.manual_loader as manual_loader


def clear_manual_loader_caches() -> None:
    manual_loader.load_manual.cache_clear()
    manual_loader.get_manual_metadata.cache_clear()


def test_manual_loader_returns_small_fallback_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        manual_loader,
        "settings",
        SimpleNamespace(
            manual_path=str(tmp_path / "missing_manual.txt"),
            simulate_large_manual=False,
            simulated_manual_repeat_count=80,
        ),
    )
    clear_manual_loader_caches()

    content = manual_loader.load_manual()

    assert content == manual_loader.BASE_MANUAL_TEXT


def test_manual_loader_expands_content_when_large_simulation_enabled(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        manual_loader,
        "settings",
        SimpleNamespace(
            manual_path=str(tmp_path / "missing_manual.txt"),
            simulate_large_manual=True,
            simulated_manual_repeat_count=3,
        ),
    )
    clear_manual_loader_caches()

    content = manual_loader.load_manual()

    assert content.count(manual_loader.BASE_MANUAL_TEXT) == 3
    assert len(content) > len(manual_loader.BASE_MANUAL_TEXT)


def test_manual_metadata_contains_fingerprint_and_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(
        manual_loader,
        "settings",
        SimpleNamespace(
            manual_path=str(tmp_path / "missing_manual.txt"),
            simulate_large_manual=True,
            simulated_manual_repeat_count=2,
        ),
    )
    clear_manual_loader_caches()

    metadata = manual_loader.get_manual_metadata()

    assert metadata["path"].endswith("missing_manual.txt")
    assert metadata["characters"] > 0
    assert len(metadata["sha256"]) == 64
    assert metadata["simulate_large_manual"] is True
