from app.manual_loader import get_manual_metadata, load_manual


def test_manual_loader_returns_content():
    assert load_manual()


def test_manual_metadata_contains_fingerprint():
    metadata = get_manual_metadata()

    assert metadata["path"]
    assert metadata["characters"] > 0
    assert len(metadata["sha256"]) == 64
