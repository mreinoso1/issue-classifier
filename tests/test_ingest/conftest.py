import httpx
import pytest


@pytest.fixture
def github_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer test-token",
    }


@pytest.fixture
def mock_client():
    """Build an httpx client whose transport never leaves the process."""

    def _make(handler) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    return _make
