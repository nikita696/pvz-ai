import os

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_openwebui_backend_status_page(page):
    base_url = os.getenv("E2E_BASE_URL")
    if not base_url:
        pytest.skip("Set E2E_BASE_URL to run browser smoke tests.")

    page.goto(base_url.rstrip("/"), wait_until="domcontentloaded")
    page.wait_for_load_state("load")

    expect(page.get_by_text("pvz-ai OpenAI-compatible backend")).to_be_visible()
    expect(page.get_by_text("/v1/models")).to_be_visible()
