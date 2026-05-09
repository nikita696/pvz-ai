import os

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_gradio_chat_smoke(page):
    base_url = os.getenv("E2E_BASE_URL")
    if not base_url:
        pytest.skip("Set E2E_BASE_URL to run browser smoke tests.")

    page.goto(f"{base_url.rstrip('/')}/chat", wait_until="domcontentloaded")
    page.wait_for_load_state("load")
    textbox = page.get_by_role("textbox").last
    textbox.fill("Привет")
    textbox.press("Enter")

    expect(page.get_by_text("Echo: Привет")).to_be_visible(timeout=30000)
