"""Tests for the client-side ("Live") Surface Explorer mode.

The Live mode embeds `surface_explorer_client.html` (an in-browser quadratic
OLS fit + Plotly surface) via st.components.v1.html, so slider/axis changes
recompute in the browser with no Streamlit rerun. These tests guard the
Python integration; the browser-side behaviour is covered by the Playwright
suite in ui_tests/.
"""
import os

from streamlit.testing.v1 import AppTest

APP_SCRIPT = "src/app.py"
BOOT_TIMEOUT = 60
LIVE_OPTION = "⚡ Live (client-side — instant sliders)"
CLIENT_HTML = os.path.join(
    os.path.dirname(__file__), "..", "src", "views", "surface_explorer_client.html"
)


def test_client_html_present_and_injectable():
    """The embedded explorer file must exist and keep the contract the
    Streamlit view depends on: the exact injection anchor line, the
    in-browser OLS fit, and Plotly."""
    assert os.path.exists(CLIENT_HTML), f"missing {CLIENT_HTML}"
    html = open(CLIENT_HTML, encoding="utf-8").read()
    # surface_explorer_view._render_client_side() string-replaces this line
    assert "var INJECTED = null;" in html
    assert "function fitOLS" in html
    assert "function buildDesign" in html
    assert "Plotly" in html


def test_live_mode_cold_boot_no_exception():
    """Switching the Surface Explorer to Live mode with no project loaded
    must render the embedded component without raising."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=BOOT_TIMEOUT)
    at.run()
    assert not at.exception

    at.radio(key="se_view_mode").set_value(LIVE_OPTION).run()
    assert not at.exception, f"Live mode raised on cold boot: {at.exception}"


def test_classic_mode_still_default():
    """The Classic (server-side) explorer must remain the default so the
    existing behaviour and the 10-tab app test are unaffected."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=BOOT_TIMEOUT)
    at.run()
    assert not at.exception
    assert at.radio(key="se_view_mode").value == "Classic (server-side)"


def test_session_injection_builds_blob(analyzed_demo):
    """_session_injection turns a loaded project into the data blob the
    client-side explorer consumes."""
    from views.surface_explorer_view import _session_injection
    fake_session = {
        "exp_df": analyzed_demo["exp_df"],
        "independent_vars": analyzed_demo["independent_vars"],
        "dependent_vars": analyzed_demo["dependent_vars"],
    }
    inj = _session_injection(fake_session)
    assert inj is not None
    assert set(analyzed_demo["independent_vars"]).issubset(set(inj["headers"]))
    assert inj["response"] in inj["headers"]
    assert len(inj["rows"]) >= 3
    assert all(len(row) == len(inj["headers"]) for row in inj["rows"])


def test_session_injection_none_without_project():
    """No loaded project -> no injection (the explorer falls back to upload)."""
    from views.surface_explorer_view import _session_injection
    assert _session_injection({}) is None


def test_live_mode_with_seeded_project_no_exception(analyzed_demo, trained_ols):
    """Live mode with a seeded project must inject the project data and
    render the embedded explorer without raising."""
    from test_app_apptest import _seed_analyzed_project
    at = AppTest.from_file(APP_SCRIPT, default_timeout=BOOT_TIMEOUT)
    _seed_analyzed_project(at, analyzed_demo, trained_ols)
    at.run()
    assert not at.exception
    at.radio(key="se_view_mode").set_value(LIVE_OPTION).run()
    assert not at.exception, f"Live mode raised with a seeded project: {at.exception}"
