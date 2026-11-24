import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from she import webapp

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "sample_input.json").read_text(encoding="utf-8"))


def test_design_payload_is_json_safe_and_consistent():
    result = webapp.run_design(copy.deepcopy(SAMPLE))
    json.dumps(result, allow_nan=False)  # no NaN/Infinity leaks to the browser
    assert result["feasible"] and result["thermal"]["duty_w"] >= SAMPLE["target_duty_w"] * (1 - 1e-6)
    assert all(check["pass"] for check in result["checks"])
    assert result["geometry"]["number_of_baffles"] > 1


def test_rate_mode_rates_the_entered_geometry():
    result = webapp.run_rating(copy.deepcopy(SAMPLE))
    assert result["geometry"]["tube_count"] == SAMPLE["geometry"]["tube_count"]
    assert not result["feasible"]  # 240 tubes at 0.65 m are too small for this duty, which the page must show
    assert result["iterations"] is None


def test_mechanical_block_is_validated_and_used():
    default = webapp.run_design(copy.deepcopy(SAMPLE))
    data = copy.deepcopy(SAMPLE) | {"mechanical": {"design_pressure_pa": 2e6}}
    assert webapp.run_design(data)["geometry"]["tubesheet_thickness_m"] > default["geometry"]["tubesheet_thickness_m"]
    with pytest.raises(ValueError, match="mechanical"):
        webapp.run_design(copy.deepcopy(SAMPLE) | {"mechanical": {"nope": 1}})
    with pytest.raises(ValueError, match="design_pressure_pa"):
        webapp.run_design(copy.deepcopy(SAMPLE) | {"mechanical": {"design_pressure_pa": -1}})


def test_report_is_an_xlsx_file():
    assert webapp.build_report(copy.deepcopy(SAMPLE))[:2] == b"PK"


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), webapp.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, body, headers=None):
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_http_endpoints(server):
    with urllib.request.urlopen(server + "/") as page:
        assert b"Shell-and-Tube Designer" in page.read()
    with urllib.request.urlopen(server + "/api/sample") as sample:
        assert json.loads(sample.read())["target_duty_w"] == SAMPLE["target_duty_w"]
    status, body = _post(server + "/api/design", json.dumps(SAMPLE).encode())
    assert status == 200 and json.loads(body)["feasible"]


def test_http_errors_are_clear_and_cross_origin_is_refused(server):
    status, body = _post(server + "/api/design", b"{bad")
    assert status == 400 and "invalid JSON" in json.loads(body)["error"]
    status, body = _post(server + "/api/design", json.dumps(SAMPLE | {"tube_lenght_m": 5}).encode())
    assert status == 400 and "unknown field" in json.loads(body)["error"]
    status, body = _post(server + "/api/design", json.dumps(SAMPLE).encode(), {"Origin": "http://evil.example"})
    assert status == 403
    status, _ = _post(server + "/api/nothing", b"{}")
    assert status == 404
