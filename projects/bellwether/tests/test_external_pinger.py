"""External scheduler configuration tests (offline)."""

from scripts import setup_external_pinger as pinger


def test_job_config_dispatches_workflow_every_15_minutes():
    config = pinger.job_config("test-token")

    assert config["url"].endswith("watcher-audit.yml/dispatches")
    assert config["requestMethod"] == 1
    assert config["schedule"]["minutes"] == [7, 22, 37, 52]
    assert config["extendedData"]["headers"]["Authorization"] == "Bearer test-token"
    assert '"trigger_source": "cron-job.org"' in config["extendedData"]["body"]


def test_api_request_keeps_tls_verification_enabled(monkeypatch):
    calls = []

    class Response:
        content = b'{"jobs": []}'

        def raise_for_status(self):
            pass

        def json(self):
            return {"jobs": []}

    def fake_request(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(pinger.requests, "request", fake_request)

    assert pinger.api_request("GET", "/jobs", "secret") == {"jobs": []}
    assert calls[0][1].get("verify", True) is True


def test_upsert_creates_job_when_missing(monkeypatch):
    calls = []

    def fake_request(method, path, api_key, payload=None):
        calls.append((method, path, api_key, payload))
        if (method, path) == ("GET", "/jobs"):
            return {"jobs": []}
        return {"jobId": 123}

    monkeypatch.setattr(pinger, "api_request", fake_request)

    assert pinger.upsert_job("cjo-key", "gh-token") == ("created", 123)
    assert calls[1][0:3] == ("PUT", "/jobs", "cjo-key")


def test_upsert_updates_job_with_matching_title(monkeypatch):
    calls = []

    def fake_request(method, path, api_key, payload=None):
        calls.append((method, path, api_key, payload))
        return {"jobs": [{"jobId": 456, "title": pinger.JOB_TITLE}]}

    monkeypatch.setattr(pinger, "api_request", fake_request)

    assert pinger.upsert_job("cjo-key", "gh-token") == ("updated", 456)
    assert calls[1][0:3] == ("PATCH", "/jobs/456", "cjo-key")
