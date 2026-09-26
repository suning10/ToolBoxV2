"""``RouteTemplatePrometheusMiddleware`` must survive FastAPI's nested ``include_router`` and stay low-cardinality."""

import pytest
from fastapi import (
    APIRouter,
    FastAPI,
)
from fastapi.testclient import TestClient
from starlette_prometheus.middleware import (
    REQUESTS,
    RESPONSES,
)

from app.core.metrics import (
    UNMATCHED_PATH,
    RouteTemplatePrometheusMiddleware,
)


@pytest.fixture
def client():
    inner = APIRouter()
    inner.add_api_route("/session/{session_id}/name", lambda session_id: {"id": session_id}, methods=["PATCH"])
    inner.add_api_route("/items/{item_id:int}", lambda item_id: {"id": item_id}, methods=["GET"])
    inner.add_api_route("/boom", lambda: 1 / 0, methods=["GET"])
    api = APIRouter()
    api.include_router(inner, prefix="/auth")  # nested include, like app/api/v1/api.py
    app = FastAPI()
    app.add_middleware(RouteTemplatePrometheusMiddleware)
    app.include_router(api, prefix="/api/v1")
    app.add_api_route("/flat", lambda: {"ok": True}, methods=["GET"])
    return TestClient(app, raise_server_exceptions=False)


def requests_total(method, template):
    return REQUESTS.labels(method=method, path_template=template)._value.get()


def test_request_through_nested_routers_does_not_crash(client):
    # regression: starlette-prometheus read ``route.path`` on FastAPI's ``_IncludedRouter`` -> 500 on every API route
    assert client.patch("/api/v1/auth/session/abc/name").status_code == 200


def test_label_is_the_full_template_including_include_router_prefixes(client):
    template = "/api/v1/auth/session/{session_id}/name"
    before = requests_total("PATCH", template)

    client.patch("/api/v1/auth/session/abc/name")
    client.patch("/api/v1/auth/session/def/name")

    assert requests_total("PATCH", template) == before + 2  # two different ids, ONE label


def test_typed_path_parameters_are_templated_too(client):
    template = "/api/v1/auth/items/{item_id:int}"
    before = requests_total("GET", template)

    client.get("/api/v1/auth/items/7")

    assert requests_total("GET", template) == before + 1


def test_flat_routes_are_labelled_by_their_path(client):
    before = requests_total("GET", "/flat")

    client.get("/flat")

    assert requests_total("GET", "/flat") == before + 1


def test_unknown_urls_share_a_single_bounded_label(client):
    before = requests_total("GET", UNMATCHED_PATH)

    for path in ("/nope", "/wp-admin/setup.php", "/api/v1/auth/../etc/passwd"):
        assert client.get(path).status_code == 404

    assert requests_total("GET", UNMATCHED_PATH) == before + 3


def test_response_status_is_recorded_against_the_template(client):
    template = "/api/v1/auth/boom"
    before = RESPONSES.labels(method="GET", path_template=template, status_code=500)._value.get()

    assert client.get("/api/v1/auth/boom").status_code == 500

    assert RESPONSES.labels(method="GET", path_template=template, status_code=500)._value.get() == before + 1
