"""Validate docker-compose.yml structure (no docker daemon required)."""

import os

import yaml

COMPOSE_PATH = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")


def _load_compose():
    with open(COMPOSE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_docker_compose_has_backend_and_frontend_services():
    compose = _load_compose()
    services = compose["services"]
    assert "backend" in services
    assert "frontend" in services


def test_docker_compose_backend_exposes_port_8000():
    compose = _load_compose()
    ports = compose["services"]["backend"]["ports"]
    assert any("8000" in str(p) for p in ports), ports


def test_docker_compose_frontend_exposes_port_5173():
    compose = _load_compose()
    ports = compose["services"]["frontend"]["ports"]
    assert any("5173" in str(p) for p in ports), ports


def test_docker_compose_frontend_depends_on_healthy_backend():
    compose = _load_compose()
    frontend = compose["services"]["frontend"]
    depends_on = frontend.get("depends_on", {})
    assert "backend" in depends_on
    assert depends_on["backend"].get("condition") == "service_healthy"


def test_docker_compose_defines_and_mounts_data_volume():
    compose = _load_compose()
    assert "marketplace-data" in compose.get("volumes", {})
    backend_volumes = compose["services"]["backend"].get("volumes", [])
    assert any("marketplace-data" in str(v) for v in backend_volumes)


def test_docker_compose_backend_has_healthcheck():
    compose = _load_compose()
    backend = compose["services"]["backend"]
    assert "healthcheck" in backend
    assert "/api/health" in str(backend["healthcheck"]["test"])
