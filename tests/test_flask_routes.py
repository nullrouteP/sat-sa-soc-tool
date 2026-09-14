"""
tests/test_flask_routes.py
===========================
Flask route tests using the test_client().
Covers:
 - HTTP 200 for every expected valid route
 - HTTP 404 for invalid entity IDs on entity-specific routes
 - /api/sector_stats returns valid JSON with the expected schema
 - /export/<entity_id> returns valid CSV with required headers
 - /export/<entity_id>/pdf returns a valid PDF binary
 - Role-based view toggle (manager / analyst) on / and /raw_anomalies
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def client():
    """Create a Flask test client operating against the real data/ files."""
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def _first_entity_id() -> str:
    """Read the first entity_id from the real entities.csv."""
    import pandas as pd
    entities_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'entities.csv'))
    return entities_df.iloc[0]['entity_id']


INVALID_ENTITY_ID = '00000000-dead-beef-ffff-000000000000'


# ---------------------------------------------------------------------------
# Index / Dashboard tests
# ---------------------------------------------------------------------------

class TestIndexRoute:
    def test_index_returns_200(self, client):
        resp = client.get('/')
        assert resp.status_code == 200

    def test_index_manager_view(self, client):
        resp = client.get('/?view=manager')
        assert resp.status_code == 200

    def test_index_analyst_view(self, client):
        resp = client.get('/?view=analyst')
        assert resp.status_code == 200

    def test_index_invalid_view_defaults_gracefully(self, client):
        """Unknown view parameter must not crash the route."""
        resp = client.get('/?view=superadmin')
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Entity detail page
# ---------------------------------------------------------------------------

class TestEntityRoute:
    def test_valid_entity_returns_200(self, client):
        eid = _first_entity_id()
        resp = client.get(f'/entity/{eid}')
        assert resp.status_code == 200

    def test_invalid_entity_returns_404(self, client):
        resp = client.get(f'/entity/{INVALID_ENTITY_ID}')
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Finding detail page
# ---------------------------------------------------------------------------

class TestFindingRoute:
    def test_valid_finding_returns_200(self, client):
        eid = _first_entity_id()
        resp = client.get(f'/finding/{eid}/0')
        assert resp.status_code == 200

    def test_invalid_entity_on_finding_returns_404(self, client):
        resp = client.get(f'/finding/{INVALID_ENTITY_ID}/0')
        assert resp.status_code == 404

    def test_trace_chain_returns_200(self, client):
        eid = _first_entity_id()
        resp = client.get(f'/finding/{eid}/0/trace')
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------

class TestStaticRoutes:
    def test_raw_anomalies_returns_200(self, client):
        resp = client.get('/raw_anomalies')
        assert resp.status_code == 200

    def test_raw_anomalies_manager_view(self, client):
        resp = client.get('/raw_anomalies?view=manager')
        assert resp.status_code == 200

    def test_sector_benchmarks_returns_200(self, client):
        resp = client.get('/sector_benchmarks')
        assert resp.status_code == 200

    def test_diagnostics_returns_200(self, client):
        resp = client.get('/diagnostics')
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

class TestApiSectorStats:
    def test_returns_200_and_json(self, client):
        resp = client.get('/api/sector_stats')
        assert resp.status_code == 200
        assert 'application/json' in resp.content_type

    def test_response_is_valid_json_object(self, client):
        resp = client.get('/api/sector_stats')
        data = json.loads(resp.data)
        assert isinstance(data, (dict, list)), "Expected a JSON object or array"

    def test_response_has_sector_data(self, client):
        """Each sector in the response should have at least an entity/anomaly count."""
        resp = client.get('/api/sector_stats')
        data = json.loads(resp.data)
        # Accept dict-of-counts (sector -> int), dict-of-dicts, or list-of-objects
        if isinstance(data, dict):
            for sector, stats in data.items():
                assert isinstance(sector, str), f"Sector key must be a string, got {type(sector)}"
                assert isinstance(stats, (dict, int)), f"Sector value must be a dict or count, got {type(stats)}"
        elif isinstance(data, list):
            for item in data:
                assert isinstance(item, dict)


# ---------------------------------------------------------------------------
# Export routes
# ---------------------------------------------------------------------------

class TestExportRoutes:
    def test_csv_export_returns_200(self, client):
        eid = _first_entity_id()
        resp = client.get(f'/export/{eid}')
        assert resp.status_code == 200

    def test_csv_content_type(self, client):
        eid = _first_entity_id()
        resp = client.get(f'/export/{eid}')
        assert 'text/csv' in resp.content_type

    def test_csv_has_plain_english_summary_header(self, client):
        """The CSV export must contain the 'Plain English Summary' column."""
        eid = _first_entity_id()
        resp = client.get(f'/export/{eid}')
        content = resp.data.decode('utf-8')
        assert 'Plain English Summary' in content

    def test_csv_invalid_entity_returns_404(self, client):
        resp = client.get(f'/export/{INVALID_ENTITY_ID}')
        assert resp.status_code == 404

    def test_pdf_export_returns_200(self, client):
        eid = _first_entity_id()
        resp = client.get(f'/export/{eid}/pdf')
        assert resp.status_code == 200

    def test_pdf_content_type(self, client):
        eid = _first_entity_id()
        resp = client.get(f'/export/{eid}/pdf')
        assert resp.content_type == 'application/pdf'

    def test_pdf_has_valid_header(self, client):
        """The response body must start with the %%PDF- magic bytes."""
        eid = _first_entity_id()
        resp = client.get(f'/export/{eid}/pdf')
        assert resp.data.startswith(b'%PDF-'), "PDF response does not start with %PDF- magic bytes"

    def test_pdf_minimum_size(self, client):
        """A valid PDF with findings must be at least 5 KB."""
        eid = _first_entity_id()
        resp = client.get(f'/export/{eid}/pdf')
        assert len(resp.data) > 5 * 1024, f"PDF unexpectedly small: {len(resp.data)} bytes"

    def test_pdf_invalid_entity_returns_404(self, client):
        resp = client.get(f'/export/{INVALID_ENTITY_ID}/pdf')
        assert resp.status_code == 404
