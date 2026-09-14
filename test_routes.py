from app import app
import pandas as pd

def run_tests():
    client = app.test_client()
    entities_df = pd.read_csv('data/entities.csv')
    sample_eid = entities_df.iloc[0]['entity_id']
    
    routes_to_test = [
        ('/', 200),
        ('/?view=manager', 200),
        ('/?view=analyst', 200),
        (f'/entity/{sample_eid}', 200),
        (f'/finding/{sample_eid}/0', 200),
        (f'/finding/{sample_eid}/0/trace', 200),
        ('/raw_anomalies', 200),
        ('/raw_anomalies?view=manager', 200),
        ('/sector_benchmarks', 200),
        ('/diagnostics', 200),
        ('/api/sector_stats', 200),
        (f'/export/{sample_eid}', 200),
        (f'/export/{sample_eid}/pdf', 200)
    ]
    
    print("Testing SAT-SA Flask Application Routes...")
    for route, expected_status in routes_to_test:
        response = client.get(route)
        assert response.status_code == expected_status, f"Route {route} failed with status {response.status_code}"
        
        # Extra checks for export routes
        if route.endswith('/pdf'):
            assert response.content_type == 'application/pdf', f"PDF route content type mismatch: {response.content_type}"
            assert response.data.startswith(b'%PDF-'), "PDF response does not start with %PDF- header"
            assert len(response.data) > 3000, f"PDF response is suspiciously small ({len(response.data)} bytes)"
            print(f" [PASS] {route} -> 200 OK (Valid PDF, {len(response.data)} bytes)")
        elif '/export/' in route:
            assert 'text/csv' in response.content_type, f"CSV route content type mismatch: {response.content_type}"
            csv_text = response.data.decode('utf-8')
            assert 'Plain English Summary' in csv_text, "CSV export missing 'Plain English Summary' header"
            print(f" [PASS] {route} -> 200 OK (Valid CSV with Plain English Summary)")
        else:
            print(f" [PASS] {route} -> {response.status_code} OK")

    print("\nALL ROUTES AND EXPORT FORMATS VALIDATED SUCCESSFULLY!")

if __name__ == '__main__':
    run_tests()
