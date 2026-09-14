import json
import pandas as pd
from pdf_report import generate_entity_pdf

def test_pdf():
    with open('data/findings.json') as f:
        findings = json.load(f)
    entities_df = pd.read_csv('data/entities.csv')
    alerts_df = pd.read_csv('data/alerts.csv')
    cases_df = pd.read_csv('data/cases.csv')

    for _, row in entities_df.iterrows():
        entity = row.to_dict()
        ef = [f for f in findings if f.get('entity_id') == entity['entity_id']]
        print(f"Generating PDF for {entity['entity_name']} ({len(ef)} findings)...")
        pdf_bytes = generate_entity_pdf(entity, ef, alerts_df, cases_df)
        assert pdf_bytes.startswith(b'%PDF-'), f"Invalid PDF header for {entity['entity_name']}"
        assert len(pdf_bytes) > 2000, f"PDF too small for {entity['entity_name']}"
        print(f" -> OK! Size: {len(pdf_bytes)} bytes")
    print("ALL ENTITY PDFS GENERATED SUCCESSFULLY!")

if __name__ == '__main__':
    test_pdf()
