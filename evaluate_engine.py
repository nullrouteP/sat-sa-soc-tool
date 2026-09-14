import pandas as pd
import json
import os

DATA_DIR = "data"

def main():
    ground_truth_path = os.path.join(DATA_DIR, 'ground_truth.csv')
    findings_path = os.path.join(DATA_DIR, 'findings.json')
    
    if not os.path.exists(ground_truth_path) or not os.path.exists(findings_path):
        print("Missing data files. Make sure to run generate_dataset.py and detection_engine.py first.")
        return
        
    gt_df = pd.read_csv(ground_truth_path)
    with open(findings_path, 'r') as f:
        findings = json.load(f)
        
    # We will match by alert_id or case_id or (entity_id + missing_telemetry)
    # Create sets for easier comparison
    
    # GT tuples: (entity_id, alert_id, case_id, anomaly_type)
    gt_set = set()
    for _, row in gt_df.iterrows():
        # Handle nan values for alert/case ids in missing telemetry
        a_id = row['alert_id'] if pd.notna(row['alert_id']) else None
        c_id = row['case_id'] if pd.notna(row['case_id']) else None
        gt_set.add((row['entity_id'], a_id, c_id, row['anomaly_type']))
        
    # Findings tuples
    findings_set = set()
    for f in findings:
        findings_set.add((f['entity_id'], f.get('alert_id'), f.get('case_id'), f['rule_triggered']))
        
    true_positives = gt_set.intersection(findings_set)
    false_positives = findings_set - gt_set
    false_negatives = gt_set - findings_set
    
    precision = len(true_positives) / len(findings_set) if len(findings_set) > 0 else 0
    recall = len(true_positives) / len(gt_set) if len(gt_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("=== Evaluation Results ===")
    print(f"Total Injected Anomalies (Ground Truth): {len(gt_set)}")
    print(f"Total Findings Generated: {len(findings_set)}")
    print(f"True Positives (Successfully Caught): {len(true_positives)}")
    print(f"False Positives (False Alarms): {len(false_positives)}")
    print(f"False Negatives (Missed Anomalies): {len(false_negatives)}")
    print(f"---")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    
    if len(false_positives) > 0:
        print("\nNote on False Positives: Since we generated random data, some normal data might unintentionally meet the criteria for a rule. This is expected behavior in synthetic data generation without strict constraints.")

if __name__ == "__main__":
    main()
