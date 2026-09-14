import pandas as pd
import numpy as np
import random
from faker import Faker
import uuid
from datetime import datetime, timedelta
import os

fake = Faker()

# Configuration
NUM_ENTITIES = 10
ASSETS_PER_ENTITY = (20, 100) # Min, Max
TOTAL_ALERTS = 5000
DAYS_OF_DATA = 90
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Constants
SECTORS = ['Banking', 'Power', 'Telecom', 'Healthcare', 'Transport']
SIZE_TIERS = ['Small', 'Medium', 'Large']
ASSET_TYPES = ['Server', 'Endpoint', 'Network_Device', 'Database']
CRITICALITIES = ['Low', 'Medium', 'High', 'Critical']
SEVERITIES = ['Low', 'Medium', 'High', 'Critical']
ALERT_CATEGORIES = ['Malware', 'Unauthorized_Access', 'Data_Exfil', 'Policy_Violation', 'Anomaly']
SOURCE_SYSTEMS = ['EDR', 'Firewall', 'IDS', 'DLP']
CLOSURE_REASONS = ['True Positive', 'False Positive', 'Benign', 'Resolved']
ESCALATION_LEVELS = ['None', 'Tier 2', 'Tier 3', 'Management']

def generate_entities(num_entities):
    entities = []
    for _ in range(num_entities):
        entities.append({
            'entity_id': str(uuid.uuid4()),
            'entity_name': fake.company(),
            'sector': random.choice(SECTORS),
            'size_tier': random.choice(SIZE_TIERS)
        })
    return pd.DataFrame(entities)

def generate_assets(entities_df):
    assets = []
    for entity_id in entities_df['entity_id']:
        num_assets = random.randint(*ASSETS_PER_ENTITY)
        for _ in range(num_assets):
            assets.append({
                'asset_id': str(uuid.uuid4()),
                'entity_id': entity_id,
                'asset_type': random.choice(ASSET_TYPES),
                'criticality': random.choice(CRITICALITIES)
            })
    return pd.DataFrame(assets)

def generate_base_alerts_and_cases(entities_df, assets_df, num_alerts, days_of_data):
    alerts = []
    cases = []
    ground_truth = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_of_data)
    
    # We will exclude one high-criticality asset for the missing telemetry anomaly
    # Let's pick a specific entity and one of its Critical assets
    target_entity_mt = entities_df.iloc[0]['entity_id']
    target_asset_mt = assets_df[(assets_df['entity_id'] == target_entity_mt) & (assets_df['criticality'] == 'Critical')]['asset_id'].iloc[0]
    
    # Let's pick a specific entity for templated investigations
    target_entity_ti = entities_df.iloc[1]['entity_id']
    templated_notes = "Reviewed system logs and correlated events. No malicious activity found. Closing as False Positive."

    available_assets = assets_df[assets_df['asset_id'] != target_asset_mt]
    
    for _ in range(num_alerts):
        # Pick random asset
        asset_row = available_assets.sample(1).iloc[0]
        entity_id = asset_row['entity_id']
        asset_id = asset_row['asset_id']
        
        timestamp = fake.date_time_between(start_date=start_date, end_date=end_date)
        severity = random.choice(SEVERITIES)
        
        alert_id = str(uuid.uuid4())
        case_id = str(uuid.uuid4())
        
        # Decide if this will be an anomaly (approx 5-10% chance)
        is_anomaly = random.random() < 0.08
        anomaly_type = None
        
        # Default normal values
        status = 'Closed'
        ack_delay_mins = random.uniform(1, 60)
        acknowledged_at = timestamp + timedelta(minutes=ack_delay_mins)
        investigation_mins = random.uniform(15, 60 * 24 * 3) # 15 mins to 3 days
        closed_at = acknowledged_at + timedelta(minutes=investigation_mins)
        
        notes_text = fake.paragraph(nb_sentences=random.randint(3, 8))
        notes_len = len(notes_text.split())
        
        escalated = 'Yes' if severity == 'Critical' or (severity == 'High' and random.random() > 0.5) else 'No'
        escalation_level = random.choice(ESCALATION_LEVELS[1:]) if escalated == 'Yes' else 'None'
        escalated_at = acknowledged_at + timedelta(minutes=random.uniform(10, 60)) if escalated == 'Yes' else None
        
        root_cause_documented = 'Yes' if severity in ['High', 'Critical'] else random.choice(['Yes', 'No'])
        
        # Inject specific anomalies based on rules
        if is_anomaly:
            anomaly_category = random.choice([
                'fast_closure_no_investigation',
                'critical_no_escalation',
                'stale_open_case',
                'root_cause_missing_on_critical'
            ])
            
            if anomaly_category == 'fast_closure_no_investigation':
                severity = random.choice(['High', 'Critical'])
                closed_at = acknowledged_at + timedelta(seconds=random.randint(10, 110)) # under 2 mins
                notes_text = random.choice(["False Positive", "Resolved", "OK"])
                notes_len = len(notes_text.split())
                anomaly_type = anomaly_category
                
            elif anomaly_category == 'critical_no_escalation':
                severity = 'Critical'
                escalated = 'No'
                escalation_level = 'None'
                escalated_at = None
                anomaly_type = anomaly_category
                
            elif anomaly_category == 'stale_open_case':
                status = random.choice(['Open', 'Investigating'])
                timestamp = fake.date_time_between(start_date=start_date, end_date=end_date - timedelta(days=35))
                acknowledged_at = timestamp + timedelta(minutes=random.uniform(1, 60))
                closed_at = None
                anomaly_type = anomaly_category
                
            elif anomaly_category == 'root_cause_missing_on_critical':
                severity = 'Critical'
                root_cause_documented = 'No'
                anomaly_type = anomaly_category
        
        # Entity-specific Templated Investigation Anomaly
        if not is_anomaly and entity_id == target_entity_ti and random.random() < 0.2:
            notes_text = templated_notes
            notes_len = len(notes_text.split())
            anomaly_type = 'templated_investigation'
            is_anomaly = True

        # Append alert
        alerts.append({
            'alert_id': alert_id,
            'entity_id': entity_id,
            'asset_id': asset_id,
            'timestamp': timestamp.isoformat(),
            'severity': severity,
            'alert_category': random.choice(ALERT_CATEGORIES),
            'source_system': random.choice(SOURCE_SYSTEMS),
            'status': status,
            'acknowledged_at': acknowledged_at.isoformat() if acknowledged_at else None,
            'closed_at': closed_at.isoformat() if closed_at else None,
            'closure_reason': random.choice(CLOSURE_REASONS) if status == 'Closed' else None
        })
        
        # Append case
        cases.append({
            'case_id': case_id,
            'alert_id': alert_id,
            'entity_id': entity_id,
            'opened_at': acknowledged_at.isoformat() if acknowledged_at else None,
            'closed_at': closed_at.isoformat() if closed_at else None,
            'investigator_notes_length': notes_len,
            'investigator_notes': notes_text,
            'escalated': escalated,
            'escalated_at': escalated_at.isoformat() if escalated_at else None,
            'escalation_level': escalation_level,
            'root_cause_documented': root_cause_documented
        })
        
        # Append ground truth if anomaly
        if is_anomaly:
            ground_truth.append({
                'entity_id': entity_id,
                'alert_id': alert_id,
                'case_id': case_id,
                'is_injected_anomaly': 'Yes',
                'anomaly_type': anomaly_type
            })
            
    # Add ground truth for missing telemetry
    ground_truth.append({
        'entity_id': target_entity_mt,
        'alert_id': None,
        'case_id': None,
        'is_injected_anomaly': 'Yes',
        'anomaly_type': 'missing_telemetry'
    })

    return pd.DataFrame(alerts), pd.DataFrame(cases), pd.DataFrame(ground_truth)

def inject_statistical_outliers(alerts_df, cases_df, ground_truth_df, entities_df):
    """
    Deliberately skew aggregate case data for target entities to create genuine
    population outliers for Layer B statistical detection rules:
    - MTTC outlier entity: inflates closed_at timestamps so mean close time is >2.0 std above peers.
    - Escalation-rate outlier entity: suppresses escalated='Yes' on High cases so escalation rate is <-2.0 std below peers.
    Derives injection strength as a multiple of the natural population's variance.
    Preserves existing individual record anomalies to maintain 100% recall.
    """
    # Select distinct target entities (different from missing_telemetry [0] and templated_investigation [1])
    target_entity_mttc = entities_df.iloc[2]['entity_id']
    name_mttc = entities_df.iloc[2]['entity_name']
    target_entity_esc = entities_df.iloc[3]['entity_id']
    name_esc = entities_df.iloc[3]['entity_name']

    # Convert timestamps temporarily to compute natural population metrics
    ack_dt = pd.to_datetime(alerts_df['acknowledged_at'])
    closed_dt = pd.to_datetime(alerts_df['closed_at'])

    # Baseline MTTC per entity
    mttc_by_entity = {}
    for eid in entities_df['entity_id']:
        valid = (alerts_df['entity_id'] == eid) & closed_dt.notna() & ack_dt.notna()
        mttc_by_entity[eid] = (closed_dt[valid] - ack_dt[valid]).dt.total_seconds().mean() / 60

    # Baseline Escalation Rate per entity for High/Critical
    cases_merged = cases_df.merge(alerts_df[['alert_id', 'severity']], on='alert_id')
    esc_by_entity = {}
    for eid in entities_df['entity_id']:
        hc = cases_merged[(cases_merged['entity_id'] == eid) & cases_merged['severity'].isin(['High', 'Critical'])]
        esc_by_entity[eid] = (hc['escalated'] == 'Yes').sum() / len(hc) if len(hc) > 0 else 0.0

    mttc_values = list(mttc_by_entity.values())
    esc_values = list(esc_by_entity.values())

    mttc_std = float(np.std(mttc_values, ddof=1))
    esc_std = float(np.std(esc_values, ddof=1))
    peer_mean_esc = float(np.mean(esc_values))

    # Keep track of already injected record-level anomalies so we don't distort them
    injected_alert_ids = set(ground_truth_df['alert_id'].dropna().unique())

    # --- 1. MTTC Outlier Injection ---
    # Derive shift dynamically from natural variance (multiplier of 6.0 guarantees z > 2.0)
    mttc_multiplier = 6.0
    mttc_shift_mins = mttc_multiplier * mttc_std

    normal_closed_mask = (alerts_df['entity_id'] == target_entity_mttc) & closed_dt.notna() & (~alerts_df['alert_id'].isin(injected_alert_ids))
    total_closed_count = ((alerts_df['entity_id'] == target_entity_mttc) & closed_dt.notna()).sum()
    normal_closed_count = normal_closed_mask.sum()
    per_case_shift = mttc_shift_mins * (total_closed_count / normal_closed_count) if normal_closed_count > 0 else mttc_shift_mins

    # Inflate closed_at on alerts_df
    # (use .mask() instead of a partial .loc assignment — avoids the
    # datetime64 unit-mismatch TypeError newer pandas raises when an
    # added Timedelta produces a different resolution than the column)
    closed_dt = closed_dt.mask(normal_closed_mask, closed_dt + pd.Timedelta(minutes=per_case_shift))
    alerts_df['closed_at'] = closed_dt.apply(lambda x: x.isoformat() if pd.notna(x) else None)

    # Synchronize cases_df closed_at
    cases_closed_dt = pd.to_datetime(cases_df['closed_at'])
    cases_mask = (cases_df['entity_id'] == target_entity_mttc) & cases_closed_dt.notna() & (~cases_df['alert_id'].isin(injected_alert_ids))
    cases_closed_dt = cases_closed_dt.mask(cases_mask, cases_closed_dt + pd.Timedelta(minutes=per_case_shift))
    cases_df['closed_at'] = cases_closed_dt.apply(lambda x: x.isoformat() if pd.notna(x) else None)

    final_mttc_valid = (alerts_df['entity_id'] == target_entity_mttc) & closed_dt.notna() & ack_dt.notna()
    final_mttc = (closed_dt[final_mttc_valid] - ack_dt[final_mttc_valid]).dt.total_seconds().mean() / 60

    # --- 2. Escalation Rate Outlier Injection ---
    # Derive target escalation rate from natural variance (multiplier of 6.0 guarantees z < -2.0)
    esc_multiplier = 6.0
    target_esc_rate = max(0.0, peer_mean_esc - esc_multiplier * esc_std)

    target_hc = cases_merged[(cases_merged['entity_id'] == target_entity_esc) & cases_merged['severity'].isin(['High', 'Critical'])]
    current_esc_count = (target_hc['escalated'] == 'Yes').sum()
    desired_esc_count = int(np.floor(target_esc_rate * len(target_hc)))
    num_to_suppress = max(0, current_esc_count - desired_esc_count)

    # Suppress escalation on High cases that are not injected anomalies
    eligible_high = target_hc[(target_hc['severity'] == 'High') & (target_hc['escalated'] == 'Yes') & (~target_hc['alert_id'].isin(injected_alert_ids))]
    sample_size = min(num_to_suppress, len(eligible_high))
    suppressed_alert_ids = set(eligible_high.sample(n=sample_size, random_state=42)['alert_id'].tolist())

    suppress_mask = cases_df['alert_id'].isin(suppressed_alert_ids)
    cases_df.loc[suppress_mask, 'escalated'] = 'No'
    cases_df.loc[suppress_mask, 'escalation_level'] = 'None'
    cases_df.loc[suppress_mask, 'escalated_at'] = None

    # Recalculate final escalation rate
    updated_merged = cases_df.merge(alerts_df[['alert_id', 'severity']], on='alert_id')
    final_hc = updated_merged[(updated_merged['entity_id'] == target_entity_esc) & updated_merged['severity'].isin(['High', 'Critical'])]
    final_esc = (final_hc['escalated'] == 'Yes').sum() / len(final_hc) if len(final_hc) > 0 else 0.0

    # --- 3. Log to Ground Truth ---
    new_gt_records = [
        {
            'entity_id': target_entity_mttc,
            'alert_id': None,
            'case_id': None,
            'is_injected_anomaly': 'Yes',
            'anomaly_type': 'peer_benchmark_mttc_outlier'
        },
        {
            'entity_id': target_entity_esc,
            'alert_id': None,
            'case_id': None,
            'is_injected_anomaly': 'Yes',
            'anomaly_type': 'peer_benchmark_escalation_outlier'
        }
    ]
    ground_truth_df = pd.concat([ground_truth_df, pd.DataFrame(new_gt_records)], ignore_index=True)

    summary_info = {
        'mttc': {
            'entity_id': target_entity_mttc,
            'name': name_mttc,
            'baseline_mttc': mttc_by_entity[target_entity_mttc],
            'final_mttc': final_mttc,
            'shift_mins': mttc_shift_mins,
            'std': mttc_std,
            'multiplier': mttc_multiplier
        },
        'esc': {
            'entity_id': target_entity_esc,
            'name': name_esc,
            'baseline_esc': esc_by_entity[target_entity_esc],
            'final_esc': final_esc,
            'target_esc': target_esc_rate,
            'std': esc_std,
            'multiplier': esc_multiplier
        }
    }

    return alerts_df, cases_df, ground_truth_df, summary_info

def main():
    print("Generating entities...")
    entities_df = generate_entities(NUM_ENTITIES)
    entities_df.to_csv(os.path.join(DATA_DIR, 'entities.csv'), index=False)
    
    print("Generating assets...")
    assets_df = generate_assets(entities_df)
    assets_df.to_csv(os.path.join(DATA_DIR, 'asset_inventory.csv'), index=False)
    
    print("Generating alerts and cases (with injected anomalies)...")
    alerts_df, cases_df, ground_truth_df = generate_base_alerts_and_cases(entities_df, assets_df, TOTAL_ALERTS, DAYS_OF_DATA)
    
    print("Injecting entity-level statistical outliers (Layer B)...")
    alerts_df, cases_df, ground_truth_df, summary = inject_statistical_outliers(alerts_df, cases_df, ground_truth_df, entities_df)

    alerts_df.to_csv(os.path.join(DATA_DIR, 'alerts.csv'), index=False)
    cases_df.to_csv(os.path.join(DATA_DIR, 'cases.csv'), index=False)
    ground_truth_df.to_csv(os.path.join(DATA_DIR, 'ground_truth.csv'), index=False)
    
    print(f"Generation complete! Datasets saved to {DATA_DIR}/")
    print(f"Total Entities: {len(entities_df)}")
    print(f"Total Assets: {len(assets_df)}")
    print(f"Total Alerts/Cases: {len(alerts_df)}")
    print(f"Injected Anomalies: {len(ground_truth_df)}")

    # Summary of injected statistical outliers
    m = summary['mttc']
    e = summary['esc']
    print("\n=== Statistical Outlier Injections Summary ===")
    print(f"Entity {m['entity_id']} ({m['name']}): MTTC inflated from {m['baseline_mttc']:.1f} mins to {m['final_mttc']:.1f} mins (+{m['shift_mins']:.1f} mins, {m['multiplier']:.1f}x natural std of {m['std']:.1f} mins), target z-score >2.0")
    print(f"Entity {e['entity_id']} ({e['name']}): Escalation rate suppressed from {e['baseline_esc']:.1%} to {e['final_esc']:.1%} (target <={e['target_esc']:.1%}, {e['multiplier']:.1f}x natural std of {e['std']:.3f}), target z-score <-2.0\n")

if __name__ == "__main__":
    main()

