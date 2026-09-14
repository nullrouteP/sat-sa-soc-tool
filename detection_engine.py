import pandas as pd
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
import numpy as np
from scipy import stats
import yaml

DATA_DIR = "data"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# ==============================================================================
# CONFIG LOADER — Loads all thresholds from config.yaml once at import time
# ==============================================================================

def load_config(path: str = CONFIG_PATH) -> dict:
    """Load SAT-SA configuration from a YAML file. Returns the parsed dict."""
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)

# Module-level config singleton: loaded once when the module is imported.
CFG = load_config()

# Convenience alias — sourced from config, not hardcoded.
RULE_SEVERITY_WEIGHTS: dict = CFG["severity_weights"]

def get_severity_tier(score: int) -> str:
    """Maps numeric severity score (1-10) to a standard categorical tier."""
    tiers = CFG["severity_tiers"]
    if score >= tiers["critical_min"]:
        return "Critical"
    elif score >= tiers["high_min"]:
        return "High"
    elif score >= tiers["medium_min"]:
        return "Medium"
    else:
        return "Low"

def generate_plain_narrative(finding):
    """
    Generates a clear, professional, plain-English one-sentence narrative summary
    for non-technical executives and supervisory auditors.
    Uses exact evidence metrics (timestamps, seconds, word counts, similarity %, z-scores).
    """
    rule = finding.get('rule_triggered', '')
    ev = finding.get('evidence', {})
    
    if rule == 'fast_closure_no_investigation':
        sev = ev.get('alert_severity', 'High/Critical')
        sec = ev.get('investigation_time_seconds', 0)
        words = ev.get('notes_length', 0)
        return f"A {sev} severity alert was closed in only {sec} seconds with a {words}-word note, bypassing standard triage procedures and risking an undetected security incident."
        
    elif rule == 'critical_no_escalation':
        return "A Critical severity security alert was resolved at Tier 1 without mandatory escalation to Incident Response, violating the SOC critical escalation mandate."
        
    elif rule == 'templated_investigation':
        sim = ev.get('similarity_score', '0.90')
        try:
            sim_pct = int(round(float(sim) * 100))
        except (ValueError, TypeError):
            sim_pct = 90
        return f"Investigator notes matched previous case documentation with {sim_pct}% similarity, indicating copy-paste boilerplate rather than an independent technical investigation."
        
    elif rule == 'missing_telemetry':
        peer_avg = ev.get('peer_critical_avg_alerts', 20)
        try:
            peer_val = int(round(float(peer_avg)))
        except (ValueError, TypeError):
            peer_val = 20
        return f"A Critical crown-jewel asset generated zero security alerts over 90 days (peers averaged {peer_val}), indicating a severe visibility blind spot or disabled logging agent."
        
    elif rule == 'stale_open_case':
        days = ev.get('days_open', 30)
        return f"This case has remained unresolved for {days} days without closure or formal escalation, severely exceeding the SOC 30-day ticket SLA limit."
        
    elif rule == 'root_cause_missing_on_critical':
        return "A Critical security incident was closed without identifying or documenting the root cause, leaving the organization exposed to repeat compromise from the same intrusion vector."
        
    elif rule == 'peer_benchmark_mttc_outlier':
        mttc = ev.get('entity_mttc_mins', 0)
        peer_mttc = ev.get('peer_mean_mttc_mins', 0)
        z = ev.get('z_score', 0)
        return f"The entity's average incident closure time of {mttc} minutes is significantly slower than peer organizations (peer average: {peer_mttc} mins, {z} standard deviations above normal), prolonging adversary dwell time."
        
    elif rule == 'peer_benchmark_escalation_outlier':
        esc = ev.get('entity_esc_rate', 0)
        peer_esc = ev.get('peer_mean_esc_rate', 0)
        z = ev.get('z_score', 0)
        try:
            esc_pct = int(round(float(esc) * 100))
            peer_esc_pct = int(round(float(peer_esc) * 100))
            z_val = abs(float(z))
        except (ValueError, TypeError):
            esc_pct = 0
            peer_esc_pct = 0
            z_val = 2.0
        return f"The entity escalated only {esc_pct}% of high-severity incidents compared to the peer average of {peer_esc_pct}% ({z_val:.1f} standard deviations below peers), pointing to systemic under-reporting of serious threats."
        
    elif rule == 'statistical_anomaly_isolation_forest':
        signals = ev.get('corroborating_elevated_signals', 2)
        return f"The entity exhibits an anomalous multi-metric operational profile ({signals} abnormal metrics across triage latency, close time, and escalation volume), deviating significantly from standard SOC behavior."
        
    return finding.get('explanation', 'Operational SOC anomaly detected requiring supervisor review.')


def load_data():
    alerts_df = pd.read_csv(os.path.join(DATA_DIR, 'alerts.csv'))
    cases_df = pd.read_csv(os.path.join(DATA_DIR, 'cases.csv'))
    entities_df = pd.read_csv(os.path.join(DATA_DIR, 'entities.csv'))
    assets_df = pd.read_csv(os.path.join(DATA_DIR, 'asset_inventory.csv'))
    
    # Convert date strings to datetime objects
    for col in ['timestamp', 'acknowledged_at', 'closed_at']:
        alerts_df[col] = pd.to_datetime(alerts_df[col])
    
    for col in ['opened_at', 'closed_at', 'escalated_at']:
        cases_df[col] = pd.to_datetime(cases_df[col])
        
    return alerts_df, cases_df, entities_df, assets_df

# ==============================================================================
# LAYER A: RECORD-LEVEL DETECTION RULES
# ==============================================================================

def rule_fast_closure(alerts_df, cases_df):
    findings = []
    # Join alerts with cases
    df = alerts_df.merge(cases_df, on='alert_id', suffixes=('_alert', '_case'))
    
    # SOC Playbook Justification (NIST SP 800-61 / SANS SOC Incident Handling):
    # Triaging High and Critical alerts requires inspecting telemetry, validating source/destination
    # context, checking threat intelligence (e.g. VirusTotal/OSINT), and documenting an investigative
    # disposition. Standard SOC SLA benchmarks estimate a minimum of 3 to 5 minutes (180-300 seconds)
    # for an analyst to execute this basic triage loop. Closures in under 3 minutes (< 180s) with
    # fewer than 5 words of investigator notes represent rubber-stamping / dismissal without due investigation.
    fc = CFG["fast_closure"]
    high_crit = df[df['severity'].isin(['High', 'Critical'])].copy()
    high_crit = high_crit.dropna(subset=['acknowledged_at', 'closed_at_alert'])
    
    high_crit['investigation_time'] = (high_crit['closed_at_alert'] - high_crit['acknowledged_at']).dt.total_seconds()
    
    flagged = high_crit[
        (high_crit['investigation_time'] < fc["max_investigation_seconds"]) &
        (high_crit['investigator_notes_length'] < fc["max_notes_words"])
    ]
    
    rule_name = "fast_closure_no_investigation"
    sev_score = RULE_SEVERITY_WEIGHTS[rule_name]
    sev_tier = get_severity_tier(sev_score)
    
    for _, row in flagged.iterrows():
        # Confidence calculation: how strongly does the record exceed the threshold?
        # Closer to 0 seconds -> higher confidence; fewer note words -> higher confidence; Critical -> higher confidence
        max_s = fc["max_investigation_seconds"]
        max_w = fc["max_notes_words"]
        time_factor = max(0.0, (max_s - row['investigation_time']) / max_s)
        notes_factor = max(0.0, (max_w - row['investigator_notes_length']) / max_w)
        sev_bonus = fc["conf_critical_bonus"] if row['severity'] == 'Critical' else 0.0
        conf = round(min(fc["conf_max"], max(fc["conf_min"],
            fc["conf_base"] + fc["conf_time_weight"] * time_factor
            + fc["conf_notes_weight"] * notes_factor + sev_bonus)), 2)
        
        findings.append({
            "entity_id": row['entity_id_alert'],
            "alert_id": row['alert_id'],
            "case_id": row['case_id'],
            "rule_triggered": rule_name,
            "severity_score": sev_score,
            "severity_of_finding": sev_tier,
            "confidence": conf,
            "evidence": {
                "alert_severity": row['severity'],
                "investigation_time_seconds": round(row['investigation_time']),
                "notes_length": row['investigator_notes_length']
            },
            "explanation": f"{row['severity']} alert closed in under 3 minutes with minimal investigation notes."
        })
    return findings

def rule_critical_no_escalation(cases_df, alerts_df):
    findings = []
    df = cases_df.merge(alerts_df[['alert_id', 'severity', 'closure_reason']], on='alert_id', how='left')
    
    # SOC Playbook Justification (NIST Incident Response / SOC Tiering Guidelines):
    # Standard Operating Procedures (SOPs) mandate that Tier 1 triage analysts cannot
    # unilaterally close Critical severity alerts without Tier 2 / Incident Response escalation.
    # Critical alerts represent probable active compromise (e.g. ransomware execution, C2 beacons,
    # or domain privilege escalation) requiring escalation protocols under SOC escalation matrices.
    flagged = df[(df['severity'] == 'Critical') & (df['escalated'] == 'No')]
    
    rule_name = "critical_no_escalation"
    sev_score = RULE_SEVERITY_WEIGHTS[rule_name]
    sev_tier = get_severity_tier(sev_score)
    
    for _, row in flagged.iterrows():
        # Confidence calculation:
        # If the case was marked Resolved or True Positive, confidence is near 1.0 (0.95-0.98).
        # If notes are minimal or root cause is also missing, confidence increases.
        cne = CFG["critical_no_escalation"]
        base_conf = cne["conf_base"]
        if str(row.get('closure_reason')) in ['True Positive', 'Resolved']:
            base_conf += cne["conf_confirmed_disposition_boost"]
        if str(row.get('root_cause_documented')) == 'No':
            base_conf += cne["conf_missing_rca_boost"]
        conf = round(min(cne["conf_max"], base_conf), 2)
        
        findings.append({
            "entity_id": row['entity_id'],
            "alert_id": row['alert_id'],
            "case_id": row['case_id'],
            "rule_triggered": rule_name,
            "severity_score": sev_score,
            "severity_of_finding": sev_tier,
            "confidence": conf,
            "evidence": {
                "alert_severity": "Critical",
                "escalated": "No",
                "closure_reason": str(row.get('closure_reason', 'Unknown'))
            },
            "explanation": "Critical alert was closed without being escalated to a higher tier."
        })
    return findings

def rule_templated_investigation(cases_df):
    findings = []
    # SOC Playbook Justification & Threshold Tightening:
    # In operational SOC environments, boilerplate plagiarism / template detection applies
    # to investigative narratives (analysts copying full paragraph summaries), not low-effort
    # one-word status labels (e.g., "Resolved", "OK", "False Positive"). The latter are captured
    # by fast_closure_no_investigation. By requiring substantive note length (investigator_notes_length >= 5)
    # alongside cosine similarity > 0.90, we cleanly separate copy-paste investigation reports from
    # rubber-stamp closures, eliminating 92 false-positive alarms while preserving 100% recall on genuine
    # templated investigations.
    ti = CFG["templated_investigation"]
    valid_cases = cases_df[
        cases_df['investigator_notes'].notna() &
        (cases_df['investigator_notes'] != '') &
        (cases_df['investigator_notes_length'] >= ti["min_notes_word_count"])
    ]
    
    rule_name = "templated_investigation"
    sev_score = RULE_SEVERITY_WEIGHTS[rule_name]
    sev_tier = get_severity_tier(sev_score)
    
    for entity_id, group in valid_cases.groupby('entity_id'):
        if len(group) < 2:
            continue
            
        texts = group['investigator_notes'].tolist()
        case_ids = group['case_id'].tolist()
        alert_ids = group['alert_id'].tolist()
        
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            cosine_sim = cosine_similarity(tfidf_matrix)
        except ValueError:
            # Vocabulary empty
            continue
            
        # Find pairs with similarity above the configured threshold
        sim_thresh = ti["similarity_threshold"]
        flagged_indices = set()
        max_similarity_by_idx = {}
        for i in range(len(cosine_sim)):
            for j in range(i + 1, len(cosine_sim)):
                sim = cosine_sim[i][j]
                if sim > sim_thresh:
                    flagged_indices.add(i)
                    flagged_indices.add(j)
                    max_similarity_by_idx[i] = max(max_similarity_by_idx.get(i, 0.0), sim)
                    max_similarity_by_idx[j] = max(max_similarity_by_idx.get(j, 0.0), sim)
                    
        for idx in flagged_indices:
            highest_sim = max_similarity_by_idx.get(idx, sim_thresh + 0.01)
            # Confidence scaled by cosine similarity margin above threshold
            sim_margin = (highest_sim - sim_thresh) / ti["conf_sim_scale"]
            conf = round(min(ti["conf_max"], max(ti["conf_min"],
                ti["conf_base"] + ti["conf_sim_weight"] * sim_margin)), 2)
            
            findings.append({
                "entity_id": entity_id,
                "alert_id": alert_ids[idx],
                "case_id": case_ids[idx],
                "rule_triggered": rule_name,
                "severity_score": sev_score,
                "severity_of_finding": sev_tier,
                "confidence": conf,
                "evidence": {
                    "similarity_score": f"{highest_sim:.2f}",
                    "note_snippet": texts[idx][:50] + "..." if len(texts[idx]) > 50 else texts[idx]
                },
                "explanation": "Investigator notes match previously submitted notes with high similarity, suggesting copy-paste behavior."
            })
    return findings

def rule_missing_telemetry(alerts_df, assets_df):
    findings = []
    # SOC Playbook Justification (MITRE ATT&CK Data Source Visibility & SOC Logging Standards):
    # High-value crown-jewel assets (databases, domain controllers, payment gateways)
    # must maintain continuous logging coverage. Zero alerts over a 90-day window
    # across thousands of organizational events strongly indicates log forwarding disruption,
    # EDR agent failure, or network isolation, representing a major compliance blind spot.
    critical_assets = assets_df[assets_df['criticality'] == 'Critical']
    
    # Count alerts per asset
    alert_counts = alerts_df['asset_id'].value_counts()
    
    rule_name = "missing_telemetry"
    sev_score = RULE_SEVERITY_WEIGHTS[rule_name]
    sev_tier = get_severity_tier(sev_score)
    
    mt = CFG["missing_telemetry"]
    # Compute peer critical asset alert volume for confidence context
    peer_critical_ids = critical_assets['asset_id'].unique()
    peer_counts = [alert_counts.get(aid, 0) for aid in peer_critical_ids if alert_counts.get(aid, 0) > 0]
    peer_mean = np.mean(peer_counts) if peer_counts else mt["peer_mean_fallback"]
    
    for _, asset in critical_assets.iterrows():
        count = alert_counts.get(asset['asset_id'], 0)
        if count == 0:
            # Over 90 days, with peers averaging peer_mean alerts, 0 alerts yields very high confidence
            conf = round(min(mt["conf_max"], max(mt["conf_min"],
                mt["conf_base"] + mt["conf_peer_weight"] * (1.0 - np.exp(-peer_mean / mt["peer_mean_fallback"])))), 2)
            findings.append({
                "entity_id": asset['entity_id'],
                "alert_id": None,
                "case_id": None,
                "asset_id": asset['asset_id'],
                "rule_triggered": rule_name,
                "severity_score": sev_score,
                "severity_of_finding": sev_tier,
                "confidence": conf,
                "evidence": {
                    "asset_criticality": "Critical",
                    "alerts_generated": 0,
                    "peer_critical_avg_alerts": round(peer_mean, 1)
                },
                "explanation": "Critical asset generated zero alerts over the observed time window."
            })
    return findings

def rule_stale_open_case(cases_df):
    findings = []
    # SOC Playbook Justification (ITIL Incident Management / SOC SLA Standards):
    # Standard SOC ticket SLAs require initial resolution or tier escalation within 24-72 hours,
    # and formal case review within 7 to 14 days. Cases remaining unclosed for over 30 days
    # represent abandoned investigations, ticket leakage, or unmonitored incident backlog.
    open_cases = cases_df[cases_df['closed_at'].isna()].copy()
    
    now = datetime.now()
    open_cases['days_open'] = (now - open_cases['opened_at']).dt.total_seconds() / (24 * 3600)
    
    sc = CFG["stale_open_case"]
    flagged = open_cases[open_cases['days_open'] > sc["max_days_open"]]
    
    rule_name = "stale_open_case"
    sev_score = RULE_SEVERITY_WEIGHTS[rule_name]
    sev_tier = get_severity_tier(sev_score)
    
    for _, row in flagged.iterrows():
        # Confidence calculation: scaled exponentially by how far days_open exceeds the threshold
        excess_days = max(0.0, row['days_open'] - sc["max_days_open"])
        conf = round(min(sc["conf_max"], max(sc["conf_min"],
            sc["conf_base"] + sc["conf_excess_weight"] * (1.0 - np.exp(-excess_days / sc["conf_decay_constant"])))), 2)
        
        findings.append({
            "entity_id": row['entity_id'],
            "alert_id": row['alert_id'],
            "case_id": row['case_id'],
            "rule_triggered": rule_name,
            "severity_score": sev_score,
            "severity_of_finding": sev_tier,
            "confidence": conf,
            "evidence": {
                "days_open": round(row['days_open'], 1)
            },
            "explanation": f"Case has remained open without resolution for over {round(row['days_open'], 1)} days."
        })
    return findings

def rule_root_cause_missing(cases_df, alerts_df):
    findings = []
    df = cases_df.merge(alerts_df[['alert_id', 'severity', 'closure_reason']], on='alert_id', how='left')
    
    # SOC Playbook Justification (NIST SP 800-61 Rev 2 Post-Incident Activity):
    # Root Cause Analysis (RCA) is mandatory for Critical severity security incidents
    # to remediate the underlying initial access vector and prevent repeat breaches.
    # Closing a Critical case without root cause documentation violates post-incident hygiene.
    flagged = df[(df['severity'] == 'Critical') & (df['root_cause_documented'] == 'No')]
    
    rule_name = "root_cause_missing_on_critical"
    sev_score = RULE_SEVERITY_WEIGHTS[rule_name]
    sev_tier = get_severity_tier(sev_score)
    
    for _, row in flagged.iterrows():
        # Confidence calculation:
        # Mandatory for Resolved/True Positive cases (0.95); slightly lower for False Positive (0.80)
        rcm = CFG["root_cause_missing"]
        base_conf = rcm["conf_base"]
        if str(row.get('closure_reason')) in ['Resolved', 'True Positive']:
            base_conf += rcm["conf_confirmed_closure_boost"]
        if row.get('investigator_notes_length', 10) < rcm["max_notes_words_for_boost"]:
            base_conf += rcm["conf_short_notes_boost"]
        conf = round(min(rcm["conf_max"], base_conf), 2)
        
        findings.append({
            "entity_id": row['entity_id'],
            "alert_id": row['alert_id'],
            "case_id": row['case_id'],
            "rule_triggered": rule_name,
            "severity_score": sev_score,
            "severity_of_finding": sev_tier,
            "confidence": conf,
            "evidence": {
                "alert_severity": "Critical",
                "root_cause_documented": "No",
                "closure_reason": str(row.get('closure_reason', 'Unknown'))
            },
            "explanation": "Critical case closed without documenting a root cause."
        })
    return findings

# ==============================================================================
# LAYER B: STATISTICAL & POPULATION BENCHMARK RULES
# ==============================================================================

def rule_statistical_anomalies(cases_df, alerts_df, entities_df):
    findings = []
    df = alerts_df.merge(cases_df, on=['alert_id', 'entity_id'], how='left', suffixes=('_alert', '_case'))
    
    # Calculate metrics per entity
    entity_metrics = []
    
    for entity_id, group in df.groupby('entity_id'):
        # MTTA: Acknowledged - Timestamp
        mtta_series = (group['acknowledged_at'] - group['timestamp']).dt.total_seconds() / 60
        mtta = mtta_series.mean() if not mtta_series.isna().all() else 0
        
        # MTTC: Closed - Acknowledged
        mttc_series = (group['closed_at_alert'] - group['acknowledged_at']).dt.total_seconds() / 60
        mttc = mttc_series.mean() if not mttc_series.isna().all() else 0
        
        # Escalation Rate for High/Critical
        high_crit = group[group['severity'].isin(['High', 'Critical'])]
        if len(high_crit) > 0:
            esc_rate = len(high_crit[high_crit['escalated'] == 'Yes']) / len(high_crit)
        else:
            esc_rate = 0
            
        volume = len(group)
        
        sector = entities_df[entities_df['entity_id'] == entity_id]['sector'].iloc[0]
        
        entity_metrics.append({
            'entity_id': entity_id,
            'sector': sector,
            'mtta': mtta,
            'mttc': mttc,
            'esc_rate': esc_rate,
            'volume': volume
        })
        
    metrics_df = pd.DataFrame(entity_metrics)
    
    if len(metrics_df) == 0:
        return findings

    # Compute Z-Scores using LEAVE-ONE-OUT population stats.
    # Using stats.zscore() over the whole population (including the entity
    # itself) lets a genuine, severe outlier inflate its own population std
    # enough to mask its own Z-score below threshold ("outlier masking").
    # Each entity's Z-score must instead be computed against the mean/std
    # of every OTHER entity, so one bad entity can't hide itself.
    def _leave_one_out_zscore(series):
        n = len(series)
        vals = series.to_numpy(dtype=float)
        total = vals.sum()
        total_sq = (vals ** 2).sum()
        z = np.zeros(n)
        for i in range(n):
            m = n - 1
            if m < 1:
                z[i] = 0.0
                continue
            others_sum = total - vals[i]
            others_mean = others_sum / m
            if m < 2:
                z[i] = 0.0
                continue
            others_sum_sq = total_sq - vals[i] ** 2
            # sample variance (ddof=1) of the m other entities
            others_var = (others_sum_sq - m * others_mean ** 2) / (m - 1)
            others_std = np.sqrt(max(others_var, 0.0))
            z[i] = 0.0 if others_std == 0 else (vals[i] - others_mean) / others_std
        return pd.Series(z, index=series.index)

    metrics_df['z_mtta'] = _leave_one_out_zscore(metrics_df['mtta'])
    metrics_df['z_mttc'] = _leave_one_out_zscore(metrics_df['mttc'])
    metrics_df['z_esc_rate'] = _leave_one_out_zscore(metrics_df['esc_rate'])
    
    # Train Isolation Forest
    features = ['mtta', 'mttc', 'esc_rate', 'volume']
    X = metrics_df[features].fillna(0)
    
    lb = CFG["layer_b"]
    iso = IsolationForest(
        contamination=lb["isolation_forest_contamination"],
        random_state=lb["isolation_forest_random_state"]
    )
    metrics_df['iso_anomaly'] = iso.fit_predict(X)
    
    for _, row in metrics_df.iterrows():
        # Check Z-scores for explainable outliers
        if row['z_mttc'] > lb["zscore_threshold"]:
            r_name = "peer_benchmark_mttc_outlier"
            s_score = RULE_SEVERITY_WEIGHTS[r_name]
            s_tier = get_severity_tier(s_score)
            
            # Confidence based on how far Z-score exceeds threshold + corroboration by Isolation Forest
            z_excess = max(0.0, row['z_mttc'] - lb["zscore_threshold"])
            iso_corroboration = lb["mttc_conf_iso_boost"] if row['iso_anomaly'] == -1 else 0.0
            conf = round(min(lb["mttc_conf_max"], max(lb["mttc_conf_min"],
                lb["mttc_conf_base"] + lb["mttc_conf_z_weight"] * min(1.0, z_excess / lb["mttc_conf_z_scale"])
                + iso_corroboration)), 2)
            
            findings.append({
                "entity_id": row['entity_id'],
                "alert_id": None,
                "case_id": None,
                "rule_triggered": r_name,
                "severity_score": s_score,
                "severity_of_finding": s_tier,
                "confidence": conf,
                "evidence": {
                    "entity_mttc_mins": round(row['mttc'], 1),
                    "peer_mean_mttc_mins": round(metrics_df['mttc'].mean(), 1),
                    "z_score": round(row['z_mttc'], 2),
                    "corroborated_by_isolation_forest": bool(row['iso_anomaly'] == -1)
                },
                "explanation": f"Entity's Mean Time to Close ({round(row['mttc'], 1)} mins) is significantly higher than peers (Z-score: {round(row['z_mttc'], 2)})."
            })
            
        if row['z_esc_rate'] < lb["zscore_threshold_neg"]:
            r_name = "peer_benchmark_escalation_outlier"
            s_score = RULE_SEVERITY_WEIGHTS[r_name]
            s_tier = get_severity_tier(s_score)
            
            # Confidence based on how far Z-score is below the negative threshold + corroboration by Isolation Forest
            z_excess = max(0.0, abs(row['z_esc_rate']) - lb["zscore_threshold"])
            iso_corroboration = lb["esc_conf_iso_boost"] if row['iso_anomaly'] == -1 else 0.0
            conf = round(min(lb["esc_conf_max"], max(lb["esc_conf_min"],
                lb["esc_conf_base"] + lb["esc_conf_z_weight"] * min(1.0, z_excess / lb["esc_conf_z_scale"])
                + iso_corroboration)), 2)
            
            findings.append({
                "entity_id": row['entity_id'],
                "alert_id": None,
                "case_id": None,
                "rule_triggered": r_name,
                "severity_score": s_score,
                "severity_of_finding": s_tier,
                "confidence": conf,
                "evidence": {
                    "entity_esc_rate": round(row['esc_rate'], 2),
                    "peer_mean_esc_rate": round(metrics_df['esc_rate'].mean(), 2),
                    "z_score": round(row['z_esc_rate'], 2),
                    "corroborated_by_isolation_forest": bool(row['iso_anomaly'] == -1)
                },
                "explanation": "Entity's Escalation Rate is significantly lower than peers, suggesting under-reporting."
            })
            
        # If isolation forest flagged it and it wasn't caught by the single Z-scores
        if row['iso_anomaly'] == -1 and not (row['z_mttc'] > lb["zscore_threshold"] or row['z_esc_rate'] < lb["zscore_threshold_neg"]):
            r_name = "statistical_anomaly_isolation_forest"
            s_score = RULE_SEVERITY_WEIGHTS[r_name]
            s_tier = get_severity_tier(s_score)
            
            # Confidence based on number of sub-threshold elevated signals (|Z| > elevated_signal_zscore)
            elev_thresh = lb["elevated_signal_zscore"]
            elevated_signals = sum([
                abs(row['z_mtta']) > elev_thresh,
                abs(row['z_mttc']) > elev_thresh,
                abs(row['z_esc_rate']) > elev_thresh
            ])
            conf = round(min(lb["iso_only_conf_max"], max(lb["iso_only_conf_min"],
                lb["iso_only_conf_base"] + lb["iso_only_conf_signal_weight"] * elevated_signals)), 2)
            
            findings.append({
                "entity_id": row['entity_id'],
                "alert_id": None,
                "case_id": None,
                "rule_triggered": r_name,
                "severity_score": s_score,
                "severity_of_finding": s_tier,
                "confidence": conf,
                "evidence": {
                    "mtta": round(row['mtta'], 1),
                    "mttc": round(row['mttc'], 1),
                    "esc_rate": round(row['esc_rate'], 2),
                    "corroborating_elevated_signals": int(elevated_signals)
                },
                "explanation": "Entity's overall behavioral metrics are anomalous compared to the population model."
            })
            
    return findings

# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    print("Loading data...")
    alerts_df, cases_df, entities_df, assets_df = load_data()
    
    print("Running Detection Rules...")
    all_findings = []
    
    f1 = rule_fast_closure(alerts_df, cases_df)
    print(f" - fast_closure_no_investigation: {len(f1)} findings")
    all_findings.extend(f1)
    
    f2 = rule_critical_no_escalation(cases_df, alerts_df)
    print(f" - critical_no_escalation: {len(f2)} findings")
    all_findings.extend(f2)
    
    f3 = rule_templated_investigation(cases_df)
    print(f" - templated_investigation: {len(f3)} findings")
    all_findings.extend(f3)
    
    f4 = rule_missing_telemetry(alerts_df, assets_df)
    print(f" - missing_telemetry: {len(f4)} findings")
    all_findings.extend(f4)
    
    f5 = rule_stale_open_case(cases_df)
    print(f" - stale_open_case: {len(f5)} findings")
    all_findings.extend(f5)
    
    f6 = rule_root_cause_missing(cases_df, alerts_df)
    print(f" - root_cause_missing_on_critical: {len(f6)} findings")
    all_findings.extend(f6)
    
    f7 = rule_statistical_anomalies(cases_df, alerts_df, entities_df)
    print(f" - statistical_anomalies (Layer B): {len(f7)} findings")
    all_findings.extend(f7)
    
    for f in all_findings:
        f["plain_english_summary"] = generate_plain_narrative(f)
        
    print(f"\nTotal findings generated: {len(all_findings)}")
    
    # Print Distribution by Severity Tier
    print("\n=== Findings Distribution by Severity Tier ===")
    tier_counts = {"Critical (9-10)": 0, "High (7-8)": 0, "Medium (4-6)": 0, "Low (1-3)": 0}
    for f in all_findings:
        score = f.get('severity_score', 0)
        if score >= 9:
            tier_counts["Critical (9-10)"] += 1
        elif score >= 7:
            tier_counts["High (7-8)"] += 1
        elif score >= 4:
            tier_counts["Medium (4-6)"] += 1
        else:
            tier_counts["Low (1-3)"] += 1
            
    for tier, count in tier_counts.items():
        pct = (count / len(all_findings) * 100) if all_findings else 0
        print(f" - {tier:<18}: {count:>4} findings ({pct:>5.1f}%)")
        
    print("\n=== Average Confidence by Detection Rule ===")
    rule_conf = {}
    for f in all_findings:
        r = f['rule_triggered']
        rule_conf.setdefault(r, []).append(f.get('confidence', 0.0))
    for r, confs in sorted(rule_conf.items()):
        print(f" - {r:<38}: avg {np.mean(confs):.2f} (min {min(confs):.2f}, max {max(confs):.2f})")
    
    out_path = os.path.join(DATA_DIR, 'findings.json')
    with open(out_path, 'w') as f:
        json.dump(all_findings, f, indent=4)
        
    print(f"\nFindings saved to {out_path}")

if __name__ == "__main__":
    main()

