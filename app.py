from flask import Flask, render_template, jsonify, abort, Response, request
import json
import pandas as pd
import os
import io
import csv
from detection_engine import generate_plain_narrative
from pdf_report import generate_entity_pdf

app = Flask(__name__)
DATA_DIR = "data"

def load_data():
    try:
        with open(os.path.join(DATA_DIR, 'findings.json'), 'r') as f:
            findings = json.load(f)
    except FileNotFoundError:
        findings = []

    # Ensure all findings have plain_english_summary populated
    for f in findings:
        if not f.get('plain_english_summary'):
            f['plain_english_summary'] = generate_plain_narrative(f)

    try:
        entities_df = pd.read_csv(os.path.join(DATA_DIR, 'entities.csv'))
    except FileNotFoundError:
        entities_df = pd.DataFrame(columns=['entity_id', 'entity_name', 'sector', 'size_tier'])
        
    try:
        alerts_df = pd.read_csv(os.path.join(DATA_DIR, 'alerts.csv'))
    except FileNotFoundError:
        alerts_df = pd.DataFrame(columns=['alert_id', 'entity_id', 'timestamp', 'severity'])

    try:
        cases_df = pd.read_csv(os.path.join(DATA_DIR, 'cases.csv'))
    except FileNotFoundError:
        cases_df = pd.DataFrame(columns=['case_id', 'alert_id', 'entity_id', 'opened_at', 'closed_at'])

    return findings, entities_df, alerts_df, cases_df

@app.route('/')
def index():
    findings, entities_df, alerts_df, _ = load_data()
    view_mode = request.args.get('view', 'manager').lower()
    if view_mode not in ['manager', 'analyst']:
        view_mode = 'manager'
    
    # Calculate finding statistics per entity
    finding_counts = {}
    finding_scores = {}
    critical_counts = {}
    high_counts = {}
    medium_counts = {}
    entity_primary_rule = {}
    
    for f in findings:
        eid = f.get('entity_id')
        if eid:
            finding_counts[eid] = finding_counts.get(eid, 0) + 1
            score = f.get('severity_score', 5)
            finding_scores[eid] = finding_scores.get(eid, 0) + score
            if score >= 9:
                critical_counts[eid] = critical_counts.get(eid, 0) + 1
            elif score >= 7:
                high_counts[eid] = high_counts.get(eid, 0) + 1
            else:
                medium_counts[eid] = medium_counts.get(eid, 0) + 1
            
            current_top = entity_primary_rule.get(eid)
            if not current_top or score > current_top[1]:
                entity_primary_rule[eid] = (f.get('rule_triggered', 'Anomaly'), score)
            
    # Build ranked entities list
    ranked_entities = []
    for _, row in entities_df.iterrows():
        eid = row['entity_id']
        count = finding_counts.get(eid, 0)
        tot_score = finding_scores.get(eid, 0)
        crit_c = critical_counts.get(eid, 0)
        high_c = high_counts.get(eid, 0)
        med_c = medium_counts.get(eid, 0)
        primary = entity_primary_rule.get(eid, ('None', 0))[0]
        
        if crit_c >= 10 or tot_score >= 300:
            status = "Immediate Action Required"
            status_class = "badge-critical"
        elif high_c >= 15 or tot_score >= 150:
            status = "Elevated Risk"
            status_class = "badge-high"
        else:
            status = "Routine Oversight"
            status_class = "badge-medium"
            
        ranked_entities.append({
            'entity_id': eid,
            'name': row['entity_name'],
            'sector': row['sector'],
            'size_tier': row.get('size_tier', 'Medium'),
            'findings_count': count,
            'risk_score': tot_score,
            'critical_count': crit_c,
            'high_count': high_c,
            'medium_count': med_c,
            'primary_violation': primary.replace('_', ' ').title(),
            'status': status,
            'status_class': status_class
        })
        
    # In manager view, rank by total risk_score; in analyst view, rank by findings_count
    if view_mode == 'manager':
        ranked_entities.sort(key=lambda x: x['risk_score'], reverse=True)
    else:
        ranked_entities.sort(key=lambda x: x['findings_count'], reverse=True)
        
    # Enrich findings with entity name and index for direct linking
    entity_name_map = dict(zip(entities_df['entity_id'], entities_df['entity_name']))
    entity_idx_counter = {}
    enriched_findings = []
    for f in findings:
        eid = f.get('entity_id')
        idx = entity_idx_counter.get(eid, 0)
        entity_idx_counter[eid] = idx + 1
        item = dict(f)
        item['entity_name'] = entity_name_map.get(eid, 'Unknown Entity')
        item['finding_idx'] = idx
        enriched_findings.append(item)
        
    # Sort by severity_score descending, then confidence descending
    sorted_findings = sorted(enriched_findings, key=lambda x: (x.get('severity_score', 0), x.get('confidence', 0.0)), reverse=True)
    top_manager_findings = sorted_findings[:12]
    
    total_findings = len(findings)
    total_entities = len(entities_df)
    total_alerts = len(alerts_df)
    
    critical_findings_count = sum(1 for f in findings if f.get('severity_score', 0) >= 9)
    high_findings_count = sum(1 for f in findings if 7 <= f.get('severity_score', 0) <= 8)
    avg_confidence = (sum(f.get('confidence', 0.0) for f in findings) / total_findings * 100) if total_findings else 0.0
    
    return render_template('index.html', 
                           view_mode=view_mode,
                           entities=ranked_entities, 
                           top_findings=top_manager_findings,
                           analyst_findings=sorted_findings[:50],
                           total_findings=total_findings,
                           total_entities=total_entities,
                           total_alerts=total_alerts,
                           critical_count=critical_findings_count,
                           high_count=high_findings_count,
                           avg_confidence=round(avg_confidence, 1))

@app.route('/entity/<entity_id>')
def entity_detail(entity_id):
    findings, entities_df, alerts_df, cases_df = load_data()
    
    entity_row = entities_df[entities_df['entity_id'] == entity_id]
    if len(entity_row) == 0:
        abort(404)
        
    entity = entity_row.iloc[0].to_dict()
    entity_findings = [f for f in findings if f.get('entity_id') == entity_id]
    
    # 1. Prepare data for daily trend chart (backward compatibility)
    entity_alerts = alerts_df[alerts_df['entity_id'] == entity_id].copy()
    daily_trend_data = {}
    if not entity_alerts.empty:
        entity_alerts['timestamp_dt'] = pd.to_datetime(entity_alerts['timestamp'], format='ISO8601')
        entity_alerts['date'] = entity_alerts['timestamp_dt'].dt.date
        daily_counts = entity_alerts.groupby('date').size()
        daily_trend_data = {str(k): int(v) for k, v in daily_counts.items()}
        
    # 2. Time-Series Trend View: weekly bucketing for MTTC, Escalation Rate, and Finding Count
    entity_cases = cases_df[cases_df['entity_id'] == entity_id].copy()
    
    weekly_labels = []
    weekly_mttc = []
    weekly_esc_rate = []
    weekly_finding_counts = []
    weekly_alert_counts = []
    
    if not entity_alerts.empty:
        entity_alerts['week_start'] = entity_alerts['timestamp_dt'].dt.to_period('W').apply(lambda r: r.start_time.strftime('%Y-%m-%d'))
        all_weeks = sorted(entity_alerts['week_start'].dropna().unique())
        
        # Alert to week mapping
        alert_to_week = dict(zip(entity_alerts['alert_id'], entity_alerts['week_start']))
        
        # Calculate MTTC per alert
        entity_alerts['ack_dt'] = pd.to_datetime(entity_alerts['acknowledged_at'], format='ISO8601')
        entity_alerts['closed_dt'] = pd.to_datetime(entity_alerts['closed_at'], format='ISO8601')
        entity_alerts['mttc_mins'] = (entity_alerts['closed_dt'] - entity_alerts['ack_dt']).dt.total_seconds() / 60
        
        # Merge cases with alert week and severity
        cases_with_week = entity_cases.merge(entity_alerts[['alert_id', 'week_start', 'severity', 'mttc_mins']], on='alert_id', how='left')
        
        # Count findings per week using alert_id link
        findings_per_week = {}
        for f in entity_findings:
            w = alert_to_week.get(f.get('alert_id'))
            if w:
                findings_per_week[w] = findings_per_week.get(w, 0) + 1
                
        for w in all_weeks:
            w_cases = cases_with_week[cases_with_week['week_start'] == w]
            w_alerts = entity_alerts[entity_alerts['week_start'] == w]
            
            # MTTC: mean in minutes across alerts closed in that week
            valid_mttc = w_alerts['mttc_mins'].dropna()
            mttc_val = round(float(valid_mttc.mean()), 1) if len(valid_mttc) > 0 else 0.0
            
            # Escalation rate: % of High/Critical cases escalated
            hc_cases = w_cases[w_cases['severity'].isin(['High', 'Critical'])]
            if len(hc_cases) > 0:
                esc_rate_val = round(float((hc_cases['escalated'] == 'Yes').sum() / len(hc_cases) * 100), 1)
            else:
                esc_rate_val = 0.0
                
            f_count = int(findings_per_week.get(w, 0))
            a_count = int(len(w_alerts))
            
            weekly_labels.append(w)
            weekly_mttc.append(mttc_val)
            weekly_esc_rate.append(esc_rate_val)
            weekly_finding_counts.append(f_count)
            weekly_alert_counts.append(a_count)
            
    trend_metrics = {
        'labels': [f"Wk {i+1} ({w[5:]})" for i, w in enumerate(weekly_labels)],
        'raw_dates': weekly_labels,
        'mttc': weekly_mttc,
        'escalation_rate': weekly_esc_rate,
        'findings_count': weekly_finding_counts,
        'alert_volume': weekly_alert_counts
    }
        
    return render_template('entity.html', 
                           entity=entity, 
                           findings=entity_findings,
                           trend_data=json.dumps(daily_trend_data),
                           trend_metrics=trend_metrics,
                           trend_metrics_json=json.dumps(trend_metrics))

@app.route('/api/sector_stats')
def api_sector_stats():
    findings, entities_df, _, _ = load_data()
    
    # Count findings by sector
    sector_counts = {}
    for f in findings:
        eid = f.get('entity_id')
        if eid:
            sector = entities_df[entities_df['entity_id'] == eid]['sector'].iloc[0]
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
    return jsonify(sector_counts)

@app.route('/finding/<entity_id>/<int:finding_idx>')
def finding_detail(entity_id, finding_idx):
    findings, entities_df, _, _ = load_data()
    
    entity_findings = [f for f in findings if f.get('entity_id') == entity_id]
    if finding_idx < 0 or finding_idx >= len(entity_findings):
        abort(404)
        
    finding = entity_findings[finding_idx]
    
    entity_row = entities_df[entities_df['entity_id'] == entity_id]
    entity = entity_row.iloc[0].to_dict() if not entity_row.empty else None
    
    return render_template('finding.html', finding=finding, entity=entity, index=finding_idx)

@app.route('/finding/<entity_id>/<int:finding_idx>/trace')
def finding_trace(entity_id, finding_idx):
    findings, entities_df, alerts_df, cases_df = load_data()
    
    entity_findings = [f for f in findings if f.get('entity_id') == entity_id]
    if finding_idx < 0 or finding_idx >= len(entity_findings):
        abort(404)
        
    finding = entity_findings[finding_idx]
    
    entity_row = entities_df[entities_df['entity_id'] == entity_id]
    entity = entity_row.iloc[0].to_dict() if not entity_row.empty else {'entity_name': 'Unknown', 'entity_id': entity_id}
    
    # Retrieve linked alert if available
    alert = None
    alert_id = finding.get('alert_id')
    if alert_id:
        alert_row = alerts_df[alerts_df['alert_id'] == alert_id]
        if not alert_row.empty:
            alert = alert_row.iloc[0].to_dict()
            
    # Retrieve linked case if available
    case = None
    case_id = finding.get('case_id')
    if case_id:
        case_row = cases_df[cases_df['case_id'] == case_id]
        if not case_row.empty:
            case = case_row.iloc[0].to_dict()
    elif alert_id:
        case_row = cases_df[cases_df['alert_id'] == alert_id]
        if not case_row.empty:
            case = case_row.iloc[0].to_dict()
            
    RULE_DESCRIPTIONS = {
        "critical_no_escalation": {
            "title": "Critical Alert Closed Without Escalation",
            "layer": "Layer A (Record-Level Rule)",
            "policy": "SOC-SOP-201: Mandatory Tier 2 Escalation for Critical Incidents",
            "threshold": "Alert Severity == Critical AND Escalation Status == No",
            "risk_rationale": "High-risk active compromise contained prematurely at Tier 1 without forensic validation or Incident Commander authorization."
        },
        "fast_closure_no_investigation": {
            "title": "Fast Alert Closure With Insufficient Investigation",
            "layer": "Layer A (Record-Level Rule)",
            "policy": "SOC-SOP-104: Minimum Triage Duration & Disposition Standards",
            "threshold": "Investigation Time < 180s AND Investigator Notes Length < 5 words",
            "risk_rationale": "Superficial closure / rubber-stamping on severe alert risking unmitigated intrusions."
        },
        "templated_investigation": {
            "title": "Templated / Boilerplate Investigation Narrative",
            "layer": "Layer A (Record-Level Rule)",
            "policy": "SOC-QA-305: Independent Narrative Analysis Requirement",
            "threshold": "Cosine Similarity > 0.90 across investigator notes within entity",
            "risk_rationale": "Repeated copy-pasting of identical narrative notes across multiple incidents indicating low-effort analyst compliance."
        },
        "missing_telemetry": {
            "title": "Zero Security Telemetry on Crown-Jewel Asset",
            "layer": "Layer A (Record-Level Rule)",
            "policy": "SOC-SEC-402: Critical Asset Continuous Visibility Mandate",
            "threshold": "Asset Criticality == Critical AND Total Alerts in 90 Days == 0",
            "risk_rationale": "Complete monitoring blind spot on a mission-critical system (agent down or unforwarded logs)."
        },
        "stale_open_case": {
            "title": "Unresolved Case Exceeding 30-Day SLA",
            "layer": "Layer A (Record-Level Rule)",
            "policy": "SOC-SLA-102: Incident Lifecycle & Case Aging Limit",
            "threshold": "Case Status != Closed AND Elapsed Time > 30 Days",
            "risk_rationale": "Abandoned investigation or unmanaged incident backlog exceeding maximum allowable SLA."
        },
        "root_cause_missing_on_critical": {
            "title": "Critical Incident Closed Without Root Cause Documentation",
            "layer": "Layer A (Record-Level Rule)",
            "policy": "SOC-PIR-501: Post-Incident Root Cause Analysis Requirement",
            "threshold": "Alert Severity == Critical AND Root Cause Documented == No",
            "risk_rationale": "Closure of confirmed severe incident without identifying initial intrusion vector, inviting repeat breach."
        },
        "peer_benchmark_mttc_outlier": {
            "title": "Entity MTTC Significantly Exceeds Sector Peers",
            "layer": "Layer B (Entity-Level Statistical Rule)",
            "policy": "SOC-PERF-601: Cross-Entity MTTC Operational Baseline",
            "threshold": "Z-Score(Entity MTTC) > +2.0 standard deviations above peer population",
            "risk_rationale": "Systematic operational latency in containing incidents, drastically increasing attacker dwell time."
        },
        "peer_benchmark_escalation_outlier": {
            "title": "Entity Escalation Rate Significantly Suppressed",
            "layer": "Layer B (Entity-Level Statistical Rule)",
            "policy": "SOC-PERF-602: Cross-Entity Escalation Rate Baseline",
            "threshold": "Z-Score(Entity Escalation Rate) < -2.0 standard deviations below peer population",
            "risk_rationale": "Abnormally low escalation rate across high-severity alerts, indicating chronic under-reporting."
        },
        "statistical_anomaly_isolation_forest": {
            "title": "Multivariate Behavioral Metric Anomaly",
            "layer": "Layer B (Machine Learning Ensemble)",
            "policy": "SOC-ML-701: Unsupervised Population Behavior Monitoring",
            "threshold": "IsolationForest Anomaly Score == -1 across MTTA, MTTC, Escalation Rate, and Volume",
            "risk_rationale": "Entity exhibits divergent operational characteristics across multiple dimensions simultaneously."
        }
    }
    
    rule_info = RULE_DESCRIPTIONS.get(finding.get('rule_triggered'), {
        "title": finding.get('rule_triggered', 'Detected Anomaly'),
        "layer": "Detection Engine Rule",
        "policy": "SOC Standard Operational Benchmark",
        "threshold": "Threshold criterion defined in engine",
        "risk_rationale": "Automated supervisory finding."
    })
    
    return render_template('trace.html',
                           finding=finding,
                           entity=entity,
                           alert=alert,
                           case=case,
                           rule_info=rule_info,
                           index=finding_idx)

@app.route('/raw_anomalies')
def raw_anomalies():
    findings, entities_df, _, _ = load_data()
    view_mode = request.args.get('view', 'analyst').lower()
    if view_mode not in ['manager', 'analyst']:
        view_mode = 'analyst'
    
    entity_name_map = dict(zip(entities_df['entity_id'], entities_df['entity_name']))
    entity_idx_counter = {}
    enriched_findings = []
    
    for f in findings:
        eid = f.get('entity_id')
        idx = entity_idx_counter.get(eid, 0)
        entity_idx_counter[eid] = idx + 1
        item = dict(f)
        item['entity_name'] = entity_name_map.get(eid, 'Unknown')
        item['finding_idx'] = idx
        enriched_findings.append(item)
        
    if view_mode == 'manager':
        enriched_findings.sort(key=lambda x: (x.get('severity_score', 0), x.get('confidence', 0)), reverse=True)
            
    return render_template('raw_anomalies.html', findings=enriched_findings, view_mode=view_mode)

@app.route('/sector_benchmarks')
def sector_benchmarks():
    findings, entities_df, alerts_df, _ = load_data()
    
    # Calculate stats per sector
    sector_stats = {}
    for sector in entities_df['sector'].unique():
        sector_stats[sector] = {
            'entities': 0,
            'anomalies': 0,
            'alerts': 0
        }
        
    for _, row in entities_df.iterrows():
        sector_stats[row['sector']]['entities'] += 1
        
    for f in findings:
        eid = f.get('entity_id')
        if eid:
            sector = entities_df[entities_df['entity_id'] == eid]['sector'].iloc[0]
            sector_stats[sector]['anomalies'] += 1
            
    for _, row in alerts_df.iterrows():
        sector = entities_df[entities_df['entity_id'] == row['entity_id']]['sector'].iloc[0]
        sector_stats[sector]['alerts'] += 1
        
    return render_template('sector_benchmarks.html', sector_stats=sector_stats)

@app.route('/diagnostics')
def diagnostics():
    findings, entities_df, alerts_df, _ = load_data()
    return render_template('diagnostics.html', total_findings=len(findings), total_alerts=len(alerts_df))

@app.route('/export/<entity_id>')
def export_entity_report(entity_id):
    findings, entities_df, _, _ = load_data()
    entity_findings = [f for f in findings if f.get('entity_id') == entity_id]
    
    if not entity_findings:
        abort(404)
        
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['Rule Triggered', 'Severity Score', 'Severity Tier', 'Confidence', 'Plain English Summary', 'Explanation', 'Alert ID', 'Case ID'])
    
    # Rows
    for f in entity_findings:
        writer.writerow([
            f.get('rule_triggered', ''),
            f.get('severity_score', ''),
            f.get('severity_of_finding', ''),
            f.get('confidence', ''),
            f.get('plain_english_summary', ''),
            f.get('explanation', ''),
            f.get('alert_id', ''),
            f.get('case_id', '')
        ])
        
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=sat_sa_report_{entity_id}.csv"}
    )

@app.route('/export/<entity_id>/pdf')
def export_entity_pdf(entity_id):
    findings, entities_df, alerts_df, cases_df = load_data()
    entity_row = entities_df[entities_df['entity_id'] == entity_id]
    if len(entity_row) == 0:
        abort(404)
        
    entity = entity_row.iloc[0].to_dict()
    entity_findings = [f for f in findings if f.get('entity_id') == entity_id]
    
    pdf_bytes = generate_entity_pdf(entity, entity_findings, alerts_df, cases_df)
    
    safe_name = entity.get('entity_name', 'entity').lower().replace(' ', '_').replace(',', '')
    filename = f"sat_sa_executive_summary_{safe_name}_{entity_id[:8]}.pdf"
    
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "application/pdf"
        }
    )

if __name__ == '__main__':
    import os as _os
    _debug = _os.environ.get('SAT_SA_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    app.run(debug=_debug, port=5000)

