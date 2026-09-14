"""
tests/test_detection_rules.py
==============================
Unit tests for every detection rule in detection_engine.py.

Fixture data is hand-crafted in each test so the expected outcome (flag / no-flag)
is precisely known without relying on the full synthetic dataset.
"""

import sys
import os
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# Ensure repo root is importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detection_engine import (
    CFG,
    RULE_SEVERITY_WEIGHTS,
    get_severity_tier,
    rule_fast_closure,
    rule_critical_no_escalation,
    rule_templated_investigation,
    rule_missing_telemetry,
    rule_stale_open_case,
    rule_root_cause_missing,
    rule_statistical_anomalies,
    load_config,
)


# ---------------------------------------------------------------------------
# Helper: minimal alert and case DataFrames sharing a common alert_id
# ---------------------------------------------------------------------------

def _make_alert(alert_id='a1', entity_id='e1', severity='High',
                investigation_time_secs=60, notes_length=2,
                asset_id='asset-1', source_system='SIEM',
                alert_category='Malware', closure_reason='Resolved'):
    """Build a one-row alerts DataFrame with all required columns."""
    now = datetime.utcnow()
    ack = now - timedelta(seconds=investigation_time_secs + 10)
    closed = ack + timedelta(seconds=investigation_time_secs)
    return pd.DataFrame([{
        'alert_id': alert_id,
        'entity_id': entity_id,
        'asset_id': asset_id,
        'severity': severity,
        'timestamp': now - timedelta(minutes=30),
        'acknowledged_at': ack,
        'closed_at': closed,
        'closure_reason': closure_reason,
        'alert_category': alert_category,
        'source_system': source_system,
    }])


def _make_case(alert_id='a1', entity_id='e1', case_id='c1',
               escalated='No', root_cause='Yes',
               days_open=None, closed=True,
               notes='investigation complete', notes_length=3):
    """Build a one-row cases DataFrame with all required columns."""
    now = datetime.utcnow()
    opened_at = now - timedelta(days=days_open if days_open else 5)
    closed_at = now if closed else None
    return pd.DataFrame([{
        'case_id': case_id,
        'alert_id': alert_id,
        'entity_id': entity_id,
        'escalated': escalated,
        'escalated_at': now if escalated == 'Yes' else None,
        'opened_at': opened_at,
        'closed_at': closed_at,
        'root_cause_documented': root_cause,
        'investigator_notes': notes,
        'investigator_notes_length': notes_length,
    }])


# ===========================================================================
# CONFIG & SEVERITY TIER TESTS
# ===========================================================================

class TestConfig:
    """Verify config loading and basic structural integrity."""

    def test_load_config_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict), "load_config() must return a dict"

    def test_all_rule_keys_present_in_severity_weights(self):
        """Every expected rule name must have a severity weight."""
        expected_rules = {
            'critical_no_escalation',
            'missing_telemetry',
            'root_cause_missing_on_critical',
            'fast_closure_no_investigation',
            'peer_benchmark_escalation_outlier',
            'peer_benchmark_mttc_outlier',
            'stale_open_case',
            'statistical_anomaly_isolation_forest',
            'templated_investigation',
        }
        cfg = load_config()
        assert expected_rules == set(cfg['severity_weights'].keys())

    def test_severity_weight_values_in_range(self):
        """All severity weights must be integers in 1-10."""
        for rule, weight in CFG['severity_weights'].items():
            assert 1 <= weight <= 10, f"{rule} weight {weight} out of 1-10 range"

    def test_get_severity_tier_critical(self):
        assert get_severity_tier(10) == "Critical"
        assert get_severity_tier(9) == "Critical"

    def test_get_severity_tier_high(self):
        assert get_severity_tier(8) == "High"
        assert get_severity_tier(7) == "High"

    def test_get_severity_tier_medium(self):
        assert get_severity_tier(6) == "Medium"
        assert get_severity_tier(5) == "Medium"
        assert get_severity_tier(4) == "Medium"

    def test_get_severity_tier_low(self):
        assert get_severity_tier(3) == "Low"
        assert get_severity_tier(1) == "Low"


# ===========================================================================
# RULE: fast_closure_no_investigation
# ===========================================================================

class TestRuleFastClosure:
    """
    Threshold (from config): investigation_time < 180s AND notes_words < 5.
    Only High and Critical severity alerts are evaluated.
    """

    def _run(self, secs, words, severity='High'):
        alerts = _make_alert(severity=severity, investigation_time_secs=secs)
        # investigator_notes_length lives on the case record, not the alert
        cases = _make_case(notes_length=words)
        return rule_fast_closure(alerts, cases)

    def test_flags_when_both_conditions_met(self):
        """Short time + short notes on High alert must produce a finding."""
        findings = self._run(secs=60, words=2, severity='High')
        assert len(findings) == 1
        f = findings[0]
        assert f['rule_triggered'] == 'fast_closure_no_investigation'
        assert f['severity_score'] == RULE_SEVERITY_WEIGHTS['fast_closure_no_investigation']
        assert f['severity_of_finding'] == 'High'   # score=8 -> High

    def test_flags_critical_severity(self):
        """Critical alerts also trigger the rule and get a confidence bonus."""
        findings = self._run(secs=30, words=1, severity='Critical')
        assert len(findings) == 1
        # Critical bonus should push confidence above the High counterpart
        findings_high = self._run(secs=30, words=1, severity='High')
        assert findings[0]['confidence'] > findings_high[0]['confidence']

    def test_no_finding_when_time_above_threshold(self):
        """Investigation time above threshold should not trigger the rule."""
        findings = self._run(secs=200, words=2)
        assert len(findings) == 0

    def test_no_finding_when_notes_above_threshold(self):
        """Sufficient notes length should not trigger the rule."""
        findings = self._run(secs=60, words=6)
        assert len(findings) == 0

    def test_no_finding_for_low_severity(self):
        """Low severity alerts are outside the rule scope."""
        findings = self._run(secs=30, words=1, severity='Low')
        assert len(findings) == 0

    def test_no_finding_for_medium_severity(self):
        """Medium severity alerts are outside the rule scope."""
        findings = self._run(secs=30, words=1, severity='Medium')
        assert len(findings) == 0

    def test_confidence_bounded(self):
        """Confidence must always be in [conf_min, conf_max] range."""
        fc = CFG['fast_closure']
        findings = self._run(secs=1, words=0, severity='Critical')
        assert len(findings) == 1
        conf = findings[0]['confidence']
        assert fc['conf_min'] <= conf <= fc['conf_max']


# ===========================================================================
# RULE: critical_no_escalation
# ===========================================================================

class TestRuleCriticalNoEscalation:
    """
    Threshold: severity == 'Critical' AND escalated == 'No'.
    """

    def _run(self, severity='Critical', escalated='No', closure_reason='Resolved',
             root_cause='Yes'):
        alerts = _make_alert(severity=severity, closure_reason=closure_reason)
        cases = _make_case(escalated=escalated, root_cause=root_cause)
        return rule_critical_no_escalation(cases, alerts)

    def test_flags_critical_not_escalated(self):
        findings = self._run(severity='Critical', escalated='No')
        assert len(findings) == 1
        assert findings[0]['rule_triggered'] == 'critical_no_escalation'
        assert findings[0]['severity_score'] == 10

    def test_no_flag_if_escalated(self):
        """Critical alert that was escalated must not produce a finding."""
        findings = self._run(severity='Critical', escalated='Yes')
        assert len(findings) == 0

    def test_no_flag_for_non_critical(self):
        """High severity alert should never trigger this rule."""
        findings = self._run(severity='High', escalated='No')
        assert len(findings) == 0

    def test_confidence_higher_with_confirmed_true_positive(self):
        """'True Positive' closure_reason boosts confidence above base."""
        cne = CFG['critical_no_escalation']
        findings_tp = self._run(closure_reason='True Positive')
        findings_other = self._run(closure_reason='False Positive')
        assert findings_tp[0]['confidence'] > findings_other[0]['confidence']
        assert findings_tp[0]['confidence'] <= cne['conf_max']

    def test_evidence_fields_present(self):
        findings = self._run()
        ev = findings[0]['evidence']
        assert ev['alert_severity'] == 'Critical'
        assert ev['escalated'] == 'No'


# ===========================================================================
# RULE: templated_investigation
# ===========================================================================

class TestRuleTemplatedInvestigation:
    """
    Threshold: notes_length >= 5 AND cosine_similarity > 0.90 between pairs.
    """

    BOILERPLATE = (
        "Investigated the security alert by reviewing system logs and correlating "
        "with threat intelligence feeds. No malicious activity was identified."
    )

    def _cases_with_notes(self, note1, note2, entity_id='e1', notes_length=15):
        return pd.DataFrame([
            {
                'case_id': 'c1', 'alert_id': 'a1', 'entity_id': entity_id,
                'investigator_notes': note1, 'investigator_notes_length': notes_length,
                'escalated': 'No', 'escalated_at': None,
                'opened_at': datetime.utcnow() - timedelta(days=5),
                'closed_at': datetime.utcnow(),
                'root_cause_documented': 'No',
            },
            {
                'case_id': 'c2', 'alert_id': 'a2', 'entity_id': entity_id,
                'investigator_notes': note2, 'investigator_notes_length': notes_length,
                'escalated': 'No', 'escalated_at': None,
                'opened_at': datetime.utcnow() - timedelta(days=3),
                'closed_at': datetime.utcnow(),
                'root_cause_documented': 'No',
            }
        ])

    def test_flags_near_identical_notes(self):
        """Two nearly identical substantive notes must both be flagged."""
        cases = self._cases_with_notes(self.BOILERPLATE, self.BOILERPLATE)
        findings = rule_templated_investigation(cases)
        assert len(findings) == 2
        for f in findings:
            assert f['rule_triggered'] == 'templated_investigation'

    def test_no_flag_for_distinct_notes(self):
        """Completely different notes must not trigger the rule."""
        note_a = "Malware signature detected. Process terminated. IOC quarantined. Root cause: phishing email."
        note_b = "Network anomaly observed. Port scan from external IP. Firewall rule applied. No data exfiltration detected."
        cases = self._cases_with_notes(note_a, note_b)
        findings = rule_templated_investigation(cases)
        assert len(findings) == 0

    def test_no_flag_for_short_notes(self):
        """Notes with fewer than min_notes_word_count words are excluded (rubber-stamp path)."""
        short = "Resolved."
        cases = self._cases_with_notes(short, short, notes_length=1)
        findings = rule_templated_investigation(cases)
        assert len(findings) == 0

    def test_no_flag_for_single_case_entity(self):
        """An entity with only one case cannot form a pair — no finding."""
        cases = pd.DataFrame([{
            'case_id': 'c1', 'alert_id': 'a1', 'entity_id': 'e1',
            'investigator_notes': self.BOILERPLATE, 'investigator_notes_length': 20,
            'escalated': 'No', 'escalated_at': None,
            'opened_at': datetime.utcnow() - timedelta(days=5),
            'closed_at': datetime.utcnow(),
            'root_cause_documented': 'No',
        }])
        findings = rule_templated_investigation(cases)
        assert len(findings) == 0

    def test_evidence_contains_similarity_score(self):
        cases = self._cases_with_notes(self.BOILERPLATE, self.BOILERPLATE)
        findings = rule_templated_investigation(cases)
        assert 'similarity_score' in findings[0]['evidence']
        sim = float(findings[0]['evidence']['similarity_score'])
        assert sim > CFG['templated_investigation']['similarity_threshold']


# ===========================================================================
# RULE: missing_telemetry
# ===========================================================================

class TestRuleMissingTelemetry:
    """
    Threshold: Critical asset with zero alerts in the dataset.
    """

    def _assets(self, criticality='Critical', asset_id='crown-jewel-1', entity_id='e1'):
        return pd.DataFrame([{
            'asset_id': asset_id, 'entity_id': entity_id, 'criticality': criticality,
        }])

    def _alerts_for(self, asset_id='some-other-asset'):
        """Return an alerts df that only has alerts for a DIFFERENT asset."""
        return pd.DataFrame([{
            'alert_id': 'a1', 'entity_id': 'e2', 'asset_id': asset_id,
            'severity': 'High',
            'timestamp': datetime.utcnow() - timedelta(days=10),
            'acknowledged_at': datetime.utcnow() - timedelta(days=10),
            'closed_at': datetime.utcnow() - timedelta(days=10),
            'investigator_notes_length': 5,
            'closure_reason': 'Resolved',
            'alert_category': 'Malware',
            'source_system': 'SIEM',
        }])

    def test_flags_critical_asset_with_zero_alerts(self):
        """A Critical asset with no matching alerts must be flagged."""
        assets = self._assets(criticality='Critical', asset_id='crown-jewel-1')
        # Alerts exist, but for a completely different asset
        alerts = self._alerts_for(asset_id='unrelated-asset-99')
        findings = rule_missing_telemetry(alerts, assets)
        assert len(findings) == 1
        f = findings[0]
        assert f['rule_triggered'] == 'missing_telemetry'
        assert f['evidence']['alerts_generated'] == 0
        assert f['severity_score'] == 9

    def test_no_flag_when_asset_has_alerts(self):
        """A Critical asset with at least one alert must not be flagged."""
        assets = self._assets(criticality='Critical', asset_id='crown-jewel-1')
        alerts = self._alerts_for(asset_id='crown-jewel-1')
        findings = rule_missing_telemetry(alerts, assets)
        assert len(findings) == 0

    def test_no_flag_for_non_critical_asset(self):
        """Non-critical assets are outside the rule scope."""
        assets = self._assets(criticality='High', asset_id='regular-server')
        alerts = self._alerts_for(asset_id='unrelated-asset')
        findings = rule_missing_telemetry(alerts, assets)
        assert len(findings) == 0


# ===========================================================================
# RULE: stale_open_case
# ===========================================================================

class TestRuleStaleOpenCase:
    """
    Threshold: case still open (closed_at is NaT) AND days_open > 30.
    """

    def _cases(self, days_open, closed=False):
        opened_at = datetime.utcnow() - timedelta(days=days_open)
        return pd.DataFrame([{
            'case_id': 'c1', 'alert_id': 'a1', 'entity_id': 'e1',
            'escalated': 'No', 'escalated_at': None,
            'opened_at': opened_at,
            'closed_at': datetime.utcnow() if closed else None,
            'root_cause_documented': 'No',
            'investigator_notes': 'Pending review.',
            'investigator_notes_length': 2,
        }])

    def test_flags_open_case_over_threshold(self):
        max_days = CFG['stale_open_case']['max_days_open']
        cases = self._cases(days_open=max_days + 10, closed=False)
        findings = rule_stale_open_case(cases)
        assert len(findings) == 1
        assert findings[0]['rule_triggered'] == 'stale_open_case'
        assert findings[0]['evidence']['days_open'] > max_days

    def test_no_flag_for_recent_open_case(self):
        """A case open for fewer days than the threshold must not be flagged."""
        max_days = CFG['stale_open_case']['max_days_open']
        cases = self._cases(days_open=max_days - 5, closed=False)
        findings = rule_stale_open_case(cases)
        assert len(findings) == 0

    def test_no_flag_for_closed_case(self):
        """Closed cases must never be flagged regardless of age."""
        cases = self._cases(days_open=60, closed=True)
        findings = rule_stale_open_case(cases)
        assert len(findings) == 0

    def test_confidence_increases_with_age(self):
        """Older stale cases should have higher confidence than borderline ones."""
        max_days = CFG['stale_open_case']['max_days_open']
        findings_old = rule_stale_open_case(self._cases(days_open=max_days + 90))
        findings_new = rule_stale_open_case(self._cases(days_open=max_days + 1))
        assert len(findings_old) == 1
        assert len(findings_new) == 1
        assert findings_old[0]['confidence'] > findings_new[0]['confidence']


# ===========================================================================
# RULE: root_cause_missing_on_critical
# ===========================================================================

class TestRuleRootCauseMissing:
    """
    Threshold: severity == 'Critical' AND root_cause_documented == 'No'.
    """

    def _run(self, severity='Critical', root_cause='No',
             closure_reason='Resolved', notes_length=5):
        alerts = _make_alert(severity=severity, closure_reason=closure_reason,
                             notes_length=notes_length)
        cases = _make_case(root_cause=root_cause, notes_length=notes_length,
                           notes='investigated malware infection initial access via phishing')
        return rule_root_cause_missing(cases, alerts)

    def test_flags_critical_missing_rca(self):
        findings = self._run(severity='Critical', root_cause='No')
        assert len(findings) == 1
        assert findings[0]['rule_triggered'] == 'root_cause_missing_on_critical'
        assert findings[0]['severity_score'] == 8

    def test_no_flag_when_rca_documented(self):
        findings = self._run(severity='Critical', root_cause='Yes')
        assert len(findings) == 0

    def test_no_flag_for_non_critical(self):
        findings = self._run(severity='High', root_cause='No')
        assert len(findings) == 0

    def test_confidence_boosted_by_confirmed_closure(self):
        rcm = CFG['root_cause_missing']
        findings_confirmed = self._run(closure_reason='True Positive')
        findings_other = self._run(closure_reason='False Positive')
        assert findings_confirmed[0]['confidence'] > findings_other[0]['confidence']
        assert findings_confirmed[0]['confidence'] <= rcm['conf_max']

    def test_evidence_fields_present(self):
        findings = self._run()
        ev = findings[0]['evidence']
        assert ev['root_cause_documented'] == 'No'
        assert ev['alert_severity'] == 'Critical'


# ===========================================================================
# RULE: rule_statistical_anomalies (Layer B — Z-score + IsolationForest)
# ===========================================================================

class TestRuleStatisticalAnomalies:
    """
    Layer B rules require population-level data.
    We construct a minimal entities set (4 entities) where one is a clear
    MTTC outlier and one is a clear escalation rate outlier, so the
    Z-scores reliably exceed the 2.0 threshold.
    """

    def _build_population(self):
        """
        Build four entities:
          - e1: normal MTTC (10 min), normal escalation (60%)
          - e2: normal MTTC (12 min), normal escalation (55%)
          - e3: extremely high MTTC (500 min) -> MTTC outlier
          - e4: zero escalation rate          -> escalation outlier
        """
        entities_df = pd.DataFrame([
            {'entity_id': 'e1', 'sector': 'Banking'},
            {'entity_id': 'e2', 'sector': 'Banking'},
            {'entity_id': 'e3', 'sector': 'Banking'},
            {'entity_id': 'e4', 'sector': 'Banking'},
        ])

        base_ts = datetime(2025, 1, 1)
        rows_alerts = []
        rows_cases = []

        def _add_entity(eid, mttc_mins, esc_rate, n=20):
            num_yes = round(esc_rate * n)
            for i in range(n):
                aid = f'{eid}-a{i}'
                cid = f'{eid}-c{i}'
                ts = base_ts + timedelta(hours=i * 3)
                ack = ts + timedelta(minutes=2)
                closed = ack + timedelta(minutes=mttc_mins)
                severity = 'High'
                # Use a count-based assignment (not i/n < rate) so the resulting
                # escalation rate exactly matches esc_rate regardless of n —
                # the old threshold comparison collapsed distinct rates like
                # 0.55/0.58/0.6 to the same value at small n.
                escalated = 'Yes' if i < num_yes else 'No'
                rows_alerts.append({
                    'alert_id': aid, 'entity_id': eid, 'asset_id': f'asset-{eid}',
                    'severity': severity, 'timestamp': ts,
                    'acknowledged_at': ack, 'closed_at': closed,
                    'investigator_notes_length': 5,
                    'closure_reason': 'Resolved',
                    'alert_category': 'Malware', 'source_system': 'SIEM',
                })
                rows_cases.append({
                    'case_id': cid, 'alert_id': aid, 'entity_id': eid,
                    'escalated': escalated, 'escalated_at': ack if escalated == 'Yes' else None,
                    'opened_at': ts, 'closed_at': closed,
                    'root_cause_documented': 'Yes',
                    'investigator_notes': 'investigation complete',
                    'investigator_notes_length': 5,
                })

        _add_entity('e1', mttc_mins=10, esc_rate=0.6)
        _add_entity('e2', mttc_mins=12, esc_rate=0.55)
        _add_entity('e3', mttc_mins=500, esc_rate=0.58)  # MTTC outlier
        _add_entity('e4', mttc_mins=11, esc_rate=0.0)    # escalation outlier

        alerts_df = pd.DataFrame(rows_alerts)
        cases_df = pd.DataFrame(rows_cases)

        # Parse datetime columns
        for col in ['timestamp', 'acknowledged_at', 'closed_at']:
            alerts_df[col] = pd.to_datetime(alerts_df[col])
        for col in ['opened_at', 'closed_at', 'escalated_at']:
            cases_df[col] = pd.to_datetime(cases_df[col])

        return alerts_df, cases_df, entities_df

    def test_mttc_outlier_flagged(self):
        """Entity with extremely high MTTC should trigger peer_benchmark_mttc_outlier."""
        alerts, cases, entities = self._build_population()
        findings = rule_statistical_anomalies(cases, alerts, entities)
        mttc_findings = [f for f in findings if f['rule_triggered'] == 'peer_benchmark_mttc_outlier']
        assert len(mttc_findings) >= 1, "Expected at least one MTTC outlier finding"
        entity_ids = [f['entity_id'] for f in mttc_findings]
        assert 'e3' in entity_ids, "Entity e3 (high MTTC) must be flagged"

    def test_escalation_outlier_flagged(self):
        """Entity with zero escalation rate should trigger peer_benchmark_escalation_outlier."""
        alerts, cases, entities = self._build_population()
        findings = rule_statistical_anomalies(cases, alerts, entities)
        esc_findings = [f for f in findings if f['rule_triggered'] == 'peer_benchmark_escalation_outlier']
        assert len(esc_findings) >= 1, "Expected at least one escalation outlier finding"
        entity_ids = [f['entity_id'] for f in esc_findings]
        assert 'e4' in entity_ids, "Entity e4 (zero escalation) must be flagged"

    def test_normal_entities_not_flagged_by_zscore_rules(self):
        """Entities e1 and e2 (normal metrics) must not trigger Z-score rules."""
        alerts, cases, entities = self._build_population()
        findings = rule_statistical_anomalies(cases, alerts, entities)
        zscore_rules = {'peer_benchmark_mttc_outlier', 'peer_benchmark_escalation_outlier'}
        flagged_eids = {f['entity_id'] for f in findings if f['rule_triggered'] in zscore_rules}
        assert 'e1' not in flagged_eids
        assert 'e2' not in flagged_eids

    def test_returns_empty_for_no_data(self):
        """Empty DataFrames must return an empty findings list, not crash."""
        entities_df = pd.DataFrame(columns=['entity_id', 'sector'])
        alerts_df = pd.DataFrame(columns=['alert_id', 'entity_id', 'asset_id', 'severity',
                                          'timestamp', 'acknowledged_at', 'closed_at',
                                          'investigator_notes_length', 'closure_reason',
                                          'alert_category', 'source_system'])
        cases_df = pd.DataFrame(columns=['case_id', 'alert_id', 'entity_id', 'escalated',
                                         'escalated_at', 'opened_at', 'closed_at',
                                         'root_cause_documented', 'investigator_notes',
                                         'investigator_notes_length'])
        findings = rule_statistical_anomalies(cases_df, alerts_df, entities_df)
        assert findings == []
