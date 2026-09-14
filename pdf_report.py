import io
from datetime import datetime
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

from detection_engine import generate_plain_narrative

REMEDIATION_GUIDELINES = {
    "critical_no_escalation": (
        "Tier Escalation Matrix Enforcement",
        "Implement mandatory SIEM/ITSM workflow restrictions preventing Tier 1 triage analysts from closing Critical severity alerts without Tier 2 / Incident Response lead sign-off. Immediately audit all unescalated Critical incidents."
    ),
    "fast_closure_no_investigation": (
        "Triage SLA & Rubber-Stamping Review",
        "Establish an automated floor for investigation time (minimum 180 seconds) on High and Critical alerts. Require structured triage checklists and substantive investigative notes before alert disposition."
    ),
    "missing_telemetry": (
        "Crown-Jewel Telemetry Health Verification",
        "Initiate urgent diagnostic health checks on EDR agent connectivity, syslog forwarding, and network sensors across all Critical crown-jewel assets. Deploy dead-man alerting for agent silence exceeding 24 hours."
    ),
    "root_cause_missing_on_critical": (
        "Mandatory Root Cause Analysis (RCA)",
        "Institute mandatory post-incident RCA reviews for all Critical security incidents. Require SecOps leadership sign-off on threat vector attribution and mitigation prior to ticket closure."
    ),
    "templated_investigation": (
        "Investigation Quality & Documentation Standards",
        "Deploy automated text-similarity checks across closed case notes. Conduct analyst retraining on technical documentation standards and strictly prohibit copy-pasting boilerplate summaries."
    ),
    "stale_open_case": (
        "Incident Backlog Hygiene & Stale Ticket Sprints",
        "Enforce automated ticket notifications and manager escalation at 14 and 30 days of inactivity. Conduct weekly backlog grooming sprints to prevent abandoned or leaked investigations."
    ),
    "peer_benchmark_mttc_outlier": (
        "Mean Time to Close (MTTC) Optimization",
        "Analyze Tier 2 handoffs, automation playbooks (SOAR), and forensic workflow bottlenecks. Entity MTTC significantly exceeds sector peer benchmarks, prolonging adversary dwell time."
    ),
    "peer_benchmark_escalation_outlier": (
        "Threat Escalation Calibration & Retraining",
        "Conduct a targeted audit of high-severity alert triage criteria. The entity escalates high-severity incidents at a rate significantly below sector peers, indicating potential under-reporting."
    ),
    "statistical_anomaly_isolation_forest": (
        "Supervisory Operational Workflow Review",
        "Initiate an in-depth supervisory review of operational workflows. Multi-metric anomaly modeling indicates significant systemic deviation from normal SOC performance baselines."
    ),
}

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 11 * 72 - 30, "SAT-SA | Supervisory Analytics Tool for SOC Assessment — Executive Audit Report")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(36, 11 * 72 - 34, 8.5 * 72 - 36, 11 * 72 - 34)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 36, 8.5 * 72 - 36, 36)
        
        self.drawString(36, 24, "CONFIDENTIAL // FOR SUPERVISORY & OVERSIGHT USE ONLY")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 36, 24, page_str)
        self.restoreState()


def generate_entity_pdf(entity, findings, alerts_df, cases_df):
    """
    Generates a professional PDF Executive Summary for the specified entity.
    Returns: bytes (PDF binary data)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=44
    )
    
    # Page width = 8.5 * 72 = 612. Usable width = 612 - 72 = 540
    usable_width = 540
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0284c7')
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#475569')
    )

    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )
    
    subhead_style = ParagraphStyle(
        'SubheadStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#1e293b')
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )

    finding_title_style = ParagraphStyle(
        'FindingTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#0f172a')
    )

    narrative_style = ParagraphStyle(
        'NarrativeStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )

    tech_style = ParagraphStyle(
        'TechStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.0,
        leading=11,
        textColor=colors.HexColor('#475569')
    )

    badge_crit = ParagraphStyle(
        'BadgeCrit',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
        alignment=1
    )
    
    badge_high = ParagraphStyle(
        'BadgeHigh',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
        alignment=1
    )

    badge_med = ParagraphStyle(
        'BadgeMed',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
        alignment=1
    )

    story = []
    
    # Ensure plain_english_summary exists on all findings
    for f in findings:
        if not f.get('plain_english_summary'):
            f['plain_english_summary'] = generate_plain_narrative(f)

    # 1. HEADER BANNER
    header_table_data = [
        [
            Paragraph("<b>SAT-SA | SUPERVISORY ANALYTICS TOOL FOR SOC ASSESSMENT</b>", subtitle_style),
            Paragraph(f"<b>Audit Date:</b> {datetime.now().strftime('%B %d, %Y')}", meta_style)
        ],
        [
            Paragraph("Executive Supervisory Audit & Compliance Report", title_style),
            Paragraph("<b>Classification:</b> RESTRICTED // INTERNAL USE ONLY", meta_style)
        ]
    ]
    header_table = Table(header_table_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # 2. ENTITY METADATA & EXECUTIVE KPI SUMMARY
    # Calculate entity-level metrics
    entity_id = entity.get('entity_id', '')
    entity_alerts = alerts_df[alerts_df['entity_id'] == entity_id] if not alerts_df.empty else pd.DataFrame()
    entity_cases = cases_df[cases_df['entity_id'] == entity_id] if not cases_df.empty else pd.DataFrame()
    
    # Calculate MTTC
    mttc_val = "N/A"
    if not entity_alerts.empty and 'acknowledged_at' in entity_alerts.columns and 'closed_at' in entity_alerts.columns:
        ack_dt = pd.to_datetime(entity_alerts['acknowledged_at'], format='ISO8601', errors='coerce')
        cls_dt = pd.to_datetime(entity_alerts['closed_at'], format='ISO8601', errors='coerce')
        mttc_s = (cls_dt - ack_dt).dt.total_seconds() / 60
        valid_mttc = mttc_s.dropna()
        if len(valid_mttc) > 0:
            mttc_val = f"{valid_mttc.mean():.1f} mins"
            
    # Calculate Escalation Rate for High/Critical
    esc_rate_val = "N/A"
    if not entity_cases.empty and not entity_alerts.empty:
        merged_cases = entity_cases.merge(entity_alerts[['alert_id', 'severity']], on='alert_id', how='left')
        hc_cases = merged_cases[merged_cases['severity'].isin(['High', 'Critical'])]
        if len(hc_cases) > 0:
            rate = (hc_cases['escalated'] == 'Yes').sum() / len(hc_cases) * 100
            esc_rate_val = f"{rate:.1f}%"

    total_findings = len(findings)
    crit_count = sum(1 for f in findings if f.get('severity_score', 0) >= 9)
    high_count = sum(1 for f in findings if 7 <= f.get('severity_score', 0) <= 8)
    med_count = sum(1 for f in findings if 4 <= f.get('severity_score', 0) <= 6)
    total_risk_score = sum(f.get('severity_score', 5) for f in findings)
    
    if crit_count >= 10 or total_risk_score >= 300:
        posture_status = "IMMEDIATE ACTION REQUIRED"
        posture_color = colors.HexColor("#dc2626")
    elif high_count >= 15 or total_risk_score >= 150:
        posture_status = "ELEVATED SUPERVISORY RISK"
        posture_color = colors.HexColor("#ea580c")
    else:
        posture_status = "ROUTINE OVERSIGHT"
        posture_color = colors.HexColor("#0284c7")

    meta_kpi_table_data = [
        [
            Paragraph("<b>MONITORED ENTITY DETAILS</b>", subhead_style),
            Paragraph("<b>SUPERVISORY RISK POSTURE</b>", subhead_style)
        ],
        [
            Paragraph(f"<b>Organization:</b> {entity.get('entity_name', 'Unknown')}<br/>"
                      f"<b>Entity UUID:</b> <font face='Courier'>{entity_id}</font><br/>"
                      f"<b>Industry Sector:</b> {entity.get('sector', 'Unknown')}<br/>"
                      f"<b>Scale Tier:</b> {entity.get('size_tier', 'Medium')} Enterprise<br/>"
                      f"<b>Observation Scope:</b> 90-Day Continuous Telemetry", body_style),
            Paragraph(f"<b>Audit Finding Volume:</b> {total_findings} Policy Violations<br/>"
                      f"<b>Critical Tier (Scores 9-10):</b> <font color='#b91c1c'><b>{crit_count}</b></font> findings<br/>"
                      f"<b>High Tier (Scores 7-8):</b> <font color='#c2410c'><b>{high_count}</b></font> findings<br/>"
                      f"<b>Medium Tier (Scores 4-6):</b> <font color='#b45309'><b>{med_count}</b></font> findings<br/>"
                      f"<b>Resolution Velocity (Avg MTTC):</b> {mttc_val}<br/>"
                      f"<b>High/Crit Escalation Rate:</b> {esc_rate_val}<br/>"
                      f"<b>Overall Posture:</b> <font color='{posture_color.hexval()}'><b>{posture_status}</b></font>", body_style)
        ]
    ]
    meta_kpi_table = Table(meta_kpi_table_data, colWidths=[260, 280])
    meta_kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_kpi_table)
    story.append(Spacer(1, 14))

    # 3. GROUPED FINDINGS DOSSIER (Sorted by severity_score descending, then confidence descending)
    story.append(Paragraph("<b>Audited Findings & Policy Violations Dossier</b>", section_header_style))
    story.append(Paragraph("Findings are ranked by assessed severity score and algorithmic confidence. Each violation includes an executive narrative, specific numerical telemetry evidence, and technical references.", meta_style))
    story.append(Spacer(1, 8))

    # Sort findings
    sorted_findings = sorted(
        findings,
        key=lambda x: (x.get('severity_score', 0), x.get('confidence', 0.0)),
        reverse=True
    )

    if not sorted_findings:
        story.append(Paragraph("<i>No compliance anomalies or policy violations detected for this entity during the audit window.</i>", body_style))
        story.append(Spacer(1, 14))
    else:
        # Group into tiers: Critical (9-10), High (7-8), Medium (4-6), Low (1-3)
        current_tier = None
        
        for f in sorted_findings:
            score = f.get('severity_score', 5)
            tier_name = f.get('severity_of_finding', 'Medium')
            
            # Format evidence string
            ev = f.get('evidence', {})
            ev_items = []
            for k, v in ev.items():
                k_clean = k.replace('_', ' ').capitalize()
                ev_items.append(f"{k_clean}: <b>{v}</b>")
            ev_str = " &bull; ".join(ev_items) if ev_items else "None recorded"
            
            # References
            ref_items = []
            if f.get('alert_id'):
                ref_items.append(f"Alert ID: <font face='Courier'>{f['alert_id'][:18]}...</font>")
            if f.get('case_id'):
                ref_items.append(f"Case ID: <font face='Courier'>{f['case_id'][:18]}...</font>")
            if f.get('asset_id'):
                ref_items.append(f"Asset ID: <font face='Courier'>{f['asset_id']}</font>")
            ref_str = " | ".join(ref_items) if ref_items else "Entity-Wide Metric Evaluation"

            # Badge styling
            if score >= 9:
                tier_bg = colors.HexColor("#dc2626")
                badge_style = badge_crit
            elif score >= 7:
                tier_bg = colors.HexColor("#ea580c")
                badge_style = badge_high
            else:
                tier_bg = colors.HexColor("#b45309")
                badge_style = badge_med
                
            conf_pct = int(round(f.get('confidence', 0.8) * 100))
            rule_clean = f.get('rule_triggered', 'Unknown').replace('_', ' ').title()

            finding_box_data = [
                [
                    Paragraph(f"<b>{rule_clean}</b> &nbsp; <font face='Courier' size=7 color='#64748b'>({f.get('rule_triggered')})</font>", finding_title_style),
                    Paragraph(f"<b>{tier_name.upper()} ({score}/10)</b>", badge_style),
                    Paragraph(f"<b>{conf_pct}% Conf.</b>", badge_med)
                ],
                [
                    Paragraph(f"<b>Executive Summary:</b> {f.get('plain_english_summary')}", narrative_style),
                    "",
                    ""
                ],
                [
                    Paragraph(f"<b>Evidence & Telemetry:</b> {ev_str}<br/><b>Audit Chain:</b> {ref_str}", meta_style),
                    "",
                    ""
                ],
                [
                    Paragraph(f"<b>Technical SOC Context:</b> {f.get('explanation')}", tech_style),
                    "",
                    ""
                ]
            ]
            
            finding_table = Table(finding_box_data, colWidths=[380, 85, 75])
            finding_table.setStyle(TableStyle([
                ('SPAN', (0, 1), (2, 1)),
                ('SPAN', (0, 2), (2, 2)),
                ('SPAN', (0, 3), (2, 3)),
                ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#f1f5f9")),
                ('BACKGROUND', (1, 0), (1, 0), tier_bg),
                ('BACKGROUND', (2, 0), (2, 0), colors.HexColor("#0284c7")),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#ffffff")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor("#cbd5e1")),
                ('LINEBELOW', (0, 1), (-1, 1), 0.5, colors.HexColor("#f1f5f9")),
                ('LINEBELOW', (0, 2), (-1, 2), 0.5, colors.HexColor("#f1f5f9")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(KeepTogether([finding_table, Spacer(1, 6)]))

    story.append(Spacer(1, 10))

    # 4. REMEDIATION & GOVERNANCE ROADMAP
    story.append(KeepTogether([
        Paragraph("<b>Targeted Supervisory Remediation & Governance Roadmap</b>", section_header_style),
        Paragraph("Based on the specific violation patterns and algorithmic outliers detected during the audit window, the supervisory authority mandates the following corrective measures for SOC leadership:", meta_style),
        Spacer(1, 6)
    ]))

    triggered_rules = sorted(list(set(f.get('rule_triggered') for f in findings if f.get('rule_triggered'))))
    
    if not triggered_rules:
        story.append(Paragraph("<i>No specific corrective actions required. Maintain existing SOC operational controls and monitoring.</i>", body_style))
    else:
        rec_table_rows = []
        for idx, rule in enumerate(triggered_rules, 1):
            title, desc = REMEDIATION_GUIDELINES.get(
                rule,
                ("Operational Workflow Review", "Conduct a standard operational and technical review of the alerted incident workflows.")
            )
            rule_clean = rule.replace('_', ' ').title()
            
            rec_cell = Paragraph(
                f"<b>{idx}. {title}</b> &nbsp; <font color='#64748b' size=7.5><i>(Governing Rule: {rule_clean})</i></font><br/>"
                f"<font color='#334155'>{desc}</font>",
                body_style
            )
            rec_table_rows.append([rec_cell])
            
        rec_table = Table(rec_table_rows, colWidths=[540])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(rec_table)

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
