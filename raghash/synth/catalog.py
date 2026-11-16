"""Static catalogue for the synthetic corpus: departments, regions, fact templates, boilerplate."""
from __future__ import annotations

from dataclasses import dataclass

DEPARTMENTS = {
    "hr": "People Operations",
    "it": "IT Security",
    "finance": "Finance Operations",
    "engineering": "Platform Engineering",
}
REGIONS = ["India", "United States", "United Kingdom", "Germany", "Singapore"]
FAQ_OWNER = "Employee Communications"

# Scope sets. A "scope" is who/what a rule applies to. Region is a second scope dimension
# (it is part of the fact key), so the same rule may legitimately differ across regions.
ROLES = ("interns", "associates", "senior staff", "managers", "directors")
TENURE = (
    "employees with under 1 year of service",
    "employees with 1 to 3 years of service",
    "employees with more than 3 years of service",
)
TEAMS = ("engineering teams", "support teams", "sales teams", "operations teams")
SYSTEMS = ("production systems", "staging systems", "corporate laptops", "contractor accounts")
SERVICES = ("tier-1 services", "tier-2 services", "internal tools")
TRAVEL = ("domestic travel", "regional travel", "intercontinental travel")
PURCHASES = ("goods purchases", "services purchases", "software purchases")
ALL = ("all employees",)
ORDERED_SCOPE_SETS = (ROLES, TENURE)  # values are sorted ascending along these


@dataclass(frozen=True)
class FactTemplate:
    dept: str
    topic: str
    subject: str
    scopes: tuple
    kind: str  # "int" | "req" (required / not required) | "allow" (allowed / not allowed)
    lo: int = 0
    hi: int = 0
    step: int = 1
    unit: str = ""
    max_scopes: int = 2  # how many scopes one document states for this template


TEMPLATES: list = []


def _topic(dept, topic, rows):
    for row in rows:
        subject, scopes, kind = row[:3]
        lo, hi, step, unit = row[3:] if len(row) > 3 else (0, 0, 1, "")
        TEMPLATES.append(FactTemplate(dept, topic, subject, scopes, kind, lo, hi, step, unit))


# ----------------------------------------------------------------------------- HR
_topic("hr", "Leave Policy", [
    ("annual leave entitlement", ROLES, "int", 10, 30, 1, "days per year"),
    ("maximum carry-forward of unused leave", ROLES, "int", 0, 15, 1, "days"),
    ("notice required for planned leave", ALL, "int", 3, 30, 1, "days"),
    ("medical certificate for sick leave", ALL, "req"),
])
_topic("hr", "Remote Work Policy", [
    ("required office attendance", TEAMS, "int", 0, 5, 1, "days per week"),
    ("home-office equipment stipend", ROLES, "int", 100, 1000, 50, "USD"),
    ("core collaboration window", ALL, "int", 3, 7, 1, "hours per day"),
    ("manager approval for working from another city", ALL, "req"),
])
_topic("hr", "Probation and Notice Policy", [
    ("probation period", ROLES, "int", 1, 6, 1, "months"),
    ("notice period after probation", ROLES, "int", 15, 90, 15, "days"),
    ("notice period during probation", ALL, "int", 7, 30, 1, "days"),
    ("exit interview", ALL, "req"),
])
_topic("hr", "Performance Review Policy", [
    ("number of review cycles", ALL, "int", 1, 4, 1, "per year"),
    ("rating appeal window", ALL, "int", 5, 30, 1, "days"),
    ("minimum time in role before promotion", ROLES, "int", 6, 36, 6, "months"),
    ("peer feedback", ALL, "req"),
])
_topic("hr", "Parental Leave Policy", [
    ("maternity leave", TENURE, "int", 12, 26, 1, "weeks"),
    ("paternity leave", TENURE, "int", 1, 12, 1, "weeks"),
    ("adoption leave", ALL, "int", 4, 16, 1, "weeks"),
    ("phased return to work", ALL, "allow"),
])
_topic("hr", "Mandatory Training Policy", [
    ("mandatory training time", ROLES, "int", 4, 40, 2, "hours per year"),
    ("security awareness refresher interval", ALL, "int", 3, 12, 1, "months"),
    ("deadline for harassment prevention training after joining", ALL, "int", 15, 90, 5, "days"),
    ("external certification reimbursement", ROLES, "int", 0, 2000, 100, "USD"),
])

# ----------------------------------------------------------------------------- IT
_topic("it", "Password and Access Policy", [
    ("minimum password length", ALL, "int", 8, 20, 1, "characters"),
    ("password rotation interval", SYSTEMS, "int", 30, 365, 15, "days"),
    ("multi-factor authentication", SYSTEMS, "req"),
    ("account lockout threshold", ALL, "int", 3, 10, 1, "failed attempts"),
])
_topic("it", "VPN and Remote Access Policy", [
    ("VPN session timeout", ALL, "int", 15, 720, 15, "minutes"),
    ("split tunnelling", ALL, "allow"),
    ("remote access from personal device", ROLES, "allow"),
    ("maximum concurrent VPN sessions", ROLES, "int", 1, 5, 1, "sessions"),
])
_topic("it", "Device Management Policy", [
    ("laptop refresh cycle", ROLES, "int", 2, 5, 1, "years"),
    ("personal device on the corporate network", ALL, "allow"),
    ("full-disk encryption", ALL, "req"),
    ("lost device reporting window", ALL, "int", 1, 48, 1, "hours"),
])
_topic("it", "Incident Response Policy", [
    ("P1 incident response time", SERVICES, "int", 5, 60, 5, "minutes"),
    ("P2 incident response time", SERVICES, "int", 1, 8, 1, "hours"),
    ("post-incident review deadline", ALL, "int", 2, 14, 1, "days"),
    ("customer notification deadline for data incidents", ALL, "int", 24, 96, 12, "hours"),
])
_topic("it", "Backup and Recovery Policy", [
    ("backup retention", SERVICES, "int", 7, 365, 7, "days"),
    ("recovery point objective", SERVICES, "int", 1, 24, 1, "hours"),
    ("restore test frequency", ALL, "int", 1, 12, 1, "months"),
    ("off-site backup copy", SERVICES, "req"),
])
_topic("it", "Software Procurement Policy", [
    ("software request approval time", ALL, "int", 2, 15, 1, "business days"),
    ("security review for new vendor", ALL, "req"),
    ("licence audit interval", ALL, "int", 3, 12, 1, "months"),
    ("approved spend without review", ROLES, "int", 100, 5000, 100, "USD"),
])

# ----------------------------------------------------------------------------- Finance
_topic("finance", "Expense Policy", [
    ("expense approval limit", ROLES, "int", 100, 10000, 100, "USD"),
    ("receipt threshold", ALL, "int", 10, 100, 5, "USD"),
    ("reimbursement submission deadline", ALL, "int", 15, 90, 5, "days"),
    ("alcohol reimbursement", ALL, "allow"),
])
_topic("finance", "Travel Policy", [
    ("hotel nightly cap", TRAVEL, "int", 80, 400, 10, "USD"),
    ("daily meal allowance", TRAVEL, "int", 20, 120, 5, "USD"),
    ("advance booking requirement", TRAVEL, "int", 3, 30, 1, "days"),
    ("business class travel", ROLES, "allow"),
])
_topic("finance", "Procurement Policy", [
    ("competitive quotes threshold", PURCHASES, "int", 1000, 25000, 500, "USD"),
    ("purchase order approval time", ALL, "int", 1, 10, 1, "business days"),
    ("vendor onboarding time", ALL, "int", 5, 30, 1, "business days"),
    ("conflict-of-interest declaration", ALL, "req"),
])
_topic("finance", "Invoicing and Payments Policy", [
    ("standard payment terms", ALL, "int", 15, 90, 15, "days"),
    ("late payment fee", ALL, "int", 1, 5, 1, "percent"),
    ("invoice approval limit", ROLES, "int", 500, 50000, 500, "USD"),
    ("advance payment to vendors", ALL, "allow"),
])
_topic("finance", "Corporate Card Policy", [
    ("monthly card limit", ROLES, "int", 500, 20000, 500, "USD"),
    ("statement reconciliation deadline", ALL, "int", 3, 21, 1, "days"),
    ("cash withdrawal", ALL, "allow"),
    ("card replacement time", ALL, "int", 2, 10, 1, "business days"),
])
_topic("finance", "Budget and Reporting Policy", [
    ("budget submission deadline before quarter start", ALL, "int", 10, 45, 1, "days"),
    ("variance explanation threshold", ALL, "int", 3, 15, 1, "percent"),
    ("forecast refresh frequency", TEAMS, "int", 1, 6, 1, "months"),
    ("mid-quarter budget reallocation", TEAMS, "allow"),
])

# ----------------------------------------------------------------------------- Engineering
_topic("engineering", "Code Review Policy", [
    ("number of approvals required per pull request", TEAMS, "int", 1, 3, 1, "approvals"),
    ("maximum pull request size", ALL, "int", 200, 1000, 50, "lines"),
    ("review turnaround target", ALL, "int", 4, 48, 4, "hours"),
    ("self-merge of approved change", TEAMS, "allow"),
])
_topic("engineering", "Release and Deployment Policy", [
    ("deployment freeze before major holidays", ALL, "int", 1, 10, 1, "days"),
    ("rollback decision window", SERVICES, "int", 5, 60, 5, "minutes"),
    ("canary traffic share", SERVICES, "int", 1, 25, 1, "percent"),
    ("Friday production deployment", SERVICES, "allow"),
])
_topic("engineering", "On-Call Policy", [
    ("on-call rotation length", TEAMS, "int", 1, 4, 1, "weeks"),
    ("alert acknowledgement time", SERVICES, "int", 5, 30, 5, "minutes"),
    ("compensatory time off per on-call week", TEAMS, "int", 0, 3, 1, "days"),
    ("secondary on-call engineer", SERVICES, "req"),
])
_topic("engineering", "Service Reliability Policy", [
    ("monthly error budget", SERVICES, "int", 5, 120, 5, "minutes"),
    ("availability review frequency", ALL, "int", 1, 6, 1, "months"),
    ("load test before launch", SERVICES, "req"),
    ("status page update interval during incidents", SERVICES, "int", 10, 60, 10, "minutes"),
])
_topic("engineering", "Logging and Data Retention Policy", [
    ("application log retention", SERVICES, "int", 14, 365, 7, "days"),
    ("audit log retention", ALL, "int", 1, 7, 1, "years"),
    ("personal data in logs", ALL, "allow"),
    ("log access review interval", ALL, "int", 1, 12, 1, "months"),
])
_topic("engineering", "Security Review Policy", [
    ("dependency scan frequency", SERVICES, "int", 1, 30, 1, "days"),
    ("penetration test frequency", ALL, "int", 1, 4, 1, "per year"),
    ("critical vulnerability fix deadline", SERVICES, "int", 1, 14, 1, "days"),
    ("security sign-off for new service", ALL, "req"),
])

TOPICS = list(dict.fromkeys((t.dept, t.topic) for t in TEMPLATES))

# ----------------------------------------------------------------------------- Boilerplate
PURPOSE = [
    "This document sets out how {dept_name} manages the {topic_lc} for employees in {region}.",
    "The purpose of this policy is to give employees a clear and consistent reference for the {topic_lc}.",
    "It exists so that managers and employees apply the same rules to the {topic_lc} across {region}.",
    "The rules below reflect statutory requirements in {region} and the company's internal risk guidance.",
    "{dept_name} owns this policy and reviews it as part of the annual governance cycle.",
    "Employees are expected to read this policy as part of their onboarding and whenever it changes.",
]
SCOPE = [
    "This policy applies to all permanent, fixed-term and contract staff working in {region}.",
    "Employees seconded to {region} from another country follow this policy for the duration of the assignment.",
    "Where a local statute in {region} is more generous than this policy, the statute prevails.",
    "Third-party contractors are covered only where their agreement explicitly refers to this policy.",
    "Staff on long-term assignment outside {region} should consult {dept_name} for the rules that apply to them.",
    "This policy does not replace individual employment contracts, which remain the primary source of terms.",
]
DETAILS = [
    "The following rules apply in {region}.",
    "The values below are in force for employees in {region}.",
    "The rules for {region} are set out below.",
]
EXCEPTIONS = [
    "Exceptions must be requested in writing and approved by the head of department and {dept_name}.",
    "Approved exceptions are recorded in the exceptions register and reviewed every quarter.",
    "Emergency exceptions may be granted verbally but must be confirmed in writing within two working days.",
    "Repeated exception requests for the same issue will trigger a review of the relevant rule.",
    "Disputes about how this policy applies are escalated to the {dept_name} lead for a final decision.",
    "No exception can override a legal or regulatory obligation in {region}.",
]
CONTACTS = [
    "Questions about this policy can be sent to {email}.",
    "The {dept_name} service desk is available during normal office hours on working days.",
    "Urgent matters outside office hours should be raised through the on-call channel for {dept_name}.",
    "Feedback on this policy is welcome and should be sent to {email}.",
]
POOLS = {
    "purpose": PURPOSE,
    "scope": SCOPE,
    "details": DETAILS,
    "exceptions": EXCEPTIONS,
    "contacts": CONTACTS,
}
REVISION_NOTES = [
    "Revision note: figures were reviewed and updated following the annual policy review.",
    "Revision note: this version replaces the previous issue and clarifies several thresholds.",
    "Revision note: updated to align with changes requested by the policy governance board.",
]
FAQ_FILLER = [
    ("Who owns this policy?", "{dept_name} owns this policy and answers questions about it."),
    ("Where can I find the full policy text?", "The full text is published on the intranet under the {dept_name} section."),
    ("Can my manager grant an exception on their own?", "No. Exceptions need written approval as described in the policy."),
    ("How often is this reviewed?", "It is reviewed at least once a year, or sooner if regulations change."),
    ("Does this apply to contractors?", "Only where their contract explicitly says so."),
    ("What if my local law differs?", "The more generous of the two applies in {region}."),
]
