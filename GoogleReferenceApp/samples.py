SAMPLE_MEDICAL_HISTORY = """PATIENT MEDICAL HISTORY SUMMARY
Patient Name: Jordan Taylor (Age 36, California Resident)
Primary Care Physician: Dr. Maya Patel, MD

1. CHRONIC & PRE-EXISTING CONDITIONS
- Type 2 Diabetes Mellitus (diagnosed 2022, controlled with medication)
- Mild Persistent Asthma (diagnosed 2018)
- Seasonal Allergic Rhinitis

2. CURRENT PRESCRIPTION MEDICATIONS
- Metformin 500mg twice daily (Generic Tier 1)
- Insulin Glargine basal pen (Preferred Brand Tier 2, monthly refill)
- Albuterol HFA rescue inhaler (as needed)

3. ANTICIPATED CARE & UTILIZATION (NEXT 12 MONTHS)
- Quarterly Endocrinologist specialist visits (4 visits/year)
- Bi-annual HbA1c diagnostic lab work and annual diabetic retinal screening
- Annual preventive wellness exam and flu immunization
- Outpatient mental health therapy (bi-weekly telehealth sessions)"""

SAMPLE_FINANCIAL_BACKGROUND = """HOUSEHOLD FINANCIAL BACKGROUND & W-2 INCOME SUMMARY
Tax Filing Year: 2026 (Single Filer, Household Size: 1)
Annual Gross Income: $74,000
Monthly Net Take-Home Pay: $4,850
Monthly Fixed Obligations (Rent, Utilities, Student Loan, Groceries): $3,650
Monthly Disposable Healthcare Budget: $1,200
Emergency Savings & HSA Balance: $6,200"""

SAMPLE_DOCUMENTS = [
    {
        "id": "sample-aca-silver-v1",
        "label": "ACA Silver 70 PPO Health Plan (v1 — Standard)",
        "fileName": "PacificCare_Silver_70_PPO_2026_v1.txt",
        "rawText": """SUMMARY OF BENEFITS AND COVERAGE — PACIFICCARE SILVER 70 PPO (PLAN YEAR 2026)

This Health Insurance Policy ("Policy") is issued effective January 1, 2026 ("Effective Date") and expires December 31, 2026 ("Expiration Date"), by PacificCare Health Plan, Inc., located at 400 Capitol Mall, Suite 900, Sacramento, CA 95814 ("Insurer"), to Enrolled Member ("Policyholder").

1. PREMIUM, DEDUCTIBLE & OUT-OF-POCKET LIMITS
1.1 Monthly Premium: $410.00 per month ($4,920 annually), due on the 1st of each month. Automatic renewal applies on January 1, 2027 unless written notice of non-renewal is delivered 30 days prior (deadline: December 2, 2026).
1.2 Annual Deductible: $4,500 Individual In-Network ($9,000 Out-of-Network).
1.3 Out-of-Pocket Maximum: $8,700 Individual In-Network. Once reached, Insurer pays 100% of covered essential health benefits.

2. COVERED MEDICAL SERVICES & ESSENTIAL HEALTH BENEFITS
2.1 Preventive Care: All ACA-mandated preventive wellness exams, immunizations, and routine screenings are covered at $0 copay (100% covered) with no deductible when using In-Network providers.
2.2 Primary Care & Specialist Visits: Primary care office visits are subject to a $40 copay (deductible waived). Specialist visits including Endocrinology and Pulmonology require a $75 copay.
2.3 Chronic Condition & Pre-Existing Coverage: Pre-existing conditions including Type 2 Diabetes, asthma, hypertension, and cardiovascular disease are fully covered from day one with no waiting period and no lifetime dollar limits.
2.4 Mental Health & Telehealth: Outpatient mental health therapy and behavioral health counseling are covered on parity with medical benefits at $0 telehealth copay or $40 in-person copay.

3. PRESCRIPTION DRUG FORMULARY & COST SHARE
3.1 Generic Medications (Tier 1): Covered at $15 copay per 30-day supply with deductible waived.
3.2 Preferred Brand & Insulin Cap (Tier 2): Covered at $45 copay; insulin prescriptions are capped at $35 per 30-day supply.
3.3 Specialty Drugs (Tier 4): Specialty biologics require 30% coinsurance after deductible and mandatory step therapy (fail-first) plus prior authorization before dispensing.

4. PRIOR AUTHORIZATION, NETWORK & EXCLUSIONS
4.1 Prior Authorization Penalties: Failure to obtain prior authorization at least 72 hours before non-emergency MRI imaging or outpatient surgery results in a 50% benefit penalty charged directly to Policyholder.
4.2 Out-of-Network Lab Work: Out-of-network diagnostic laboratory testing is covered at 50% coinsurance after deductible, and Policyholder may be subject to balance billing above Usual and Customary rates.
4.3 Surprise Billing Exception: Notwithstanding Section 4.2, emergency services and ancillary lab/radiology services performed at an In-Network facility are protected under the federal No Surprises Act at In-Network cost-sharing only.

5. GOVERNING LAW & APPEALS
5.1 This Policy is governed by the laws of the State of California and federal Affordable Care Act (ACA) regulations. Internal and independent external appeals are available within 180 days of an adverse benefit determination.

IN WITNESS WHEREOF, PacificCare Health Plan, Inc. executes this Policy as of January 1, 2026.
INSURER: PacificCare Health Plan, Inc.
Signed: /s/ R. Alvarez, Chief Underwriting Officer
Date: January 1, 2026
POLICYHOLDER: Jordan Taylor
Signed: /s/ Jordan Taylor
Date: January 1, 2026""",
    },
    {
        "id": "sample-aca-silver-v2",
        "label": "ACA Silver 70 PPO Health Plan (v2 — Lower Deductible)",
        "fileName": "PacificCare_Silver_70_PPO_2026_v2.txt",
        "rawText": """SUMMARY OF BENEFITS AND COVERAGE — PACIFICCARE SILVER 70 PPO (REVISED 2026)

This Health Insurance Policy ("Policy") is issued effective January 1, 2026 ("Effective Date") and expires December 31, 2026 ("Expiration Date"), by PacificCare Health Plan, Inc., located at 400 Capitol Mall, Suite 900, Sacramento, CA 95814 ("Insurer"), to Enrolled Member ("Policyholder").

1. PREMIUM, DEDUCTIBLE & OUT-OF-POCKET LIMITS
1.1 Monthly Premium: $350.00 per month ($4,200 annually), due on the 1st of each month.
1.2 Annual Deductible: $2,500 Individual In-Network ($6,000 Out-of-Network).
1.3 Out-of-Pocket Maximum: $6,500 Individual In-Network. Once reached, Insurer pays 100% of covered essential health benefits.

2. COVERED MEDICAL SERVICES & ESSENTIAL HEALTH BENEFITS
2.1 Preventive Care: All ACA-mandated preventive wellness exams, immunizations, and routine screenings are covered at $0 copay (100% covered) with no deductible.
2.2 Primary Care & Specialist Visits: Primary care office visits are $25 copay (deductible waived). Specialist visits including Endocrinology are $50 copay (deductible waived).
2.3 Chronic Condition & Pre-Existing Coverage: Pre-existing conditions including Type 2 Diabetes, asthma, hypertension, and maternity care are fully covered from day one with zero waiting period and zero lifetime or annual dollar limits.
2.4 Mental Health & Telehealth: Outpatient mental health therapy and telehealth visits are covered at $0 copay for the first 12 visits and $25 thereafter.

3. PRESCRIPTION DRUG FORMULARY & COST SHARE
3.1 Generic Medications (Tier 1): Covered at $10 copay per 30-day supply with deductible waived.
3.2 Preferred Brand & Insulin Cap (Tier 2): Covered at $35 copay; insulin and diabetes testing supplies are capped at $35 per 30-day supply with deductible waived.

4. PRIOR AUTHORIZATION & NO SURPRISES PROTECTION
4.1 Prior Authorization: Standard prior authorization applies to inpatient admissions; diagnostic outpatient bloodwork and routine imaging require no prior authorization.
4.2 No Surprises Act Protection: Emergency services and ancillary laboratory/radiology services are always billed at In-Network cost-sharing rates with zero balance billing.

5. GOVERNING LAW & APPEALS
5.1 This Policy is governed by the laws of the State of California and federal ACA regulations.

IN WITNESS WHEREOF, PacificCare Health Plan, Inc. executes this Policy as of January 1, 2026.
INSURER: PacificCare Health Plan, Inc.
Signed: /s/ R. Alvarez, Chief Underwriting Officer
Date: January 1, 2026
POLICYHOLDER: Jordan Taylor
Signed: /s/ Jordan Taylor
Date: January 1, 2026""",
    },
    {
        "id": "sample-short-term",
        "label": "Short-Term Limited Duration Medical Plan (High Risk)",
        "fileName": "LibertyGuard_ShortTerm_Medical_2026.txt",
        "rawText": """SHORT-TERM LIMITED DURATION MEDICAL INSURANCE CERTIFICATE

THIS POLICY IS NOT ACA-COMPLIANT MINIMUM ESSENTIAL COVERAGE UNDER FEDERAL LAW.
INSURER: LibertyGuard Indemnity Co. (Address not specified)
POLICYHOLDER: Jordan Taylor

1. PREMIUM, DEDUCTIBLE & LIFETIME CAP
1.1 Monthly Premium: $215.00 per month ($2,580 annually). Policy automatically renews at Insurer's then-current rate unless cancelled 60 days prior.
1.2 Annual Deductible: $10,000 Individual Deductible. Policyholder pays 50% coinsurance after deductible for all covered hospital and specialist claims.
1.3 Out-of-Pocket & Benefit Cap: Total Plan Lifetime Maximum Benefit is strictly capped at $250,000. Expenses exceeding $250,000 are excluded and paid 100% by Policyholder.

2. PRE-EXISTING CONDITION EXCLUSIONS & WAITING PERIODS
2.1 Pre-Existing Condition Exclusion: Any medical condition, illness, injury, diabetes, asthma, hypertension, or prescription medication diagnosed or treated within five (5) years prior to the Effective Date is strictly excluded for the entire policy term.
2.2 Waiting Period: A 12-month waiting period applies to outpatient surgery, imaging, and diagnostic specialist consultations.

3. EXCLUDED ESSENTIAL BENEFITS & PRESCRIPTIONS
3.1 Prescription Drugs Excluded: Outpatient prescription medications including insulin, inhalers, and specialty biologics are NOT covered under this Policy.
3.2 Mental Health & Maternity Excluded: Mental health counseling, substance use treatment, preventive wellness exams, and maternity care are completely excluded.
3.3 Balance Billing & Out-of-Network: There is zero out-of-network coverage; Policyholder is responsible for 100% of out-of-network hospital charges and balance billing.

INSURER: LibertyGuard Indemnity Co.
Signed: /s/ D. Mercer
Date: [BLANK]
POLICYHOLDER: [UNSIGNED]
Date: [BLANK]""",
    },
    {
        "id": "sample-medical-bill",
        "label": "Non-Policy Document (Outpatient Clinic Invoice)",
        "fileName": "Metro_Diagnostics_Medical_Bill.txt",
        "rawText": """PATIENT BILLING STATEMENT — INVOICE #MED-99214
Provider: Metro Diagnostics Outpatient Lab
720 Medical Center Pkwy, Austin, TX 78701
Patient: Jordan Taylor
Service Date: October 12, 2026
Payment Status: Paid in Full via HSA Card ****8104

Itemized Charges:
- Comprehensive Metabolic Panel (CPT 80053) = $145.00
- Hemoglobin A1C Screening (CPT 83036) = $65.00
Subtotal: $210.00
Amount Due: $0.00 (Paid in Full)

Thank you for visiting Metro Diagnostics. Contact billing@metrodiag.example with questions.""",
    },
]
