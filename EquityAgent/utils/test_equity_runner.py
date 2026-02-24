from __future__ import annotations

import json
from equity_agent import EquityAgent


def make_fake_patients():
    # 10 dummy examples with varied demographics for testing
    return [
        {"id": "P001", "race": "Black", "sex": "Female", "age": 68, "location": "Urban", "imaging_findings": "optic_disc_cupping"},
        {"id": "P002", "race": "White", "sex": "Male", "age": 55, "location": "Rural", "imaging_findings": "normal"},
        {"id": "P003", "race": "Asian", "sex": "Female", "age": 72, "location": "Urban", "imaging_findings": "macular_drusen"},
        {"id": "P004", "race": "Hispanic", "sex": "Male", "age": 45, "location": "Suburban", "imaging_findings": "optic_disc_asymmetry"},
        {"id": "P005", "race": "Black", "sex": "Male", "age": 59, "location": "Urban", "imaging_findings": "retinal_hemorrhage"},
        {"id": "P006", "race": "White", "sex": "Female", "age": 32, "location": "Rural", "imaging_findings": "normal"},
        {"id": "P007", "race": "Native American", "sex": "Female", "age": 66, "location": "Rural", "imaging_findings": "peripapillary_atrophy"},
        {"id": "P008", "race": "Black", "sex": "Female", "age": 80, "location": "Urban", "imaging_findings": "narrow_angles"},
        {"id": "P009", "race": "Asian", "sex": "Male", "age": 50, "location": "Suburban", "imaging_findings": "epiretinal_membrane"},
        {"id": "P010", "race": "White", "sex": "Male", "age": 74, "location": "Urban", "imaging_findings": "advanced_cup_to_disc_ratio"},
    ]


def main():
    patients = [make_fake_patients()[0]]

    agent = EquityAgent()

    print("EquityAgent running...")
    results = agent.analyze_patients(patients)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
