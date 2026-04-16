from __future__ import annotations

import json
from EquityAgent.equity_agent import EquityAgent


def make_patient_summaries():
    return [
        (
            "Patient P003 is a 72-year-old Asian female living in an urban area with delayed retina follow-up because specialty access has been inconsistent. "
            "Current symptoms include gradually reduced central vision, distortion while reading, and nighttime glare. "
            "Mirage imaging findings note bilateral macular drusen, pigmentary change near the fovea, and no convincing diabetic retinopathy pattern. "
            "Retfound flags macular drusen with moderate-to-high AMD concern and mild optic disc asymmetry, but does not strongly support proliferative retinopathy or advanced glaucomatous loss. "
            "Earlier upstream models reported mildly reduced visual acuity, family history of age-related macular degeneration, no diabetes, and no documented prior glaucoma diagnosis."
        ),
        (
            "Patient P011 is a 64-year-old Black male from a rural area presenting for evaluation of progressive peripheral vision loss and intermittent headaches. "
            "Mirage describes enlarged peripapillary atrophy, and subtle nerve fiber layer thinning. "
            "Retfound marks only low-to-moderate glaucoma concern despite optic nerve cupping and gives no strong AMD or diabetic retinopathy signal. "
            "Prior models and chart context note elevated intraocular pressure at an outside screening visit, transportation barriers, and missed ophthalmology follow-up over the last year."
        ),
    ]


def main():
    agent = EquityAgent()
    patient_summaries = make_patient_summaries()[1] # try 0 or 1

#    print("EquityAgent running...")
    results = agent.analyze_patients(patient_summaries, output_format="text")

    print(results)


if __name__ == "__main__":
    main()

