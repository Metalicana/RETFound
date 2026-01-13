import json
import os

class EquityRiskTool:
    def __init__(self, data_path="data/equity_stats.json"):
        # Dynamic Loading: If your interns update the JSON, the agent gets smarter automatically.
        try:
            with open(data_path, 'r') as f:
                self.risk_db = json.load(f)
            print(f"[EquityTool] Loaded risk stats from {data_path}")
        except FileNotFoundError:
            print(f"[EquityTool] CRITICAL: {data_path} not found. Using empty DB.")
            self.risk_db = {}

    def check_risk(self, disease, race, age=None):
        # 1. Normalize Race Input
        race_key = "White" # Default
        r_low = race.lower()
        if "black" in r_low or "african" in r_low: race_key = "Black"
        elif "hispanic" in r_low or "latino" in r_low: race_key = "Hispanic"
        elif "asian" in r_low or "chinese" in r_low: race_key = "Asian"
        
        # 2. Normalize Disease Input
        d_key = None
        dis_low = disease.lower()
        if "glaucoma" in dis_low: d_key = "Glaucoma"
        elif "diabet" in dis_low or "dr" in dis_low: d_key = "DR"
        elif "amd" in dis_low or "macular" in dis_low: d_key = "AMD"

        if not d_key:
            return {"status": "skipped", "reason": "Disease not in equity database."}

        # 3. Lookup
        stats = self.risk_db.get(d_key, {}).get(race_key)
        
        if not stats:
            return {
                "risk_level": "BASELINE", 
                "message": f"No specific disparity stats for {race_key} with {d_key}."
            }

        # 4. Add Age Context (Logic Layer on top of Data)
        response = stats.copy()
        if d_key == "Glaucoma" and race_key == "Black" and age and age < 50:
             response["URGENT_ALERT"] = "Early Onset Glaucoma is aggressive in this group."

        return response