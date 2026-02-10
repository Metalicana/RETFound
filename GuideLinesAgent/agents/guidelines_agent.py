from tools.evidence.orchestrator import retrieve_guidelines
import json

class GuidelinesAgent:
    def __init__(self):
        self.name = "GuidelinesAgent"

    # input: clinical question / suspected diagnosis from other agents
    def consult(self, query: str, max_results: int = 5):
        print(f"[{self.name}] Consulting for query: '{query}'...")
        evidence = retrieve_guidelines(query)
        evidence["evidence"] = evidence.get("evidence", [])[:max_results]

        return evidence


# test
if __name__ == "__main__":
    agent = GuidelinesAgent()

    query = "Normal-tension glaucoma IOP management treatment" # LET RECEIVE QUERY FROM OTHER AGENTS, currently simple query for testing
    results = agent.consult(query, max_results=5)

    print("\n--- Retrieved Evidence ---")
    print(json.dumps(results, indent=2))
