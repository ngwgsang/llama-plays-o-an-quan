import unittest

from o_an_quan.prompts import CLASSIFICATION_PROMPT, DECISION_PROMPT, PERSONAS, RULES, SYSTEM_PROMPT

PAPER_PERSONAS = {
    "BalancedAgent", "DefensiveAgent", "QuickPlayAgent", "RiskTakerAgent", "StrategicControlAgent",
}


class TestPrompts(unittest.TestCase):
    def test_all_paper_personas_are_loaded(self):
        self.assertTrue(PAPER_PERSONAS <= set(PERSONAS))
        for name in PAPER_PERSONAS:
            self.assertTrue(PERSONAS[name].startswith(f"# {name}"))

    def test_rules_cover_extra_rules(self):
        for rule in ["E1", "E2", "E3", "E4", "E5"]:
            self.assertIn(rule, RULES)

    def test_templates_format(self):
        prompt = DECISION_PROMPT.format(
            team="A", persona=PERSONAS["BalancedAgent"], game_state="<G>",
            available_positions="A1, A2", history="<H>", rules=RULES,
        )
        for section in ["**Persona**", "**Game States**", "**History**", "**Game Rules**", "**Task**"]:
            self.assertIn(section, prompt)
        self.assertIn("<think-text>", CLASSIFICATION_PROMPT.format(think="<think-text>"))
        self.assertIn("Ô Ăn Quan", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
