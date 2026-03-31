from types import SimpleNamespace

from app.routing import decide_route, normalize_route


def test_normalize_route_handles_supported_labels():
    assert normalize_route("technical_specialist") == "technical_specialist"
    assert normalize_route("Compliance Auditor") == "compliance_auditor"
    assert normalize_route("support concierge") == "support_concierge"


def test_decide_route_returns_model_selected_route():
    class RecordingModel:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="technical_specialist")

    model = RecordingModel()
    route_name = decide_route("Which expert should handle this?", model)

    assert route_name == "technical_specialist"
    assert model.messages is not None
    assert model.messages[-1].content.endswith("Which expert should handle this?")


def test_decide_route_normalizes_non_canonical_model_output():
    class StubModel:
        def invoke(self, _messages):
            return SimpleNamespace(content="The best route is Compliance Auditor")

    route_name = decide_route("Can Tier 1 accounts use crypto?", StubModel())

    assert route_name == "compliance_auditor"


def test_decide_route_handles_list_based_model_content():
    class StubModel:
        def invoke(self, _messages):
            return SimpleNamespace(content=[{"text": "support_concierge"}])

    route_name = decide_route("Explain the onboarding process simply.", StubModel())

    assert route_name == "support_concierge"
