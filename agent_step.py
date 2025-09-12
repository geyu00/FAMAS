import json
from typing import Optional


class AgentStep:
    def __init__(self, agent: str, action: str, label: str):
        self.agent = agent
        self.action = action
        self.label = label

    # def __init__(self, agent: str, action: str):
    #     self.agent = agent
    #     self.action = action
    #     self.label = ""

    def __eq__(self, other):
        if not isinstance(other, AgentStep):
            return NotImplemented
        return (
            self.agent == other.agent and
            self.action == other.action and
            self.label == other.label
        )

    def __hash__(self):
        return hash((self.agent, self.action, self.label))

    # def __repr__(self):
    #     return f"{self.agent}###{self.action}###{self.label}"
    
    def __repr__(self):
        return json.dumps({
            "agent": self.agent,
            "action": self.action,
            "label": self.label
        })

    @staticmethod
    def from_string(step_str: str) -> Optional["AgentStep"]:
        parts = step_str.strip().split("###")
        if len(parts) < 4:
            return None
        return AgentStep(parts[1].strip(), parts[2].strip(), parts[3].strip())
    
    @staticmethod
    def from_jsonl_line(line: str, is_hierarchical=False) -> Optional["AgentStep"]:
        try:
            obj = json.loads(line)
            if not is_hierarchical:
                return AgentStep(obj["agent"], obj["action"], obj["label"])
            return AgentStep(obj["agent"], obj["action"], "")
        except Exception:
            return None
