"""Strict high-level intent boundary. No motor values, commands, or code fields."""
import json
import re
import uuid
from datetime import datetime, timezone
from .config import HERE


class TaskError(ValueError):
    pass


def prompt_text(value):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 1000:
        raise TaskError("Enter an instruction between 1 and 1,000 characters.")
    if any(ord(c) < 32 and c not in "\n\t" for c in value):
        raise TaskError("Control characters are not accepted.")
    return value.strip()


def validate_intent(value):
    schema = json.loads((HERE / "intent.schema.json").read_text())
    def check(obj, spec, path):
        if spec["type"] == "object":
            if not isinstance(obj, dict) or set(obj) != set(spec["required"]):
                raise TaskError(f"Invalid fields at {path}.")
            for key, child in spec["properties"].items():
                check(obj[key], child, f"{path}.{key}")
        elif not isinstance(obj, str) or not spec.get("minLength", 0) <= len(obj) <= spec.get("maxLength", 500):
            raise TaskError(f"Invalid text at {path}.")
        if "enum" in spec and obj not in spec["enum"]:
            raise TaskError(f"Unsupported value at {path}.")
    check(value, schema, "task")
    if (value["decision"] == "accepted") != (value["task_type"] != "unsupported"):
        raise TaskError("Inconsistent task decision.")
    if value["decision"] == "accepted" and value["target"]["object"] == "unknown":
        raise TaskError("An accepted task needs an identifiable target.")
    return value


def build_task(intent, prompt):
    value = validate_intent(intent)
    if value["decision"] != "accepted":
        raise TaskError("Unsupported task: " + value["reason"])
    kind = value["task_type"]
    # The model chooses a high-level task; the server owns the workflow vocabulary.
    stages = (["Observe target", "Review grasp", "Approach", "Grasp", "Lift", "Verify hold"]
              if kind == "pick_up" else ["Observe target", "Review both views", "Report observations"])
    criteria = (["Operator verifies the intended object is held above the surface.",
                 "Fresh camera evidence and fault-free follower feedback are required."]
                if kind == "pick_up" else ["Operator verifies target visibility in fresh camera views."])
    return {"id": uuid.uuid4().hex, "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(), "prompt": prompt,
            "task_type": kind, "target": dict(value["target"]),
            "action": "Pick up" if kind == "pick_up" else "Inspect",
            "stages": stages, "completion_criteria": criteria, "reason": value["reason"],
            "requires_operator_approval": True, "execution_enabled": False,
            "perception_verified": False}


def mock_intent(prompt):
    """Explicit simulation only, never a silent fallback for unavailable Codex."""
    s = prompt.lower().strip().rstrip(".!?")
    match = re.fullmatch(r"(pick up|inspect|look at) (?:the |a )?(?:(blue|red|yellow|green) )?(block|cube|object)", s)
    return {"decision": "accepted" if match else "rejected",
            "task_type": ("pick_up" if match[1] == "pick up" else "inspect") if match else "unsupported",
            "target": {"object": match[3] if match else "unknown", "color": (match[2] or "unspecified") if match else "unspecified"},
            "reason": "Simulated intent for UI testing; no Codex call." if match else "Only a single block pickup or inspection is supported."}
