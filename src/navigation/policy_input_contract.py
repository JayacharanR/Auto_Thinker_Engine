"""Version declarations for the camera-only packet and optional own actuator status."""

LEGACY_SCHEMA = "navigation-sensor-packet-v1"
ACTUATOR_SCHEMA = "navigation-ego-action-step-v2"
SUPPORTED_SCHEMAS = (LEGACY_SCHEMA, ACTUATOR_SCHEMA)
ACTUATOR_FIELDS = ("observation_frame", "action_frame", "steer", "throttle", "brake")


def policy_input_schema(state):
    schema = state.get("policy_input_schema", LEGACY_SCHEMA)
    if type(schema) is not str or schema not in SUPPORTED_SCHEMAS:
        raise ValueError("Unsupported declared policy input schema")
    return schema


def validate_config_schema(config, schema):
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError("Unsupported declared policy input schema")
    if schema == LEGACY_SCHEMA:
        return
    extension = config.get("policy_input_extensions", {}).get(schema, {})
    if (
        extension.get("input") != "ego_action_feedback"
        or extension.get("fields") != list(ACTUATOR_FIELDS)
        or extension.get("identical_for") != ["constant_velocity", "rssm", "jepa"]
        or extension.get("other_actor_teacher_world_inputs") is not False
        or extension.get("user_approved") is not True
    ):
        raise ValueError("Own actuator input requires its exact approved versioned declaration")
