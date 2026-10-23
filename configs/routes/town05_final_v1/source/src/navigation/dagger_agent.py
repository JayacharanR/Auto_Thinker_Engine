"""DAgger training rollout; labeled privileged assistance, not camera evaluation."""

import json

from src.navigation.collection_agent import CollectionExpert, controls
from src.navigation.dagger import CorrectionMixer
from src.navigation.policy_ipc import PolicyClient
from src.navigation.runtime import atomic_json


def get_entry_point():
    return "CorrectionCollector"


class CorrectionCollector(CollectionExpert):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self.driver_label = "DAGGER TRAINING | PRIVILEGED ASSISTANCE ON"
        self.trace_policy_kind = "dagger_camera_with_privileged_expert"
        self.mixer = CorrectionMixer()
        self.cached_packet = None
        self.student_response = None
        self.selected_intervention = None
        self.correction_trace = (self.out / "dagger_trace.jsonl").open("w")
        self.client = PolicyClient(
            self.navigation_spec["student"]["checkpoint"],
            self.out / "policy_stderr.log",
            rule_overrides=False,
        )

    def collection_locations(self, split, entry):
        return set(
            self.navigation_spec["collection"]["dagger_guard"]["allowed_junctions"][entry["town"]]
        )

    def collection_metadata(self, metadata):
        guard = self.navigation_spec["collection"]["dagger_guard"]
        return {
            **metadata,
            "driver": "dagger_camera_with_privileged_expert",
            "mode": "dagger",
            "policy_checkpoint_sha256": self.navigation_spec["student"]["sha256"],
            "dagger_guard_sha256": guard["guard_sha256"],
            "internal_manifest_sha256": guard["internal_manifest_sha256"],
            "trace_control_semantics": "trace.jsonl=expert proposal; "
            "dagger_trace.jsonl=student/selected; shards=next-tick executed",
            "perturbation_policy": "Student with privileged teacher braking/hazard control; "
            "3s teacher recovery after4s stationary teacher-go; collection only",
            "future_targets": "Instantaneous expert correction, no counterfactual future",
        }

    def packet(self, input_data, timestamp):
        if self.cached_packet is None or self.cached_packet.frame != int(input_data["front"][0]):
            self.cached_packet = super().packet(input_data, timestamp)
        return self.cached_packet

    def select_control(self, control, input_data, timestamp):
        packet = self.packet(input_data, timestamp)  # guard before every20Hz policy call
        response = self.client.step(packet)
        teacher = controls(control)
        hazard = any(
            self.teacher_data[k]
            for k in ("vehicle_hazard", "walker_hazard", "light_hazard", "stop_sign_hazard")
        )
        selected, reason = self.mixer.select(
            timestamp,
            response["control"],
            teacher,
            float(packet.ego_motion[0]),
            float(self.teacher_data["target_speed"]),
            bool(hazard),
        )
        self.student_response, self.selected_intervention = response, reason
        self.correction_trace.write(
            json.dumps(
                {
                    "frame": packet.frame,
                    "timestamp": timestamp,
                    "student_requested_control": response["control"],
                    "teacher_control": teacher,
                    "selected_control": selected,
                    "intervention": reason,
                    "teacher_target_speed_mps": float(self.teacher_data["target_speed"]),
                    "student_target_speed_mps": response["target_speed_mps"],
                    "student_brake_probability": response["brake_probability"],
                    "ego_speed_mps": float(packet.ego_motion[0]),
                },
                allow_nan=False,
            )
            + "\n"
        )
        if self.frame_count % 20 == 0:
            self.correction_trace.flush()
            path = self.out.parent.parent / "progress.json"
            progress = json.loads(path.read_text())
            progress.update(
                stage="M2 DAgger collection",
                collection_samples=self.collector.writer.samples,
                privileged_intervention_ticks=sum(
                    v for k, v in self.mixer.counts.items() if k.startswith("teacher_")
                ),
                training_mixture_not_camera_evaluation=True,
            )
            atomic_json(path, progress)
        control.steer, control.throttle, control.brake = selected
        return control, reason

    def extra_facts(self):
        planner = self._waypoint_planner
        light = planner.next_traffic_lights[planner.route_index]
        return {
            "student_requested_control": self.student_response["control"],
            "student_target_speed_mps": self.student_response["target_speed_mps"],
            "student_brake_probability": self.student_response["brake_probability"],
            "selected_control_source": self.selected_intervention or "student",
            "expert_label_valid": True,
            "instantaneous_expert_correction_only": True,
            "applicable_light_id": light.id if light is not None else None,
            "signal_visibility_valid": None,
            "counterfactual_safe_label": False,
        }

    def destroy(self, results=None):
        if getattr(self, "client", None):
            self.client.close()
        if getattr(self, "correction_trace", None):
            self.correction_trace.close()
        super().destroy(results)
        if not getattr(self, "out", None):
            return
        path = self.out / "collection_profile.json"
        if path.exists() and getattr(self, "mixer", None):
            profile = json.loads(path.read_text())
            counts = dict(self.mixer.counts)
            profile.update(
                dagger_control_ticks=counts,
                privileged_intervention_ticks=sum(
                    v for k, v in counts.items() if k.startswith("teacher_")
                ),
                safety_stop_ticks=counts.get("teacher_braking", 0)
                + counts.get("teacher_hazard", 0),
                training_mixture_not_camera_evaluation=True,
            )
            atomic_json(path, profile)
