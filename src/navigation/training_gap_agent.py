"""Privileged TRAIN-only teacher stopping-gap trial; canonical expert unchanged."""

from src.navigation.collection import training_light_gap
from src.navigation.collection_agent import CollectionExpert


def get_entry_point():
    return "TrainingGapExpert"


class TrainingGapExpert(CollectionExpert):
    def collection_metadata(self, metadata):
        metadata = super().collection_metadata(metadata)
        gap = training_light_gap(self.navigation_spec)
        if gap is None:
            raise ValueError("TrainingGapExpert requires an explicit guarded light-gap trial")
        return dict(
            metadata,
            training_light_gap_m=gap,
            driver="privileged_pdm_lite_training_gap",
            standard_light_gap_m=float(self.config.idm_red_light_minimum_distance),
        )

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self.original_gap = self.config.idm_red_light_minimum_distance
        self.config.idm_red_light_minimum_distance = training_light_gap(self.navigation_spec)
        self.driver_label = (
            f"PRIVILEGED TRAIN EXPERT: GAP{self.config.idm_red_light_minimum_distance:g}m"
        )
        self.trace_policy_kind = "privileged_pdm_lite_training_gap"

    def extra_facts(self):
        return dict(
            training_light_gap_m=float(self.config.idm_red_light_minimum_distance),
            light_gap_semantics=(
                "PDM-Lite IDM standstill gap relative to planner reference; "
                "not painted/evaluator line or verified visibility"
            ),
        )
