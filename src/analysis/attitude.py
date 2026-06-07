import numpy as np

from .monitor import AnalysisMonitor
from ..maths.rotations import A_from_q


class BodyAxisAlignmentMonitor(AnalysisMonitor):
    def __init__(
        self,
        name="body_axis_alignment",
        body_axis=np.array([1.0, 0.0, 0.0]),
        reference_axis_eci=np.array([1.0, 0.0, 0.0]),
        use_absolute=True,
    ):
        super().__init__(name)
        self.body_axis = self._unit_vector(body_axis, "body_axis")
        self.reference_axis_eci = self._unit_vector(
            reference_axis_eci, "reference_axis_eci"
        )
        self.use_absolute = bool(use_absolute)
        self.times = []
        self.alignments = []

    @property
    def mean_alignment(self):
        return float(np.mean(self.alignments)) if self.alignments else 0.0

    def sample(self, sat):
        body_to_eci = A_from_q(sat.quaternion).T
        body_axis_eci = body_to_eci @ self.body_axis
        alignment = float(np.dot(self.reference_axis_eci, body_axis_eci))
        if self.use_absolute:
            alignment = abs(alignment)

        self.times.append(float(sat.time))
        self.alignments.append(alignment)

    def _unit_vector(self, value, name):
        vector = np.asarray(value, dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError(f"{name} must be non-zero.")
        return vector / norm
