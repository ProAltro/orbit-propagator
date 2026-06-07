import numpy as np

from .base import Component


DEFAULT_FACE_NORMALS_BODY = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
)
DEFAULT_DRAG_AREAS_M2 = np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1])
DEFAULT_SRP_AREAS_M2 = DEFAULT_DRAG_AREAS_M2.copy()


class BodySurfaceModel(Component):
    def __init__(
        self,
        normals_body=DEFAULT_FACE_NORMALS_BODY,
        drag_areas_m2=DEFAULT_DRAG_AREAS_M2,
        srp_areas_m2=None,
        drag_coefficient=2.2,
        specular=0.3,
        diffuse=0.3,
        name="body_surfaces",
    ):
        super().__init__(name)
        self.normals_body = np.asarray(normals_body, dtype=float)
        self.drag_areas_m2 = np.asarray(drag_areas_m2, dtype=float)
        self.srp_areas_m2 = np.asarray(
            self.drag_areas_m2 if srp_areas_m2 is None else srp_areas_m2,
            dtype=float,
        )
        self.drag_coefficient = float(drag_coefficient)
        self.specular = self._as_face_array(specular, "specular")
        self.diffuse = self._as_face_array(diffuse, "diffuse")

        if self.normals_body.ndim != 2 or self.normals_body.shape[1] != 3:
            raise ValueError("normals_body must have shape (N, 3).")
        if self.drag_areas_m2.shape != (len(self.normals_body),):
            raise ValueError("drag_areas_m2 must match normals_body length.")
        if self.srp_areas_m2.shape != (len(self.normals_body),):
            raise ValueError("srp_areas_m2 must match normals_body length.")

        norms = np.linalg.norm(self.normals_body, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("surface normals must be non-zero.")
        self.normals_body = self.normals_body / norms[:, None]

    def _as_face_array(self, value, label):
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            return np.full(len(self.normals_body), float(values))
        if values.shape != (len(self.normals_body),):
            raise ValueError(f"{label} must be a scalar or match normals_body length.")
        return values

    def projected_drag_area(self, direction_body):
        direction_body = np.asarray(direction_body, dtype=float)
        norm = np.linalg.norm(direction_body)
        if norm == 0.0:
            return 0.0

        unit_direction = direction_body / norm
        projections = self.normals_body @ (-unit_direction)
        mask = projections > 0.0
        return float(np.sum(projections[mask] * self.drag_areas_m2[mask]))
