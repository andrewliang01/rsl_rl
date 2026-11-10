"""Deployment-input-only actual-point terrain memory and small recovery."""
from .kinematics import G1KinematicChain,KinematicMotionPrior
from .state import RollingPointMemory
from .inputs import PointMemoryInputBuilder
from .recovery import EvidenceSurfaceCompleter

EVIDENCE_FIELDS=('current_anchor_z_m','current_mask','current_offset_cell_x','current_offset_cell_y',
 'current_height_span_m','history_anchor_z_m','history_mask','history_offset_cell_x',
 'history_offset_cell_y','history_age_s','transport_residual_hint_m')
