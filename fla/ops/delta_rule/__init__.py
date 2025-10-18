# -*- coding: utf-8 -*-

from .chunk import chunk_delta_rule
from .fused_chunk import fused_chunk_delta_rule
from .fused_recurrent import fused_recurrent_delta_rule
from .fused_recurrent_osla import fused_recurrent_delta_rule as fused_recurrent_delta_rule_osla

__all__ = [
    'fused_chunk_delta_rule',
    'fused_recurrent_delta_rule',
    'chunk_delta_rule',
    'fused_recurrent_delta_rule_osla'
]
