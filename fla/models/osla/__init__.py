# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.osla.configuration_delta_net import OSLAConfig
from fla.models.osla.modeling_delta_net import OSLAForCausalLM, OSLAModel

AutoConfig.register(OSLAConfig.model_type, OSLAConfig, exist_ok=True)
AutoModel.register(OSLAConfig, OSLAModel, exist_ok=True)
AutoModelForCausalLM.register(OSLAConfig, OSLAForCausalLM, exist_ok=True)

__all__ = ['OSLAConfig', 'OSLAForCausalLM', 'OSLAModel']
