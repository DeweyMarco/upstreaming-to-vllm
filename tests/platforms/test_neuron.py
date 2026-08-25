# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest

from vllm.platforms.neuron import NeuronFramework, NeuronPlatform


@pytest.mark.parametrize(
    ("specified_framework", "tnx_installed", "nxd_installed"),
    [
        (NeuronFramework.TRANSFORMERS_NEURONX, True, True),
        (NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE, True, True),
    ],
)
def test_explicit_neuron_framework_selection(monkeypatch, specified_framework,
                                             tnx_installed, nxd_installed):
    monkeypatch.setenv("VLLM_NEURON_FRAMEWORK", specified_framework.value)

    with patch.object(NeuronPlatform,
                      "is_transformers_neuronx",
                      return_value=tnx_installed), patch.object(
                          NeuronPlatform,
                          "is_neuronx_distributed_inference",
                          return_value=nxd_installed):
        assert (NeuronPlatform().get_neuron_framework_to_use() ==
                specified_framework)


def test_neuron_framework_defaults_to_nxd(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_FRAMEWORK", raising=False)

    with patch.object(NeuronPlatform,
                      "is_transformers_neuronx",
                      return_value=True), patch.object(
                          NeuronPlatform,
                          "is_neuronx_distributed_inference",
                          return_value=True):
        assert (NeuronPlatform().get_neuron_framework_to_use() ==
                NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE)


def test_invalid_neuron_framework_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_FRAMEWORK", "unknown-framework")

    with patch.object(NeuronPlatform,
                      "is_transformers_neuronx",
                      return_value=True), patch.object(
                          NeuronPlatform,
                          "is_neuronx_distributed_inference",
                          return_value=True), pytest.raises(
                              ValueError, match="unknown-framework"):
        NeuronPlatform().get_neuron_framework_to_use()


def test_unavailable_neuron_framework_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_FRAMEWORK",
                       NeuronFramework.TRANSFORMERS_NEURONX.value)

    with patch.object(NeuronPlatform,
                      "is_transformers_neuronx",
                      return_value=False), patch.object(
                          NeuronPlatform,
                          "is_neuronx_distributed_inference",
                          return_value=True), pytest.raises(
                              RuntimeError, match="not installed"):
        NeuronPlatform().get_neuron_framework_to_use()


def test_missing_neuron_frameworks_are_rejected(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_FRAMEWORK", raising=False)

    with patch.object(NeuronPlatform,
                      "is_transformers_neuronx",
                      return_value=False), patch.object(
                          NeuronPlatform,
                          "is_neuronx_distributed_inference",
                          return_value=False), pytest.raises(
                              RuntimeError, match="No Neuron framework"):
        NeuronPlatform().get_neuron_framework_to_use()
