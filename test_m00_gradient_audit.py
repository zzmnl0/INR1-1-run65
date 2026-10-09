import inspect
import random

import numpy as np
import torch

import run_m00_gradient_audit as audit


def test_audit_entry_has_no_training_optimizer_or_validation_route():
    source = inspect.getsource(audit)
    assert 'tf.train_fsia' not in source
    assert '_NoOpOptimizer' not in source
    assert '.step(' not in source
    assert '.backward(' not in source
    assert 'clip_grad_norm_' not in source


def test_parameter_groups_are_cumulative_numel_and_cover_without_overlap():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    groups, size = audit._parameter_groups(model)
    ranges = [tuple(value.tolist()) for value in groups.values()]
    flat = [index for group in ranges for index in group]
    assert size == sum(parameter.numel() for parameter in model.parameters())
    assert flat == list(range(size))
    assert len(set(flat)) == size


def test_gradient_vector_sum_and_zero_support_are_explicit():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    total = (parameter.square()).sum()
    left = parameter[0].square()
    right = parameter[1].square()
    total_vector = audit._gradient_vector(total, [parameter])
    pieces = (audit._gradient_vector(left, [parameter])
              + audit._gradient_vector(right, [parameter]))
    assert torch.allclose(total_vector, pieces)
    stats = audit._supported_stats(torch.zeros(2), torch.ones(2), 0, 0)
    assert stats == {'value': None, 'reason': 'no finite supported points/profiles'}


def test_state_snapshot_restores_parameters_buffers_modes_grads_and_rng():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
    parameters = list(model.parameters())
    parameters[0].grad = torch.ones_like(parameters[0])
    model.eval()
    snapshot = audit._state_snapshot(model, parameters)
    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    with torch.no_grad():
        for value in model.state_dict().values():
            if torch.is_floating_point(value):
                value.add_(1)
    model.train()
    for parameter in parameters:
        parameter.requires_grad_(False)
        parameter.grad = None
    audit._restore_state(model, snapshot)
    assert model.training is False
    assert all(parameter.requires_grad for parameter in parameters)
    assert parameters[0].grad is not None
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshot['state_dict'][key])
    checks = audit._state_matches(model, parameters, snapshot)
    assert checks['model_state_unchanged']
    assert checks['rng_state_restored']
