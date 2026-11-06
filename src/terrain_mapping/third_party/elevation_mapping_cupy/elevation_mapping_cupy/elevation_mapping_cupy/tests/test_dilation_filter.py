import cupy as cp

from elevation_mapping_cupy.kernels.custom_kernels import dilation_filter_kernel


def _run(values, mask, radius, *, alias=False):
    values = cp.asarray(values, dtype=cp.float32)
    mask = cp.asarray(mask, dtype=cp.float32)
    operation = dilation_filter_kernel(values.shape[1], values.shape[0], radius)
    if alias:
        operation(values, mask, values, mask, size=values.size)
        return values.get(), mask.get()

    output = cp.zeros_like(values)
    output_mask = cp.zeros_like(mask)
    operation(values, mask, output, output_mask, size=values.size)
    return output.get(), output_mask.get()


def test_dilation_chooses_nearest_valid_cell():
    values = cp.zeros((11, 11), dtype=cp.float32)
    mask = cp.zeros_like(values)
    values[2, 2] = 22.0
    mask[2, 2] = 1.0
    values[5, 6] = 56.0
    mask[5, 6] = 1.0

    output, output_mask = _run(values, mask, radius=3)

    assert output[5, 5] == 56.0
    assert output_mask[5, 5] == 1.0


def test_dilation_does_not_wrap_between_rows():
    values = cp.zeros((7, 7), dtype=cp.float32)
    mask = cp.zeros_like(values)
    values[2, 5] = 25.0
    mask[2, 5] = 1.0

    output, output_mask = _run(values, mask, radius=3)

    assert output[3, 1] == 0.0
    assert output_mask[3, 1] == 0.0


def test_dilation_is_safe_when_inputs_alias_outputs():
    values = cp.zeros((9, 9), dtype=cp.float32)
    mask = cp.zeros_like(values)
    values[4, 4] = 7.0
    mask[4, 4] = 1.0

    output, output_mask = _run(values, mask, radius=2, alias=True)

    assert output[4, 4] == 7.0
    assert output[4, 5] == 7.0
    assert output_mask[4, 5] == 1.0
    assert output_mask[2, 2] == 0.0


def test_large_radius_distance_transform_respects_radius():
    values = cp.zeros((32, 32), dtype=cp.float32)
    mask = cp.zeros_like(values)
    values[16, 16] = 9.0
    mask[16, 16] = 1.0

    output, output_mask = _run(values, mask, radius=10)

    assert output[16, 26] == 9.0
    assert output_mask[16, 26] == 1.0
    assert output[16, 27] == 0.0
    assert output_mask[16, 27] == 0.0
