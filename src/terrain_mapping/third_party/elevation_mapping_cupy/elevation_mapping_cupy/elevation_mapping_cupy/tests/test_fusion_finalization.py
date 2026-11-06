import cupy as cp

from elevation_mapping_cupy.kernels.custom_kernels import finalize_map_kernel


def _empty_state(size):
    previous = cp.zeros((7, size, size), dtype=cp.float32)
    previous[1].fill(10.0)
    proposals = cp.zeros_like(previous)
    visibility = cp.zeros((3, size, size), dtype=cp.float32)
    visibility[2].fill(cp.inf)
    output = cp.empty_like(previous)
    return previous, proposals, visibility, output


def test_endpoint_update_wins_over_same_frame_visibility_cleanup():
    size = 5
    row = col = 2
    previous, proposals, visibility, output = _empty_state(size)
    previous[0, row, col] = 1.0
    previous[1, row, col] = 0.1
    previous[2, row, col] = 1.0
    previous[4, row, col] = 1.0
    proposals[0, row, col] = 2.0
    proposals[1, row, col] = 0.2
    proposals[2, row, col] = 1.0
    visibility[0, row, col] = 2.0
    visibility[1, row, col] = 20.0
    visibility[2, row, col] = -3.0
    finalize = finalize_map_kernel(size, size, 1.0, 10.0, 0.01)

    finalize(previous, proposals, visibility, output, size=size * size)

    assert float(output[0, row, col]) == 2.0
    assert float(output[1, row, col]) == cp.float32(0.2)
    assert float(output[2, row, col]) == 1.0
    assert float(output[4, row, col]) == 0.0
    assert float(output[5, row, col]) == 2.0
    assert float(output[6, row, col]) == 0.0


def test_visibility_proposal_updates_invalid_cell_upper_bound():
    size = 5
    row = col = 2
    previous, proposals, visibility, output = _empty_state(size)
    visibility[2, row, col] = -0.4
    finalize = finalize_map_kernel(size, size, 1.0, 10.0, 0.01)

    finalize(previous, proposals, visibility, output, size=size * size)

    assert float(output[0, row, col]) == 0.0
    assert float(output[1, row, col]) == 10.0
    assert float(output[2, row, col]) == 0.0
    assert cp.isclose(output[5, row, col], -0.4)
    assert float(output[6, row, col]) == 1.0
