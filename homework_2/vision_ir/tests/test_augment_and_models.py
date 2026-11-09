import numpy as np
import pytest
import torch
from PIL import Image

from vision_ir.augment import AUGMENTATIONS, augment
from vision_ir.models import BACKBONES, layer_indices, preprocess


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", sorted(AUGMENTATIONS))
def test_every_augmentation_returns_a_new_rgb_image(photo, name):
    edited = augment(photo, name)
    assert edited.mode == "RGB"
    assert edited is not photo
    assert np.asarray(photo).shape == (120, 160, 3)  # original left untouched


@pytest.mark.parametrize("name", sorted(AUGMENTATIONS))
def test_augmentations_are_deterministic(photo, name):
    assert np.array_equal(np.asarray(augment(photo, name)), np.asarray(augment(photo, name)))


def test_rotation_swaps_width_and_height(photo):
    assert augment(photo, "rotate 90°").size == (120, 160)


def test_unknown_augmentation_is_rejected(photo):
    with pytest.raises(KeyError):
        augment(photo, "sepia")


@pytest.mark.parametrize("key", sorted(BACKBONES))
def test_preprocess_produces_square_normalized_batches(photo, key):
    backbone = BACKBONES[key]
    pixels = preprocess([photo, photo.rotate(90, expand=True)], backbone)
    size = backbone.image_size
    assert pixels.shape == (2, 3, size, size)
    assert pixels.dtype == torch.float32
    assert torch.isfinite(pixels).all()


def test_layer_indices_are_sorted_unique_and_end_at_last_block():
    assert layer_indices(12) == [2, 3, 4, 6, 8, 9, 10, 12]
    assert layer_indices(4) == [1, 2, 3, 4]
    assert layer_indices(32)[-1] == 32


def test_backbones_pin_revisions():
    for backbone in BACKBONES.values():
        assert len(backbone.revision) == 40
