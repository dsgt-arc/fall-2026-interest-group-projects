import csv

import numpy as np
import pytest
import torch
from PIL import Image

import vision_ir.embed as embed_module
from vision_ir.augment import AUGMENTATIONS
from vision_ir.data import (
    DOMAINS,
    IMAGES_PER_GROUP,
    _group_of,
    copy_queries,
    gallery_items,
    read_photos,
    selected_groups,
)
from vision_ir.embed import ImageItem, embed
from vision_ir.evaluate import mean_pair_cosine, standardize
from vision_ir.tasks import copy_scores, same_group_scores, same_group_task


def test_selected_groups_skip_faces_and_cover_five_domains():
    groups = selected_groups()
    assert len(groups) == 125
    assert max(groups) < 1000  # groups 1000–1199 are celebrity faces
    assert {name for name in groups.values()} == {d.name for d in DOMAINS}
    assert all(group % 8 == 0 for group in groups)


@pytest.mark.parametrize("name, group", [
    ("images/0_430b273cb7915d83.jpg", 0),
    ("images/400_sketch_12.JPEG", 400),
    ("images/800_110434895315_10.JPG", 800),
    ("images/", None),
    ("README.txt", None),
])
def test_group_parsing(name, group):
    assert _group_of(name) == group


@pytest.fixture
def fake_dataset(tmp_path):
    """A complete 125-group manifest with tiny solid-color images."""
    root = tmp_path / "gpr1200"
    (root / "images").mkdir(parents=True)
    with (root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "group", "domain"])
        for group, domain in selected_groups().items():
            for index in range(IMAGES_PER_GROUP):
                name = f"{group}_{index}.jpg"
                color = (group % 256, index * 20, 100)
                Image.new("RGB", (40, 30), color).save(root / "images" / name)
                writer.writerow([name, group, domain])
    return root


def test_read_photos_validates_and_sorts(fake_dataset):
    photos = read_photos(fake_dataset)
    assert len(photos) == 1250
    assert photos[0].group == 0 and photos[-1].group == max(selected_groups())


def test_read_photos_rejects_missing_groups(fake_dataset):
    manifest = fake_dataset / "manifest.csv"
    lines = manifest.read_text().splitlines()
    manifest.write_text("\n".join(lines[:-1]) + "\n")  # drop one image
    with pytest.raises(ValueError):
        read_photos(fake_dataset)


def test_same_group_task_marks_nine_correct_images_per_query(fake_dataset):
    photos = read_photos(fake_dataset)
    relevant, ignore = same_group_task(photos)
    assert (relevant.sum(axis=1) == IMAGES_PER_GROUP - 1).all()
    assert not (relevant & ignore).any()


def test_copy_queries_cover_every_group_and_edit(fake_dataset):
    photos = read_photos(fake_dataset)
    copies = copy_queries(photos)
    assert len(copies) == 125 * len(AUGMENTATIONS)
    assert len({copy.item.item_id for copy in copies}) == len(copies)
    assert {copy.edit for copy in copies} == set(AUGMENTATIONS)


def test_perfect_embeddings_score_one(fake_dataset):
    photos = read_photos(fake_dataset)
    groups = sorted({photo.group for photo in photos})
    one_hot = np.eye(len(groups))[[groups.index(photo.group) for photo in photos]]
    scores = same_group_scores(one_hot, photos)
    assert scores["mAP"] == pytest.approx(1.0) and scores["P@1"] == pytest.approx(1.0)


def test_copy_scores_find_exact_originals(fake_dataset):
    photos = read_photos(fake_dataset)
    copies = copy_queries(photos)
    gallery = np.eye(len(photos))  # every image is its own direction
    position = {photo.image_id: i for i, photo in enumerate(photos)}
    queries = gallery[[position[copy.source_id] for copy in copies]]
    scores = copy_scores(queries, gallery, photos, copies)
    assert scores["mean"] == pytest.approx(1.0)
    assert set(scores) == set(AUGMENTATIONS) | {"mean", "misses that were group-mates"}
    assert np.isnan(scores["misses that were group-mates"])  # no misses at all


def test_copy_scores_count_group_mate_misses(fake_dataset):
    photos = read_photos(fake_dataset)
    copies = copy_queries(photos)
    gallery = np.eye(len(photos))
    position = {photo.image_id: i for i, photo in enumerate(photos)}
    # Every query points at the image right after its source: a group-mate, never the source.
    queries = gallery[[position[copy.source_id] + 1 for copy in copies]]
    scores = copy_scores(queries, gallery, photos, copies)
    assert scores["mean"] == pytest.approx(0.0)
    assert scores["misses that were group-mates"] == pytest.approx(1.0)


def test_standardize_uses_reference_statistics():
    reference = np.array([[1.0, 10.0], [3.0, 30.0]])
    np.testing.assert_allclose(standardize(reference, reference), [[-1, -1], [1, 1]], atol=1e-6)


def test_mean_pair_cosine_detects_crowded_vectors():
    assert mean_pair_cosine(np.array([[1.0, 0.0], [2.0, 0.0], [5.0, 0.0]])) == pytest.approx(1.0)
    assert mean_pair_cosine(np.eye(4)) == pytest.approx(0.0)


class _FakeEncoder:
    calls = 0

    def __call__(self, images):
        _FakeEncoder.calls += 1
        values = torch.tensor([[float(np.asarray(image).mean()), 1.0] for image in images])
        return {"final": values, "layers": {1: values * 2}}


def test_embed_caches_results_and_invalidates_on_new_options(fake_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(embed_module, "load_encoder", lambda *args, **kwargs: _FakeEncoder())
    items = gallery_items(read_photos(fake_dataset))[:5]
    options = {"cache_dir": tmp_path, "output_dir": tmp_path / "emb", "device": torch.device("cpu")}
    first = embed("DiT", items, batch_size=2, **options)
    calls = _FakeEncoder.calls
    again = embed("DiT", items, batch_size=2, **options)
    assert _FakeEncoder.calls == calls  # loaded from disk
    np.testing.assert_allclose(first.final, again.final)
    assert first.rows([items[1].item_id]).shape == (1, 2)
    embed("DiT", items, batch_size=2, timestep=50, **options)
    assert _FakeEncoder.calls > calls  # different options → recomputed


def test_embed_rejects_duplicate_ids(tmp_path):
    item = ImageItem("same", tmp_path / "a.jpg")
    with pytest.raises(ValueError):
        embed("ViT", [item, item], cache_dir=tmp_path, output_dir=tmp_path, device=torch.device("cpu"))
