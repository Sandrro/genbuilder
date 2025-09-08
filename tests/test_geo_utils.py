import sys
import types
import pathlib

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Avoid importing heavy optional dependencies
sys.modules.setdefault('skgeom', types.SimpleNamespace())
_fake_neighbors = types.SimpleNamespace(NearestNeighbors=object)
sys.modules.setdefault('sklearn', types.SimpleNamespace(neighbors=_fake_neighbors))
sys.modules.setdefault('sklearn.neighbors', _fake_neighbors)
sys.modules.setdefault('cv2', types.SimpleNamespace())
_fake_morphology = types.SimpleNamespace(medial_axis=lambda *a, **k: None)
sys.modules.setdefault('skimage', types.SimpleNamespace(morphology=_fake_morphology))
sys.modules.setdefault('skimage.morphology', _fake_morphology)

from shapely.geometry import box, Point, MultiLineString, LineString, GeometryCollection

from geo_utils import get_extend_line
from example_canonical_transform import modified_skel_to_medaxis


def test_get_extend_line_handles_empty_and_line_intersections():
    block = box(0, 0, 10, 10)

    # Empty intersection: starting point lies outside the block
    a = Point(20, 20)
    b = Point(20, 21)
    line = get_extend_line(a, b, block, False)
    assert list(line.coords) == [(20.0, 20.0), (20.0, 20.0)]

    # Intersection returning a LineString when the start is on the boundary
    a2 = Point(0, 5)
    b2 = Point(0, 7)
    line2 = get_extend_line(a2, b2, block, False, is_extend_from_end=True)
    assert list(line2.coords) == [(0.0, 7.0), (0.0, 5.0)]


def test_modified_skel_to_medaxis_handles_collections():
    block = box(0, 0, 10, 10)

    multi = MultiLineString([[(1, 1), (2, 2), (3, 3), (4, 4)], [(5, 5), (6, 6)]])
    res = modified_skel_to_medaxis(multi, block)
    assert isinstance(res, LineString)

    gc = GeometryCollection([LineString([(1, 1), (9, 9)]), Point(0, 0)])
    res2 = modified_skel_to_medaxis(gc, block)
    assert isinstance(res2, LineString)

    empty = GeometryCollection([])
    res3 = modified_skel_to_medaxis(empty, block)
    assert res3.is_empty
