import pytest
from shapely.geometry import Polygon, Point

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from geo_utils import get_extend_line


@pytest.fixture
def block():
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def _assert_no_output(capsys):
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def _assert_point_intersection(line, block):
    intersect = block.boundary.intersection(line)
    assert intersect.geom_type == "Point"


def test_get_extend_line_vertical(block, capsys):
    a = Point(5, 2)
    b = Point(5, 8)
    line = get_extend_line(a, b, block, isfront=False)
    _assert_no_output(capsys)
    _assert_point_intersection(line, block)


def test_get_extend_line_horizontal(block, capsys):
    a = Point(2, 5)
    b = Point(8, 5)
    line = get_extend_line(a, b, block, isfront=False)
    _assert_no_output(capsys)
    _assert_point_intersection(line, block)


def test_get_extend_line_diagonal(block, capsys):
    a = Point(2, 2)
    b = Point(5, 5)
    line = get_extend_line(a, b, block, isfront=False)
    _assert_no_output(capsys)
    _assert_point_intersection(line, block)
