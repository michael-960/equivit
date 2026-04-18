import pytest
import equivit


@pytest.mark.parametrize(
    "lattice",
    [equivit.geometry.Hexagon(n) for n in range(100)] +\
    [equivit.geometry.Honeycomb(n) for n in range(1,100)] +\
    [equivit.geometry.Square(n) for n in range(100)] +\
    [equivit.geometry.Triangle(n) for n in range(100)]
)
def test_lattice_actions(lattice):
    lattice.action.validate()