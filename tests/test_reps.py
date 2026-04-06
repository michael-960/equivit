import equivit
import pytest



@pytest.mark.parametrize('group', [equivit.geometry.CyclicGroup(n) for n in range(1, 20)] + [equivit.geometry.DihedralGroup(n) for n in range(1, 20)])
def test_real_irreps(group):
    irreps = group.real_irreps()
    for irrep in irreps.values():
        irrep.validate()