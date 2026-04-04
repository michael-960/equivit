import equivit
import numpy as np



def test_cycclic_groups():
    for n in range(1, 20):
        G = equivit.geometry.CyclicGroup(n)

        assert G.order() == n
        assert len(G) == n

        if n > 1:
            assert G['r'] is G.from_value(1)
        else:
            assert G is equivit.geometry.TRIVIAL_GROUP
        
        for i in range(n):
            g = G.from_value(i)
            assert g.value == i
            assert g is G.from_value(i)

        for g in G:
            assert len(set([g*h for h in G])) == len(G)


def test_dihedral_groups():
    for n in range(1, 20):
        G = equivit.geometry.DihedralGroup(n)

        assert G.order() == 2*n
        assert len(G) == 2*n

        if n > 1:
            assert G['r'] is G.from_value((0, 1))
            assert G['t'] is G.from_value((1, 0))
        else:
            assert G['t'] is G.from_value((1, 0)) 
        
        for i in range(2):
            for j in range(n):
                g = G.from_value((i, j))
                assert g.value == (i, j)
                assert g is G.from_value((i, j))

        for g in G:
            assert len(set([g*h for h in G])) == len(G)


def test_group_action():

    for n in [1,2,3,4,5,6,7,8,9,10,12,15,17,30,40,48]:
        hexagon = equivit.geometry.Hexagon(n)

        group = hexagon.symmetry_group
        action = hexagon.action

        projections = equivit.geometry.decompose_set_action(action)
        irreps = group.real_irreps()


        for irrep_name, projection in projections.items():
            irrep = irreps[irrep_name]
            for p in projection:
                assert p.shape[0] == action.num_elements
                assert p.shape[1] == irrep.dim

                for g in group:
                    proj = p.to_dense().numpy()
                    error = proj[action(g.inv()),:] - proj @ irrep(g)
                    assert np.max(np.abs(error)) < 1e-13
