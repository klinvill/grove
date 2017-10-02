from grove.fermion_transforms.jwtransform import JWTransform
from itertools import product
from pyquil.paulis import sI, sX, sY, sZ, PauliTerm


def test_create():
    """
    Testing creation operator produces 0.5 * (X - iY)
    """
    jw = JWTransform()
    creation = jw.create(5)
    pt1 = 0.5
    pt2 = -0.5j
    for i in range(5):
        pt1 *= sZ(i)
        pt2 *= sZ(i)
    pt1 *= sX(5)
    pt2 *= sY(5)
    ps = pt1 + pt2
    assert ps == creation


def test_annihilation():
    """
    Testing creation operator produces 0.5 * (X - iY)
    """
    jw = JWTransform()
    annihilation = jw.kill(5)
    pt1 = 0.5
    pt2 = 0.5j
    for i in range(5):
        pt1 *= sZ(i)
        pt2 *= sZ(i)
    pt1 *= sX(5)
    pt2 *= sY(5)
    ps = pt1 + pt2
    assert ps == annihilation


def test_hopping():
    """
    Hopping term tests
    """
    jw = JWTransform()
    hopping = jw.create(2) * jw.kill(0) + jw.create(0) * jw.kill(2)
    pt1 = 0.5 * PauliTerm('X', 0) * PauliTerm('Z', 1) * PauliTerm('X', 2)
    pt2 = 0.5 * PauliTerm('Y', 0) * PauliTerm('Z', 1) * PauliTerm('Y', 2)
    ps = pt1 + pt2
    assert ps == hopping


def test_multi_ops():
    """
    test construction of Paulis for product of second quantized operators
    """
    jw = JWTransform()
    # test on one particle density matrix
    for p, q, in product(range(6), repeat=2):
        truth = jw.create(p) * jw.kill(q)
        prod_ops_out = jw.product_ops([p, q], [-1, 1])
        assert truth == prod_ops_out

    # test on two particle density matrix
    for p, q, r, s in product(range(4), repeat=4):
        truth = jw.create(p) * jw.create(q) * jw.kill(s) * jw.kill(r)
        prod_ops_out = jw.product_ops([p, q, s, r], [-1, -1, 1, 1])
        assert truth == prod_ops_out


def test_one_body():
    jw = JWTransform()
    assert jw.one_body_term(1, 1) == (jw.product_ops([1, 1], [-1, 1])
                                      + jw.product_ops([1, 1], [-1, 1]))
    assert jw.one_body_term(0, 2) == (jw.product_ops([0, 2], [-1, 1])
                                      + jw.product_ops([2, 0], [-1, 1]))
    # Should be symmetric
    assert jw.one_body_term(0, 2) == jw.one_body_term(2, 0)


def test_two_body():
    jw = JWTransform()
    assert jw.two_body_term(1, 1, 1, 1) == (jw.product_ops([1, 1, 1, 1], [-1, -1, 1, 1])
                                            + jw.product_ops([1, 1, 1, 1], [-1, -1, 1, 1]))
    assert jw.two_body_term(1, 2, 3, 4) == (jw.product_ops([1, 2, 3, 4], [-1, -1, 1, 1])
                                            + jw.product_ops([4, 3, 2, 1], [-1, -1, 1, 1]))
    assert jw.two_body_term(1, 2, 2, 3) == (jw.product_ops([1, 2, 2, 3], [-1, -1, 1, 1])
                                            + jw.product_ops([3, 2, 2, 1], [-1, -1, 1, 1]))
    assert jw.two_body_term(1, 2, 2, 1) == (jw.product_ops([1, 2, 2, 1], [-1, -1, 1, 1])
                                            + jw.product_ops([1, 2, 2, 1], [-1, -1, 1, 1]))
