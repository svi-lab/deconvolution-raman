import numpy as np
import matplotlib.pyplot as plt

compounds = {'NS': np.array([49.22, 0, 50.78]),
             'NS2': np.array([65.97, 0, 34.03]),
             'N2CS3': np.array([50.03, 15.57, 34.41]),
             'NC2S2': np.array([40.83, 38.11, 21.06]),
             'NC2S3': np.array([50.86, 31.65, 17.49]),
             'NC3S6': np.array([61.03, 28.48, 10.49]),
             'CS': np.array([51.72, 48.28, 0]),
             'C2S': np.array([34.88, 65.12, 0]),
             'C': np.array([0, 100., 0]),
             'S': np.array([100., 0, 0]),
             'N': np.array([0, 0, 100.]),
             'void': np.zeros(3)}

oxides_masses = np.array([60.08, 56.08, 61.98])
batch_masses = np.array([60.08, 100.09, 105.99])

def molmass(*molar_composition):
    """
    Parameters
    ----------
    molar_composition : float, float, float or np.ndarray 3 * 1
        molar composition in the order SiO2, CaO, Na2O

    Notes
    -----
    Converts the molar composition of a glass into mass fraction. Works with a
    three-float input or a three-float numpy.ndarray input.
    Returns a three element numpy.ndarray of mass fractions in the order SiO2,
    CaO, Na2O.

    See also
    --------
    ternary.massmol : reciprocal operation
    """
    if not isinstance(molar_composition, np.ndarray):
        molar_composition = np.array(molar_composition)
    masses = oxides_masses * molar_composition
    return 100 * masses / masses.sum()

def massmol(*mass_composition):
    """
    Parameters
    ----------
    mass_composition : float, float, float or np.ndarray 3 * 1
        mass composition in the order SiO2, CaO, Na2O

    Notes
    -----
    Converts the mass composition of a glass into molar fraction. Works with a
    three-float input or a three-float numpy.ndarray input.
    Returns a three element numpy.ndarray of molar fractions in the order SiO2,
    CaO, Na2O.

    See also
    --------
    ternary.molmass : reciprocal operation
    """
    if len(mass_composition) > 2 and not isinstance(mass_composition, np.ndarray):
        mass_composition = np.array(mass_composition)
    moles = mass_composition[0] / oxides_masses
    return 100 * moles / moles.sum(-1)[:, np.newaxis]


def ternary_to_cartesian(a, b, c):
    """
    Parameters
    ----------
    a : numpy.ndarray, flat
        Content of the first species, corresponding to the lower left
        pole of the diagram
    b : numpy.ndarray, flat
        Content of the second species, corresponding to the lower right
        pole of the diagram
    c : numpy.ndarray, flat
        Content of the third species, corresponding to the upper pole of
        the diagram

    Notes
    -----
    Converts an equilateral ternary plot where a = 100% is placed at
    (x, y) = (0, 0) and b = 100% at (1, 0) and c=100% is (0.5, sqrt(3)/2)
    to cartesian coordinates.

    Examples
    --------
    >>> i = Carte.image[Carte.image.sum(axis=2) != 0]
    >>> b, c, a = np.hsplit(i, 3)
    >>> x, y = ternary_to_cartesian(a, b, c)

    See also
    --------
    cartesian_to_ternary : reciprocal operation
    """
    
    x = .5 * (2 * b + c) / (a + b + c)
    y = np.sqrt(.75) * c / (a + b + c)
    return x, y


def cartesian_to_ternary(x, y):
    """
    Parameters
    ----------
    x : numpy.ndarray, flat
        x-coordinate in the 2D cartesian plot
    y : numpy.ndarray, flat
        y-coordinate in the 2D cartesian plot

    Notes
    -----
    Converts cartesian coordinates to an ternary coordinates where
    a = 100% is placed at (x, y) = (0, 0) and b = 100% at (1, 0) and
    c=100% is (0.5, sqrt(3)/2).

    Examples
    --------
    >>> color = cartesian_to_ternary(xpos, ypos)

    See also
    --------
    ternary_to_cartesian : reciprocal operation
    """
    
    a = 1 - x - 1 / np.sqrt(3) * y
    b = x - 1 / np.sqrt(3) * y
    c = 2 / np.sqrt(3) * y
    return np.column_stack((b, c, a))


def elements_to_oxides(si, ca, na):
    """
    Parameters
    ----------
    si: numpy.ndarray
        Silicon array
    ca: numpy.ndarray
        Calcium array
    na: numpy.ndarray
        Sodium array

    Notes
    -----
    Converts element weight percents into the corresponding oxides weight
    percents, assuming SiO2, CaO and Na2O are the only oxides.

    Examples
    --------
    >>> sio2, cao, na2o = elements_to_oxides(si, ca, na)
    """

    M = {'Si': 28.09, 'SiO2': 60.08, 'Ca': 40.08, 'CaO': 56.08,
         'Na': 22.99, 'Na2O': 61.98}
    sio2 = M['SiO2'] * si / M['Si']
    cao = M['CaO'] * ca / M['Ca']
    na2o = M['Na2O'] * na / (2 * M['Na'])
    return sio2, cao, na2o


def ternary_frame(flat=True):
    """
    Parameters
    ----------
    flat : bool, default True
        Should the frame be flat, ie two-dimensional.

    Notes
    -----
    Prepares a matplotlib.figure instance where drawing ternary plots is
    easy : triangular boundaries, orthogonal and same-scale axes for a
    flat (ie 2D) frame, 3d subplot for a non-flat frame.

    Examples
    --------
    >>> fig = ternary_frame(flat=False)
    >>> ax = fig.get_axes()[0]
    >>> ax.plot(x, y, z)
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    # The coordinates of the triangle vertices
    vertices = np.array([[0, 1, .5, 0],
                         [0, 0, np.sqrt(.75), 0],
                         [0, 0, 0, 0]])
    if flat is True:
        fig.set_size_inches(10, 10)
        plt.axes([0, 0, 1, 1])
        plt.axis('equal')
        plt.plot(vertices[0], vertices[1], 'k-')
        plt.text(-0.05, 0, "Na2O")
        plt.text(1.05, 0, "SiO2")
        plt.text(.5, .88, "CaO")
    else:
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(vertices[0], vertices[1], vertices[2], 'k-')
        ax.text(-0.05, 0, 0, "Na2O")
        ax.text(1.1, 0, 0, "SiO2")
        ax.text(.5, .88, 0, "CaO")
    return fig


def inside_triangle(point):
    """
    Parameters
    ----------
    point : numpy.ndarray, shape n * 2
        x, y coordinate of the point

    Notes
    -----
    Returns True if the point is inside the triangle of ternary
    coordinates (0, 0) (1, 0) (.5, sqrt(.75)). A point on the edge of
    the triangle or on a vertex is considered inside it.

    Examples
    --------
    >>> points = np.array([[.1, .1], [.5, .9], [.8, .2]])
    >>> inside_triangle(points)
    array([ True, False,  True], dtype=bool)
    """
    
    A = np.array([0, 0], dtype=np.float64)
    B = np.array([1, 0], dtype=np.float64)
    C = np.array([.5, np.sqrt(.75)], dtype=np.float64)
    v1 = np.cross(A - point, B - point)
    v2 = np.cross(B - point, C - point)
    v3 = np.cross(C - point, A - point)
    return (np.array([v1, v2, v3]) >= 0).all(axis=0)


def mesh(num, min=0, max=.9):
    num = np.sqrt(num * 2)
    r = np.linspace(min, max, num=num)
    raw_mesh = np.meshgrid(r, r)
    mesh = np.column_stack(tuple(m.flatten() for m in raw_mesh))
    mesh = mesh[inside_triangle(mesh)]
    return mesh

if __name__ == '__main__':
    m = mesh(225)
    print(m.shape)
    plt.scatter(m[:, 0], m[:, 1])
    plt.show()
