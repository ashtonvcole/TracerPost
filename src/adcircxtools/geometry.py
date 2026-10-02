"""geometry.py

Tools for mesh-related geometry, especially on the surface of a sphere or
geoid.
"""

import numpy

def central_angle(lon1: float | numpy.ndarray, lat1: float | numpy.ndarray,
    lon2: float | numpy.ndarray, lat2: float | numpy.ndarray,
    use_deg: bool = True):
    """Compute the central angle between two points on a sphere, given longitude
    and latitude.

    This is computed from the haversine fomula.

    hav(theta) = hav(lat2 - lat1) + cos(lat1) * cos(lat2) * hav(lon2 - lon1)

    Let a = hav(theta). Then a numerically stable way to compute theta exploits
    arctangent.

    theta = atan2()

    Arguments:
        lon1 (float | numpy.ndarray): The longitude of the first point.
        lat1 (float | numpy.ndarray): The latitude of the first point.
        lon2 (float | numpy.ndarray): The longitude of the second point.
        lat2 (float | numpy.ndarray): The latitude of the second point.
        use_deg (bool, optional): Whether the inputs are provided in degrees. If
            True, the output will also be converted to degrees. Default is True.

    Returns:
        float | numpy.ndarray: The central angle between the two points on a
            sphere, in the same units as the inputs.
    """
    # Ensure computations are in radians
    if use_deg:
        lon1 = numpy.deg2rad(lon1)
        lat1 = numpy.deg2rad(lat1)
        lon2 = numpy.deg2rad(lon2)
        lat2 = numpy.deg2rad(lat2)

    # Compute haversine of theta
    a = (numpy.sin((lat2 - lat1) / 2.0) ** 2 +
        numpy.cos(lat1) * numpy.cos(lat2) * numpy.sin((lon2 - lon1) / 2.0) ** 2)

    # Compute theta
    theta = 2 * numpy.atan2(numpy.sqrt(a), numpy.sqrt(1 - a))

    # Convert back to the user's units
    if use_deg:
        theta = numpy.rad2deg(theta)
    return theta

def cpp(coord: numpy.ndarray, lon0: float, lat0: float, R: float,
    use_deg: bool = True) -> numpy.ndarray:
    """Carte parallelogrammatique projection used in ADCIRC and DGSWEM.

    x = R * (lon - lon0) * cos(lat0)

    y = R * lat

    Arguments:
        coord (numpy.ndarray): The list of longitudes and latitudes to be
            projected. Of shape (num_coord, 2).
        lon0 (float): The longitude of the center of the projection.
        lat0 (float): The latitude of the center of the projection.
        R (float): The radius of the sphere used for projection.
        use_deg (bool, optional): Whether the inputs are provided in degrees. If
            True, the coordinates will be converted to radians for the
            projection. Default is True.

    Returns:
        numpy.ndarray
    """
    if use_deg:
        coord = numpy.deg2rad(coord)
        lon0 = numpy.deg2rad(lon0)
        lat0 = numpy.deg2rad(lat0)
    projected_coord = numpy.zeros(coord.shape)
    projected_coord[:, 0] = R * (coord[:, 0] - lon0) * numpy.cos(lat0)
    projected_coord[:, 1] = R * coord[:, 1]
    return projected_coord
