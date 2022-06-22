from math import sin, cos, atan2, asin, radians, degrees, sqrt
import pygeohash as pgh
from geopy import distance
from scipy import spatial
import numbers
from pyspark.sql import functions as F
from pyspark.sql import types as T

EARTH_RADIUS = 6371009
UNIT_FACTOR = {"m": 1.0, "km": 1000.0}


def to_latlon_decimal_degrees(loc, input_format, radius=EARTH_RADIUS):
    """
    Parameters
    ----------

    loc
        Location to format.
        If input_format is "dd", [lat, lon] format is required.
        If input_format is "dms", [[d1,m1,s1], [d2,m2,s2]] format is required.
        If input_format is "radian", [lat_radian, lon_radian] format is required.
        If input_format is "cartesian", [x, y, z] format is required.
        If input_format is "geohash", string format is required.
    input_format
        "dd", "dms", "radian", "cartesian", "geohash".
    radius
        Radius of Earth. (Default value = EARTH_RADIUS)

    Returns
    -------
    List
        Formatted location: [lat, lon]
    """
    if loc is None:
        return None
    if isinstance(loc, (list, tuple)):
        if any(i is None for i in loc):
            return None
    if isinstance(loc[0], (list, tuple)):
        if any(i is None for i in loc[0] + loc[1]):
            return None

    if input_format == "dd":
        # loc = [lat, lon]
        return [float(loc[0]), float(loc[1])]

    elif input_format == "dms":
        # loc = [[d1,m1,s1], [d2,m2,s2]]
        d1, m1, s1, d2, m2, s2 = loc[0] + loc[1]
        lat = d1 + float(m1) / 60 + float(s1) / 3600
        lon = d2 + float(m2) / 60 + float(s2) / 3600

    elif input_format == "radian":
        # loc = [lat_radian, lon_radian]
        lat_rad, lon_rad = loc
        lat = degrees(float(lat_rad))
        lon = degrees(float(lon_rad))

    elif input_format == "cartesian":
        # loc = [x, y, z]
        x, y, z = loc
        lat = degrees(float(asin(z / radius)))
        lon = degrees(float(atan2(y, x)))

    elif input_format == "geohash":
        # loc = geohash
        lat, lon = list(pgh.decode(loc))

    return [lat, lon]


def decimal_degrees_to_degrees_minutes_seconds(dd):
    """
    Parameters
    ----------

    dd
        Float value in decimal degree.

    Returns
    -------
    List
        [degree, minute, second]

    """
    if dd is None:
        return [None, None, None]
    else:
        minute, second = divmod(dd * 3600, 60)
        degree, minute = divmod(minute, 60)
        return [degree, minute, second]


def from_latlon_decimal_degrees(
    loc, output_format, radius=EARTH_RADIUS, geohash_precision=8
):

    """
    Parameters
    ----------

    loc
        Location to format with format [lat, lon].
    output_format
        "dd", "dms", "radian", "cartesian", "geohash".
    radius
        Radius of Earth. (Default value = EARTH_RADIUS)
    geohash_precision
        Precision of the resultant geohash.
        This argument is only used when output_format is "geohash". (Default value = 8)

    Returns
    -------
    String if output_format is "geohash", list otherwise.
        [lat, lon] if output_format is "dd".
        [[d1,m1,s1], [d2,m2,s2]] if output_format is "dms".
        [lat_radian, lon_radian] if output_format is "radian".
        [x, y, z] if output_format is "cartesian".
        string if output_format is "geohash".
    """
    # loc = [lat, lon]
    if loc is None:
        lat, lon = None, None
    else:
        lat, lon = loc[0], loc[1]

    if output_format == "dd":
        return [lat, lon]

    elif output_format == "dms":
        return [
            decimal_degrees_to_degrees_minutes_seconds(lat),
            decimal_degrees_to_degrees_minutes_seconds(lon),
        ]

    elif output_format == "radian":
        if (lat is None) | (lon is None):
            return [None, None]
        else:
            lat_rad = radians(float(lat))
            lon_rad = radians(float(lon))
            return [lat_rad, lon_rad]

    elif output_format == "cartesian":
        if (lat is None) | (lon is None):
            return [None, None, None]
        else:
            lat_rad = radians(float(lat))
            lon_rad = radians(float(lon))
            x = radius * cos(lat_rad) * cos(lon_rad)
            y = radius * cos(lat_rad) * sin(lon_rad)
            z = radius * sin(lat_rad)
            return [x, y, z]

    elif output_format == "geohash":
        if (lat is None) | (lon is None):
            return None
        else:
            return pgh.encode(lat, lon, geohash_precision)


def haversine_distance(loc1, loc2, loc_format, unit="m", radius=EARTH_RADIUS):

    """
    Parameters
    ----------

    loc1
        The first location.
        If loc_format is "dd", [lat, lon] format is required.
        If loc_format is "radian", [lat_radian, lon_radian] format is required.
    loc2
        The second location .
        If loc_format is "dd", [lat, lon] format is required.
        If loc_format is "radian", [lat_radian, lon_radian] format is required.
    loc_format
        "dd", "radian".
    unit
        "m", "km".
        Unit of the result. (Default value = "m")
    radius
        Radius of Earth. (Default value = EARTH_RADIUS)

    Returns
    -------
    Float
    """
    # loc1 = [lat1, lon1]; loc2 = [lat2, lon2]
    if None in [loc1, loc2]:
        return None
    if None in loc1 + loc2:
        return None
    if loc_format not in ["dd", "radian"]:
        raise TypeError("Invalid input for loc_format")

    lat1, lon1 = float(loc1[0]), float(loc1[1])
    lat2, lon2 = float(loc2[0]), float(loc2[1])

    if loc_format == "dd":
        lat1, lon1 = radians(lat1), radians(lon1)
        lat2, lon2 = radians(lat2), radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = radius * c / UNIT_FACTOR[unit]

    return distance


def vincenty_distance(loc1, loc2, unit="m", ellipsoid="WGS-84"):
    """
    Vincenty's formulae are two related iterative methods used in geodesy to calculate
    the distance between two points on the surface of a spheroid.

    Parameters
    ----------

    loc1
        The first location. [lat, lon] format is required.
    loc2
        The second location. [lat, lon] format is required.
    unit
        "m", "km".
        Unit of the result. (Default value = "m")
    ellipsoid
        "WGS-84", "GRS-80", "Airy (1830)", "Intl 1924", "Clarke (1880)", "GRS-67".
        The ellipsoidal model to use. For more information, please refer to geopy.distance.ELLIPSOIDS.
        (Default value = "WGS-84")

    Returns
    -------
    Float
    """

    if None in [loc1, loc2]:
        return None
    if None in loc1 + loc2:
        return None

    loc_distance = distance.distance(loc1, loc2, ellipsoid=ellipsoid)

    if unit == "m":
        return loc_distance.m
    else:
        return loc_distance.km


def euclidean_distance(loc1, loc2):
    """
    The Euclidean distance between 2 lists loc1 and loc2, is defined as
    .. math::
       {\\|loc1-loc2\\|}_2

    Parameters
    ----------

    loc1
        The first location. [x1, y1, z1] format is required.
    loc2
        The second location. [x2, y2, z2] format is required.

    Returns
    -------
    Float
    """
    if None in [loc1, loc2]:
        return None
    if None in loc1 + loc2:
        return None

    # loc1 = [x1, y1, z1]; loc2 = [x2, y2, z2]
    euclidean_distance = spatial.distance.euclidean(loc1, loc2)
    return euclidean_distance


def point_in_polygon(x, y, polygon):
    """
    Check whether (x,y) is inside a polygon

    Parameters
    ----------

    x
        x coordinate/longitude
    y
        y coordinate/latitude
    polygon
        polygon consists of list of (x,y)s

    Returns
    -------
    Integer
        1 if (x, y) is in the polygon and 0 otherwise.
    """
    if (x is None) | (y is None):
        return None

    counter = 0
    for index, poly in enumerate(polygon):
        # Check whether x and y are numbers
        if not isinstance(x, numbers.Number) or not isinstance(y, numbers.Number):
            raise TypeError("Input coordinate should be of type float")

        # Check whether poly is list of (x,y)s
        if any([not isinstance(i, numbers.Number) for point in poly for i in point]):
            # Polygon from multipolygon have extra bracket - that need to be removed
            poly = poly[0]
            if any(
                [not isinstance(i, numbers.Number) for point in poly for i in point]
            ):
                raise TypeError("The polygon is invalid")

        # Check if point is a vertex
        test_vertex = (x, y) if isinstance(poly[0], tuple) else [x, y]
        if test_vertex in poly:
            return 1

        # Check if point is on a boundary
        poly_length = len(poly)
        for i in range(poly_length):
            if i == 0:
                p1 = poly[0]
                p2 = poly[1]
            else:
                p1 = poly[i - 1]
                p2 = poly[i]
            if (
                p1[1] == p2[1]
                and p1[1] == y
                and (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]))
            ):
                return 1
            if (
                p1[0] == p2[0]
                and p1[0] == x
                and (min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]))
            ):
                return 1

        # Check if the point is inside
        p1x, p1y = poly[0]
        for i in range(1, poly_length + 1):
            p2x, p2y = poly[i % poly_length]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xints:
                                counter += 1
            p1x, p1y = p2x, p2y

    if counter % 2 == 0:
        return 0
    else:
        return 1


def point_in_polygons(x, y, polygon_list):
    """
    Check whether (x,y) is inside any polygon in a list of polygon

    Parameters
    ----------

    x
        x coordinate/longitude
    y
        y coordinate/latitude
    polygon_list
        A list of polygon(s)

    Returns
    -------
    Integer
        1 if (x, y) is inside any polygon of polygon_list and 0 otherwise.
    """

    flag_list = []
    for polygon in polygon_list:
        flag_list.append(point_in_polygon(x, y, polygon))
    return int(any(flag_list))


def f_point_in_polygons(polygon_list):
    return F.udf(lambda x, y: point_in_polygons(x, y, polygon_list), T.IntegerType())
