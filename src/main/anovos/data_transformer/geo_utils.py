from math import sin, cos, atan2, asin, radians, degrees, sqrt
import pygeohash as pgh
from geopy import distance
from scipy import spatial
import numbers
import warnings
from pyspark.sql import functions as F
from pyspark.sql import types as T

EARTH_RADIUS = 6371009
UNIT_FACTOR = {"m": 1.0, "km": 1000.0}


def in_range(loc, loc_format="dd"):
    """
    This function helps to check if the input location is in range based on loc_format.

    Parameters
    ----------
    loc
        Location to check if in range.
        If loc_format is "dd", [lat, lon] format is required.
        If loc_format is "dms", [[d1,m1,s1], [d2,m2,s2]] format is required.
        If loc_format is "radian", [lat_radian, lon_radian] format is required.
        If loc_format is "cartesian", [x, y, z] format is required.
        If loc_format is "geohash", string format is required.
    loc_format
        "dd", "dms", "radian", "cartesian", "geohash". (Default value = "dd")
    """
    if loc_format == "dd":
        try:
            lat, lon = [int(float(i)) for i in loc]
        except:
            lat, lon = None, None
    else:
        try:
            lat, lon = [
                int(float(i)) for i in to_latlon_decimal_degrees(loc, loc_format)
            ]
        except:
            lat, lon = None, None

    if None not in [lat, lon]:
        if lat > 90 or lat < -90 or lon > 180 or lon < -180:
            warnings.warn(
                "Rows may contain unintended values due to longitude and/or latitude values being out of the "
                "valid range"
            )


def to_latlon_decimal_degrees(loc, input_format, radius=EARTH_RADIUS):
    """
    This function helps to format input location into [lat,lon] format
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
        try:
            lat = float(loc[0])
            lon = float(loc[1])
        except:
            lat, lon = None, None
            warnings.warn(
                "Rows dropped due to invalid longitude and/or latitude values"
            )

    elif input_format == "dms":
        # loc = [[d1,m1,s1], [d2,m2,s2]]
        try:
            d1, m1, s1, d2, m2, s2 = [float(i) for i in (loc[0] + loc[1])]
            lat = d1 + m1 / 60 + s1 / 3600
            lon = d2 + m2 / 60 + s2 / 3600
        except:
            lat, lon = None, None
            warnings.warn(
                "Rows dropped due to invalid longitude and/or latitude values"
            )

    elif input_format == "radian":
        # loc = [lat_radian, lon_radian]
        try:
            lat = degrees(float(loc[0]))
            lon = degrees(float(loc[1]))
        except:
            lat, lon = None, None
            warnings.warn(
                "Rows dropped due to invalid longitude and/or latitude values"
            )

    elif input_format == "cartesian":
        # loc = [x, y, z]
        try:
            x, y, z = [float(i) for i in loc]
            lat = degrees(float(asin(z / radius)))
            lon = degrees(float(atan2(y, x)))
        except:
            lat, lon = None, None
            warnings.warn("Rows dropped due to invalid cartesian values")

    elif input_format == "geohash":
        # loc = geohash
        try:
            lat, lon = list(pgh.decode(loc))
        except:
            lat, lon = None, None
            warnings.warn("Rows dropped due to an invalid geohash entry")

    in_range((lat, lon))

    return [lat, lon]


def decimal_degrees_to_degrees_minutes_seconds(dd):
    """
    This function helps to divide float value dd in decimal degree into [degreee, minute, second]
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
    This function helps to transform [lat,lon] locations into desired output_format.
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
    This function helps to calculate the haversine distance between loc1 and loc2.
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

    try:
        lat1, lon1 = float(loc1[0]), float(loc1[1])
        lat2, lon2 = float(loc2[0]), float(loc2[1])
    except:
        return None

    in_range((lat1, lon1), loc_format)
    in_range((lat2, lon2), loc_format)

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

    in_range(loc1)
    in_range(loc2)

    try:
        loc_distance = distance.distance(loc1, loc2, ellipsoid=ellipsoid)
    except:
        return None

    if unit == "m":
        return loc_distance.m
    else:
        return loc_distance.km


def euclidean_distance(loc1, loc2, unit="m"):
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
    unit
        "m", "km".
        Unit of the result. (Default value = "m")

    Returns
    -------
    Float
    """
    if None in [loc1, loc2]:
        return None
    if None in loc1 + loc2:
        return None

    try:
        loc1 = [float(i) for i in loc1]
        loc2 = [float(i) for i in loc2]
    except:
        return None

    in_range(loc1, "cartesian")
    in_range(loc2, "cartesian")

    # loc1 = [x1, y1, z1]; loc2 = [x2, y2, z2]
    euclidean_distance = spatial.distance.euclidean(loc1, loc2)

    if unit == "km":
        euclidean_distance /= 1000

    return euclidean_distance


def point_in_polygon(x, y, polygon):
    """
    This function helps to check whether (x,y) is inside a polygon

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

    try:
        x = float(x)
        y = float(y)
    except:
        return None

    in_range((y, x))

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
        for i in range(poly_length - 1):
            p1 = poly[i]
            p2 = poly[i + 1]
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
        for i in range(poly_length):
            p1x, p1y = poly[i]
            p2x, p2y = poly[(i + 1) % poly_length]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xints:
                                counter += 1

    if counter % 2 == 0:
        return 0
    else:
        return 1


def point_in_polygons(x, y, polygon_list, south_west_loc=[], north_east_loc=[]):
    """
    This function helps to check whether (x,y) is inside any polygon in a list of polygon

    Parameters
    ----------
    x
        x coordinate/longitude
    y
        y coordinate/latitude
    polygon_list
        A list of polygon(s)
    south_west_loc
        The south-west point (x_sw, y_sw) of the bounding box of the polygons, if available.
        0 will be directly returned if x < x_sw or y < y_sw (Default value = [])
    north_east_loc
        The north-east point (x_ne, y_ne) of the bounding box of the polygons, if available. (Default value = [])
        0 will be directly returned if x > x_ne or y > y_ne (Default value = [])

    Returns
    -------
    Integer
        1 if (x, y) is inside any polygon of polygon_list and 0 otherwise.
    """
    if (x is None) | (y is None):
        return None

    try:
        x = float(x)
        y = float(y)
    except:
        warnings.warn("Rows dropped due to invalid longitude and/or latitude values")
        return None

    in_range((y, x))

    if south_west_loc:
        if (x < south_west_loc[0]) or (y < south_west_loc[1]):
            return 0

    if north_east_loc:
        if (x > north_east_loc[0]) or (y > north_east_loc[1]):
            return 0

    flag_list = []
    for polygon in polygon_list:
        flag_list.append(point_in_polygon(x, y, polygon))
    return int(any(flag_list))


def f_point_in_polygons(polygon_list, south_west_loc=[], north_east_loc=[]):
    return F.udf(
        lambda x, y: point_in_polygons(
            x, y, polygon_list, south_west_loc, north_east_loc
        ),
        T.IntegerType(),
    )


COUNTRY_BOUNDING_BOXES = {
    "AW": ("Aruba", (-70.2809842, 12.1702998, -69.6409842, 12.8102998)),
    "AF": ("Afghanistan", (60.5176034, 29.3772, 74.889862, 38.4910682)),
    "AO": ("Angola", (11.4609793, -18.038945, 24.0878856, -4.3880634)),
    "AI": ("Anguilla", (-63.6391992, 18.0615454, -62.7125449, 18.7951194)),
    "AL": ("Albania", (19.1246095, 39.6448625, 21.0574335, 42.6610848)),
    "AD": ("Andorra", (1.4135781, 42.4288238, 1.7863837, 42.6559357)),
    "AE": ("United Arab Emirates", (51.498, 22.6444, 56.3834, 26.2822)),
    "AR": ("Argentina", (-73.5600329, -55.1850761, -53.6374515, -21.781168)),
    "AM": ("Armenia", (43.4471395, 38.8404775, 46.6333087, 41.300712)),
    "AS": ("American Samoa", (-171.2951296, -14.7608358, -167.9322899, -10.8449746)),
    "AQ": ("Antarctica", (-180.0, -85.0511287, 180.0, -60.0)),
    "AG": ("Antigua and Barbuda", (-62.5536517, 16.7573901, -61.447857, 17.929)),
    "AU": ("Australia", (72.2460938, -55.3228175, 168.2249543, -9.0882278)),
    "AT": ("Austria", (9.5307487, 46.3722761, 17.160776, 49.0205305)),
    "AZ": ("Azerbaijan", (44.7633701, 38.3929551, 51.0090302, 41.9502947)),
    "BI": ("Burundi", (29.0007401, -4.4693155, 30.8498462, -2.3096796)),
    "BE": ("Belgium", (2.3889137, 49.4969821, 6.408097, 51.5516667)),
    "BJ": ("Benin", (0.776667, 6.0398696, 3.843343, 12.4092447)),
    "BF": ("Burkina Faso", (-5.5132416, 9.4104718, 2.4089717, 15.084)),
    "BD": ("Bangladesh", (88.0075306, 20.3756582, 92.6804979, 26.6382534)),
    "BG": ("Bulgaria", (22.3571459, 41.2353929, 28.8875409, 44.2167064)),
    "BH": ("Bahrain", (50.2697989, 25.535, 50.9233693, 26.6872444)),
    "BS": ("Bahamas", (-80.7001941, 20.7059846, -72.4477521, 27.4734551)),
    "BA": ("Bosnia and Herzegovina", (15.7287433, 42.5553114, 19.6237311, 45.2764135)),
    "BL": ("Saint Barthélemy", (-63.06639, 17.670931, -62.5844019, 18.1375569)),
    "BY": ("Belarus", (23.1783344, 51.2575982, 32.7627809, 56.17218)),
    "BZ": ("Belize", (-89.2262083, 15.8857286, -87.3098494, 18.496001)),
    "BM": ("Bermuda", (-65.1232222, 32.0469651, -64.4109842, 32.5913693)),
    "BO": (
        "Bolivia (Plurinational State of)",
        (-69.6450073, -22.8982742, -57.453, -9.6689438),
    ),
    "BR": ("Brazil", (-73.9830625, -33.8689056, -28.6341164, 5.2842873)),
    "BB": ("Barbados", (-59.8562115, 12.845, -59.2147175, 13.535)),
    "BN": ("Brunei Darussalam", (114.0758734, 4.002508, 115.3635623, 5.1011857)),
    "BT": ("Bhutan", (88.7464724, 26.702016, 92.1252321, 28.246987)),
    "BW": ("Botswana", (19.9986474, -26.9059669, 29.375304, -17.778137)),
    "CF": ("Central African Republic", (14.4155426, 2.2156553, 27.4540764, 11.001389)),
    "CA": ("Canada", (-141.00275, 41.6765556, -52.3231981, 83.3362128)),
    "CC": (
        "Cocos (Keeling) Islands",
        (96.612524, -12.4055983, 97.1357343, -11.6213132),
    ),
    "CH": ("Switzerland", (5.9559113, 45.817995, 10.4922941, 47.8084648)),
    "CL": ("Chile", (-109.6795789, -56.725, -66.0753474, -17.4983998)),
    "CN": ("China", (73.4997347, 8.8383436, 134.7754563, 53.5608154)),
    "CI": ("Côte d'Ivoire", (-8.601725, 4.1621205, -2.493031, 10.740197)),
    "CM": ("Cameroon", (8.3822176, 1.6546659, 16.1921476, 13.083333)),
    "CD": (
        "Congo, Democratic Republic of the",
        (12.039074, -13.459035, 31.3056758, 5.3920026),
    ),
    "CG": ("Congo", (11.0048205, -5.149089, 18.643611, 3.713056)),
    "CK": ("Cook Islands", (-166.0856468, -22.15807, -157.1089329, -8.7168792)),
    "CO": ("Colombia", (-82.1243666, -4.2316872, -66.8511907, 16.0571269)),
    "KM": ("Comoros", (43.025305, -12.621, 44.7451922, -11.165)),
    "CV": ("Cabo Verde", (-25.3609478, 14.8031546, -22.6673416, 17.2053108)),
    "CR": ("Costa Rica", (-87.2722647, 5.3329698, -82.5060208, 11.2195684)),
    "CU": ("Cuba", (-85.1679702, 19.6275294, -73.9190004, 23.4816972)),
    "CX": ("Christmas Island", (105.5336422, -10.5698515, 105.7130159, -10.4123553)),
    "KY": ("Cayman Islands", (-81.6313748, 19.0620619, -79.5110954, 19.9573759)),
    "CY": ("Cyprus", (32.0227581, 34.4383706, 34.8553182, 35.913252)),
    "CZ": ("Czechia", (12.0905901, 48.5518083, 18.859216, 51.0557036)),
    "DE": ("Germany", (5.8663153, 47.2701114, 15.0419319, 55.099161)),
    "DJ": ("Djibouti", (41.7713139, 10.9149547, 43.6579046, 12.7923081)),
    "DM": ("Dominica", (-61.6869184, 15.0074207, -61.0329895, 15.7872222)),
    "DK": ("Denmark", (7.7153255, 54.4516667, 15.5530641, 57.9524297)),
    "DO": ("Dominican Republic", (-72.0574706, 17.2701708, -68.1101463, 21.303433)),
    "DZ": ("Algeria", (-8.668908, 18.968147, 11.997337, 37.2962055)),
    "EC": ("Ecuador", (-92.2072392, -5.0159314, -75.192504, 1.8835964)),
    "EG": ("Egypt", (24.6499112, 22.0, 37.1153517, 31.8330854)),
    "ER": ("Eritrea", (36.4333653, 12.3548219, 43.3001714, 18.0709917)),
    "EH": ("Western Sahara", (-17.3494721, 20.556883, -8.666389, 27.6666834)),
    "ES": ("Spain", (-18.3936845, 27.4335426, 4.5918885, 43.9933088)),
    "EE": ("Estonia", (21.3826069, 57.5092997, 28.2100175, 59.9383754)),
    "ET": ("Ethiopia", (32.9975838, 3.397448, 47.9823797, 14.8940537)),
    "FI": ("Finland", (19.0832098, 59.4541578, 31.5867071, 70.0922939)),
    "FJ": ("Fiji", (172.0, -21.9434274, -178.5, -12.2613866)),
    "FK": (
        "Falkland Islands (Malvinas)",
        (-61.7726772, -53.1186766, -57.3662367, -50.7973007),
    ),
    "FR": ("France", (-5.4534286, 41.2632185, 9.8678344, 51.268318)),
    "FO": ("Faroe Islands", (-7.6882939, 61.3915553, -6.2565525, 62.3942991)),
    "FM": (
        "Micronesia (Federated States of)",
        (137.2234512, 0.827, 163.2364054, 10.291),
    ),
    "GA": ("Gabon", (8.5002246, -4.1012261, 14.539444, 2.3182171)),
    "GB": (
        "United Kingdom of Great Britain and Northern Ireland",
        (-14.015517, 49.674, 2.0919117, 61.061),
    ),
    "GE": ("Georgia", (39.8844803, 41.0552922, 46.7365373, 43.5864294)),
    "GG": ("Guernsey", (-2.6751703, 49.4155331, -2.501814, 49.5090776)),
    "GH": ("Ghana", (-3.260786, 4.5392525, 1.2732942, 11.1748562)),
    "GI": ("Gibraltar", (-5.3941295, 36.100807, -5.3141295, 36.180807)),
    "GN": ("Guinea", (-15.5680508, 7.1906045, -7.6381993, 12.67563)),
    "GM": ("Gambia", (-17.0288254, 13.061, -13.797778, 13.8253137)),
    "GW": ("Guinea-Bissau", (-16.894523, 10.6514215, -13.6348777, 12.6862384)),
    "GQ": ("Equatorial Guinea", (5.4172943, -1.6732196, 11.3598628, 3.989)),
    "GR": ("Greece", (19.2477876, 34.7006096, 29.7296986, 41.7488862)),
    "GD": ("Grenada", (-62.0065868, 11.786, -61.1732143, 12.5966532)),
    "GL": ("Greenland", (-74.1250416, 59.515387, -10.0288759, 83.875172)),
    "GT": ("Guatemala", (-92.3105242, 13.6345804, -88.1755849, 17.8165947)),
    "GU": ("Guam", (144.563426, 13.182335, 145.009167, 13.706179)),
    "GY": ("Guyana", (-61.414905, 1.1710017, -56.4689543, 8.6038842)),
    "HK": ("Hong Kong", (114.0028131, 22.1193278, 114.3228131, 22.4393278)),
    "HN": ("Honduras", (-89.3568207, 12.9808485, -82.1729621, 17.619526)),
    "HR": ("Croatia", (13.2104814, 42.1765993, 19.4470842, 46.555029)),
    "HT": ("Haiti", (-75.2384618, 17.9099291, -71.6217461, 20.2181368)),
    "HU": ("Hungary", (16.1138867, 45.737128, 22.8977094, 48.585257)),
    "ID": ("Indonesia", (94.7717124, -11.2085669, 141.0194444, 6.2744496)),
    "IM": ("Isle of Man", (-4.7946845, 54.0539576, -4.3076853, 54.4178705)),
    "IN": ("India", (68.1113787, 6.5546079, 97.395561, 35.6745457)),
    "IO": (
        "British Indian Ocean Territory",
        (71.036504, -7.6454079, 72.7020157, -5.037066),
    ),
    "IE": ("Ireland", (-11.0133788, 51.222, -5.6582363, 55.636)),
    "IR": (
        "Iran (Islamic Republic of)",
        (44.0318908, 24.8465103, 63.3332704, 39.7816502),
    ),
    "IQ": ("Iraq", (38.7936719, 29.0585661, 48.8412702, 37.380932)),
    "IS": ("Iceland", (-25.0135069, 63.0859177, -12.8046162, 67.353)),
    "IL": ("Israel", (34.2674994, 29.4533796, 35.8950234, 33.3356317)),
    "IT": ("Italy", (6.6272658, 35.2889616, 18.7844746, 47.0921462)),
    "JM": ("Jamaica", (-78.5782366, 16.5899443, -75.7541143, 18.7256394)),
    "JE": ("Jersey", (-2.254512, 49.1625179, -2.0104193, 49.2621288)),
    "JO": ("Jordan", (34.8844372, 29.183401, 39.3012981, 33.3750617)),
    "JP": ("Japan", (122.7141754, 20.2145811, 154.205541, 45.7112046)),
    "KZ": ("Kazakhstan", (46.4932179, 40.5686476, 87.3156316, 55.4421701)),
    "KE": ("Kenya", (33.9098987, -4.8995204, 41.899578, 4.62)),
    "KG": ("Kyrgyzstan", (69.2649523, 39.1728437, 80.2295793, 43.2667971)),
    "KH": ("Cambodia", (102.3338282, 9.4752639, 107.6276788, 14.6904224)),
    "KI": ("Kiribati", (-179.1645388, -7.0516717, -164.1645388, 7.9483283)),
    "KN": ("Saint Kitts and Nevis", (-63.051129, 16.895, -62.3303519, 17.6158146)),
    "KR": ("Korea, Republic of", (124.354847, 32.9104556, 132.1467806, 38.623477)),
    "KW": ("Kuwait", (46.5526837, 28.5243622, 49.0046809, 30.1038082)),
    "LA": (
        "Lao People's Democratic Republic",
        (100.0843247, 13.9096752, 107.6349989, 22.5086717),
    ),
    "LB": ("Lebanon", (34.8825667, 33.0479858, 36.625, 34.6923543)),
    "LR": ("Liberia", (-11.6080764, 4.1555907, -7.367323, 8.5519861)),
    "LY": ("Libya", (9.391081, 19.5008138, 25.3770629, 33.3545898)),
    "LC": ("Saint Lucia", (-61.2853867, 13.508, -60.6669363, 14.2725)),
    "LI": ("Liechtenstein", (9.4716736, 47.0484291, 9.6357143, 47.270581)),
    "LK": ("Sri Lanka", (79.3959205, 5.719, 82.0810141, 10.035)),
    "LS": ("Lesotho", (27.0114632, -30.6772773, 29.4557099, -28.570615)),
    "LT": ("Lithuania", (20.653783, 53.8967893, 26.8355198, 56.4504213)),
    "LU": ("Luxembourg", (4.9684415, 49.4969821, 6.0344254, 50.430377)),
    "LV": ("Latvia", (20.6715407, 55.6746505, 28.2414904, 58.0855688)),
    "MO": ("Macao", (113.5281666, 22.0766667, 113.6301389, 22.2170361)),
    "MF": (
        "Saint Martin (French part)",
        (-63.3605643, 17.8963535, -62.7644063, 18.1902778),
    ),
    "MA": ("Morocco", (-17.2551456, 21.3365321, -0.998429, 36.0505269)),
    "MC": ("Monaco", (7.4090279, 43.7247599, 7.4398704, 43.7519311)),
    "MD": ("Moldova, Republic of", (26.6162189, 45.4674139, 30.1636756, 48.4918695)),
    "MG": ("Madagascar", (43.2202072, -25.6071002, 50.4862553, -11.9519693)),
    "MV": ("Maldives", (72.3554187, -0.9074935, 73.9700962, 7.3106246)),
    "MX": ("Mexico", (-118.59919, 14.3886243, -86.493266, 32.7186553)),
    "MH": ("Marshall Islands", (163.4985095, -0.5481258, 178.4985095, 14.4518742)),
    "MK": ("North Macedonia", (20.4529023, 40.8536596, 23.034051, 42.3735359)),
    "ML": ("Mali", (-12.2402835, 10.147811, 4.2673828, 25.001084)),
    "MT": ("Malta", (13.9324226, 35.6029696, 14.8267966, 36.2852706)),
    "MM": ("Myanmar", (92.1719423, 9.4399432, 101.1700796, 28.547835)),
    "ME": ("Montenegro", (18.4195781, 41.7495999, 20.3561641, 43.5585061)),
    "MN": ("Mongolia", (87.73762, 41.5800276, 119.931949, 52.1496)),
    "MP": ("Northern Mariana Islands", (144.813338, 14.036565, 146.154418, 20.616556)),
    "MZ": ("Mozambique", (30.2138197, -26.9209427, 41.0545908, -10.3252149)),
    "MR": ("Mauritania", (-17.068081, 14.7209909, -4.8333344, 27.314942)),
    "MS": ("Montserrat", (-62.450667, 16.475, -61.9353818, 17.0152978)),
    "MU": ("Mauritius", (56.3825151, -20.725, 63.7151319, -10.138)),
    "MW": ("Malawi", (32.6703616, -17.1296031, 35.9185731, -9.3683261)),
    "MY": ("Malaysia", (105.3471939, -5.1076241, 120.3471939, 9.8923759)),
    "YT": ("Mayotte", (45.0183298, -13.0210119, 45.2999917, -12.6365902)),
    "NA": ("Namibia", (11.5280384, -28.96945, 25.2617671, -16.9634855)),
    "NC": ("New Caledonia", (162.6034343, -23.2217509, 167.8109827, -17.6868616)),
    "NE": ("Niger", (0.1689653, 11.693756, 15.996667, 23.517178)),
    "NG": ("Nigeria", (2.676932, 4.0690959, 14.678014, 13.885645)),
    "NI": ("Nicaragua", (-87.901532, 10.7076565, -82.6227023, 15.0331183)),
    "NU": ("Niue", (-170.1595029, -19.3548665, -169.5647229, -18.7534559)),
    "NL": ("Netherlands", (1.9193492, 50.7295671, 7.2274985, 53.7253321)),
    "NO": ("Norway", (4.0875274, 57.7590052, 31.7614911, 71.3848787)),
    "NP": ("Nepal", (80.0586226, 26.3477581, 88.2015257, 30.446945)),
    "NR": ("Nauru", (166.9091794, -0.5541334, 166.9589235, -0.5025906)),
    "NZ": ("New Zealand", (-179.059153, -52.8213687, 179.3643594, -29.0303303)),
    "OM": ("Oman", (52, 16.4649608, 60.054577, 26.7026737)),
    "PK": ("Pakistan", (60.872855, 23.5393916, 77.1203914, 37.084107)),
    "PA": ("Panama", (-83.0517245, 7.0338679, -77.1393779, 9.8701757)),
    "PN": ("Pitcairn", (-130.8049862, -25.1306736, -124.717534, -23.8655769)),
    "PE": ("Peru", (-84.6356535, -20.1984472, -68.6519906, -0.0392818)),
    "PH": ("Philippines", (114.0952145, 4.2158064, 126.8072562, 21.3217806)),
    "PW": ("Palau", (131.0685462, 2.748, 134.7714735, 8.222)),
    "PG": ("Papua New Guinea", (136.7489081, -13.1816069, 151.7489081, 1.8183931)),
    "PL": ("Poland", (14.1229707, 49.0020468, 24.145783, 55.0336963)),
    "PR": ("Puerto Rico", (-67.271492, 17.9268695, -65.5897525, 18.5159789)),
    "KP": (
        "Korea (Democratic People's Republic of)",
        (124.0913902, 37.5867855, 130.924647, 43.0089642),
    ),
    "PT": ("Portugal", (-31.5575303, 29.8288021, -6.1891593, 42.1543112)),
    "PY": ("Paraguay", (-62.6442036, -27.6063935, -54.258, -19.2876472)),
    "PS": ("Palestine, State of", (34.0689732, 31.2201289, 35.5739235, 32.5521479)),
    "PF": ("French Polynesia", (-154.9360599, -28.0990232, -134.244799, -7.6592173)),
    "QA": ("Qatar", (50.5675, 24.4707534, 52.638011, 26.3830212)),
    "RE": ("Réunion", (55.2164268, -21.3897308, 55.8366924, -20.8717136)),
    "RO": ("Romania", (20.2619773, 43.618682, 30.0454257, 48.2653964)),
    "RU": ("Russian Federation", (19.6389, 41.1850968, 180, 82.0586232)),
    "RW": ("Rwanda", (28.8617546, -2.8389804, 30.8990738, -1.0474083)),
    "SA": ("Saudi Arabia", (34.4571718, 16.29, 55.6666851, 32.1543377)),
    "SD": ("Sudan", (21.8145046, 8.685278, 39.0576252, 22.224918)),
    "SN": ("Senegal", (-17.7862419, 12.2372838, -11.3458996, 16.6919712)),
    "SG": ("Singapore", (103.6920359, 1.1304753, 104.0120359, 1.4504753)),
    "SH": (
        "Saint Helena, Ascension and Tristan da Cunha",
        (-5.9973424, -16.23, -5.4234153, -15.704),
    ),
    "SJ": ("Svalbard and Jan Mayen", (-9.6848146, 70.6260825, 34.6891253, 81.028076)),
    "SB": ("Solomon Islands", (155.3190556, -13.2424298, 170.3964667, -4.81085)),
    "SL": ("Sierra Leone", (-13.5003389, 6.755, -10.271683, 9.999973)),
    "SV": ("El Salvador", (-90.1790975, 12.976046, -87.6351394, 14.4510488)),
    "SM": ("San Marino", (12.4033246, 43.8937002, 12.5160665, 43.992093)),
    "SO": ("Somalia", (40.98918, -1.8031969, 51.6177696, 12.1889121)),
    "PM": ("Saint Pierre and Miquelon", (-56.6972961, 46.5507173, -55.9033333, 47.365)),
    "RS": ("Serbia", (18.8142875, 42.2322435, 23.006309, 46.1900524)),
    "ST": ("Sao Tome and Principe", (6.260642, -0.2135137, 7.6704783, 1.9257601)),
    "SR": ("Suriname", (-58.070833, 1.8312802, -53.8433358, 6.225)),
    "SK": ("Slovakia", (16.8331891, 47.7314286, 22.56571, 49.6138162)),
    "SI": ("Slovenia", (13.3754696, 45.4214242, 16.5967702, 46.8766816)),
    "SE": ("Sweden", (10.5930952, 55.1331192, 24.1776819, 69.0599699)),
    "SZ": ("Eswatini", (30.7908, -27.3175201, 32.1349923, -25.71876)),
    "SC": ("Seychelles", (45.9988759, -10.4649258, 56.4979396, -3.512)),
    "SY": ("Syrian Arab Republic", (35.4714427, 32.311354, 42.3745687, 37.3184589)),
    "TC": (
        "Turks and Caicos Islands",
        (-72.6799046, 20.9553418, -70.8643591, 22.1630989),
    ),
    "TD": ("Chad", (13.47348, 7.44107, 24.0, 23.4975)),
    "TG": ("Togo", (-0.1439746, 5.926547, 1.8087605, 11.1395102)),
    "TH": ("Thailand", (97.3438072, 5.612851, 105.636812, 20.4648337)),
    "TJ": ("Tajikistan", (67.3332775, 36.6711153, 75.1539563, 41.0450935)),
    "TK": ("Tokelau", (-172.7213673, -9.6442499, -170.9797586, -8.3328631)),
    "TM": ("Turkmenistan", (52.335076, 35.129093, 66.6895177, 42.7975571)),
    "TL": ("Timor-Leste", (124.0415703, -9.5642775, 127.5335392, -8.0895459)),
    "TO": ("Tonga", (-179.3866055, -24.1034499, -173.5295458, -15.3655722)),
    "TT": ("Trinidad and Tobago", (-62.083056, 9.8732106, -60.2895848, 11.5628372)),
    "TN": ("Tunisia", (7.5219807, 30.230236, 11.8801133, 37.7612052)),
    "TR": ("Turkey", (25.6212891, 35.8076804, 44.8176638, 42.297)),
    "TV": ("Tuvalu", (175.1590468, -9.9939389, 178.7344938, -5.4369611)),
    "TW": ("Taiwan, Province of China", (114.3599058, 10.374269, 122.297, 26.4372222)),
    "TZ": (
        "Tanzania, United Republic of",
        (29.3269773, -11.761254, 40.6584071, -0.9854812),
    ),
    "UG": ("Uganda", (29.573433, -1.4823179, 35.000308, 4.2340766)),
    "UA": ("Ukraine", (22.137059, 44.184598, 40.2275801, 52.3791473)),
    "UY": ("Uruguay", (-58.4948438, -35.7824481, -53.0755833, -30.0853962)),
    "US": ("United States of America", (-125.0011, 24.9493, -66.9326, 49.5904)),
    "UZ": ("Uzbekistan", (55.9977865, 37.1821164, 73.1397362, 45.590118)),
    "VA": ("Holy See", (12.4457442, 41.9002044, 12.4583653, 41.9073912)),
    "VC": (
        "Saint Vincent and the Grenadines",
        (-61.6657471, 12.5166548, -60.9094146, 13.583),
    ),
    "VE": (
        "Venezuela (Bolivarian Republic of)",
        (-73.3529632, 0.647529, -59.5427079, 15.9158431),
    ),
    "VG": ("Virgin Islands (British)", (-65.159094, 17.623468, -64.512674, 18.464984)),
    "VI": ("Virgin Islands (U.S.)", (-65.159094, 17.623468, -64.512674, 18.464984)),
    "VN": ("Viet Nam", (102.14441, 8.1790665, 114.3337595, 23.393395)),
    "VU": ("Vanuatu", (166.3355255, -20.4627425, 170.449982, -12.8713777)),
    "WF": ("Wallis and Futuna", (-178.3873749, -14.5630748, -175.9190391, -12.9827961)),
    "WS": ("Samoa", (-173.0091864, -14.2770916, -171.1929229, -13.2381892)),
    "YE": ("Yemen", (41.60825, 11.9084802, 54.7389375, 19.0)),
    "ZA": ("South Africa", (16.3335213, -47.1788335, 38.2898954, -22.1250301)),
    "ZM": ("Zambia", (21.9993509, -18.0765945, 33.701111, -8.2712822)),
    "ZW": ("Zimbabwe", (25.2373, -22.4241096, 33.0683413, -15.6097033)),
}


def point_in_country_approx(lat, lon, country):
    c = COUNTRY_BOUNDING_BOXES[country]

    if (lat is None) | (lon is None):
        return None

    try:
        lat = float(lat)
        lon = float(lon)
    except:
        warnings.warn("Rows dropped due to invalid longitude and/or latitude values")
        return None

    in_range((lat, lon))

    if (c[1][1] <= lat <= c[1][3]) and (c[1][0] <= lon <= c[1][2]):
        return 1
    else:
        return 0
