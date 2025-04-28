import re
from pathlib import Path

timestamp_pattern = re.compile("^(?P<timestamp>[0-9]{8}_[0-9]{6})_")
slide_pattern = re.compile("(?:^|_)S(?P<slide>[0-9])(?:_|$)")
cycle_pattern = re.compile("(?:^|_)C(?P<cycle>[0-9]+)(?:_|[.]|$)")
pool_pattern = re.compile("(?:^|_)P(?P<pool>[0-9]+)(?:_|[.]|$)")
spot_pattern = re.compile("(?:^|_)N(?P<spot>[0-9]+)(?:_|[.]|$)")
fov_pattern = re.compile("(?:^|_)F(?:OV)?(?P<fov>[0-9]+)(?:_|[.]|$)")
zslice_pattern = re.compile("(?:^|_)Z(?P<zslice>[0-9]+)(?:_|[.]|$)")

def get_fov_number(filepath):
    filename = Path(filepath).name
    m = fov_pattern.search(filename)
    return int(m['fov']) if m else None

def convertLabels(x: str, labels: dict, to_name=True) -> str:
    """Finds the protein name or label

    Args:
        x (str): the protein name or label
        labels (dict): protein dictionary with ProbeID as keys.
        to_name (bool, optional): True if converting label to name; False if reverse. Defaults to True.

    Returns:
        str: if to_name is True, the protein name; else, the protein label
    """
    if to_name:
        return labels.get(x)  

    else:  
        for label, name in labels.items():
            if name == x:
                return label