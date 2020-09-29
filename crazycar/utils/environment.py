def get_observation_shape(x):
    """
    Get observation shape for each key (recursive)

    Args:
        x: dictionary of observation

    Returns:
        dictionary of shape
    """

    d = {}
    if isinstance(x, dict):
        for k in x.keys():
            d[k] = get_observation_shape(x[k])
    else:
        return x.shape
    return d
