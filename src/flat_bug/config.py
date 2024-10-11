import os, yaml
from collections import OrderedDict
from typing import List, Iterable,Union, Any

from flat_bug import logger

# Add support for OrderedDict in PyYAML
yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_dict(data.items()), Dumper=yaml.SafeDumper)

CFG_PARAMS = [
    "SCORE_THRESHOLD",
    "IOU_THRESHOLD",
    "MINIMUM_TILE_OVERLAP",
    "EDGE_CASE_MARGIN",
    "MIN_MAX_OBJ_SIZE",
    "MAX_MASK_SIZE",
    "PREFER_POLYGONS",
    "EXPERIMENTAL_NMS_OPTIMIZATION",
    "TIME",
    "TILE_SIZE",
    "BATCH_SIZE"
]

CFG_DESCRIPTION = {
    "SCORE_THRESHOLD": "Minimum score for a prediction to be considered.",
    "IOU_THRESHOLD": "Minimum IOU for a prediction to be considered a duplicate.",
    "MINIMUM_TILE_OVERLAP": "Minimum overlap between tiles when splitting the image.",
    "EDGE_CASE_MARGIN": "Margin for edge cases. How far from the edge of the image a prediction can be.",
    "MIN_MAX_OBJ_SIZE": "Minimum and maximum size of a bounding box. Size is measured as the square root of the area, i.e. the side-length if the bounding box was a square.",
    "MAX_MASK_SIZE": "Loss of precision may occur if the mask is larger than this, but all shapes are possible. No effect if PREFER_POLYGONS is enabled.",
    "PREFER_POLYGONS": "Convert masks to polygons as soon as possible, and only use the polygons for further processing - no loss of precision, but only single polygons without holes can be represented, performance impact may depend on hardware and use-case.",
    "EXPERIMENTAL_NMS_OPTIMIZATION": "Experimental optimization for NMS. Improves performance significantly when there are many predictions.",
    "TIME": "Enable to print time taken for each step. Can incur a performance penalty.",
    "TILE_SIZE": "Fixed by the model architecture - do not change unless you know what you are doing.",
    "BATCH_SIZE": "Used for model initialization and batched tile processing."
}

DEFAULT_CFG = {
    "SCORE_THRESHOLD": 0.2,
    "IOU_THRESHOLD": 0.2,
    "MINIMUM_TILE_OVERLAP": 384,
    "EDGE_CASE_MARGIN": 8,
    "MIN_MAX_OBJ_SIZE": (32, 10**8),
    "MAX_MASK_SIZE": 1024,
    "PREFER_POLYGONS": True,
    "EXPERIMENTAL_NMS_OPTIMIZATION": True,
    "TIME": False,
    "TILE_SIZE": 1024,
    "BATCH_SIZE": 16
}

LEGACY_CFG = {
    "SCORE_THRESHOLD": 0.5,
    "IOU_THRESHOLD": 0.5,
    "MINIMUM_TILE_OVERLAP": 256,
    "EDGE_CASE_MARGIN": 128,
    "MIN_MAX_OBJ_SIZE": (16, 1024),
    "MAX_MASK_SIZE": 1024,
    "PREFER_POLYGONS": True,
    "EXPERIMENTAL_NMS_OPTIMIZATION": True,
    "TIME": False,
    "TILE_SIZE": 1024,
    "BATCH_SIZE": 16
}

def get_type_def(
        obj : Any, 
        tuple_list_interchangeable : bool=False
    ) -> Union[Any, List[Any]]:
    """
    Generates a dynamic type definition for an object.

    The type definition schema is defined like this:
    - If the object is a tuple or a list, the first element is the type of the object, and the second element is a list of type definitions for the elements of the object.
    - If the object is not a tuple or a list, the type definition is the type of the object.

    For example;
        - the type definition for the object `(1, "A", True)` would be `[tuple, [int, str, bool]]`.
        - the type definition for the object `[[2, "B"], [3, "C"]]` would be `[list, [[list, [int, str]], [list, [int, str]]]]`.

    Parameters:
        obj (Any): The object to generate a type definition for.
        tuple_list_interchangeable (bool): If True, tuples and lists are considered interchangeable.

    Returns:
        Union[Any, List[Any]]: The type definition for the object.
    """
    if isinstance(obj, (tuple, list)):
        otype = type(obj)
        if tuple_list_interchangeable and otype in (tuple, list):
            otype = (tuple, list)
        return [otype, [get_type_def(i, tuple_list_interchangeable) for i in obj]]
    return type(obj)

CFG_TYPES = {k : get_type_def(DEFAULT_CFG[k], tuple_list_interchangeable=True) for k in DEFAULT_CFG}

def check_types(
        value : Any, 
        expected_type : Union[List[Any], Iterable[type], type], 
        key : str="<Not specified>", 
        strict : bool=True
    ) -> bool:
    """
    Recursively check if the type of a value matches the expected type.

    If the expected type is a list, the first element is the type of the value, and the second element is a list of types that the elements of the value match, a single type that all elements should match or a tuple/type of types that all elements should match any of.

    Parameters:
        value: The value to check.
        expected_type: The expected type of the value.
        key: Name of the value to use in error messages.
        strict: If True, raise an error if the check fails.

    Returns:
        bool: True if the check passes, and False if strict is False and the check fails. Raises an error otherwise.

    Raises:
        ValueError: If the expected type list does not have exactly 2 elements.
        TypeError: If the expected type is not a list, an iterable or a 'type' object.
        TypeError: If the number of types in the list does not match the number of items in the value.
        TypeError: If the value does not match the expected type.
    """
    try:
        # If expected type is a list, recursively check the types of the elements
        if isinstance(expected_type, list):
            # Check that an expected type has been supplied for both the value and its elements
            if len(expected_type) != 2:
                raise ValueError(f"Expected type list must have exactly 2 elements, got {len(expected_type)} for key: {key}.")
            # Check that the value matches the expected type
            check_types(value, expected_type[0], key, strict)
            # If the expected type of the elements is a list, each element of the value should match the corresponding element of the expected type list
            if isinstance(expected_type[1], list):
                # Check that the number of types in the list matches the number of items in the value
                if len(value) != len(expected_type[1]):
                    raise TypeError(f"Expected number of types ({len(expected_type[1])}) does not match number of items in value ({len(value)}) for key: {key}.")
                # Check that each item in the value matches the corresponding type in the expected type list
                for item, et in zip(value, expected_type[1]):
                    check_types(item, et, key, strict)
            # If the expected type of the elements is a single type, each element of the value should match the expected type
            elif isinstance(expected_type[1], type):
                for item in value:
                    check_types(item, expected_type[1], key, strict)
            # If the expected type of the elements is a tuple, each element of the value should match any of the types in the tuple
            elif isinstance(expected_type[1], tuple):
                check_types(value, expected_type[1], key, strict)
            # If the expected type of the elements is an iterable, the value should be an iterable and each element of the value should match the corresponding type in the expected type iterable
            elif hasattr(expected_type[1], "__iter__") and hasattr(expected_type[1], "__len__"):
                assert len(expected_type[1]) == len(value), f"Expected number of types ({len(expected_type[1])}) does not match number of items in value ({len(value)}) for key: {key}."
                errors = []
                for item, et in zip(value, expected_type[1]):
                    try:
                        check_types(item, et, key, strict)
                    except TypeError as e:
                        errors.append(str(e))
                if len(errors) != 0:
                    raise TypeError("\n  - ".join(errors))
            else:
                raise TypeError(f"Invalid expected type. Expected 'list', 'type', 'tuple' or an iterable got {type(expected_type[1])} for key: {key}.")
        # If the expected type is an iterable, check if the value is an instance of any of the types in the iterable
        elif hasattr(expected_type, "__iter__") and not isinstance(expected_type, type):
            if not any([check_types(value, e, key, False) for e in expected_type]):
                raise TypeError(f"Expected one of {et}, got {type(value)} for key: {key}.")
        # If the expected type is a 'type' object, check if the value is an instance of the type
        elif isinstance(expected_type, type):
            if not isinstance(value, expected_type):
                raise TypeError(f"Expected {expected_type}, got {type(value)} for key {key}.")
        # If the expected type is None, check if the value is None
        elif expected_type is None:
            if not value is None:
                raise TypeError(f"Expected None, got {type(value)} for key: {key}.")
        # If the expected type is typing.Any, pass everything
        elif expected_type is Any:
            pass
        # If the expected type is not a list, a iterable or a 'type' object raise an error
        else:
            raise TypeError(f"Invalid expected type. Expected 'list', an iterable, 'type' or 'typing.Any' got {type(expected_type)} for key: {key}.")
        # If no errors are raised, return True
        return True
    # If an error is raised, return False if strict is False, otherwise raise the error
    except Exception as e:
        if strict:
            raise e
        else:
            return False

def check_cfg_types(
        cfg : dict, 
        strict : bool = False
    ) -> bool:
    """
    Check if the config is a dictionary and that the types of the values in the config dictionary are correct.

    Parameters:
        cfg (dict): The config dictionary to check.
        strict (bool): If True, raise an error if a key is not recognized.
    
    Returns:
        bool: True if all checks pass, raises an error otherwise.
    """
    # Check if cfg is a dictionary
    if not isinstance(cfg, dict):
        raise TypeError(f"Invalid config type. Expected dict, got {type(cfg)}.")
    # Check if the types of the values in the config dictionary are correct
    for key in cfg:
        if key not in CFG_TYPES:
            if strict:
                raise KeyError(f"Config parameter {key} not recognized.")
            else:
                pass
        else:
            check_types(cfg[key], CFG_TYPES[key], key)
    # If no errors are raised, return True
    return True

def read_cfg(
        path : Union[str, os.PathLike], 
        strict : bool=False
    ) -> dict:
    """
    Load and validate the config file.

    Missing keys are replaced with default values.

    Parameters:
        config (Union[str, os.PathLike]): The path to the config file.
        strict (bool): If True, raise an error if a key is not recognized.

    Returns:
        dict: The config dictionary.
    """
    # Check if config is a string or path-like object
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"Invalid config location. Expected str or os.PathLike, got {type(path)}.")
    # Check if the config file is a YAML file
    if not (path.endswith(".yaml") or path.endswith(".yml")):
        raise ValueError(f"Cannot read config. Expected YAML file, got {path}.")
    # Check if config file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file {path} not found.")
    # Load config file
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Type check config
    check_cfg_types(cfg, strict)
    # Replace missing keys with default values
    for key in DEFAULT_CFG:
        if key not in cfg:
            cfg[key] = DEFAULT_CFG[key]
    # Return config
    return cfg

def write_cfg(
        cfg : dict, 
        path : Union[str, os.PathLike], 
        overwrite : bool=False
    ) -> Union[str, os.PathLike]:
    """
    Save the config dictionary to a YAML file.

    Parameters:
        cfg (dict): The config dictionary to save.
        path (Union[str, os.PathLike]): The path to save the config file.
        overwrite (bool): If True, overwrite the file if it already exists.

    Returns:
        Union[str, os.PathLike]: The path to the saved config file.
    """
    # Check if path is a string or path-like object
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"Invalid config location. Expected str or os.PathLike, got {type(path)}.")
    # Check if path is a YAML file
    if not (path.endswith(".yaml") or path.endswith(".yml")):
        raise ValueError(f"Cannot save config. Expected YAML file, got {path}.")
    # Check if path exists
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"Config file {path} already exists.")
    # Type check config
    check_cfg_types(cfg)
    # Convert the MIN_MAX_OBJ_SIZE tuple to a list (prettier in the YAML file)
    if isinstance(cfg["MIN_MAX_OBJ_SIZE"], tuple):
        cfg["MIN_MAX_OBJ_SIZE"] = list(cfg["MIN_MAX_OBJ_SIZE"])
    # Create an ordered dict to control the order of the keys in the config YAML
    sorted_cfg = OrderedDict()
    # First add the known keys in the order they are defined in CFG_PARAMS
    for key in CFG_PARAMS:
        if key in cfg:
            sorted_cfg[key] = cfg[key]
    # Then add any additional keys that are not in CFG_PARAMS in the order they appear in the config dictionary
    for key in cfg:
        if key not in sorted_cfg:
            sorted_cfg[key] = cfg[key]
    # Save config file
    with open(path, "w") as f:
        # OBS: will fail if not using yaml.SafeDumper (default with yaml.safe_dump). If another dumper is to be used, the representer for OrderedDict must be added manually.
        yaml.safe_dump(sorted_cfg, f, sort_keys=False, default_flow_style=None)
    # Return the path to the saved config YAML file
    return path

if __name__ == "__main__":
    # Print a helpful message:
    logger.info(
        "####################################################################"
        "###################### Flat-Bug Configuration ######################"
        "####################################################################"
        "\nConfigurable parameters:"
    )
    logger.info(
        "\n".join([f"\t- {key} ({CFG_TYPES[key]}): {CFG_DESCRIPTION[key]}" for key in CFG_PARAMS])
    )
    logger.info(
        "\nParameters can either be specified with a YAML file or manually:"
        "\t* `fb_predict --config <YML_PATH>`"
        "\t* `flat_bug.predictor.Predictor.__init__(..., cfg=<YML_PATH>, ...)`"
        "\t* `flat_bug.predictor.Predictor.set_hyperparameters(<PARAM_i>=<VALUE_i>, <PARAM_j>=<VALUE_j>, ...)`"
        "\nAny parameters not specified will be set to default values."
    )