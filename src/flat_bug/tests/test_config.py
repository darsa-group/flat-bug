import unittest

import os, tempfile, copy

from flat_bug.config import get_type_def, check_types, check_cfg_types, read_cfg, write_cfg, DEFAULT_CFG, CFG_TYPES

TEST_OBJECTS = {
    "float": 1.23,
    "int": 1,
    "string": "string",
    "dict": {"key": "value"},
    "bool": True,
    "list of ints": [1, 2, 3],
    "tuple of ints": (1, 2, 3),
    "mixed list": [1, "string", True],
    "list of mixed list and mixed tuple": [[1, "a"], ("b", 2)]
}

TEST_OBJECTS_TYPES_LIST_TUPLE_NOT_INTERCHANGEABLE = {
    "float": float,
    "int": int,
    "string": str,
    "dict": dict,
    "bool": bool,
    "list of ints": [list, [int, int, int]],
    "tuple of ints": [tuple, [int, int, int]],
    "mixed list": [list, [int, str, bool]],
    "list of mixed list and mixed tuple": [list, [[list, [int, str]], [tuple, [str, int]]]]
}

TEST_OBJECTS_TYPES_LIST_TUPLE_INTERCHANGEABLE = {
    "float": float,
    "int": int,
    "string": str,
    "dict": dict,
    "bool": bool,
    "list of ints": [(tuple, list), [int, int, int]],
    "tuple of ints": [(tuple, list), [int, int, int]],
    "mixed list": [(tuple, list), [int, str, bool]],
    "list of mixed list and mixed tuple": [(tuple, list), [[(tuple, list), [int, str]], [(tuple, list), [str, int]]]]
}

def check_equals_recursive(obj1, obj2):
    if isinstance(obj1, (tuple, list)):
        if len(obj1) != len(obj2):
            return False
        for i, (o1, o2) in enumerate(zip(obj1, obj2)):
            if not check_equals_recursive(o1, o2):
                return False
        return True
    return obj1 == obj2

class TestConfig(unittest.TestCase):
    def test_check_types(self):
        for i, (key, obj) in enumerate(TEST_OBJECTS.items()):
            expected_type = get_type_def(obj)
            check_types(obj, expected_type, f"Object '{key}' ({i})")
            check_types(obj, TEST_OBJECTS_TYPES_LIST_TUPLE_NOT_INTERCHANGEABLE[key], f"Object '{key}' ({i})")
            check_types(obj, TEST_OBJECTS_TYPES_LIST_TUPLE_INTERCHANGEABLE[key], f"Object '{key}' ({i})")
    
    def test_check_cfg_types(self):
        try:
            check_cfg_types(DEFAULT_CFG, strict=True)
        except Exception as e:
            type(e)(f"Error raised when checking the types of the default config:\n" + str(e))
        altered_cfg = copy.deepcopy(DEFAULT_CFG)
        altered_cfg["UNKNOWN_KEY"] = "value"
        with self.assertRaises(KeyError, msg="Failed to raise an error when checking config with unknown key and strict=True"):
            check_cfg_types(altered_cfg, strict=True)
        try:
            check_cfg_types(altered_cfg, strict=False)
        except Exception as e:
            raise type(e)(f"Error raised when checking config with unknown key and strict=False:\n" + str(e))

    def test_get_type_def(self):
        error_msg = \
        """
        Failed to generate the correct type definitions for the test objects with tuple_list_interchangeable={}.
        TEST_OBJECTS_TYPES should be a dictionary with:
            - keys: same as TEST_OBJECTS, 
            - values: the type definitions of the corresponding values in TEST_OBJECTS, that pass the check_types function.
        Either TEST_OBJECTS_TYPES is incorrect, get_type_def is not generating the correct type definitions or test_check_types did not pass.
        """
        try:
            for key, expected_type in TEST_OBJECTS_TYPES_LIST_TUPLE_NOT_INTERCHANGEABLE.items():
                obj = TEST_OBJECTS[key]
                type_def = get_type_def(obj, tuple_list_interchangeable=False)
                # Check that the generated type definition is the same as the expected type definition
                self.assertTrue(check_equals_recursive(type_def, expected_type), f"\nFailed on object:\n'{key}' => {obj}\nwith generated type definition:\n{type_def}\nand expected type definition:\n{expected_type}")
                # Check that the object passes the type definition - no need to assertTrue, since check_types will raise an error if it fails
                check_types(obj, type_def, f"Object {key}")
        except Exception as e:
            raise type(e)(str(e) + error_msg.format(False))
        try:
            for key, expected_type in TEST_OBJECTS_TYPES_LIST_TUPLE_INTERCHANGEABLE.items():
                obj = TEST_OBJECTS[key]
                type_def = get_type_def(obj, tuple_list_interchangeable=True)
                # Check that the generated type definition is the same as the expected type definition
                self.assertTrue(check_equals_recursive(type_def, expected_type), f"\nFailed on object:\n'{key}' => {obj}\nwith generated type definition:\n{type_def}\nand expected type definition:\n{expected_type}")
                # Check that the object passes the type definition - no need to assertTrue, since check_types will raise an error if it fails
                check_types(obj, type_def, f"Object {key}")
        except Exception as e:
            raise type(e)(str(e) + error_msg.format(True))

    def test_default_cfg(self):
        try:
            check_cfg_types(DEFAULT_CFG, strict=True)
        except Exception as e:
            raise type(e)(f"Error raised when checking the types and keys of the default config:\n" + str(e))

    def test_write_read_cfg(self):
        orig_cfg = copy.deepcopy(DEFAULT_CFG)
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_cfg_file = os.path.join(tmpdir, "test.cfg")
            with self.assertRaises(ValueError, msg="Failed to raise an error when writing a config file with an invalid extension"):
                write_cfg(orig_cfg, invalid_cfg_file)
            cfg_file = os.path.join(tmpdir, "test.yaml")
            write_cfg(orig_cfg, cfg_file)
            new_cfg = read_cfg(cfg_file)
            alter_cfg = copy.deepcopy(orig_cfg)
            alter_cfg["UNKNOWN_KEY"] = "value"
            alter_file = os.path.join(tmpdir, "alter_test.yaml")
            write_cfg(alter_cfg, alter_file)
            with self.assertRaises(KeyError, msg="Failed to raise an error when reading a config file with unknown key and strict=True"):
                read_cfg(alter_file, strict=True)
            try:
                read_cfg(alter_file)
            except Exception as e:
                type(e)(f"Error raised when reading a config file with unknown key and strict=False:\n" + str(e))
        self.assertTrue(check_types(new_cfg, get_type_def(orig_cfg), "Reconstructed Config", strict=False), f"Failed to reconstruct the original config with comparable types after writing and reading.")
        self.assertTrue(check_equals_recursive(orig_cfg, new_cfg), "Failed to reconstruct the values of the original config after writing and reading. Although the types are comparable, the values are not equal.")

if __name__ == '__main__':
    unittest.main()