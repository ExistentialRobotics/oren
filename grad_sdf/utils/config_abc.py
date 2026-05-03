import argparse
import os
import pathlib
import sys
from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import ClassVar, Literal, Union, get_args, get_origin

import numpy as np
from ruamel import yaml

try:
    import torch
except ImportError:

    print("Warning: torch is not installed, set torch.Tensor to np.ndarray.")

    # dummy torch module
    class torch:
        Tensor = np.ndarray


__all__ = [
    "ConfigABC",
    "MultiConfigArgumentParser",
]


@dataclass
class ConfigABC:
    """Abstract base class for configuration dataclasses."""

    _registry: ClassVar[dict] = {}
    identifier: str = None

    @staticmethod
    def get_identifier(cls):
        return ".".join([cls.__module__, cls.__qualname__])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        identifier = ConfigABC.get_identifier(cls)
        if cls.__name__ == "ConfigABC" or identifier in ConfigABC._registry:
            return
        ConfigABC._registry[identifier] = cls

    def __post_init__(self):
        self.identifier = ".".join([self.__module__, self.__class__.__qualname__])

    @staticmethod
    def get(identifier: str):
        return ConfigABC._registry[identifier]

    def as_dict(self) -> dict:
        all_fields = fields(self)
        yaml_dict = dict()
        for field in all_fields:
            value = getattr(self, field.name)
            if value is None:
                field_type = field.type
            else:
                field_type = type(value)
            if is_dataclass(field_type):
                if not issubclass(field_type, ConfigABC):
                    raise TypeError(f"Filed {field.name} is dataclass but not a ConfigABC subclass.")
                if value is None:
                    value = field_type()
                assert (
                    value.identifier is not None
                ), f"Config object {field.name} has empty identifier. Do you forget to call super().__post_init__()?"
                yaml_dict[field.name] = value.as_dict()
            elif isinstance(value, tuple):  # convert tuple to list
                yaml_dict[field.name] = list(value)
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    yaml_dict[field.name] = value.item()
                else:
                    yaml_dict[field.name] = value.tolist()
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    yaml_dict[field.name] = value.item()
                else:
                    yaml_dict[field.name] = value.tolist()
            elif isinstance(value, (int, float, str, bool, dict, list)):
                yaml_dict[field.name] = value
            elif isinstance(value, pathlib.Path):
                yaml_dict[field.name] = str(value)
            elif value is None or value is MISSING:
                yaml_dict[field.name] = None
            else:
                raise TypeError(f"Unsupported type {type(value).__qualname__} for field {field.name}")
        return yaml_dict

    @staticmethod
    def _get_true_field_type(field):
        field_type = get_origin(field.type)

        if field_type is None:
            field_type = field.type  # rollback to original type

        type_args = get_args(field.type)
        choices = None

        if field_type is Union:
            args = get_args(field.type)
            if type(None) in args:  # optional
                if args[1] is type(None):
                    field_type = args[0]
                else:
                    field_type = args[1]
        elif field_type is Literal:
            field_type = str
            choices = type_args

        return field_type, type_args, choices

    @staticmethod
    def _load_field_value(field, value, field_type=None):
        if field_type in [None, type(None)]:
            field_type, type_args, choices = ConfigABC._get_true_field_type(field)
        else:
            type_args = get_args(field_type)
            choices = None

        if field_type is Union:
            for t in type_args:
                if t is type(None):
                    continue
                # noinspection PyBroadException
                try:
                    return ConfigABC._load_field_value(field, value, t)
                except Exception:
                    pass
            raise TypeError(f"Cannot convert value {value} to any type in Union {type_args}")
        else:
            if issubclass(field_type, ConfigABC):  # ConfigABC subclass
                assert value is not None, f"Field {field.name} is subclass of ConfigABC but has None value."
                return ConfigABC.get(value["identifier"]).from_dict(value)

            if is_dataclass(field_type):  # dataclass but not ConfigABC subclass
                raise TypeError(
                    f"Filed {field.name} is dataclass but not a ConfigABC subclass. "
                    "Please use ConfigABC subclass instead of dataclass only."
                )

            if value is None or value is MISSING:
                return None

            if issubclass(field_type, np.ndarray):
                return np.array(value)

            if issubclass(field_type, torch.Tensor):
                return torch.tensor(value)

            if issubclass(field_type, pathlib.Path):
                return pathlib.Path(value)

            if field_type is str and choices is not None:
                assert value in choices, f"Value {value} is not in choices {choices} for field {field.name}"
                return value

            try:
                return field_type(value)
            except Exception as e:
                # import pdb
                #
                # pdb.set_trace()
                raise TypeError(f"Error when converting field {field.name} to type {field.type}: {e}")

    @classmethod
    def from_dict(cls, yaml_dict: dict):
        all_fields = fields(cls)
        assert len(all_fields) > 0, "No fields found, did you forget to add @dataclass decorator to your class?"
        kwargs = {}
        for field in all_fields:
            if field.name not in yaml_dict:
                continue
            kwargs[field.name] = ConfigABC._load_field_value(field, yaml_dict[field.name])
            del yaml_dict[field.name]
        if len(yaml_dict) > 0:
            print(f"Warning: Unknown fields {list(yaml_dict.keys())} are ignored.")
        return cls(**kwargs)  # type: ignore

    def as_yaml(self, yaml_path: Union[str, pathlib.Path]):
        yaml_obj = yaml.YAML()
        yaml_obj.indent(mapping=4, sequence=6, offset=4)
        with open(yaml_path, "w") as file:
            yaml_obj.dump(self.as_dict(), file)

    @staticmethod
    def load_yaml_file(yaml_path: Union[str, pathlib.Path]) -> dict:

        def update_dict(src: dict, dst: dict):
            for key, value in src.items():
                if key in dst and isinstance(value, dict) and isinstance(dst[key], dict):
                    update_dict(value, dst[key])
                else:
                    dst[key] = value

        with open(yaml_path, "r") as file:
            yaml_dict = yaml.YAML(typ="safe").load(file)
        if yaml_dict is not None and "base_config" in yaml_dict:
            base_config_path = yaml_dict["base_config"]
            del yaml_dict["base_config"]
            if not os.path.isabs(base_config_path):
                base_config_path = os.path.join(os.path.dirname(yaml_path), base_config_path)
            base_yaml_dict = ConfigABC.load_yaml_file(base_config_path)
            update_dict(yaml_dict, base_yaml_dict)
            yaml_dict = base_yaml_dict
        return yaml_dict

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, pathlib.Path]):
        yaml_dict = ConfigABC.load_yaml_file(yaml_path)
        if yaml_dict is None:
            print(f"Warning: Empty config file {yaml_path}, using default values.")
            return cls()
        return cls.from_dict(yaml_dict)

    class ArgumentParser(argparse.ArgumentParser):
        def __init__(self, cls, **kwargs):
            super().__init__(**kwargs)
            self.add_argument(
                "--config",
                type=str,
                metavar="CONFIG_PATH",
                default=None,
                required=False,
                help="Path to config file.",
            )
            self.add_argument(
                "--create-config",
                action="store_true",
                help="Create a config file with default values and save it to CONFIG_PATH.",
            )
            self._cls = cls
            self._added_arguments_for_cls = False

        def format_help(self) -> str:
            self._add_arguments_for_cls(self._cls, None)
            return super().format_help()

        def _add_arguments_for_cls(self, cls, cls_obj=None, prefix: str = ""):
            if self._added_arguments_for_cls:
                return
            self._added_arguments_for_cls = True
            self._add_arguments_for_cls_recursive(cls, cls_obj, prefix)

        @staticmethod
        def _get_default_value(field):
            no_default = field.default is None or field.default is MISSING
            no_default_factory = field.default_factory is MISSING

            if no_default and no_default_factory:
                required = True
                if get_origin(field.type) is Union and type(None) in get_args(field.type):  # optional
                    required = False
                return None, required

            if no_default:
                return field.default_factory(), False

            return field.default, False

        def _add_arguments_for_cls_recursive(self, cls, cls_obj=None, prefix: str = ""):
            if hasattr(cls, "help_dict"):
                help_dict = cls.help_dict
            else:
                help_dict = {}

            all_fields = fields(cls)
            for field in all_fields:
                default_value, required = self._get_default_value(field)
                if cls_obj is not None and cls_obj is not MISSING:
                    # get default value from config object if it is loaded from config file
                    default_value = getattr(cls_obj, field.name)

                if len(prefix) == 0:
                    argname = field.name
                else:
                    argname = f"{prefix}.{field.name}"

                help_str = help_dict.get(argname, "")
                dst = argname
                argname = argname.replace("_", "-")
                flag = f"--{argname}"

                if default_value is not None:
                    if len(help_str) > 0:
                        help_str = help_str + " "
                    help_str = help_str + f"Default: {default_value}"
                    required = False  # default value is provided, so it is not required

                field_type, type_args, choices = ConfigABC._get_true_field_type(field)
                if field_type is Union:
                    if default_value is None:
                        field_type = type_args[0]
                    else:
                        field_type = type(default_value)
                if issubclass(field_type, ConfigABC):
                    if not is_dataclass(field_type):
                        raise TypeError(f"Filed {field.name} is a subclass of ConfigABC but not a dataclass.")
                    self._add_arguments_for_cls_recursive(field_type, default_value, prefix=argname)
                    continue

                if is_dataclass(field_type):
                    raise TypeError(f"Filed {field.name} is dataclass but not a ConfigABC subclass.")

                if field_type in (list, tuple):
                    if len(type_args) == 1:
                        field_type = type_args[0]
                    else:
                        field_type = None
                    self.add_argument(
                        flag,
                        type=field_type,
                        nargs="+",
                        default=default_value,
                        required=required,
                        help=help_str,
                    )
                elif field_type in [np.ndarray, torch.Tensor]:
                    self.add_argument(
                        flag,
                        nargs="+",
                        type=float,  # only support float for now
                        default=default_value,
                        required=required,
                        help=help_str,
                    )
                elif field_type in (int, float):
                    self.add_argument(flag, type=field_type, default=default_value, required=required, help=help_str)
                elif field_type == str:
                    if field.name == "identifier":
                        self.add_argument(
                            flag,
                            type=str,
                            default=ConfigABC.get_identifier(cls),
                            help=f"Config class path. Default: {ConfigABC.get_identifier(cls)}. ",
                        )
                    else:
                        self.add_argument(
                            flag,
                            type=str,
                            default=default_value,
                            choices=choices,
                            required=required,
                            help=help_str,
                        )
                elif field_type == bool:
                    if default_value:
                        self.add_argument(f"--no-{argname}", action="store_false", dest=dst, help="Set to False")
                    else:
                        self.add_argument(flag, action="store_true", dest=dst, help="Set to True")
                elif field_type == dict:
                    self.add_argument(flag, type=eval, default=default_value, required=required, help=help_str)
                elif field_type == pathlib.Path:
                    self.add_argument(
                        flag,
                        type=pathlib.Path,
                        default=default_value,
                        required=required,
                        help=help_str,
                    )
                else:
                    raise TypeError(f"Unsupported type {field_type.__qualname__} for field {field.name}")

        def parse_known_args(self, args: list = None, namespace=None):
            if args is None:
                args = list(sys.argv[1:])
            if not isinstance(args, list):
                args = list(args)

            try:
                config_arg_index = args.index("--config")
                if len(args) <= config_arg_index + 1:  # no value provided
                    raise RuntimeError("Please specify config file path with --config.")
                config = args[config_arg_index + 1]
                if os.path.exists(config):
                    config_obj = self._cls.from_yaml(config)
                else:
                    assert "--create-config" in args, f"Config file {config} does not exist."
                    config_obj = None
            except ValueError:
                config = None
                config_obj = None

            if "--help" in args or "-h" in args:
                self._add_arguments_for_cls(self._cls, config_obj)
                super().parse_known_args(args, namespace)
                exit(0)

            if "--create-config" in args:
                if config_obj is not None:
                    overwrite = input(f"Config file {config} already exists. Overwrite? [y/N] ")
                    if overwrite.lower() != "y":
                        exit(0)
                try:
                    config_arg_index = args.index("--config")
                except ValueError:
                    print("Please specify config file path with --config.")
                    exit(1)
                config = args[config_arg_index + 1]
                self._cls().as_yaml(config)
                exit(0)

            self._add_arguments_for_cls(self._cls, config_obj)
            args, unknown = super().parse_known_args(args)

            if args.config is not None:  # config_obj is not None
                args = vars(args)
                del args["config"]
                del args["create_config"]

                def args_update_config(arg_dict: dict, _config_obj):
                    config_fields = fields(_config_obj)
                    config_fields = {field.name: field for field in config_fields}
                    for key, value in arg_dict.items():
                        key: str
                        key = key.replace("-", "_")

                        if key.endswith(".identifier"):  # ignore commands like "--xxx.identifier"
                            continue
                        ind = key.find(".")
                        if ind == -1:  # leaf attribute
                            field_type = type(getattr(_config_obj, key))
                            value = ConfigABC._load_field_value(config_fields[key], value, field_type)
                            if value is not None:
                                setattr(_config_obj, key, value)
                        else:  # nested attribute
                            prefix = key[:ind]
                            suffix = key[ind + 1 :]
                            assert hasattr(_config_obj, prefix), f"Unknown attribute {prefix}"
                            args_update_config({suffix: value}, getattr(_config_obj, prefix))

                args_update_config(args, config_obj)
                return config_obj, unknown
            else:

                def args_to_yaml_dict(arg_dict: dict):
                    result = {}
                    for key, value in arg_dict.items():
                        key: str
                        ind = key.find(".")
                        if ind == -1:
                            assert key not in result, f"Duplicate key"
                            result[key] = value
                        else:
                            prefix = key[:ind]
                            suffix = key[ind + 1 :]
                            if prefix not in result:
                                result[prefix] = dict()
                            assert suffix not in result[prefix]
                            result[prefix].update(args_to_yaml_dict({suffix: value}))
                    return result

                args = vars(args)
                del args["config"]
                del args["create_config"]
                return self._cls.from_dict(args_to_yaml_dict(args)), unknown

    @classmethod
    def get_argparser(cls):
        return ConfigABC.ArgumentParser(cls)


class MultiConfigArgumentParser(argparse.ArgumentParser):
    class CompleteHelpAction(getattr(argparse, "_HelpAction")):
        def __call__(self, parser, namespace, values, option_string=None):
            parser.print_help()
            subparsers_actions = [
                action
                for action in getattr(parser, "_actions")
                if isinstance(action, getattr(argparse, "_SubParsersAction"))
            ]
            print()
            for subparsers_action in subparsers_actions:
                for choice, subparser in subparsers_action.choices.items():
                    print(f"Command: {choice}")
                    print(subparser.format_help())
                    print()
            parser.exit()

    def __init__(self, config_parser_dict: dict, **kwargs):
        kwargs["add_help"] = False
        super().__init__(**kwargs)
        self.add_argument("--help", "-h", action=self.CompleteHelpAction, help="show this help message and exit")
        self._my_subparsers = self.add_subparsers(
            title="commands",
            description="Available commands",
            required=True,
            dest="command",
            parser_class=ConfigABC.ArgumentParser,
        )
        self._my_subparsers_config_dict = config_parser_dict
        self._my_subparsers_dict = {}
        for key, config_kwargs in config_parser_dict.items():
            assert "cls" in config_kwargs, "cls is required in config_kwargs"
            self._my_subparsers_dict[key] = self._my_subparsers.add_parser(key, **config_kwargs)
        self.command = None

    def parse_known_args(self, args=None, namespace=None):
        args, unknown = super().parse_known_args(args, namespace)
        self.command = args.command
        if self.command in self._my_subparsers_dict:
            config_cls = self._my_subparsers_config_dict[self.command]["cls"]
            config_kwargs = vars(args)
            del config_kwargs["command"]
            config = config_cls(**config_kwargs)
            return config, unknown
        return args, unknown
