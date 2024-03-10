from typing import Optional, List
from dataclasses import dataclass
import inspect

from argparse import ArgumentParser, Namespace


class RecSysArgumentParser:
    parser: ArgumentParser = ArgumentParser()

    def parse(self) -> Namespace:
        self.parser.add_argument('--skip-training', '-s',
                                 action='store_true',
                                 help='Boolean flag to determine if training should be skipped.',
                                 required=False)
        return self.parser.parse_args()


@dataclass
class BaseArguments:
    skip_training: Optional[List[str] | str]

    @classmethod
    def from_dict(cls, env):
        """Allows the definition of the class even if extra arguments are provided in the input"""
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })
