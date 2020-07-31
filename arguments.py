import argparse
import sys
import time

__ALL__ = ['args']

# class Args(TypedArgs):

#     def __init__(self):
#         parser = argparse.ArgumentParser()
#         self.experiment_path: Optional[str] = parser.add_argument(
#             '-e',
#             '--experiment-path',
#             type=str,
#             nargs='?',
#             default='!default',
#             required=False,
#             help='path to save your experiment'
#         )

#         self.config: Optional[str] = parser.add_argument(
#             '-c',
#             '--config',
#             type=str,
#             nargs='?',
#             required=False,
#             help='path to config files'
#         )

#         self.debug: bool = parser.add_argument(
#             '-d',
#             '--debug',
#             action='store_true',
#             help='debug mode'
#         )

#         self.resume: str = parser.add_argument(
#             '-r',
#             '--resume',
#             type=str,
#             help='resume an experiment'
#         )

#         # This arg is for distributed
#         self.local_rank: int = parser.add_argument(
#             '--local_rank',
#             default=0,
#             type=int
#         )

#         self.parse_known_args_from(parser)
#         self.create_experiment_path()

#     def create_experiment_path(self):
#         timestamp = get_timestamp()
#         if self.experiment_path is None:
#             if self.debug:
#                 experiment_path = os.path.join(
#                     'output', '{}_debug'.format(timestamp))
#             else:
#                 experiment_path = os.path.join('output', timestamp)

#         else:
#             # No experiment path
#             if self.experiment_path == '!default':
#                 experiment_path = None
#             else:
#                 experiment_path = self.experiment_path

#         if experiment_path is not None and self.local_rank == 0:
#             try:
#                 os.makedirs(experiment_path)
#             except Exception as e:
#                 if not F.ask_remove_older_experiment_path(experiment_path):
#                     raise e

#         self.experiment_path = experiment_path


def get_timestamp(fmt: str = '%Y%m%d_%H%M%S') -> str:
    timestamp = time.strftime(fmt, time.localtime())
    return timestamp


def create_experiment_path(self):
    timestamp = get_timestamp()
    if self.experiment_path is None:
        if self.debug:
            experiment_path = os.path.join(
                'output', '{}_debug'.format(timestamp))
        else:
            experiment_path = os.path.join('output', timestamp)

    else:
        # No experiment path
        if self.experiment_path == '!default':
            experiment_path = None
        else:
            experiment_path = self.experiment_path

    if experiment_path is not None and self.local_rank == 0:
        try:
            os.makedirs(experiment_path)
        except Exception as e:
            if not F.ask_remove_older_experiment_path(experiment_path):
                raise e

    self.experiment_path = experiment_path


parser = argparse.ArgumentParser()
parser.add_argument(
    '-e',
    '--experiment-path',
    type=str,
    nargs='?',
    default='!default',
    required=False,
    help='path to save your experiment'
)
parser.add_argument(
    '-c',
    '--config',
    type=str,
    nargs='?',
    required=False,
    help='path to config files'
)
parser.add_argument(
    '-d',
    '--debug',
    action='store_true',
    help='debug mode'
)
parser.add_argument(
    '-r',
    '--resume',
    type=str,
    help='resume an experiment'
)
parser.add_argument(
    '--local_rank',
    default=0,
    type=int
)


parser.add_argument  # TODO

args = parser.parse_args()
