from pyhocon import ConfigFactory
from arguments import args

config = ConfigFactory.parse_file(args.config)
