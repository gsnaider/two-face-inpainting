import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
  '--config_file',
  type=argparse.FileType(mode='r'))

parser.add_argument('--max_steps', type=int)
parser.add_argument('--gen_learning_rate', type=float)
parser.add_argument('--lambda_rec', type=float, default=1.0)
parser.add_argument('--foo', type=float, default=5.5)


args, _ = parser.parse_known_args()

if args.config_file:
  data = json.load(args.config_file)
  delattr(args, 'config_file')
  arg_dict = args.__dict__
  for key, value in data.items():
    if isinstance(value, list):
      arg_dict[key].extend(value)
    else:
      arg_dict[key] = value


print(args.max_steps)
print(args.gen_learning_rate)
print(args.lambda_rec)
print(args.foo)

print(type(args.max_steps))
print(type(args.gen_learning_rate))
print(type(args.lambda_rec))
