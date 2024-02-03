from main import main
from config.config import parse_args
from multitask.multitask import multitask

if __name__ == '__main__':
    args = parse_args()
    
    if args.multitask:
        multitask(args)
    else:
        main(args)
