from opt import parse_opts


def main():
    from src.trainer import run
    run(parse_opts())


if __name__ == '__main__':
    main()
