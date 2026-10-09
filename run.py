import sys

from factory.utils import ensure_utf8_console


def main():
    ensure_utf8_console()
    if "--cli" in sys.argv:
        from cli import main as cli_main
        sys.argv = ["cli.py"] + [a for a in sys.argv[1:] if a != "--cli"]
        return cli_main()
    from factory.gui import launch_gui
    launch_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
