"""Use mjlab's supported train/play CLI with the local task registered."""


def train():
    from .task import register

    register()
    from mjlab.scripts.train import main

    main()


def play():
    from .task import register

    register()
    from mjlab.scripts.play import main

    main()
