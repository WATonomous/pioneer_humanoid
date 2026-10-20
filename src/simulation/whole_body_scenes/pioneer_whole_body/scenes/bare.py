"""The studio and an empty workbench."""
from pioneer_whole_body import scene
from pioneer_whole_body.studio import add_bench, add_studio


@scene("bare")
def build(spec):
    add_studio(spec)
    add_bench(spec)
