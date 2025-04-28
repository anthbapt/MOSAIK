from napari_cosmx.gemini import Gemini

def test_launch_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    gem = Gemini('data/Liver-S2', viewer=viewer)
    assert isinstance(gem, Gemini)

def test_add_channel(make_napari_viewer):
    viewer = make_napari_viewer()
    gem = Gemini('data/Liver-S2', viewer=viewer)
    gem.add_channel('DNA', colormap='blue')
    assert 'DNA' in viewer.layers, "Morphology layer not added to viewer"
    cm = viewer.layers['DNA'].colormap.name
    assert cm == 'blue', f"Colormap is {cm}, not blue as requested"