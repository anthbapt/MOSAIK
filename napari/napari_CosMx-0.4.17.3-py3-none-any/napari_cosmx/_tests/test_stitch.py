from napari_cosmx.utils.stitch_images import main as stitch_images
from napari_cosmx.gemini import Gemini

def test_stitch_images(make_napari_viewer, tmp_path):
    stitch_images(args_list=['-i', 'data/Liver-S2-source/CellStatsDir',
         '-f', 'data/Liver-S2-source/RunSummary',
         '-o', tmp_path.as_posix()])
    viewer = make_napari_viewer()
    gem = Gemini(tmp_path.as_posix(), viewer=viewer)
    assert isinstance(gem, Gemini)
