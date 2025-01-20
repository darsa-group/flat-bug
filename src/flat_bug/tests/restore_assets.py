import os

if __name__ == "__main__":
    ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))

    prediction_assets = [os.path.join(ASSET_DIR, f) for f in os.listdir(ASSET_DIR) if f.startswith("pyramid_" or f.startswith("single_scale_"))]

    for asset in prediction_assets:
        with open(asset, "w") as f:
            f.write("ERDA Pointer\n")
    