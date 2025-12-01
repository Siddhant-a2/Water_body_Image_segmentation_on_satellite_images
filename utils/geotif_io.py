def readTiff(path):
    raise NotImplementedError(
        "readTiff is not used in this Streamlit app. "
        "Images are read using rasterio in app.py."
    )


def writeTiff(im_data, geotrans, proj, path):
    raise NotImplementedError(
        "writeTiff is not used in this Streamlit app. "
        "If you need GeoTIFF export, implement it directly with rasterio."
    )
