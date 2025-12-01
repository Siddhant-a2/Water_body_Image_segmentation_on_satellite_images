// 1. Define Area of Interest (ROI)
// If you haven't drawn a polygon, this sets a default point (San Francisco).
// PLEASE DRAW A POLYGON AND RENAME IT 'roi' FOR YOUR SPECIFIC AREA.
var roi = typeof roi !== 'undefined' ? roi : 
  ee.Geometry.Point([-122.4194, 37.7749]).buffer(5000).bounds();

// 2. Define Time Range
var startDate = '2023-01-01';
var endDate = '2023-12-31';

// 3. Cloud Masking Function
// This removes cloudy pixels using the QA60 band so your download is clear.
function maskS2clouds(image) {
  var qa = image.select('QA60');
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000); // Scale to 0-1 reflectance
}

// 4. Load the Sentinel-2 Harmonized Surface Reflectance Collection
var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(startDate, endDate)
  .filterBounds(roi)
  // Pre-filter to get less cloudy images to begin with (<20% cloud cover)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds);

// 5. Select the specific bands requested
// 10m bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
// 20m bands: B11 (SWIR1), B12 (SWIR2)
var selectedBands = collection.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']);

// 6. Create a Composite
// We create a median composite to get a single, cloud-free image for the year.
var composite = selectedBands.median().clip(roi);

// 7. Visualization (Optional - checks if the image looks correct)
var visParams = {
  bands: ['B4', 'B3', 'B2'], // True Color
  min: 0.0,
  max: 0.3,
};
Map.centerObject(roi, 12);
Map.addLayer(composite, visParams, 'Sentinel-2 Composite');

// 8. Export to Google Drive
// Note: B11 and B12 are natively 20m, but we export at 10m to match the
// VNIR bands. GEE will nearest-neighbor resample them automatically.
Export.image.toDrive({
  image: composite,
  description: 'Sentinel2_Selected_Bands_Export',
  folder: 'GEE_Exports', // The folder in your Google Drive
  scale: 10,             // Resolution in meters (matches B2, B3, B4, B8)
  region: roi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});