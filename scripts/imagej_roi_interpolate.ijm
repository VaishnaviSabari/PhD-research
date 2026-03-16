// ImageJ Macro: ROI detection and interpolation
for (i = 1; i <= nSlices; i++) {
    setSlice(i);
    run("Select All");
    getStatistics(area, mean, min, max, std, histogram);
    run("Select None");
    if (max == 255) {
        run("Create Selection");
        roiManager("add");
    }
}
if (roiManager("count") >= 2) {
    roiManager("Interpolate ROIs");
} else {
    showMessage("Need at least 2 ROIs for interpolation.");
}
