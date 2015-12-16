
function componentToHex(c) {
    var hex = c.toString(16);
    return hex.length == 1 ? "0" + hex : hex;
}

function rgbToHex(r, g, b) {
    return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
}

function getColorForFocus() {
    return '#F58800'
}

function getColorFromVec(vec) {
    lb = 100
    hb = 200

    rgbToHex()
}