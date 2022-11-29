// photoshop script
// function: read mouse position and open a file

var activeDoc = app.activeDocument

// alert(activeDoc)

// app.showColorPicker()

var l = app.foregroundColor.lab.l
var a = app.foregroundColor.lab.a
var b = app.foregroundColor.lab.b

var minn = -128
var maxn = 127
a = 1.0 - (a - minn) / (maxn - minn)
b = 1.0 - (b - minn) / (maxn - minn)
// alert(a)
// alert(b)

var folder = new Folder("results")
var files = folder.getFiles()
// alert(files.length)
num_files = files.length

var min_dist2 = 1000000
var min_dist2_index = 0

for (var j = 0; j < num_files; j++) {
    var file = files[j].toString()
    var filename = file.split("/")
    var length = filename.length
    filename = filename[length-1]
    length = filename.length
    filename = filename.substring(0, length-4)
    var x = filename.split("_")[0]
    var y = filename.split("_")[1]
//     alert(x + ' ' + y + ', ' + a + ' ' + b)
    var dist2 = (x - a) * (x - a) + (y - b) * (y - b)
    if (min_dist2 > dist2) {
        min_dist2 = dist2
        min_dist2_index = j
    }
}

var file = files[min_dist2_index].toString()

fileObj = new File(file)

var doc = open(fileObj)

