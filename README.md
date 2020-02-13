ConvertOnnxModel  
2018-07-19..2020-02-13  
Dwayne Robinson (FDwR)  

Converts a binary ONNX model file to text (which can be edited in any simple text editor) and vice versa.
Can also convert input/output tensor protobuf to CSV/PNG and vice versa.

# Usage
    ConvertOnnxModel.exe [options] inputFilename [outputFilename]

# Example usage
    ConvertOnnxModel.exe Foo.onnx Foo.txt // ONNX model binary file to text representation
    ConvertOnnxModel.exe Foo.txt Foo.onnx // ONNX model text representation to binary file
    ConvertOnnxModel.exe -tensor Foo.pb Foo.csv // Tensor protobuf file to comma separated values
    ConvertOnnxModel.exe -tensor -dimensions 224,224 -datatype uint8 -row 2 -column 1,225 Foo.csv Foo.dat
    ConvertOnnxModel.exe -tensor file.pb file.txt // Tensor brotobuf to text
    ConvertOnnxModel.exe -tensor file.pb file.png // Tensor protobuf to PNG image
    ConvertOnnxModel.exe -tensor file.png file.pb // PNG image to tensor protobuf
    ConvertOnnxModel.exe -tensor -dimensions 3,480,640 file.csv file.png // CSV to PNG with dimensions
    ConvertOnnxModel.exe -tensor -datatype float64 con.csv foo.dat // CSV file to raw data array
    ConvertOnnxModel.exe -tensor -datatype uint16 foo.dat con.csv // raw data array to CSV file

# Parameters
* input/output files: graph (onnx/pb/text) or tensor (pb/text/csv/dat).
* -tensor: convert tensor instead of graph.
* -graph: convert graph (default).
* -dimensions: explicit tensor dimensions for .csv or .dat file. Defaults to 1D element count from source data.
* -datatype: tensor element type (float16,float32,float64,int8,uint8,int16,uint16,int32,uint32,int64,uint64,bool8,complex64,complex128). This isn't usually needed unless reading from raw data.
* -row: single row or range for .csv.
* -column: single column or range for .csv.

# File Types
* .onnx - Open Neural Network Exchange model/graph binary file.
* .txt - Open Neural Network Exchange model/graph text file.
* .pb - Protobuf binary file, either tensor or graph (depending on -tensor or -graph). The dimensions are data type are contained in the file.
* .csv - Comma separated value. Contain raw values, no dimensions. The dimensions should be specified if input.
csv/dat).
* .png - Portable Network Graphics image file.
* .dat/.bin - Raw binary data.

# Building
Load Visual Studio solution, and build.

# Build Google Protobuf .lib yourself:
The protobuf-3.5.1 directory in this project contains the bare minimum .lib and .h files to build
the executable. If you need a different configuration:

Download/clone: https://github.com/protocolbuffers/protobuf/releases/tag/v3.5.1

    cd protobuf-3.5.1\cmake\build\solution\
    cmake -G "Visual Studio 15 2017" -DCMAKE_INSTALL_PREFIX=../../../../install ../.. -Dprotobuf_BUILD_TESTS=OFF
    start protobuf-3.5.1\cmake\build\solution\protobuf.sln

Build libprotobuf.vcxproj project. Find the .lib files under protobuf-3.5.1\cmake\build\solution\*.

# Updating ONNX Proto file.

Copy the newest version of `onnx.proto` from https://github.com/onnx/onnx/blob/master/onnx/onnx.proto.

Run the ProtoBuf compiler.

    protobuf3.5.1\bin\protoc.exe onnx.proto --cpp_out=. --error_format=msvs
