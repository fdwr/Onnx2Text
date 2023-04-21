Onnx2Text  
2018-07-19..2023-03-02  
Dwayne Robinson (FDwR)  

- Converts a binary [ONNX](https://github.com/onnx/onnx) model file to text (which can be edited in any simple text editor) and vice versa.
- Converts an ONNX model to GraphViz dot file.
- Converts an ONNX tensor protobuf to text/CSV/PNG/NPY and vice versa.
- Generates output tensor values (ones, zeros, iota series, random).
- Exports model tensors to directory of tensor files.

# Motivation

I needed to make small edits to existing models for debugging, and I didn't want to have to write actual code every time, but rather just use a command line tool. Additionally I needed a simple tool that predictably _just worked_ by itself regardless of what machine I'm using and is easily shareable (not a tool that depends on your current environment and needed pip install this or pip install that, but oops, not that incompatible version of the library, and don't forget to install Cmake and protobuf and add them to your path before the egg/wheel thingie is built... 😑).

# Usage
    Onnx2Text.exe [options] inputFilename outputFilename

# Example usage

- Convert model to/from ONNX binary protobuf and prototxt format (open with Notepad):
    - `Onnx2Text input.onnx output.prototxt`  <i>plain `.txt` works too</i>
    - `Onnx2Text input.prototxt output.onnx`
    - `Onnx2Text input.prototxt con:` // print to console

- Just show information like how many times each operator is used:
    - `Onnx2Text -information model.onnx`

- Write GraphViz dot file (download GraphViz separately):
    - `Onnx2Text input.onnx output.dot`
    - `dot.exe output.dot -Tpng -O`  (or -Tsvg)

- Zero weights in ONNX binary protobuf:
    - `Onnx2Text -zeromodelvalues input.onnx output.onnx`

- Export model from ONNX protobuf to NumPy tensors/data files:
    - `Onnx2Text resnet50.onnx x:\resnet_*.npy`
    - `Onnx2Text squeezenet.onnx z:\folder\*_weight.dat`

- Convert tensor between ONNX protobuf, CSV, raw data, numpy, PNG:
    - `Onnx2Text input.onnxtensor output.csv`
    - `Onnx2Text input.pb output.png`
    - `Onnx2Text -datatype uint8 -dimensions 224,224 Foo.csv Foo.dat`
    - `Onnx2Text input.npy output.onnxtensor`

- Generate tensor from randomness:
    - `Onnx2Text -dimensions 3,4 -datatype float16 generate(random,1,24) output.onnxtensor`

# Parameters
* input/output files - graph (onnx/pb/text) or tensor (onnxtensor/npy/pb/text/csv/dat).
* `-dimensions` - explicit tensor dimensions for .csv or .dat file. Defaults to 1D element count from source data. Pass "()" to indicate 0D scalar.
* `-datatype` - tensor element type (float16,float32,float64,int8,uint8,int16,uint16,int32,uint32,int64,uint64,bool8,float16m7e8s1/bfloat16). This isn't usually needed unless reading from raw data.
* `-zeromodelvalues` - zero any tensor values in model (clears model initializer weights - useful for sharing confidential models without revealing trained results) except tiny 1D tensors needed for shapes.
* `-row` - single row or range for .csv.
* `-column` - single column or range for .csv.
* `-scale` - scale tensor values during conversion.
* `-inversescale` - scale tensor values during conversion by reciprocal (e.g. 255 means 1/255).
* `-normalizevalues` - should normalize values in tensor 0 to 1.
* `-information` - display more verbose file information (output file is not needed).
* `-tensor` - specifies the input file is a tensor (only needed if ambiguous file type like .pb).
* `-graph` - specifies the input file is a model (only needed if ambiguous file type like .pb).

# File Types
* Model file types:
    * .onnx - Open Neural Exchange model protobuf
    * .pb - Google Protobuf (with -graph)
    * .txt/.prototxt - Protobuf text
* Tensor file types:
    * .onnxtensor - Open Neural Exchange tensor
    * .pb - Google Protobuf (with -tensor)
    * .csv - Comma Separate Values (no dimensions, just data)
    * .png - Image (Portable Network Graphics)
    * .jpg - Image (Joint Photographic Experts Group)
    * .npy - NumPyArray single tensor
    * .dat/.bin - Raw binary data (no header, just contiguous array elements)
    * generate() - Generator tensor input pseudo filename:
        * generate(ones) - all ones. [1,1,1,1...]
        * generate(zeros) - all zeros [0,0,0,0...]
        * generate(values,3) - specific value [3,3,3,3...]
        * generate(iota,1,2) - increasing sequence [1,3,5...]
        * generate(random,1,100) - random values between min/max [31,56,2,69...]

# Building
Load Visual Studio solution (Onnx2Text.sln), and build.

The target is Windows, as there are dependencies on WIC (Windows 7+) for image loading/saving, but it *might* compile fine for Linux too if you `#ifdef`'d those parts and don't need image conversion. Though, I don't use Linux frequently enough to support that target. 

# Build Google Protobuf .lib yourself:
The protobuf-3.5.1 directory in this project contains the bare minimum .lib and .h files to build
the executable. If you need a different configuration:

Download/clone:
https://github.com/protocolbuffers/protobuf/releases/tag/v3.5.1
or `git clone -b v3.5.1 https://github.com/google/protobuf.git`

    md protobuf-3.5.1\cmake\build\solution\
    cd protobuf-3.5.1\cmake\build\solution\
    cmake -G "Visual Studio 15 2017" -DCMAKE_INSTALL_PREFIX=../../../../install ../.. -Dprotobuf_BUILD_TESTS=OFF
    #cmake -G "Visual Studio 15 2017" -DCMAKE_INSTALL_PREFIX=../../../../install ../.. -Dprotobuf_BUILD_TESTS=OFF -Ax64
    start protobuf-3.5.1\cmake\build\solution\protobuf.sln
    # Build it

Build libprotobuf.vcxproj project. Find the .lib files under protobuf-3.5.1\cmake\build\solution\*.

# Updating ONNX Proto file.

Copy the newest version of `onnx.proto` from https://github.com/onnx/onnx/blob/master/onnx/onnx.proto.

Replace "LITE_RUNTIME" with "option optimize_for = CODE_SIZE;

Run the ProtoBuf compiler.

    protobuf3.5.1\bin\protoc.exe onnx.proto --cpp_out=. --error_format=msvs

    (local copy here: protobuf-3.5.1/cmake/build/solution/MinSizeRel/protoc.exe)
