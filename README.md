ConvertOnnxModel  
2018-07-19..2021-11-19  
Dwayne Robinson (FDwR)  

- Converts a binary [ONNX](https://github.com/onnx/onnx) model file to text (which can be edited in any simple text editor) and vice versa.
- Converts an ONNX tensor protobuf to text/CSV/PNG/NPY and vice versa.
- Generates output tensor values (ones, zeros, iota series, random).
- Exports model tensors to directory of tensor files.

# Usage
    ConvertOnnxModel.exe [options] inputFilename [outputFilename]

# Example usage

- Model to model
    - Model from ONNX binary protobuf format to prototxt format  
        `ConvertOnnxModel input.onnx output.prototxt`
    - Model in prototxt text format to binary protobuf ONNX format  
        `ConvertOnnxModel input.prototxt output.onnx`
    - Model from ONNX binary protobuf back to binary with zeroed tensor values  
        `ConvertOnnxModel -zeromodelvalues input.onnx output.onnx`

- Model to multiple tensors
    - Model from ONNX binary protobuf format to directory of NumPy tensors  
        `ConvertOnnxModel resnet50.onnx x:\resnet_*.npy`
    - Model from ONNX binary protobuf format to directory of raw data files  
        `ConvertOnnxModel squeezenet.onnx z:\folder\*_weight.dat`

- Tensor to tensor
    - Tensor from ONNX binary protobuf to comma separated values  
        `ConvertOnnxModel input.onnxtensor output.csv`
    - Tensor from ONNX binary protobuf (.pb) to image file  
        `ConvertOnnxModel -tensor input.pb output.png`
    - Tensor data as comma separated values to raw binary file  
        `ConvertOnnxModel -datatype uint8 -dimensions 224,224 Foo.csv Foo.dat`
    - Tensor from NumPy array format to protobuf ONNX binary format  
        `ConvertOnnxModel input.npy output.onnxtensor`

- Tensor from generated randomness to ONNX binary protobuf format  
    `ConvertOnnxModel -dimensions 3,4 -datatype float16 generate(random,1,24) output.onnxtensor`

# Parameters
* input/output files - graph (onnx/pb/text) or tensor (onnxtensor/npy/pb/text/csv/dat).
* -tensor - specifies the input file is a tensor (only needed if ambiguous file type like .pb).
* -graph - specifies the input file is a model (only needed if ambiguous file type like .pb).
* -dimensions - explicit tensor dimensions for .csv or .dat file. Defaults to 1D element count from source data. Pass "()" to indicate 0D scalar.
* -datatype - tensor element type (float16,float32,float64,int8,uint8,int16,uint16,int32,uint32,int64,uint64,bool8). This isn't usually needed unless reading from raw data.
* -zeromodelvalues - zero any tensor values in model (clears model initializer weights - useful for sharing confidential models without revealing trained results) except tiny 1D tensors needed for shapes.
* -row - single row or range for .csv.
* -column - single column or range for .csv.
* -scale - scale tensor values during conversion.
* -inversescale - scale tensor values during conversion by reciprocal (e.g. 255 means 1/255).
* -normalizevalues - should normalize values in tensor 0 to 1.

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
Load Visual Studio solution (ConvertOnnxModel.sln), and build.

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

Run the ProtoBuf compiler.

    protobuf3.5.1\bin\protoc.exe onnx.proto --cpp_out=. --error_format=msvs
