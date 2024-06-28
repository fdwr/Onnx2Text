del ./build/Onnx2Text.pikensoft.*.zip > nul
\programs\file\7-Zip\7z.exe a -tzip -mx=9 ./build/Onnx2Text.pikensoft.%date%.zip -i@PackageFileList.txt %*
start https://github.com/fdwr/Onnx2Text/releases
