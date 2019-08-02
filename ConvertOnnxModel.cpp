#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING 1 // For Google protobuf using std::iterator as a base class in C++17.
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING 1
#define NOMINMAX

#include <iostream>
#include <fstream>
#include <sstream>
#include <string_view>
#include <filesystem>
#include <algorithm>
#include <numeric>

#pragma warning(push)
#pragma warning(disable: 4146)
#include "onnx-ml.pb.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#pragma warning(pop)

#include <d3d12.h>
#include <wrl/client.h>
#include <wincodec.h>

extern "C"
{
    HRESULT WINAPI WICCreateImagingFactory_Proxy(
        __in UINT SDKVersion,
        __deref_out IWICImagingFactory** IWICImagingFactory
    );
}

using Microsoft::WRL::ComPtr;

////////////////////////////////////////////////////////////////////////////////

void ThrowBadHResultRuntimeErrorWithMessage(HRESULT hr)
{
    std::stringstream stream;
    stream << "Failing HRESULT: 0x" << std::hex << hr;
    std::string result = std::move(stream.str());
    throw std::runtime_error(result.c_str());
}

#ifndef THROW_IF_FAILED
#define THROW_IF_FAILED(hr) {if (FAILED(hr)) ThrowBadHResultRuntimeErrorWithMessage(hr);}
#endif

std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>,wchar_t> g_converterToUtf8;

template <typename T>
class span
{
public:
    span() = default;

    template<typename ContiguousContainer>
    constexpr span(ContiguousContainer& container)
    :   begin_(std::data(container)),
        end_(begin_ + std::size(container))
    {
    }

    constexpr span(std::initializer_list<T> i)
    :   begin_(std::data(i)),
        end_(begin_ + std::size(i))
    {
    }

    span(T* begin, T* end)
    :   begin_(begin),
        end_(end)
    {
    }

    span(T* begin, size_t elementCount)
    :   begin_(begin),
        end_(begin + elementCount)
    {
    }

    T* data() noexcept { return begin_; }
    T* begin() noexcept { return begin_; }
    T* end() noexcept { return end_; }
    T const* data() const noexcept { return begin_; }
    T const* begin() const noexcept { return begin_; }
    T const* end() const noexcept { return end_; }
    bool empty() const noexcept { return end_ == begin_; }
    size_t size() const noexcept { return end_ - begin_; }
    size_t size_bytes() const noexcept { return sizeof(T) * size(); }
    T& operator[](size_t index) const noexcept { return begin_[index]; }
    span<T> subspan(size_t index, size_t count) { return span<T>(begin_ + index, begin_ + index + count); }

protected:
    T* begin_ = nullptr;
    T* end_ = nullptr;
};

template<typename T>
struct HalfOpenRange
{
    static_assert(std::is_trivial<T>::value);

    T begin;
    T end;

    constexpr HalfOpenRange() noexcept : begin(static_cast<T>(0)), end(static_cast<T>(0)) {}
    constexpr HalfOpenRange(T initialValue) noexcept : begin(initialValue), end(initialValue) {}
    constexpr HalfOpenRange(T initialBegin, T initialEnd) noexcept : begin(initialBegin), end(initialEnd) {}

    static HalfOpenRange None() { return { static_cast<T>(0), static_cast<T>(0) }; }
    static HalfOpenRange All() { return {std::numeric_limits<T>::min(), std::numeric_limits<T>::max()}; }

    bool IsEmpty() const noexcept { return begin == end; }
    bool empty() const noexcept { return IsEmpty(); }

    bool Contains(T value) const noexcept { return value >= begin && value < end; }
    bool Contains(HalfOpenRange const& other) const noexcept { return other.begin >= begin && other.end <= end;}

    void ExpandAllIfEmpty() noexcept
    {
        if (begin == end)
        {
            *this = All();
        }
    }
};

using HalfOpenRangeUint32 = HalfOpenRange<uint32_t>;

// Reinterprets a span of data from one type to another.
// The input parameter can be any contiguous container with data() and size() methods,
// including gsl::span, std::array, and std::vector.
template <typename NewType, typename OldTypeContainer>
span<NewType> reinterpret_span(OldTypeContainer& oldSpan)
{
    using OldType = decltype(*oldSpan.data());
    size_t newElementCount = static_cast<size_t>(oldSpan.size()) * sizeof(OldType) / sizeof(NewType);
    assert(newElementCount * sizeof(NewType) == oldSpan.size() * sizeof(OldType));

    NewType* p = reinterpret_cast<NewType*>(oldSpan.data());
    return span<NewType>(p, p + newElementCount);
}

std::string ReadTextFile(wchar_t const* inputFilename)
{
    std::string fileData;
    std::ifstream file(inputFilename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::ios::failure("Could not open input file.");
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    fileData.resize(size);
    file.read(fileData.data(), size);
    fileData.resize(size);

    return fileData;
}

void WriteBinaryFile(wchar_t const* outputFilename, span<const char> fileData)
{
    // Create any intermediate output path needed.
    std::experimental::filesystem::path path(outputFilename);
    if (path.has_parent_path())
    {
        path.remove_filename();
        // .filename() lies when referring to the root directory, saying there is a filename when
        // there actually is not. So instead we check whether the current path equals the root.
        std::experimental::filesystem::path root = path.root_path();
        if (path != root)
        {
            std::experimental::filesystem::create_directory(path);
        }
    }

    std::ofstream file(outputFilename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::ios::failure("Could not open output file.");
    }

    file.write(fileData.data(), fileData.size());
}

enum class FileExtensionType
{
    Unknown,             // .onnx
    OnnxModel,           // .onnx
    GoogleProtobuf,      // .pb
    Text,                // .txt
    CommaSeparatedValue, // .csv
    Image,               // .png / .jpg
    RawData,             // .dat / .bin - raw binary, dump of tensor values as-is
};

FileExtensionType GetFileExtensionType(std::wstring_view filename)
{
    std::wstring_view filenameExtension = filename.substr(filename.find_last_of(L".") + 1);
    if (filenameExtension == L"pb"  ) return FileExtensionType::GoogleProtobuf;
    if (filenameExtension == L"onnx") return FileExtensionType::OnnxModel;
    if (filenameExtension == L"txt" ) return FileExtensionType::Text;
    if (filenameExtension == L"csv" ) return FileExtensionType::CommaSeparatedValue;
    if (filenameExtension == L"dat" ) return FileExtensionType::RawData;
    if (filenameExtension == L"bin" ) return FileExtensionType::RawData;
    if (filenameExtension == L"png" ) return FileExtensionType::Image;
    if (filenameExtension == L"jpg" ) return FileExtensionType::Image;
    if (filenameExtension == L"jpeg") return FileExtensionType::Image;
    return FileExtensionType::Unknown;
}

size_t GetDataTypeElementByteSize(onnx::TensorProto::DataType dataType)
{
    switch (dataType)
    {
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:    return 2;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      return 4;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     return 8;
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       return 1;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      return 1;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       return 1;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     return 2;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      return 2;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     return 4;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      return 4;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     return 8;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      return 8;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  return 8;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: return 16;
    default: throw std::ios::failure("Unsupported data type in tensor.");
    }
}

void ReadCsv(
    span<const char> text,
    onnx::TensorProto::DataType dataType,
    HalfOpenRangeUint32 rowRange,
    HalfOpenRangeUint32 columnRange,
    /*out*/std::vector<char>& byteData
    )
{
    byteData.clear();

    rowRange.ExpandAllIfEmpty();
    columnRange.ExpandAllIfEmpty();

    size_t elementByteSize = GetDataTypeElementByteSize(dataType);
    size_t byteDataSize = 0;
    uint32_t row = 1, column = 1;

    std::string unquotedText;

    constexpr char quote = '\"';

    char const* begin = text.data();
    char const* end = text.data() + text.size();
    while (begin != end)
    {
        char const* numberStart = begin;

        // Skip leading spaces.
        while (begin != end && *begin == ' ')
        {
            ++begin;
        }

        // Read quoted field.
        if (begin != end && *begin == quote)
        {
            unquotedText.clear();

            while (++begin != end)
            {
                auto ch = *begin;
                if (ch == quote)
                {
                    ++begin; // skip the quote.
                    if (begin != end && *begin != quote)
                    {
                        break; // Found ending quote.
                    }
                }
                unquotedText.push_back(ch);
            }

            numberStart = unquotedText.data();
        }

        // Write the value to the byte buffer.
        if (rowRange.Contains(row) && columnRange.Contains(column))
        {
            // Read the numeric value.
            char* numberEnd;
            double value = strtod(numberStart, &numberEnd);

            byteData.resize(byteDataSize + elementByteSize);
            void* data = &byteData[byteDataSize];
            byteDataSize += elementByteSize;

            switch (dataType)
            {
            case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      *reinterpret_cast<float*>   (data) = static_cast<float>   (value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     *reinterpret_cast<double*>  (data) = static_cast<double>  (value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       *reinterpret_cast<bool*>    (data) = static_cast<bool>    (value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      *reinterpret_cast<uint8_t*> (data) = static_cast<uint8_t> (value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       *reinterpret_cast<int8_t*>  (data) = static_cast<int8_t>  (value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     *reinterpret_cast<uint16_t*>(data) = static_cast<uint16_t>(value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      *reinterpret_cast<int16_t*> (data) = static_cast<int16_t> (value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     *reinterpret_cast<uint32_t*>(data) = static_cast<uint32_t>(value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      *reinterpret_cast<int32_t*> (data) = static_cast<int32_t> (value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     *reinterpret_cast<uint64_t*>(data) = static_cast<uint64_t>(value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      *reinterpret_cast<int64_t*> (data) = static_cast<int64_t> (value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  *reinterpret_cast<float*>   (data) = static_cast<float>   (value); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: *reinterpret_cast<double*>  (data) = static_cast<double>  (value); break;
            // case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16
            default: throw std::ios::failure("Unsupported data type in tensor for raw output.");
            }
        }

        ++column;

        // Skip any following comma or line return.
        for (; begin != end; ++begin)
        {
            char ch = *begin;
            if (ch == ',') // Comma ends value.
            {
                ++begin;
                break;
            }
            if (ch == '\x000A' || ch == '\x000D') // Line feed or carriage return ends value.
            {
                ++begin;
                ++row;
                column = 1;
                if (ch == '\x000D' && begin != end && *begin == '\x000A')
                {
                    ++begin;
                }
                break;
            }
        }
    }
}

// Simpler version just for reading command line arguments.
void ReadCsv(span<const char> text, /*out*/std::vector<int32_t>& values)
{
    values.clear();
    char const* begin = text.data();
    char const* end = text.data() + text.size();
    while (begin != end)
    {
        char* valueEnd;
        uint32_t value = strtol(begin, &valueEnd, 10);
        values.push_back(value);
        if (valueEnd != end && *valueEnd == ',')
        {
            ++valueEnd;
        }
        begin = valueEnd;
    }
}

void WriteCsv(
    /*out*/span<char const> byteData,
    onnx::TensorProto::DataType dataType,
    /*out*/std::string& text
    )
{
    text.clear();

    size_t elementByteSize = GetDataTypeElementByteSize(dataType);
    size_t byteDataSize = 0;

    char buffer[40];

    // Round off any potential padding.
    byteData = span<char const>(byteData.data(), (byteData.size() / elementByteSize) * elementByteSize);

    char const* begin = byteData.data();
    char const* end = byteData.data() + byteData.size();
    while (begin != end)
    {
        // Read the next value from the type buffer.
        double value;
        void const* data = begin;
        begin += elementByteSize;

        switch (dataType)
        {
        case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      value = static_cast<double>(*reinterpret_cast<const float*>   (data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     value = static_cast<double>(*reinterpret_cast<const double*>  (data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       value = static_cast<double>(*reinterpret_cast<const bool*>    (data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      value = static_cast<double>(*reinterpret_cast<const uint8_t*> (data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       value = static_cast<double>(*reinterpret_cast<const int8_t*>  (data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     value = static_cast<double>(*reinterpret_cast<const uint16_t*>(data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      value = static_cast<double>(*reinterpret_cast<const int16_t*> (data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     value = static_cast<double>(*reinterpret_cast<const uint32_t*>(data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      value = static_cast<double>(*reinterpret_cast<const int32_t*> (data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     value = static_cast<double>(*reinterpret_cast<const uint64_t*>(data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      value = static_cast<double>(*reinterpret_cast<const int64_t*> (data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  value = static_cast<double>(*reinterpret_cast<const float*>   (data)); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: value = static_cast<double>(*reinterpret_cast<const double*>  (data)); break;
        // case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16
        default: throw std::ios::failure("Unsupported data type in tensor for raw output.");
        }

        // Write the next value as value.
        sprintf_s(buffer, "%g", value);
        text.append(buffer);
        if (begin != end)
        {
            text.push_back(',');
        }
    }
}

std::vector<int32_t> ResolveEmptyDimensions(
    span<int32_t const> defaultDimensions,
    span<char const> byteData,
    onnx::TensorProto::DataType dataType
    )
{
    std::vector<int32_t> resolvedDimensions(defaultDimensions.begin(), defaultDimensions.end());

    // Use a 1D array equal to the element count if no dimensions were given.
    if (resolvedDimensions.empty())
    {
        size_t elementByteSize = GetDataTypeElementByteSize(dataType);
        size_t elementCount = byteData.size() / elementByteSize;
        resolvedDimensions.push_back(static_cast<int32_t>(elementCount));
    }

    return resolvedDimensions;
}

void ConvertModel(
    _In_z_ wchar_t const* inputFilename,
    _In_z_ wchar_t const* outputFilename
    )
{
    FileExtensionType inputFileExtensionType  = GetFileExtensionType(std::wstring_view(inputFilename));
    FileExtensionType outputFileExtensionType = GetFileExtensionType(std::wstring_view(outputFilename));

    onnx::ModelProto model;

    bool succeeded = false;
    if (inputFileExtensionType == FileExtensionType::Text)
    {
        std::string modelString = ReadTextFile(inputFilename);

        // Essentially "google::protobuf::TextFormat::ParseFromString(modelString, &model)"
        // except that we need to pass the flag to allow field numbers.

        google::protobuf::TextFormat::Parser parser;
        parser.AllowFieldNumber(true);
        succeeded = parser.ParseFromString(modelString, &model);
    }
    else if (inputFileExtensionType == FileExtensionType::OnnxModel
          || inputFileExtensionType == FileExtensionType::GoogleProtobuf)
    {
        std::ifstream ifs(inputFilename, std::ios::binary);
        succeeded = model.ParseFromIstream(&ifs);
    }
    else
    {
        throw std::invalid_argument("Unknown input graph file extension.");
    }

    if (!succeeded)
    {
        throw std::ios::failure("Could not parse input graph file.");
    }

    if (outputFileExtensionType == FileExtensionType::Text)
    {
        // Write the whole model to a text file.
        // Use the stream instead of google::protobuf::TextFormat::PrintToString,
        // which can fail for models that are >= 200MBs by running out of memory.
        std::ofstream outputFile(outputFilename, std::ios::out);
        std::unique_ptr<google::protobuf::io::ZeroCopyOutputStream> output(new google::protobuf::io::OstreamOutputStream(&outputFile));
        succeeded = google::protobuf::TextFormat::Print(model, output.get());
    }
    else if (outputFileExtensionType == FileExtensionType::OnnxModel
          || outputFileExtensionType == FileExtensionType::GoogleProtobuf)
    {
        std::ofstream os(outputFilename, std::ios::binary);
        succeeded = model.SerializeToOstream(&os);
    }
    else
    {
        throw std::invalid_argument("Unknown output graph file extension.");
    }

    if (!succeeded)
    {
        throw std::ios::failure("Could not serialize output graph file.");
    }
}

uint32_t ComputeElementCount(span<int32_t const> dimensions)
{
    return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int32_t>());
}

// Aligns with onnx::TensorProto::DataType.
const char* g_elementDataTypeNames[] =
{
    "undefined",    // Undefined = 0,
    "float32",      // Float32 = 1,
    "uint8",        // Uint8 = 2,
    "int8",         // Int8 = 3,
    "uint16",       // Uint16 = 4,
    "int16",        // Int16 = 5,
    "int32",        // Int32 = 6,
    "int64",        // Int64 = 7,
    "string8",      // StringChar8 = 8,
    "bool8",        // Bool = 9,
    "float16",      // Float16 = 10,
    "float64",      // Float64 = 11,
    "uint32",       // Uint32 = 12,
    "uint64",       // Uint64 = 13,
    "complex64",    // Complex64 = 14,
    "complex128",   // Complex128 = 15,
};

constexpr uint32_t g_elementDataTypeByteSizes[] =
{
    0, // Undefined = 0,
    4, // Float32 = 1,
    1, // Uint8 = 2,
    1, // Int8 = 3,
    2, // Uint16 = 4,
    2, // Int16 = 5,
    4, // Int32 = 6,
    8, // Int64 = 7,
    0, // StringChar8 = 8,
    1, // Bool = 9,
    2, // Float16 = 10,
    8, // Float64 = 11,
    4, // Uint32 = 12,
    8, // Uint64 = 13,
    8, // Complex64 = 14,
    16,// Complex128 = 15,
};

std::string_view GetStringNameFromDataType(onnx::TensorProto::DataType dataType) noexcept
{
    size_t index = static_cast<size_t>(dataType);
    return g_elementDataTypeNames[index < std::size(g_elementDataTypeNames) ? index : 0];
}

uint32_t GetByteSizeFromDataType(onnx::TensorProto::DataType dataType) noexcept
{
    size_t index = static_cast<size_t>(dataType);
    return g_elementDataTypeByteSizes[index < std::size(g_elementDataTypeNames) ? index : 0];
}

onnx::TensorProto::DataType GetDataTypeFromStringName(std::string_view name) noexcept
{
    auto i = std::find(std::begin(g_elementDataTypeNames), std::end(g_elementDataTypeNames), name);
    return (i == std::end(g_elementDataTypeNames))
        ? onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED
        : onnx::TensorProto::DataType(i - std::begin(g_elementDataTypeNames));
}

bool IsRecognizedChannelLayoutString(std::string_view channelLayoutString)
{
    return channelLayoutString == "nchw" || channelLayoutString == "nhwc";
}

struct Struct128Bit
{
    uint32_t data[4];
};

void RearrangeChannels(
    onnx::TensorProto::DataType dataType,
    span<int32_t const> dimensions,
    std::string_view originalChannelLayoutString,
    std::string_view desiredChannelLayoutString,
    /*inout*/ std::vector<uint8_t>& pixelBytes
    )
{
    if (!IsRecognizedChannelLayoutString(originalChannelLayoutString) || !IsRecognizedChannelLayoutString(desiredChannelLayoutString))
    {
        throw std::invalid_argument("Channel layout must be nchw or nhwc.");
    }

    if (originalChannelLayoutString == desiredChannelLayoutString)
    {
        return; // No work.
    }
    if (dimensions.size() <= 2)
    {
        return; // Nothing channels reorder since only 1 exists.
    }

    std::vector<uint8_t> destinationPixelBytes(pixelBytes.size());

    // Flatten the count of all the dimensions before the color channel.
    const span<int32_t const> chwDimensions = dimensions.subspan(dimensions.size() - 3, 3);
    const span<int32_t const> batchDimensions = dimensions.subspan(0, dimensions.size() - 3);
    const uint32_t destinationElementByteSize = GetByteSizeFromDataType(dataType);
    const uint32_t channelCount   = chwDimensions[0];
    const uint32_t heightCount    = chwDimensions[1];
    const uint32_t widthCount     = chwDimensions[2];
    const uint32_t batchCount     = ComputeElementCount(batchDimensions);
    const uint32_t totalByteCount = destinationElementByteSize * channelCount * heightCount * widthCount * batchCount;

    if (totalByteCount != destinationPixelBytes.size())
    {
        throw std::invalid_argument("Pixel total byte count does not match dimension counts.");
    }

    size_t destinationByteOffset = 0;
    size_t sourceOffset0 = 0, sourceOffset1 = 0, sourceOffset2 = 0, sourceOffset3 = 0;

    uint32_t sourceStride0, sourceStride1, sourceStride2, sourceStride3;
    uint32_t count0, count1, count2, count3;
    if (desiredChannelLayoutString == "nchw")
    {
        sourceStride1  = destinationElementByteSize;   // channel stride
        sourceStride3  = channelCount * sourceStride1; // width stride
        sourceStride2  = sourceStride3 * widthCount;   // height stride
        sourceStride0  = sourceStride2 * heightCount;  // batch stride
        count0 = batchCount;
        count1 = channelCount;
        count2 = heightCount;
        count3 = widthCount;
    }
    else if (desiredChannelLayoutString == "nhwc")
    {
        sourceStride2  = destinationElementByteSize;    // width stride
        sourceStride1  = sourceStride2 * widthCount;    // height stride
        sourceStride3  = sourceStride1 * heightCount;   // channel stride
        sourceStride0  = sourceStride3 * channelCount;  // batch stride
        count0 = batchCount;
        count1 = heightCount;
        count2 = widthCount;
        count3 = channelCount;
    }

    // This simple function only supports swapping NCHW <-> NHWC, no transpose or flipping or
    // arbitrary dimension remapping.
    for (uint32_t index0 = 0; index0 < count0; ++index0)
    {
        sourceOffset1 = sourceOffset0;
        sourceOffset0 += sourceStride0;
        for (uint32_t index1 = 0; index1 < count1; ++index1)
        {
            sourceOffset2 = sourceOffset1;
            sourceOffset1 += sourceStride1;
            for (uint32_t index2 = 0; index2 < count2; ++index2)
            {
                sourceOffset3 = sourceOffset2;
                sourceOffset2 += sourceStride2;
                for (uint32_t index3 = 0; index3 < count3; ++index3)
                {
                    switch (destinationElementByteSize)
                    {
                    case 1: destinationPixelBytes[destinationByteOffset] = pixelBytes[sourceOffset3]; break;
                    case 2: reinterpret_cast<uint16_t&>(destinationPixelBytes[destinationByteOffset]) = reinterpret_cast<uint16_t&>(pixelBytes[sourceOffset3]); break;
                    case 4: reinterpret_cast<uint32_t&>(destinationPixelBytes[destinationByteOffset]) = reinterpret_cast<uint32_t&>(pixelBytes[sourceOffset3]); break;
                    case 8: reinterpret_cast<uint64_t&>(destinationPixelBytes[destinationByteOffset]) = reinterpret_cast<uint64_t&>(pixelBytes[sourceOffset3]); break;
                    case 16: reinterpret_cast<Struct128Bit&>(destinationPixelBytes[destinationByteOffset]) = reinterpret_cast<Struct128Bit&>(pixelBytes[sourceOffset3]); break;
                    }
                    sourceOffset3 += sourceStride3;
                    destinationByteOffset += destinationElementByteSize;
                }
            }
        }
    }

    pixelBytes = std::move(destinationPixelBytes);
}

struct PixelFormatAttributes
{
    std::string_view pixelFormatString;
    WICPixelFormatGUID const& guid;
    uint8_t channelCount;
    uint8_t bytesPerChannel; // Only accepts homogenous channels.
    onnx::TensorProto::DataType dataType;
};

constexpr PixelFormatAttributes g_pixelFormatAttributes[] =
{
    {"gray8", GUID_WICPixelFormat8bppGray, 1, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {"b8g8r8", GUID_WICPixelFormat24bppBGR, 3, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {"r8g8b8", GUID_WICPixelFormat24bppRGB, 3, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {"b8g8r8a8", GUID_WICPixelFormat32bppBGRA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {"r8g8b8a8", GUID_WICPixelFormat32bppPRGBA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {"pb8g8r8a8", GUID_WICPixelFormat32bppPBGRA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {"pr8g8b8a8", GUID_WICPixelFormat32bppPRGBA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {"b32g32r32", GUID_WICPixelFormat32bppPRGBA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {"r32g32b32x32", GUID_WICPixelFormat128bppRGBFloat, 4, 4, onnx::TensorProto::DataType::TensorProto_DataType_FLOAT},
};

bool ResolvePixelFormat(
    std::string_view pixelFormatString,
    _Out_ WICPixelFormatGUID const*& pixelFormatGuid,
    _Out_ uint32_t& channelCount,
    _Out_ uint32_t& bytesPerChannel,
    _Out_ onnx::TensorProto::DataType& dataType
    )
{
    pixelFormatGuid = nullptr;
    bytesPerChannel = 0;
    channelCount = 0;
    dataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;

    for (auto& attributes : g_pixelFormatAttributes)
    {
        if (pixelFormatString == attributes.pixelFormatString)
        {
            pixelFormatGuid = &attributes.guid;
            channelCount = attributes.channelCount;
            bytesPerChannel = attributes.bytesPerChannel;
            dataType = attributes.dataType;
            return true;
        }
    }
    return false;
}

bool ResolvePixelFormat(
    WICPixelFormatGUID const& pixelFormatGuid,
    _Out_ std::string_view& pixelFormatString,
    _Out_ uint32_t& channelCount,
    _Out_ uint32_t& bytesPerChannel,
    _Out_ onnx::TensorProto::DataType& dataType
    )
{
    pixelFormatString = std::string_view{};
    bytesPerChannel = 0;
    channelCount = 0;
    dataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;

    for (auto& attributes : g_pixelFormatAttributes)
    {
        if (pixelFormatGuid == attributes.guid)
        {
            pixelFormatString = attributes.pixelFormatString;
            channelCount = attributes.channelCount;
            bytesPerChannel = attributes.bytesPerChannel;
            dataType = attributes.dataType;
            return true;
        }
    }
    return false;
}

bool ResolvePixelFormat(
    uint32_t channelCount,
    onnx::TensorProto::DataType dataType,
    _Out_ std::string_view& pixelFormatString,
    _Out_ WICPixelFormatGUID const*& pixelFormatGuid,
    _Out_ uint32_t& bytesPerChannel
    )
{
    pixelFormatString = std::string_view{};
    pixelFormatGuid = nullptr;
    bytesPerChannel = 0;

    for (auto& attributes : g_pixelFormatAttributes)
    {
        if (channelCount == attributes.channelCount && dataType == attributes.dataType)
        {
            pixelFormatString = attributes.pixelFormatString;
            pixelFormatGuid = &attributes.guid;
            bytesPerChannel = attributes.bytesPerChannel;
            return true;
        }
    }
    return false;
}

template <typename T, size_t sourceElementByteStride = sizeof(T)>
void ConvertElementTypeToUInt8(
    _In_reads_bytes_(elementCount * sourceElementByteStride) uint8_t const* source,
    _In_reads_(elementCount) uint8_t* destination,
    size_t elementCount
    )
{
    // std::copy gives warnings about casting, but we explicitly do want the cast, even if there is bit loss.
    for (; elementCount != 0; --elementCount)
    {
        T const* recastSource = reinterpret_cast<T const*>(source);
        *destination++ = static_cast<uint8_t>(*recastSource);
        source += sourceElementByteStride;
    }
}

uint32_t GetElementCountFromByteSpan(onnx::TensorProto::DataType dataType, span<uint8_t const> source)
{
    const uint32_t elementByteSize = GetByteSizeFromDataType(dataType);
    if (elementByteSize == 0)
    {
        throw std::invalid_argument("Unknown element data type.");
    }
    return static_cast<uint32_t>(source.size_bytes() / elementByteSize);
}

void ConvertElementTypeToUInt8(
    onnx::TensorProto::DataType dataType,
    span<uint8_t const> source,
    span<uint8_t> destination
    )
{
    const uint32_t sourceElementCount = GetElementCountFromByteSpan(dataType, source);
    if (sourceElementCount != destination.size_bytes())
    {
        throw std::invalid_argument("Source and destination must have same element count.");
    }

    switch (dataType)
    {
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      ConvertElementTypeToUInt8<float>   (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     ConvertElementTypeToUInt8<double>  (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       ConvertElementTypeToUInt8<bool>    (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      ConvertElementTypeToUInt8<uint8_t> (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       ConvertElementTypeToUInt8<int8_t>  (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     ConvertElementTypeToUInt8<uint16_t>(source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      ConvertElementTypeToUInt8<int16_t> (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     ConvertElementTypeToUInt8<uint32_t>(source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      ConvertElementTypeToUInt8<int32_t> (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     ConvertElementTypeToUInt8<uint64_t>(source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      ConvertElementTypeToUInt8<int64_t> (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  ConvertElementTypeToUInt8<float, sizeof(float)*2>(source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: ConvertElementTypeToUInt8<double, sizeof(double)*2>(source.data(), destination.data(), sourceElementCount); break;
    // case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16
    default: throw std::ios::failure("Unsupported data type in tensor.");
    }
}

void LoadImageData(
    _In_z_ wchar_t const* inputFilename, // Alternately could specify a span<const uint8_t>.
    std::string_view pixelFormatString,
    onnx::TensorProto::DataType& dataType,
    /*out*/ std::vector<int32_t>& dimensions,
    /*out*/ std::vector<uint8_t>& pixelBytes
    )
{
    pixelBytes.clear();

    WICPixelFormatGUID const* resolvedPixelFormatGuid = nullptr;
    uint32_t channelCount, bytesPerChannel;
    ResolvePixelFormat(pixelFormatString, /*out*/ resolvedPixelFormatGuid, /*out*/ channelCount, /*out*/ bytesPerChannel, /*out*/ dataType);

    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(WICCreateImagingFactory_Proxy(WINCODEC_SDK_VERSION1, OUT &wicFactory));

    // Decompress the image using WIC.
    ComPtr<IWICStream> stream;
    ComPtr<IWICBitmapDecoder> decoder;
    ComPtr<IWICBitmapFrameDecode> bitmapFrame;
    ComPtr<IWICFormatConverter> converter;
    IWICBitmapSource* pixelSource = nullptr;

    THROW_IF_FAILED(wicFactory->CreateStream(&stream));
    THROW_IF_FAILED(stream->InitializeFromFilename(inputFilename, GENERIC_READ));
    #if 0
    THROW_IF_FAILED(stream->InitializeFromMemory(
        const_cast<uint8_t*>(fileBytes.data()),
        static_cast<uint32_t>(fileBytes.size_bytes()))
        );
    #endif
    THROW_IF_FAILED(wicFactory->CreateDecoderFromStream(stream.Get(), nullptr, WICDecodeMetadataCacheOnLoad, OUT &decoder));
    THROW_IF_FAILED(decoder->GetFrame(0, OUT &bitmapFrame));

    WICPixelFormatGUID actualPixelFormatGuid;
    THROW_IF_FAILED(bitmapFrame->GetPixelFormat(/*out*/ &actualPixelFormatGuid));

    // Convert format to 32bppPBGRA - which D2D uses and the test's blend function expects.
    if (resolvedPixelFormatGuid != nullptr && actualPixelFormatGuid != *resolvedPixelFormatGuid)
    {
        THROW_IF_FAILED(wicFactory->CreateFormatConverter(&converter));
        THROW_IF_FAILED(converter->Initialize(
            bitmapFrame.Get(),
            *resolvedPixelFormatGuid,
            WICBitmapDitherTypeNone,
            nullptr,
            0.f,
            WICBitmapPaletteTypeMedianCut
            ));
        pixelSource = converter.Get();
    }
    // Just use the type in the image, but verify that it's a type we actually recognize.
    else
    {
        if (!ResolvePixelFormat(actualPixelFormatGuid, /*out*/ pixelFormatString, /*out*/ channelCount, /*out*/ bytesPerChannel, /*out*/ dataType))
        {
            throw std::invalid_argument("Pixel format in image is not recognized for reading.");
        }
        pixelSource = bitmapFrame.Get();
    }

    // Copy the pixels out of the IWICBitmapSource.
    uint32_t width, height;
    THROW_IF_FAILED(pixelSource->GetSize(OUT &width, OUT &height));
    const uint32_t bytesPerPixel = channelCount * bytesPerChannel;
    const uint32_t rowByteStride = width * bytesPerPixel;
    const uint32_t bufferByteSize = rowByteStride * height;
    pixelBytes.resize(bufferByteSize);
    WICRect rect = {0, 0, static_cast<INT>(width), static_cast<INT>(height)};
    THROW_IF_FAILED(pixelSource->CopyPixels(&rect, rowByteStride, bufferByteSize, OUT pixelBytes.data()));

    dimensions.assign({1, int32_t(channelCount), int32_t(height), int32_t(width)});
}

void StoreImageData(
    span<const uint8_t> pixelBytes,
    std::string_view pixelFormatString, // currently only supports "b8g8r8".
    onnx::TensorProto::DataType dataType,
    span<int32_t const> dimensions,
    _In_z_ wchar_t const* outputFilename
    )
{
    std::wstring_view filename = std::wstring_view(outputFilename);
    std::wstring_view filenameExtension = filename.substr(filename.find_last_of(L".") + 1);
    if (filenameExtension != L"png")
    {
        throw std::invalid_argument("Only .png is supported for writing files.");
    }
    if (dimensions.size() < 2)
    {
        throw std::invalid_argument("Dimensions must be at least 2 for height and width.");
    }
    // TODO: Support non-8bit pixel types.
    if (pixelFormatString != "b8g8r8")
    {
        throw std::invalid_argument("Only supported pixelFormatString is b8g8r8.");
    }

    // TODO: Support non-8bit pixel types.
    // For now, convert any format larger than 8-bits to 8-bit.
    std::vector<uint8_t> pixelBytesBuffer;
    if (dataType != onnx::TensorProto::DataType::TensorProto_DataType_UINT8)
    {
        const uint32_t sourceElementCount = GetElementCountFromByteSpan(dataType, pixelBytes);
        pixelBytesBuffer.resize(sourceElementCount);
        ConvertElementTypeToUInt8(dataType, pixelBytes, /*out*/ pixelBytesBuffer);
        pixelBytes = pixelBytesBuffer;
        dataType = onnx::TensorProto::DataType::TensorProto_DataType_UINT8;
    }

    WICPixelFormatGUID const* resolvedPixelFormatGuid = nullptr;
    const uint32_t channelCount = dimensions.size() >= 3 ? dimensions[dimensions.size() - 3] : 1;
    uint32_t bytesPerChannel;
    if (!ResolvePixelFormat(channelCount, dataType, /*out*/ pixelFormatString, /*out*/ resolvedPixelFormatGuid, /*out*/ bytesPerChannel))
    {
        throw std::invalid_argument("Pixel format is not supported for writing.");
    };

    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(WICCreateImagingFactory_Proxy(WINCODEC_SDK_VERSION1, OUT &wicFactory));

    // Decompress the image using WIC.
    ComPtr<IWICStream> stream;
    ComPtr<IWICBitmapEncoder> encoder;
    ComPtr<IWICBitmapFrameEncode> bitmapFrame;
    ComPtr<IPropertyBag2> propertybag;

    THROW_IF_FAILED(wicFactory->CreateStream(&stream));
    THROW_IF_FAILED(stream->InitializeFromFilename(outputFilename, GENERIC_WRITE));
    THROW_IF_FAILED(wicFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, OUT &encoder));
    THROW_IF_FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache));
    THROW_IF_FAILED(encoder->CreateNewFrame(OUT &bitmapFrame, &propertybag));

    #if 0
    // This is how you customize the TIFF output.
    {
        PROPBAG2 option = {};
        option.pstrName = L"TiffCompressionMethod";
        VARIANT varValue;    
        VariantInit(&varValue);
        varValue.vt = VT_UI1;
        varValue.bVal = WICTiffCompressionZIP;      
        THROW_IF_FAILED(propertybag->Write(1, &amp; option, &amp; varValue));
        THROW_IF_FAILED(bitmapFrame->Initialize(propertybag));
    }
    #endif
    THROW_IF_FAILED(bitmapFrame->Initialize(propertybag.Get()));

    const span<int32_t const> hwDimensions = dimensions.subspan(dimensions.size() - 2, 2);
    const uint32_t height = hwDimensions[0];
    const uint32_t width = hwDimensions[1];
    const uint32_t bytesPerPixel = channelCount * bytesPerChannel;
    const uint32_t rowByteStride = width * bytesPerPixel;
    const uint32_t bufferByteSize = rowByteStride * height;

    THROW_IF_FAILED(bitmapFrame->SetSize(width, height));

    WICPixelFormatGUID actualPixelFormatGuid = *resolvedPixelFormatGuid;
    THROW_IF_FAILED(bitmapFrame->SetPixelFormat(/*inout*/ &actualPixelFormatGuid));

    // Assign to a temporary large buffer if the specified dimensions are larger than the passed
    // pixel content. The remaining pixels will just be empty blackness.
    if (bufferByteSize > pixelBytes.size_bytes())
    {
        pixelBytesBuffer.reserve(bufferByteSize);
        pixelBytesBuffer.assign(pixelBytes.begin(), pixelBytes.end());
        pixelBytesBuffer.resize(bufferByteSize);
        pixelBytes = pixelBytesBuffer;
    }

    // Why is the WritePixels input parameter not const??
    BYTE* recastPixelBytes = const_cast<BYTE*>(reinterpret_cast<BYTE const*>(pixelBytes.data()));
    THROW_IF_FAILED(bitmapFrame->WritePixels(height, rowByteStride, bufferByteSize, recastPixelBytes));
    THROW_IF_FAILED(bitmapFrame->Commit());
    THROW_IF_FAILED(encoder->Commit());
}

void MakeTensor(
    span<char const> byteData,
    onnx::TensorProto::DataType dataType,
    span<int32_t const> dimensions,
    std::string_view name,
    _Inout_ onnx::TensorProto& onnxTensor
    )
{
    // Write name, tensor element type, dimensions, and raw byte data.
    onnxTensor.set_name(name.data(), name.size());

    for (auto d : dimensions)
    {
        onnxTensor.add_dims(d);
    }

    onnxTensor.set_data_type(dataType);
    onnxTensor.set_raw_data(byteData.data(), byteData.size());
}

template<typename OutputElementType, typename Iterator, typename ContiguousByteOutputContainer>
void CopyOnnxTensorDataToBuffer(
    Iterator begin,
    Iterator end,
    size_t elementCount,
    ContiguousByteOutputContainer& outputContainer
    )
{
    static_assert(sizeof(*outputContainer.begin()) == 1);
    constexpr size_t inputElementByteSize = sizeof(*begin);
    constexpr size_t outputElementSize = sizeof(OutputElementType);
    outputContainer.resize(elementCount * outputElementSize);

    span<OutputElementType> outputValues = reinterpret_span<OutputElementType>(outputContainer);
    size_t index = 0;
    for (auto i = begin; i != end; ++i)
    {
        outputValues[index++] = static_cast<OutputElementType>(*i);
    }
}

std::string GetOnnxTensorRawByteData(onnx::TensorProto tensor)
{
    std::string values;
    if (tensor.has_raw_data())
    {
        values = tensor.raw_data();
    }
    else
    {
        switch (tensor.data_type())
        {
        case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:    CopyOnnxTensorDataToBuffer<uint16_t>(tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      CopyOnnxTensorDataToBuffer<float>   (tensor.float_data().begin(),  tensor.float_data().end(),  tensor.float_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     CopyOnnxTensorDataToBuffer<double>  (tensor.double_data().begin(), tensor.double_data().end(), tensor.double_data_size(), values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       CopyOnnxTensorDataToBuffer<bool>    (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      CopyOnnxTensorDataToBuffer<uint8_t> (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       CopyOnnxTensorDataToBuffer<int8_t>  (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     CopyOnnxTensorDataToBuffer<uint16_t>(tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      CopyOnnxTensorDataToBuffer<int16_t> (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     CopyOnnxTensorDataToBuffer<uint32_t>(tensor.uint64_data().begin(), tensor.uint64_data().end(), tensor.uint64_data_size(), values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      CopyOnnxTensorDataToBuffer<int32_t> (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     CopyOnnxTensorDataToBuffer<uint64_t>(tensor.uint64_data().begin(), tensor.uint64_data().end(), tensor.uint64_data_size(), values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      CopyOnnxTensorDataToBuffer<int64_t> (tensor.int64_data().begin(),  tensor.int64_data().end(),  tensor.int64_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  CopyOnnxTensorDataToBuffer<float>   (tensor.float_data().begin(),  tensor.float_data().end(),  tensor.float_data_size(),  values); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: CopyOnnxTensorDataToBuffer<double>  (tensor.double_data().begin(), tensor.double_data().end(), tensor.double_data_size(), values); break;
        default: throw std::ios::failure("Unsupported data type in tensor for raw output.");
        }
    }

    return values;
}

void ConvertTensor(
    _In_z_ wchar_t const* inputFilename,
    span<int32_t const> dimensions,
    onnx::TensorProto::DataType dataType,
    HalfOpenRangeUint32 rowRange,           // matters for CSV files
    HalfOpenRangeUint32 columnRange,        // matters for CSV files
    std::string_view pixelFormatString,     // matters for image files
    std::string_view channelLayoutString,   // matters for image files
    _In_z_ wchar_t const* outputFilename
    )
{
    FileExtensionType inputFileExtensionType  = GetFileExtensionType(std::wstring_view(inputFilename));
    FileExtensionType outputFileExtensionType = GetFileExtensionType(std::wstring_view(outputFilename));

    // Set defaults.
    if (dataType == onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED)
    {
        dataType = onnx::TensorProto::DataType::TensorProto_DataType_FLOAT;
    }

    if (channelLayoutString.empty())
    {
        channelLayoutString = "nchw";
    }

    onnx::TensorProto tensor;
    std::vector<int32_t> resolvedDimensions;

    if (!dimensions.empty())
    {
        if (inputFileExtensionType == FileExtensionType::Text
        ||  inputFileExtensionType == FileExtensionType::OnnxModel
        ||  inputFileExtensionType == FileExtensionType::GoogleProtobuf
        ||  inputFileExtensionType == FileExtensionType::Image)
        {
            throw std::invalid_argument("\"dimensions\" are invalid when reading from this file type.");
        }
    }

    bool succeeded = true;
    if (inputFileExtensionType == FileExtensionType::Text)
    {
        std::string modelString = ReadTextFile(inputFilename);
        succeeded = google::protobuf::TextFormat::ParseFromString(modelString, &tensor);
    }
    else if (inputFileExtensionType == FileExtensionType::OnnxModel
          || inputFileExtensionType == FileExtensionType::GoogleProtobuf)
    {
        std::ifstream ifs(inputFilename, std::ios::binary);
        succeeded = tensor.ParseFromIstream(&ifs);
    }
    else if (inputFileExtensionType == FileExtensionType::RawData)
    {
        std::string byteData = ReadTextFile(inputFilename);
        resolvedDimensions = ResolveEmptyDimensions(dimensions, byteData, dataType);

        MakeTensor(byteData, dataType, resolvedDimensions, "", /*out*/ tensor);
    }
    else if (inputFileExtensionType == FileExtensionType::CommaSeparatedValue)
    {
        std::string text = ReadTextFile(inputFilename);
        std::vector<char> byteData;
        resolvedDimensions = ResolveEmptyDimensions(dimensions, byteData, dataType);

        ReadCsv(/*out*/ text, dataType, rowRange, columnRange, /*out*/ byteData);
        MakeTensor(byteData, dataType, resolvedDimensions, "", /*out*/ tensor);
    }
    else if (inputFileExtensionType == FileExtensionType::Image)
    {
        std::vector<uint8_t> pixelBytes;
        LoadImageData(
            inputFilename,
            pixelFormatString,
            /*out*/ dataType, // Ignore the passed data type, using the image's data type instead.
            /*out*/ resolvedDimensions,
            /*out*/ pixelBytes
            );

        RearrangeChannels(dataType, resolvedDimensions, "nhwc", channelLayoutString, /*inout*/ pixelBytes);
        MakeTensor(reinterpret_span<const char>(pixelBytes), dataType, resolvedDimensions, "", /*out*/ tensor);
    }
    else
    {
        throw std::invalid_argument("Unknown input tensor file extension.");
    }

    if (!succeeded)
    {
        throw std::ios::failure("Could not parse input tensor file.");
    }

    // Read the data type and dimensions back from the tensor.
    dataType = tensor.data_type();
    if (resolvedDimensions.empty())
    {
        for (auto v : tensor.dims())
        {
            resolvedDimensions.push_back(static_cast<int32_t>(v));
        }
    }

    // Print details.
    {
        std::string dimensionsText;
        WriteCsv(reinterpret_span<char const>(resolvedDimensions), onnx::TensorProto::DataType::TensorProto_DataType_INT32, /*out*/ dimensionsText);
        printf("Tensor data type: %s, Dimensions: %s\r\n", GetStringNameFromDataType(dataType).data(), dimensionsText.c_str());
    }

    if (outputFileExtensionType == FileExtensionType::Text)
    {
        std::string modelString;
        if (succeeded = google::protobuf::TextFormat::PrintToString(tensor, &modelString))
        {
            WriteBinaryFile(outputFilename, modelString);
        }
    }
    else if (outputFileExtensionType == FileExtensionType::OnnxModel
          || outputFileExtensionType == FileExtensionType::GoogleProtobuf)
    {
        std::ofstream os(outputFilename, std::ios::binary);
        succeeded = tensor.SerializeToOstream(&os);
    }
    else if (outputFileExtensionType == FileExtensionType::RawData)
    {
        std::string byteData = GetOnnxTensorRawByteData(tensor);
        WriteBinaryFile(outputFilename, byteData);
    }
    else if (outputFileExtensionType == FileExtensionType::CommaSeparatedValue)
    {
        std::string byteData = GetOnnxTensorRawByteData(tensor);
        std::string text;
        WriteCsv(byteData, tensor.data_type(), /*out*/ text);
        WriteBinaryFile(outputFilename, text);
    }
    else if (outputFileExtensionType == FileExtensionType::Image)
    {
        std::string byteData = GetOnnxTensorRawByteData(tensor);
        std::vector<uint8_t> pixelBytes(byteData.data(), byteData.data() + byteData.size());
        RearrangeChannels(
            dataType,
            resolvedDimensions,
            channelLayoutString,
            "nhwc",
            /*inout*/ pixelBytes
            );
        StoreImageData(
            reinterpret_span<const uint8_t>(pixelBytes),
            "b8g8r8",
            dataType, // Ignore the passed data type, using the image's data type instead.
            resolvedDimensions,
            outputFilename
            );
    }
    else
    {
        throw std::invalid_argument("Unknown output tensor file extension.");
    }

    if (!succeeded)
    {
        throw std::ios::failure("Could not serialize tensor output file.");
    }
}

void PrintUsage()
{
    std::cout << "ConvertOnnxModel 2018-07-19..2019-08-01 FDR\r\n"
                 "Example usage:\r\n"
                 "    ConvertOnnxModel.exe Foo.onnx Foo.txt\r\n"
                 "    ConvertOnnxModel.exe Foo.txt Foo.onnx\r\n"
                 "    ConvertOnnxModel.exe -tensor Foo.pb Foo.csv\r\n"
                 "    ConvertOnnxModel.exe -tensor -dimensions 224,224 -datatype uint8 -row 2 -column 1,225 Foo.csv Foo.dat\r\n"
                 "    ConvertOnnxModel.exe -tensor Foo.pb Foo.png\r\n"
                 "\r\n"
                 "Parameters:\r\n"
                 "     input file: graph (onnx/pb/text) or tensor (pb/text/csv/dat)\r\n"
                 "    output file: graph (onnx/pb/text) or tensor (pb/text/csv/dat)\r\n"
                 "        -tensor: convert tensor instead of graph.\r\n"
                 "         -graph: convert graph (default).\r\n"
                 "    -dimensions: explicit tensor dimensions for .csv or .dat file. Defaults to\r\n"
                 "                 1D element count from source data.\r\n"
                 "      -datatype: tensor element type (float16,float32,float64,int8,uint8,int16,\r\n"
                 "                 uint16,int32,uint32,int64,uint64,bool8,complex64,complex128).\r\n"
                 "           -row: single row or range for .csv.\r\n"
                 "        -column: single column or range for .csv.\r\n"
                 ;
}

void ReadOpenHalfRange(std::string_view text, _Out_ HalfOpenRangeUint32& range)
{
    range = range.All();
    std::vector<int32_t> values;
    ReadCsv(text, /*out*/values);

    switch (values.size())
    {
    case 1: range = HalfOpenRangeUint32(values.front(), values.front() + 1); break;
    case 2: range = HalfOpenRangeUint32(values[0], values[1] + 1); break;
    case 0: throw std::invalid_argument("No values for range. Expect 1 or 2 values.");
    default: throw std::invalid_argument("Too many values for range. Expect 1 or 2 values.");
    }
}

enum class ConversionMode
{
    Unknown,
    Tensor,
    Graph,
    Total
};

const char* g_conversionModeNames[3] =
{
    "Unknown",
    "Tensor",
    "Graph",
};
static_assert(std::extent<decltype(g_conversionModeNames)>::value == uint32_t(ConversionMode::Total));

ConversionMode GetConversionModeFromFileExtensionType(FileExtensionType fileExtensionType)
{
    switch (fileExtensionType)
    {
    case FileExtensionType::Unknown: return ConversionMode::Unknown;
    case FileExtensionType::OnnxModel: return ConversionMode::Graph;
    case FileExtensionType::GoogleProtobuf: return ConversionMode::Unknown;
    case FileExtensionType::Text: return ConversionMode::Unknown;
    case FileExtensionType::CommaSeparatedValue: return ConversionMode::Tensor;
    case FileExtensionType::Image: return ConversionMode::Tensor;
    case FileExtensionType::RawData: return ConversionMode::Unknown;
    }
    return ConversionMode::Unknown;
}

int Main(int argc, wchar_t** argv)
{
    std::wstring inputFilename, outputFilename;
    std::string pixelFormatString, channelLayoutString;
    ConversionMode conversionMode = ConversionMode::Unknown;
    std::vector<int32_t> dimensions;
    onnx::TensorProto::DataType dataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
    HalfOpenRangeUint32 rowRange = {}, columnRange = {};

    for (int i = 1; i < argc; ++i)
    {
        auto argument = std::wstring_view(argv[i]);
        if (argument.front() == '-')
        {
            if (argument == L"-tensor")
            {
                conversionMode = ConversionMode::Tensor;
            }
            else if (argument == L"-graph")
            {
                conversionMode = ConversionMode::Graph;
            }
            else if (argument == L"-dimensions")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Dimensions expected: -dimensions 224,224");
                }

                std::string s = g_converterToUtf8.to_bytes(argv[i]);
                ReadCsv(s, /*out*/dimensions);
            }
            else if (argument == L"-datatype")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Data type expected: -datatype float32 (uint8, int32, int16, bool8, float64...)");
                }

                std::string s = g_converterToUtf8.to_bytes(argv[i]);
                dataType = GetDataTypeFromStringName(s);
            }
            else if (argument == L"-row")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Row number expected: -row 3 / -row 3,7");
                }
                std::string s = g_converterToUtf8.to_bytes(argv[i]);
                ReadOpenHalfRange(s, /*out*/ rowRange);
            }
            else if (argument == L"-column")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Column number expected: -column 2 / -column 2,100");
                }
                std::string s = g_converterToUtf8.to_bytes(argv[i]);
                ReadOpenHalfRange(s, /*out*/ columnRange);
            }
            else
            {
                throw std::invalid_argument("Unknown argument.");
            }
        }
        else
        {
            if (inputFilename.empty())
            {
                inputFilename = argument;
            }
            else if (outputFilename.empty())
            {
                outputFilename = argument;
            }
            else
            {
                throw std::invalid_argument("Too many files passed.");
            }
        }
    }

    if (inputFilename.empty() && outputFilename.empty())
    {
        std::cerr << "No input or output file names given.\r\n";
        PrintUsage();
        return EXIT_FAILURE;
    }

    // Print to console if no output file is given.
    if (outputFilename.empty())
    {
        outputFilename = L"con.txt";
    }

    // Deduce conversion mode from filename extension.
    if (conversionMode == ConversionMode::Unknown)
    {
        FileExtensionType inputFileExtensionType = GetFileExtensionType(std::wstring_view(inputFilename));
        FileExtensionType outputFileExtensionType = GetFileExtensionType(std::wstring_view(outputFilename));
        conversionMode = GetConversionModeFromFileExtensionType(inputFileExtensionType);
        if (conversionMode == ConversionMode::Unknown)
        {
            conversionMode = GetConversionModeFromFileExtensionType(outputFileExtensionType);
        }
    }

    printf(
        "Input filename:  %S\r\n"
        "Output filename: %S\r\n"
        "Conversion mode: %s\r\n",
        inputFilename.c_str(),
        outputFilename.c_str(),
        g_conversionModeNames[uint32_t(conversionMode)]
        );

    if (conversionMode == ConversionMode::Tensor)
    {
        ConvertTensor(
            inputFilename.c_str(),
            dimensions,
            dataType,
            rowRange,
            columnRange,
            pixelFormatString,
            channelLayoutString,
            outputFilename.c_str()
        );
    }
    else if (conversionMode == ConversionMode::Graph)
    {
        if (!dimensions.empty())
        {
            throw std::invalid_argument("\"-dimensions\" may only be specified for \"-tensor\" conversion.");
        }
        if (dataType != onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED)
        {
            throw std::invalid_argument("\"-datatype\" may only be specified for \"-tensor\" conversion.");
        }

        ConvertModel(inputFilename.c_str(), outputFilename.c_str());
    }
    else // conversionMode == ConversionMode::Unknown
    {
        throw std::invalid_argument("Conversion mode cannot be determined from the filename extensions alone. Specify -graph or -tensor.");
    }

    return EXIT_SUCCESS;
}

int wmain(int argc, wchar_t** argv)
{
    try
    {
        return Main(argc, argv);
    }
    catch (std::exception const& e)
    {
        std::cout << e.what();
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cout << "Unknown error.";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
