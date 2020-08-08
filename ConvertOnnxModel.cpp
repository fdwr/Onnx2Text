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
#include <codecvt>
#include <charconv>
#include <random>

#pragma warning(push)
#pragma warning(disable: 4146)
#include "onnx.pb.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#pragma warning(pop)

#include "half/half.hpp"

#include <d3d12.h>
#include <wrl/client.h>
#include <wincodec.h>

using float16 = half_float::half;

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
    span<T> subspan(size_t index, size_t count) const noexcept { return span<T>(begin_ + index, begin_ + index + count); }
    span<T> subrange(size_t begin, size_t end) const noexcept { return span<T>(begin_ + begin, begin_ + end); }
    span<T> first(size_t count) const noexcept { return span<T>(begin_, begin_ + count); }
    span<T> last(size_t count) const noexcept { return span<T>(end_ - count, end_); }

    T& front() noexcept { return *begin_; }
    T& back()  noexcept { return *end_; }
    T const& front() const noexcept { return *begin_; }
    T const& back()  const noexcept { return *end_; }
    T consume_front() noexcept { return *begin_++; }
    T consume_back()  noexcept { return *--end_; }
    void pop_front() noexcept { ++begin_; }
    void pop_back()  noexcept { --end_; }
    void pop_front(size_t n) noexcept { begin_ += n; }
    void pop_back(size_t n)  noexcept { end_ -= n; }

protected:
    T* begin_ = nullptr;
    T* end_ = nullptr;
};

template<typename T>
span<const std::byte> as_bytes(span<T> oldSpan)
{
    return span<const std::byte>(reinterpret_cast<const std::byte*>(oldSpan.data()), oldSpan.size_bytes());
}

template<typename T>
span<std::byte> as_writeable_bytes(span<T> oldSpan)
{
    return span<std::byte>(reinterpret_cast<std::byte*>(oldSpan.data()), oldSpan.size_bytes());
}

template<typename T>
span<const std::byte> struct_as_bytes(T& data)
{
    return span<const std::byte>(reinterpret_cast<const std::byte*>(std::addressof(data)), sizeof(data));
}

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

template <typename ContainerType>
auto clamped_span(ContainerType& oldSpan, size_t offset, size_t count) -> span<decltype(*oldSpan.data())>
{
    size_t endOffset = offset + count;
    size_t maxOffset = oldSpan.size();
    beginOffset = std::min(offset, maxOffset);
    endOffset = (endOffset > maxOffset || endOffset < offset) ? maxOffset : endOffset;

    auto* p = oldSpan.data();
    return span<NewType>(p + beginOffset, p + endOffset);
}

// Reads a byte array from std::vector/std::string/std::array as a struct.
template <typename NewStructType, typename OldTypeContainer>
const NewStructType& read_as(OldTypeContainer&& oldSpan)
{
    span<const std::byte> byteSpan = as_bytes(oldSpan);
    size_t byteSize = byteSpan.size_bytes();
    if (sizeof(NewStructType) > byteSize)
    {
        throw std::runtime_error("Span is too small to be cast to new data type.");
    }
    return *reinterpret_cast<const NewStructType*>(byteSpan.data());
}

// Why? How? How are basic functions like this missing from the standard library?
template <typename ContainerType1, typename ContainerType2>
bool equals(ContainerType1&& c1, ContainerType2&& c2)
{
    return std::equal(c1.begin(), c1.end(), c2.begin(), c2.end());
}

// e.g. starts_with(some_string_view, "prefix");
template <typename ContainerType1, typename ContainerType2>
bool starts_with(ContainerType1&& fullSequence, ContainerType2&& prefix)
{
    return fullSequence.size() >= prefix.size()
        && std::equal(fullSequence.begin(), fullSequence.begin() + prefix.size(), prefix.begin(), prefix.end());
}

// e.g. starts_with(some_string_view, "prefix");
template <typename ContainerType1, typename ContainerType2>
bool ends_with(ContainerType1&& fullSequence, ContainerType2&& suffix)
{
    return fullSequence.size() >= suffix.size()
        && std::equal(fullSequence.end() -  + suffix.size(), fullSequence.end(), suffix.begin(), suffix.end());
}

template <typename ContainerType>
auto find(ContainerType&& container, decltype(*container.data())&& value) -> decltype(container.begin())
{
    return std::find(container.begin(), container.end(), value);
}

template <typename ContainerType>
auto tokenize(
    ContainerType&& container,
    std::remove_reference_t<decltype(*container.data())> divider
    ) -> std::vector<span< std::remove_reference_t<decltype(*container.data())> >>
{
    using ElementType = std::remove_reference_t<decltype(*container.data())>;

    std::vector<span<ElementType>> v;

    for (auto i = container.begin(), end = container.end(); i != end; )
    {
        auto next = std::find(i, container.end(), divider);
        v.push_back(span<ElementType>(&*i, &*next));
        if (next != end)
        {
            ++next;
        }
        i = next;
    }

    return v;
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
    std::filesystem::path path(outputFilename);
    if (path.has_parent_path())
    {
        path.remove_filename();
        // .filename() lies when referring to the root directory, saying there is a filename when
        // there actually is not. So instead we check whether the current path equals the root.
        std::filesystem::path root = path.root_path();
        if (path != root)
        {
            std::filesystem::create_directory(path);
        }
    }

    std::ofstream file(outputFilename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::ios::failure("Could not open output file.");
    }

    file.write(fileData.data(), fileData.size());
}

enum class FileType
{
    Unknown,             // .onnx
    OnnxModel,           // .onnx
    GoogleProtobuf,      // .pb
    Text,                // .txt
    CommaSeparatedValue, // .csv
    Image,               // .png / .jpg
    RawData,             // .dat / .bin - raw binary, dump of tensor values as-is
    NumPyArray,          // .npy (not .npz zip files with multiple arrays in them)
    OnnxTensor,          // .onnxtensor
    TensorGenerator,     // generator:
};

size_t GetFileExtensionOffset(std::wstring_view filename)
{
    size_t extensionOffset = filename.find_last_of(L".");
    extensionOffset = (extensionOffset != std::string::npos) ? extensionOffset + 1 : filename.size();
    return extensionOffset;
}

FileType GetFileType(std::wstring_view filename)
{
    size_t extensionOffset = GetFileExtensionOffset(filename);
    std::wstring_view filenameExtension = filename.substr(extensionOffset);
    if (starts_with(filename, std::wstring_view(L"generate("))) return FileType::TensorGenerator;
    if (filenameExtension == L"pb"  ) return FileType::GoogleProtobuf;
    if (filenameExtension == L"onnx") return FileType::OnnxModel;
    if (filenameExtension == L"txt" ) return FileType::Text;
    if (filenameExtension == L"csv" ) return FileType::CommaSeparatedValue;
    if (filenameExtension == L"dat" ) return FileType::RawData;
    if (filenameExtension == L"bin" ) return FileType::RawData;
    if (filenameExtension == L"png" ) return FileType::Image;
    if (filenameExtension == L"jpg" ) return FileType::Image;
    if (filenameExtension == L"jpeg") return FileType::Image;
    if (filenameExtension == L"npy")  return FileType::NumPyArray;
    if (filenameExtension == L"onnxtensor") return FileType::OnnxTensor;
    return FileType::Unknown;
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

union ScalarValueUnion
{
    double floatValue;
    int64_t intValue;
    uint64_t uintValue;
};

enum class CsvValueNumberClass
{
    Int,
    Uint,
    Float,
    Hex,
};

CsvValueNumberClass GetCsvValueNumberClass(
    onnx::TensorProto::DataType dataType,
    bool shouldPrintRawBytes
    )
{
    CsvValueNumberClass valueNumberClass = CsvValueNumberClass::Uint;
    switch (dataType)
    {
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:    valueNumberClass = CsvValueNumberClass::Float; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      valueNumberClass = CsvValueNumberClass::Float; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     valueNumberClass = CsvValueNumberClass::Float; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  valueNumberClass = CsvValueNumberClass::Float; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: valueNumberClass = CsvValueNumberClass::Float; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       valueNumberClass = CsvValueNumberClass::Uint ; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      valueNumberClass = CsvValueNumberClass::Uint ; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     valueNumberClass = CsvValueNumberClass::Uint ; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     valueNumberClass = CsvValueNumberClass::Uint ; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     valueNumberClass = CsvValueNumberClass::Uint ; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       valueNumberClass = CsvValueNumberClass::Int  ; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      valueNumberClass = CsvValueNumberClass::Int  ; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      valueNumberClass = CsvValueNumberClass::Int  ; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      valueNumberClass = CsvValueNumberClass::Int  ; break;
    default: throw std::ios::failure("Unsupported data type in tensor for CSV output.");
    }

    // For printing raw hex, always print as hex digits regardless of actual data type.
    if (shouldPrintRawBytes)
    {
        valueNumberClass = CsvValueNumberClass::Hex;
    }

    return valueNumberClass;
}

void WriteTensorValue(
    void* data, // Must point to memory that has at least the number of bytes specified by the dataType.
    onnx::TensorProto::DataType dataType,
    ScalarValueUnion value
    )
{
    switch (dataType)
    {
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:    *reinterpret_cast<float16*> (data) = static_cast<float>   (value.floatValue); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      *reinterpret_cast<float*>   (data) = static_cast<float>   (value.floatValue); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     *reinterpret_cast<double*>  (data) = static_cast<double>  (value.floatValue); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  *reinterpret_cast<float*>   (data) = static_cast<float>   (value.floatValue); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: *reinterpret_cast<double*>  (data) = static_cast<double>  (value.floatValue); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       *reinterpret_cast<bool*>    (data) = static_cast<bool>    (value.uintValue ); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      *reinterpret_cast<uint8_t*> (data) = static_cast<uint8_t> (value.uintValue ); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     *reinterpret_cast<uint16_t*>(data) = static_cast<uint16_t>(value.uintValue ); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     *reinterpret_cast<uint32_t*>(data) = static_cast<uint32_t>(value.uintValue ); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     *reinterpret_cast<uint64_t*>(data) = static_cast<uint64_t>(value.uintValue ); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       *reinterpret_cast<int8_t*>  (data) = static_cast<int8_t>  (value.intValue  ); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      *reinterpret_cast<int16_t*> (data) = static_cast<int16_t> (value.intValue  ); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      *reinterpret_cast<int32_t*> (data) = static_cast<int32_t> (value.intValue  ); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      *reinterpret_cast<int64_t*> (data) = static_cast<int64_t> (value.intValue  ); break;
    default: throw std::ios::failure("Unsupported data type for tensor.");
    }
}

void WriteTensorValues(
    /*out*/ span<uint8_t> arrayByteData,
    onnx::TensorProto::DataType dataType,
    ScalarValueUnion value
    )
{
    const size_t elementByteSize = GetDataTypeElementByteSize(dataType);
    if (elementByteSize == 0)
    {
        return;
    }

    size_t alignedByteCount = arrayByteData.size_bytes() / elementByteSize * elementByteSize;
    for (size_t i = 0; i < alignedByteCount; i += elementByteSize)
    {
        WriteTensorValue(/*out*/ &arrayByteData[i], dataType, value);
    }
}

void WriteTensorValues(
    /*out*/ span<uint8_t> arrayByteData,
    onnx::TensorProto::DataType dataType,
    std::function<ScalarValueUnion()> valueGetter
    )
{
    const size_t elementByteSize = GetDataTypeElementByteSize(dataType);
    if (elementByteSize == 0)
    {
        return;
    }

    size_t alignedByteCount = arrayByteData.size_bytes() / elementByteSize * elementByteSize;
    for (size_t i = 0; i < alignedByteCount; i += elementByteSize)
    {
        ScalarValueUnion value = valueGetter();
        WriteTensorValue(/*out*/ &arrayByteData[i], dataType, value);
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

    CsvValueNumberClass valueNumberClass = GetCsvValueNumberClass(dataType, /*shouldPrintRawBytes*/ false);

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
            ScalarValueUnion value = {};

            // Read the numeric value.
            bool shouldReadRawBytes = false;
            if (numberStart[0] == '0' && numberStart[1] == 'x')
            {
                shouldReadRawBytes = true;
                std::from_chars(numberStart + 2, end, /*out*/ value.uintValue);
            }
            else
            {
                switch (valueNumberClass)
                {
                case CsvValueNumberClass::Int:   std::from_chars(numberStart, end, /*out*/ value.intValue);      break;
                case CsvValueNumberClass::Uint:  std::from_chars(numberStart, end, /*out*/ value.uintValue);     break;
                case CsvValueNumberClass::Float: std::from_chars(numberStart, end, /*out*/ value.floatValue);    break;
                case CsvValueNumberClass::Hex:   std::from_chars(numberStart, end, /*out*/ value.uintValue, 16); break;
                }
            }

            byteData.resize(byteDataSize + elementByteSize);
            void* data = &byteData[byteDataSize];
            byteDataSize += elementByteSize;

            if (shouldReadRawBytes)
            {
                switch (dataType)
                {
                case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:
                case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:
                case onnx::TensorProto::DataType::TensorProto_DataType_INT8:
                    *reinterpret_cast<uint8_t*> (data) = static_cast<uint8_t>(value.uintValue); break;
                case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:
                case onnx::TensorProto::DataType::TensorProto_DataType_INT16:
                case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:
                    *reinterpret_cast<uint16_t*>(data) = static_cast<uint16_t>(value.uintValue); break;
                case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:
                case onnx::TensorProto::DataType::TensorProto_DataType_INT32:
                case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:
                case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:
                    *reinterpret_cast<uint32_t*>(data) = static_cast<uint32_t>(value.uintValue); break;
                case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:
                case onnx::TensorProto::DataType::TensorProto_DataType_INT64:
                case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:
                case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128:
                    *reinterpret_cast<uint64_t*>(data) = static_cast<uint64_t>(value.uintValue); break;
                default: throw std::ios::failure("Unsupported data type in tensor for CSV input.");
                }
            }
            else
            {
                WriteTensorValue(/*out*/ data, dataType, value);
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

// Writes tensor data to a string (not directly to a file).
void WriteCsv(
    /*out*/span<char const> byteData,
    onnx::TensorProto::DataType dataType,
    bool shouldPrintRawBytes, // Print raw hex bit values instead of formatted numbers.
    /*out*/std::string& text
    )
{
    text.clear();

    size_t elementByteSize = GetDataTypeElementByteSize(dataType);

    char buffer[40];

    // Round off any potential padding.
    byteData = span<char const>(byteData.data(), (byteData.size() / elementByteSize) * elementByteSize);

    CsvValueNumberClass valueNumberClass = GetCsvValueNumberClass(dataType, shouldPrintRawBytes);

    char const* begin = byteData.data();
    char const* end = byteData.data() + byteData.size();

    ScalarValueUnion value;

    while (begin != end)
    {
        // Read the next value from the type buffer.
        value = {};
        void const* data = begin;
        begin += elementByteSize;

        if (shouldPrintRawBytes)
        {
            switch (dataType)
            {
            case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:
            case onnx::TensorProto::DataType::TensorProto_DataType_INT8:
                value.uintValue = *reinterpret_cast<const uint8_t*> (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:
            case onnx::TensorProto::DataType::TensorProto_DataType_INT16:
            case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:
                value.uintValue = *reinterpret_cast<const uint16_t*>(data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:
            case onnx::TensorProto::DataType::TensorProto_DataType_INT32:
            case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:
            case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:
                value.uintValue = *reinterpret_cast<const uint32_t*>(data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:
            case onnx::TensorProto::DataType::TensorProto_DataType_INT64:
            case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:
            case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128:
                value.uintValue = *reinterpret_cast<const uint64_t*>(data); break;
            default: throw std::ios::failure("Unsupported data type in tensor for CSV output.");
            }
        }
        else
        {
            switch (dataType)
            {
            case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      value.floatValue = *reinterpret_cast<const float*>   (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     value.floatValue = *reinterpret_cast<const double*>  (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       value.uintValue  = *reinterpret_cast<const bool*>    (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      value.uintValue  = *reinterpret_cast<const uint8_t*> (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       value.intValue   = *reinterpret_cast<const int8_t*>  (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     value.uintValue  = *reinterpret_cast<const uint16_t*>(data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      value.intValue   = *reinterpret_cast<const int16_t*> (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     value.uintValue  = *reinterpret_cast<const uint32_t*>(data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      value.intValue   = *reinterpret_cast<const int32_t*> (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     value.uintValue  = *reinterpret_cast<const uint64_t*>(data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      value.intValue   = *reinterpret_cast<const int64_t*> (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  value.floatValue = *reinterpret_cast<const float*>   (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: value.floatValue = *reinterpret_cast<const double*>  (data); break;
            case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:    value.floatValue = *reinterpret_cast<const float16*> (data); break;
            default: throw std::ios::failure("Unsupported data type in tensor for CSV output.");
            }
        }

        switch (valueNumberClass)
        {
        case CsvValueNumberClass::Int:   sprintf_s(buffer, "%lld",   value.intValue);   break;
        case CsvValueNumberClass::Uint:  sprintf_s(buffer, "%llu",   value.uintValue);  break;
        case CsvValueNumberClass::Float: sprintf_s(buffer, "%g",     value.floatValue); break;
        case CsvValueNumberClass::Hex:   sprintf_s(buffer, "0x%llX", value.uintValue);  break;
        }

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
    // Return a 1D array if no dimensions were given, equal to the element count.
    std::vector<int32_t> resolvedDimensions(defaultDimensions.begin(), defaultDimensions.end());

    if (resolvedDimensions.empty())
    {
        size_t elementByteSize = GetDataTypeElementByteSize(dataType);
        size_t elementCount = byteData.size() / elementByteSize;
        resolvedDimensions.push_back(static_cast<int32_t>(elementCount));
    }

    return resolvedDimensions;
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

constexpr bool g_isFloatingPointElementDataType[] =
{
    false, // Undefined = 0,
    true , // Float32 = 1,
    false, // Uint8 = 2,
    false, // Int8 = 3,
    false, // Uint16 = 4,
    false, // Int16 = 5,
    false, // Int32 = 6,
    false, // Int64 = 7,
    false, // StringChar8 = 8,
    false, // Bool = 9,
    true , // Float16 = 10,
    true , // Float64 = 11,
    false, // Uint32 = 12,
    false, // Uint64 = 13,
    true , // Complex64 = 14,
    true , // Complex128 = 15,
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

uint32_t GetByteSizeFromDimensions(span<int32_t const> dimensions, onnx::TensorProto::DataType dataType) noexcept
{
    return ComputeElementCount(dimensions) * GetByteSizeFromDataType(dataType);
}

onnx::TensorProto::DataType GetDataTypeFromStringName(std::string_view name) noexcept
{
    auto i = std::find(std::begin(g_elementDataTypeNames), std::end(g_elementDataTypeNames), name);
    return (i == std::end(g_elementDataTypeNames))
        ? onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED
        : onnx::TensorProto::DataType(i - std::begin(g_elementDataTypeNames));
}

bool IsFloatingPointDataType(onnx::TensorProto::DataType dataType) noexcept
{
    size_t index = static_cast<size_t>(dataType);
    return g_isFloatingPointElementDataType[index < std::size(g_elementDataTypeNames) ? index : 0];
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

void SwapBytes(/*inout*/ span<uint8_t> arrayByteData, uint32_t elementByteSize)
{
    switch (elementByteSize)
    {
    // case 1: NOP

    case 2:
        {
            auto s16 = reinterpret_span<uint16_t>(arrayByteData);
            for (auto& u : s16)
            {
                uint32_t v = u;
                u = ((v & 0x00FF) << 8) |
                    ((v & 0xFF00) >> 8);
            }
        }
        break;

    case 4:
    case 8:
    case 16:
        {
            auto s32 = reinterpret_span<uint32_t>(arrayByteData);
            for (auto& u : s32)
            {
                uint32_t v = u;
                u = ((v & 0x000000FF) << 24) |
                    ((v & 0x0000FF00) << 8);
                    ((v & 0x00FF0000) >> 8);
                    ((v & 0xFF000000) >> 24);
            }

            if (elementByteSize == 8)
            {
                for (uint32_t i = 0; i < static_cast<uint32_t>(s32.size() & ~0x1); i += 2)
                {
                    std::swap(s32[i + 0], s32[i + 1]);
                }
            }
            else if (elementByteSize == 16)
            {
                for (uint32_t i = 0; i < static_cast<uint32_t>(s32.size() & ~0x3); i += 4)
                {
                    std::swap(s32[i + 0], s32[i + 3]);
                    std::swap(s32[i + 1], s32[i + 2]);
                }
            }
        }
        break;
    }
}

void MapNumPyArrayDataTypeToOnnx(
    std::string_view numPyElementType,
    /*out*/onnx::TensorProto::DataType& dataType,
    /*out*/ bool& isBackwardsEndian // Backwards endian which stores greatest bytes at lowest bytes.
    )
{
    dataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
    isBackwardsEndian = false;

    onnx::TensorProto::DataType resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
    uint32_t elementByteSize = 0;

    // https://docs.python.org/2/library/array.html#module-array
    // https://numpy.org/devdocs/reference/arrays.dtypes.html
    for (char c : numPyElementType)
    {
        switch (c)
        {
        case '?': resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_BOOL;   break; // boolean
        case 'b': resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_INT8;   break; // signed byte
        case 'B': resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UINT8;  break; // unsigned byte
        case 'h': resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_INT16;  break; // signed short
        case 'H': resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UINT16; break; // unsigned short
        case 'i': resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_INT32;  break; // signed integer
        case 'u': resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UINT32; break; // unsigned integer
        case 'f': resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_FLOAT;  break; // float
        case 'd': resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE; break; // float64
        case '>': isBackwardsEndian = true; break;    // (backwards-endian)
        case '<': isBackwardsEndian = false; break;   // (logical-endian)

        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            elementByteSize = elementByteSize * 10 + c - '0';
            break;

        default:
            assert(false);
            resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
            break;
        }
    }

    if (elementByteSize > 0)
    {
        switch (resolvedDataType)
        {
        case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:
            switch (elementByteSize)
            {
            case 1: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UINT8; break;
            case 2: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UINT16; break;
            case 4: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UINT32; break;
            case 8: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UINT64; break;
            default: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED; break;
            }
            break;

        case onnx::TensorProto::DataType::TensorProto_DataType_INT8:
        case onnx::TensorProto::DataType::TensorProto_DataType_INT16:
        case onnx::TensorProto::DataType::TensorProto_DataType_INT32:
            switch (elementByteSize)
            {
            case 1: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_INT8; break;
            case 2: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_INT16; break;
            case 4: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_INT32; break;
            case 8: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_INT64; break;
            default: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED; break;
            }
            break;

        case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:
        case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:
            switch (elementByteSize)
            {
            case 2: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16; break;
            case 4: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_FLOAT; break;
            case 8: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE; break;
            default: resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED; break;
            }
            break;
        }
    }

    dataType = resolvedDataType;
}

void MapOnnxDataTypeToNumPyArray(
    onnx::TensorProto::DataType dataType,
    bool isBackwardsEndian, // Backwards endian which stores greatest bytes at lowest bytes.
    /*out*/ std::string& numPyElementType
    )
{
    numPyElementType.push_back(isBackwardsEndian ? '>' : '<');

    // https://docs.python.org/2/library/array.html#module-array
    // https://numpy.org/devdocs/reference/arrays.dtypes.html
    std::string_view characterCode;
    switch (dataType)
    {
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:    characterCode = "?"   /*'?'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:    characterCode = "i1"  /*'b'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:   characterCode = "u1"  /*'B'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:   characterCode = "i2" /*'h'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:   characterCode = "i4" /*'i'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:   characterCode = "i8" /*'i'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:  characterCode = "u2" /*'H'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:  characterCode = "u4" /*'u'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:  characterCode = "u8" /*'u'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16: characterCode = "f2" /*'f'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:   characterCode = "f4" /*'f'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:  characterCode = "f8" /*'d'*/; break;
    default: characterCode = "?";  assert(false);
    }
    numPyElementType.append(characterCode);
}

class PythonDictionaryLexer
{
public:
    enum class TokenType
    {
        Error,
        End,
        OpeningBrace,
        ClosingBrace,
        OpeningParenthesis,
        ClosingParenthesis,
        Identifier,
        String,
        Colon,
        Comma,
        Number,
    };

    PythonDictionaryLexer(span<const char> text) : text_(text)
    {
    }

    PythonDictionaryLexer(std::string_view text) : text_(text.data(), text.data() + text.size())
    {
    }

    bool empty()
    {
        return text_.empty();
    }

    struct { span<const char> token; TokenType tokenType; } Read()
    {
        span<const char> token;
        TokenType tokenType = TokenType::End;

        // Skip spaces.
        for (; !text_.empty() && isspace(text_.front()); text_.pop_front())
        {
        }

        if (!text_.empty())
        {
            token = text_.subspan(0, 1);
            char ch = text_.consume_front();

            switch (ch)
            {
            case '{': tokenType = TokenType::OpeningBrace; break;
            case '}': tokenType = TokenType::ClosingBrace; break;
            case '(': tokenType = TokenType::OpeningParenthesis; break;
            case ')': tokenType = TokenType::ClosingParenthesis; break;
            case ':': tokenType = TokenType::Colon; break;
            case ',': tokenType = TokenType::Comma; break;
            case '\'':
            case '\"':
                {
                    tokenType = TokenType::String;
                    char leadingQuoteMark = ch;
                    ch = 0;
                    token.pop_front(); // Skip leading quote.

                    // Read until the closing quote mark.
                    for (; !text_.empty() && (ch = text_.front(), ch != leadingQuoteMark && ch != '\r' && ch != '\n'); text_.pop_front())
                    {
                    }
                    token = {token.begin(), text_.begin()};

                    if (ch == leadingQuoteMark)
                    {
                        text_.pop_front(); // Skip closing quote mark.
                    }
                    else
                    {
                        tokenType = TokenType::Error; // Unclosed string.
                    }
                }
                break;

            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
                tokenType = TokenType::Number;
                for (; !text_.empty() && (ch = text_.front(), isdigit(ch) || ch == '.'); text_.pop_front())
                {
                }
                token = {token.begin(), text_.begin()};
                break;

            default:
                // Check alphanumeric identifier.
                if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))
                {
                    tokenType = TokenType::Identifier;
                    for (; !text_.empty() && (ch = text_.front(), isalnum(ch) || ch == '.'); text_.pop_front())
                    {
                    }
                    token = {token.begin(), text_.begin()};
                }
                else
                {
                    tokenType = TokenType::Error;
                }
            }
        }

        return { token, tokenType };
    }

    std::map<std::string_view, std::string_view> ReadDictionary()
    {
        int indentLevel = 0;

        std::map<std::string_view, std::string_view> map;
        std::string_view currentKey;
        std::string_view currentValue;
        bool haveKey = false;

        auto appendCurrentKeyValue = [&]()
        {
            if (haveKey)
            {
                map.insert({currentKey, currentValue});
                currentKey = {};
                currentValue = {};
            }
            haveKey = false;
        };

        while (true)
        {
            auto [token, tokenType] = Read();

            bool extendToken = false;
            switch (tokenType)
            {
            case TokenType::Error: goto End;
            case TokenType::End: goto End;
            case TokenType::OpeningParenthesis: ++indentLevel; break;
            case TokenType::ClosingParenthesis: --indentLevel; extendToken = (indentLevel == 1); break;
            case TokenType::OpeningBrace: ++indentLevel; break;
            case TokenType::ClosingBrace: --indentLevel; extendToken = (indentLevel == 1); break;

            case TokenType::Comma:
                if (indentLevel == 1)
                {
                    appendCurrentKeyValue();
                }
                break;

            case TokenType::Colon:
                if (indentLevel == 1)
                {
                    haveKey = true;
                }
                break;

            default:
                extendToken = true;
                break;
            }

            // Glom multiple tokens together into a larger unit.
            if (indentLevel > 1 || extendToken)
            {
                auto& keyOrValue = (haveKey) ? currentValue : currentKey;
                if (keyOrValue.empty())
                {
                    keyOrValue = { token.data(), token.size() };
                }
                else
                {
                    keyOrValue = {keyOrValue.data(), size_t(token.end() - keyOrValue.data()) };
                }
            }
        }
    End:

        appendCurrentKeyValue();

        assert(indentLevel == 0);

        return map;
    }

    void ParseIntegers(/*out*/ std::vector<int32_t>& numbers)
    {
        while (true)
        {
            auto [token, tokenType] = Read();

            switch (tokenType)
            {
            case TokenType::End:
            case TokenType::Error:
                goto End;

            case TokenType::Number:
                {
                    //char const* endptr = token.end();
                    uint32_t value = 0;
                    std::from_chars(token.begin(), token.end(), /*out*/ value);
                    numbers.push_back(value);
                }
                break;
            }
        }
    End:;
    }

private:
    span<const char> text_;
};

class PythonDictionaryWriter
{
public:
    std::string_view GetText() const
    {
        return text_;
    }

    void Append(std::string_view text)
    {
        text_.append(text);
    }

    void WriteKeyValueUnquoted(std::string_view key, std::string_view value)
    {
        text_.append(key);
        text_.append(":");
        text_.append(value);
        text_.append(", ");
    }

    void WriteKeyValue(std::string_view key, std::string_view value)
    {
        text_.push_back('\'');
        text_.append(key);
        text_.append("\':\'");
        text_.append(value);
        text_.append("\', ");
    }

    void WriteIntegers(span<const int32_t> numbers, std::string_view brackets)
    {
        if (!brackets.empty())
        {
            text_.push_back(brackets.front());
        }
        for (auto n : numbers)
        {
            char buffer[11];
            auto result = std::to_chars(std::begin(buffer), std::end(buffer), n);
            text_.append(std::begin(buffer), result.ptr);
            text_.append(",");
        }
        if (!brackets.empty())
        {
            text_.push_back(brackets.back());
        }
    }

private:
    std::string text_;
};

struct NumPyArrayHeaderV1
{
    uint8_t signature[6]; // "\x0093NUMPY"
    uint8_t majorVersion;
    uint8_t minorVersion;
    uint16_t dictionaryLength; // Confusingly instead labeled "HEADER_LEN" in the documentation.
};

struct NumPyArrayHeaderV2
{
    uint8_t signature[6]; // "\x0093NUMPY"
    uint8_t majorVersion;
    uint8_t minorVersion;
    uint32_t dictionaryLength; // Confusingly instead labeled "HEADER_LEN" in the documentation.
};

void ReadNpy(
    span<const char> fileData,
    /*out*/onnx::TensorProto::DataType& dataType,
    /*out*/std::vector<int32_t>& dimensions,
    /*out*/std::vector<char>& arrayByteData
    )
{
    dataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
    dimensions.clear();
    arrayByteData.clear();

    using namespace std::literals;

    if (fileData.size_bytes() < sizeof(NumPyArrayHeaderV1))
    {
        throw std::ios::failure("NumPy array header signature is invalid.");
    }

    auto& headerV1 = read_as<NumPyArrayHeaderV1>(fileData);
    auto& headerV2 = read_as<NumPyArrayHeaderV2>(fileData);
    if (headerV1.majorVersion >= 3)
    {
        throw std::ios::failure("Versions > 4 unsupported.");
    }

    size_t dictionaryOffset = (headerV1.majorVersion >= 2) ? sizeof(NumPyArrayHeaderV2) : sizeof(NumPyArrayHeaderV1);
    size_t dictionaryLength = (headerV1.majorVersion >= 2) ? headerV2.dictionaryLength : headerV1.dictionaryLength;
    size_t dataByteOffset = dictionaryOffset + dictionaryLength;

    PythonDictionaryLexer lexer(fileData.subrange(dictionaryOffset, fileData.size_bytes()));
    std::map<std::string_view, std::string_view> dictionary = lexer.ReadDictionary();

    bool isBackwardsEndian = false;
    bool hasIncreasingStrides = false;

    for (auto& i : dictionary)
    {
        if (i.first == "descr"sv)
        {
            MapNumPyArrayDataTypeToOnnx(i.second, /*out*/ dataType, /*out*/ isBackwardsEndian);
        }
        else if (i.first == "fortran_order"sv)
        {
            hasIncreasingStrides = (i.second == "True"sv);
        }
        else if (i.first == "shape"sv)
        {
            PythonDictionaryLexer shapeLexer(i.second);
            shapeLexer.ParseIntegers(dimensions);
        }
    }

    arrayByteData.assign(fileData.data() + dataByteOffset, fileData.end());
    const uint32_t elementByteSize = GetByteSizeFromDataType(dataType);
    const uint32_t totalElementCount = ComputeElementCount(dimensions);
    const uint32_t totalByteSize = elementByteSize * totalElementCount;
    if (arrayByteData.size() < totalByteSize)
    {
        arrayByteData.resize(totalByteSize);
    }

    // Assuming that we're running on a logical endian machine.
    // If not, lots of other places would break too anyway.
    if (isBackwardsEndian)
    {
        SwapBytes(/*inout*/ reinterpret_span<uint8_t>(arrayByteData), totalByteSize);
    }
    if (hasIncreasingStrides)
    {
        // TODO - augment RearrangeChannels to support reversing stride.
        throw std::ios::failure("Fortran stride order unsupported.");
    }
}

// Writes tensor data to in memory file data (not directly to file).
void WriteNpy(
    /*out*/span<char const> arrayByteData,
    onnx::TensorProto::DataType dataType,
    /*out*/span<int32_t> dimensions,
    /*out*/std::string& fileData
    )
{
    NumPyArrayHeaderV1 headerFixedPart = { {uint8_t('\x0093'),'N','U','M','P','Y'}, 1,0, 0 };

    PythonDictionaryWriter dictionaryWriter;
    PythonDictionaryWriter numberWriter;

    // Format dictionary fields.
    std::string numPyElementType;
    MapOnnxDataTypeToNumPyArray(dataType, /*isBackwardsEndian*/ false, /*out*/ numPyElementType);
    numberWriter.WriteIntegers(dimensions, "()");

    dictionaryWriter.Append("{");
    dictionaryWriter.WriteKeyValue("descr", numPyElementType);
    dictionaryWriter.WriteKeyValueUnquoted("'fortran_order'", "False");
    dictionaryWriter.WriteKeyValueUnquoted("'shape'", numberWriter.GetText());
    dictionaryWriter.Append("}");

    // Compute header length for alignment.
    uint32_t headerLength = sizeof(headerFixedPart);
    headerLength += static_cast<uint32_t>(dictionaryWriter.GetText().size());
    headerLength++; // For new line.
    headerLength = (headerLength + 63) & ~63; // For rounding up to multiple of 64 alignment.

    // Write header, including fixed size part, dictionary, and alignment padding.
    headerFixedPart.dictionaryLength = static_cast<uint16_t>(headerLength - sizeof(headerFixedPart));
    fileData.append(reinterpret_cast<const char*>(&headerFixedPart), sizeof(headerFixedPart));
    fileData.append(dictionaryWriter.GetText());
    fileData.append(headerLength - fileData.size(), ' ');
    fileData.back() = '\x000A'; // Terminate with new line.
    // Note the spec says "It is terminated by a newline (\n) and padded with spaces (\x20)",
    // but that's wrong. It's actually "padding with spaces and then terminated by a newline".
    // Otherwise Numpy 1.18.5 barfs (1.19 works fine either way).
    // https://numpy.org/devdocs/reference/generated/numpy.lib.format.html

    fileData.append(arrayByteData.begin(), arrayByteData.end());
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

void PrintTensorInfo(
    std::string_view name,
    std::wstring_view fileName,
    span<const int32_t> dimensions,
    onnx::TensorProto::DataType dataType
    )
{
    std::string dimensionsText;
    WriteCsv(
        reinterpret_span<char const>(dimensions),
        onnx::TensorProto::DataType::TensorProto_DataType_INT32,
        /*shouldPrintRawBytes*/ false,
        /*out*/ dimensionsText
    );

    const uint32_t byteSize = GetByteSizeFromDimensions(dimensions, dataType);
    if (!fileName.empty())
    {
        printf(
            "Tensor \"%.*S\", datatype: %s, dimensions: %s (%d bytes)\n",
            uint32_t(fileName.size()),
            fileName.data(),
            GetStringNameFromDataType(dataType).data(),
            dimensionsText.c_str(),
            byteSize
        );
    }
    else
    {
        printf(
            "Tensor \"%.*s\", datatype: %s, dimensions: %s, (%d bytes)\n",
            uint32_t(name.size()),
            name.data(),
            GetStringNameFromDataType(dataType).data(),
            dimensionsText.c_str(),
            byteSize
        );
    }
}

void ConvertModel(
    _In_z_ wchar_t const* inputFilename,
    _In_z_ wchar_t const* outputFilename
    )
{
    FileType inputFileType  = GetFileType(std::wstring_view(inputFilename));
    FileType outputFileType = GetFileType(std::wstring_view(outputFilename));

    onnx::ModelProto model;

    bool succeeded = false;
    if (inputFileType == FileType::Text)
    {
        std::string modelString = ReadTextFile(inputFilename);

        // Essentially "google::protobuf::TextFormat::ParseFromString(modelString, &model)"
        // except that we need to pass the flag to allow field numbers.

        google::protobuf::TextFormat::Parser parser;
        parser.AllowFieldNumber(true);
        succeeded = parser.ParseFromString(modelString, &model);
    }
    else if (inputFileType == FileType::OnnxModel
          || inputFileType == FileType::GoogleProtobuf)
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

    if (outputFileType == FileType::Text)
    {
        // Write the whole model to a text file.
        // Use the stream instead of google::protobuf::TextFormat::PrintToString,
        // which can fail for models that are >= 200MBs by running out of memory.
        std::ofstream outputFile(outputFilename, std::ios::out);
        std::unique_ptr<google::protobuf::io::ZeroCopyOutputStream> output(new google::protobuf::io::OstreamOutputStream(&outputFile));
        succeeded = google::protobuf::TextFormat::Print(model, output.get());
    }
    else if (outputFileType == FileType::OnnxModel
          || outputFileType == FileType::GoogleProtobuf)
    {
        std::ofstream os(outputFilename, std::ios::binary);
        succeeded = model.SerializeToOstream(&os);
    }
    else if (outputFileType == FileType::NumPyArray
          || outputFileType == FileType::OnnxTensor
          || outputFileType == FileType::RawData)
    {
        // enumerate all the tensor initializers, and dump their contents.

        std::wstring initialFileName(outputFilename);
        std::wstring currentFileName(outputFilename);
        size_t substitutionOffset = initialFileName.find('*', 0);
        if (substitutionOffset != std::string::npos)
        {
            initialFileName.erase(substitutionOffset, 1);
        }
        else
        {
            substitutionOffset = GetFileExtensionOffset(initialFileName);
            if (substitutionOffset > 0 && initialFileName[substitutionOffset - 1] == '.')
            {
                --substitutionOffset;
            }
        }

        onnx::GraphProto const& graphProto = model.graph();
        for (const onnx::TensorProto& onnxTensor : graphProto.initializer())
        {
            // Read data type, dimensions, and name.
            onnx::TensorProto::DataType dataType = onnxTensor.data_type();

            std::vector<int32_t> dimensions;
            for (auto v : onnxTensor.dims())
            {
                dimensions.push_back(static_cast<int32_t>(v));
            }
            const uint32_t elementCount = ComputeElementCount(dimensions);

            std::wstring name = g_converterToUtf8.from_bytes(onnxTensor.name());
            currentFileName.assign(initialFileName);
            currentFileName.insert(substitutionOffset, name);

            PrintTensorInfo(onnxTensor.name(), currentFileName, dimensions, dataType);

            switch (outputFileType)
            {
            case FileType::OnnxTensor:
                {
                    std::ofstream os(currentFileName, std::ios::binary);
                    succeeded = onnxTensor.SerializeToOstream(&os);
                }
                break;

            case FileType::RawData:
                {
                    std::string arrayByteData = GetOnnxTensorRawByteData(onnxTensor);
                    WriteBinaryFile(currentFileName.c_str(), arrayByteData);
                }
                break;

            case FileType::NumPyArray:
                {
                    std::string fileData;
                    std::string arrayByteData = GetOnnxTensorRawByteData(onnxTensor);
                    WriteNpy(arrayByteData, onnxTensor.data_type(), dimensions, /*out*/ fileData);
                    WriteBinaryFile(currentFileName.c_str(), fileData);
                }
                break;
            }
        }
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

// Downcasts a given element type to uint8 (e.g. for pixel imagery).
template <typename T, size_t sourceElementByteStride = sizeof(T)>
void ConvertElementTypeToUInt8(
    _In_reads_bytes_(elementCount * sourceElementByteStride) uint8_t const* source,
    _Out_writes_(elementCount) uint8_t* destination,
    size_t elementCount
    )
{
    // std::copy gives warnings about casting, but we explicitly do want the cast, even if there is bit loss.
    for (; elementCount != 0; --elementCount)
    {
        T const* recastSource = reinterpret_cast<T const*>(source);
        *destination++ = static_cast<uint8_t>(*recastSource * 255);
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

void GenerateTensorSequence(
    std::wstring_view fileName,
    onnx::TensorProto::DataType dataType,
    span<int32_t const> dimensions,
    /*out*/ std::vector<char>& arrayByteData
    )
{
    arrayByteData.clear();
    arrayByteData.resize(GetByteSizeFromDimensions(dimensions, dataType));

    // Strip off "generate(".
    if (!starts_with(fileName, std::wstring_view(L"generate("))
    ||  !ends_with(fileName, std::wstring_view(L")")))
    {
        return;
    }
    span<const wchar_t> generatorType = fileName;
    generatorType.pop_front(9);
    generatorType.pop_back(1);

    // Tokenize using commas.
    std::vector<span<const wchar_t>> tokens = tokenize(generatorType, L',');
    if (tokens.empty())
    {
        throw std::invalid_argument("Expected a value generator after generate (e.g. ones, zeros, values, iota, random");
    }

    ScalarValueUnion valueUnion = {};
    const bool isFloatingPointDataType = IsFloatingPointDataType(dataType);

    auto GetFloatNumberFromToken = [](span<const wchar_t> token)
    {
        float value = 0;
        std::string tokenU8 = g_converterToUtf8.to_bytes(token.begin(), token.end());
        std::from_chars(tokenU8.data(), tokenU8.data() + tokenU8.size(), /*out*/ value);
        return value;
    };

    if (equals(tokens.front(), std::wstring_view(L"zeros"))) // (not "zeroes" which is a verb)
    {
        // The default getter already returns zeros.
        WriteTensorValues(/*out*/ reinterpret_span<uint8_t>(arrayByteData), dataType, valueUnion);
    }
    else if (equals(tokens.front(), std::wstring_view(L"ones")))
    {
        if (isFloatingPointDataType)
        {
            valueUnion.floatValue = 1.0;
        }
        else
        {
            valueUnion.uintValue = 1;
        }
        WriteTensorValues(/*out*/ reinterpret_span<uint8_t>(arrayByteData), dataType, valueUnion);
    }
    else if (equals(tokens.front(), std::wstring_view(L"values")))
    {
        float value = (tokens.size() > 1) ? GetFloatNumberFromToken(tokens[1]) : 1.0f;
        if (isFloatingPointDataType)
        {
            valueUnion.floatValue = value;
        }
        else
        {
            valueUnion.intValue = static_cast<int64_t>(value);
        }
        WriteTensorValues(/*out*/ reinterpret_span<uint8_t>(arrayByteData), dataType, valueUnion);
    }
    else if (equals(tokens.front(), std::wstring_view(L"iota")))
    {
        float startingValue = (tokens.size() > 1) ? GetFloatNumberFromToken(tokens[1]) : 0.0f;
        float increment = (tokens.size() > 2) ? GetFloatNumberFromToken(tokens[2]) : 1.0f;

        std::function<ScalarValueUnion()> getter;
        if (isFloatingPointDataType)
        {
            valueUnion.floatValue = startingValue;
            getter = [&]()->ScalarValueUnion
            {
                ScalarValueUnion newValueUnion = valueUnion;
                valueUnion.floatValue += increment;
                return newValueUnion;
            };
        }
        else // Integer
        {
            valueUnion.intValue = static_cast<int64_t>(startingValue);
            getter = [&]()->ScalarValueUnion
            {
                ScalarValueUnion newValueUnion = valueUnion;
                valueUnion.intValue += int64_t(increment);
                return newValueUnion;
            };
        }
        WriteTensorValues(/*out*/ reinterpret_span<uint8_t>(arrayByteData), dataType, getter);
    }
    else if (equals(tokens.front(), std::wstring_view(L"random")))
    {
        float min = (tokens.size() > 1) ? GetFloatNumberFromToken(tokens[1]) : 1.0f;
        float max = (tokens.size() > 2) ? GetFloatNumberFromToken(tokens[2]) : 255.0f;

        std::function<ScalarValueUnion()> getter;
        std::default_random_engine generator;
        std::uniform_int_distribution<int> intDistribution(static_cast<int>(min), static_cast<int>(max));
        std::uniform_real_distribution<float> floatDistribution(min, max);

        if (isFloatingPointDataType)
        {
            getter = [&]()->ScalarValueUnion
            {
                ScalarValueUnion value = {};
                value.floatValue = floatDistribution(generator);
                return value;
            };
        }
        else // Integer
        {
            getter = [&]()->ScalarValueUnion
            {
                ScalarValueUnion value = {};
                value.intValue = intDistribution(generator);
                return value;
            };
        }

        WriteTensorValues(/*out*/ reinterpret_span<uint8_t>(arrayByteData), dataType, getter);
    }
    else
    {
        throw std::invalid_argument("Not a valid mode for generate().");
    }
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

void ConvertTensor(
    _In_z_ wchar_t const* inputFilename,
    span<int32_t const> dimensions,
    onnx::TensorProto::DataType dataType,
    HalfOpenRangeUint32 rowRange,           // matters for CSV files
    HalfOpenRangeUint32 columnRange,        // matters for CSV files
    std::string_view pixelFormatString,     // matters for image files
    std::string_view channelLayoutString,   // matters for image files
    bool shouldPrintRawBytes,               // for printing CSV
    _In_z_ wchar_t const* outputFilename
    )
{
    FileType inputFileType  = GetFileType(std::wstring_view(inputFilename));
    FileType outputFileType = GetFileType(std::wstring_view(outputFilename));

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
    std::vector<int32_t> resolvedDimensions(dimensions.begin(), dimensions.end());

    if (!dimensions.empty())
    {
        if (inputFileType == FileType::Text
        ||  inputFileType == FileType::OnnxModel
        ||  inputFileType == FileType::GoogleProtobuf
        ||  inputFileType == FileType::OnnxTensor
        ||  inputFileType == FileType::NumPyArray
        ||  inputFileType == FileType::Image)
        {
            throw std::invalid_argument("\"dimensions\" are invalid when reading from this file type.");
        }
    }

    bool succeeded = true;
    if (inputFileType == FileType::Text)
    {
        std::string modelString = ReadTextFile(inputFilename);
        succeeded = google::protobuf::TextFormat::ParseFromString(modelString, &tensor);
    }
    else if (inputFileType == FileType::OnnxTensor
          || inputFileType == FileType::GoogleProtobuf)
    {
        std::ifstream ifs(inputFilename, std::ios::binary);
        succeeded = tensor.ParseFromIstream(&ifs);
    }
    else if (inputFileType == FileType::RawData)
    {
        std::string arrayByteData = ReadTextFile(inputFilename);
        resolvedDimensions = ResolveEmptyDimensions(dimensions, arrayByteData, dataType);

        MakeTensor(arrayByteData, dataType, resolvedDimensions, "", /*out*/ tensor);
    }
    else if (inputFileType == FileType::CommaSeparatedValue)
    {
        std::string text = ReadTextFile(inputFilename);
        std::vector<char> arrayByteData;

        ReadCsv(text, dataType, rowRange, columnRange, /*out*/ arrayByteData);
        resolvedDimensions = ResolveEmptyDimensions(dimensions, arrayByteData, dataType);
        MakeTensor(arrayByteData, dataType, resolvedDimensions, "", /*out*/ tensor);
    }
    else if (inputFileType == FileType::NumPyArray)
    {
        std::string fileData = ReadTextFile(inputFilename);
        std::vector<char> arrayByteData;

        ReadNpy(fileData, /*out*/ dataType, /*out*/ resolvedDimensions, /*out*/ arrayByteData);
        MakeTensor(arrayByteData, dataType, resolvedDimensions, "", /*out*/ tensor);
    }
    else if (inputFileType == FileType::Image)
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
    else if (inputFileType == FileType::TensorGenerator)
    {
        std::vector<char> arrayByteData;
        GenerateTensorSequence(inputFilename, dataType, dimensions, /*out*/ arrayByteData);
        MakeTensor(arrayByteData, dataType, resolvedDimensions, "", /*out*/ tensor);
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
    dataType = onnx::TensorProto::DataType(tensor.data_type());
    if (resolvedDimensions.empty())
    {
        for (auto v : tensor.dims())
        {
            resolvedDimensions.push_back(static_cast<int32_t>(v));
        }
    }

    // Print details.
    PrintTensorInfo(tensor.name(), L"", resolvedDimensions, dataType);

    if (outputFileType == FileType::Text)
    {
        std::string modelString;
        if (succeeded = google::protobuf::TextFormat::PrintToString(tensor, &modelString))
        {
            WriteBinaryFile(outputFilename, modelString);
        }
    }
    else if (outputFileType == FileType::OnnxTensor
          || outputFileType == FileType::GoogleProtobuf)
    {
        std::ofstream os(outputFilename, std::ios::binary);
        succeeded = tensor.SerializeToOstream(&os);
    }
    else if (outputFileType == FileType::RawData)
    {
        std::string arrayByteData = GetOnnxTensorRawByteData(tensor);
        WriteBinaryFile(outputFilename, arrayByteData);
    }
    else if (outputFileType == FileType::CommaSeparatedValue)
    {
        std::string arrayByteData = GetOnnxTensorRawByteData(tensor);
        std::string text;
        WriteCsv(arrayByteData, onnx::TensorProto::DataType(tensor.data_type()), shouldPrintRawBytes, /*out*/ text);
        WriteBinaryFile(outputFilename, text);
    }
    else if (outputFileType == FileType::NumPyArray)
    {
        std::string fileData;
        std::string arrayByteData = GetOnnxTensorRawByteData(tensor);
        WriteNpy(arrayByteData, tensor.data_type(), resolvedDimensions, /*out*/ fileData);
        WriteBinaryFile(outputFilename, fileData);
    }
    else if (outputFileType == FileType::Image)
    {
        std::string arrayByteData = GetOnnxTensorRawByteData(tensor);
        std::vector<uint8_t> pixelBytes(arrayByteData.data(), arrayByteData.data() + arrayByteData.size());
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
    // Credits, examples, and option help.
    std::cout << "ConvertOnnxModel 2018-07-19..2020-07-11 FDR\r\n"
                 "Example usage:\r\n"
                 "    ConvertOnnxModel.exe input.onnx output.txt\r\n"
                 "    ConvertOnnxModel.exe input.txt output.onnx\r\n"
                 "    ConvertOnnxModel.exe input.onnxtensor output.csv\r\n"
                 "    ConvertOnnxModel.exe -tensor input.pb output.png\r\n"
                 "    ConvertOnnxModel.exe -tensor -dimensions 224,224 -datatype uint8 -row 2 -column 1,225 Foo.csv Foo.dat\r\n"
                 "    ConvertOnnxModel.exe input.npy output.onnxtensor\r\n"
                 "    ConvertOnnxModel.exe resnet50.onnx x:\\resnet_*.npy\r\n"
                 "    ConvertOnnxModel.exe -dimensions 3,4 -datatype float16 generate(random,1,24) output.onnxtensor\r\n"
                 "\r\n"
                 "Parameters:\r\n"
                 "     input file - graph (onnx/pb/text) or tensor (onnxtensor/pb/npy/text/csv/dat/generate)\r\n"
                 "    output file - graph (onnx/pb/text) or tensor (onnxtensor/pb/npy/text/csv/dat)\r\n"
                 "        -tensor - convert tensor instead of graph (if can't tell from file extension).\r\n"
                 "         -graph - convert graph (default).\r\n"
                 "    -dimensions - explicit tensor dimensions for .csv or .dat file. Defaults to\r\n"
                 "                  1D element count from source data.\r\n"
                 "      -datatype - tensor element type (float16,float32,float64,int8,uint8,int16,\r\n"
                 "                  uint16,int32,uint32,int64,uint64,bool8,complex64,complex128).\r\n"
                 "        -rawhex - display as raw hexadecimal when writing .csv\r\n"
                 "           -row - single row or range for .csv.\r\n"
                 "        -column - single column or range for .csv.\r\n"
                 "\r\n"
                 "File types:\r\n"
                 "    .onnx - Open Neural Exchange model protobuf\r\n"
                 "    .onnxtensor - Open Neural Exchange tensor\r\n"
                 "    .pb  - Google Protobuf (unstated type, might be tensor)\r\n"
                 "    .txt - Text\r\n"
                 "    .csv - Comma Separate Value\r\n"
                 "    .png - Image (Portable Network Graphics)\r\n"
                 "    .jpg - Image (Joint Photographic Experts Group)\r\n"
                 "    .npy - NumPyArray single tensor\r\n"
                 "    .dat/.bin - Raw binary array (no header)\r\n"
                 "    generate(...) - Generator pseudo filename\r\n"
                 "      generate(ones) - all ones. [1,1,1,1...]\r\n"
                 "      generate(zeros) - all zeros [0,0,0,0...]\r\n"
                 "      generate(values,3) - specific value [3,3,3,3...]\r\n"
                 "      generate(iota,1,2) - increasing sequence [1,3,5...]\r\n"
                 "      generate(random,1,100) - random values between min/max [31,56,2,69...]\r\n"
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

ConversionMode GetConversionModeFromFileType(FileType fileType)
{
    switch (fileType)
    {
    case FileType::Unknown: return ConversionMode::Unknown;
    case FileType::OnnxModel: return ConversionMode::Graph;
    case FileType::GoogleProtobuf: return ConversionMode::Unknown;
    case FileType::Text: return ConversionMode::Unknown;
    case FileType::CommaSeparatedValue: return ConversionMode::Tensor;
    case FileType::Image: return ConversionMode::Tensor;
    case FileType::RawData: return ConversionMode::Unknown;
    case FileType::OnnxTensor: return ConversionMode::Tensor;
    case FileType::NumPyArray: return ConversionMode::Tensor;
    case FileType::TensorGenerator: return ConversionMode::Tensor;
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
    bool shouldPrintRawBytes = false;

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
                    throw std::invalid_argument("Dimensions expected. e.g. -dimensions 3,224,224");
                }

                std::string s = g_converterToUtf8.to_bytes(argv[i]);
                ReadCsv(s, /*out*/dimensions);
            }
            else if (argument == L"-datatype")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Valid data type expected: -datatype float32 (uint8, int32, int16, bool8, float64...)");
                }

                std::string s = g_converterToUtf8.to_bytes(argv[i]);
                dataType = GetDataTypeFromStringName(s);
                if (dataType == onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED)
                {
                    throw std::invalid_argument("Valid data type expected: -datatype float32 (uint8, int32, int16, bool8, float64...)");
                }
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
            else if (argument == L"-rawhex")
            {
                shouldPrintRawBytes = true;
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
        FileType inputFileType = GetFileType(std::wstring_view(inputFilename));
        FileType outputFileType = GetFileType(std::wstring_view(outputFilename));
        conversionMode = GetConversionModeFromFileType(inputFileType);
        if (conversionMode == ConversionMode::Unknown)
        {
            conversionMode = GetConversionModeFromFileType(outputFileType);
        }
    }

    printf(
        "Input filename:  %S\r\n"
        "Output filename: %S\r\n"
        "Conversion mode: %s\r\n"
        "\r\n",
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
            shouldPrintRawBytes,
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
        std::cout << e.what() << "\r\n";
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cout << "Unknown error\r\n.";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
