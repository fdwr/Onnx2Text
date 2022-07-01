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

inline char ToChar(std::byte c) { return static_cast<char>(c); }
inline char* ToChar(std::byte* p) { return reinterpret_cast<char*>(p); }
inline char const* ToChar(std::byte const* p) { return reinterpret_cast<char const*>(p); }
inline char ToChar(char8_t c) { return static_cast<char>(c); }
inline char* ToChar(char8_t* p) { return reinterpret_cast<char*>(p); }
inline char const* ToChar(char8_t const* const p) { return reinterpret_cast<char const* const>(p); }
inline char** ToChar(char8_t** p) { return reinterpret_cast<char**>(p); }
inline char* const* ToChar(char8_t* const* const p) { return reinterpret_cast<char* const* const>(p); }
inline char const* const* ToChar(char8_t const* const* const p) { return reinterpret_cast<char const* const* const>(p); }
inline unsigned char* ToUChar(char* p) { return reinterpret_cast<unsigned char*>(p); }
inline unsigned char* ToUChar(char8_t* p) { return reinterpret_cast<unsigned char*>(p); }
inline unsigned char* ToUChar(std::byte* p) { return reinterpret_cast<unsigned char*>(p); }
inline char8_t* ToUtf8Char(char* p) { return reinterpret_cast<char8_t*>(p); }
inline std::u8string_view ToUtf8Char(std::string_view s) { return std::u8string_view(reinterpret_cast<char8_t const*>(s.data()), s.size()); }

std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> g_converterToUtf8;

inline std::u8string ToUtf8String(std::wstring_view source)
{
    static_assert(sizeof(wchar_t) == sizeof(char16_t), "Doesn't work on Linux, expecting UTF-16 for wchar_t.");
    std::u8string dest;
    std::string temporary = g_converterToUtf8.to_bytes(source.data(), source.data() + source.size());
    dest = std::move(*reinterpret_cast<std::u8string const*>(&temporary));
    return dest;
}

inline std::u8string ToUtf8String(std::u16string_view source)
{
    static_assert(sizeof(wchar_t) == sizeof(char16_t), "Doesn't work on Linux, expecting UTF-16 for wchar_t.");
    std::u8string dest;
    std::string temporary = g_converterToUtf8.to_bytes(
        reinterpret_cast<wchar_t const*>(source.data()),
        reinterpret_cast<wchar_t const*>(source.data() + source.size())
    );
    dest = std::move(*reinterpret_cast<std::u8string const*>(&temporary));
    return dest;
}

inline std::u16string ToUtf16String(std::u8string_view source)
{
    std::u16string dest;
    std::wstring temporary = g_converterToUtf8.from_bytes(ToChar(source.data()), ToChar(source.data() + source.size()));
    dest = std::move(*reinterpret_cast<std::u16string const*>(&temporary));
    return dest;
}

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
    std::string result;
    std::stringstream stream;
    stream << "Failing HRESULT: 0x" << std::hex << hr;
    stream.str(/*out*/ result);
    throw std::runtime_error(result.c_str());
}

#ifndef THROW_IF_FAILED
#define THROW_IF_FAILED(hr) {if (FAILED(hr)) ThrowBadHResultRuntimeErrorWithMessage(hr);}
#endif

template <typename T>
class span
{
public:
    span() = default;

    constexpr span(span<T>& s) = default;

    template<typename ContiguousContainerType>
    constexpr span(ContiguousContainerType&& container)
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
    T& back()  noexcept { return *(end_ - 1); }
    T const& front() const noexcept { return *begin_; }
    T const& back()  const noexcept { return *(end_ - 1); }
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

template<
    typename ContiguousContainerType,
    typename ElementType = std::remove_reference_t<decltype(*std::declval<ContiguousContainerType>().data())>
>
auto make_span(ContiguousContainerType& container) -> span<ElementType>
{
    auto* begin = std::data(container);
    return span<ElementType>(begin, begin + std::size(container));
}

template<typename ContiguousContainerType>
span<const std::byte> as_bytes(ContiguousContainerType const& container)
requires (std::is_const_v<std::remove_pointer_t<decltype(container.data())>>)
{
    auto oldSpan = make_span(container);
    return span<const std::byte>(reinterpret_cast<const std::byte*>(oldSpan.data()), oldSpan.size_bytes());
}

template<typename ContiguousContainerType>
span<std::byte> as_bytes(ContiguousContainerType& container)
requires (!std::is_const_v<std::remove_pointer_t<decltype(container.data())>>)
{
    auto oldSpan = make_span(container);
    return span<std::byte>(reinterpret_cast<std::byte*>(oldSpan.data()), oldSpan.size_bytes());
}

template<typename T>
span<const std::byte> struct_as_bytes(T const& data)
requires (std::is_const_v<decltype(data)>)
{
    return span<const std::byte>(reinterpret_cast<const std::byte*>(std::addressof(data)), sizeof(data));
}

template<typename T>
span<std::byte> struct_as_bytes(T const& data)
requires (!std::is_const_v<decltype(data)>)
{
    return span<std::byte>(reinterpret_cast<const std::byte*>(std::addressof(data)), sizeof(data));
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
    using NewType = decltype(*oldSpan.data());
    size_t endOffset = offset + count;
    size_t maxOffset = oldSpan.size();
    size_t beginOffset = std::min(offset, maxOffset);
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

template<
    typename ContiguousContainerType,
    typename ElementType = std::remove_reference_t<decltype(*std::declval<ContiguousContainerType>().data())>
>
auto append_data(ContiguousContainerType& v, span<ElementType const> s)
{
    // Whyyyy is basic functionality like std::vector::append() missing from the standard? -_-
    v.insert(v.end(), s.begin(), s.end());
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

template<typename OutputType, typename InputType> OutputType clamp_cast(InputType input)
{
    // Determine the larger type to decide which numeric limits to clamp to.
    using InputLimits = std::numeric_limits<InputType>;
    using OutputLimits = std::numeric_limits<OutputType>;
    constexpr int inputMaxDigits = std::max(InputLimits::max_exponent, InputLimits::digits);
    constexpr int outputMaxDigits = std::max(OutputLimits::max_exponent, OutputLimits::digits);
    constexpr bool isEitherTypeUnsigned = std::is_unsigned_v<InputType> || std::is_unsigned_v<OutputType>;
    constexpr bool isOutputTypeLarger = outputMaxDigits > inputMaxDigits;

    InputType lowestValue  = isEitherTypeUnsigned ? static_cast<InputType>(0) :
                             isOutputTypeLarger ? InputLimits::lowest() :
                             static_cast<InputType>(OutputLimits::lowest());
    InputType highestValue = isOutputTypeLarger ? InputLimits::max() :
                             static_cast<InputType>(OutputLimits::max());

    return static_cast<OutputType>(std::clamp<InputType>(input, lowestValue, highestValue));
}

// Read the file, calling back to set the size and return a span to the data to write into.
void ReadBinaryFileWithCallback(
    wchar_t const* inputFilename,
    std::function<span<std::byte>(size_t newSize)> setDataSize
    )
{
    std::ifstream file(inputFilename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::ios::failure("Could not open input file.");
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    span<std::byte> s = setDataSize(size);
    file.read(ToChar(s.data()), size);
}

template<typename T = std::vector<std::byte>>
T ReadBinaryFile(wchar_t const* inputFilename)
{
    T fileData;
    ReadBinaryFileWithCallback(
        inputFilename,
        [&](size_t newSize)
        {
            fileData.resize(newSize);
            return as_bytes(fileData);
        }
    );
    return fileData;
}

void WriteBinaryFileBytes(wchar_t const* outputFilename, span<std::byte const> fileData)
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

    file.write(ToChar(fileData.data()), fileData.size());
}

template <typename ContainerType>
void WriteBinaryFile(wchar_t const* outputFilename, ContainerType const& fileData)
{
    return WriteBinaryFileBytes(outputFilename, as_bytes(fileData));
}

enum class FileType
{
    Unknown,             // .onnx
    OnnxModel,           // .onnx
    GoogleProtobuf,      // .pb
    Text,                // .txt / .prototxt
    CommaSeparatedValue, // .csv
    Image,               // .png / .jpg
    RawData,             // .dat / .bin - raw binary, dump of tensor values as-is
    NumPyArray,          // .npy (not .npz zip files with multiple arrays in them)
    OnnxTensor,          // .onnxtensor
    TensorGenerator,     // generator:
    GraphVizDot,         // .dot
};

size_t GetFileExtensionOffset(std::wstring_view filename)
{
    size_t extensionOffset = filename.find_last_of(L".");
    extensionOffset = (extensionOffset != std::wstring_view::npos) ? extensionOffset + 1 : filename.size();
    return extensionOffset;
}

struct Mapping
{
    std::wstring_view filenameExtension;
    FileType fileType;
};
const static Mapping fileTypeMappings[] =
{
    { L"pb", FileType::GoogleProtobuf },
    { L"onnx", FileType::OnnxModel },
    { L"txt" , FileType::Text },
    { L"prototxt" , FileType::Text },
    { L"csv" , FileType::CommaSeparatedValue },
    { L"dat" , FileType::RawData },
    { L"bin" , FileType::RawData },
    { L"bmp" , FileType::Image },
    { L"png" , FileType::Image },
    { L"jpg" , FileType::Image },
    { L"jpeg", FileType::Image },
    { L"npy" , FileType::NumPyArray },
    { L"onnxtensor", FileType::OnnxTensor },
    { L"tensorproto", FileType::OnnxTensor },
    { L"dot", FileType::GraphVizDot },
};

FileType GetFileType(std::wstring_view filename)
{
    size_t extensionOffset = GetFileExtensionOffset(filename);
    std::wstring_view filenameExtension = filename.substr(extensionOffset);
    if (starts_with(filename, std::wstring_view(L"generate("))) return FileType::TensorGenerator;
    for (auto& mapping : fileTypeMappings)
    {
        if (filenameExtension == mapping.filenameExtension)
        {
            return mapping.fileType;
        }
    }

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
    /*out*/ span<std::byte> arrayByteData,
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
    /*out*/ span<std::byte> arrayByteData,
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
    std::u8string_view text,
    onnx::TensorProto::DataType dataType,
    HalfOpenRangeUint32 rowRange,
    HalfOpenRangeUint32 columnRange,
    /*out*/std::vector<std::byte>& byteData
    )
{
    byteData.clear();

    rowRange.ExpandAllIfEmpty();
    columnRange.ExpandAllIfEmpty();

    size_t elementByteSize = GetDataTypeElementByteSize(dataType);
    size_t byteDataSize = 0;
    uint32_t row = 1, column = 1;

    std::u8string unquotedText;

    CsvValueNumberClass valueNumberClass = GetCsvValueNumberClass(dataType, /*shouldPrintRawBytes*/ false);

    constexpr char quote = '\"';

    char8_t const* begin = text.data();
    char8_t const* end = text.data() + text.size();
    while (begin != end)
    {
        char8_t const* numberStart = begin;

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
                std::from_chars(ToChar(numberStart + 2), ToChar(end), /*out*/ value.uintValue);
            }
            else
            {
                switch (valueNumberClass)
                {
                case CsvValueNumberClass::Int:   std::from_chars(ToChar(numberStart), ToChar(end), /*out*/ value.intValue);      break;
                case CsvValueNumberClass::Uint:  std::from_chars(ToChar(numberStart), ToChar(end), /*out*/ value.uintValue);     break;
                case CsvValueNumberClass::Float: std::from_chars(ToChar(numberStart), ToChar(end), /*out*/ value.floatValue);    break;
                case CsvValueNumberClass::Hex:   std::from_chars(ToChar(numberStart), ToChar(end), /*out*/ value.uintValue, 16); break;
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
void ReadCsv(std::u8string_view text, /*out*/std::vector<int32_t>& values)
{
    values.clear();
    const char8_t* begin = text.data();
    const char8_t* end = text.data() + text.size();

    // Special case of empty dimensions.
    if (text == u8"[]" || text == u8"()")
    {
        return;
    }

    while (begin != end)
    {
        char8_t* valueEnd;
        uint32_t value = strtol(ToChar(begin), ToChar(&valueEnd), 10);
        values.push_back(value);
        if (valueEnd != end && *valueEnd == u8',')
        {
            ++valueEnd;
        }
        begin = valueEnd;
    }
}

// Writes tensor data to a string (not directly to a file).
void WriteCsv(
    span<std::byte const> byteData,
    onnx::TensorProto::DataType dataType,
    bool shouldPrintRawBytes, // Print raw hex bit values instead of formatted numbers.
    /*out*/std::u8string& text
    )
{
    text.clear();

    size_t elementByteSize = GetDataTypeElementByteSize(dataType);

    char8_t buffer[40];

    // Round off any potential trailing padding.
    byteData = span<std::byte const>(byteData.data(), (byteData.size() / elementByteSize) * elementByteSize);

    CsvValueNumberClass valueNumberClass = GetCsvValueNumberClass(dataType, shouldPrintRawBytes);

    std::byte const* begin = byteData.data();
    std::byte const* end = byteData.data() + byteData.size();

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
        case CsvValueNumberClass::Int:   sprintf_s(ToChar(buffer), std::size(buffer), "%lld",   value.intValue);   break;
        case CsvValueNumberClass::Uint:  sprintf_s(ToChar(buffer), std::size(buffer), "%llu",   value.uintValue);  break;
        case CsvValueNumberClass::Float: sprintf_s(ToChar(buffer), std::size(buffer), "%g",     value.floatValue); break;
        case CsvValueNumberClass::Hex:   sprintf_s(ToChar(buffer), std::size(buffer), "0x%llX", value.uintValue);  break;
        }

        text.append(buffer);
        if (begin != end)
        {
            text.push_back(',');
        }
    }
}

bool AreDimensionsSpecified(span<int32_t const> dimensions)
{
    return dimensions.size() != 1 || dimensions.back() >= 0;
}

std::vector<int32_t> ResolveUnspecifiedDimensions(
    span<int32_t const> defaultDimensions,
    span<std::byte const> byteData,
    onnx::TensorProto::DataType dataType
    )
{
    std::vector<int32_t> resolvedDimensions;

    if (AreDimensionsSpecified(defaultDimensions))
    {
        resolvedDimensions.assign(defaultDimensions.begin(), defaultDimensions.end());
    }
    else
    {
        // Return a 1D array if no dimensions were given, equal to the element count.
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
const char8_t* g_elementDataTypeNames[] =
{
    u8"undefined",    // Undefined = 0,
    u8"float32",      // Float32 = 1,
    u8"uint8",        // Uint8 = 2,
    u8"int8",         // Int8 = 3,
    u8"uint16",       // Uint16 = 4,
    u8"int16",        // Int16 = 5,
    u8"int32",        // Int32 = 6,
    u8"int64",        // Int64 = 7,
    u8"string8",      // StringChar8 = 8,
    u8"bool8",        // Bool = 9,
    u8"float16",      // Float16 = 10,
    u8"float64",      // Float64 = 11,
    u8"uint32",       // Uint32 = 12,
    u8"uint64",       // Uint64 = 13,
    u8"complex64",    // Complex64 = 14,
    u8"complex128",   // Complex128 = 15,
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

std::u8string_view GetStringNameFromDataType(onnx::TensorProto::DataType dataType) noexcept
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

onnx::TensorProto::DataType GetDataTypeFromStringName(std::u8string_view name) noexcept
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

bool IsRecognizedChannelLayoutString(std::u8string_view channelLayoutString)
{
    return channelLayoutString == u8"nchw" || channelLayoutString == u8"nhwc";
}

struct Struct128Bit
{
    uint32_t data[4];
};

void RearrangeChannels(
    onnx::TensorProto::DataType dataType,
    span<int32_t const> dimensions,
    std::u8string_view originalChannelLayoutString,
    std::u8string_view desiredChannelLayoutString,
    /*inout*/ std::vector<std::byte>& pixelBytes
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
        return; // No channels to reorder since only 1 channel exists.
    }

    std::vector<std::byte> destinationPixelBytes(pixelBytes.size());

    // Flatten the count of all the dimensions before the color channel.
    const span<int32_t const> chwDimensions = dimensions.subspan(dimensions.size() - 3, 3);
    const span<int32_t const> batchDimensions = dimensions.subspan(0, dimensions.size() - 3);
    const uint32_t destinationElementByteSize = GetByteSizeFromDataType(dataType);
    const uint32_t channelCount   = chwDimensions[0];
    const uint32_t heightCount    = chwDimensions[1];
    const uint32_t widthCount     = chwDimensions[2];
    const uint32_t batchCount     = ComputeElementCount(batchDimensions);
    const uint32_t totalByteCount = destinationElementByteSize * channelCount * heightCount * widthCount * batchCount;

    if (totalByteCount > destinationPixelBytes.size())
    {
        throw std::invalid_argument("Pixel total byte count exceeds dimension counts.");
    }

    size_t destinationByteOffset = 0;
    size_t sourceOffset0 = 0, sourceOffset1 = 0, sourceOffset2 = 0, sourceOffset3 = 0;

    uint32_t sourceStride0 = 0, sourceStride1 = 0, sourceStride2 = 0, sourceStride3 = 0;
    uint32_t count0 = 0, count1 = 0, count2 = 0, count3 = 0;
    if (desiredChannelLayoutString == u8"nchw")
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
    else if (desiredChannelLayoutString == u8"nhwc")
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
                    case 1: reinterpret_cast<uint8_t&>(destinationPixelBytes[destinationByteOffset]) = reinterpret_cast<uint8_t&>(pixelBytes[sourceOffset3]); break;
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

    case 4: // 32-bit
    case 8: // 64-bit
    case 16: // 128-bit
        {
            auto s32 = reinterpret_span<uint32_t>(arrayByteData);
            for (auto& u : s32)
            {
                uint32_t v = u;
                u = ((v & 0x000000FF) << 24) |
                    ((v & 0x0000FF00) << 8)  |
                    ((v & 0x00FF0000) >> 8)  |
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

template<typename DataType>
void RescaleArray(
    double prebias,
    double scale,
    /*inout*/ span<std::byte> arrayByteData
    )
{
    auto recastedData = reinterpret_span<DataType>(arrayByteData);
    constexpr double lowestValue  = double(std::numeric_limits<DataType>::lowest());
    constexpr double highestValue = double(std::numeric_limits<DataType>::max());
    auto functor = [=](DataType& v)
    {
        double clampedValue = std::clamp<double>((v + prebias) * scale, lowestValue, highestValue);
        v = static_cast<DataType>(clampedValue);
    };

    std::for_each(recastedData.begin(), recastedData.end(), functor);
}

void RescaleArray(
    onnx::TensorProto::DataType dataType,
    double prebias,
    double scale,
    /*inout*/ span<std::byte> arrayByteData
    )
{
    switch (dataType)
    {
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:   RescaleArray<bool    >(prebias, scale, /*inout*/ arrayByteData); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:  RescaleArray<uint8_t >(prebias, scale, /*inout*/ arrayByteData); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16: RescaleArray<uint16_t>(prebias, scale, /*inout*/ arrayByteData); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32: RescaleArray<uint32_t>(prebias, scale, /*inout*/ arrayByteData); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:   RescaleArray<int8_t  >(prebias, scale, /*inout*/ arrayByteData); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:  RescaleArray<int16_t >(prebias, scale, /*inout*/ arrayByteData); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:  RescaleArray<int32_t >(prebias, scale, /*inout*/ arrayByteData); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:  RescaleArray<float   >(prebias, scale, /*inout*/ arrayByteData); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE: RescaleArray<double  >(prebias, scale, /*inout*/ arrayByteData); break;
    default:
        assert(false); // Could not have reached here because we only set a known subset.
    }
}

template<typename InputDataType, typename OutputDataType>
std::pair<OutputDataType, OutputDataType> ArrayMinMax(
    /*inout*/ span<std::byte> arrayByteData
    )
{
    auto recastedData = reinterpret_span<InputDataType>(arrayByteData);
    InputDataType lowestValue  = std::numeric_limits<InputDataType>::max();
    InputDataType highestValue = std::numeric_limits<InputDataType>::lowest();
    auto functor = [&](InputDataType v) mutable
    {
        if (v < lowestValue ) lowestValue  = v;
        if (v > highestValue) highestValue = v;
    };

    std::for_each(recastedData.begin(), recastedData.end(), functor);

    return std::pair<OutputDataType, OutputDataType>(
        static_cast<OutputDataType>(lowestValue),
        static_cast<OutputDataType>(highestValue)
    );
}

std::pair<double, double> ArrayMinMax(
    onnx::TensorProto::DataType dataType,
    /*inout*/ span<std::byte> arrayByteData
    )
{
    switch (dataType)
    {
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:   return ArrayMinMax<bool    , double>(arrayByteData);
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:  return ArrayMinMax<uint8_t , double>(arrayByteData);
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16: return ArrayMinMax<uint16_t, double>(arrayByteData);
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32: return ArrayMinMax<uint32_t, double>(arrayByteData);
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:   return ArrayMinMax<int8_t  , double>(arrayByteData);
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:  return ArrayMinMax<int16_t , double>(arrayByteData);
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:  return ArrayMinMax<int32_t , double>(arrayByteData);
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:  return ArrayMinMax<float   , double>(arrayByteData);
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE: return ArrayMinMax<double  , double>(arrayByteData);
    default:
        assert(false); // Could not have reached here because we only set a known subset.
        return { 0.0, 1.0 };
    }
}

void MapNumPyArrayDataTypeToOnnx(
    std::u8string_view numPyElementType,
    /*out*/onnx::TensorProto::DataType& dataType,
    /*out*/ bool& isBackwardsEndian // Backwards endian which stores greatest bytes at lowest bytes.
    )
{
    dataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
    isBackwardsEndian = false;

    onnx::TensorProto::DataType resolvedDataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
    uint32_t elementByteSize = 0;

    #if !(defined(_M_IX86) || defined(_M_X64) || defined(_M_ARM) || defined(_M_ARM64))
    // Technically ARM machines can accept either, but the vast majority of ARM machines
    // default to logical endian, including all 2021 Windows ones and Android phones.
    static_assert(false, "Double check that endianness is specified correctly for this architecture when using '='.");
    #endif

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
        case '=': isBackwardsEndian = false; break;   // (logical-endian since targeting x86)
        case '|': isBackwardsEndian = false; break;   // not applicable

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

        default:
            assert(false); // Could not have reached here because we only set a known subset.
        }
    }

    dataType = resolvedDataType;
}

void AppendOnnxDataTypeToNumPyArray(
    onnx::TensorProto::DataType dataType,
    bool isBackwardsEndian, // Backwards endian which stores greatest bytes at lowest bytes.
    /*inout*/ std::u8string& numPyElementType
    )
{
    numPyElementType.push_back(isBackwardsEndian ? '>' : '<');

    // https://docs.python.org/2/library/array.html#module-array
    // https://numpy.org/devdocs/reference/arrays.dtypes.html
    std::u8string_view characterCode;
    switch (dataType)
    {
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:    characterCode = u8"?"  /*'?'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:    characterCode = u8"i1" /*'b'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:   characterCode = u8"u1" /*'B'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:   characterCode = u8"i2" /*'h'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:   characterCode = u8"i4" /*'i'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:   characterCode = u8"i8" /*'i'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:  characterCode = u8"u2" /*'H'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:  characterCode = u8"u4" /*'u'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:  characterCode = u8"u8" /*'u'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16: characterCode = u8"f2" /*'f'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:   characterCode = u8"f4" /*'f'*/; break;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:  characterCode = u8"f8" /*'d'*/; break;
    default: characterCode = u8"?";  assert(false);
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

    PythonDictionaryLexer(span<char8_t const> text) : text_(text)
    {
    }

    PythonDictionaryLexer(std::u8string_view text) : text_(text)
    {
    }

    PythonDictionaryLexer(span<std::byte const> text) : text_(reinterpret_span<char8_t const>(text))
    {
    }

    bool empty()
    {
        return text_.empty();
    }

    struct ReadStruct { span<const char8_t> token; TokenType tokenType; };
    ReadStruct Read()
    {
        static_assert(int(char(-42)) == int(uint8_t(-42)), "char must be unsigned. Signed char makes no freaking sense at all. Fix your compiler.");
        span<const char8_t> token;
        TokenType tokenType = TokenType::End;

        // Skip spaces.
        for (; !text_.empty() && isspace(uint8_t(text_.front())); text_.pop_front())
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

    std::map<std::u8string_view, std::u8string_view> ReadDictionary()
    {
        int indentLevel = 0;

        std::map<std::u8string_view, std::u8string_view> map;
        std::u8string_view currentKey;
        std::u8string_view currentValue;
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
                    //char8_t const* endptr = token.end();
                    uint32_t value = 0;
                    std::from_chars(ToChar(token.begin()), ToChar(token.end()), /*out*/ value);
                    numbers.push_back(value);
                }
                break;

            default:
                ; // Skip anything else.
            }
        }
    End:;
    }

private:
    span<const char8_t> text_;
};

class PythonDictionaryWriter
{
public:
    std::u8string_view GetText() const
    {
        return text_;
    }

    span<std::byte const> const GetBytes() const
    {
        return reinterpret_span<std::byte const>(text_);
    }

    void Append(std::u8string_view text)
    {
        text_.append(text);
    }

    void WriteKeyValueUnquoted(std::u8string_view key, std::u8string_view value)
    {
        text_.append(key);
        text_.append(u8":");
        text_.append(value);
        text_.append(u8", ");
    }

    void WriteKeyValue(std::u8string_view key, std::u8string_view value)
    {
        text_.push_back('\'');
        text_.append(key);
        text_.append(u8"\':\'");
        text_.append(value);
        text_.append(u8"\', ");
    }

    void WriteIntegers(span<const int32_t> numbers, std::u8string_view brackets)
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
            text_.append(u8",");
        }
        if (!brackets.empty())
        {
            text_.push_back(brackets.back());
        }
    }

private:
    std::u8string text_;
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
    span<std::byte const> fileData,
    /*out*/onnx::TensorProto::DataType& dataType,
    /*out*/std::vector<int32_t>& dimensions,
    /*out*/std::vector<std::byte>& arrayByteData
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

    PythonDictionaryLexer lexer(fileData.subrange(dictionaryOffset, dataByteOffset));
    std::map<std::u8string_view, std::u8string_view> dictionary = lexer.ReadDictionary();

    bool isBackwardsEndian = false;
    bool hasIncreasingStrides = false;

    for (auto& i : dictionary)
    {
        if (i.first == u8"descr"sv)
        {
            MapNumPyArrayDataTypeToOnnx(i.second, /*out*/ dataType, /*out*/ isBackwardsEndian);
        }
        else if (i.first == u8"fortran_order"sv)
        {
            hasIncreasingStrides = (i.second == u8"True"sv);
        }
        else if (i.first == u8"shape"sv)
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
    span<std::byte const> arrayByteData,
    onnx::TensorProto::DataType dataType,
    span<int32_t const> dimensions,
    /*out*/std::vector<std::byte>& fileData
    )
{
    NumPyArrayHeaderV1 headerFixedPart = { {uint8_t('\x0093'),'N','U','M','P','Y'}, 1,0, 0 };

    PythonDictionaryWriter dictionaryWriter;
    PythonDictionaryWriter numberWriter;

    // Format dictionary fields.
    std::u8string numPyElementType;
    AppendOnnxDataTypeToNumPyArray(dataType, /*isBackwardsEndian*/ false, /*inout*/ numPyElementType);
    numberWriter.WriteIntegers(dimensions, u8"()");

    dictionaryWriter.Append(u8"{");
    dictionaryWriter.WriteKeyValue(u8"descr", numPyElementType);
    dictionaryWriter.WriteKeyValueUnquoted(u8"'fortran_order'", u8"False");
    dictionaryWriter.WriteKeyValueUnquoted(u8"'shape'", numberWriter.GetText());
    dictionaryWriter.Append(u8"}");

    // Compute header length for alignment.
    uint32_t headerLength = sizeof(headerFixedPart);
    headerLength += static_cast<uint32_t>(dictionaryWriter.GetText().size());
    headerLength++; // For new line.
    headerLength = (headerLength + 63) & ~63; // For rounding up to multiple of 64 alignment.

    // Write header, including fixed size part, dictionary, and alignment padding.
    headerFixedPart.dictionaryLength = static_cast<uint16_t>(headerLength - sizeof(headerFixedPart));
    append_data(/*inout*/ fileData, { reinterpret_cast<std::byte const*>(&headerFixedPart), sizeof(headerFixedPart) });
    append_data(/*inout*/ fileData, dictionaryWriter.GetBytes());
    fileData.insert(fileData.end(), headerLength - fileData.size(), std::byte{' '});
    fileData.back() = std::byte{ '\x000A' }; // Terminate with new line.
    // Note the spec says "It is terminated by a newline (\n) and padded with spaces (\x20)",
    // but that's wrong. It's actually "padding with spaces and then terminated by a newline".
    // Otherwise Numpy 1.18.5 barfs (1.19 works fine either way).
    // https://numpy.org/devdocs/reference/generated/numpy.lib.format.html

    append_data(/*inout*/ fileData, arrayByteData);
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
    constexpr size_t outputElementSize = sizeof(OutputElementType);
    outputContainer.resize(elementCount * outputElementSize);

    span<OutputElementType> outputValues = reinterpret_span<OutputElementType>(outputContainer);
    size_t index = 0;
    for (auto i = begin; i != end; ++i)
    {
        outputValues[index++] = static_cast<OutputElementType>(*i);
    }
}

std::vector<std::byte> GetOnnxTensorRawByteData(onnx::TensorProto const& tensor)
{
    std::vector<std::byte> bytes;
    if (tensor.has_raw_data())
    {
        std::string const& s = tensor.raw_data();
        span<std::byte const> b = as_bytes(s);
        bytes.assign(b.begin(), b.end());
    }
    else
    {
        switch (tensor.data_type())
        {
        case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:    CopyOnnxTensorDataToBuffer<uint16_t>(tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      CopyOnnxTensorDataToBuffer<float>   (tensor.float_data().begin(),  tensor.float_data().end(),  tensor.float_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     CopyOnnxTensorDataToBuffer<double>  (tensor.double_data().begin(), tensor.double_data().end(), tensor.double_data_size(), bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       CopyOnnxTensorDataToBuffer<bool>    (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      CopyOnnxTensorDataToBuffer<uint8_t> (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       CopyOnnxTensorDataToBuffer<int8_t>  (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     CopyOnnxTensorDataToBuffer<uint16_t>(tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      CopyOnnxTensorDataToBuffer<int16_t> (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     CopyOnnxTensorDataToBuffer<uint32_t>(tensor.uint64_data().begin(), tensor.uint64_data().end(), tensor.uint64_data_size(), bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      CopyOnnxTensorDataToBuffer<int32_t> (tensor.int32_data().begin(),  tensor.int32_data().end(),  tensor.int32_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     CopyOnnxTensorDataToBuffer<uint64_t>(tensor.uint64_data().begin(), tensor.uint64_data().end(), tensor.uint64_data_size(), bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      CopyOnnxTensorDataToBuffer<int64_t> (tensor.int64_data().begin(),  tensor.int64_data().end(),  tensor.int64_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  CopyOnnxTensorDataToBuffer<float>   (tensor.float_data().begin(),  tensor.float_data().end(),  tensor.float_data_size(),  bytes); break;
        case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: CopyOnnxTensorDataToBuffer<double>  (tensor.double_data().begin(), tensor.double_data().end(), tensor.double_data_size(), bytes); break;
        default: throw std::ios::failure("Unsupported data type in tensor for raw output.");
        }
    }

    return bytes;
}

void PrintTensorInfo(
    std::u8string_view name,
    std::wstring_view fileName,
    span<int32_t const> dimensions,
    onnx::TensorProto::DataType dataType
    )
{
    std::u8string dimensionsText;
    WriteCsv(
        as_bytes(dimensions),
        onnx::TensorProto::DataType::TensorProto_DataType_INT32,
        /*shouldPrintRawBytes*/ false,
        /*out*/ dimensionsText
    );

    const uint32_t byteSize = GetByteSizeFromDimensions(dimensions, dataType);
    if (!fileName.empty())
    {
        printf(
            "Tensor:\n"
            "  Name: \"%.*ls\"\n"
            "  Data type: %hs\n"
            "  Dimensions: %hs\n"
            "  Byte size: %d bytes\n"
            ,
            uint32_t(fileName.size()),
            fileName.data(),
            ToChar(GetStringNameFromDataType(dataType).data()),
            ToChar(dimensionsText.c_str()),
            byteSize
        );
    }
    else
    {
        printf(
            "Tensor:\n"
            "  name: \"%.*hs\"\n"
            "  datatype: %hs\n"
            "  dimensions: %hs\n"
            "  byte size: %d bytes\n"
            ,
            uint32_t(name.size()),
            ToChar(name.data()),
            ToChar(GetStringNameFromDataType(dataType).data()),
            ToChar(dimensionsText.c_str()),
            byteSize
        );
    }
}

void ZeroModelTensorWeights(onnx::ModelProto& model)
{
    onnx::GraphProto* graphProto = model.mutable_graph();
    int initializerSize = graphProto->initializer_size();
    for (int i = 0; i < initializerSize; ++i)
    {
        onnx::TensorProto* onnxTensor = graphProto->mutable_initializer(i);

        // Do not zero small 1D shape tensors, which would break model execution
        // used by Reshape/Expand/ConstantOfShape.
        auto dims = onnxTensor->dims();
        if (dims.size() == 1 && std::all_of(dims.begin(), dims.end(), [](int64_t dim) {return dim <= 8; }))
        {
            continue;
        }

        std::vector<std::byte> tensorValues = GetOnnxTensorRawByteData(*onnxTensor);
        onnxTensor->clear_int32_data();
        onnxTensor->clear_int64_data();
        onnxTensor->clear_uint64_data();
        onnxTensor->clear_float_data();
        onnxTensor->clear_double_data();
        std::fill(tensorValues.begin(), tensorValues.end(), std::byte{});
        onnxTensor->set_raw_data(tensorValues.data(), tensorValues.size());
    }
}

void DisplayModelInformation(onnx::ModelProto const& model)
{
    onnx::GraphProto const& graphProto = model.graph();
    printf(
        "Graph:\n"
        "  total nodes: %d\n"
        ,
        graphProto.node_size()
    );

    std::map<std::string, uint32_t> operatorTypeCounts;
    for (const onnx::NodeProto& node : graphProto.node())
    {
        auto operatorType = node.op_type();
        operatorTypeCounts[operatorType]++;
    }

    // Print the counts of each type of operator (e.g. "Conv" x 12).
    printf("Nodes:\n");
    for (auto& operatorTypeCount : operatorTypeCounts)
    {
        printf("  \"%s\",%d\n", operatorTypeCount.first.c_str(), operatorTypeCount.second);
    }
}

std::string GetSanitizedGraphVizIdentifier(std::string_view name)
{
    // This isn't complete because other characters could be found inside ONNX files,
    // but it handles the most problematic ones at least.
    std::string newName(name);
    for (char& c : newName)
    {
        switch (c)
        {
        case '\\':
        case '/':
        case '\"':
        case '|':
        case '<':
        case '>':
        case ':':
        case '?':
        case '*':
        case ' ':
            c = '_';
            break;
        }
    }
    return newName;
}

std::string GetSanitizedGraphVizLabel(std::string_view name)
{
    std::string newName(name);
    for (char& c : newName)
    {
        switch (c)
        {
        case '\'':
        case '\"':
            c = '_';
            break;
        }
    }
    return newName;
}

void ConvertOnnxToGraphViz(
    onnx::ModelProto const& model,
    bool showConstantTensors,
    std::ofstream& outputFile
    )
{
    const onnx::GraphProto& onnxGraph = model.graph();

    char const* header =
R"(digraph Graph {
// Settings
layout=dot
rankdir=TB;
edge [headport="n", tailport="s", arrowsize=0.5 ];
graph [pencolor=transparent, fontname="Segoe UI", fontsize=10, color=black]
splines=polyline;
nodesep=0.025;
ranksep=0.25;
center=true;
outputorder=edgesfirst;
)";
    char const* footer = "}\n";
    char const* inputsOutputsComment = "\n// Graph input/output tensors\n";
    char const* constantTensorsComment = "\n// Constant tensors\n";
    char const* edgesComment = "\n// Edges\n";
    char const* operatorsComment = "\n// Operators\n";
    char const* operatorNodeStyle =     R"(node [style="filled, rounded", color=black, fillcolor="#E0E0F0FF", penwidth=1, shape=rectangle, fontname="Segoe UI", fontsize=9, height=.2, width=1, margin="0.02, 0.02" ];)" "\n";
    char const* inputOutputNodeStyle =  R"(node [style=filled, color=black, fillcolor="#C0F0C0FF", penwidth=1, shape=rectangle, fontname="Segoe UI", fontsize=9, height=.2, width=0.8, margin="0.04, 0.04" ];)" "\n";
    char const* constantNodeStyle =     R"(node [style=filled, color=black, fillcolor="#D0E0D0FF", penwidth=1, shape=rectangle, fontname="Segoe UI", fontsize=9, height=.2, width=0.8, margin="0.04, 0.04" ];)" "\n";
    char const* intermediateNodeStyle = R"(node [color=transparent, fillcolor="#00000000", penwidth=0, shape=rectangle, fontname="Segoe UI", fontsize=9, height=.2, width=0.8, margin="0.01, 0.01" ];)" "\n";

    auto nodes = onnxGraph.node();
    std::vector<std::string> sanitizedNodeNames;
    sanitizedNodeNames.reserve(nodes.size());
    std::set<std::string, std::less<>> constantTensors;
    std::vector<std::string_view> constantTensorsInOrder;

    outputFile << header;

    // Gather the list of constant tensors defined with static weights inside the graph.
    for (const onnx::TensorProto& tensorProto : onnxGraph.initializer())
    {
        const std::string& name = tensorProto.name();
        constantTensors.insert(name);
    }

    // Write input/output tensors.
    outputFile << inputsOutputsComment;
    outputFile << inputOutputNodeStyle;

    // Exclude any constant tensors. Newer models do not have this issue, as they only declare
    // true inputs, but some old models defined all constant tensors as graph inputs too, which
    // confuses later logic when trying to bind tensors to inputs (e.g. opset 9 squeezenet).
    for (const onnx::ValueInfoProto& valueInfo : onnxGraph.input())
    {
        const std::string& name = valueInfo.name();
        if (!constantTensors.contains(name))
        {
            outputFile << GetSanitizedGraphVizIdentifier(name);
            outputFile << '\n';
        }
    }

    for (const onnx::ValueInfoProto& valueInfo : onnxGraph.output())
    {
        outputFile << GetSanitizedGraphVizIdentifier(valueInfo.name());
        outputFile << '\n';
    }

    if (showConstantTensors)
    {
        // Write constant tensors.
        outputFile << constantTensorsComment;
        outputFile << constantNodeStyle;

        // Enumerate them in node binding order rather than the arbitrary order they are listed in the model
        // or in alphabetic order, since node binding order is more intuitive when looking at the graph
        // since the inputs follow the ONNX order.
        constantTensorsInOrder.reserve(constantTensors.size());
        for (const onnx::NodeProto& node : nodes)
        {
            for (const std::string& tensorName : node.input())
            {
                if (constantTensors.contains(tensorName))
                {
                    constantTensorsInOrder.push_back(tensorName);
                }
            }
            for (const std::string& tensorName : node.output())
            {
                if (constantTensors.contains(tensorName))
                {
                    constantTensorsInOrder.push_back(tensorName);
                }
            }
        }

        for (auto& name : constantTensorsInOrder)
        {
            outputFile << GetSanitizedGraphVizIdentifier(name);
            outputFile << '\n';
        }
    }

    // Write operator nodes.
    outputFile << operatorsComment;
    outputFile << operatorNodeStyle;

    // Enumerate all nodes, generating sanitized names (avoid illegal characters in GraphViz identifier names).
    for (int nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex)
    {
        const onnx::NodeProto& node = nodes[nodeIndex];
        const std::string& nodeName = node.name();
        const std::string& operatorTypeName = node.op_type();

        std::string sanitizedNodeName;
        if (nodeName.empty())
        {
            // Just use current node index for unique name.
            sanitizedNodeName = std::to_string(nodeIndex);
        }
        else
        {
            sanitizedNodeName = GetSanitizedGraphVizIdentifier(nodeName);
        }

        outputFile << sanitizedNodeName << " [label=\"" << GetSanitizedGraphVizLabel(operatorTypeName) << "\"]\n";
        sanitizedNodeNames.push_back(std::move(sanitizedNodeName));
    }

    // Write all edges.
    outputFile << edgesComment;
    outputFile << intermediateNodeStyle;

    auto writeEdge = [&](std::string_view sanitizedNodeName, std::string_view tensorName, bool isInput) -> void
    {
        if (!showConstantTensors && constantTensors.contains(tensorName))
        {
            return; // Don't show constant tensors, which can be noisy.
        }
        auto sanitizedTensorName = GetSanitizedGraphVizIdentifier(tensorName);

        // Change the edge direction depending on whether input or output edge.
        std::string_view first = sanitizedNodeName;
        std::string_view second = sanitizedTensorName;
        if (isInput)
        {
            std::swap(first, second);
        }
        
        outputFile << first << " -> " << second << "\n";
    };

    for (int nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex)
    {
        const onnx::NodeProto& node = nodes[nodeIndex];
        std::string_view sanitizedNodeName = sanitizedNodeNames[nodeIndex];

        for (const std::string& tensorName : node.input())
        {
            writeEdge(sanitizedNodeName, tensorName, true);
        }
        for (const std::string& tensorName : node.output())
        {
            writeEdge(sanitizedNodeName, tensorName, false);
        }
    }

    outputFile << footer;
}

void LoadModel(
    _In_z_ wchar_t const* inputFilename,
    bool shouldZeroModelValues,
    /*inout*/ onnx::ModelProto& model
    )
{
    FileType inputFileType  = GetFileType(std::wstring_view(inputFilename));

    // Read the input file.

    bool succeeded = false;
    if (inputFileType == FileType::Text)
    {
        std::string modelString = ReadBinaryFile<std::string>(inputFilename);

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
    else if (inputFileType != FileType::Unknown)
    {
        throw std::invalid_argument("File type is not supported for input.");
    }
    else
    {
        throw std::invalid_argument("Unknown input graph file extension.");
    }

    if (!succeeded)
    {
        throw std::ios::failure("Could not parse input graph file.");
    }

    if (shouldZeroModelValues)
    {
        ZeroModelTensorWeights(/*inout*/ model);
    }
}

void StoreModel(
    _In_z_ wchar_t const* outputFilename,
    onnx::ModelProto const& model
    )
{
    // Write the output file (either another model or directory of tensors).

    FileType outputFileType = GetFileType(std::wstring_view(outputFilename));

    bool succeeded = false;
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
          || outputFileType == FileType::CommaSeparatedValue
          || outputFileType == FileType::RawData)
    {
        // enumerate all the tensor initializers, and dump their contents.

        std::wstring initialFileName(outputFilename);
        std::wstring currentFileName(outputFilename);
        size_t substitutionOffset = initialFileName.find('*', 0);
        if (substitutionOffset != std::wstring::npos)
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
            // Someone changed the data type enum in protobuf to a raw int32, which forced manual casting :/.
            onnx::TensorProto::DataType dataType = static_cast<onnx::TensorProto::DataType>(onnxTensor.data_type());

            std::vector<int32_t> dimensions;
            for (auto v : onnxTensor.dims())
            {
                dimensions.push_back(static_cast<int32_t>(v));
            }

            std::wstring name = g_converterToUtf8.from_bytes(onnxTensor.name());
            currentFileName.assign(initialFileName);
            currentFileName.insert(substitutionOffset, name);

            PrintTensorInfo(ToUtf8Char(onnxTensor.name()), currentFileName, dimensions, dataType);

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
                    std::vector<std::byte> arrayByteData = GetOnnxTensorRawByteData(onnxTensor);
                    WriteBinaryFile(currentFileName.c_str(), arrayByteData);
                }
                break;

            case FileType::CommaSeparatedValue:
                {
                    std::u8string text;
                    std::vector<std::byte> arrayByteData = GetOnnxTensorRawByteData(onnxTensor);
                    WriteCsv(arrayByteData, static_cast<onnx::TensorProto::DataType>(onnxTensor.data_type()), /*shouldPrintRawBytes*/false, /*out*/ text);
                    WriteBinaryFile(currentFileName.c_str(), text);
                }
                break;

            case FileType::NumPyArray:
                {
                    std::vector<std::byte> fileData;
                    std::vector<std::byte> arrayByteData = GetOnnxTensorRawByteData(onnxTensor);
                    WriteNpy(arrayByteData, static_cast<onnx::TensorProto::DataType>(onnxTensor.data_type()), dimensions, /*out*/ fileData);
                    WriteBinaryFile(currentFileName.c_str(), fileData);
                }
                break;

            default:
                assert(false);
                // Switch statement could not have been entered due to `if` above.
            }
        }
    }
    else if (outputFileType == FileType::GraphVizDot)
    {
        std::ofstream outputFile(outputFilename, std::ios::out);
        ConvertOnnxToGraphViz(model, /*showConstantTensors*/ false, outputFile);
        succeeded = true;

    }
    else if (outputFileType != FileType::Unknown)
    {
        throw std::invalid_argument("File type is not supported for output.");
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
    std::u8string_view pixelFormatString;
    WICPixelFormatGUID const& guid;
    uint8_t channelCount;
    uint8_t bytesPerChannel; // Only accepts homogenous channels.
    onnx::TensorProto::DataType dataType;
};

constexpr PixelFormatAttributes g_pixelFormatAttributes[] =
{
    {u8"gray8", GUID_WICPixelFormat8bppGray, 1, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {u8"pal8", GUID_WICPixelFormat8bppIndexed, 1, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {u8"b8g8r8", GUID_WICPixelFormat24bppBGR, 3, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {u8"r8g8b8", GUID_WICPixelFormat24bppRGB, 3, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {u8"b8g8r8a8", GUID_WICPixelFormat32bppBGRA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {u8"r8g8b8a8", GUID_WICPixelFormat32bppPRGBA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {u8"pb8g8r8a8", GUID_WICPixelFormat32bppPBGRA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {u8"pr8g8b8a8", GUID_WICPixelFormat32bppPRGBA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {u8"b32g32r32", GUID_WICPixelFormat32bppPRGBA, 4, 1, onnx::TensorProto::DataType::TensorProto_DataType_UINT8},
    {u8"r32g32b32x32", GUID_WICPixelFormat128bppRGBFloat, 4, 4, onnx::TensorProto::DataType::TensorProto_DataType_FLOAT},
};

bool ResolvePixelFormat(
    std::u8string_view pixelFormatString,
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
    _Out_ std::u8string_view& pixelFormatString,
    _Out_ uint32_t& channelCount,
    _Out_ uint32_t& bytesPerChannel,
    _Out_ onnx::TensorProto::DataType& dataType
    )
{
    pixelFormatString = std::u8string_view{};
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
    _Out_ std::u8string_view& pixelFormatString,
    _Out_ WICPixelFormatGUID const*& pixelFormatGuid,
    _Out_ uint32_t& bytesPerChannel
    )
{
    pixelFormatString = std::u8string_view{};
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
        *destination++ = static_cast<uint8_t>(*recastSource);
        source += sourceElementByteStride;
    }
}

// Downcasts a given element type to uint8 (e.g. for pixel imagery).
template <typename T, size_t sourceElementByteStride = sizeof(T)>
void ConvertElementTypeToUInt8Clamped(
    _In_reads_bytes_(elementCount * sourceElementByteStride) uint8_t const* source,
    _Out_writes_(elementCount) uint8_t* destination,
    size_t elementCount
    )
{
    // std::copy gives warnings about casting, but we explicitly do want the cast, even if there is bit loss.
    for (; elementCount != 0; --elementCount)
    {
        T const* recastSource = reinterpret_cast<T const*>(source);
        *destination++ = clamp_cast<uint8_t, T>(*recastSource);
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

void ConvertElementTypeToUInt8Clamped(
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
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:      ConvertElementTypeToUInt8Clamped<float>   (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:     ConvertElementTypeToUInt8Clamped<double>  (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:       ConvertElementTypeToUInt8Clamped<bool>    (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:      ConvertElementTypeToUInt8Clamped<uint8_t> (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:       ConvertElementTypeToUInt8Clamped<int8_t>  (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:     ConvertElementTypeToUInt8Clamped<uint16_t>(source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:      ConvertElementTypeToUInt8Clamped<int16_t> (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:     ConvertElementTypeToUInt8Clamped<uint32_t>(source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:      ConvertElementTypeToUInt8Clamped<int32_t> (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:     ConvertElementTypeToUInt8Clamped<uint64_t>(source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:      ConvertElementTypeToUInt8Clamped<int64_t> (source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:  ConvertElementTypeToUInt8Clamped<float, sizeof(float)*2>(source.data(), destination.data(), sourceElementCount); break;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128: ConvertElementTypeToUInt8Clamped<double, sizeof(double)*2>(source.data(), destination.data(), sourceElementCount); break;
    // case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16
    default: throw std::ios::failure("Unsupported data type in tensor.");
    }
}

void LoadImageData(
    _In_z_ wchar_t const* inputFilename, // Alternately could specify a span<const uint8_t>.
    std::u8string_view pixelFormatString,
    onnx::TensorProto::DataType& dataType,
    /*out*/ std::vector<int32_t>& dimensions,
    /*out*/ std::vector<std::byte>& pixelBytes
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
    THROW_IF_FAILED(pixelSource->CopyPixels(&rect, rowByteStride, bufferByteSize, OUT ToUChar(pixelBytes.data())));

    dimensions.assign({1, int32_t(channelCount), int32_t(height), int32_t(width)});
}

void StoreImageData(
    span<const uint8_t> pixelBytes,
    std::u8string_view pixelFormatString, // currently only supports "b8g8r8".
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
    if (dimensions.empty())
    {
        throw std::invalid_argument("Dimensions are empty. They must be at least 1D for images.");
    }
    std::vector<int32_t> reshapedDimensions;
    if (dimensions.size() < 2)
    {
        reshapedDimensions.assign(dimensions.begin(), dimensions.end());
        reshapedDimensions.insert(reshapedDimensions.begin(), 2 - reshapedDimensions.size(), 1);
        dimensions = reshapedDimensions;
    }
    // TODO: Support non-8bit pixel types.
    if (pixelFormatString != u8"b8g8r8")
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
        ConvertElementTypeToUInt8Clamped(dataType, pixelBytes, /*out*/ pixelBytesBuffer);
        pixelBytes = pixelBytesBuffer;
        dataType = onnx::TensorProto::DataType::TensorProto_DataType_UINT8;
    }

    WICPixelFormatGUID const* resolvedPixelFormatGuid = nullptr;
    const uint32_t channelCount = dimensions.size() >= 3 ? dimensions[dimensions.size() - 3] : 1;
    uint32_t bytesPerChannel;
    if (!ResolvePixelFormat(channelCount, dataType, /*out*/ pixelFormatString, /*out*/ resolvedPixelFormatGuid, /*out*/ bytesPerChannel))
    {
        printf("Channel count = %d\n", channelCount);
        throw std::invalid_argument("Pixel format is not supported for writing. Verify the channel layout.");
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
    /*out*/ std::vector<std::byte>& arrayByteData
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
        std::u8string tokenU8 = ToUtf8String({ token.begin(), token.size() });
        std::from_chars(ToChar(tokenU8.data()), ToChar(tokenU8.data() + tokenU8.size()), /*out*/ value);
        return value;
    };

    if (equals(tokens.front(), std::wstring_view(L"zeros"))) // (not "zeroes" which is a verb)
    {
        // The default getter already returns zeros.
        WriteTensorValues(/*out*/ arrayByteData, dataType, valueUnion);
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
        WriteTensorValues(/*out*/ arrayByteData, dataType, valueUnion);
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
        WriteTensorValues(/*out*/ arrayByteData, dataType, valueUnion);
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
        WriteTensorValues(/*out*/ arrayByteData, dataType, getter);
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

        WriteTensorValues(/*out*/ arrayByteData, dataType, getter);
    }
    else
    {
        throw std::invalid_argument("Not a valid mode for generate().");
    }
}

void MakeTensor(
    span<std::byte const> byteData,
    onnx::TensorProto::DataType dataType,
    span<int32_t const> dimensions,
    std::u8string_view name,
    _Inout_ onnx::TensorProto& onnxTensor
    )
{
    // Write name, tensor element type, dimensions, and raw byte data.
    onnxTensor.set_name(ToChar(name.data()), name.size());

    for (auto d : dimensions)
    {
        onnxTensor.add_dims(d);
    }

    onnxTensor.set_data_type(dataType);
    onnxTensor.set_raw_data(byteData.data(), byteData.size());
}

void LoadTensor(
    _In_z_ wchar_t const* inputFilename,
    span<int32_t const> dimensions,
    onnx::TensorProto::DataType dataType,
    HalfOpenRangeUint32 rowRange,           // matters for CSV files
    HalfOpenRangeUint32 columnRange,        // matters for CSV files
    std::u8string_view pixelFormatString,   // matters for image files
    std::u8string_view channelLayoutString, // matters for image files
    bool shouldNormalizeValues,
    double scale, // default = 1.0
    /*inout*/ onnx::TensorProto& tensor
)
{
    FileType inputFileType = GetFileType(std::wstring_view(inputFilename));

    // Set defaults.
    if (dataType == onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED)
    {
        dataType = onnx::TensorProto::DataType::TensorProto_DataType_FLOAT;
    }

    if (channelLayoutString.empty())
    {
        channelLayoutString = u8"nchw";
    }

    std::vector<int32_t> resolvedDimensions(dimensions.begin(), dimensions.end());

    if (AreDimensionsSpecified(dimensions))
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
        std::string modelString = ReadBinaryFile<std::string>(inputFilename);
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
        std::vector<std::byte> arrayByteData = ReadBinaryFile(inputFilename);
        resolvedDimensions = ResolveUnspecifiedDimensions(dimensions, arrayByteData, dataType);

        MakeTensor(arrayByteData, dataType, resolvedDimensions, u8"", /*out*/ tensor);
    }
    else if (inputFileType == FileType::CommaSeparatedValue)
    {
        std::u8string text = ReadBinaryFile<std::u8string>(inputFilename);
        std::vector<std::byte> arrayByteData;

        ReadCsv(text, dataType, rowRange, columnRange, /*out*/ arrayByteData);
        resolvedDimensions = ResolveUnspecifiedDimensions(dimensions, arrayByteData, dataType);
        MakeTensor(arrayByteData, dataType, resolvedDimensions, u8"", /*out*/ tensor);
    }
    else if (inputFileType == FileType::NumPyArray)
    {
        std::vector<std::byte> fileData = ReadBinaryFile(inputFilename);
        std::vector<std::byte> arrayByteData;

        ReadNpy(fileData, /*out*/ dataType, /*out*/ resolvedDimensions, /*out*/ arrayByteData);
        MakeTensor(arrayByteData, dataType, resolvedDimensions, u8"", /*out*/ tensor);
    }
    else if (inputFileType == FileType::Image)
    {
        std::vector<std::byte> pixelBytes;
        LoadImageData(
            inputFilename,
            pixelFormatString,
            /*out*/ dataType, // Ignore the passed data type, using the image's data type instead.
            /*out*/ resolvedDimensions,
            /*out*/ pixelBytes
        );

        RearrangeChannels(dataType, resolvedDimensions, u8"nhwc", channelLayoutString, /*inout*/ pixelBytes);
        MakeTensor(pixelBytes, dataType, resolvedDimensions, u8"", /*out*/ tensor);
    }
    else if (inputFileType == FileType::TensorGenerator)
    {
        std::vector<std::byte> arrayByteData;
        GenerateTensorSequence(inputFilename, dataType, dimensions, /*out*/ arrayByteData);
        MakeTensor(arrayByteData, dataType, resolvedDimensions, u8"", /*out*/ tensor);
    }
    else
    {
        throw std::invalid_argument("Unknown input tensor file extension.");
    }

    if (!succeeded)
    {
        throw std::ios::failure("Could not parse input tensor file.");
    }

    std::vector<std::byte> adjustedTensorByteData;
    auto getArrayByteData = [&]()
    {
        if (adjustedTensorByteData.empty())
        {
            adjustedTensorByteData = GetOnnxTensorRawByteData(tensor);
        }
    };

    // TODO: Move this between load and store functions.
    // TODO: If the output data type has a wider range than the input data type,
    // then upcast it first. Otherwise we might get an output array of all zeros.
    // TODO: Add inputDataType and outputDataType.
    // TODO: Error if PNG and outputDataType != uint8.
    double prebias = 0.0;
    if (shouldNormalizeValues)
    {
        getArrayByteData();
        std::pair<double, double> range = ArrayMinMax(static_cast<onnx::TensorProto::DataType>(tensor.data_type()), adjustedTensorByteData);
        double totalRange = (range.second - range.first);
        prebias = -range.first;
        scale *= (1.0 / totalRange);
    }

    if (scale != 1.0)
    {
        getArrayByteData();
        RescaleArray(static_cast<onnx::TensorProto::DataType>(tensor.data_type()), prebias, scale, /*inout*/ adjustedTensorByteData);
    }

    if (!adjustedTensorByteData.empty())
    {
        tensor.set_raw_data(adjustedTensorByteData.data(), adjustedTensorByteData.size());
    }
}

void StoreTensor(
    std::u8string_view channelLayoutString, // matters for image files
    bool shouldPrintRawBytes,               // for printing CSV
    _In_z_ wchar_t const* outputFilename,
    /*inout*/ onnx::TensorProto const& tensor
    )
{
    FileType outputFileType = GetFileType(std::wstring_view(outputFilename));

    // Read the data type and dimensions back from the tensor.
    onnx::TensorProto::DataType dataType = onnx::TensorProto::DataType(tensor.data_type());
    std::vector<int32_t> resolvedDimensions;
    for (auto v : tensor.dims())
    {
        resolvedDimensions.push_back(static_cast<int32_t>(v));
    }

    // Print details.
    PrintTensorInfo(ToUtf8Char(tensor.name()), L"", resolvedDimensions, dataType);

    std::vector<std::byte> arrayByteData;
    auto getArrayByteData = [&]()
    {
        if (arrayByteData.empty())
        {
            arrayByteData = GetOnnxTensorRawByteData(tensor);
        }
    };

    if (channelLayoutString.empty())
    {
        channelLayoutString = u8"nchw";
    }

    bool succeeded = true;
    if (outputFileType == FileType::Text)
    {
        std::string modelString;
        succeeded = google::protobuf::TextFormat::PrintToString(tensor, &modelString);
        if (succeeded)
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
        getArrayByteData();
        WriteBinaryFile(outputFilename, arrayByteData);
    }
    else if (outputFileType == FileType::CommaSeparatedValue)
    {
        getArrayByteData();
        std::u8string text;
        WriteCsv(arrayByteData, onnx::TensorProto::DataType(tensor.data_type()), shouldPrintRawBytes, /*out*/ text);
        WriteBinaryFile(outputFilename, text);
    }
    else if (outputFileType == FileType::NumPyArray)
    {
        getArrayByteData();
        std::vector<std::byte> fileData;
        WriteNpy(arrayByteData, static_cast<onnx::TensorProto::DataType>(tensor.data_type()), resolvedDimensions, /*out*/ fileData);
        WriteBinaryFile(outputFilename, fileData);
    }
    else if (outputFileType == FileType::Image)
    {
        getArrayByteData();
        std::vector<std::byte> pixelBytes(arrayByteData.data(), arrayByteData.data() + arrayByteData.size());
        RearrangeChannels(
            dataType,
            resolvedDimensions,
            channelLayoutString,
            u8"nhwc",
            /*inout*/ pixelBytes
            );
        StoreImageData(
            reinterpret_span<const uint8_t>(pixelBytes),
            u8"b8g8r8",
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
    std::cout << "Onnx2Text 2018-07-19..2022-06-29 FDwR\r\n"
                 "\r\n"
                 "Converts:\r\n"
                 "    - binary ONNX model file to proto text and back.\r\n"
                 "    - tensors to/from prototext/ONNX/CSV/PNG/NPY and vice versa.\r\n"
                 "    - generated values to output tensor (ones, zeros, iota series, random).\r\n"
                 "    - model to directory of tensor files.\r\n"
                 "\r\n"
                 "Example usage:\r\n"
                 "\r\n"
                 "    Onnx2Text.exe -options inputfile outputfile\r\n"
                 "\r\n"
                 "    Convert model to/from ONNX binary protobuf and prototxt format:\r\n"
                 "        Onnx2Text input.onnx output.prototxt\r\n"
                 "        Onnx2Text input.prototxt output.onnx\r\n"
                 "\r\n"
                 "    Write GraphViz dot file (download GraphViz separately):\r\n"
                 "        Onnx2Text input.onnx output.dot\r\n"
                 "        dot.exe output.dot -Tpng -O   (or -Tsvg)\r\n"
                 "\r\n"
                 "    Zero weights in ONNX binary protobuf:\r\n"
                 "        Onnx2Text -zeromodelvalues input.onnx output.onnx\r\n"
                 "\r\n"
                 "    Export model from ONNX protobuf to NumPy tensors/data files:\r\n"
                 "        Onnx2Text resnet50.onnx x:\\resnet_*.npy\r\n"
                 "        Onnx2Text squeezenet.onnx z:\\folder\\*_weight.dat\r\n"
                 "\r\n"
                 "    Convert tensor between ONNX protobuf, CSV, raw data, numpy, PNG:\r\n"
                 "        Onnx2Text input.onnxtensor output.csv\r\n"
                 "        Onnx2Text input.pb output.png\r\n"
                 "        Onnx2Text -datatype uint8 -dimensions 224,224 Foo.csv Foo.dat\r\n"
                 "        Onnx2Text input.npy output.onnxtensor\r\n"
                 "\r\n"
                 "    Generate tensor from randomness:\r\n"
                 "        Onnx2Text -dimensions 3,4 -datatype float16 generate(random,1,24) output.onnxtensor\r\n"
                 "\r\n"
                 "Parameters:\r\n"
                 "        inputfile - either a graph or tensor (see file types below)\r\n"
                 "       outputfile - either a graph or tensor (see file types below)\r\n"
                 "          -tensor - specifies the input file is a tensor\r\n"
                 "                    (only needed if can't tell from file extension, like with .pb).\r\n"
                 "           -graph - specifies the input file is a graph model\r\n"
                 "                    (only needed if can't tell from file extension, like with .pb).\r\n"
                 "      -dimensions - explicit tensor dimensions for .csv or .dat file which do not\r\n"
                 "                    store dimensions internally. Defaults to 1D otherwise.\r\n"
                 "                    Pass \"[]\" to indicate a 0D scalar.\r\n"
                 "        -datatype - tensor element type (float16,float32,float64,int8,uint8,int16,\r\n"
                 "                    uint16,int32,uint32,int64,uint64,bool8).\r\n"
                 " -zeromodelvalues - zero any tensor values (clears model initializer weights)\r\n"
                 "          -rawhex - display as raw hexadecimal when writing .csv\r\n"
                 "             -row - single row or range for .csv\r\n"
                 "          -column - single column or range for .csv\r\n"
                 "           -scale - scale tensor values during conversion\r\n"
                 "    -inversescale - scale tensor values during conversion by reciprocal (e.g. 255 means 1/255)\r\n"
                 " -normalizevalues - should normalize values in tensor 0 to 1\r\n"
                 "     -information - display more verbose file information (output file is not needed)\r\n"
                 //"   -channellayout - either nchw or nhwc (needed for images)\r\n" // Not functional enough to enable yet.
                 "\r\n"
                 "File types:\r\n"
                 "    Model file types:\r\n"
                 "        .onnx - Open Neural Exchange model protobuf\r\n"
                 "        .pb - Google Protobuf (with -graph)\r\n"
                 "        .txt/.prototxt - Protobuf text\r\n"
                 "    Tensor file types:\r\n"
                 "        .onnxtensor - Open Neural Exchange tensor\r\n"
                 "        .pb - Google Protobuf (with -tensor)\r\n"
                 "        .csv - Comma Separate Value\r\n"
                 "        .png - Image (Portable Network Graphics)\r\n"
                 "        .jpg - Image (Joint Photographic Experts Group)\r\n"
                 "        .npy - NumPyArray single tensor\r\n"
                 "        .dat/.bin - Raw binary array (no header)\r\n"
                 "        generators - pseudo input filename:\r\n"
                 "            generate(ones) - all ones [1,1,1,1...]\r\n"
                 "            generate(zeros) - all zeros [0,0,0,0...]\r\n"
                 "            generate(values,3) - specific value [3,3,3,3...]\r\n"
                 "            generate(iota,1,2) - increasing sequence [1,3,5...]\r\n"
                 "            generate(random,1,100) - random values between min/max [31,56,2,69...]\r\n"
                 ;
}

void ReadOpenHalfRange(std::u8string_view text, _Out_ HalfOpenRangeUint32& range)
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

char8_t const* g_conversionModeNames[3] =
{
    u8"Unknown",
    u8"Tensor",
    u8"Graph",
};
static_assert(std::extent<decltype(g_conversionModeNames)>::value == uint32_t(ConversionMode::Total));

ConversionMode GetConversionModeFromFileType(FileType fileType)
{
    switch (fileType)
    {
    case FileType::Unknown: return ConversionMode::Unknown;
    case FileType::OnnxModel: return ConversionMode::Graph;
    case FileType::GoogleProtobuf: return ConversionMode::Tensor; // Assume .pb is tensor for ONNX model zoo/backend usage, even though .pb is technically an agnostic container type.
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
    std::u8string pixelFormatString, channelLayoutString;
    ConversionMode conversionMode = ConversionMode::Unknown;
    std::vector<int32_t> dimensions = {-1};
    onnx::TensorProto::DataType dataType = onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
    HalfOpenRangeUint32 rowRange = {}, columnRange = {};
    double scale = 1.0;
    bool shouldNormalizeValues = false;
    bool shouldPrintRawBytes = false;
    bool shouldZeroModelValues = false;
    bool displayMoreInformation = false;

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

                std::u8string s = ToUtf8String(argv[i]);
                ReadCsv(s, /*out*/dimensions);
            }
            else if (argument == L"-datatype")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Valid data type expected: -datatype float32 (uint8, int32, int16, bool8, float64...)");
                }

                std::u8string s = ToUtf8String(argv[i]);
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
                std::u8string s = ToUtf8String(argv[i]);
                ReadOpenHalfRange(s, /*out*/ rowRange);
            }
            else if (argument == L"-column")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Column number expected: -column 2 / -column 2,100");
                }
                std::u8string s = ToUtf8String(argv[i]);
                ReadOpenHalfRange(s, /*out*/ columnRange);
            }
            else if (argument == L"-rawhex")
            {
                shouldPrintRawBytes = true;
            }
            else if (argument == L"-zeromodelvalues")
            {
                shouldZeroModelValues = true;
            }
            else if (argument == L"-scale")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Scale expects a value");
                }
                std::u8string s = ToUtf8String(argv[i]);
                scale = atof(reinterpret_cast<char*>(s.data()));
            }
            else if (argument == L"-inversescale")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Inverse scale expects a value");
                }
                std::u8string s = ToUtf8String(argv[i]);
                scale = 1.0f / atof(reinterpret_cast<char*>(s.data()));
            }
            else if (argument == L"-normalizevalues")
            {
                shouldNormalizeValues = true;
            }
            else if (argument == L"-information" || argument == L"-info")
            {
                displayMoreInformation = true;
            }
            else if (argument == L"-help")
            {
                PrintUsage();
                return EXIT_SUCCESS;
            }
            #if 0 // Not functional enough to enable yet. todo: Rename to "layout", and ensure image conversion works.
            else if (argument == L"-channellayout")
            {
                if (++i >= argc)
                {
                    throw std::invalid_argument("Channel layout string expected: -channellayout nchw (or nhwc)");
                }
                channelLayoutString = ToUtf8String(argv[i]);
            }
            #endif
            else
            {
                char buffer[256];
                sprintf_s(buffer, std::size(buffer), "Unknown argument: %S", argument.data());
                throw std::invalid_argument(std::string(buffer));
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
    if (outputFilename.empty() && !displayMoreInformation)
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

    printf("Input filename:  %ls\r\n", inputFilename.c_str());
    if (!outputFilename.empty())
    {
        printf(
            "Output filename: %ls\r\n"
            "Conversion mode: %hs\r\n"
            ,
            outputFilename.c_str(),
            ToChar(g_conversionModeNames[uint32_t(conversionMode)])
        );
    }
    printf("\r\n");

    if (conversionMode == ConversionMode::Tensor)
    {
        if (shouldZeroModelValues)
        {
            throw std::invalid_argument("\"-zeromodelvalues\" passed but input was not a model.");
        }

        onnx::TensorProto tensor;
        LoadTensor(
            inputFilename.c_str(),
            dimensions,
            dataType,
            rowRange,
            columnRange,
            pixelFormatString,
            channelLayoutString,
            shouldNormalizeValues,
            scale,
            /*inout*/ tensor
        );

        if (!outputFilename.empty())
        {
            StoreTensor(
                channelLayoutString,
                shouldPrintRawBytes,
                outputFilename.c_str(),
                tensor
            );
        }
        else // At least display information.
        {
            onnx::TensorProto::DataType resolvedDataType = onnx::TensorProto::DataType(tensor.data_type());
            std::vector<int32_t> resolvedDimensions;
            for (auto v : tensor.dims())
            {
                resolvedDimensions.push_back(static_cast<int32_t>(v));
            }
            PrintTensorInfo(ToUtf8Char(tensor.name()), L"", resolvedDimensions, resolvedDataType);
        }
    }
    else if (conversionMode == ConversionMode::Graph)
    {
        if (AreDimensionsSpecified(dimensions))
        {
            throw std::invalid_argument("\"-dimensions\" may only be specified for \"-tensor\" conversion.");
        }
        if (dataType != onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED)
        {
            throw std::invalid_argument("\"-datatype\" may only be specified for \"-tensor\" conversion.");
        }

        onnx::ModelProto model;
        LoadModel(inputFilename.c_str(), shouldZeroModelValues, /*inout*/ model);
        if (displayMoreInformation)
        {
            DisplayModelInformation(model);
        }
        if (!outputFilename.empty())
        {
            StoreModel(outputFilename.c_str(), model);
        }
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

    // return EXIT_SUCCESS;
}
