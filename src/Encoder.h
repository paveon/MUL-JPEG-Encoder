#ifndef MUL_ENCODER_H
#define MUL_ENCODER_H

#include <cstdint>
#include <cstdio>
#include <vector>
#include <array>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <optional>
#include <thread>
#include <immintrin.h>
#include <fftw3.h>

#include "EncoderDefaultValues.h"


template<class T>
struct AlignedAllocator {
    typedef T value_type;

    AlignedAllocator() = default;

    template<class U>
    constexpr AlignedAllocator(const AlignedAllocator<U> &) noexcept {
    }

    T *allocate(std::size_t n) {
        if (n > std::size_t(-1) / sizeof(T))
            throw std::bad_alloc();
        if (auto p = static_cast<T *>(_mm_malloc(n * sizeof(T), 32)))
            return p;

        throw std::bad_alloc();
    }

    void deallocate(T *p, std::size_t) noexcept {
        _mm_free(p);
    }
};

template<class T, class U>
bool operator==(const AlignedAllocator<T> &, const AlignedAllocator<U> &) {
    return true;
}

template<class T, class U>
bool operator!=(const AlignedAllocator<T> &, const AlignedAllocator<U> &) {
    return false;
}


struct Pixel {
    u_char r, g, b;
};

enum Component {
    Y = 0,
    Cb = 1,
    Cr = 2
};

enum class DownsampleMode {
    None, X, XY
};

const uint8_t MarkerBegin = 0xFF;

const std::array<uint8_t, 9> JFIF_MarkerValues{0xD8, 0xE0, 0xDB, 0xC0, 0xC4, 0xDA, 0xD9, 0xFE, 0xC1};
enum JFIF_Marker {
    SOI, APP0, DQT, SOF0, DHT, SOS, EOI, COM, SOF1
};


struct ElementCode {
    uint16_t m_Value{};
    uint8_t m_BitCount{};

    ElementCode() = default;

    ElementCode(uint16_t value, uint8_t bitCount) : m_Value(value), m_BitCount(bitCount) {}
};


struct BitBuffer {
    std::vector<uint8_t> m_Data;
    uint32_t m_Index = 0;
    uint8_t m_BitIndex = 0;

    explicit BitBuffer(size_t size) : m_Data(size) {}

    template<size_t SIZE>
    void Push(const std::array<uint8_t, SIZE>& data) {
        assert(m_BitIndex == 0);
        for (size_t i = 0; i < SIZE; ++i) {
            m_Data[m_Index++] = data[i];
        }
    }

    void Push(const std::vector<uint8_t>& data) {
        assert(m_BitIndex == 0);
        for (unsigned char i : data) {
            m_Data[m_Index++] = i;
        }
    }

    void Push(uint8_t byte) {
        if (m_BitIndex == 0) {
            m_Data[m_Index++] = byte;
        } else {
            uint8_t remaining = 8;
            uint8_t freeBits = 8u - m_BitIndex;
            uint8_t maskLength = std::min(freeBits, remaining);
            remaining -= maskLength;
            uint8_t mask = ((1u << maskLength) - 1u) << remaining;
            m_Data[m_Index++] += ((byte & mask) >> remaining);
            mask = (1u << remaining) - 1u;
            m_Data[m_Index] = (byte & mask) << freeBits;
        }
    }

    void Push(const ElementCode &code) {
        uint8_t remaining = code.m_BitCount;
        while (remaining > 0) {
            uint8_t freeBits = 8u - m_BitIndex;
            uint8_t maskLength = std::min(freeBits, remaining);
            uint8_t leftShift = freeBits - maskLength;
            remaining -= maskLength;
            uint16_t mask = ((1u << maskLength) - 1u) << remaining;

            m_Data[m_Index] += ((code.m_Value & mask) >> remaining) << leftShift;
            m_BitIndex += maskLength;
            if (m_BitIndex >= 8u) {
                m_BitIndex %= 8u;

                if (m_Data[m_Index] == MarkerBegin) {
                    // Specify that this is not a marker by a zero byte (is ignored)
                    m_Index++;
                    m_Data[m_Index] = 0x00;
                }
                m_Index++;
            }
        }
    }

    void PushMarker(JFIF_Marker marker, uint16_t length) {
        Push(0xFF);
        Push(JFIF_MarkerValues[marker]);
        if (length > 0) {
            length += 2;
            Push(length >> 8u);
            Push(length & 0xFF);
        }
    }

    void FlushRemainingBits() {
        if (m_Data[m_Index] > 0) {
            m_Data[m_Index++] = 0xFF;
            m_BitIndex = 0;
        }
    }

    void OutputToFile(const char *filepath) {
        std::ofstream file(filepath, std::ios::out | std::ios::binary);
        if (!file) {
            assert(false);
        }
        file.write(reinterpret_cast<char *>(m_Data.data()), m_Index);
    }
};


class Encoder {
public:
    enum class OutputMode {
        GRAYSCALE,
        RGB
    };

private:
    struct EncodingMetadata {
        int32_t width, height, channels;
        size_t luminanceWidthPadded;
        size_t luminanceHeightPadded;
        size_t chrominanceHeight;
        size_t chrominanceWidth;
        size_t luminanceBlocksX;
        size_t luminanceBlocksY;
        size_t chrominanceBlocksX;
    };

    static float rgbToY(float r, float g, float b) { return 0.299f * r + 0.587f * g + 0.114f * b - 128.0f; }

    static float rgbToCb(float r, float g, float b) { return -0.1687f * r - 0.3313f * g + 0.5f * b; }

    static float rgbToCr(float r, float g, float b) { return 0.5f * r - 0.4187f * g - 0.0813f * b; }

    const size_t m_ThreadCount = std::thread::hardware_concurrency();

    uint8_t m_QualityLevel = DefaultQualityLevel;

    DownsampleMode m_Downsampling = DownsampleMode::None;
    bool m_DownsampleX = false;
    bool m_DownsampleY = false;

    std::array<uint8_t, BlockSize> m_ZigZagLuminanceQT{};
    std::array<uint8_t, BlockSize> m_ZigZagChrominanceQT{};

    std::array<std::array<uint8_t, 16>, 6> m_CodeCounts{};
    std::array<std::vector<uint8_t>, 6> m_SortedCodes;

    std::array<std::array<ElementCode, 256>, 6> m_HuffmanTables;

    // Thread private data
    std::vector<std::array<uint64_t, 256>> m_ThreadHistograms;
    std::vector<std::vector<const ElementCode *>> m_ThreadBuffers;
    std::vector<std::array<int16_t, 3>> m_ThreadLastDCs;
    std::vector<std::array<int16_t, 3>> m_ThreadFirstDCs;
    std::vector<std::array<size_t, 3>> m_ThreadDcIndices;
    std::vector<std::vector<std::pair<size_t, const ElementCode *>>> m_ThreadEncodeIndices;

    /* 256bit AVX loads require 32 byte alignment */
    __attribute__((aligned(32))) std::array<float, BlockSize> m_DctCoefficients1D{};
    __attribute__((aligned(32))) std::array<float, BlockSize> m_CoefficientsFFTW{};

    std::array<std::vector<float, AlignedAllocator<float>>, 3> m_ComponentBlocks;

    ElementCode m_CodeTable[2 * MaxDCTValue];

    fftwf_plan m_ExecutionPlan;

    void DownsampleComponent(float *data, bool vertical, size_t width, size_t height);

    void WriteMetadata(BitBuffer &buffer, uint16_t width, uint16_t height, uint8_t channels, bool optimizedTables);

    void GenerateHuffmanTable(size_t idx);

    void MergeThreadHistograms(size_t channelCount);

    void GenerateOptimalHuffmanTable(size_t idx);

    void SeparateComponents(const u_char *imageData, const EncodingMetadata &info, bool parallel);

    void DCT1D(float *block, float *output);

    void DCT_FFTW(float *block);

    void EncodeThreadChunk(size_t threadID, size_t threadStep, const EncodingMetadata &info);


    int16_t ProcessBlock(size_t threadID,
                         std::vector<const ElementCode *> &buffer,
                         float *blockData,
                         Component componentType,
                         std::optional<int16_t> previousDC);

public:
    Encoder();

    ~Encoder() {
        fftwf_destroy_plan(m_ExecutionPlan);
        fftwf_cleanup();
    }

    void SetQualityLevel(uint8_t value);

    void SetDownsamplingMode(DownsampleMode mode);

    void EncodeJPEG(const char *filepath, const char *outFilepath, OutputMode mode, bool optimizeTables, bool parallel);
};


#endif //MUL_ENCODER_H
