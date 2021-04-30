#include "Encoder.h"
#include <iostream>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <future>
#include <queue>
#include <immintrin.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image.h>
#include <stb_image_write.h>


uint32_t RoundUp(uint32_t value, uint32_t multiple) { return (value + multiple - 1) & ~(multiple - 1); }

uint8_t DownsampleModeToCode(DownsampleMode mode) {
    switch (mode) {
        case DownsampleMode::X:
            return 0x21;
        case DownsampleMode::XY:
            return 0x22;
        default:
            return 0x11;
    }
}


void Encoder::SetQualityLevel(uint8_t value) {
    m_QualityLevel = std::clamp(value, (uint8_t) 1, (uint8_t) 100);

    uint32_t S = m_QualityLevel < 50 ? (5000 / m_QualityLevel) : (200 - 2 * m_QualityLevel);
    for (size_t i = 0; i < BlockSize; ++i) {
        m_ZigZagLuminanceQT[i] = std::clamp((S * DefaultLuminanceQT[ZigZagIndices[i]] + 50) / 100, 1u, 255u);
        m_ZigZagChrominanceQT[i] = std::clamp((S * DefaultChrominanceQT[ZigZagIndices[i]] + 50) / 100, 1u, 255u);
    }
}


void Encoder::SetDownsamplingMode(DownsampleMode mode) {
    m_Downsampling = mode;
    switch (m_Downsampling) {
        case DownsampleMode::X:
            m_DownsampleX = true;
            m_DownsampleY = false;
            break;
        case DownsampleMode::XY:
            m_DownsampleX = m_DownsampleY = true;
            break;
        default:
            m_DownsampleX = m_DownsampleY = false;
            break;
    }
}


void Encoder::DownsampleComponent(float *data, bool vertical, size_t width, size_t height) {
    assert(width % 2 == 0 && height % 2 == 0);

    size_t widthDownsampled = width / 2;
    size_t heightDownsampled = height / 2;
    if (vertical) {
        // 4:2:0, halved in both dimensions
        for (size_t row = 0; row < heightDownsampled; ++row) {
            float *srcRow1 = &data[width * row * 2];
            float *srcRow2 = &data[width * (row * 2 + 1)];
            for (size_t col = 0; col < widthDownsampled; ++col) {
                data[(width * row) + col] = (srcRow1[col * 2] + srcRow1[(col * 2) + 1] +
                                             srcRow2[col * 2] + srcRow2[(col * 2) + 1]) / 4.0f;
            }
        }
    } else {
        // 4:2:2, halved only in horizontal dimension
        for (size_t row = 0; row < height; ++row) {
            float *rowData = &data[row * width];
            for (size_t col = 0; col < widthDownsampled; ++col) {
                data[(width * row) + col] = (rowData[col * 2] + rowData[(col * 2) + 1]) / 2.0f;
            }
        }
    }
}


void Encoder::WriteMetadata(BitBuffer &buffer, uint16_t width,
                            uint16_t height, uint8_t channels, bool optimizedTables) {

    const bool chrominanceEnabled = channels == STBI_rgb;
    const uint8_t quantizationTableCount = chrominanceEnabled ? 2 : 1;

    const uint8_t headerPayloadSize = 14;
    const uint8_t majorVersion = 1;
    const uint8_t minorVersion = 1;
    const uint8_t horizontalDensity = 1;
    const uint8_t verticalDensity = 1;
    const std::array<uint8_t, 20> fileHeader{
            MarkerBegin, JFIF_MarkerValues[JFIF_Marker::SOI],
            MarkerBegin, JFIF_MarkerValues[JFIF_Marker::APP0],
            0, headerPayloadSize + 2,
            'J', 'F', 'I', 'F', 0,
            majorVersion, minorVersion,
            0,
            0, horizontalDensity, 0, verticalDensity,
            0, 0};
    buffer.Push(fileHeader);

    /* Store quantization tables metadata */
    const uint8_t LuminanceQT_ID = 0x00;
    const uint8_t ChrominanceQT_ID = 0x01;
    buffer.PushMarker(JFIF_Marker::DQT, (BlockSize + 1) * quantizationTableCount);
    buffer.Push(LuminanceQT_ID);
    buffer.Push(m_ZigZagLuminanceQT);
    if (chrominanceEnabled) {
        buffer.Push(ChrominanceQT_ID);
        buffer.Push(m_ZigZagChrominanceQT);
    }

    buffer.PushMarker(JFIF_Marker::SOF1, 6 + (3 * channels));
    // Height and width in big endian
    buffer.Push(0x08); // 8 bits per channel
    buffer.Push(height >> 8u);
    buffer.Push(height & 0xFF);
    buffer.Push(width >> 8u);
    buffer.Push(width & 0xFF);

    const uint8_t SubsamplingDisabledCode = 0x11;
    const uint8_t Y_ID = 1;
    const uint8_t Cb_ID = 2;
    const uint8_t Cr_ID = 3;
    std::array<std::array<uint8_t, 3>, 3> componentInfo{
            std::array<uint8_t, 3>{Y_ID, DownsampleModeToCode(m_Downsampling), LuminanceQT_ID},
            {Cb_ID, SubsamplingDisabledCode, ChrominanceQT_ID},
            {Cr_ID, SubsamplingDisabledCode, ChrominanceQT_ID}
    };
    buffer.Push(channels);
    for (size_t channelIdx = 0; channelIdx < channels; ++channelIdx) {
        buffer.Push(componentInfo[channelIdx]);
    }

    uint8_t huffmanTableCount = optimizedTables ? 3 : 2;
    uint16_t totalSize = 0;
    for (uint8_t tableID = 0; tableID < huffmanTableCount; tableID++) {
        totalSize += 2 + 32 + m_SortedCodes[(tableID * 2)].size() + m_SortedCodes[(tableID * 2) + 1].size();
    }

    buffer.PushMarker(JFIF_Marker::DHT, totalSize);
    for (uint8_t tableID = 0; tableID < huffmanTableCount; tableID++) {
        buffer.Push(0x00 + tableID);
        buffer.Push(m_CodeCounts[tableID * 2]);
        buffer.Push(m_SortedCodes[tableID * 2]);
        buffer.Push(0x10 + tableID);
        buffer.Push(m_CodeCounts[tableID * 2 + 1]);
        buffer.Push(m_SortedCodes[tableID * 2 + 1]);
    }

    buffer.PushMarker(JFIF_Marker::SOS, 1 + (2 * channels) + 3);
    buffer.Push(channels);
    std::array<std::array<uint8_t, 2>, 3> huffmanTableInfo{
            std::array<uint8_t, 2>{Y_ID, 0x00},
            {Cb_ID, 0x11},
            {Cr_ID, (uint8_t) (optimizedTables ? 0x22 : 0x11)}
    };
    for (size_t componentIdx = 0; componentIdx < channels; ++componentIdx) {
        buffer.Push(huffmanTableInfo[componentIdx]);
    }

    std::array<uint8_t, 3> spectral{0, 63, 0};
    buffer.Push(spectral);
}


void Encoder::GenerateHuffmanTable(size_t idx) {
    const std::array<uint8_t, 16> &codeCounts = m_CodeCounts[idx];
    const std::vector<uint8_t> &values = m_SortedCodes[idx];
    std::array<ElementCode, 256> &huffmanTable = m_HuffmanTables[idx];

    auto huffmanCode = 0;
    size_t valueIdx = 0;
    for (size_t bitLength = 1; bitLength <= codeCounts.size(); bitLength++) {
        for (auto i = 0; i < codeCounts[bitLength - 1]; i++) {
            huffmanTable[values[valueIdx]] = ElementCode(huffmanCode++, bitLength);
            valueIdx++;
        }

        huffmanCode <<= 1;
    }
}


/* Reduce 256-bit AVX float vector into single float value */
inline float hsum_floats_avx(__m256 x) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}


void Encoder::DCT1D(float *block, float *output) {
    // Compute rows first
    float tmpOutput[BlockSize] = {0.0f};
    for (size_t outRow = 0; outRow < BlockWidth; outRow++) {
        float *outputRow = &tmpOutput[outRow * BlockWidth];
        __m256 dataRow = _mm256_load_ps(&block[outRow * BlockWidth]);


        for (size_t outCol = 0; outCol < BlockWidth; outCol++) {
            const float *coeffBlock = &m_DctCoefficients1D[outCol * BlockWidth];

            __m256 coeffRow = _mm256_load_ps(coeffBlock);
            outputRow[outCol] += hsum_floats_avx(_mm256_mul_ps(dataRow, coeffRow));
        }
    }

    /* 256bit AVX loads require 32 byte alignment */
    __attribute__((aligned(32))) float colTmp[BlockWidth] = {0.0f};

    for (size_t outCol = 0; outCol < BlockWidth; outCol++) {
        // Store column in temp array for better cache locality
        for (size_t row = 0; row < BlockWidth; row++) {
            colTmp[row] = tmpOutput[(row * BlockWidth) + outCol];
        }
        __m256 dataCol = _mm256_load_ps(colTmp);

        for (size_t outRow = 0; outRow < BlockWidth; outRow++) {
            const float *coeffBlock = &m_DctCoefficients1D[outRow * BlockWidth];
            float *outValue = &output[(outRow * BlockWidth) + outCol];

            __m256 coeffRow = _mm256_load_ps(coeffBlock);
            *outValue += hsum_floats_avx(_mm256_mul_ps(dataCol, coeffRow));

        }
    }
}


void Encoder::DCT_FFTW(float *block) {
    fftwf_execute_r2r(m_ExecutionPlan, block, block);
    for (size_t i = 0; i < BlockSize; i += 8) {
        __m256 dataChunk = _mm256_load_ps(&block[i]);
        __m256 coeffChunk = _mm256_load_ps(&m_CoefficientsFFTW[i]);
        __m256 result = _mm256_mul_ps(dataChunk, coeffChunk);
        _mm256_store_ps(&block[i], result);
    }
}


int16_t Encoder::ProcessBlock(size_t threadID,
                              std::vector<const ElementCode *> &buffer,
                              float *blockData,
                              Component componentType,
                              std::optional<int16_t> previousDC) {

    // std::array<float, BlockSize> outputDCT{};
    // DCT1D(blockData, outputDCT.data());
    DCT_FFTW(blockData);

    const uint8_t *zigZagQT = componentType == Component::Y ? m_ZigZagLuminanceQT.data() : m_ZigZagChrominanceQT.data();
    ElementCode *huffmanDC = m_HuffmanTables[componentType * 2].data();
    ElementCode *huffmanAC = m_HuffmanTables[componentType * 2 + 1].data();
    uint64_t *dcHistogram = m_ThreadHistograms[(threadID * 6) + (componentType * 2)].data();
    uint64_t *acHistogram = m_ThreadHistograms[(threadID * 6) + (componentType * 2) + 1].data();

    size_t lastValueIdx = 0;
    std::array<int16_t, BlockSize> quantizedData{};
    for (size_t i = 0; i < quantizedData.size(); ++i) {
        float value = blockData[ZigZagIndices[i]];
        float quantized = value / (float) zigZagQT[i];
        quantizedData[i] = quantized + (quantized >= 0 ? 0.5f : -0.5f);
        if (quantizedData[i] != 0) {
            lastValueIdx = i;
        }
    }


    int16_t DC = quantizedData[0];
    if (previousDC.has_value()) {
        int16_t dcDiff = DC - previousDC.value();
        if (dcDiff != 0) {
            // output category code followed by the value itself
            const ElementCode *valueCode = &m_CodeTable[MaxDCTValue + dcDiff];
            const ElementCode *categoryCode = &huffmanDC[valueCode->m_BitCount];
            buffer.push_back(categoryCode);
            buffer.push_back(valueCode);
            dcHistogram[valueCode->m_BitCount]++;
        } else {
            buffer.push_back(&huffmanDC[dcDiff]);
            dcHistogram[dcDiff]++;
        }
    }

    uint8_t zeroCount = 0;
    for (size_t i = 1; i <= lastValueIdx; ++i) {
        int16_t value = quantizedData[i];
        if (value == 0) {
            zeroCount++;
            if (zeroCount == 16) {
                buffer.push_back(&huffmanAC[0xF0]);
                zeroCount = 0;
                acHistogram[0xF0]++;
            }
        } else {
            const ElementCode *valueCode = &m_CodeTable[MaxDCTValue + value];
            uint8_t huffmanIdx = (zeroCount << 4u) + valueCode->m_BitCount;
            const ElementCode *categoryCode = &huffmanAC[huffmanIdx];
            buffer.push_back(categoryCode);
            buffer.push_back(valueCode);
            zeroCount = 0;
            acHistogram[huffmanIdx]++;
        }
    }

    if (lastValueIdx < (BlockSize - 1)) {
        // Output Huffman code of EOB (0x00)
        buffer.push_back(&huffmanAC[0x00]);
        acHistogram[0x00]++;
    }

    return DC;
}


Encoder::Encoder() {
    uint8_t numBits = 1;
    int32_t mask = 1;
    for (int16_t value = 1; value < MaxDCTValue; value++) {
        if (value > mask) {
            numBits++;
            mask = (mask << 1) | 1;
        }
        m_CodeTable[MaxDCTValue - value] = ElementCode(mask - value, numBits);
        m_CodeTable[MaxDCTValue + value] = ElementCode(value, numBits);
    }

    const float SqrtCoeff = 1.0f / 1.414213562373095048801688724209f;

    // Precompute 1D DCT coefficients
    for (size_t outIdx = 0; outIdx < BlockWidth; outIdx++) {
        float lambda = outIdx == 0 ? SqrtCoeff : 1.0f;
        float *coeffRow = &m_DctCoefficients1D[outIdx * BlockWidth];

        for (size_t inIdx = 0; inIdx < BlockWidth; inIdx++) {
            float cosArg = (outIdx * M_PI * (inIdx + 0.5f)) / 8.0f;
            coeffRow[inIdx] = 0.5f * lambda * std::cos(cosArg);
        }
    }

    // Precompute adjustment coefficients for FFTW DCT
    for (size_t y = 0; y < BlockWidth; ++y) {
        float lambda1 = y == 0 ? SqrtCoeff : 1.0f;
        for (size_t x = 0; x < BlockWidth; ++x) {
            float lambda2 = x == 0 ? SqrtCoeff : 1.0f;
            m_CoefficientsFFTW[(y * BlockWidth) + x] = 0.0625f * lambda1 * lambda2;
        }
    }

    // Determine best possible FFTW plan for our DCT transformation
    float tmpInput[64];
    m_ExecutionPlan = fftwf_plan_r2r_2d(BlockWidth, BlockWidth,
                                        tmpInput, tmpInput,
                                        FFTW_REDFT10, FFTW_REDFT10,
                                        FFTW_EXHAUSTIVE);
}


void Encoder::EncodeThreadChunk(size_t threadID, size_t threadStep, const EncodingMetadata &info) {
    std::vector<const ElementCode *> &buffer = m_ThreadBuffers[threadID];
    std::array<int16_t, 3> &lastDCs = m_ThreadLastDCs[threadID];
    std::array<int16_t, 3> &firstDCs = m_ThreadFirstDCs[threadID];
    std::array<size_t, 3> &indices = m_ThreadDcIndices[threadID];

    size_t horizontalStep = m_DownsampleX ? 2 : 1;
    size_t verticalStep = m_DownsampleY ? 2 : 1;
    size_t chrominanceStep = horizontalStep * verticalStep;

    size_t yStart = threadID * threadStep;
    size_t yEnd = std::min(yStart + threadStep, info.luminanceBlocksY);
    size_t chrominanceBlockIdx = (yStart * info.luminanceBlocksX) / chrominanceStep;
    std::optional<int16_t> previousDC_Y{};
    std::optional<int16_t> previousDC_Cb{};
    std::optional<int16_t> previousDC_Cr{};
    if (threadID == 0) {
        previousDC_Y = 0;
        previousDC_Cb = 0;
        previousDC_Cr = 0;
    }

    float *yBlocks = m_ComponentBlocks[Component::Y].data();
    float *cbBlocks = m_ComponentBlocks[Component::Cb].data();
    float *crBlocks = m_ComponentBlocks[Component::Cr].data();

    size_t blockIdx = 0;
    for (size_t y = yStart; y < yEnd; y += verticalStep) {
        for (size_t x = 0; x < info.luminanceBlocksX; x += horizontalStep) {

            for (size_t yOffset = 0; yOffset < verticalStep; yOffset++) {
                for (size_t xOffset = 0; xOffset < horizontalStep; xOffset++) {
                    size_t xBlockPos = x + xOffset;
                    size_t yBlockPos = y + yOffset;
                    float *yBlock = &yBlocks[(yBlockPos * info.luminanceBlocksX + xBlockPos) * BlockSize];
                    previousDC_Y = ProcessBlock(threadID, buffer, yBlock, Component::Y, previousDC_Y);

                    if (blockIdx == 0 && (yOffset + xOffset) == 0) {
                        firstDCs[0] = previousDC_Y.value();
                    }
                }
            }

            if (info.channels > 1) {
                float *cbBlock = &cbBlocks[chrominanceBlockIdx * BlockSize];
                float *crBlock = &crBlocks[chrominanceBlockIdx * BlockSize];

                size_t cbDcOffset = buffer.size();
                previousDC_Cb = ProcessBlock(threadID, buffer, cbBlock, Component::Cb, previousDC_Cb);

                size_t crDcOffset = buffer.size();
                previousDC_Cr = ProcessBlock(threadID, buffer, crBlock, Component::Cr, previousDC_Cr);

                chrominanceBlockIdx++;

                if (blockIdx == 0) {
                    indices[0] = 0;
                    indices[1] = cbDcOffset;
                    indices[2] = crDcOffset;
                    firstDCs[1] = previousDC_Cb.value();
                    firstDCs[2] = previousDC_Cr.value();
                }
            }

            blockIdx++;
        }
    }
    lastDCs[0] = previousDC_Y.value();
    lastDCs[1] = previousDC_Cb.value();
    lastDCs[2] = previousDC_Cr.value();
}


void Encoder::MergeThreadHistograms(size_t channelCount) {
    for (size_t channelIdx = 0; channelIdx < channelCount; channelIdx++) {
        auto &dcHistogram = m_ThreadHistograms[channelIdx * 2];
        auto &acHistogram = m_ThreadHistograms[channelIdx * 2 + 1];

        for (size_t i = 0; i < 256; i++) {
            for (size_t threadIdx = 1; threadIdx < m_ThreadCount; threadIdx++) {
                dcHistogram[i] += m_ThreadHistograms[(threadIdx * 6) + (channelIdx * 2)][i];
                acHistogram[i] += m_ThreadHistograms[(threadIdx * 6) + (channelIdx * 2) + 1][i];
            }
        }
    }
}


void Encoder::GenerateOptimalHuffmanTable(size_t idx) {
    // Special comparator, if two symbols have same frequencies, then
    // the one with higher ordinal value is considered to be "smaller"
    auto freqComp = [&](auto &x, auto &y) {
        if (x.second == y.second)
            return x.first < y.first;

        return x.second > y.second;
    };


    const std::array<uint64_t, 256> &histogram = m_ThreadHistograms[idx];

    std::vector<std::pair<uint16_t, uint64_t>> frequencyHeap;
    frequencyHeap.reserve(128);
    for (size_t symbol = 0; symbol < histogram.size(); symbol++) {
        uint64_t frequency = histogram[symbol];
        if (frequency > 0) {
            frequencyHeap.emplace_back(symbol, frequency);
        }
    }
    size_t nonzeroSymbols = frequencyHeap.size();

    const uint16_t reservedSymbol = 256;
    frequencyHeap.emplace_back(reservedSymbol, 1);
    std::make_heap(frequencyHeap.begin(), frequencyHeap.end(), freqComp);

    std::array<uint8_t, 257> codeSizes{};
    std::array<int16_t, 257> others{};
    std::fill(others.begin(), others.end(), -1);

    // Find Huffman code sizes, from Figure K.1 of JPEG spec,
    // symbol searching implemented via min-heap
    while (frequencyHeap.size() > 1) {
        std::pop_heap(frequencyHeap.begin(), frequencyHeap.end(), freqComp);
        auto symbol1 = frequencyHeap.back();
        frequencyHeap.pop_back();

        std::pop_heap(frequencyHeap.begin(), frequencyHeap.end(), freqComp);
        auto symbol2 = frequencyHeap.back();
        frequencyHeap.pop_back();

        frequencyHeap.emplace_back(symbol1.first, symbol1.second + symbol2.second);
        std::push_heap(frequencyHeap.begin(), frequencyHeap.end(), freqComp);

        uint16_t v1 = symbol1.first;
        codeSizes[v1]++;
        while (others[v1] >= 0) {
            v1 = others[v1];
            codeSizes[v1]++;
        }

        uint16_t v2 = symbol2.first;
        others[v1] = v2;
        codeSizes[v2]++;
        while (others[v2] >= 0) {
            v2 = others[v2];
            codeSizes[v2]++;
        }
    }

    // Count number of codes of each bit length, Figure K.2 from JPEG spec
    const size_t MAX_CODE_LENGTH = 32;
    std::array<uint8_t, MAX_CODE_LENGTH + 1> codesPerBitsize{};
    for (uint8_t codeLength : codeSizes) {
        if (codeLength != 0) {
            codesPerBitsize[codeLength]++;
        }
    }

    // Adjust_BITS procedure from Figure K.3 of JPEG spec
    const size_t MAX_CODE_LENGTH_ADJUSTED = 16;
    for (size_t i = MAX_CODE_LENGTH; i > MAX_CODE_LENGTH_ADJUSTED;) {
        if (codesPerBitsize[i] > 0) {
            size_t j = i - 1;
            do {
                j--;
            } while (codesPerBitsize[j] == 0);

            codesPerBitsize[i] -= 2;
            codesPerBitsize[i - 1]++;
            codesPerBitsize[j + 1] += 2;
            codesPerBitsize[j]--;
        } else {
            i--;
        }
    }
    size_t i = MAX_CODE_LENGTH_ADJUSTED;
    while (codesPerBitsize[i] == 0) {
        i--;
    }
    codesPerBitsize[i]--;

    std::copy_n(&codesPerBitsize[1], m_CodeCounts[idx].size(), m_CodeCounts[idx].begin());

    std::vector<uint8_t> &sortedCodes = m_SortedCodes[idx];
    sortedCodes.resize(nonzeroSymbols);

    size_t k = 0;
    for (i = 1; i <= MAX_CODE_LENGTH; i++) {
        for (size_t j = 0; j < 256; j++) {
            if (codeSizes[j] == i) {
                sortedCodes[k++] = j;
            }
        }
    }

    GenerateHuffmanTable(idx);
}


void Encoder::EncodeJPEG(const char *filepath,
                         const char *outFilepath,
                         OutputMode mode,
                         bool optimizeTables,
                         bool parallel) {

    EncodingMetadata metadata{};

    std::unique_ptr<u_char> imageData(stbi_load(filepath, &metadata.width, &metadata.height,
                                                &metadata.channels, STBI_rgb));
    if (!imageData) {
        std::cerr << "[Error] Image file '" << filepath << "' does not exists or is invalid." << std::endl;
        return;
    }

    if (mode == OutputMode::RGB) {
        if (metadata.channels != STBI_rgb) {
            std::cout << "[Warning] Input image contains only one channel, output image will be in grayscale."
                      << std::endl;

            mode = OutputMode::GRAYSCALE;
        }
    } else {
        metadata.channels = STBI_grey;
    }

    const size_t xMultiple = m_DownsampleX ? BlockWidth * 2 : BlockWidth;
    const size_t yMultiple = m_DownsampleY ? BlockWidth * 2 : BlockWidth;

    metadata.luminanceWidthPadded = RoundUp(metadata.width, xMultiple);
    metadata.luminanceHeightPadded = RoundUp(metadata.height, yMultiple);
    metadata.chrominanceWidth = m_DownsampleX ? metadata.luminanceWidthPadded / 2 : metadata.luminanceWidthPadded;
    metadata.chrominanceHeight = m_DownsampleY ? metadata.luminanceHeightPadded / 2 : metadata.luminanceHeightPadded;

    metadata.luminanceBlocksX = metadata.luminanceWidthPadded / BlockWidth;
    metadata.luminanceBlocksY = metadata.luminanceHeightPadded / BlockWidth;
    const size_t luminanceBlockCount = metadata.luminanceBlocksX * metadata.luminanceBlocksY;
    metadata.chrominanceBlocksX = m_DownsampleX ? metadata.luminanceBlocksX / 2 : metadata.luminanceBlocksX;

    SeparateComponents(imageData.get(), metadata, parallel);

    if (!optimizeTables) {
        m_SortedCodes[0].resize(DcLuminanceCodes.size());
        m_SortedCodes[1].resize(AcLuminanceCodes.size());
        m_SortedCodes[2].resize(DcChrominanceCodes.size());
        m_SortedCodes[3].resize(AcChrominanceCodes.size());

        std::copy_n(DcLuminanceCodeCounts.begin(), 16, m_CodeCounts[0].begin());
        std::copy_n(AcLuminanceCodeCounts.begin(), 16, m_CodeCounts[1].begin());
        std::copy_n(DcChrominanceCodeCounts.begin(), 16, m_CodeCounts[2].begin());
        std::copy_n(AcChrominanceCodeCounts.begin(), 16, m_CodeCounts[3].begin());

        std::copy_n(DcLuminanceCodes.begin(), m_SortedCodes[0].size(), m_SortedCodes[0].begin());
        std::copy_n(AcLuminanceCodes.begin(), m_SortedCodes[1].size(), m_SortedCodes[1].begin());
        std::copy_n(DcChrominanceCodes.begin(), m_SortedCodes[2].size(), m_SortedCodes[2].begin());
        std::copy_n(AcChrominanceCodes.begin(), m_SortedCodes[3].size(), m_SortedCodes[3].begin());

        for (size_t tableID = 0; tableID < 2; tableID++) {
            GenerateHuffmanTable(tableID * 2);
            GenerateHuffmanTable(tableID * 2 + 1);
        }
        std::copy(m_HuffmanTables[Component::Cb * 2].begin(), m_HuffmanTables[Component::Cb * 2].end(),
                  m_HuffmanTables[Component::Cr * 2].begin());

        std::copy(m_HuffmanTables[Component::Cb * 2 + 1].begin(), m_HuffmanTables[Component::Cb * 2 + 1].end(),
                  m_HuffmanTables[Component::Cr * 2 + 1].begin());
    }

    float *yBlocks = m_ComponentBlocks[Component::Y].data();
    float *cbBlocks = m_ComponentBlocks[Component::Cb].data();
    float *crBlocks = m_ComponentBlocks[Component::Cr].data();
    size_t horizontalStep = m_DownsampleX ? 2 : 1;
    size_t verticalStep = m_DownsampleY ? 2 : 1;
    size_t chrominanceStep = horizontalStep * verticalStep;

    if (parallel) {

        size_t threadStep = RoundUp(std::ceil((float) metadata.luminanceBlocksY / m_ThreadCount), verticalStep);
        size_t threadLuminanceBlockCount = threadStep * metadata.luminanceBlocksX;

        m_ThreadBuffers.resize(m_ThreadCount);
        m_ThreadLastDCs.resize(m_ThreadCount);
        m_ThreadFirstDCs.resize(m_ThreadCount);
        m_ThreadDcIndices.resize(m_ThreadCount);
        m_ThreadEncodeIndices.resize(m_ThreadCount);

        // Two histograms per one component per one thread
        m_ThreadHistograms.resize(m_ThreadCount * 6);

        size_t threadBufferSizeEstimate = threadLuminanceBlockCount * 13;
        if (mode == OutputMode::RGB) {
            threadBufferSizeEstimate += (threadBufferSizeEstimate / chrominanceStep) * 2;
        }

        std::vector<std::future<void>> workers;
        for (size_t threadIdx = 0; threadIdx < m_ThreadCount; threadIdx++) {
            m_ThreadBuffers[threadIdx].reserve(threadBufferSizeEstimate);
            m_ThreadEncodeIndices[threadIdx].reserve(threadBufferSizeEstimate / 2);

            workers.emplace_back(std::async(std::launch::async, &Encoder::EncodeThreadChunk,
                                            this,
                                            threadIdx, threadStep,
                                            std::ref(metadata)));
        }

        // Wait for all threads to finish
        for (auto &future : workers) {
            future.get();
        }

        if (optimizeTables) {
            MergeThreadHistograms(metadata.channels);

            // Generate two optimal Huffman tables (DC + AC) for each component
            for (auto channel = 0; channel < metadata.channels; channel++) {
                GenerateOptimalHuffmanTable(channel * 2);
                GenerateOptimalHuffmanTable(channel * 2 + 1);
            }
        }


        // 128 bytes per 8x8 block should be more than the theoretical limit
        BitBuffer outputBuffer(luminanceBlockCount * BlockSize * metadata.channels * 2);
        WriteMetadata(outputBuffer, metadata.width, metadata.height, metadata.channels, optimizeTables);


        for (size_t threadIdx = 0; threadIdx < m_ThreadCount; ++threadIdx) {
            const std::vector<const ElementCode *> &chunkBuffer = m_ThreadBuffers[threadIdx];
            std::vector<size_t> dcIndices;
            std::copy_n(m_ThreadDcIndices[threadIdx].data(), metadata.channels, std::back_inserter(dcIndices));

            auto encodeDC = [&](size_t componentIdx) {
                const auto &huffmanTable = m_HuffmanTables[componentIdx * 2];

                int16_t dcDiff =
                        m_ThreadFirstDCs[threadIdx][componentIdx] - m_ThreadLastDCs[threadIdx - 1][componentIdx];
                if (dcDiff != 0) {
                    // output category code followed by the value itself
                    ElementCode valueCode = m_CodeTable[MaxDCTValue + dcDiff];
                    ElementCode categoryCode = huffmanTable[valueCode.m_BitCount];
                    outputBuffer.Push(categoryCode);
                    outputBuffer.Push(valueCode);
                } else {
                    outputBuffer.Push(huffmanTable[0x00]);
                }
            };

            size_t componentIdx = 0;
            for (size_t i = 0; i < chunkBuffer.size(); ++i) {
                if (threadIdx > 0 && !dcIndices.empty() && dcIndices.front() == i) {
                    encodeDC(componentIdx);
                    dcIndices.erase(dcIndices.begin());
                    componentIdx++;
                }

                outputBuffer.Push(*chunkBuffer[i]);
            }
        }

        outputBuffer.FlushRemainingBits();
        outputBuffer.PushMarker(JFIF_Marker::EOI, 0);
        outputBuffer.OutputToFile(outFilepath);

    } else {
        size_t bufferSizeEstimate = luminanceBlockCount * 13;
        if (mode == OutputMode::RGB) {
            bufferSizeEstimate += (bufferSizeEstimate / chrominanceStep) * 2;
        }

        // Two histograms per one component per one thread
        m_ThreadHistograms.resize(mode == OutputMode::RGB ? 6 : 2);
        m_ThreadBuffers.resize(1);
        m_ThreadBuffers[0].reserve(bufferSizeEstimate);

        std::optional<int16_t> previousDC_Y = 0;
        std::optional<int16_t> previousDC_Cb = 0;
        std::optional<int16_t> previousDC_Cr = 0;
        size_t chrominanceBlockIdx = 0;
        for (size_t y = 0; y < metadata.luminanceBlocksY; y += verticalStep) {
            for (size_t x = 0; x < metadata.luminanceBlocksX; x += horizontalStep) {

                for (size_t yOffset = 0; yOffset < verticalStep; yOffset++) {
                    for (size_t xOffset = 0; xOffset < horizontalStep; xOffset++) {
                        size_t xBlockPos = x + xOffset;
                        size_t yBlockPos = y + yOffset;
                        float *blockData_Y = &yBlocks[(yBlockPos * metadata.luminanceBlocksX + xBlockPos) * BlockSize];
                        previousDC_Y = ProcessBlock(0, m_ThreadBuffers[0], blockData_Y, Component::Y, previousDC_Y);
                    }
                }

                if (metadata.channels > 1) {
                    float *blockData_Cb = &cbBlocks[chrominanceBlockIdx * BlockSize];
                    float *blockData_Cr = &crBlocks[chrominanceBlockIdx * BlockSize];
                    previousDC_Cb = ProcessBlock(0, m_ThreadBuffers[0], blockData_Cb, Component::Cb, previousDC_Cb);
                    previousDC_Cr = ProcessBlock(0, m_ThreadBuffers[0], blockData_Cr, Component::Cr, previousDC_Cr);
                    chrominanceBlockIdx++;
                }
            }
        }

        if (optimizeTables) {
            // Generate two optimal Huffman tables (DC + AC) for each component
            for (auto channel = 0; channel < metadata.channels; channel++) {
                GenerateOptimalHuffmanTable(channel * 2);
                GenerateOptimalHuffmanTable(channel * 2 + 1);
            }
        }

        // 128 bytes per 8x8 block should be more than the theoretical limit
        BitBuffer outputBuffer(luminanceBlockCount * BlockSize * metadata.channels * 2);
        WriteMetadata(outputBuffer, metadata.width, metadata.height, metadata.channels, optimizeTables);
        for (const ElementCode *code : m_ThreadBuffers[0]) {
            outputBuffer.Push(*code);
        }

        outputBuffer.FlushRemainingBits();
        outputBuffer.PushMarker(JFIF_Marker::EOI, 0);
        outputBuffer.OutputToFile(outFilepath);
    }
}


void Encoder::SeparateComponents(const u_char *imageData, const EncodingMetadata &info, bool parallel) {
    size_t luminancePixelCount = info.luminanceWidthPadded * info.luminanceHeightPadded;
    size_t chrominancePixelCount = info.chrominanceWidth * info.chrominanceHeight;

    std::array<std::vector<float>, 3> components;
    components[0].resize(luminancePixelCount);
    m_ComponentBlocks[0].resize(luminancePixelCount);

    if (info.channels > 1) {
        components[1].resize(luminancePixelCount);
        components[2].resize(luminancePixelCount);
        m_ComponentBlocks[1].resize(chrominancePixelCount);
        m_ComponentBlocks[2].resize(chrominancePixelCount);
    }

    auto *pixel = reinterpret_cast<const Pixel *>(imageData);
    for (auto y = 0; y < info.height; ++y) {
        size_t dstRowOffset = (y * info.luminanceWidthPadded);
        for (auto x = 0; x < info.width; ++x) {
            components[Component::Y][dstRowOffset + x] = rgbToY(pixel->r, pixel->g, pixel->b);

            if (info.channels > 1) {
                components[Component::Cb][dstRowOffset + x] = rgbToCb(pixel->r, pixel->g, pixel->b);
                components[Component::Cr][dstRowOffset + x] = rgbToCr(pixel->r, pixel->g, pixel->b);
            }

            pixel++;
        }
    }


    // Extend each component to a multiple of 8 or 16 based on used subsampling method
    size_t widthDiff = info.luminanceWidthPadded - info.width;
    size_t heightDiff = info.luminanceHeightPadded - info.height;

    auto threadWork = [&](size_t componentIdx) {
        float *values = components[componentIdx].data();
        if (widthDiff > 0) {
            for (auto row = 0; row < info.height; ++row) {
                float *rowData = &values[row * info.luminanceWidthPadded];
                std::fill_n(&rowData[info.width], widthDiff, rowData[info.width - 1]);
            }
        }
        if (heightDiff > 0) {
            float *src = &values[(info.height - 1) * info.luminanceWidthPadded];
            float *dst = &values[info.height * info.luminanceWidthPadded];
            for (size_t i = 0; i < heightDiff; ++i) {
                std::copy_n(src, info.luminanceWidthPadded, dst);
                dst += info.luminanceWidthPadded;
            }
        }


        // Split components into blocks of 8x8
        if (componentIdx > 0 && m_DownsampleX) {
            DownsampleComponent(values, m_DownsampleY, info.luminanceWidthPadded, info.luminanceHeightPadded);
        }

        float *srcData = components[componentIdx].data();
        float *dstBlocks = m_ComponentBlocks[componentIdx].data();

        size_t rows = info.luminanceHeightPadded;
        size_t blocksX = info.luminanceBlocksX;
        if (componentIdx > 0) {
            rows = info.chrominanceHeight;
            blocksX = info.chrominanceBlocksX;
        }

        for (size_t y = 0; y < rows; ++y) {
            size_t startOffset = (y % BlockWidth) * BlockWidth;
            float *srcRowData = &srcData[y * info.luminanceWidthPadded];
            float *dstRowData = &dstBlocks[(y / BlockWidth) * BlockSize * blocksX + startOffset];
            for (size_t blockIdx = 0; blockIdx < blocksX; ++blockIdx) {
                std::copy_n(srcRowData, BlockWidth, dstRowData);
                dstRowData += BlockSize;
                srcRowData += BlockWidth;
            }
        }
    };

    std::vector<std::future<void>> workers;
    for (int componentIdx = 0; componentIdx < info.channels; componentIdx++) {
        if (parallel) {
            workers.emplace_back(std::async(std::launch::async, threadWork, componentIdx));
        } else {
            threadWork(componentIdx);
        }
    }

    if (parallel) {
        // Wait for all threads to finish
        for (auto &future : workers) {
            future.get();
        }
    }
}
