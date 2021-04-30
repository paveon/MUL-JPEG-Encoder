#include "Encoder.h"
#include "ArgumentParser.h"


void PrintUsage() {
    std::cout << "[Usage] jpeg_encoder filepath [-rgb] [-optimize] [-subsample VALUE] [-q VALUE] [-o FILEPATH]\n"
                 "Options:\n"
                 "   filepath           Path to the input image.\n"
                 "   -rgb               Output RGB image.\n"
                 "   -optimize          Calculate optimized Huffman tables.\n"
                 "   -mt                Multi-threaded compression.\n"
                 "   -subsample VALUE   Subsample chrominance component.\n"
                 "                      Possible values are '4:2:0' (2x horizontal + vertical)\n"
                 "                      and '4:2:2' (2x horizontal only).\n"
                 "   -q VALUE           JPEG Quality level, possible values in range <1, 100>.\n"
                 "   -o FILEPATH        Specify output path for JPEG image.\n"
                 "   -h | --help        Show usage.\n";
    std::cout.flush();
}


int main(int argc, char **argv) {
    Parser::ArgumentParser parser(argc, argv);
    parser.AddArgument("filepath");
    parser.AddSwitch("-rgb", Parser::Arg::Optional);
    parser.AddSwitch("-optimize", Parser::Arg::Optional);
    parser.AddSwitch("-mt", Parser::Arg::Optional);
    parser.AddSwitch("-subsample", Parser::Arg::Optional | Parser::Arg::HasValue);
    parser.AddSwitch("-o", Parser::Arg::Optional | Parser::Arg::HasValue);
    parser.AddSwitch("-q", Parser::Arg::Optional | Parser::Arg::HasValue);
    parser.AddSwitch("-h", Parser::Arg::Optional);
    parser.AddSwitch("--help", Parser::Arg::Optional);

    try {
        parser.Parse();
    }
    catch (const Parser::ArgException& e) {
        if (!(parser.HasArgument("-h") || parser.HasArgument("--help")))
            std::cerr << "[Exception] " << e.what() << std::endl;

        PrintUsage();
        return EXIT_FAILURE;
    }

    if (parser.HasArgument("-h") || parser.HasArgument("--help")) {
        PrintUsage();
        return EXIT_SUCCESS;
    }

    Encoder encoder;

    if (parser.HasArgument("-q")) {
        const std::string qualityLevelStr(*parser["-q"]);
        try {
            encoder.SetQualityLevel(std::stoi(qualityLevelStr));
        }
        catch (const std::exception &e) {
            std::cout << "[Warning] Invalid quality setting, using default quality level of 50." << std::endl;
        }
    } else {
        encoder.SetQualityLevel(DefaultQualityLevel);
    }

    auto outputMode = parser.HasArgument("-rgb") ? Encoder::OutputMode::RGB : Encoder::OutputMode::GRAYSCALE;

    if (outputMode == Encoder::OutputMode::RGB && parser.HasArgument("-subsample")) {
        const std::string subsamplingMode(*parser["-subsample"]);
        if (subsamplingMode == "4:2:0") {
            encoder.SetDownsamplingMode(DownsampleMode::XY);
        } else if (subsamplingMode == "4:2:2") {
            encoder.SetDownsamplingMode(DownsampleMode::X);
        } else {
            std::cout << "[Warning] Invalid subsampling mode, only 4:2:0, 4:2:2 modes are allowed." << std::endl;
        }
    }

    const std::string inputPath(*parser["filepath"]);
    const std::string outputPath(parser.HasArgument("-o") ? (std::string)*parser["-o"] : "compressed.jpg");
    bool optimizeTables = parser.HasArgument("-optimize");
    bool parallel = parser.HasArgument("-mt");
    encoder.EncodeJPEG(inputPath.data(), outputPath.data(), outputMode, optimizeTables, parallel);

    return EXIT_SUCCESS;
}
