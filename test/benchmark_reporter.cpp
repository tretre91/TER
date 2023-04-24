#include <catch2/catch_config.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "util.hpp"

/**
 * @brief Catch2 reporter used for the benchmarks
 */
class BenchmarkReporter : public Catch::StreamingReporterBase
{
private:
	using Catch::StreamingReporterBase::StreamingReporterBase;

	Catch::Verbosity m_verbosityLevel;
	std::string m_currentTestCase = "";

public:
	static std::string getDescription() { return "Reporter indicating which benchmark is currently being run"; }

	void testRunStarting(const Catch::TestRunInfo& info) override {
		Catch::StreamingReporterBase::testRunStarting(info);

		const auto& config = util::config_data;
		m_verbosityLevel = config.verbosity;

		util::bench.output(&m_stream);

		fmt::print("RNG Seed initialized to {}\n", config.rngSeed);
		if (!config.testsOrTags.empty()) {
			fmt::print("Active filters: {}\n\n", fmt::join(config.testsOrTags, "; "));
		} else {
			fmt::print("No filters\n\n");
		}
	}

	void testCaseStarting(const Catch::TestCaseInfo& info) override {
		Catch::StreamingReporterBase::testCaseStarting(info);

		const auto title = fmt::format("\n## Benchmark \"{}\"", info.name);
		fmt::print(m_stream, "{}\n", title);
		if (m_verbosityLevel == Catch::Verbosity::High) {
			fmt::print("{}\n", title);
		}
		m_currentTestCase = info.name;
	}

	void sectionStarting(const Catch::SectionInfo& info) override {
		Catch::StreamingReporterBase::sectionStarting(info);

		if (info.name == m_currentTestCase) {
			return;
		}

		const auto title = fmt::format("\n### {}", info.name);
		fmt::print(m_stream, "{}\n", title);
		if (m_verbosityLevel == Catch::Verbosity::High) {
			fmt::print("{}\n", title);
		}
	}

	void testCasePartialStarting(const Catch::TestCaseInfo& info, std::uint64_t partNumber) override {
		Catch::StreamingReporterBase::testCasePartialStarting(info, partNumber);

		if (m_verbosityLevel == Catch::Verbosity::High) {
			fmt::print("- {}#{} ...\n", info.name, partNumber);
		}
	}
};

CATCH_REGISTER_REPORTER("benchmark", BenchmarkReporter);
