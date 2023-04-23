#include <catch2/catch_session.hpp>
#include <fstream>

#include "util.hpp"

int main(int argc, char* argv[]) {
	Catch::Session session;

	std::string bench_output_file;

	using namespace Catch::Clara;
	auto cli = session.cli()
	           | Opt(bench_output_file, "filename")["--bench-output"]
				 ("Output file where the benchmark results will be stored instead of outputing to stdout");

	session.cli(cli);

	int return_code = session.applyCommandLine(argc, argv);
	if (return_code != 0) {
		return return_code;
	}

	std::ofstream file;
	if (!bench_output_file.empty()) {
		file.open(bench_output_file);
		util::benchmark_output = &file;
		util::bench.output(util::benchmark_output);
	}

	return session.run();
}
