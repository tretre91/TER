set_languages("c++20")
add_rules("mode.debug", "mode.release")

package("eve")
	add_urls("https://github.com/jfalcou/eve.git")

	on_install(function (package)
		os.cp("include", package:installdir())
	end)
package_end()

add_requires("eve", "catch2 3.3.2", "nanobench 4.3.11", "fmt 9.1.0")
add_requires("openblas", {system = true})

target("gemm")
	set_kind("headeronly")
	add_headerfiles("include/gemm/**/*.hpp")
	add_includedirs("include", {public = true})
	add_packages("eve", {public = true})
	set_warnings("allextra")

target("test")
	set_kind("binary")
	add_files("test/*.cpp")
	add_deps("gemm")
	add_packages("nanobench", "catch2", "openblas", "fmt")
	add_cxxflags("-march=native")
	set_warnings("allextra")
