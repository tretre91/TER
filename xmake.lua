set_languages("c++20")
add_rules("mode.debug", "mode.release")

package("eve")
	add_urls("https://github.com/jfalcou/eve.git")

	on_install(function (package)
		os.cp("include", package:installdir())
	end)
package_end()

package("nanobench")
	add_urls("https://github.com/martinus/nanobench.git")

	on_install(function (package)
		os.cp("src/include/nanobench.h", package:installdir("include"))
	end)
package_end()

add_requires("eve", "nanobench v4.3.11", "fmt")
add_requires("openblas", {system = true})

target("eblas")
	set_kind("headeronly")
	add_headerfiles("include/*.hpp")
	add_includedirs("include", {public = true})
	add_packages("eve", {public = true})
	set_warnings("allextra")

target("test")
	set_kind("binary")
	add_files("test/*.cpp")
	add_deps("eblas")
	add_packages("nanobench", "openblas", "fmt")
	add_cxxflags("-mavx2", "-mfma")
	set_warnings("allextra")
	set_optimize("fastest") -- -O3
