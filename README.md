[![Actions Status](https://github.com/jkorsvik/NMBUFlowTorch/workflows/MacOS/badge.svg)](https://github.com/jkorsvik/NMBUFlowTorch/actions)
[![Actions Status](https://github.com/jkorsvik/NMBUFlowTorch/workflows/Windows/badge.svg)](https://github.com/jkorsvik/NMBUFlowTorch/actions)
[![Actions Status](https://github.com/jkorsvik/NMBUFlowTorch/workflows/Ubuntu/badge.svg)](https://github.com/jkorsvik/NMBUFlowTorch/actions)
[![codecov](https://codecov.io/gh/jkorsvik/NMBUFlowTorch/branch/master/graph/badge.svg)](https://codecov.io/gh/jkorsvik/NMBUFlowTorch)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/jkorsvik/NMBUFlowTorch)](https://github.com/jkorsvik/NMBUFlowTorch/releases)

# NMBUFlowTorch

A simple C++ implementation of Neural Nets, inspired by the functionality of Tensorflow and pyTorch.

## Features
* like the functional API of tensorflow
* using eigen3 library
* more to come

### Build and Install Features

* Modern **CMake** configuration and project, which, to the best of my
knowledge, uses the best practices,

* An example of a **Clang-Format** config, inspired from the base *Google* model,
with minor tweaks. This is aimed only as a starting point, as coding style
is a subjective matter, everyone is free to either delete it (for the *LLVM*
default) or supply their own alternative,

* **Static analyzers** integration, with *Clang-Tidy* and *Cppcheck*, the former
being the default option,

* **Doxygen** support, through the `ENABLE_DOXYGEN` option, which you can enable
if you wish to use it,

* **Unit testing** support, through *GoogleTest* (with an option to enable
*GoogleMock*) or *Catch2*,

* **Code coverage**, enabled by using the `ENABLE_CODE_COVERAGE` option, through
*Codecov* CI integration,

* **Package manager support**, with *Conan* and *Vcpkg*, through their respective
options

* **CI workflows for Windows, Linux and MacOS** using *GitHub Actions*, making
use of the caching features, to ensure minimum run time,

* **.md templates** for: *README*, *Contributing Guideliness*,
*Issues* and *Pull Requests*,

* **Permissive license** to allow you to integrate it as easily as possible. The
template is licensed under the [Unlicense](https://unlicense.org/),

* Options to build as a header-only library or executable, not just a static or
shared library.

* **Ccache** integration, for speeding up rebuild times

## Getting Started

These instructions will get you a copy of the project up and running on your local
machine for development and testing purposes.

### Prerequisites

This project is meant as a template and a good starting point for learning how create larger
c++ projects, compile and run them.
can be change to better suit the needs of the developer(s). If you wish to use the
template *as-is*, meaning using the versions recommended here, then you will need:

* **CMake v3.15+** - found at [https://cmake.org/](https://cmake.org/)

* **C++ Compiler** - needs to support at least the **C++17** standard, i.e. *MSVC*,
*GCC*, *Clang*

> ***Note:*** *You also need to be able to provide ***CMake*** a supported
[generator](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).*

* **Different data** - [MNIST](https://github.com/wichtounet/mnist) AND https://home.bawue.de/~horsch/teaching/inf205/src/image-benchmark.zip *MSVC*,
*GCC*, *Clang*

### If using VSCode

There are few select extensions which are recommended
* https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack
* 

### Installing

It is fairly easy to install the project, all you need to do is clone if from
**GitHub**.

If you wish to clone the repository, you simply need
to run:

```bash
git clone https://github.com/jkorsvik/NMBUFlowTorch/
```
or 
```bash
git clone git@github.com:jkorsvik/NMBUFlowTorch.git
```


***Install requirements (cmake, conan, etc..)***

```bash
source install_reqs.sh
```

***Build and install project as an executable***
```bash
source automatic_rebuild_and_install.sh
```
***If you just want to rebuild fast and you are happy with the install location, run:***
```bash
source cached_build_and_install.sh
```
The executables of the project will then be in the 

To install an already built project, you need to run the `install` target with CMake.
For example:

```bash
cmake --build build --target install --config Release

# a more general syntax for that command is:
cmake --build <build_directory> --target install --config <desired_config>
```

If you have not built the project yet, the `automatic_rebuild_and_install.sh` will do fine, otherwise follow the next section:

>**Note**: *If you want to supress a lot of warnings when building, see the [CMakeLists.txt](CMakeLists.txt) at line 145 and 146, and uncomment the preferred.*

## Building the project

To build the project, all you need to do, ***after correctly
[installing the project](README.md#Installing)***, is run a similar **CMake** routine
to the the one below:

```bash
mkdir build/ && cd build/
cmake .. -DCMAKE_INSTALL_PREFIX=/absolute/path/to/custom/install/directory
cmake --build . --target install
```

> ***Note:*** *The custom ``CMAKE_INSTALL_PREFIX`` can be omitted if you wish to
install in [the default install location](https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html).*

More options that you can set for the project can be found in the
[`cmake/StandardSettings.cmake` file](cmake/StandardSettings.cmake). For certain
options additional configuration may be needed in their respective `*.cmake` files
(i.e. Conan needs the `CONAN_REQUIRES` and might need the `CONAN_OPTIONS` to be setup
for it work correctly; the two are set in the [`cmake/Conan.cmake` file](cmake/Conan.cmake)).

## Generating the documentation

In order to generate documentation for the project, you need to configure the build
to use Doxygen. This is easily done, by modifying the workflow shown above as follows:

```bash
mkdir build/ && cd build/
cmake .. -D<project_name>_ENABLE_DOXYGEN=1 -DCMAKE_INSTALL_PREFIX=/absolute/path/to/custom/install/directory
cmake --build . --target doxygen-docs
```

> ***Note:*** *This will generate a `docs/` directory in the **project's root directory**.*

## Running the tests

By default, the template uses [Google Test](https://github.com/google/googletest/)
for unit testing. Unit testing can be disabled in the options, by setting the
`ENABLE_UNIT_TESTING` (from
[cmake/StandardSettings.cmake](cmake/StandardSettings.cmake)) to be false. To run
the tests, simply use CTest, from the build directory, passing the desire
configuration for which to run tests for. An example of this procedure is:

```bash
cd build          # if not in the build directory already
ctest -C Release  # or `ctest -C Debug` or any other configuration you wish to test

# you can also run tests with the `-VV` flag for a more verbose output (i.e.
#GoogleTest output as well)
```

### End to end tests

If applicable, should be presented here.

### Coding style tests

If applicable, should be presented here.

## Contributing


### Devolopment
After finishing getting a copy of the project, with any of the methods above, create
a new folder in the `include/` folder, with the name of your project.  Edit
`cmake/SourcesAndHeaders.cmake` to add your files.

You will also need to rename the `cmake/ProjectConfig.cmake.in` file to start with
the ***exact name of your project***. Such as `cmake/MyNewProjectConfig.cmake.in`.
You should also make the same changes in the GitHub workflows provided, notably
[`.github/workflows/ubuntu.yml`](.github/workflows/ubuntu.yml), in which you should
replace the CMake option `-DProject_ENABLE_CODE_COVERAGE=1` to
`-DMyNewProject_ENABLE_CODE_COVERAGE=1`.

Finally, change `"Project"` from `CMakeLists.txt`, from

```cmake
project(
  "nmbuflowtorch"
  VERSION 0.1.0
  LANGUAGES CXX
)
```

## Versioning

This project makes use of [SemVer](http://semver.org/) for versioning. A list of
existing versions can be found in the
[project's releases](https://github.com/jkorsvik/NMBUFlowTorch/releases).

## Authors

* **Jon-Mikkel Ryen Korsvik** - [@jkorsvik](https://github.com/jkorsvik)
* **Jørgen Navjord** - [@navjordj](https://github.com/navjordj)


## Inspiration Repos
* https://github.com/filipdutescu/modern-cpp-template Used for bootstrapping a c++ project and for managing building, installing, structure, as well as testing.
* https://github.com/krocki/MLP-CXX 
* https://github.com/TheFirstGuy/ann
* https://github.com/leondavi/cppSANN
* https://github.com/Black-Phoenix/CUDA-MLP



## License

This project is licensed under the [Unlicense](https://unlicense.org/) - see the
[LICENSE](LICENSE) file for details
