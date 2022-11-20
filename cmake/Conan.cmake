if(${PROJECT_NAME}_ENABLE_CONAN)
  #
  # Setup Conan requires and options here:
  #

  set(${PROJECT_NAME}_CONAN_REQUIRES gtest/1.12.1 eigen/3.4.0 boost/1.80.0)
  set(${PROJECT_NAME}_CONAN_OPTIONS "")

  #
  # If `conan.cmake` (from https://github.com/conan-io/cmake-conan) does not exist, download it.
  #
  if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(
      STATUS
      "Downloading conan.cmake from https://github.com/conan-io/cmake-conan..."
    )
    file(
      DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.16/conan.cmake"
      "${CMAKE_BINARY_DIR}/conan.cmake"
    )
    message(STATUS "Cmake-Conan downloaded succesfully.")
  endif()

  include(${CMAKE_BINARY_DIR}/conan.cmake)

  conan_add_remote(
    NAME conancenter
    URL
    https://center.conan.io
  )

  conan_cmake_run(
    REQUIRES
    ${${PROJECT_NAME}_CONAN_REQUIRES}
    OPTIONS
    ${${PROJECT_NAME}_CONAN_OPTIONS}
    BASIC_SETUP
    CMAKE_TARGETS # Individual targets to link to
    BUILD
    missing
  )

  conan_basic_setup()

  verbose_message("Conan is setup and all requires have been installed.")
endif()
