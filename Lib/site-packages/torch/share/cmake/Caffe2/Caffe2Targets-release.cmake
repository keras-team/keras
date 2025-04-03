#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "c10" for configuration "Release"
set_property(TARGET c10 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(c10 PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/c10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/c10.dll"
  )

list(APPEND _cmake_import_check_targets c10 )
list(APPEND _cmake_import_check_files_for_c10 "${_IMPORT_PREFIX}/lib/c10.lib" "${_IMPORT_PREFIX}/lib/c10.dll" )

# Import target "torch_cpu" for configuration "Release"
set_property(TARGET torch_cpu APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(torch_cpu PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/torch_cpu.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "fbgemm"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/torch_cpu.dll"
  )

list(APPEND _cmake_import_check_targets torch_cpu )
list(APPEND _cmake_import_check_files_for_torch_cpu "${_IMPORT_PREFIX}/lib/torch_cpu.lib" "${_IMPORT_PREFIX}/lib/torch_cpu.dll" )

# Import target "torch" for configuration "Release"
set_property(TARGET torch APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(torch PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/torch.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/torch.dll"
  )

list(APPEND _cmake_import_check_targets torch )
list(APPEND _cmake_import_check_files_for_torch "${_IMPORT_PREFIX}/lib/torch.lib" "${_IMPORT_PREFIX}/lib/torch.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
