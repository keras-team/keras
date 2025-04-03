#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TBB::tbb" for configuration "Release"
set_property(TARGET TBB::tbb APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TBB::tbb PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/openvino/libs/tbb12.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/openvino/libs/tbb12.dll"
  )

list(APPEND _cmake_import_check_targets TBB::tbb )
list(APPEND _cmake_import_check_files_for_TBB::tbb "${_IMPORT_PREFIX}/openvino/libs/tbb12.dll" )

# Import target "TBB::tbbmalloc" for configuration "Release"
set_property(TARGET TBB::tbbmalloc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TBB::tbbmalloc PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/openvino/libs/tbbmalloc.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/openvino/libs/tbbmalloc.dll"
  )

list(APPEND _cmake_import_check_targets TBB::tbbmalloc )
list(APPEND _cmake_import_check_files_for_TBB::tbbmalloc "${_IMPORT_PREFIX}/openvino/libs/tbbmalloc.dll" )

# Import target "TBB::tbbmalloc_proxy" for configuration "Release"
set_property(TARGET TBB::tbbmalloc_proxy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TBB::tbbmalloc_proxy PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/openvino/libs/tbbmalloc_proxy.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "TBB::tbbmalloc"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/openvino/libs/tbbmalloc_proxy.dll"
  )

list(APPEND _cmake_import_check_targets TBB::tbbmalloc_proxy )
list(APPEND _cmake_import_check_files_for_TBB::tbbmalloc_proxy "${_IMPORT_PREFIX}/openvino/libs/tbbmalloc_proxy.dll" )

# Import target "TBB::tbbbind_2_5" for configuration "Release"
set_property(TARGET TBB::tbbbind_2_5 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TBB::tbbbind_2_5 PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/openvino/libs/tbbbind_2_5.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/openvino/libs/tbbbind_2_5.dll"
  )

list(APPEND _cmake_import_check_targets TBB::tbbbind_2_5 )
list(APPEND _cmake_import_check_files_for_TBB::tbbbind_2_5 "${_IMPORT_PREFIX}/openvino/libs/tbbbind_2_5.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
