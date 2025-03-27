#-----------------------------------------------------------------------------
# Adept
#-----------------------------------------------------------------------------

SET(ADEPT_VERSION 2.0)

# Adept-2 has been built and moved to this repo in include and lib //porchetta

SET (ADEPT_INCLUDE_DIRS  ${CMAKE_SOURCE_DIR}/include/adept-2.0/)
SET (ADEPT_LIBRARIES     ${CMAKE_SOURCE_DIR}/lib/adept-2.0/libadept.so)

INCLUDE_DIRECTORIES(${ADEPT_INCLUDE_DIRS})

#ExternalProject_Add(${proj}
#  SOURCE_DIR         ${ADEPT_UNPACK_PATH}/adept-${ADEPT_VERSION}
#  CONFIGURE_COMMAND  ""
#  BUILD_COMMAND make libadept.a
#  BUILD_IN_SOURCE 1
#  INSTALL_COMMAND ""
#) 
