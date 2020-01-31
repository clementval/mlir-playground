add_library(oaru SHARED oaru.cu)

set(OARU_LIBARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

install(TARGETS oaru DESTINATION bin)


