

file(COPY ${MLIR_LIT_SITE_CONFIG} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})


set(LIT_SITE_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py")
set(LIT_LOCAL_CFG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py")

set(UPDATE_CFG_CMD "${CMAKE_CURRENT_SOURCE_DIR}/update_config")

configure_file(update_config.in ${UPDATE_CFG_CMD} @ONLY)
configure_file(lit.cfg.py.in ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py @ONLY)

add_custom_target(update-config ${UPDATE_CFG_CMD})

add_custom_target(
  check-openacc ${LIT_PATH} "${CMAKE_SOURCE_DIR}/openacc/" 
  DEPENDS update-config
)
