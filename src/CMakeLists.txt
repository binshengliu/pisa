add_executable(create_freq_index create_freq_index.cpp)
target_link_libraries(create_freq_index
  pisa
  CLI11
)
install(TARGETS create_freq_index DESTINATION bin)

add_executable(optimal_hybrid_index optimal_hybrid_index.cpp)
target_include_directories(optimal_hybrid_index PRIVATE ${STXXL_INCLUDE_DIRS})
target_link_libraries(optimal_hybrid_index
  ${STXXL_LIBRARIES}
  pisa
)
set_target_properties(optimal_hybrid_index PROPERTIES
            CXX_STANDARD 14
)
install(TARGETS optimal_hybrid_index DESTINATION bin)

add_executable(create_wand_data create_wand_data.cpp)
target_link_libraries(create_wand_data
  pisa
  CLI11
)
install(TARGETS create_wand_data DESTINATION bin)

add_executable(queries queries.cpp)
target_link_libraries(queries
  pisa
  CLI11
)
install(TARGETS queries DESTINATION bin)

add_executable(evaluate_queries evaluate_queries.cpp)
target_link_libraries(evaluate_queries
  pisa
  CLI11
)
install(TARGETS evaluate_queries DESTINATION bin)

add_executable(thresholds thresholds.cpp)
target_link_libraries(thresholds
  pisa
  CLI11
)
install(TARGETS thresholds DESTINATION bin)

add_executable(profile_queries profile_queries.cpp)
target_link_libraries(profile_queries
  pisa
)
install(TARGETS profile_queries DESTINATION bin)

add_executable(profile_decoding profile_decoding.cpp)
target_link_libraries(profile_decoding
  pisa
)
install(TARGETS profile_decoding DESTINATION bin)

add_executable(shuffle_docids shuffle_docids.cpp)
target_link_libraries(shuffle_docids
  pisa
)
install(TARGETS shuffle_docids DESTINATION bin)

add_executable(recursive_graph_bisection recursive_graph_bisection.cpp)
target_link_libraries(recursive_graph_bisection
  pisa
  CLI11
)
install(TARGETS recursive_graph_bisection DESTINATION bin)

add_executable(evaluate_collection_ordering evaluate_collection_ordering.cpp)
target_link_libraries(evaluate_collection_ordering
  pisa
  )
install(TARGETS evaluate_collection_ordering DESTINATION bin)

add_executable(parse_collection parse_collection.cpp)
target_link_libraries(parse_collection
  pisa
  CLI11
  wapopp
)
install(TARGETS parse_collection DESTINATION bin)

add_executable(invert invert.cpp)
target_link_libraries(invert
  CLI11
  pisa
)
install(TARGETS invert DESTINATION bin)

add_executable(read_collection read_collection.cpp)
target_link_libraries(read_collection
  pisa
  CLI11
)
install(TARGETS read_collection DESTINATION bin)

add_executable(partition_fwd_index partition_fwd_index.cpp)
target_link_libraries(partition_fwd_index
  pisa
  CLI11
)
install(TARGETS partition_fwd_index DESTINATION bin)

add_executable(compute_intersection compute_intersection.cpp)
target_link_libraries(compute_intersection
  pisa
  CLI11
)
install(TARGETS compute_intersection DESTINATION bin)

add_executable(lexicon lexicon.cpp)
target_link_libraries(lexicon
  pisa
  CLI11
)
install(TARGETS lexicon DESTINATION bin)

add_executable(extract_topics extract_topics.cpp)
target_link_libraries(extract_topics
  pisa
  CLI11
)
install(TARGETS extract_topics DESTINATION bin)
