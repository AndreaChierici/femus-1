CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
  

  

  ADD_SUBDIRECTORY(./poisson/)
  ADD_SUBDIRECTORY(./poisson_nonauto/)
  ADD_SUBDIRECTORY(./poisson_nonauto_conv/)


  ADD_SUBDIRECTORY(./coupled_biharmonic/)
  ADD_SUBDIRECTORY(./coupled_biharmonic_test/)
  ADD_SUBDIRECTORY(./coupled_biharmonic_four_blocks/)


  ADD_SUBDIRECTORY(./uncoupled_biharmonic/)
  ADD_SUBDIRECTORY(./coupled_biharmonic_multiple_domains_sol/)
  ADD_SUBDIRECTORY(./uncoupled_biharmonic_multiple_domains_sol/)
  ADD_SUBDIRECTORY(./hermann_miyoshi/)
  ADD_SUBDIRECTORY(./optimal_control/)

  ADD_SUBDIRECTORY(./coupled_biharmonic_nonauto/)
  ADD_SUBDIRECTORY(./coupled_biharmonic_nonauto_conv/)

