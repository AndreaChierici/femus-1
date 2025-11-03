// #pragma omp requires unified_shared_memory
#include <vector>
int main() {
  std::vector<double> a(100,1.0);
  #pragma omp target
  a[0] = 2.0;
  return 0;
}
