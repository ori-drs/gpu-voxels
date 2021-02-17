// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Mark Finean
 * \date    2020-11-30
 *
 */
//----------------------------------------------------------------------
#ifndef INHERITSIGNEDDISTANCEVOXELMAP_H
#define INHERITSIGNEDDISTANCEVOXELMAP_H

#include <gpu_voxels/voxelmap/DistanceVoxelMap.h>

using namespace gpu_voxels;

namespace gpu_voxels {
namespace voxelmap {

class NormalDistanceVoxelMap : public DistanceVoxelMap {
  public: 
    NormalDistanceVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type) : DistanceVoxelMap(dim, voxel_side_length, map_type) {};
};

class InverseDistanceVoxelMap : public DistanceVoxelMap {
  public: 
    InverseDistanceVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type) : DistanceVoxelMap(dim, voxel_side_length, map_type) {};
};

class InheritSignedDistanceVoxelMap: public virtual NormalDistanceVoxelMap, public virtual InverseDistanceVoxelMap{
  public:
    InheritSignedDistanceVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type);
    //   NormalDistanceVoxelMap(dim, voxel_side_length, map_type), 
    //   InverseDistanceVoxelMap(dim, voxel_side_length, map_type) {
    ~InheritSignedDistanceVoxelMap();

    // }

    // InheritSignedDistanceVoxelMap(Voxel* dev_data, Voxel* inv_dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type);

  void clearMaps();

  void occupancyMerge(boost::shared_ptr<ProbVoxelMap> maintainedProbVoxmap, float occupancy_threshold, float free_threshold);

  // GpuVoxelsMapSharedPtr getNormal();
  // GpuVoxelsMapSharedPtr getInverse();

  NormalDistanceVoxelMap* getNormal();
  InverseDistanceVoxelMap* getInverse();

  void getSignedDistancesToHost(std::vector<float>& host_result_map);
  void getSignedDistancesAndGradientsToHost(std::vector<VectorSdfGrad>& host_result_map);

  void parallelBanding3DSigned();
  void parallelBanding3DUnsigned();

  void parallelBanding3DParallelSigned(uint32_t m1 = 1, uint32_t m2 = 1, uint32_t m3 = 1, uint32_t m1_blocksize = gpu_voxels::PBA_DEFAULT_M1_BLOCK_SIZE, uint32_t m2_blocksize = gpu_voxels::PBA_DEFAULT_M2_BLOCK_SIZE, uint32_t m3_blocksize = gpu_voxels::PBA_DEFAULT_M3_BLOCK_SIZE, bool detailtimer = false);
  // void parallelBanding3DCustomSUnsigned(uint32_t m1 = 1, uint32_t m2 = 1, uint32_t m3 = 1, uint32_t m1_blocksize = gpu_voxels::PBA_DEFAULT_M1_BLOCK_SIZE, uint32_t m2_blocksize = gpu_voxels::PBA_DEFAULT_M2_BLOCK_SIZE, uint32_t m3_blocksize = gpu_voxels::PBA_DEFAULT_M3_BLOCK_SIZE, bool detailtimer = false);

  private:
    // create two CUDA streams
    cudaStream_t s1, s2;
    thrust::device_vector<float>* dev_output;

};

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif // INHERITSIGNEDDISTANCEVOXELMAP_H
