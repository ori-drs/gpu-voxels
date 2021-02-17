// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Mark Finean
 * \date    2020-11-30
 *
 */
//----------------------------------------------------------------------
#ifndef SIGNEDDISTANCEVOXELMAP_H
#define SIGNEDDISTANCEVOXELMAP_H

#include <gpu_voxels/voxelmap/DistanceVoxelMap.h>

using namespace gpu_voxels;

namespace gpu_voxels {
namespace voxelmap {

class SignedDistanceVoxelMap
{
public:

  boost::shared_ptr<DistanceVoxelMap> pbaDistanceVoxmap_;
  boost::shared_ptr<DistanceVoxelMap> pbaInverseDistanceVoxmap_;

  SignedDistanceVoxelMap(boost::shared_ptr<DistanceVoxelMap> pbaDistanceVoxmap, boost::shared_ptr<DistanceVoxelMap> pbaInverseDistanceVoxmap, const Vector3ui dim, const float voxel_side_length);

// getVoxelMapSize
  void clearMaps();

  void occupancyMerge(boost::shared_ptr<ProbVoxelMap> maintainedProbVoxmap, float occupancy_threshold, float free_threshold);

  void parallelBanding3D();
  // void parallelBanding3DMark(uint32_t m1 = 1, uint32_t m2 = 1, uint32_t m3 = 1, uint32_t m1_blocksize = gpu_voxels::PBA_DEFAULT_M1_BLOCK_SIZE, uint32_t m2_blocksize = gpu_voxels::PBA_DEFAULT_M2_BLOCK_SIZE, uint32_t m3_blocksize = gpu_voxels::PBA_DEFAULT_M3_BLOCK_SIZE, bool detailtimer = false);


  DistanceVoxelMap* getNormal();
  DistanceVoxelMap* getInverse();

  void getSignedDistancesToHost(std::vector<float>& host_result_map);
  void getSignedDistancesAndGradientsToHost(std::vector<VectorSdfGrad>& host_result_map);

protected:
  const Vector3ui map_dims_;
  const float voxel_side_length_;
//getSignedDistancesAndGradientsToHost
//get
};

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif // SIGNEDDISTANCEVOXELMAP_H
