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

  DistanceVoxelMap* pbaDistanceVoxmap_;
  DistanceVoxelMap* pbaInverseDistanceVoxmap_;

  SignedDistanceVoxelMap(DistanceVoxelMap* pbaDistanceVoxmap, DistanceVoxelMap* pbaInverseDistanceVoxmap);

// getVoxelMapSize
  void clearMaps();
  
//mergeOccupied or mergeFree
//parallelBanding3D
//getSignedDistancesToHost
//getSignedDistancesAndGradientsToHost
//get
};

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif // SIGNEDDISTANCEVOXELMAP_H
