// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Mark Finean
 * \date    2020-11-30
 */
//----------------------------------------------------------------------
#ifndef SIGNEDDISTANCEVOXELMAP_HPP
#define SIGNEDDISTANCEVOXELMAP_HPP

#include <gpu_voxels/voxelmap/SignedDistanceVoxelMap.h>

namespace gpu_voxels {
namespace voxelmap {

SignedDistanceVoxelMap::SignedDistanceVoxelMap(DistanceVoxelMap* pbaDistanceVoxmap, DistanceVoxelMap* pbaInverseDistanceVoxmap) :
  pbaDistanceVoxmap_(pbaDistanceVoxmap), pbaInverseDistanceVoxmap_(pbaInverseDistanceVoxmap)

{

};

void SignedDistanceVoxelMap::clearMaps(){
  pbaDistanceVoxmap_->clearMap();
  pbaInverseDistanceVoxmap_->clearMap();
}


} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif // SIGNEDDISTANCEVOXELMAP_HPP

