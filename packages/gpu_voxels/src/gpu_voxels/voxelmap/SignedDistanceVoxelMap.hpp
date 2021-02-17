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

SignedDistanceVoxelMap::SignedDistanceVoxelMap(boost::shared_ptr<DistanceVoxelMap> pbaDistanceVoxmap, boost::shared_ptr<DistanceVoxelMap> pbaInverseDistanceVoxmap, const Vector3ui dim, const float voxel_side_length) :
  pbaDistanceVoxmap_(pbaDistanceVoxmap), pbaInverseDistanceVoxmap_(pbaInverseDistanceVoxmap), map_dims_(dim), voxel_side_length_(voxel_side_length){};


void SignedDistanceVoxelMap::clearMaps(){
  pbaDistanceVoxmap_->clearMap();
  pbaInverseDistanceVoxmap_->clearMap();
}


void SignedDistanceVoxelMap::occupancyMerge(boost::shared_ptr<ProbVoxelMap> maintainedProbVoxmap, float occupancy_threshold, float free_threshold){
  pbaDistanceVoxmap_->mergeOccupied(maintainedProbVoxmap, Vector3ui(), occupancy_threshold);
  pbaInverseDistanceVoxmap_->mergeFree(maintainedProbVoxmap, Vector3ui(), free_threshold);
}

void SignedDistanceVoxelMap::parallelBanding3D(){
  pbaDistanceVoxmap_->parallelBanding3D();
  pbaInverseDistanceVoxmap_->parallelBanding3D();     
}

DistanceVoxelMap* SignedDistanceVoxelMap::getNormal(){
  return (DistanceVoxelMap*) pbaDistanceVoxmap_.get(); 
}


DistanceVoxelMap* SignedDistanceVoxelMap::getInverse(){
  return (DistanceVoxelMap*) pbaInverseDistanceVoxmap_.get(); 
}

void SignedDistanceVoxelMap::getSignedDistancesToHost(std::vector<float>& host_result_map){
  pbaDistanceVoxmap_->getSignedDistancesToHost(pbaInverseDistanceVoxmap_, host_result_map);
}


void SignedDistanceVoxelMap::getSignedDistancesAndGradientsToHost(std::vector<VectorSdfGrad>& host_result_map){
  pbaDistanceVoxmap_->getSignedDistancesAndGradientsToHost(pbaInverseDistanceVoxmap_, host_result_map);
}

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif // SIGNEDDISTANCEVOXELMAP_HPP

