// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Mark Finean
 * \date    2020-11-30
 */
//----------------------------------------------------------------------
#ifndef INHERITSIGNEDDISTANCEVOXELMAP_HPP
#define INHERITSIGNEDDISTANCEVOXELMAP_HPP

#include <gpu_voxels/voxelmap/InheritSignedDistanceVoxelMap.h>

namespace gpu_voxels {
namespace voxelmap {

InheritSignedDistanceVoxelMap::InheritSignedDistanceVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type) : 
  NormalDistanceVoxelMap(dim, voxel_side_length, map_type), 
  InverseDistanceVoxelMap(dim, voxel_side_length, map_type) {
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    dev_output = new thrust::device_vector<float>(this->NormalDistanceVoxelMap::m_voxelmap_size);
  }

InheritSignedDistanceVoxelMap::~InheritSignedDistanceVoxelMap(){
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
  }

// InheritSignedDistanceVoxelMap::InheritSignedDistanceVoxelMap(Voxel* dev_data, Voxel* inv_dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
//   NormalDistanceVoxelMap(dev_data, dim, voxel_side_length, map_type), InverseDistanceVoxelMap(inv_dev_data, dim, voxel_side_length, map_type) {};


void InheritSignedDistanceVoxelMap::clearMaps(){
  this->NormalDistanceVoxelMap::clearMap();
  this->InverseDistanceVoxelMap::clearMap();
}


void InheritSignedDistanceVoxelMap::occupancyMerge(boost::shared_ptr<ProbVoxelMap> maintainedProbVoxmap, float occupancy_threshold, float free_threshold){
  this->NormalDistanceVoxelMap::mergeOccupied(maintainedProbVoxmap, Vector3ui(), occupancy_threshold);
  this->InverseDistanceVoxelMap::mergeFree(maintainedProbVoxmap, Vector3ui(), free_threshold);
}


NormalDistanceVoxelMap* InheritSignedDistanceVoxelMap::getNormal(){
  NormalDistanceVoxelMap* ptr = this; 
  return ptr;
}


InverseDistanceVoxelMap* InheritSignedDistanceVoxelMap::getInverse(){
  InverseDistanceVoxelMap* ptr = this; 
  return ptr;
}

// void InheritSignedDistanceVoxelMap::getSignedDistancesToHost(std::vector<float>& host_result_map){
//   this->NormalDistanceVoxelMap::getSignedDistancesToHost(boost::dynamic_pointer_cast<DistanceVoxelMap>(GpuVoxelsMapSharedPtr(getInverse())), host_result_map);
// }

void InheritSignedDistanceVoxelMap::getSignedDistancesToHost(std::vector<float>& host_result_map)
{  
  
  thrust::device_ptr<DistanceVoxel> voxel_begin(this->NormalDistanceVoxelMap::m_dev_data);
  thrust::device_ptr<DistanceVoxel> voxel_end(this->NormalDistanceVoxelMap::m_dev_data + this->NormalDistanceVoxelMap::m_voxelmap_size);
  thrust::device_ptr<DistanceVoxel> other_voxel_begin(this->InverseDistanceVoxelMap::m_dev_data);  
  // thrust::device_vector<float> dev_output(this->NormalDistanceVoxelMap::m_voxelmap_size);
 
  thrust::counting_iterator<int> count_start(0);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(voxel_begin, count_start)), 
                    thrust::make_zip_iterator(thrust::make_tuple(voxel_end, count_start + this->NormalDistanceVoxelMap::getVoxelMapSize())), 
                    thrust::make_zip_iterator(thrust::make_tuple(other_voxel_begin, count_start)), 
                    dev_output->begin(),
                    SignedDistanceFunctor(this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_voxel_side_length));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  thrust::copy(dev_output->begin(), dev_output->end(), host_result_map.begin());
}

void InheritSignedDistanceVoxelMap::getSignedDistancesAndGradientsToHost(std::vector<VectorSdfGrad>& host_result_map){
  this->NormalDistanceVoxelMap::getSignedDistancesAndGradientsToHost(boost::dynamic_pointer_cast<DistanceVoxelMap>(GpuVoxelsMapSharedPtr(getInverse())), host_result_map);
}

void InheritSignedDistanceVoxelMap::parallelBanding3D(){
  this->NormalDistanceVoxelMap::parallelBanding3D();
  this->InverseDistanceVoxelMap::parallelBanding3D();     
}

void InheritSignedDistanceVoxelMap::parallelBanding3DMark(uint32_t m1, uint32_t m2, uint32_t m3, uint32_t arg_m1_blocksize, uint32_t arg_m2_blocksize, uint32_t arg_m3_blocksize, bool detailtimer) {

  // if (this->NormalDistanceVoxelMap::m_dim.x != this->NormalDistanceVoxelMap::m_dim.y || this->NormalDistanceVoxelMap::m_dim.x % 64)
  // {
  //   LOGGING_ERROR(VoxelmapLog, "parallelBanding3D: dimX and dimY must be equal; they also must be divisible by 64" << endl);
  //   //return; //TODO: check whether this is the right check; why not 32?
  // }

  bool sync_always = detailtimer;
//   if (sync_always); //ifndef IC_PERFORMANCE_MONITOR there would be a compiler warning otherwise

//   //optimise m1,m2,m3; m3 is especially detrimental? (increases divergence)
//   // m2, m3 works on dim.y first, then dim.x after transpose
//   m1 = max(1, min(m1, this->NormalDistanceVoxelMap::m_dim.z)); //band count in phase1
//   m2 = max(1, min(m2, this->NormalDistanceVoxelMap::m_dim.y)); //band count in phase2
//   m3 = max(1, min(m3, this->NormalDistanceVoxelMap::m_dim.y)); //band count in phase3

//   // ensure m_dim.z is multiple of m1
//   if (this->NormalDistanceVoxelMap::m_dim.z % m1) {
//     LOGGING_WARNING(VoxelmapLog, "PBA: m1 does not cleanly divide m_dim.z: " << this->NormalDistanceVoxelMap::m_dim.z << "%" << m1 << " = " << (this->NormalDistanceVoxelMap::m_dim.z % m1) << ", reverting to default m1 = 1" << endl);
//     m1 = 1;
//   }

//   // ensure m_dim.x and m_dim.y are multiples of m2
//   if ((this->NormalDistanceVoxelMap::m_dim.x % m2) || (this->NormalDistanceVoxelMap::m_dim.y % m2)) {
//     LOGGING_WARNING(VoxelmapLog, "PBA: m2 does not cleanly divide m_dim.x and m_dim.y: " << this->NormalDistanceVoxelMap::m_dim.x << "%" << m2 << " = " << (this->NormalDistanceVoxelMap::m_dim.x % m2) << 
//               ", " << this->NormalDistanceVoxelMap::m_dim.y << "%" << m2 << " = " << (this->NormalDistanceVoxelMap::m_dim.y % m2) <<  ", reverting to default m2 = 1" << endl);
//     m2 = 1;
//   }

//   // ensure m_dim.x and m_dim.y are multiples of m3
//   if ((this->NormalDistanceVoxelMap::m_dim.x % m3) || (this->NormalDistanceVoxelMap::m_dim.y % m3)) {
//     LOGGING_WARNING(VoxelmapLog, "PBA: m3 does not cleanly divide m_dim.x and m_dim.y: " << this->NormalDistanceVoxelMap::m_dim.x << "%" << m3 << " = " << (this->NormalDistanceVoxelMap::m_dim.x % m3) << 
//               ", " << this->NormalDistanceVoxelMap::m_dim.y << "%" << m3 << " = " << (this->NormalDistanceVoxelMap::m_dim.y % m3) <<  ", reverting to default m3 = 1" << endl);
//     m3 = 1;
//   }

//   LOGGING_DEBUG(VoxelmapLog, "PBA: m1: " << m1 << ", m2: " << m2 << ", m3: " << m3 << ", detailtimer: " << detailtimer << endl);

// #ifdef IC_PERFORMANCE_MONITOR
//   if (detailtimer) PERF_MON_START("detailtimer");
// #endif

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

// #ifdef IC_PERFORMANCE_MONITOR
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D warmup sync");
//   PERF_MON_START("pbatimer");
// #endif


  thrust::device_ptr<DistanceVoxel> original_begin_3d(this->NormalDistanceVoxelMap::m_dev_data);
  thrust::device_ptr<DistanceVoxel> original_end_3d(this->NormalDistanceVoxelMap::m_dev_data + this->NormalDistanceVoxelMap::m_voxelmap_size);
  thrust::device_ptr<DistanceVoxel> inverse_original_begin_3d(this->InverseDistanceVoxelMap::m_dev_data);
  thrust::device_ptr<DistanceVoxel> inverse_original_end_3d(this->InverseDistanceVoxelMap::m_dev_data + this->InverseDistanceVoxelMap::m_voxelmap_size);
  
  thrust::device_vector<DistanceVoxel> initial_map(original_begin_3d, original_end_3d);
  thrust::device_vector<DistanceVoxel> initial_inverse_map(inverse_original_begin_3d, inverse_original_end_3d);

  thrust::device_ptr<DistanceVoxel> distance_map_begin = original_begin_3d;
  thrust::device_ptr<DistanceVoxel> distance_inverse_map_begin = inverse_original_begin_3d;

// #ifdef IC_PERFORMANCE_MONITOR
//   if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D initialisation device_vector created");
// #endif


// #ifdef IC_PERFORMANCE_MONITOR
//   HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D initialisation done");
//   PERF_MON_PRINT_AND_RESET_INFO_P("pbatimer", "parallelBanding3D init done", "pbaprefix");
// #endif

  // PBA phase 1
  //     optimise: could work as series of simple transforms in one array

  //TODO: ensure blocksize divides m_dim.* evenly

  //in total m1*dim.x*dim.y threads
  //within warp threads should access x-neighbors
  dim3 m1_block_size(min(arg_m1_blocksize, this->NormalDistanceVoxelMap::m_dim.x)); // optimize blocksize
  dim3 m1_grid_size(this->NormalDistanceVoxelMap::m_dim.x / m1_block_size.x, this->NormalDistanceVoxelMap::m_dim.y, m1); //m1 bands


  // if (this->NormalDistanceVoxelMap::getDimensions().x < m1_block_size.x) {
  //   //TODO: check for phase1-3 that m1-3 and blocksizes are always safe and kernels will not cause memory acces violations
  //   LOGGING_ERROR_C(VoxelmapLog, DistanceVoxelMap, "ERROR: PBA require dimensions.x >= PBA_BLOCKSIZE (" << PBA_BLOCKSIZE << ")" << endl);
  // }
  // HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  //flood foward and backward within bands
  //there are m1 vertical bands
  //TODO optimise: could work in-place
  kernelPBAphase1FloodZ
      <<< m1_grid_size, m1_block_size, 0, s1>>>
      (distance_map_begin, distance_map_begin, this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_dim.z / m1); //distance_map is output
  

  kernelPBAphase1FloodZ
      <<< m1_grid_size, m1_block_size, 0, s2 >>>
      (distance_inverse_map_begin, distance_inverse_map_begin, this->InverseDistanceVoxelMap::m_dim, this->InverseDistanceVoxelMap::m_dim.z / m1); //distance_map is output
  
  // HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//   CHECK_CUDA_ERROR();
//   // -> blöcke enthalten gelbe vertikale balken, solange min 1 obstacle enthalten

// #ifdef IC_PERFORMANCE_MONITOR
//   if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 1 floodZ done");
// #endif

  //pass information between bands
  //optimise: propagate and update could be in same kernel
  // if (m1 > 1) {
  //   kernelPBAphase1PropagateInterband
  //       <<< m1_grid_size, m1_block_size, 0, s1 >>>
  //       (distance_map_begin, initial_map.begin(), this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_dim.z / m1); //buffer b to a
  //   kernelPBAphase1PropagateInterband
  //       <<< m1_grid_size, m1_block_size, 0, s2 >>>
  //       (distance_inverse_map_begin, initial_inverse_map.begin(), this->InverseDistanceVoxelMap::m_dim, this->InverseDistanceVoxelMap::m_dim.z / m1); //buffer b to a
  //   // CHECK_CUDA_ERROR();
  //   // -> initial_map enthält obstacle infos und interband head/tail infos
  // }

// #ifdef IC_PERFORMANCE_MONITOR
//   if (m1 > 1) {
//     if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//     if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 1 interband done");
//   }
// #endif

  // if (m1 > 1) {
  //   kernelPBAphase1Update
  //         <<< m1_grid_size, m1_block_size, 0, s1  >>>
  //         (initial_map.begin(), distance_map_begin, this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_dim.z / m1); //buffer to b; a is Links (top,bottom), b is Color (voxel)
  //   kernelPBAphase1Update
  //         <<< m1_grid_size, m1_block_size, 0, s2  >>>
  //         (initial_inverse_map.begin(), distance_inverse_map_begin, this->InverseDistanceVoxelMap::m_dim, this->InverseDistanceVoxelMap::m_dim.z / m1); //buffer to b; a is Links (top,bottom), b is Color (voxel)
  //   CHECK_CUDA_ERROR();
  // }
  // end of phase 1: distance_map contains the S_ij obstacle information

// #ifdef IC_PERFORMANCE_MONITOR
//   if (m1 > 1) {
//     if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//     if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 1 update done");
//   }
// #endif

  // compute proximate points locally in each band
  dim3 m2_block_size(min(arg_m2_blocksize, min(this->NormalDistanceVoxelMap::m_dim.x, this->NormalDistanceVoxelMap::m_dim.y))); // optimize blocksize
  dim3 m2_grid_size = dim3(this->NormalDistanceVoxelMap::m_dim.x / m2_block_size.x, m2, this->NormalDistanceVoxelMap::m_dim.z); // m2 bands per column
  
  // if ((this->NormalDistanceVoxelMap::getDimensions().x < arg_m2_blocksize) || (this->NormalDistanceVoxelMap::getDimensions().y < arg_m2_blocksize)) {
  //   //TODO: check for phase1-3 that m1-3 and blocksizes are always safe and kernels will not cause memory acces violations
  //   LOGGING_ERROR_C(VoxelmapLog, DistanceVoxelMap, "ERROR: PBA requires dimensions.x and .y >= arg_m2_blocksize (" << arg_m2_blocksize << ")" << endl);
  // }

  kernelPBAphase2ProximateBackpointers
      <<< m2_grid_size, m2_block_size, 0, s1  >>>
     (distance_map_begin, initial_map.begin(), this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_dim.y / m2); //output stack/singly linked list with backpointers; some elements are skipped
  kernelPBAphase2ProximateBackpointers
      <<< m2_grid_size, m2_block_size, 0, s2  >>>
     (distance_inverse_map_begin, initial_inverse_map.begin(), this->InverseDistanceVoxelMap::m_dim, this->InverseDistanceVoxelMap::m_dim.y / m2); //output stack/singly linked list with backpointers; some elements are skipped
  // CHECK_CUDA_ERROR();

// #ifdef IC_PERFORMANCE_MONITOR
//   if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 2 backpointer done");
// #endif

  // distance_map will be shadowed by an array of int16 for CreateForward and MergeBands
//  thrust::device_ptr<pba_fw_ptr_t> forward_ptrs_begin((pba_fw_ptr_t*)(distance_map_begin.get()));

  thrust::device_ptr<pba_fw_ptr_t> forward_ptrs_begin((pba_fw_ptr_t*)(distance_map_begin.get()));
  thrust::device_ptr<pba_fw_ptr_t> inverse_forward_ptrs_begin((pba_fw_ptr_t*)(distance_inverse_map_begin.get()));

  // if (m2 > 1) {
  //   kernelPBAphase2CreateForwardPointers
  //       <<< m2_grid_size, m2_block_size, 0, s1  >>>
  //       (initial_map.begin(), forward_ptrs_begin, this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_dim.y / m2); //read stack, write forward pointers
  //   kernelPBAphase2CreateForwardPointers
  //       <<< m2_grid_size, m2_block_size, 0, s2  >>>
  //       (initial_inverse_map.begin(), inverse_forward_ptrs_begin, this->InverseDistanceVoxelMap::m_dim, this->InverseDistanceVoxelMap::m_dim.y / m2); //read stack, write forward pointers    
  //   // CHECK_CUDA_ERROR();
  // }

// #ifdef IC_PERFORMANCE_MONITOR
//   if (m2 > 1) {
//     if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//     if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 2 forward done");
//   }
// #endif

  // repeatedly merge two bands into one
  for (int band_count = m2; band_count > 1; band_count /= 2) {
    dim3 m2_merge_grid_size = dim3(this->NormalDistanceVoxelMap::m_dim.x / m2_block_size.x, band_count / 2, this->NormalDistanceVoxelMap::m_dim.z);

    kernelPBAphase2MergeBands
        <<< m2_merge_grid_size, m2_block_size, 0, s1  >>>
        (initial_map.begin(), forward_ptrs_begin, this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_dim.y / band_count); //update both stack and forward_ptrs
    kernelPBAphase2MergeBands
        <<< m2_merge_grid_size, m2_block_size, 0, s2  >>>
        (initial_inverse_map.begin(), inverse_forward_ptrs_begin, this->InverseDistanceVoxelMap::m_dim, this->InverseDistanceVoxelMap::m_dim.y / band_count); //update both stack and forward_ptrs    
    // CHECK_CUDA_ERROR();

    // if (detailtimer) LOGGING_INFO(VoxelmapLog, "kernelPBAphase2MergeBands finished merging with band_size " << (this->NormalDistanceVoxelMap::m_dim.y / band_count) << endl);

// #ifdef IC_PERFORMANCE_MONITOR
//     if (m2 > 1) {
//       if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//       if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 2 merge iteration done");
//     }
// #endif
  }
  // end of phase 2: initial_ contains P_i information; y coordinates were replaced by back-pointers; y coordinate is implicitly equal to voxel position.y

  // TODO: benchmark and/or delete texture usage: (initialResDesc, texDesc and initialTexObj)
  //TODO: use template specialisation to implement; run once with and without textures

  // Specify texture
  struct cudaResourceDesc initialResDesc;
  memset(&initialResDesc, 0, sizeof(initialResDesc));
  initialResDesc.resType = cudaResourceTypeLinear;
  initialResDesc.res.linear.devPtr = thrust::raw_pointer_cast(initial_map.data());
  initialResDesc.res.linear.sizeInBytes = initial_map.size()*sizeof(int);

  //TODO!
  initialResDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
  initialResDesc.res.linear.desc.x = 32; // bits per channel

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  cudaTextureObject_t initialTexObj = 0;
  cudaCreateTextureObject(&initialTexObj, &initialResDesc, &texDesc, NULL);

  // Inverse
  struct cudaResourceDesc initialInverseResDesc;
  memset(&initialInverseResDesc, 0, sizeof(initialInverseResDesc));
  initialInverseResDesc.resType = cudaResourceTypeLinear;
  initialInverseResDesc.res.linear.devPtr = thrust::raw_pointer_cast(initial_inverse_map.data());
  initialInverseResDesc.res.linear.sizeInBytes = initial_inverse_map.size()*sizeof(int);

  //TODO!
  initialInverseResDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
  initialInverseResDesc.res.linear.desc.x = 32; // bits per channel

  struct cudaTextureDesc texDescInverse;
  memset(&texDescInverse, 0, sizeof(texDescInverse));
  texDescInverse.addressMode[0] = cudaAddressModeClamp;
  texDescInverse.filterMode = cudaFilterModePoint;
  texDescInverse.readMode = cudaReadModeElementType;
  texDescInverse.normalizedCoords = 0;

  // Create texture object
  cudaTextureObject_t initialTexObjInverse = 0;
  cudaCreateTextureObject(&initialTexObjInverse, &initialInverseResDesc, &texDescInverse, NULL);



  dim3 m3_block_size(min(arg_m3_blocksize, min(this->NormalDistanceVoxelMap::m_dim.x, this->NormalDistanceVoxelMap::m_dim.y)), m3); // y bands; block_size threads in total
  dim3 m3_grid_size = dim3(this->NormalDistanceVoxelMap::m_dim.x / m3_block_size.x, 1, this->NormalDistanceVoxelMap::m_dim.z);

//  // phase 3: read from input_, write to distance_map
//  //optimise: scale PBA_M3_BLOCKX to m3; PBA_M3_BLOCKX*m3 should not be too small

  // if ((this->NormalDistanceVoxelMap::getDimensions().x < arg_m3_blocksize) || (this->NormalDistanceVoxelMap::getDimensions().y < arg_m3_blocksize)) {
  //   //TODO: check for phase1-3 that m1-3 and blocksizes are always safe and kernels will not cause memory acces violations
  //   LOGGING_ERROR_C(VoxelmapLog, DistanceVoxelMap, "ERROR: PBA requires dimensions.x and .y >= arg_m2_blocksize (" << arg_m2_blocksize << ")" << endl);
  // }
  //distance map is write-only during phase3
  kernelPBAphase3Distances
      <<< m3_grid_size, m3_block_size, 0, s1  >>>
        (initialTexObj, distance_map_begin, this->NormalDistanceVoxelMap::m_dim);
  kernelPBAphase3Distances
      <<< m3_grid_size, m3_block_size, 0, s2  >>>
        (initialTexObjInverse, distance_inverse_map_begin, this->InverseDistanceVoxelMap::m_dim);  
        
  // CHECK_CUDA_ERROR();
      //  (initial_map.begin(), distance_map_begin, this->m_dim);
  // phase 3 done: distance_map contains final result

// #ifdef IC_PERFORMANCE_MONITOR
//   if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 3 done");
// #endif

  dim3 transpose_block(PBA_TILE_DIM, PBA_TILE_DIM);
  dim3 transpose_grid(this->NormalDistanceVoxelMap::m_dim.x / transpose_block.x, this->NormalDistanceVoxelMap::m_dim.y / transpose_block.y, this->NormalDistanceVoxelMap::m_dim.z); //maximum blockDim.y/z is 64K
  
  // transpose x/y within every z-layer; need to transpose obstacle coordinates as well
  // transpose in-place to reuse input/ouput scheme of phase 2&3
  // optimise: check bare transpose performance in-place vs non-inplace
  //TODO: ensure m_dim x/y divisible by PBA_TILE_DIM
  kernelPBA3DTransposeXY<<<transpose_grid, transpose_block, 0, s1 >>>
                        (distance_map_begin); //optimise: remove thrust wrapper?
  kernelPBA3DTransposeXY<<<transpose_grid, transpose_block, 0, s2 >>>
                        (distance_inverse_map_begin); //optimise: remove thrust wrapper?                        
  // CHECK_CUDA_ERROR();

// #ifdef IC_PERFORMANCE_MONITOR
//   if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first transpose done");
// #endif

  //second phase2&3:
  // setze m2_grid_size erneut!
  // compute proximate points locally in each band

  kernelPBAphase2ProximateBackpointers
      <<< m2_grid_size, m2_block_size, 0, s1  >>>
     (distance_map_begin, initial_map.begin(), this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_dim.y / m2); //output stack/singly linked list with backpointers; some elements are skipped
  kernelPBAphase2ProximateBackpointers
      <<< m2_grid_size, m2_block_size, 0, s2  >>>
     (distance_inverse_map_begin, initial_inverse_map.begin(), this->InverseDistanceVoxelMap::m_dim, this->InverseDistanceVoxelMap::m_dim.y / m2); //output stack/singly linked list with backpointers; some elements are skipped  
  
//   CHECK_CUDA_ERROR();

// #ifdef IC_PERFORMANCE_MONITOR
//   if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second phase 2 backpointers done");
// #endif


  // if (m2 > 1) {
  //   kernelPBAphase2CreateForwardPointers
  //       <<< m2_grid_size, m2_block_size, 0, s1  >>>
  //       (initial_map.begin(), forward_ptrs_begin, this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_dim.y / m2); //read stack, write forward pointers
  //   kernelPBAphase2CreateForwardPointers
  //       <<< m2_grid_size, m2_block_size, 0, s2  >>>
  //       (initial_inverse_map.begin(), inverse_forward_ptrs_begin, this->InverseDistanceVoxelMap::m_dim, this->InverseDistanceVoxelMap::m_dim.y / m2); //read stack, write forward pointers    
  //   // CHECK_CUDA_ERROR();
  // }

// #ifdef IC_PERFORMANCE_MONITOR
//   if (m2 > 1) {
//     if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//     if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second phase 2 forwardpointers done");
//   }
// #endif


  // repeatedly merge two bands into one
  for (int band_count = m2; band_count > 1; band_count /= 2) {
    dim3 m2_merge_grid_size = dim3(this->NormalDistanceVoxelMap::m_dim.x / m2_block_size.x, band_count / 2, this->NormalDistanceVoxelMap::m_dim.z);
    kernelPBAphase2MergeBands
        <<< m2_merge_grid_size, m2_block_size, 0, s1  >>>
        (initial_map.begin(), forward_ptrs_begin, this->NormalDistanceVoxelMap::m_dim, this->NormalDistanceVoxelMap::m_dim.y / band_count); //update both stack and forward_ptrs
    kernelPBAphase2MergeBands
        <<< m2_merge_grid_size, m2_block_size, 0, s2  >>>
        (initial_inverse_map.begin(), inverse_forward_ptrs_begin, this->InverseDistanceVoxelMap::m_dim, this->InverseDistanceVoxelMap::m_dim.y / band_count); //update both stack and forward_ptrs    
        
    // CHECK_CUDA_ERROR();

    // if (detailtimer) LOGGING_INFO(VoxelmapLog, "kernelPBAphase2MergeBands finished merging with band_size " << (this->NormalDistanceVoxelMap::m_dim.y / band_count) << endl);


// #ifdef IC_PERFORMANCE_MONITOR
//     if (m2 > 1) {
//       if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//       if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second phase2 merge iteration done");
//     }
// #endif
  }

  // end of phase 2: initial_ contains P_i information; y coordinates were replaced by back-pointers; y coordinate is implicitly equal to voxel position.y
  // phase 3: read from input_, write to distance_map
  //optimise: scale PBA_M3_BLOCKX to m3; PBA_M3_BLOCKX*m3 should not be too small
  kernelPBAphase3Distances
      <<< m3_grid_size, m3_block_size, 0, s1 >>>
      (initialTexObj, distance_map_begin, this->NormalDistanceVoxelMap::m_dim);
  kernelPBAphase3Distances
      <<< m3_grid_size, m3_block_size, 0, s2 >>>
      (initialTexObjInverse, distance_inverse_map_begin, this->InverseDistanceVoxelMap::m_dim);  
  // CHECK_CUDA_ERROR();
  // phase 3 done: distance_map contains final result

  //second phase2&3 done

// #ifdef IC_PERFORMANCE_MONITOR
//   if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second phase3 done");
// #endif

  kernelPBA3DTransposeXY<<<transpose_grid, transpose_block, 0, s1 >>>
                        (distance_map_begin);
  kernelPBA3DTransposeXY<<<transpose_grid, transpose_block, 0, s2 >>>
                        (distance_inverse_map_begin);  
  
  // CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Destroy texture object
  cudaDestroyTextureObject(initialTexObj);
  cudaDestroyTextureObject(initialTexObjInverse);

// #ifdef IC_PERFORMANCE_MONITOR
//   if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second transpose done");
//   PERF_MON_PRINT_AND_RESET_INFO_P("pbatimer", "parallelBanding3D compute done", "pbaprefix");
// //  PERF_MON_SUMMARY_ALL_INFO;
// #endif

}


} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif // INHERITSIGNEDDISTANCEVOXELMAP_HPP

