/*******************************************************************************
Vendor: Xilinx
Associated Filename: vadd.cpp
Purpose: VITIS vector addition

*******************************************************************************
Copyright (C) 2019 XILINX, Inc.

This file contains confidential and proprietary information of Xilinx, Inc. and
is protected under U.S. and international copyright and other intellectual
property laws.

DISCLAIMER
This disclaimer is not a license and does not grant any rights to the materials
distributed herewith. Except as otherwise provided in a valid license issued to
you by Xilinx, and to the maximum extent permitted by applicable law:
(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX
HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
in contract or tort, including negligence, or under any other theory of
liability) for any loss or damage of any kind or nature related to, arising under
or in connection with these materials, including for any direct, or any indirect,
special, incidental, or consequential loss or damage (including loss of data,
profits, goodwill, or any type of loss or damage suffered as a result of any
action brought by a third party) even if such damage or loss was reasonably
foreseeable or Xilinx had been advised of the possibility of the same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any
application requiring fail-safe performance, such as life-support or safety
devices or systems, Class III medical devices, nuclear facilities, applications
related to the deployment of airbags, or any other applications that could lead
to death, personal injury, or severe property or environmental damage
(individually and collectively, "Critical Applications"). Customer assumes the
sole risk and liability of any use of Xilinx products in Critical Applications,
subject only to applicable laws and regulations governing limitations on product
liability.

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
ALL TIMES.

*******************************************************************************/
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include "vadd.h"

static const int DATA_SIZE = 36;
static const int TIME_STEP = 36;
static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

int main(int argc, char* argv[]) {

    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}

    char* xclbinFilename = argv[1];
    
    // Compute the size of array in bytes
    size_t size_in_bytes = DATA_SIZE * sizeof(double);
    
    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary
    
    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device " 
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE; 
    }

    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load xclbin 
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);
    
    // This call will get the kernel object from program. A kernel is an 
    // OpenCL function that is executed on the FPGA. 
    cl::Kernel krnl_vector_add(program,"top");
    
    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device. 
    cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, size_in_bytes);
//    cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, size_in_bytes);
    cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, size_in_bytes);
    
    //set the kernel Arguments
    int narg=0;
    krnl_vector_add.setArg(narg++,buffer_a);
//    krnl_vector_add.setArg(narg++,buffer_b);
    krnl_vector_add.setArg(narg++,buffer_result);
    krnl_vector_add.setArg(narg++,TIME_STEP);

    //We then need to map our OpenCL buffers to get the pointers
    double *ptr_a = (double *) q.enqueueMapBuffer (buffer_a , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes);
//    int *ptr_b = (int *) q.enqueueMapBuffer (buffer_b , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes);
    double *ptr_result = (double *) q.enqueueMapBuffer (buffer_result , CL_TRUE , CL_MAP_READ , 0, size_in_bytes);

    //setting input data
    for(int i = 0 ; i< DATA_SIZE; i++){
	    ptr_a[i] = 0.3;
//	    ptr_b[i] = 20;
    }
    const int times=100;
    long int data_start2end_sum=0;
    long int kernel_start2end_sum=0;
    for(int i=0;i<times;i++){
    	using namespace std::chrono;
		time_point<high_resolution_clock> _data_trans_start,_data_trans_end,_kernel_start,_kernel_end;
		_data_trans_start=std::chrono::high_resolution_clock::now();
		// Data will be migrated to kernel space
		q.enqueueMigrateMemObjects({buffer_a},0/* 0 means from host*/);
		_kernel_start=std::chrono::high_resolution_clock::now();
		//Launch the Kernel
		q.enqueueTask(krnl_vector_add);
		_kernel_end=std::chrono::high_resolution_clock::now();
		// The result of the previous kernel execution will need to be retrieved in
		// order to view the results. This call will transfer the data from FPGA to
		// source_results vector
		q.enqueueMigrateMemObjects({buffer_result},CL_MIGRATE_MEM_OBJECT_HOST);
		_data_trans_end=std::chrono::high_resolution_clock::now();
		q.finish();
		long int data_start2end=duration_cast<microseconds>(_data_trans_end-_data_trans_start).count();
		long int kernel_start2end=duration_cast<microseconds>(_kernel_end-_kernel_start).count();
		data_start2end_sum+=data_start2end;
		kernel_start2end_sum+=kernel_start2end;
		printf("res:%f\n",ptr_result[0]);
		printf("data_start2end:%ld\n",data_start2end);
		printf("kernel_start2end:%ld\n",kernel_start2end);
    }
    printf("data_start2end_avg:%f\n",double(data_start2end_sum)/times);
    printf("kernel_start2end_avg:%f\n",double(kernel_start2end_sum)/times);

    printf("without_cl.finish");
	data_start2end_sum=0;
	kernel_start2end_sum=0;
	for(int i=0;i<times;i++){
		using namespace std::chrono;
		time_point<high_resolution_clock> _data_trans_start,_data_trans_end,_kernel_start,_kernel_end;
		_data_trans_start=std::chrono::high_resolution_clock::now();
		// Data will be migrated to kernel space
		q.enqueueMigrateMemObjects({buffer_a},0/* 0 means from host*/);
		_kernel_start=std::chrono::high_resolution_clock::now();
		//Launch the Kernel
		q.enqueueTask(krnl_vector_add);
		_kernel_end=std::chrono::high_resolution_clock::now();
		// The result of the previous kernel execution will need to be retrieved in
		// order to view the results. This call will transfer the data from FPGA to
		// source_results vector
		q.enqueueMigrateMemObjects({buffer_result},CL_MIGRATE_MEM_OBJECT_HOST);
		_data_trans_end=std::chrono::high_resolution_clock::now();
//		q.finish();
		long int data_start2end=duration_cast<microseconds>(_data_trans_end-_data_trans_start).count();
		long int kernel_start2end=duration_cast<microseconds>(_kernel_end-_kernel_start).count();
		data_start2end_sum+=data_start2end;
		kernel_start2end_sum+=kernel_start2end;
		printf("res:%f\n",ptr_result[0]);
		printf("data_start2end:%ld\n",data_start2end);
		printf("kernel_start2end:%ld\n",kernel_start2end);
	}
	printf("data_start2end_avg:%f\n",double(data_start2end_sum)/times);
	printf("kernel_start2end_avg:%f\n",double(kernel_start2end_sum)/times);

    q.enqueueUnmapMemObject(buffer_a , ptr_a);
//    q.enqueueUnmapMemObject(buffer_b , ptr_b);
    q.enqueueUnmapMemObject(buffer_result , ptr_result);
    q.finish();


    return 0;

}