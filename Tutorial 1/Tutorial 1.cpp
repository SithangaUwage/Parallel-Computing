#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <conio.h>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

// Reading the data file into a vector.
// Getting the 6th column of the vector, as that column has the air temp values.
vector<int>*Read_File(std::string filename)
{
	vector<int>*Temp_Linc = new vector<int>;
	ifstream file(filename);
	string Line;
	int Spaces = 0;

	while (std::getline(file, Line))
	{
		std::string temp;
		for (int i = 0; i < Line.size(); i++)
		{
			if (Spaces < 5)
			{
				if (Line[i] == ' ')
				{
					Spaces++;
				}
			}
			else
			{
				temp += Line[i];
			}
		}
		// the air temperatures are floats
		// changing the type to integers, so the kernels would work.
		Temp_Linc->push_back(std::stof(temp) * 100);
		Spaces = 0;
	}
	return Temp_Linc;
}


int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	// The location of the file.
	std::string fileName = "temp_lincolnshire.txt";

	// Reading the file into the Temp_Linc vector.
	vector<int>*Temp_Linc = Read_File(fileName);

	// Getting the total size of the file.
	int Inital_Size = Temp_Linc->size();
	typedef int mytype;

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - memory allocation
		//host - input

		size_t local_size = 10;
		size_t padding_size = Temp_Linc->size() % local_size;

		if (padding_size)
		{
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			Temp_Linc->insert(Temp_Linc->end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = Temp_Linc->size(); // Number of input elements
		size_t input_size = Temp_Linc->size() * sizeof(mytype); // Size in bytes *************
		size_t nr_groups = input_elements / local_size; // number of work groups
		size_t output_size = input_elements * sizeof(mytype); // Size in bytes

		// creating vectors to copy the data to the memory. 
		vector<mytype> B(input_elements);
		vector<mytype> C(input_elements);
		vector<mytype> D(input_elements);
		vector<mytype> E(input_elements);

		// buffer A is only used for reading in the data.
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);

		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, output_size);

		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &(*Temp_Linc)[0]);

		// Zero buffer on device memory for output
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_E, 0, 0, output_size);

		//cl::Kernel kernel_1 = cl::Kernel(program, "reduce_find_min");
		//kernel_1.setArg(0, buffer_A);
		//kernel_1.setArg(1, buffer_B);
		//kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		//cl::Kernel kernel_2 = cl::Kernel(program, "reduce_find_max");
		//kernel_2.setArg(0, buffer_A);
		//kernel_2.setArg(1, buffer_C);
		//kernel_2.setArg(2, cl::Local(local_size * sizeof(mytype)));

		/*cl::Kernel kernel_3 = cl::Kernel(program, "hist_simple");
		kernel_3.setArg(0, buffer_A);
		kernel_3.setArg(1, buffer_D);*/
		//kernel_3.setArg(2, cl::Local(local_size * sizeof(mytype)));

		//cl::Kernel kernel_3 = cl::Kernel(program, "sort_bitonic");
		//kernel_3.setArg(0, buffer_A);

		// the reduction pattern
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_4");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		// finding the minimum value from the data
		cl::Kernel kernel_2 = cl::Kernel(program, "reduce_find_min");
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_C);
		kernel_2.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		// finding the maximum value from the data. 
		cl::Kernel kernel_3 = cl::Kernel(program, "reduce_find_max");
		kernel_3.setArg(0, buffer_A);
		kernel_3.setArg(1, buffer_D);
		kernel_3.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		//cl::Kernel kernel_4 = cl::Kernel(program, "hist_simple");
		//kernel_4.setArg(0, buffer_A);
		//kernel_4.setArg(1, buffer_E);

		//cl::Kernel kernel_4 = cl::Kernel(program, "OE_sort");
		//kernel_4.setArg(0, buffer_A);
		//kernel_4.setArg(1, buffer_E);
		


		//queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL);
		//queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL);
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL);
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL);
		//queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL);
		//queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL);

		//queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		//queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_size, &D[0]);
		//queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, output_size, &E[0]);


		// The result are converted back to a float

		float sum = (float)B[0] / 100.0f;
		cout << "Sum of Data " << sum << endl;
		float minVal = (float)C[0] / 100.0f;
		cout << "Minimum Value " << minVal << endl;
		float maxVal = (float)D[0] / 100.0f;
		cout << "Maximum Value " << maxVal << endl;
		float Sum_AVG = B[0];
		float Average = (Sum_AVG / Temp_Linc->size()) / 100;
		cout << "Average " << Average << endl;

		// histogram and sort only works with small amount of data.
		// large amounts crash the program.

		//cout << E;
		
		//cout << "Histogram = " << E << endl;


		
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	system("pause");
	return 0;
}
