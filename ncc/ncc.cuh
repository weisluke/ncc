#pragma once

#include "array_functions.cuh"
#include "complex.cuh"
#include "ncc_functions.cuh"
#include "stopwatch.hpp"
#include "util.cuh"

#include <thrust/execution_policy.h> //for thrust::device
#include <thrust/extrema.h> // for thrust::min_element, thrust::max_element
#include <thrust/reduce.h> // for thrust::reduce

#include <chrono> //for setting random seed with clock
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <string>


template <typename T>
class NCC
{

public:
	/******************************************************************************
	default input variables
	******************************************************************************/
	std::string infile_prefix = "./";
	Complex<T> center_y = Complex<T>();
	Complex<T> half_length_y = Complex<T>(5, 5);
	Complex<int> num_pixels_y = Complex<int>(1000, 1000);
	int over_sample = 2;
	int write_maps = 1;
	int write_histograms = 1;
	std::string outfile_prefix = "./";


	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	NCC()
	{

	}


private:
	/******************************************************************************
	constant variables
	******************************************************************************/
	const std::string caustics_file = "ccf_caustics.bin";
	const std::string outfile_type = ".bin";

	/******************************************************************************
	variables for cuda device, kernel threads, and kernel blocks
	******************************************************************************/
	cudaDeviceProp cuda_device_prop;
	dim3 threads;
	dim3 blocks;

	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;
	double t_elapsed;
	double t_ncc;
	double t_reduce;

	/******************************************************************************
	derived variables
	******************************************************************************/
	int num_rows = 0;
	int num_cols = 0;

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	Complex<T>* caustics = nullptr;
	int* num_crossings = nullptr;

	int min_num = std::numeric_limits<int>::max();
	int max_num = 0;
	int histogram_length = 0;
	int* histogram = nullptr;

	

	bool set_cuda_devices(bool verbose)
	{
		/******************************************************************************
		check that a CUDA capable device is present
		******************************************************************************/
		int n_devices = 0;

		cudaGetDeviceCount(&n_devices);
		if (cuda_error("cudaGetDeviceCount", false, __FILE__, __LINE__)) return false;

		if (n_devices < 1)
		{
			std::cerr << "Error. No CUDA capable devices detected.\n";
			return false;
		}

		if (verbose)
		{
			std::cout << "Available CUDA capable devices:\n\n";

			for (int i = 0; i < n_devices; i++)
			{
				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, i);
				if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return false;

				show_device_info(i, prop);
			}
		}

		if (n_devices > 1)
		{
			std::cout << "More than one CUDA capable device detected. Defaulting to first device.\n\n";
		}
		cudaSetDevice(0);
		if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return false;
		cudaGetDeviceProperties(&cuda_device_prop, 0);
		if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return false;

		return true;
	}

	bool check_input_params(bool verbose)
	{
		std::cout << "Checking input parameters...\n";

		
		if (half_length_y.re < std::numeric_limits<T>::min() || half_length_y.im < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. half_length_y1 and half_length_y2 must both be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (num_pixels_y.re < 1 || num_pixels_y.im < 1)
		{
			std::cerr << "Error. num_pixels_y1 and num_pixels_y2 must both be integers > 0\n";
			return false;
		}

		if (over_sample < 0)
		{
			std::cerr << "Error. over_sample must be an integer >= 0\n";
			return false;
		}

		if (write_maps != 0 && write_maps != 1)
		{
			std::cerr << "Error. write_maps must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_histograms != 0 && write_histograms != 1)
		{
			std::cerr << "Error. write_histograms must be 1 (true) or 0 (false).\n";
			return false;
		}


		std::cout << "Done checking input parameters.\n\n";
		
		return true;
	}
	
	bool read_caustics(bool verbose)
	{
		std::cout << "Reading in caustics...\n";
		stopwatch.start();

		std::string fname = infile_prefix + caustics_file;

		if (!read_complex_array<T>(caustics, num_rows, num_cols, fname))
		{
			std::cerr << "Error. Unable to read caustics from file " << fname << "\n";
			return false;
		}

		set_param("num_rows", num_rows, num_rows, verbose);
		set_param("num_cols", num_cols, num_cols, verbose);
		
		if (num_rows < 1 || num_cols < 2)
		{
			std::cerr << "Error. File " << fname << " does not contain valid values for num_rows and num_cols.\n";
			return false;
		}

		t_elapsed = stopwatch.stop();
		std::cout << "Done reading in caustics. Elapsed time: " << t_elapsed << " seconds.\n\n";

		return true;
	}

	bool allocate_initialize_memory(bool verbose)
	{
		std::cout << "Allocating memory...\n";
		stopwatch.start();

		/******************************************************************************
		increase the number of pixels by 2^over_sample for initial sampling
		******************************************************************************/
		num_pixels_y.re <<= over_sample;
		num_pixels_y.im <<= over_sample;

		/******************************************************************************
		allocate memory for pixels
		******************************************************************************/
		cudaMallocManaged(&num_crossings, num_pixels_y.re * num_pixels_y.im * sizeof(int));
		if (cuda_error("cudaMallocManaged(*num_crossings)", false, __FILE__, __LINE__)) return false;

		std::cout << "Done allocating memory.\n\n";


		/******************************************************************************
		initialize pixel values
		******************************************************************************/
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_pixels_y.re, num_pixels_y.im);

		std::cout << "Initializing pixel values...\n";
		stopwatch.start();

		initialize_array_kernel<int> <<<blocks, threads>>> (num_crossings, num_pixels_y.im, num_pixels_y.re);
		if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		std::cout << "Done initializing pixel values. Elapsed time: " << t_elapsed << " seconds.\n\n";

		return true;
	}

	bool calculate_num_caustic_crossings(bool verbose)
	{
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_rows, num_cols - 1);

		unsigned long long int* percentage = nullptr;
		cudaMallocManaged(&percentage, sizeof(unsigned long long int));
		if (cuda_error("cudaMallocManaged(*percentage)", false, __FILE__, __LINE__)) return false;

		*percentage = 1;

		/******************************************************************************
		calculate number of caustic crossings and calculate time taken in seconds
		******************************************************************************/
		std::cout << "Calculating number of caustic crossings...\n";
		stopwatch.start();
		find_num_caustic_crossings_kernel<T> <<<blocks, threads>>> (caustics, num_rows, num_cols, center_y, half_length_y, num_crossings, num_pixels_y, percentage);
		if (cuda_error("find_num_caustic_crossings_kernel", true, __FILE__, __LINE__)) return false;
		t_ncc = stopwatch.stop();
		std::cout << "\nDone finding number of caustic crossings. Elapsed time: " << t_ncc << " seconds.\n\n";


		/******************************************************************************
		downsample number of caustic crossings and calculate time taken in seconds
		******************************************************************************/
		std::cout << "Downsampling number of caustic crossings...\n";
		stopwatch.start();

		for (int i = 0; i < over_sample; i++)
		{
			print_verbose("Loop " + std::to_string(i + 1) + " / " + std::to_string(over_sample) + "\n", verbose);
			num_pixels_y.re >>= 1;
			num_pixels_y.im >>= 1;

			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels_y.re, num_pixels_y.im);

			reduce_pix_array_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels_y);
			if (cuda_error("reduce_pix_array_kernel", true, __FILE__, __LINE__)) return false;

			set_threads(threads, 512);
			set_blocks(threads, blocks, num_pixels_y.im);
			for (int j = 1; j < num_pixels_y.re; j++)
			{
				shift_pix_column_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels_y, j);
			}
			set_threads(threads, 512);
			set_blocks(threads, blocks, num_pixels_y.re);
			for (int j = 1; j < num_pixels_y.im; j++)
			{
				shift_pix_row_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels_y, j);
			}
			if (cuda_error("shift_pix_kernel", true, __FILE__, __LINE__)) return false;
		}
		double t_reduce = stopwatch.stop();
		std::cout << "Done downsampling number of caustic crossings. Elapsed time: " << t_reduce << " seconds.\n\n";
		
		min_num = *thrust::min_element(thrust::device, num_crossings, num_crossings + num_pixels_y.re * num_pixels_y.im);
		if (min_num < 0)
		{
			std::cerr << "Error. Number of caustic crossings should be >= 0\n";
			return false;
		}

		return true;
	}

	bool create_histograms(bool verbose)
	{
		/******************************************************************************
		create histograms of pixel values
		******************************************************************************/

		if (write_histograms)
		{
			std::cout << "Creating histograms...\n";
			stopwatch.start();

			min_num = *thrust::min_element(thrust::device, num_crossings, num_crossings + num_pixels_y.re * num_pixels_y.im);
			max_num = *thrust::max_element(thrust::device, num_crossings, num_crossings + num_pixels_y.re * num_pixels_y.im);

			histogram_length = max_num - min_num + 1;

			cudaMallocManaged(&histogram, histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*histogram)", false, __FILE__, __LINE__)) return false;

			set_threads(threads, 512);
			set_blocks(threads, blocks, histogram_length);

			initialize_array_kernel<int> <<<blocks, threads>>> (histogram, 1, histogram_length);
			if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;

			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels_y.re, num_pixels_y.im);

			histogram_kernel<int> <<<blocks, threads>>> (num_crossings, num_pixels_y, min_num, histogram);
			if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;

			t_elapsed = stopwatch.stop();
			std::cout << "Done creating histograms. Elapsed time: " << t_elapsed << " seconds.\n\n";
		}

		/******************************************************************************
		done creating histograms of pixel values
		******************************************************************************/

		return true;
	}

	bool write_files(bool verbose)
	{
		/******************************************************************************
		stream for writing output files
		set precision to 9 digits
		******************************************************************************/
		std::ofstream outfile;
		outfile.precision(9);
		std::string fname;

		std::cout << "Writing parameter info...\n";
		fname = outfile_prefix + "ncc_parameter_info.txt";
		outfile.open(fname);
		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		outfile << "center_y1 " << center_y.re << "\n";
		outfile << "center_y2 " << center_y.im << "\n";
		outfile << "half_length_y1 " << half_length_y.re << "\n";
		outfile << "half_length_y2 " << half_length_y.im << "\n";
		outfile << "num_pixels_y1 " << num_pixels_y.re << "\n";
		outfile << "num_pixels_y2 " << num_pixels_y.im << "\n";
		outfile << "over_sample " << over_sample << "\n";
		outfile << "t_ncc " << t_ncc << "\n";
		outfile << "t_reduce " << t_reduce << "\n";
		outfile.close();
		std::cout << "Done writing parameter info to file " << fname << "\n\n";


		/******************************************************************************
		histogram of number of caustic crossings map
		******************************************************************************/
		if (write_histograms)
		{
			std::cout << "Writing number of caustic crossings histogram...\n";
			fname = outfile_prefix + "ncc_ncc_numpixels.txt";
			if (!write_histogram<int>(histogram, histogram_length, min_num, fname))
			{
				std::cerr << "Error. Unable to write caustic crossings histogram to file " << fname << "\n";
				return false;
			}
			std::cout << "Done writing number of caustic crossings histogram to file " << fname << "\n\n";
		}


		/******************************************************************************
		write number of caustic crossings
		******************************************************************************/
		if (write_maps)
		{
			std::cout << "Writing number of caustic crossings...\n";
			fname = outfile_prefix + "ncc_ncc" + outfile_type;
			if (!write_array<int>(num_crossings, num_pixels_y.im, num_pixels_y.re, fname))
			{
				std::cerr << "Error. Unable to write number of caustic crossings to file " << fname << "\n";
				return false;
			}
			std::cout << "Done writing number of caustic crossings to file " << fname << "\n\n";
		}

		return true;
	}


public:

	bool run(bool verbose)
	{
		if (!set_cuda_devices(verbose)) return false;
		if (!check_input_params(verbose)) return false;
		if (!read_caustics(verbose)) return false;
		if (!allocate_initialize_memory(verbose)) return false;
		if (!calculate_num_caustic_crossings(verbose)) return false;
		if (!create_histograms(verbose)) return false;

		return true;
	}

	bool save(bool verbose)
	{
		if (!write_files(verbose)) return false;

		return true;
	}

};

