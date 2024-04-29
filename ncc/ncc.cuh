#pragma once

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
	variables for kernel threads and blocks
	******************************************************************************/
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

	

	bool check_input_params(bool verbose)
	{
		std::cout << "Checking input parameters...\n";

		
		if (half_length < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. half_length must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (num_pixels < 1)
		{
			std::cerr << "Error. num_pixels must be an integer > 0\n";
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
		
		if (num_rows < 1 || num_cols < 2)
		{
			std::cerr << "Error. File " << fname << " does not contain valid values for num_rows and num_cols.\n";
			return false;
		}

		t_elapsed = stopwatch.stop();
		std::cout << "Done reading in caustics. Elapsed time: " << t_elapsed << " seconds.\n\n";

		print_verbose("Recentering caustics...\n", verbose);

		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_rows, num_cols);
		recenter_caustics_kernel<T> <<<blocks, threads>>> (caustics, num_rows, num_cols, center_y);
		if (cuda_error("recenter_caustics_kernel", true, __FILE__, __LINE__)) return false;

		print_verbose("Done recentering caustics.\n\n", verbose);

		return true;
	}

	bool allocate_initialize_memory(bool verbose)
	{
		std::cout << "Allocating memory...\n";
		stopwatch.start();

		/******************************************************************************
		increase the number of pixels by 2^over_sample for initial sampling
		******************************************************************************/
		num_pixels <<= over_sample;

		/******************************************************************************
		allocate memory for pixels
		******************************************************************************/
		cudaMallocManaged(&num_crossings, num_pixels * num_pixels * sizeof(int));
		if (cuda_error("cudaMallocManaged(*num_crossings)", false, __FILE__, __LINE__)) return false;

		std::cout << "Done allocating memory.\n\n";


		/******************************************************************************
		initialize pixel values
		******************************************************************************/
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_pixels, num_pixels);

		std::cout << "Initializing pixel values...\n";
		stopwatch.start();

		initialize_array_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels, num_pixels);
		if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		std::cout << "Done initializing pixel values. Elapsed time: " << t_elapsed << " seconds.\n\n";

		return true;
	}

	bool calculate_num_caustic_crossings(bool verbose)
	{
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_rows, num_cols);


		/******************************************************************************
		calculate number of caustic crossings and calculate time taken in seconds
		******************************************************************************/
		std::cout << "Calculating number of caustic crossings...\n";
		stopwatch.start();
		find_num_caustic_crossings_kernel<T> <<<blocks, threads>>> (caustics, num_rows, num_cols, half_length, num_crossings, num_pixels);
		if (cuda_error("find_num_caustic_crossings_kernel", true, __FILE__, __LINE__)) return false;
		t_ncc = stopwatch.stop();
		std::cout << "Done finding number of caustic crossings. Elapsed time: " << t_ncc << " seconds.\n\n";


		/******************************************************************************
		downsample number of caustic crossings and calculate time taken in seconds
		******************************************************************************/
		std::cout << "Downsampling number of caustic crossings...\n";
		stopwatch.start();

		for (int i = 0; i < over_sample; i++)
		{
			print_verbose("Loop " + std::to_string(i + 1) + " / " + std::to_string(over_sample) + "\n", verbose);
			num_pixels >>= 1;

			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels, num_pixels);

			reduce_pix_array_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels);
			if (cuda_error("reduce_pix_array_kernel", true, __FILE__, __LINE__)) return false;

			set_threads(threads, 512);
			set_blocks(threads, blocks, num_pixels);

			for (int j = 1; j < num_pixels; j++)
			{
				shift_pix_column_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels, j);
			}
			for (int j = 1; j < num_pixels; j++)
			{
				shift_pix_row_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels, j);
			}
			if (cuda_error("shift_pix_kernel", true, __FILE__, __LINE__)) return false;
		}
		double t_reduce = stopwatch.stop();
		std::cout << "Done downsampling number of caustic crossings. Elapsed time: " << t_reduce << " seconds.\n\n";

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

			min_num = *thrust::min_element(thrust::device, num_crossings, num_crossings + num_pixels * num_pixels);
			max_num = *thrust::max_element(thrust::device, num_crossings, num_crossings + num_pixels * num_pixels);

			histogram_length = max_num - min_num + 1;

			cudaMallocManaged(&histogram, histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*histogram)", false, __FILE__, __LINE__)) return false;

			set_threads(threads, 512);
			set_blocks(threads, blocks, histogram_length);

			initialize_array_kernel<T> <<<blocks, threads>>> (histogram, 1, histogram_length);
			if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;

			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels, num_pixels);

			histogram_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels, min_num, histogram);
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
		outfile << "center_y1" << center_y.re << "\n";
		outfile << "center_y2" << center_y.im << "\n";
		outfile << "half_length " << half_length << "\n";
		outfile << "num_pixels " << num_pixels << "\n";
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
			if (!write_histogram<T>(histogram, histogram_length, min_num, fname))
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
			if (!write_array<int>(num_crossings, num_pixels, num_pixels, fname))
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
		if (!check_input_params(verbose)) return false;
		if (!read_caustics(verbose)) return false;
		if (!allocate_initialize_memory(verbose)) return false;
		if (!calculate_num_caustic_crossings(verbose)) return false;

		return true;
	}

	bool save(bool verbose)
	{
		if (!create_histograms(verbose)) return false;
		if (!write_files(verbose)) return false;

		return true;
	}

};

