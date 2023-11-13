#pragma once

#include "complex.cuh"
#include "ncc_functions.cuh"
#include "stopwatch.hpp"
#include "util.cuh"

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
	T half_length = static_cast<T>(5);
	int num_pixels = 1000;
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

	int* min_num = nullptr;
	int* max_num = nullptr;
	int histogram_length = 0;
	int* histogram = nullptr;

	

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

		initialize_pixels_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels);
		if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return false;

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

			cudaMallocManaged(&min_num, sizeof(int));
			if (cuda_error("cudaMallocManaged(*min_rays)", false, __FILE__, __LINE__)) return false;
			cudaMallocManaged(&max_num, sizeof(int));
			if (cuda_error("cudaMallocManaged(*max_rays)", false, __FILE__, __LINE__)) return false;

			*min_num = std::numeric_limits<int>::max();
			*max_num = std::numeric_limits<int>::min();

			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels, num_pixels);

			histogram_min_max_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels, min_num, max_num);
			if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return false;

			histogram_length = *max_num - *min_num + 1;

			cudaMallocManaged(&histogram, histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*histogram)", false, __FILE__, __LINE__)) return false;

			set_threads(threads, 512);
			set_blocks(threads, blocks, histogram_length);

			initialize_histogram_kernel<T> <<<blocks, threads>>> (histogram, histogram_length);
			if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return false;

			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels, num_pixels);

			histogram_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels, *min_num, histogram);
			if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;

			std::cout << "Done creating histograms.\n\n";
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
			if (!write_histogram<T>(histogram, histogram_length, *min_num, fname))
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

