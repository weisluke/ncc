/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "ncc.cuh"
#include "util.hpp"

#include <iostream>
#include <limits>
#include <string>


using dtype = double;
NCC<dtype> ncc;

/******************************************************************************
constants to be used
******************************************************************************/
constexpr int OPTS_SIZE = 2 * 10;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-v", "--verbose",
	"-ip", "--infile_prefix",
	"-hl", "--half_length",
	"-px", "--pixels",
	"-os", "--over_sample",
	"-wm", "--write_maps",
	"-wh", "--write_histograms",
	"-ot", "--outfile_type",
	"-o", "--outfile_prefix"
};

/******************************************************************************
default input option values
******************************************************************************/
bool verbose = false;



/******************************************************************************
Print the program usage help message

\param name -- name of the executable
******************************************************************************/
void display_usage(char* name)
{
	if (name)
	{
		std::cout << "Usage: " << name << " opt1 val1 opt2 val2 opt3 val3 ...\n";
	}
	else
	{
		std::cout << "Usage: programname opt1 val1 opt2 val2 opt3 val3 ...\n";
	}
	std::cout
		<< "                                                                               \n"
		<< "Options:\n"
		<< "  -h,--help               Show this help message.\n"
		<< "  -v,--verbose            Toggle verbose output. Takes no option value.\n"
		<< "  -ip,--infile_prefix     Specify the prefix to be used when reading in files.\n"
		<< "                          Default value: " << infile_prefix << "\n"
		<< "  -hl,--half_length       Specify the half-length of the square source plane\n"
		<< "                          region to find the number of caustic crossings in.\n"
		<< "                          Default value: " << half_length << "\n"
		<< "  -px,--pixels            Specify the number of pixels per side for the number\n"
		<< "                          of caustic crossings map. Default value: " << num_pixels << "\n"
		<< "  -os,--over_sample       Specify the power of 2 by which to oversample the\n"
		<< "                          final pixels. E.g., an input of 4 means the final\n"
		<< "                          pixel array will initially be oversampled by a value\n"
		<< "                          of 2^4 = 16 along both axes. This will require\n"
		<< "                          16*16 = 256 times more memory. Default value: " << over_sample << "\n"
		<< "  -wm,--write_maps        Specify whether to write number of caustic crossings\n"
		<< "                          maps (1) or not (0). Default value: " << write_maps << "\n"
		<< "  -wh,--write_histograms  Specify whether to write histograms (1) or not (0).\n"
		<< "                          Default value: " << write_histograms << "\n"
		<< "  -ot,--outfile_type      Specify the type of file to be output. Valid options\n"
		<< "                          are binary (.bin) or text (.txt). Default value: " << outfile_type << "\n"
		<< "  -o,--outfile_prefix     Specify the prefix to be used in output file names.\n"
		<< "                          Default value: " << outfile_prefix << "\n"
		<< "                          Lines of .txt output files are whitespace delimited.\n"
		<< "                          Filenames are:\n"
		<< "                            ncc_parameter_info  various parameter values used\n"
		<< "                                                  in calculations\n"
		<< "                            ncc_ncc_numpixels   each line contains a number of\n"
		<< "                                                  caustic crossings and the\n"
		<< "                                                  number of pixels with that\n"
		<< "                                                  many caustic crossings\n"
		<< "                            ncc_ncc             the first item is num_pixels\n"
		<< "                                                  and the second item is\n"
		<< "                                                  num_pixels followed by the\n"
		<< "                                                  number of caustic crossings\n"
		<< "                                                  for each pixel\n";
}



int main(int argc, char* argv[])
{
	/******************************************************************************
	if help option has been input, display usage message
	******************************************************************************/
	if (cmd_option_exists(argv, argv + argc, "-h") || cmd_option_exists(argv, argv + argc, "--help"))
	{
		display_usage(argv[0]);
		return -1;
	}

	/******************************************************************************
	if there are input options, but not an even number (since all options take a
	parameter), display usage message and exit
	subtract 1 to take into account that first argument array value is program name
	account for possible verbose option, which is a toggle and takes no input
	******************************************************************************/
	if ((argc - 1) % 2 != 0 &&
		!(cmd_option_exists(argv, argv + argc, "-v") || cmd_option_exists(argv, argv + argc, "--verbose")))
	{
		std::cerr << "Error. Invalid input syntax.\n";
		display_usage(argv[0]);
		return -1;
	}

	/******************************************************************************
	check that all options given are valid. use step of 2 since all input options
	take parameters (assumed to be given immediately after the option). start at 1,
	since first array element, argv[0], is program name
	account for possible verbose option, which is a toggle and takes no input
	******************************************************************************/
	for (int i = 1; i < argc; i += 2)
	{
		if (argv[i] == std::string("-v") || argv[i] == std::string("--verbose"))
		{
			verbose = true;
			i--;
			continue;
		}
		if (!cmd_option_valid(OPTS, OPTS + OPTS_SIZE, argv[i]))
		{
			std::cerr << "Error. Invalid input syntax. Unknown option " << argv[i] << "\n";
			display_usage(argv[0]);
			return -1;
		}
	}


	/******************************************************************************
	BEGIN read in options and values, checking correctness and exiting if necessary
	******************************************************************************/

	char* cmdinput = nullptr;

	for (int i = 1; i < argc; i += 2)
	{
		/******************************************************************************
		account for possible verbose option, which is a toggle and takes no input
		******************************************************************************/
		if (argv[i] == std::string("-v") || argv[i] == std::string("--verbose"))
		{
			i--;
			continue;
		}

		cmdinput = cmd_option_value(argv, argv + argc, std::string(argv[i]));

		if (argv[i] == std::string("-ip") || argv[i] == std::string("--infile_prefix"))
		{
			set_param("infile_prefix", infile_prefix, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-hl") || argv[i] == std::string("--half_length"))
		{
			try
			{
				set_param("half_length", half_length, std::stod(cmdinput), verbose);
				if (half_length < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid half_length input. half_length must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid half_length input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-px") || argv[i] == std::string("--pixels"))
		{
			try
			{
				set_param("num_pixels", num_pixels, std::stoi(cmdinput), verbose);
				if (num_pixels < 1)
				{
					std::cerr << "Error. Invalid num_pixels input. num_pixels must be an integer > 0\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_pixels input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-os") || argv[i] == std::string("--over_sample"))
		{
			try
			{
				set_param("over_sample", over_sample, std::stoi(cmdinput), verbose);
				if (over_sample < 0)
				{
					std::cerr << "Error. Invalid over_sample input. over_sample must be an integer >= 0\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid over_sample input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wm") || argv[i] == std::string("--write_maps"))
		{
			try
			{
				set_param("write_maps", write_maps, std::stoi(cmdinput), verbose);
				if (write_maps != 0 && write_maps != 1)
				{
					std::cerr << "Error. Invalid write_maps input. write_maps must be 1 (true) or 0 (false).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_maps input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wh") || argv[i] == std::string("--write_histograms"))
		{
			try
			{
				set_param("write_histograms", write_histograms, std::stoi(cmdinput), verbose);
				if (write_histograms != 0 && write_histograms != 1)
				{
					std::cerr << "Error. Invalid write_histograms input. write_histograms must be 1 (true) or 0 (false).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_histograms input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ot") || argv[i] == std::string("--outfile_type"))
		{
			set_param("outfile_type", outfile_type, make_lowercase(cmdinput), verbose);
			if (outfile_type != ".bin" && outfile_type != ".txt")
			{
				std::cerr << "Error. Invalid outfile_type. outfile_type must be .bin or .txt\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", outfile_prefix, cmdinput, verbose);
		}
	}

	std::cout << "\n";

	/******************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	******************************************************************************/


	/******************************************************************************
	check that a CUDA capable device is present
	******************************************************************************/
	int n_devices = 0;

	cudaGetDeviceCount(&n_devices);
	if (cuda_error("cudaGetDeviceCount", false, __FILE__, __LINE__)) return -1;

	if (verbose)
	{
		std::cout << "Available CUDA capable devices:\n\n";

		for (int i = 0; i < n_devices; i++)
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return -1;

			show_device_info(i, prop);
			std::cout << "\n";
		}
	}

	if (n_devices > 1)
	{
		std::cout << "More than one CUDA capable device detected. Defaulting to first device.\n\n";
	}
	cudaSetDevice(0);
	if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return -1;


	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;


	std::string fname;
	/******************************************************************************
	read in parameter info and store necessary values
	******************************************************************************/
	if (infile_type == ".bin")
	{
		fname = infile_prefix + caustics_file + infile_type;
		std::cout << "Calculating some parameter values from caustics file " << fname << "\n";

		if (!read_params<int>(num_rows, num_cols, fname))
		{
			std::cerr << "Error. Unable to read parameter values from file " << fname << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values from caustics file " << fname << "\n";
	}
	else
	{
		fname = infile_prefix + caustics_parameter_file;
		std::cout << "Calculating some parameter values from parameter info file " << fname << "\n";

		if (!read_params<int>(num_rows, num_cols, fname))
		{
			std::cerr << "Error. Unable to read parameter values from file " << fname << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values from parameter info file " << fname << "\n";
	}
	std::cout << "\n";

	/******************************************************************************
	increase the number of pixels by 2^over_sample for initial sampling
	******************************************************************************/
	num_pixels <<= over_sample;


	/******************************************************************************
	BEGIN memory allocation
	******************************************************************************/

	std::cout << "Beginning memory allocation...\n";

	dtype* xpos = nullptr;
	dtype* ypos = nullptr;
	Complex<dtype>* caustics = nullptr;
	int* num_crossings = nullptr;

	/******************************************************************************
	allocate memory for caustic x positions
	******************************************************************************/
	cudaMallocManaged(&xpos, num_rows * num_cols * sizeof(dtype));
	if (cuda_error("cudaMallocManaged(*xpos)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	allocate memory for caustic y positions
	******************************************************************************/
	cudaMallocManaged(&ypos, num_rows* num_cols * sizeof(dtype));
	if (cuda_error("cudaMallocManaged(*ypos)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	allocate memory for complex caustic positions
	******************************************************************************/
	cudaMallocManaged(&caustics, num_rows* num_cols * sizeof(Complex<dtype>));
	if (cuda_error("cudaMallocManaged(*caustics)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	allocate memory for pixels
	******************************************************************************/
	cudaMallocManaged(&num_crossings, num_pixels * num_pixels * sizeof(int));
	if (cuda_error("cudaMallocManaged(*num_crossings)", false, __FILE__, __LINE__)) return -1;

	std::cout << "Done allocating memory.\n\n";

	/******************************************************************************
	END memory allocation
	******************************************************************************/


	/******************************************************************************
	variables for kernel threads and blocks
	******************************************************************************/
	dim3 threads;
	dim3 blocks;


	/******************************************************************************
	number of threads per block, and number of blocks per grid
	uses empirical values for maximum number of threads and blocks
	******************************************************************************/
	set_threads(threads, 16, 16);
	set_blocks(threads, blocks, num_rows, num_cols);


	/******************************************************************************
	read in caustic positions
	******************************************************************************/
	if (infile_type == ".bin")
	{
		fname = infile_prefix + caustics_file + infile_type;
		std::cout << "Reading caustic positions from file " << fname << "\n";

		if (!read_complex_array<dtype>(caustics, num_rows, num_cols, fname))
		{
			std::cerr << "Error. Unable to read caustic positions from file " << fname << "\n";
			return -1;
		}

		std::cout << "Done reading caustic positions from file " << fname << "\n";
	}
	else
	{
		fname = infile_prefix + caustics_file + "_x" + infile_type;
		std::cout << "Reading caustic x positions from file " << fname << "\n";
		if (!read_re_array<dtype>(xpos, num_rows, num_cols, infile_prefix + caustics_file + "_x" + infile_type))
		{
			std::cerr << "Error. Unable to read caustic x positions from file " << fname << "\n";
			return -1;
		}
		std::cout << "Done reading caustic x positions from file " << fname << "\n";

		fname = infile_prefix + caustics_file + "_y" + infile_type;
		std::cout << "Reading caustic y positions from file " << fname << "\n";
		if (!read_re_array<dtype>(ypos, num_rows, num_cols, fname))
		{
			std::cerr << "Error. Unable to read caustic y positions from file " << fname << "\n";
			return -1;
		}
		std::cout << "Done reading caustic y positions from file " << fname << "\n";

		/******************************************************************************
		copy values into caustic array
		******************************************************************************/
		std::cout << "Copying caustic x and y positions into complex array...\n";
		copy_caustics_kernel<dtype> <<<blocks, threads>>> (xpos, ypos, caustics, num_rows, num_cols);
		if (cuda_error("copy_caustics_kernel", true, __FILE__, __LINE__)) return -1;
		std::cout << "Done copying caustic x and y positions into complex array.\n";
	}
	std::cout << "\n";


	/******************************************************************************
	redefine thread and block size to maximize parallelization
	******************************************************************************/
	set_threads(threads, 16, 16);
	set_blocks(threads, blocks, num_pixels, num_pixels);

	/******************************************************************************
	initialize pixel values
	******************************************************************************/
	print_verbose("Initializing pixel values...\n", verbose);
	initialize_pixels_kernel<dtype> <<<blocks, threads>>> (num_crossings, num_pixels);
	if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return -1;
	print_verbose("Done initializing pixel values.\n\n", verbose);


	/******************************************************************************
	redefine thread and block size to maximize parallelization
	******************************************************************************/
	set_threads(threads, 16, 16);
	set_blocks(threads, blocks, num_rows, num_cols);


	/******************************************************************************
	calculate number of caustic crossings and calculate time taken in seconds
	******************************************************************************/
	std::cout << "Calculating number of caustic crossings...\n";
	stopwatch.start();
	find_num_caustic_crossings_kernel<dtype> <<<blocks, threads>>> (caustics, num_rows, num_cols, half_length, num_crossings, num_pixels);
	if (cuda_error("find_num_caustic_crossings_kernel", true, __FILE__, __LINE__)) return -1;
	double t_ncc = stopwatch.stop();
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

		/******************************************************************************
		redefine thread and block size to maximize parallelization
		******************************************************************************/
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_pixels, num_pixels);

		reduce_pix_array_kernel<dtype> <<<blocks, threads>>> (num_crossings, num_pixels);
		if (cuda_error("reduce_pix_array_kernel", true, __FILE__, __LINE__)) return -1;

		/******************************************************************************
		redefine thread and block size to maximize parallelization
		******************************************************************************/
		set_threads(threads, 512);
		set_blocks(threads, blocks, num_pixels);

		for (int j = 1; j < num_pixels; j++)
		{
			shift_pix_column_kernel<dtype> <<<blocks, threads>>> (num_crossings, num_pixels, j);
		}
		for (int j = 1; j < num_pixels; j++)
		{
			shift_pix_row_kernel<dtype> <<<blocks, threads>>> (num_crossings, num_pixels, j);
		}
		if (cuda_error("shift_pix_kernel", true, __FILE__, __LINE__)) return -1;
	}
	double t_reduce = stopwatch.stop();
	std::cout << "Done downsampling number of caustic crossings. Elapsed time: " << t_reduce << " seconds.\n\n";


	/******************************************************************************
	create histograms of pixel values
	******************************************************************************/

	int* min_num = nullptr;
	int* max_num = nullptr;

	int* histogram = nullptr;

	int histogram_length = 0;

	if (write_histograms)
	{
		std::cout << "Creating histograms...\n";

		cudaMallocManaged(&min_num, sizeof(int));
		if (cuda_error("cudaMallocManaged(*min_rays)", false, __FILE__, __LINE__)) return -1;
		cudaMallocManaged(&max_num, sizeof(int));
		if (cuda_error("cudaMallocManaged(*max_rays)", false, __FILE__, __LINE__)) return -1;

		*min_num = std::numeric_limits<int>::max();
		*max_num = std::numeric_limits<int>::min();

		/******************************************************************************
		redefine thread and block size to maximize parallelization
		******************************************************************************/
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_pixels, num_pixels);

		histogram_min_max_kernel<dtype> <<<blocks, threads>>> (num_crossings, num_pixels, min_num, max_num);
		if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return -1;

		histogram_length = *max_num - *min_num + 1;

		cudaMallocManaged(&histogram, histogram_length * sizeof(int));
		if (cuda_error("cudaMallocManaged(*histogram)", false, __FILE__, __LINE__)) return -1;

		/******************************************************************************
		redefine thread and block size to maximize parallelization
		******************************************************************************/
		set_threads(threads, 512);
		set_blocks(threads, blocks, histogram_length);

		initialize_histogram_kernel<dtype> <<<blocks, threads>>> (histogram, histogram_length);
		if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return -1;

		/******************************************************************************
		redefine thread and block size to maximize parallelization
		******************************************************************************/
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_pixels, num_pixels);

		histogram_kernel<dtype> <<<blocks, threads>>> (num_crossings, num_pixels, *min_num, histogram);
		if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return -1;

		std::cout << "Done creating histograms.\n\n";
	}
	/******************************************************************************
	done creating histograms of pixel values
	******************************************************************************/


	/******************************************************************************
	stream for writing output files
	set precision to 9 digits
	******************************************************************************/
	std::ofstream outfile;
	outfile.precision(9);

	std::cout << "Writing parameter info...\n";
	fname = outfile_prefix + "ncc_parameter_info.txt";
	outfile.open(fname);
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return -1;
	}
	outfile << "half_length " << half_length << "\n";
	outfile << "num_pixels " << num_pixels << "\n";
	outfile << "over_sample " << over_sample << "\n";
	outfile << "t_ncc " << t_ncc << "\n";
	outfile.close();
	std::cout << "Done writing parameter info to file " << fname << "\n\n";


	/******************************************************************************
	histogram of number of caustic crossings map
	******************************************************************************/
	if (write_histograms)
	{
		std::cout << "Writing number of caustic crossings histogram...\n";
		fname = outfile_prefix + "ncc_ncc_numpixels.txt";
		if (!write_histogram<dtype>(histogram, histogram_length, *min_num, fname))
		{
			std::cerr << "Error. Unable to write caustic crossings histogram to file " << fname << "\n";
			return -1;
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
			return -1;
		}
		std::cout << "Done writing number of caustic crossings to file " << fname << "\n\n";
	}

	std::cout << "Done.\n";

	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

