/*****************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

*****************************************************************/


#include "complex.cuh"
#include "ncc_microlensing.cuh"
#include "ncc_read_write_files.cuh"
#include "parse.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <new>
#include <string>



/*constants to be used*/
constexpr int OPTS_SIZE = 2 * 10;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-ip", "--infile_prefix",
	"-it", "--infile_type",
	"-cx","--center_x",
	"-cy", "--center_y",
	"-hl", "--half_length",
	"-px", "--pixels",
	"-wm", "--write_map",
	"-ot", "--outfile_type",
	"-o", "--outfile_prefix"
};


/*default input option values*/
std::string infile_prefix = "./";
std::string infile_type = ".bin";
double cx = 0.0;
double cy = 0.0;
double half_length = 5.0;
int num_pixels = 1000;
int write_map = 1;
std::string outfile_prefix = "./";
std::string outfile_type = ".bin";

/*default variable values*/
const std::string caustics_parameter_file = "ccf_parameter_info.txt";
const std::string caustics_file = "ccf_caustics";

int num_rows = 0;
int num_cols = 0;



/************************************************************
BEGIN structure definitions and function forward declarations
************************************************************/


/************************************
Print the program usage help message

\param name -- name of the executable
************************************/
void display_usage(char* name);

/****************************************************
function to print out progress bar of loops
examples: [====    ] 50%       [=====  ] 62%

\param icurr -- current position in the loop
\param imax -- maximum position in the loop
\param num_bars -- number of = symbols inside the bar
				   default value: 50
****************************************************/
void print_progress(int icurr, int imax, int num_bars = 50);

/*********************************************************************
CUDA error checking

\param name -- to print in error msg
\param sync -- boolean of whether device needs synchronized or not
\param name -- the file being run
\param line -- line number of the source code where the error is given

\return bool -- true for error, false for no error
*********************************************************************/
bool cuda_error(const char* name, bool sync, const char* file, const int line);

/**********************************************************
END structure definitions and function forward declarations
**********************************************************/



int main(int argc, char* argv[])
{
	/*if help option has been input, display usage message*/
	if (cmd_option_exists(argv, argv + argc, std::string("-h")) || cmd_option_exists(argv, argv + argc, std::string("--help")))
	{
		display_usage(argv[0]);
		return -1;
	}

	/*if there are input options, but not an even number (since all options
	take a parameter), display usage message and exit
	subtract 1 to take into account that first argument array value is program name*/
	if ((argc - 1) % 2 != 0)
	{
		std::cerr << "Error. Invalid input syntax.\n";
		display_usage(argv[0]);
		return -1;
	}

	/*check that all options given are valid. use step of 2 since all input
	options take parameters (assumed to be given immediately after the option)
	start at 1, since first array element, argv[0], is program name*/
	for (int i = 1; i < argc; i += 2)
	{
		if (!cmd_option_valid(OPTS, OPTS + OPTS_SIZE, argv[i]))
		{
			std::cerr << "Error. Invalid input syntax.\n";
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
		cmdinput = cmd_option_value(argv, argv + argc, std::string(argv[i]));

		if (argv[i] == std::string("-ip") || argv[i] == std::string("--infile_prefix"))
		{
			infile_prefix = cmdinput;
		}
		else if (argv[i] == std::string("-it") || argv[i] == std::string("--infile_type"))
		{
			infile_type = cmdinput;
			if (infile_type != ".bin" && infile_type != ".txt")
			{
				std::cerr << "Error. Invalid infile_type. infile_type must be .bin or .txt\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-cx") || argv[i] == std::string("--center_x"))
		{
			if (valid_double(cmdinput))
			{
				cx = strtod(cmdinput, nullptr);
			}
			else
			{
				std::cerr << "Error. Invalid center_x input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-cy") || argv[i] == std::string("--center_y"))
		{
			if (valid_double(cmdinput))
			{
				cy = strtod(cmdinput, nullptr);
			}
			else
			{
				std::cerr << "Error. Invalid center_y input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-hl") || argv[i] == std::string("--half_length"))
		{
			if (valid_double(cmdinput))
			{
				half_length = strtod(cmdinput, nullptr);
				if (half_length < std::numeric_limits<double>::min())
				{
					std::cerr << "Error. Invalid half_length input. half_length must be > " << std::numeric_limits<double>::min() << "\n";
					return -1;
				}
			}
			else 
			{
				std::cerr << "Error. Invalid half_length input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-px") || argv[i] == std::string("--pixels"))
		{
			if (valid_double(cmdinput))
			{
				num_pixels = static_cast<int>(strtod(cmdinput, nullptr));
				if (num_pixels < 1)
				{
					std::cerr << "Error. Invalid num_pixels input. num_pixels must be an integer > 0\n";
					return -1;
				}
			}
			else
			{
				std::cerr << "Error. Invalid num_pixels input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wm") || argv[i] == std::string("--write_map"))
		{
			if (valid_double(cmdinput))
			{
				write_map = static_cast<int>(std::strtod(cmdinput, nullptr));
				if (write_map != 0 && write_map != 1)
				{
					std::cerr << "Error. Invalid write_map input. write_map must be 0 (false) or 1 (true).\n";
					return -1;
				}
			}
			else
			{
				std::cerr << "Error. Invalid write_map input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ot") || argv[i] == std::string("--outfile_type"))
		{
			outfile_type = cmdinput;
			if (outfile_type != ".bin" && outfile_type != ".txt")
			{
				std::cerr << "Error. Invalid outfile_type. outfile_type must be .bin or .txt\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			outfile_prefix = cmdinput;
		}
	}

	/****************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	****************************************************************************/


	/*check that a CUDA capable device is present*/
	cudaSetDevice(0);
	if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return -1;


	/*read in parameter info and store necessary values*/
	if (infile_type == ".bin")
	{
		std::cout << "Calculating some parameter values from caustics file " << infile_prefix + caustics_file + infile_type << "\n";

		if (!read_params<int>(num_rows, num_cols, infile_prefix + caustics_file + infile_type))
		{
			std::cerr << "Error. Unable to read parameter values from file " << infile_prefix + caustics_file + infile_type << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values from caustics file " << infile_prefix + caustics_file + infile_type << "\n";
	}
	else
	{
		std::cout << "Calculating some parameter values from parameter info file " << infile_prefix + caustics_parameter_file << "\n";

		if (!read_params<int>(num_rows, num_cols, infile_prefix + caustics_parameter_file))
		{
			std::cerr << "Error. Unable to read parameter values from file " << infile_prefix + caustics_parameter_file << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values from parameter info file " << infile_prefix + caustics_parameter_file << "\n";
	}


	/**********************
	BEGIN memory allocation
	**********************/

	double* xpos = nullptr;
	double* ypos = nullptr;
	Complex<double>* caustics = nullptr;
	int* num_crossings = nullptr;

	cudaMallocManaged(&xpos, num_rows * num_cols * sizeof(double));
	if (cuda_error("cudaMallocManaged(*xpos)", false, __FILE__, __LINE__)) return -1;

	cudaMallocManaged(&ypos, num_rows* num_cols * sizeof(double));
	if (cuda_error("cudaMallocManaged(*ypos)", false, __FILE__, __LINE__)) return -1;

	cudaMallocManaged(&caustics, num_rows* num_cols * sizeof(Complex<double>));
	if (cuda_error("cudaMallocManaged(*caustics)", false, __FILE__, __LINE__)) return -1;

	cudaMallocManaged(&num_crossings, num_pixels * num_pixels * sizeof(int));
	if (cuda_error("cudaMallocManaged(*num_crossings)", false, __FILE__, __LINE__)) return -1;

	/********************
	END memory allocation
	********************/

	std::cout << "\n";
	if (infile_type == ".bin")
	{
		std::cout << "Reading caustic positions from file " << infile_prefix + caustics_file + infile_type << "\n";

		if (!read_complex_array<double>(caustics, num_rows, num_cols, infile_prefix + caustics_file + infile_type))
		{
			std::cerr << "Error. Unable to read caustic positions from file " << infile_prefix + caustics_file + infile_type << "\n";
			return -1;
		}

		std::cout << "Done reading caustic positions from file " << infile_prefix + caustics_file + infile_type << "\n";
	}
	else
	{
		std::cout << "Reading caustic x positions from file " << infile_prefix + caustics_file + "_x" + infile_type << "\n";

		if (!read_re_array<double>(xpos, num_rows, num_cols, infile_prefix + caustics_file + "_x" + infile_type))
		{
			std::cerr << "Error. Unable to read caustic x positions from file " << infile_prefix + caustics_file + "_x" + infile_type << "\n";
			return -1;
		}

		std::cout << "Done reading caustic x positions from file " << infile_prefix + caustics_file + "_x" + infile_type << "\n";


		std::cout << "Reading caustic y positions from file " << infile_prefix + caustics_file + "_y" + infile_type << "\n";

		if (!read_re_array<double>(ypos, num_rows, num_cols, infile_prefix + caustics_file + "_y" + infile_type))
		{
			std::cerr << "Error. Unable to read caustic y positions from file " << infile_prefix + caustics_file + "_y" + infile_type << "\n";
			return -1;
		}

		std::cout << "Done reading caustic y positions from file " << infile_prefix + caustics_file + "_x" + infile_type << "\n";

		/*copy values into caustic array*/
		for (int i = 0; i < num_rows; i++)
		{
			for (int j = 0; j < num_cols; j++)
			{
				caustics[i * num_cols + j] = Complex<double>(xpos[i * num_cols + j], ypos[i * num_cols + j]);
			}
		}
	}


	/*offset based on the center of the desired region*/
	for (int i = 0; i < num_rows; i++)
	{
		for (int j = 0; j < num_cols; j++)
		{
			caustics[i * num_cols + j] -= Complex<double>(cx, cy);
		}
	}
	
	/*initialize pixels to 0*/
	for (int i = 0; i < num_pixels * num_pixels; i++)
	{
		num_crossings[i] = 0;
	}


	/*number of threads per block, and number of blocks per grid
	uses empirical values for maximum number of threads and blocks*/

	int num_threads_y = 16;
	int num_threads_x = 16;

	int num_blocks_y = static_cast<int>((num_cols - 1) / num_threads_y) + 1;
	if (num_blocks_y > 32768 || num_blocks_y < 1)
	{
		num_blocks_y = 32768;
	}

	int num_blocks_x = static_cast<int>((num_rows - 1) / num_threads_x) + 1;
	if (num_blocks_x > 32768 || num_blocks_x < 1)
	{
		num_blocks_x = 32768;
	}
	dim3 blocks(num_blocks_x, num_blocks_y);
	dim3 threads(num_threads_x, num_threads_y);


	/*start and end time for timing purposes*/
	std::chrono::high_resolution_clock::time_point starttime;
	std::chrono::high_resolution_clock::time_point endtime;

	std::cout << "\nCalculating number of caustic crossings...\n";
	/*get current time at start of loop*/
	starttime = std::chrono::high_resolution_clock::now();
	find_num_caustic_crossings_kernel<double> <<<blocks, threads >>> (caustics, num_rows, num_cols, half_length, num_crossings, num_pixels);
	if (cuda_error("find_num_caustic_crossings_kernel", true, __FILE__, __LINE__)) return -1;
	/*get current time at end of loop, and calculate duration in milliseconds*/
	endtime = std::chrono::high_resolution_clock::now();
	double t_ncc = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0;
	std::cout << "Done finding number of caustic crossings. Elapsed time: " << t_ncc << " seconds.\n";


	/*create histogram of pixel values*/

	const int min_ncc = *std::min_element(num_crossings, num_crossings + num_pixels * num_pixels);
	const int max_ncc = *std::max_element(num_crossings, num_crossings + num_pixels * num_pixels);

	int* histogram = new (std::nothrow) int[max_ncc - min_ncc + 1];
	if (!histogram)
	{
		std::cerr << "Error. Memory allocation for *histogram failed.\n";
		return -1;
	}
	for (int i = 0; i <= max_ncc - min_ncc; i++)
	{
		histogram[i] = 0;
	}
	for (int i = 0; i < num_pixels * num_pixels; i++)
	{
		histogram[num_crossings[i] - min_ncc]++;
	}



	/*stream for writing output files
	set precision to 9 digits*/
	std::ofstream outfile;
	outfile.precision(9);

	std::cout << "\nWriting parameter info...\n";
	outfile.open(outfile_prefix + "ncc_parameter_info.txt");
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << outfile_prefix << "ncc_parameter_info.txt\n";
		return -1;
	}
	outfile << "center_x " << cx << "\n";
	outfile << "center_y " << cy << "\n";
	outfile << "half_length " << half_length << "\n";
	outfile << "num_pixels " << num_pixels << "\n";
	outfile << "t_ncc " << t_ncc << "\n";
	outfile.close();
	std::cout << "Done writing parameter info to file " << outfile_prefix << "ncc_parameter_info\n";

	/*histogram of number of caustic crossings map*/
	std::cout << "\nWriting number of caustic crossings histogram to file " << outfile_prefix << "ncc_ncc_numpixels.txt\n";
	outfile.open(outfile_prefix + "ncc_ncc_numpixels.txt");
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << outfile_prefix << "ncc_ncc_numpixels.txt\n";
		return -1;
	}
	for (int i = 0; i <= max_ncc - min_ncc; i++)
	{
		if (histogram[i] != 0)
		{
			outfile << i + min_ncc << " " << histogram[i] << "\n";
		}
	}
	outfile.close();
	std::cout << "Done writing number of caustic crossings histogram to file " << outfile_prefix << "ncc_ncc_numpixels.txt\n";
	

	if (write_map)
	{
		/*write number of caustic crossings*/
		std::cout << "\nWriting number of caustic crossings...\n";
		if (!write_array<int>(num_crossings, num_pixels, num_pixels, outfile_prefix + "ncc_ncc" + outfile_type))
		{
			std::cerr << "Error. Unable to write number of caustic crossings to file " << outfile_prefix << "ncc_ncc" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing number of caustic crossings to file " << outfile_prefix << "ncc_ncc" + outfile_type << "\n";
	}

	std::cout << "\nDone.\n";

	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}



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
	std::cout << "Options:\n"
		<< "   -h,--help           Show this help message\n"
		<< "   -ip,--infile_prefix   Specify the prefix to be used when reading in files.\n"
		<< "                         Default value: " << infile_prefix << "\n"
		<< "   -it,--infile_type     Specify the type of input file to be used when reading\n"
		<< "                         in files. Default value: " << infile_type << "\n"
		<< "   -cx,--center_x        Specify the x coordinate of the center of the square\n"
		<< "                         region to find the number of caustic crossings in.\n"
		<< "                         Default value: " << cx << "\n"
		<< "   -cy,--center_y        Specify the y coordinate of the center of the square\n"
		<< "                         region to find the number of caustic crossings in.\n"
		<< "                         Default value: " << cy << "\n"
		<< "   -hl,--half_length     Specify the half-length of the square region to find\n"
		<< "                         the number of caustic-crossings in. Default value: " << half_length << "\n"
		<< "   -px,--pixels          Specify the number of pixels per side length.\n"
		<< "                         Default value: " << num_pixels << "\n"
		<< "   -wp,--write_map       Specify whether to write number of caustic crossings\n"
		<< "                         map. Default value: " << write_map << "\n"
		<< "   -ot,--outfile_type    Specify the type of file to be output. Valid options\n"
		<< "                         are binary (.bin) or text (.txt). Default value: " << outfile_type << "\n"
		<< "   -o,--outfile          Specify the prefix to be used in output file names.\n"
		<< "                         Default value: " << outfile_prefix << "\n"
		<< "                         Lines of output files are whitespace delimited.\n"
		<< "                         Filenames are:\n"
		<< "                            ncc_parameter_info   various parameter values used\n"
		<< "                                                    in calculations\n"
		<< "                            ncc_ncc_numpixels    each line contains a number of\n"
		<< "                                                    caustic crossings and the\n"
		<< "                                                    number of pixels with that\n"
		<< "                                                    many caustic crossings\n"
		<< "                            ncc_ncc              the first item is num_pixels\n"
		<< "                                                    and the second item is\n"
		<< "                                                    num_pixels followed by the\n"
		<< "                                                    number of caustic crossings\n"
		<< "                                                    at the center of each pixel\n";
}

void print_progress(int icurr, int imax, int num_bars)
{
	std::cout << "\r[";
	for (int i = 0; i < num_bars; i++)
	{
		if (i <= icurr * num_bars / imax)
		{
			std::cout << "=";
		}
		else
		{
			std::cout << " ";
		}
	}
	std::cout << "] " << icurr * 100 / imax << " %";
}

bool cuda_error(const char* name, bool sync, const char* file, const int line)
{
	cudaError_t err = cudaGetLastError();
	/*if the last error message is not a success, print the error code and msg
	and return true (i.e., an error occurred)*/
	if (err != cudaSuccess)
	{
		const char* errMsg = cudaGetErrorString(err);
		std::cerr << "CUDA error check for " << name << " failed at " << file << ":" << line << "\n";
		std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
		return true;
	}
	/*if a device synchronization is also to be done*/
	if (sync)
	{
		/*perform the same error checking as initially*/
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			const char* errMsg = cudaGetErrorString(err);
			std::cerr << "CUDA error check for cudaDeviceSynchronize failed at " << file << ":" << line << "\n";
			std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
			return true;
		}
	}
	return false;
}

