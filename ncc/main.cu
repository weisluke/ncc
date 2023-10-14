/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "ncc.cuh"
#include "util.hpp"

#include <iostream>
#include <limits> //for std::numeric_limits
#include <string>


using dtype = double;
NCC<dtype> ncc;

/******************************************************************************
constants to be used
******************************************************************************/
constexpr int OPTS_SIZE = 2 * 9;
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
		<< "                          Default value: " << ncc.infile_prefix << "\n"
		<< "  -hl,--half_length       Specify the half-length of the square source plane\n"
		<< "                          region to find the number of caustic crossings in.\n"
		<< "                          Default value: " << ncc.half_length << "\n"
		<< "  -px,--pixels            Specify the number of pixels per side for the number\n"
		<< "                          of caustic crossings map. Default value: " << ncc.num_pixels << "\n"
		<< "  -os,--over_sample       Specify the power of 2 by which to oversample the\n"
		<< "                          final pixels. E.g., an input of 4 means the final\n"
		<< "                          pixel array will initially be oversampled by a value\n"
		<< "                          of 2^4 = 16 along both axes. This will require\n"
		<< "                          16*16 = 256 times more memory. Default value: " << ncc.over_sample << "\n"
		<< "  -wm,--write_maps        Specify whether to write number of caustic crossings\n"
		<< "                          maps (1) or not (0). Default value: " << ncc.write_maps << "\n"
		<< "  -wh,--write_histograms  Specify whether to write histograms (1) or not (0).\n"
		<< "                          Default value: " << ncc.write_histograms << "\n"
		<< "  -o,--outfile_prefix     Specify the prefix to be used in output file names.\n"
		<< "                          Default value: " << ncc.outfile_prefix << "\n";
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
			set_param("infile_prefix", ncc.infile_prefix, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-hl") || argv[i] == std::string("--half_length"))
		{
			try
			{
				set_param("half_length", ncc.half_length, std::stod(cmdinput), verbose);
				if (ncc.half_length < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid half_length input. half_length must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
				set_param("num_pixels", ncc.num_pixels, std::stoi(cmdinput), verbose);
				if (ncc.num_pixels < 1)
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
				set_param("over_sample", ncc.over_sample, std::stoi(cmdinput), verbose);
				if (ncc.over_sample < 0)
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
				set_param("write_maps", ncc.write_maps, std::stoi(cmdinput), verbose);
				if (ncc.write_maps != 0 && ncc.write_maps != 1)
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
				set_param("write_histograms", ncc.write_histograms, std::stoi(cmdinput), verbose);
				if (ncc.write_histograms != 0 && ncc.write_histograms != 1)
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
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", ncc.outfile_prefix, cmdinput, verbose);
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
	run and save files
	******************************************************************************/
	if (!ncc.run(verbose)) return -1;
	if (!ncc.save(verbose)) return -1;


	std::cout << "Done.\n";

	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

