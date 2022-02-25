#pragma once

#include "complex.cuh"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <string>
#include <system_error>


/***********************************************************
read nrows and ncols parameters from file

\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the file to read from

\return bool -- true if file successfully read, false if not
***********************************************************/
template<typename T>
bool read_params(int& nrows, int& ncols, const std::string& fname)
{
	nrows = 0;
	ncols = 0;

	std::ifstream infile;

	if (fname.substr(fname.size() - 4) == ".txt")
	{
		infile.open(fname);

		if (!infile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}

		std::string input;
		while (infile >> input)
		{
			if (input == "num_roots")
			{
				infile >> nrows;
			}
			if (input == "num_phi")
			{
				infile >> ncols;
				ncols++;
			}
		}
		infile.close();
		if (nrows < 1 || ncols < 2)
		{
			std::cerr << "Error. File " << fname << " does not contain valid values for num_rows and num_cols.\n";
			return false;
		}
	}
	else if (fname.substr(fname.size() - 4) == ".bin")
	{
		infile.open(fname, std::ios_base::binary);

		if (!infile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}

		infile.read((char*)(&nrows), sizeof(int));
		infile.read((char*)(&ncols), sizeof(int));
		infile.close();
		if (nrows < 1 || ncols < 2)
		{
			std::cerr << "Error. File " << fname << " does not contain valid values for num_rows and num_cols.\n";
			return false;
		}
	}
	else
	{
		std::cerr << "Error. File " << fname << " is not a .bin or .txt file.\n";
		return false;
	}

	std::cout << "num_rows: " << nrows << "\n";
	std::cout << "num_cols: " << ncols << "\n";
	return true;
}

/***********************************************************
read array of complex values from disk into array

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the .bin file to read from

\return bool -- true if file successfully read, false if not
***********************************************************/
template <typename T>
bool read_complex_array(Complex<T>* vals, int nrows, int ncols, const std::string& fname)
{
	std::ifstream infile;

	if (fname.substr(fname.size() - 4) == ".bin")
	{
		std::error_code err;
		std::uintmax_t fsize = std::filesystem::file_size(fname, err);

		if (err)
		{
			std::cerr << "Error determining size of input file " << fname << "\n";
			return false;
		}

		infile.open(fname, std::ios_base::binary);

		if (!infile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}

		int temp;
		infile.read((char*)(&temp), sizeof(int));
		infile.read((char*)(&temp), sizeof(int));

		if ((fsize - 2 * sizeof(int)) == nrows * ncols * sizeof(Complex<T>))
		{
			infile.read((char*)vals, nrows * ncols * sizeof(Complex<T>));
		}
		else if ((fsize - 2 * sizeof(int)) == nrows * ncols * sizeof(Complex<float>))
		{
			Complex<float>* temp_vals = new (std::nothrow) Complex<float>[nrows * ncols];
			if (!temp_vals)
			{
				std::cerr << "Error. Memory allocation for *temp_vals failed.\n";
				return false;
			}
			infile.read((char*)temp_vals, nrows * ncols * sizeof(Complex<float>));
			for (int i = 0; i < nrows * ncols; i++)
			{
				vals[i] = Complex<T>(temp_vals[i].re, temp_vals[i].im);
			}
			delete temp_vals;
			temp_vals = nullptr;
		}
		else if ((fsize - 2 * sizeof(int)) == nrows * ncols * sizeof(Complex<double>))
		{
			Complex<double>* temp_vals = new (std::nothrow) Complex<double>[nrows * ncols];
			if (!temp_vals)
			{
				std::cerr << "Error. Memory allocation for *temp_vals failed.\n";
				return false;
			}
			infile.read((char*)temp_vals, nrows * ncols * sizeof(Complex<double>));
			for (int i = 0; i < nrows * ncols; i++)
			{
				vals[i] = Complex<T>(temp_vals[i].re, temp_vals[i].im);
			}
			delete temp_vals;
			temp_vals = nullptr;
		}
		else
		{
			std::cerr << "Error. Binary file does not contain valid single or double precision numbers.\n";
			return false;
		}

		infile.close();
	}
	else
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	return true;
}

/***********************************************************
read array of real values from disk into array

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the .txt file to read from

\return bool -- true if file successfully read, false if not
***********************************************************/
template <typename T>
bool read_re_array(T* vals, int nrows, int ncols, const std::string& fname)
{
	std::ifstream infile;

	if (fname.substr(fname.size() - 4) == ".txt")
	{
		infile.open(fname);

		if (!infile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		for (int i = 0; i < nrows * ncols; i++)
		{
			infile >> vals[i];
		}
		infile.close();
	}
	else
	{
		std::cerr << "Error. File " << fname << " is not a .txt file.\n";
		return false;
	}

	return true;
}

/**************************************************************
write array of values to disk

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the file to write to

\return bool -- true if file successfully written, false if not
**************************************************************/
template <typename T>
bool write_array(T* vals, int nrows, int ncols, const std::string& fname)
{
	std::ofstream outfile;

	if (fname.substr(fname.size() - 4) == ".txt")
	{
		outfile.precision(9);
		outfile.open(fname);

		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		for (int i = 0; i < nrows; i++)
		{
			for (int j = 0; j < ncols; j++)
			{
				outfile << vals[i * ncols + j] << " ";
			}
			outfile << "\n";
		}
	}
	else if (fname.substr(fname.size() - 4) == ".bin")
	{
		outfile.open(fname, std::ios_base::binary);

		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}

		outfile.write((char*)(&nrows), sizeof(int));
		outfile.write((char*)(&ncols), sizeof(int));
		outfile.write((char*)vals, nrows * ncols * sizeof(T));
		outfile.close();
	}
	else
	{
		std::cerr << "Error. File " << fname << " is not a .bin or .txt file.\n";
		return false;
	}

	return true;
}

