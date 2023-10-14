#pragma once

#include "complex.cuh"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <string>
#include <system_error>


/******************************************************************************
read nrows and ncols parameters from file

\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the file to read from

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template<typename T>
bool read_params(int& nrows, int& ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".bin")
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	std::ifstream infile;

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

	return true;
}

/******************************************************************************
read array of complex values from disk into array

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the .bin file to read from

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template <typename T>
bool read_complex_array(Complex<T>* vals, int nrows, int ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	std::ifstream infile;

	if (fpath.extension() == ".bin")
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
			delete[] temp_vals;
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
			delete[] temp_vals;
			temp_vals = nullptr;
		}
		else
		{
			std::cerr << "Error. Binary file does not contain valid single or double precision numbers.\n";
			infile.close();
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

/******************************************************************************
write array of values to disk

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the file to write to

\return bool -- true if file is successfully written, false if not
******************************************************************************/
template <typename T>
bool write_array(T* vals, int nrows, int ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".bin")
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	std::ofstream outfile;

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

	return true;
}

/******************************************************************************
write histogram

\param histogram -- pointer to histogram
\param n -- length of histogram
\param minnum -- minimum number of crossings
\param fname -- location of the file to write to

\return bool -- true if file is successfully written, false if not
******************************************************************************/
template <typename T>
bool write_histogram(int* histogram, int n, int minnum, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".txt")
	{
		std::cerr << "Error. File " << fname << " is not a .txt file.\n";
		return false;
	}

	std::ofstream outfile;

	outfile.precision(9);
	outfile.open(fname);

	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return false;
	}
	for (int i = 0; i < n; i++)
	{
		if (histogram[i] != 0)
		{
			outfile << i + minnum << " " << histogram[i] << "\n";
		}
	}
	outfile.close();

	return true;
}

