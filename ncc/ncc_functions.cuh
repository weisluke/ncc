﻿#pragma once

#include "complex.cuh"
#include "util.cuh"

#include <cstdint> //for std::uintmax_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <string>
#include <system_error> //for std::error_code


/******************************************************************************
return the sign of a number

\param val -- number to find the sign of

\return -1, 0, or 1
******************************************************************************/
template <typename T>
__device__ T sgn(T val)
{
	if (val < -0) return -1;
	if (val > 0) return 1;
	return 0;
}

/******************************************************************************
determine whether a point lies within a rectangular region centered on the
origin. our particular consideration is a rectangular region such that
-hl < y < hl, and -hl < x
this is so that we can keep track of how many caustics are crossed in a line
extending from a particular point (e.g., a pixel center) to positive infinity
in the x-direction

\param p0 -- point
\param hl -- half length of the square region

\return true if point in region, false if not
******************************************************************************/
template <typename T>
__device__ bool point_in_region(Complex<T> p0, T hl)
{
	if (p0.re > -hl && fabs(p0.im) < hl)
	{
		return true;
	}
	return false;
}

/******************************************************************************
return the x position of a line connecting two points, at a particular y-value
handles vertical lines by returning the x position of the first point

\param p1 -- first point
\param p2 -- second point
\param y -- y position

\return x position of connecting line at y position
******************************************************************************/
template <typename T>
__device__ T get_line_x_position(Complex<T> p1, Complex<T> p2, T y)
{
	if (p1.re == p2.re)
	{
		return p1.re;
	}
	T slope = (p2.im - p1.im) / (p2.re - p1.re);
	return (y - p1.im) / slope + p1.re;
}

/******************************************************************************
return the y position of a line connecting two points, at a particular x-value
handles vertical lines by returning the y position of the first point

\param p1 -- first point
\param p2 -- second point
\param x -- x position

\return y position of connecting line at x position
******************************************************************************/
template <typename T>
__device__ T get_line_y_position(Complex<T> p1, Complex<T> p2, T x)
{
	if (p1.re == p2.re)
	{
		return p1.im;
	}
	T slope = (p2.im - p1.im) / (p2.re - p1.re);
	return slope * (x - p1.re) + p1.im;
}

/******************************************************************************
given two points, corrects the first point so that it lies within the desired
region of the point_in_region function
this function implicitly assumes that the line connecting the two points
intersects, or lies within, the desired region

\param p1 -- first point
\param p2 -- second point
\param hl -- half length of the square region
\param npixels -- number of pixels per side for the square region

\return position of corrected point
******************************************************************************/
template <typename T>
__device__ Complex<T> corrected_point(Complex<T> p1, Complex<T> p2, T hl, int npixels)
{
	T x = p1.re;
	T y = p1.im;
	if (x <= -hl)
	{
		/******************************************************************************
		if the x position is outside of our region, calculate where the point would be
		with an x position 1/100 of a pixel inside the desired region
		1/100 used due to later code only considering line crossings at the half pixel
		mark
		******************************************************************************/
		x = -hl + 0.01 * 2 * hl / npixels;
		y = get_line_y_position(p1, p2, x);
	}
	if (fabs(y) >= hl)
	{
		/******************************************************************************
		if the y position is outside of our region, calculate where the point would be
		with a y position 1/100 of a pixel inside the desired region
		1/100 used due to later code only considering line crossings at the half pixel
		mark
		******************************************************************************/
		y = sgn(y) * (hl - 0.01 * 2 * hl / npixels);
		x = get_line_x_position(p1, p2, y);
	}

	return Complex<T>(x, y);
}

/******************************************************************************
calculate the pixel mapping of a point


\param p0 -- point
\param hl -- half length of the square region
\param npixels -- number of pixels per side for the square region

\return pixel position of point
******************************************************************************/
template <typename T, typename U>
__device__ Complex<T> point_to_pixel(Complex<U> p0, U hl, int npixels)
{
	Complex<T> result((p0 + hl * Complex<T>(1, 1)) * npixels / (2 * hl));
	return result;
}

/******************************************************************************
calculate the number of caustic crossings

\param caustics -- array of caustic positions
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param hl -- half length of the square region
\param num -- array of number of caustic crossings
\param npixels -- number of pixels per side for the square region
******************************************************************************/
template <typename T>
__global__ void find_num_caustic_crossings_kernel(Complex<T>* caustics, int nrows, int ncols, double hl, int* num, int npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < nrows; i += x_stride)
	{
		/******************************************************************************
		only able to calculate a caustic crossing if a caustic point has a succeeding
		point to form a line segment with (i.e., we are not at the end of the 2*pi
		phase chain that was traced out), hence ncols - 1
		******************************************************************************/
		for (int j = y_index; j < ncols - 1; j += y_stride)
		{
			/******************************************************************************
			initial and final point of the line segment
			we will be calculating what pixels this line segment crosses
			******************************************************************************/
			Complex<T> pt0 = caustics[i * ncols + j];
			Complex<T> pt1 = caustics[i * ncols + j + 1];

			/******************************************************************************
			if one of the endpoints lies within the region, correct the points so they both
			lie within the region
			possibly redundant if both lie within the region
			******************************************************************************/
			if (point_in_region(pt0, hl) || point_in_region(pt1, hl))
			{
				pt0 = corrected_point(pt0, pt1, hl, npixels);
				pt1 = corrected_point(pt1, pt0, hl, npixels);
			}
			/******************************************************************************
			else if both points are outside the region, but intersect the region boundaries
			(i.e., really long caustic segments), correct the points so they both lie
			within the region
			******************************************************************************/
			else if (get_line_x_position(pt0, pt1, hl) >= -hl
				|| get_line_x_position(pt0, pt1, -hl) >= -hl
				|| fabs(get_line_y_position(pt0, pt1, -hl)) <= hl)
			{
				pt0 = corrected_point(pt0, pt1, hl, npixels);
				pt1 = corrected_point(pt1, pt0, hl, npixels);
			}
			/******************************************************************************
			else continue on to the next caustic segment
			******************************************************************************/
			else
			{
				continue;
			}

			/******************************************************************************
			make complex pixel start and end positions
			******************************************************************************/
			Complex<T> pixpt0 = point_to_pixel<T, T>(pt0, hl, npixels);
			Complex<T> pixpt1 = point_to_pixel<T, T>(pt1, hl, npixels);

			/******************************************************************************
			find starting and ending pixel positions including fractional parts
			******************************************************************************/
			T ystart = pixpt0.im;
			T yend = pixpt1.im;

			/******************************************************************************
			if caustic subsegment starts or ends exactly in the middle of a pixel, shift
			starting position down by 0.01 pixels
			done to avoid double-counting crossings at that particular point, since it is
			the end of one segment and the start of another
			******************************************************************************/
			if (ystart == static_cast<int>(ystart) + 0.5)
			{
				ystart -= 0.01;
			}
			if (yend == static_cast<int>(yend) + 0.5)
			{
				yend -= 0.01;
			}

			/******************************************************************************
			offset start by half a pixel in the direction of the ordered points
			this will ensure that after casting things to integer pixel values, we are only
			using pixels for which the segment crosses the center
			******************************************************************************/
			ystart += sgn(pt1.im - pt0.im) / 2;

			/******************************************************************************
			take into account starting points between 0 and 0.5 that get pushed outside the
			desired region, i.e. below 0, if y_end < y_start (e.g., a line from y=0.4 to
			y=0.1 would get changed after the line above to look like from y=-0.1 to y=0.1,
			and would have y_pixel_start = y_pixel_end = 0, making later code think it
			crosses the center of the pixel when in reality it doesn't)
			only needs accounted for in this case due to integer truncation at 0 being the
			same for +/- small values. it is NOT a problem at other borders of the region
			******************************************************************************/
			if (ystart < 0)
			{
				continue;
			}

			/******************************************************************************
			find actual starting y value at pixel center
			******************************************************************************/
			ystart = static_cast<int>(ystart) + 0.5;

			Complex<int> ypix;

			/******************************************************************************
			our caustics are traced in a clockwise manner
			if p1.y > p0.y, then as a line from a pixel center goes to infinity towards the
			right, we are entering a caustic region if we cross this segment
			conversely, this means a line from infinity to a pixel center leaves a caustic
			region at this segment, and so we subtract one from all pixels to the left of
			the line segment
			******************************************************************************/
			if (pt0.im < pt1.im)
			{
				/******************************************************************************
				for all y-values from the start of the line segment (at a half-pixel) to the
				end (could be anywhere) in increments of one pixel
				******************************************************************************/
				for (T k = ystart; k < yend; k += 1)
				{
					/******************************************************************************
					find the x position of the caustic segment at the current y value
					find the integer pixel value for this x-position (subtracting 0.5 to account
					for the fact that we need to be to the right of the pixel center) and the
					integer y pixel value
					if the x-pixel value is greater than the number of pixels, we perform the
					subtraction on the whole pixel row
					******************************************************************************/
					T x1 = get_line_x_position(pixpt0, pixpt1, k);
					ypix = Complex<int>(x1 - 0.5, k);

					ypix.im = npixels - 1 - ypix.im;
					if (ypix.re > npixels - 1)
					{
						ypix.re = npixels - 1;
					}
					for (int l = 0; l <= ypix.re; l++)
					{
						if (ypix.im * npixels + l < 0 || ypix.im * npixels + l > npixels * npixels - 1)
						{
							printf("Error. Caustic crossing takes place outside the desired region.\n");
							continue;
						}
						atomicSub(&num[ypix.im * npixels + l], 1);
					}
				}
			}
			/******************************************************************************
			our caustics are traced in a clockwise manner
			if p1.y < p0.y, then as a line from a pixel center goes to infinity towards the
			right, we are leaving a caustic region if we cross this segment
			conversely, this means a line from infinity to a pixel center enters a caustic
			region at this segment, and so we add one to all pixels to the left of the line
			segment
			******************************************************************************/
			else if (pt0.im > pt1.im)
			{
				for (T k = ystart; k > yend; k -= 1)
				{
					/******************************************************************************
					find the x position of the caustic segment at the current y value
					find the integer pixel value for this x-position (subtracting 0.5 to account
					for the fact that we need to be to the right of the pixel center) and the
					integer y pixel value
					if the x-pixel value is greater than the number of pixels, we perform the
					addition on the whole pixel row
					******************************************************************************/
					T x1 = get_line_x_position(pixpt0, pixpt1, k);
					ypix = Complex<int>(x1 - 0.5, k);

					ypix.im = npixels - 1 - ypix.im;
					if (ypix.re > npixels - 1)
					{
						ypix.re = npixels - 1;
					}
					for (int l = 0; l <= ypix.re; l++)
					{
						if (ypix.im * npixels + l < 0 || ypix.im * npixels + l > npixels * npixels - 1)
						{
							printf("Error. Caustic crossing takes place outside the desired region.\n");
							continue;
						}
						atomicAdd(&num[ypix.im * npixels + l], 1);
					}
				}
			}
		}
	}
}

/******************************************************************************
recenter the caustics relative to the given center

\param caustics -- array of caustic positions
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param center_y -- center of the source plane receiving region
******************************************************************************/
template <typename T>
__global__ void recenter_caustics_kernel(Complex<T>* caustics, int nrows, int ncols, Complex<T> center_y)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < ncols; i += x_stride)
	{
		for (int j = y_index; j < nrows; j += y_stride)
		{
			caustics[j * ncols + i] -= center_y;
		}
	}
}

/******************************************************************************
initialize array of values to 0

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
******************************************************************************/
template <typename T>
__global__ void initialize_array_kernel(int* vals, int nrows, int ncols)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < ncols; i += x_stride)
	{
		for (int j = y_index; j < nrows; j += y_stride)
		{
			vals[j * ncols + i] = 0;
		}
	}
}

/******************************************************************************
reduce the pixel array

\param num -- array of number of caustic crossings
\param npixels -- number of pixels per side for the square region
******************************************************************************/
template <typename T>
__global__ void reduce_pix_array_kernel(int* num, int npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels; i += x_stride)
	{
		for (int j = y_index; j < npixels; j += y_stride)
		{
			int n1 = num[2 * j * 2 * npixels + 2 * i];
			int n2 = num[2 * j * 2 * npixels + 2 * i + 1];
			int n3 = num[(2 * j + 1) * 2 * npixels + 2 * i];
			int n4 = num[(2 * j + 1) * 2 * npixels + 2 * i + 1];

			num[2 * j * 2 * npixels + 2 * i] = max(max(n1, n2), max(n3, n4));
		}
	}
}

/******************************************************************************
shift the provided pixel column from 2*i to i

\param num -- array of number of caustic crossings
\param npixels -- number of pixels per side for the square region
\param column -- the column to shift to
******************************************************************************/
template <typename T>
__global__ void shift_pix_column_kernel(int* num, int npixels, int column)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < npixels; i += x_stride)
	{
		num[2 * i * 2 * npixels + column] = num[2 * i * 2 * npixels + 2 * column];
	}
}

/******************************************************************************
shift the provided pixel row from 2*i to i

\param num -- array of number of caustic crossings
\param npixels -- number of pixels per side for the square region
\param row -- the row to shift to
******************************************************************************/
template <typename T>
__global__ void shift_pix_row_kernel(int* num, int npixels, int row)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < npixels; i += x_stride)
	{
		num[row * npixels + i] = num[2 * row * 2 * npixels + i];
	}
}

/******************************************************************************
calculate the minimum and maximum number of crossings in the pixel array

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param minnum -- pointer to minimum number of crossings
\param maxnum -- pointer to maximum number of crossings
******************************************************************************/
template <typename T>
__global__ void histogram_min_max_kernel(int* pixels, int npixels, int* minnum, int* maxnum)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels; i += x_stride)
	{
		for (int j = y_index; j < npixels; j += y_stride)
		{
			atomicMin(minnum, pixels[j * npixels + i]);
			atomicMax(maxnum, pixels[j * npixels + i]);
		}
	}
}

/******************************************************************************
calculate the histogram of crossings for the pixel array

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param minnum -- minimum number of crossings
\param histogram -- pointer to histogram
******************************************************************************/
template <typename T>
__global__ void histogram_kernel(int* pixels, int npixels, int minnum, int* histogram)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels; i += x_stride)
	{
		for (int j = y_index; j < npixels; j += y_stride)
		{
			atomicAdd(&histogram[pixels[j * npixels + i] - minnum], 1);
		}
	}
}

/******************************************************************************
read array of complex values from disk into array

\param vals -- pointer to array of values
\param nrows -- pointer to number of rows in array
\param ncols -- pointer to number of columns in array
\param fname -- location of the .bin file to read from

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template <typename T>
bool read_complex_array(Complex<T>*& vals, int& nrows, int& ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".bin")
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	std::ifstream infile;

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

	infile.read((char*)(&nrows), sizeof(int));
	infile.read((char*)(&ncols), sizeof(int));
	if (nrows < 1 || ncols < 1)
	{
		std::cerr << "Error. File " << fname << " does not contain valid values for num_rows and num_cols.\n";
		return false;
	}

	/******************************************************************************
	allocate memory for array if needed
	******************************************************************************/
	if (vals == nullptr)
	{
		cudaMallocManaged(&vals, nrows * ncols * sizeof(Complex<T>));
		if (cuda_error("cudaMallocManaged(*vals)", false, __FILE__, __LINE__)) return false;
	}


	/******************************************************************************
	objects in the file are nrows + ncols + complex values
	******************************************************************************/
	if (fsize == 2 * sizeof(int) + nrows * ncols * sizeof(Complex<T>))
	{
		infile.read((char*)vals, nrows * ncols * sizeof(Complex<T>));
	}
	else if (fsize == 2 * sizeof(int) + nrows * ncols * sizeof(Complex<float>))
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
			vals[i] = Complex<T>(temp_vals[i]);
		}
		delete[] temp_vals;
		temp_vals = nullptr;
	}
	else if (fsize == 2 * sizeof(int) + nrows * ncols * sizeof(Complex<double>))
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
			vals[i] = Complex<T>(temp_vals[i]);
		}
		delete[] temp_vals;
		temp_vals = nullptr;
	}
	else
	{
		std::cerr << "Error. Binary file does not contain valid single or double precision numbers.\n";
		return false;
	}

	infile.close();

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

