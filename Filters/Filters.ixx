export module Filters;
export import ImageLoader;

import <array>;


export
{
	enum class Operation {Bigger, Smaller};

	void GrayFilter(Image& image)
	{
		constexpr float pR = 0.11f;
		constexpr float pG = 0.66f;
		constexpr float pB = 0.23f;
		for (size_t row{}; row < image.m_height; ++row)
		{
			for (size_t column{}; column < image.m_width; ++column)
			{
				RGBA pixel = image.GetPixel(row, column);
				unsigned char val = pixel.R * pR + pixel.G * pG + pixel.B * pB;
				pixel.R = val;
				pixel.G = val;
				pixel.B = val;
				image.SetPixel(row, column, pixel);
			}
		}
	}

	void RecolorOfGrayAndWhite(Image& image, float percentageR, float percentageG, float percentageB, unsigned bitOffset)
	{
		auto& data = image.m_imageData;
		for (unsigned pix = 0; pix < image.m_imageData.size() && pix + 4 * bitOffset < image.m_imageData.size(); pix += (4 * bitOffset))
		{
			auto val = data[pix];
			data[pix] = val * percentageR;
			data[pix + 1] = val * percentageG;
			data[pix + 2] = val * percentageB;
		}
	}

	void BlackAndWhite(Image& image)
	{
		for (size_t row{}; row < image.m_height; ++row)
		{
			for (size_t column{}; column < image.m_width; ++column)
			{
				RGBA pixel = image.GetPixel(row, column);
				unsigned int val = pixel.R + pixel.G + pixel.G;

				val /= 3;

				unsigned char color = val < 128 ? 255 : 0;

				pixel.R = color;
				pixel.G = color;
				pixel.B = color;
				pixel.A = color;

				image.SetPixel(row, column, pixel);
			}
		};
	}

	void Pixelate(Image& image, size_t sampleSize)
	{
		for (size_t row{}; row < image.m_height; row += sampleSize)
		{
			for (size_t column{}; column < image.m_width; column += sampleSize)
			{
				std::array<unsigned, 4> sum{};
				auto limitSampleSize = std::min(sampleSize, image.m_height - row);
				auto limitSampleSizeColumn = std::min(sampleSize, image.m_width - column);

				for (size_t r{}; r < limitSampleSize; ++r)
				{
					for (size_t c{}; c < limitSampleSizeColumn; ++c)
					{
						RGBA pixel = image.GetPixel(row + r, column + c);
						sum[0] += pixel.R;
						sum[1] += pixel.G;
						sum[2] += pixel.B;
						sum[3] += pixel.A;
					}
				}

				RGBA pixel;
				pixel.R = sum[0] / (limitSampleSize * limitSampleSizeColumn);
				pixel.G = sum[1] / (limitSampleSize * limitSampleSizeColumn);
				pixel.B = sum[2] / (limitSampleSize * limitSampleSizeColumn);
				pixel.A = sum[3] / (limitSampleSize * limitSampleSizeColumn);

				for (size_t r{}; r < limitSampleSize; ++r)
				{
					for (size_t c{}; c < limitSampleSizeColumn; ++c)
					{
						image.SetPixel(row + r, column + c, pixel);
					}
				}
			}
		}
	}

	void Liqify(Image& image, const std::pair<size_t, size_t>& pos, size_t pixelSize)
	{
		const auto startPosX = pixelSize < pos.first ? pos.first - pixelSize : 0;
		const auto startPosY = pixelSize < pos.second ? pos.second - pixelSize : 0;
		const auto endPosX = std::min(pos.first + pixelSize, static_cast<unsigned long long>(image.m_width));
		const auto endPosY = std::min(pos.second + pixelSize, static_cast<unsigned long long>(image.m_height));

		std::array sum{ 0ull, 0ull, 0ull, 0ull };
		size_t observed{};

		for (auto row{ startPosY }; row < endPosY; ++row)
		{
			for (auto column{ startPosX }; column < endPosX; ++column)
			{
				if (const double distance = std::sqrt(std::pow(static_cast<double>(pos.first) - column, 2) + std::pow(static_cast<double>(pos.second) - row, 2)); distance <= pixelSize)
				{
					const auto pixel = image.GetPixel(row, column);
					sum[0] += pixel.R;
					sum[1] += pixel.G;
					sum[2] += pixel.B;
					sum[3] += pixel.A;
					observed++;
				}
			}
		}
		if (observed == 0) return;
		std::ranges::for_each(sum, [observed](auto& val) { val /= observed; });

		for (auto row{ startPosY }; row < endPosY; ++row)
		{
			for (auto column{ startPosX }; column < endPosX; ++column)
			{
				if (const double distance = std::sqrt(std::pow(static_cast<double>(pos.first) - column, 2) + std::pow(static_cast<double>(pos.second) - row, 2)); distance <= pixelSize)
				{
					RGBA pixel{};
					pixel.R = sum[0];
					pixel.G = sum[1];
					pixel.B = sum[2];
					pixel.A = sum[3];
					image.SetPixel(row, column, pixel);
				}
			}
		}
	}
}