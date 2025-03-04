
export module Resources;
export import <utility>;
export
{
	enum
	{
		ID_OpenFile = 2,
		Filter1,
		Filter2,
		Filter3,
		Filter4,
		Filter5,
		Filter6,
		Save,
	};

	enum ID : unsigned long long
	{
		IDRGBFilter = 10000,
		IDGrayScaleFilter = 10010,
		IDBlackWhiteFilter = 10020,
		IDPixelaze = 10030,
		IDLiqify = 10040,
		IDCudaGrayscale = 10050,
	};

	using Pair = std::pair<size_t, size_t>;

	constexpr Pair AppDimension = { 800, 800 };
	
	constexpr Pair ImageDimension = { 600, 600 };
	
	constexpr Pair PanelDimension = { 200, AppDimension.second};
	
	constexpr Pair PanelPos = { ImageDimension.first, 0 };
}