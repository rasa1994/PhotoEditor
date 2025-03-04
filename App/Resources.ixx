
export module Resources;
export import <utility>;
export
{
	enum
	{
		ID_OpenFile = 2,
		Filter1 = 3,
		Filter2 = 4,
		Filter3 = 5,
		Filter4 = 6,
		Filter5,
		Save,
	};

	enum ID : unsigned long long
	{
		IDRGBFilter = 10000,
		IDGrayScaleFilter = 10010,
		IDBlackWhiteFilter = 10020,
		IDPixelaze = 10030,
		IDLiqify = 10040,
	};

	using Pair = std::pair<size_t, size_t>;

	constexpr Pair AppDimension = { 800, 800 };
	
	constexpr Pair ImageDimension = { 600, 600 };
	
	constexpr Pair PanelDimension = { 200, AppDimension.second};
	
	constexpr Pair PanelPos = { ImageDimension.first, 0 };
}