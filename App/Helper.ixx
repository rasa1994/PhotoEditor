
export module Helper;
export import Resources;
export import <vector>;
export import Filters;
export import ImageLoader;
export import <array>;

import <wx/wxprec.h>;

#ifndef WX_PRECOMP
import <wx/wx.h>;
#endif // !WX_PRECOMP

export
{
    std::pair<std::vector<unsigned char>, std::vector<unsigned char>> ConvertFromImage(const std::vector<unsigned char>& data)
    {
		std::vector<unsigned char> RGB, a;
		RGB.resize(data.size() * 3 / 4);
		a.resize(data.size() / 4);

		for (auto it{ 0 }; it < a.size(); ++it)
		{
			RGB[it * 3] = data[it * 4];
			RGB[it * 3 + 1] = data[it * 4 + 1];
			RGB[it * 3 + 2] = data[it * 4 + 2];
			a[it] = data[it * 4 + 3];
		}

		return { RGB, a };
    }

	constexpr std::pair<size_t, size_t> FromWxPoint(const wxPoint& point)
	{
		return { point.x, point.y };
	}

	class PanelFilterInterface : public wxPanel
	{
	public:
		PanelFilterInterface(wxWindow* parent, const std::function<void(Image*, bool)>& postProcessing) : wxPanel(parent, wxID_ANY, wxPoint(PanelPos.first, PanelPos.second), wxSize(PanelDimension.first, PanelDimension.second)), postProcessing(postProcessing)
			
		{
			label = std::make_shared<wxStaticText>(this, wxID_ANY, "Filter", wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL);
			label->SetFont(wxFont(12, wxFontFamily::wxFONTFAMILY_DEFAULT, wxFontStyle::wxFONTSTYLE_NORMAL, wxFontWeight::wxFONTWEIGHT_BOLD));
			yOffset += label->GetSize().y;
		}
		void SetImage(Image* image)
		{
			m_image = image;
		}

		virtual void ProcessClick(const wxPoint& point) {};
		
		void Show()
		{
			show = true;
			__super::Show();
		}

		void Hide()
		{
			show = false;
			__super::Hide();
		}

	protected:
		Image* m_image{ nullptr };
		std::function<void(Image*, bool)> postProcessing;
		std::shared_ptr < wxStaticText> label;
		int yOffset{ 0 };
		bool show{ true };
	};

	class RGBFilterPanel : public PanelFilterInterface
	{
	public:
		RGBFilterPanel(wxWindow* parent, const std::function<void(Image*, bool)>& postProcessing) : PanelFilterInterface(parent, postProcessing)
		{
			label->SetLabelText("RGBA Filter text");
			constexpr std::array text = { "R", "G", "B", "A" };
			for (size_t pixel{ 0 }; pixel < 4; ++pixel)
			{
				constexpr auto defaultSizeY = 50;
				yOffset += defaultSizeY;
				auto slider = std::make_shared< wxSlider>(this, IDRGBFilter + pixel, 0, -255, 255, wxPoint(50,  yOffset), wxDefaultSize, wxSL_HORIZONTAL);
				auto label = std::make_shared< wxStaticText>(this, wxID_ANY, text[pixel], wxPoint(10, yOffset), wxDefaultSize, wxALIGN_CENTER_HORIZONTAL);
				const auto font = wxFont(12, wxFontFamily::wxFONTFAMILY_DEFAULT, wxFontStyle::wxFONTSTYLE_NORMAL, wxFontWeight::wxFONTWEIGHT_BOLD);
				label->SetFont(font);
				auto valueLabel = std::make_shared<wxStaticText>(this, wxID_ANY, "0", wxPoint(160, yOffset), wxDefaultSize, wxALIGN_CENTER_HORIZONTAL);
				valueLabel->SetFont(font);
				m_labelsVALUES.push_back(valueLabel);
				m_labelsRGBA.push_back(label);
				slider->Bind(wxEVT_SCROLL_CHANGED, &RGBFilterPanel::SliderChanged, this);
				m_sliders.push_back({ slider, 0 });
			}
		};
		
	private:
		std::vector<std::shared_ptr<wxStaticText>> m_labelsRGBA;
		std::vector<std::shared_ptr<wxStaticText>> m_labelsVALUES;
		std::vector<std::pair<std::shared_ptr<wxSlider>, int>> m_sliders;
		void SliderChanged(wxCommandEvent& command);
	};

	class GrayScaleFilter : public PanelFilterInterface
	{
	public:
		GrayScaleFilter(wxWindow* parent, const std::function<void(Image*, bool )>& postProcessing) : PanelFilterInterface(parent, postProcessing)
		{
			label->SetLabelText("Gray scale filter");
			m_button = std::make_shared<wxButton>(this, IDGrayScaleFilter, "Apply", wxPoint(50, yOffset), wxDefaultSize);
			Bind(wxEVT_BUTTON, &GrayScaleFilter::OnClick, this);
		};
	private:
		void OnClick(wxCommandEvent& command);
		std::shared_ptr<wxButton> m_button;
	};

	class CudaGrayScaleFilter : public PanelFilterInterface
	{
	public:
		CudaGrayScaleFilter(wxWindow* parent, const std::function<void(Image*, bool)>& postProcessing) : PanelFilterInterface(parent, postProcessing)
		{
			label->SetLabelText("Cuda grayscale filter");
			m_button = std::make_shared<wxButton>(this, IDCudaGrayscale, "Apply", wxPoint(50, yOffset), wxDefaultSize);
			Bind(wxEVT_BUTTON, &CudaGrayScaleFilter::OnClick, this);
		};
	private:
		void OnClick(wxCommandEvent& command);
		std::shared_ptr<wxButton> m_button;
	};


	class BlackWhiteFilter : public PanelFilterInterface
	{
	public:
		BlackWhiteFilter(wxWindow* parent, const std::function<void(Image*, bool )>& postProcessing) : PanelFilterInterface(parent, postProcessing)
		{
			label->SetLabelText("Black and white filter");
			m_button = std::make_shared<wxButton>(this, IDGrayScaleFilter, "Apply", wxPoint(50, yOffset), wxDefaultSize);
			Bind(wxEVT_BUTTON, &BlackWhiteFilter::OnClick, this);
		};
	private:
		void OnClick(wxCommandEvent& command);
		std::shared_ptr<wxButton> m_button;
	};

	class PixelazeFilter : public PanelFilterInterface
	{
	public:
		PixelazeFilter(wxWindow* parent, const std::function<void(Image*, bool )>& postProcessing) : PanelFilterInterface(parent, postProcessing)
		{
			label->SetLabelText("Pixelaze filter");
			m_slider = std::make_shared<wxSlider>(this, IDPixelaze, 0, 1, 16, wxPoint(50, yOffset), wxDefaultSize, wxSL_HORIZONTAL);
			yOffset += 50;
			m_button = std::make_shared<wxButton>(this, IDPixelaze, "Apply", wxPoint(50, yOffset), wxDefaultSize);
			Bind(wxEVT_BUTTON, &PixelazeFilter::OnClick, this);
		};
	private:
		void OnClick(wxCommandEvent& command);
		std::shared_ptr<wxButton> m_button;
		std::shared_ptr<wxSlider> m_slider;
	};

	class LiqifyFilter : public PanelFilterInterface
	{
	public:
		LiqifyFilter(wxWindow* parent, const std::function<void(Image*, bool )>& postProcessing) : PanelFilterInterface(parent, postProcessing)
		{
			label->SetLabelText("Liqify filter");
			m_slider = std::make_shared<wxSlider>(this, IDLiqify, 0, 1, 48, wxPoint(50, yOffset), wxDefaultSize, wxSL_HORIZONTAL);
		};

		void ProcessClick(const wxPoint& point) override;

	private:
		std::shared_ptr<wxSlider> m_slider;
	};
}



void RGBFilterPanel::SliderChanged(wxCommandEvent& command)
{
	if (!m_image) return;

	const auto id = command.GetId() - IDRGBFilter;
	const auto value = m_sliders[id].first->GetValue();
	m_labelsVALUES[id]->SetLabelText(std::to_string(value));
	m_sliders[id].second = value;
	Image copy = *m_image;
	for (size_t data{ 0u }; data < copy.m_imageData.size(); data += 4)
	{
		for (size_t pixel{ 0u }; pixel < 4; ++pixel)
		{
			const int valueFormatted = copy.m_imageData[data + pixel] + m_sliders[pixel].second;
			copy.m_imageData[data + pixel] = std::clamp(valueFormatted, 0, 255);
		}
	}

	postProcessing(&copy, false);
}

void GrayScaleFilter::OnClick(wxCommandEvent& command)
{
	if (!m_image) return;
	Image copy = *m_image;
	GrayFilter(copy);
	postProcessing(&copy, true);
}

extern "C" void ApplyGrayScaleFilterCuda(unsigned char* d_input, int width, int height);

void CudaGrayScaleFilter::OnClick(wxCommandEvent& command)
{
	if (!m_image) return;
	Image copy = *m_image;
	ApplyGrayScaleFilterCuda(copy.m_imageData.data(), copy.m_width, copy.m_height);
	postProcessing(&copy, true);
}

void BlackWhiteFilter::OnClick(wxCommandEvent& command)
{
	if (!m_image) return;
	Image copy = *m_image;
	BlackAndWhite(copy);
	postProcessing(&copy, true);
}

void PixelazeFilter::OnClick(wxCommandEvent& command)
{
	if (!m_image) return;
	Image copy = *m_image;
	const auto value = m_slider->GetValue();
	Pixelate(copy, value);
	postProcessing(&copy, true);
}

void LiqifyFilter::ProcessClick(const wxPoint& point)
{
	if (!m_image) return;
	if (!show) return;
	Image copy = *m_image;
	const auto liqifySize = m_slider->GetValue();
	Liqify(copy, FromWxPoint(point), liqifySize);
	postProcessing(&copy, true);
}
