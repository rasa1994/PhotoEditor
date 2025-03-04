#pragma once

import Helper;

class Frame : public wxFrame
{
public:
	Frame();
private:
	wxImage m_mainImage;
	wxBitmap m_mainBitmap;
	wxStaticBitmap* m_staticBitmap{ nullptr };
	wxPanel* m_imagePanel;
	Image m_mainImageData;
	Image m_mainImageDataOriginal;
	std::vector<PanelFilterInterface*> m_panels;


	static void OnHello(const wxCommandEvent& event);
	static void OnAbout(const wxCommandEvent& event);
	void OnExit(const wxCommandEvent& event);
	void SelectFilter(const wxCommandEvent& event);
	void OnClickHandler(const wxMouseEvent& event);
	void OpenFile(wxCommandEvent& event);
	void SaveFile(wxCommandEvent& event);
	void SetImage(const Image* image, bool changeOriginal = false);
};

inline Frame::Frame() : wxFrame(nullptr, wxID_ANY, "Photo Editor", wxPoint(0, 0), wxSize(AppDimension.first,AppDimension.second))
{
	wxInitAllImageHandlers();
	wxMenu* menuFile = new wxMenu;
	menuFile->AppendSeparator();
	menuFile->Append(wxID_OPEN);
	menuFile->Append(wxID_ABOUT);
	menuFile->Append(wxID_EXIT);

	wxMenu* menuFilters = new wxMenu;
	menuFilters->Append(Filter1, "&Change rgba filter");
	menuFilters->Append(Filter2, "&Grayscale");
	menuFilters->Append(Filter3, "&black and white");
	menuFilters->Append(Filter4, "&Pixelate");
	menuFilters->Append(Filter5, "&Liquefy");
	menuFilters->Append(Filter6, "&Cuda grayscale");

	wxMenu* saveMenu = new wxMenu;
	saveMenu->Append(Save, "&Save");

	wxMenuBar* menuBar = new wxMenuBar;
	menuBar->Append(menuFile, "&File");
	menuBar->Append(menuFilters, "&Filters");
	menuBar->Append(saveMenu, "&Save");

	SetMenuBar(menuBar);
	CreateStatusBar();
	SetStatusText("Photo editor");

	Bind(wxEVT_MENU, &Frame::OnExit, this, wxID_EXIT);
	Bind(wxEVT_MENU, &Frame::OpenFile, this, wxID_OPEN);
	Bind(wxEVT_MENU, &Frame::SaveFile, this, Save);

	Bind(wxEVT_MENU, &Frame::SelectFilter, this, Filter1);
	Bind(wxEVT_MENU, &Frame::SelectFilter, this, Filter2);
	Bind(wxEVT_MENU, &Frame::SelectFilter, this, Filter3);
	Bind(wxEVT_MENU, &Frame::SelectFilter, this, Filter4);
	Bind(wxEVT_MENU, &Frame::SelectFilter, this, Filter5);
	Bind(wxEVT_MENU, &Frame::SelectFilter, this, Filter6);

	
	m_imagePanel = new wxPanel(this);

	m_panels.push_back(new RGBFilterPanel(this, [this](Image* image, bool replace) { SetImage(image, replace); }));
	m_panels.push_back(new GrayScaleFilter(this, [this](Image* image, bool replace) { SetImage(image, replace); }));
	m_panels.push_back(new BlackWhiteFilter(this, [this](Image* image, bool replace) { SetImage(image, replace); }));
	m_panels.push_back(new PixelazeFilter(this, [this](Image* image, bool replace) { SetImage(image, replace); }));
	m_panels.push_back(new LiqifyFilter(this, [this](Image* image, bool replace) { SetImage(image, replace); }));
	m_panels.push_back(new CudaGrayScaleFilter(this, [this](Image* image, bool replace) { SetImage(image, replace); }));

	std::ranges::for_each(m_panels, [this](const auto& panel) { panel->Hide(); });
}


inline void Frame::OnHello([[maybe_unused]] const wxCommandEvent& event)
{
	wxLogMessage("Photo editor!!!");
}

inline void Frame::OnAbout([[maybe_unused]] const wxCommandEvent& event)
{
	wxMessageBox("Simple application for running filters on selected images. Wanna be photoshop!", "About photo editor", wxOK | wxICON_INFORMATION);
}

inline void Frame::OnExit([[maybe_unused]] const wxCommandEvent& event)
{
	m_mainImage.Destroy();
	Close(true);
}

inline void Frame::SelectFilter(const wxCommandEvent& event)
{
	std::ranges::for_each(m_panels, [](const auto& panel) { panel->Hide(); });

	const auto id = event.GetId() - Filter1;
	if (id >= m_panels.size())
		return;

	m_panels[id]->Show();
}


inline void Frame::OnClickHandler(const wxMouseEvent& event)
{
	std::cout << event.GetX() << " " << event.GetY() << std::endl;
	const auto position = event.GetPosition(); 
	const std::pair<double, double> mapped = 
	{ 
		static_cast<double>(m_mainImageData.m_width) / ImageDimension.first, 
		static_cast<double>(m_mainImageData.m_height) / ImageDimension.second
	};

	const wxPoint mappedPosition = { static_cast<int>(position.x * mapped.first), static_cast<int>(position.y * mapped.second) };

	std::ranges::for_each(m_panels, [mappedPosition](const auto& panel) { panel->ProcessClick(mappedPosition); });
}


void Frame::OpenFile(wxCommandEvent& event)
{
	wxString wildcard = "Image Files (*.jpg;*.png;*.bmp)|*.jpg;*.png;*.bmp";
	wxFileDialog openFileDialog(this, "Open File", "", "", wildcard, wxFD_OPEN | wxFD_FILE_MUST_EXIST);

	// Show the file dialog
	if (openFileDialog.ShowModal() == wxID_OK)
	{
		// Get the selected file path
		wxString filePath = openFileDialog.GetPath();
#undef NO_ERROR
		ErrorType error = ErrorType::NO_ERROR;
		m_mainImageDataOriginal = m_mainImageData = LoadImageFile(filePath.utf8_string(), error);
		
		for (auto& filter : m_panels)
			filter->SetImage(&m_mainImageData);
		
		if (error != ErrorType::NO_ERROR)
		{
			std::cout << "Error loading image " << std::endl;
			assert(false);
			return;
		}
#define NO_ERROR 0l;
		SetImage(&m_mainImageData);
	}
}

#include <filesystem>

inline void Frame::SaveFile([[maybe_unused]]wxCommandEvent& event)
{
	wxString wildcard = "Image Files (*.jpg;*.png;*.bmp)|*.jpg;*.png;*.bmp";
	wxFileDialog openFileDialog(this, "Save File", "", "", wildcard, wxFD_SAVE);

	if (openFileDialog.ShowModal() == wxID_OK)
	{
#undef NO_ERROR
		ErrorType error = ErrorType::NO_ERROR;
		wxString filePath = openFileDialog.GetPath();
		WriteImage(m_mainImageData, filePath.utf8_string(), error);
		if (error != ErrorType::NO_ERROR)
		{
			std::cout << "Error saving image " << std::endl;
			assert(false);
			return;
		}
#define NO_ERROR 0l;
	}
}


inline void Frame::SetImage(const Image* image, bool changeOriginal)
{
	if (image && changeOriginal)
	{
		m_mainImageData = *image;
	}
	auto [RGB, a] = ConvertFromImage(image->m_imageData);

	auto wximage = wxImage(image->m_width, image->m_height, RGB.data(), a.data(), true);
	wximage.Rescale(ImageDimension.first, ImageDimension.second, wxImageResizeQuality::wxIMAGE_QUALITY_HIGH);
	wxBitmap bitmap(wximage);

	if (!m_staticBitmap)
	{
		m_staticBitmap = new wxStaticBitmap(m_imagePanel, wxID_ANY, bitmap);
		m_staticBitmap->Bind(wxEVT_LEFT_DOWN, &Frame::OnClickHandler, this);

		wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
		sizer->Add(m_staticBitmap, 1, wxEXPAND, 5);
		m_imagePanel->SetSizer(sizer);
		m_imagePanel->SetSize(ImageDimension.first, ImageDimension.second);
	}
	else
	{
		m_staticBitmap->SetBitmap(bitmap);
	}

	m_imagePanel->Layout();
	Refresh(true);
}

