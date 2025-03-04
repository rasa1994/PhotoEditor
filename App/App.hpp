import <wx/wxprec.h>;

#ifndef WX_PRECOMP
import <wx/wx.h>;
#endif // !WX_PRECOMP

#include "Frame.hpp"

class App : public wxApp
{
public:
	virtual bool OnInit() override;
	static void Run(int argc, char** argv)
	{
		wxApp::SetInstance(new App());
		auto instance = wxApp::GetInstance();
		wxEntryStart(argc, argv);
		instance->CallOnInit();
		instance->OnRun();
		instance->OnExit();
		wxEntryCleanup();
	}

};

bool App::OnInit()
{
	Frame* frame = new Frame();
	frame->Show(true);
	return true;
}