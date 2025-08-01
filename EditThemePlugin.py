import dearpygui.dearpygui as dpg
import configparser
import os

class EditThemePlugin():
	def __init__(self):
		self.confParser = configparser.ConfigParser()
		self._create_folders()
		self.all_saved_colors()
		self._create_UI()
  
	def _create_folders(self):
		if not os.path.exists('themes'):
			os.mkdir('themes')
		if not os.path.exists('themes/default.ini'):
			self.createDefaultTheme()
  
	def _create_UI(self):
		dpg.add_theme(tag="glob")
		self.createMenu()
		self.loadAll(None, {"file_path_name":'themes/default.ini'})

	def __str__(self) -> str:
		return "glob"

	def createTheme(self):
		with dpg.theme_component(dpg.mvAll, parent="glob"):
			self.parse_color_operation('dpg.add_theme_color(dpg.[VARGOESHERE],		self.getThemeColor("[VARGOESHERE]"),			category=dpg.mvThemeCat_Core)')
		dpg.bind_theme("glob")

	def createDefaultTheme(self):
		self.confParser['Theme'] = {}
		self.confParser['Theme']['mvthemecol_text'] = "255.0, 254.99607849121094, 254.99607849121094, 255.0"
		self.confParser['Theme']['mvthemecol_tabactive'] = "177.51373291015625, 26.478431701660156, 26.48627471923828, 255.0"
		self.confParser['Theme']['mvthemecol_slidergrabactive'] = "249.0, 66.0, 72.062744140625, 255.0"
		self.confParser['Theme']['mvthemecol_textdisabled'] = "127.0, 127.0, 127.0, 255.0"
		self.confParser['Theme']['mvthemecol_tabunfocused'] = "53.53333282470703, 22.77254867553711, 22.776470184326172, 247.0"
		self.confParser['Theme']['mvthemecol_button'] = "123.97647094726562, 0.6823529601097107, 0.6901960968971252, 255.0"
		self.confParser['Theme']['mvthemecol_windowbg'] = "12.678431510925293, 12.607843399047852, 12.607843399047852, 239.0"
		self.confParser['Theme']['mvthemecol_tabunfocusedactive'] = "107.0, 35.0, 35.00392150878906, 255.0"
		self.confParser['Theme']['mvthemecol_buttonhovered'] = "231.04705810546875, 17.870588302612305, 24.937253952026367, 255.0"
		self.confParser['Theme']['mvthemecol_childbg'] = "0.0, 0.0, 0.0, 26.764705657958984"
		self.confParser['Theme']['mvthemecol_dockingpreview'] = "249.0, 66.0, 66.00784301757812, 178.0"
		self.confParser['Theme']['mvthemecol_buttonactive'] = "249.0, 15.0, 15.011764526367188, 255.0"
		self.confParser['Theme']['mvthemecol_border'] = "109.0, 109.0, 127.0, 127.0"
		self.confParser['Theme']['mvthemecol_dockingemptybg'] = "51.0, 51.0, 51.0, 255.0"
		self.confParser['Theme']['mvthemecol_header'] = "249.0, 66.0, 66.00784301757812, 79.0"
		self.confParser['Theme']['mvthemecol_popupbg'] = "20.0, 20.0, 20.0, 239.0"
		self.confParser['Theme']['mvthemecol_plotlines'] = "155.0, 155.0, 155.0, 255.0"
		self.confParser['Theme']['mvthemecol_headerhovered'] = "249.0, 66.0, 66.00784301757812, 204.0"
		self.confParser['Theme']['mvthemecol_bordershadow'] = "0.0, 0.0, 0.0, 0.0"
		self.confParser['Theme']['mvthemecol_plotlineshovered'] = "255.0, 109.0, 89.0, 255.0"
		self.confParser['Theme']['mvthemecol_headeractive'] = "249.0, 66.0, 66.00784301757812, 255.0"
		self.confParser['Theme']['mvthemecol_framebg'] = "82.19215393066406, 83.32157135009766, 84.52941131591797, 137.0"
		self.confParser['Theme']['mvthemecol_plothistogram'] = "229.0, 178.0, 0.0, 255.0"
		self.confParser['Theme']['mvthemecol_separator'] = "109.0, 109.0, 127.0, 127.0"
		self.confParser['Theme']['mvthemecol_framebghovered'] = "249.0, 66.0, 66.00784301757812, 102.0"
		self.confParser['Theme']['mvthemecol_plothistogramhovered'] = "255.0, 153.0, 0.0, 255.0"
		self.confParser['Theme']['mvthemecol_separatorhovered'] = "191.0, 24.713726043701172, 24.713726043701172, 200.0"
		self.confParser['Theme']['mvthemecol_framebgactive'] = "255.0, 0.0, 0.0117647061124444, 195.82745361328125"
		self.confParser['Theme']['mvthemecol_tableheaderbg'] = "48.0, 48.0, 51.0, 255.0"
		self.confParser['Theme']['mvthemecol_separatoractive'] = "191.0, 25.0, 25.007843017578125, 255.0"
		self.confParser['Theme']['mvthemecol_titlebg'] = "10.0, 10.0, 10.0, 255.0"
		self.confParser['Theme']['mvthemecol_tableborderstrong'] = "79.0, 79.0, 89.0, 255.0"
		self.confParser['Theme']['mvthemecol_resizegrip'] = "249.0, 66.0, 66.00784301757812, 51.0"
		self.confParser['Theme']['mvthemecol_titlebgactive'] = "122.0, 40.0, 40.00392150878906, 255.0"
		self.confParser['Theme']['mvthemecol_tableborderlight'] = "67.67058563232422, 67.67058563232422, 76.07450866699219, 255.0"
		self.confParser['Theme']['mvthemecol_resizegriphovered'] = "249.0, 66.0, 66.00784301757812, 170.0"
		self.confParser['Theme']['mvthemecol_titlebgcollapsed'] = "0.0, 0.0, 0.0, 130.0"
		self.confParser['Theme']['mvthemecol_tablerowbg'] = "0.0, 0.0, 0.0, 0.0"
		self.confParser['Theme']['mvthemecol_resizegripactive'] = "236.68235778808594, 36.61176300048828, 36.62352752685547, 242.0"
		self.confParser['Theme']['mvthemecol_menubarbg'] = "35.0, 35.0, 35.0, 255.0"
		self.confParser['Theme']['mvthemecol_tablerowbgalt'] = "255.0, 255.0, 255.0, 15.0"
		self.confParser['Theme']['mvthemecol_tab'] = "147.0, 44.99607849121094, 45.00392150878906, 219.0"
		self.confParser['Theme']['mvthemecol_scrollbarbg'] = "5.0, 5.0, 5.0, 135.0"
		self.confParser['Theme']['mvthemecol_textselectedbg'] = "249.0, 66.0, 114.52941131591797, 89.0"
		self.confParser['Theme']['mvthemecol_tabhovered'] = "249.0, 66.0, 66.00784301757812, 204.0"
		self.confParser['Theme']['mvthemecol_scrollbargrab'] = "79.0, 79.0, 79.0, 255.0"
		self.confParser['Theme']['mvthemecol_dragdroptarget'] = "255.0, 255.0, 0.0, 229.0"
		self.confParser['Theme']['mvthemecol_scrollbargrabhovered'] = "104.0, 104.0, 104.0, 255.0"
		self.confParser['Theme']['mvthemecol_navhighlight'] = "249.0, 66.0, 66.00784301757812, 255.0"
		self.confParser['Theme']['mvthemecol_scrollbargrabactive'] = "130.0, 130.0, 130.0, 255.0"
		self.confParser['Theme']['mvthemecol_navwindowinghighlight'] = "255.0, 255.0, 255.0, 178.0"
		self.confParser['Theme']['mvthemecol_checkmark'] = "249.0, 66.0, 66.00784301757812, 255.0"
		self.confParser['Theme']['mvthemecol_navwindowingdimbg'] = "204.0, 204.0, 204.0, 51.0"
		self.confParser['Theme']['mvthemecol_slidergrab'] = "224.0, 60.99607849121094, 61.007843017578125, 255.0"
		self.confParser['Theme']['mvthemecol_modalwindowdimbg'] = "204.0, 204.0, 204.0, 89.0"
		with open('themes/default.ini', 'w') as f: 
			self.confParser.write(f, True)
		return
		#with dpg.theme() as theme_id:
		#	with dpg.theme_component(0):
		#		dpg.add_theme_color(dpg.mvThemeCol_Text                   , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TextDisabled           , (0.50 * 255, 0.50 * 255, 0.50 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_WindowBg               , (0.06 * 255, 0.06 * 255, 0.06 * 255, 0.94 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ChildBg                , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_PopupBg                , (0.08 * 255, 0.08 * 255, 0.08 * 255, 0.94 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_Border                 , (0.43 * 255, 0.43 * 255, 0.50 * 255, 0.50 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_BorderShadow           , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_FrameBg                , (0.16 * 255, 0.29 * 255, 0.48 * 255, 0.54 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered         , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.40 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive          , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.67 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TitleBg                , (0.04 * 255, 0.04 * 255, 0.04 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive          , (0.16 * 255, 0.29 * 255, 0.48 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed       , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.51 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg              , (0.14 * 255, 0.14 * 255, 0.14 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg            , (0.02 * 255, 0.02 * 255, 0.02 * 255, 0.53 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab          , (0.31 * 255, 0.31 * 255, 0.31 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered   , (0.41 * 255, 0.41 * 255, 0.41 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive    , (0.51 * 255, 0.51 * 255, 0.51 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_CheckMark              , (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_SliderGrab             , (0.24 * 255, 0.52 * 255, 0.88 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive       , (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_Button                 , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.40 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered          , (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ButtonActive           , (0.06 * 255, 0.53 * 255, 0.98 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_Header                 , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.31 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered          , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.80 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_HeaderActive           , (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_Separator              , (0.43 * 255, 0.43 * 255, 0.50 * 255, 0.50 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered       , (0.10 * 255, 0.40 * 255, 0.75 * 255, 0.78 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive        , (0.10 * 255, 0.40 * 255, 0.75 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip             , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.20 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered      , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.67 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive       , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.95 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_Tab                    , (0.18 * 255, 0.35 * 255, 0.58 * 255, 0.86 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TabHovered             , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.80 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TabActive              , (0.20 * 255, 0.41 * 255, 0.68 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused           , (0.07 * 255, 0.10 * 255, 0.15 * 255, 0.97 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive     , (0.14 * 255, 0.26 * 255, 0.42 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_DockingPreview         , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.70 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg         , (0.20 * 255, 0.20 * 255, 0.20 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_PlotLines              , (0.61 * 255, 0.61 * 255, 0.61 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_PlotLinesHovered       , (1.00 * 255, 0.43 * 255, 0.35 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram          , (0.90 * 255, 0.70 * 255, 0.00 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered   , (1.00 * 255, 0.60 * 255, 0.00 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg          , (0.19 * 255, 0.19 * 255, 0.20 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong      , (0.31 * 255, 0.31 * 255, 0.35 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight       , (0.23 * 255, 0.23 * 255, 0.25 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TableRowBg             , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt          , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.06 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg         , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.35 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_DragDropTarget         , (1.00 * 255, 1.00 * 255, 0.00 * 255, 0.90 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_NavHighlight           , (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_NavWindowingHighlight  , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.70 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_NavWindowingDimBg      , (0.80 * 255, 0.80 * 255, 0.80 * 255, 0.20 * 255))
		#		dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg       , (0.80 * 255, 0.80 * 255, 0.80 * 255, 0.35 * 255))
		#		dpg.add_theme_color(dpg.mvPlotCol_FrameBg                 , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.07 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_PlotBg                  , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.50 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_PlotBorder              , (0.43 * 255, 0.43 * 255, 0.50 * 255, 0.50 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_LegendBg                , (0.08 * 255, 0.08 * 255, 0.08 * 255, 0.94 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_LegendBorder            , (0.43 * 255, 0.43 * 255, 0.50 * 255, 0.50 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_LegendText              , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_TitleText               , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_InlayText               , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_XAxis                   , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_XAxisGrid               , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.25 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_YAxis                   , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_YAxisGrid               , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.25 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_YAxis2                  , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_YAxisGrid2              , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.25 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_YAxis3                  , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_YAxisGrid3              , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.25 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_Selection               , (1.00 * 255, 0.60 * 255, 0.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_Query                   , (0.00 * 255, 1.00 * 255, 0.44 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvPlotCol_Crosshairs              , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.50 * 255), category=dpg.mvThemeCat_Plots)
		#		dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (50, 50, 50, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, (75, 75, 75, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, (75, 75, 75, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (100, 100, 100, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (41, 74, 122, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (66, 150, 250, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (66, 150, 250, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_Link, (61, 133, 224, 200), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (66, 150, 250, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (66, 150, 250, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_Pin, (53, 150, 250, 180), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_PinHovered, (53, 150, 250, 255), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_BoxSelector, (61, 133, 224, 30), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_BoxSelectorOutline, (61, 133, 224, 150), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_GridBackground, (40, 40, 50, 200), category=dpg.mvThemeCat_Nodes)
		#		dpg.add_theme_color(dpg.mvNodeCol_GridLine, (200, 200, 200, 40), category=dpg.mvThemeCat_Nodes)
		#return theme_id

	def createMenu(self):
		pluginMenu = dpg.add_menu(label="Theme")
		self.loadTheme()
		self.saveTheme()
		self.editTheme()
		dpg.add_menu_item(label="Configure Theme", callback=lambda: dpg.configure_item('editThemeWindow', show=True), parent=pluginMenu)
		dpg.add_menu_item(label="Load Theme", callback=lambda: dpg.configure_item("loadThemeFileSelector", show=True), parent=pluginMenu)
		dpg.add_menu_item(label="Save Theme", callback=lambda: dpg.configure_item("saveThemeFileSelector", show=True), parent=pluginMenu)

	def editTheme(self):
		dpg.add_window(tag="editThemeWindow", show=False, autosize=True, no_title_bar=True, max_size=[1080,720])
		dpg.add_button(parent="editThemeWindow", label="Close", callback=lambda: dpg.configure_item("editThemeWindow", show=False))
		dpg.add_button(label="Load Theme", callback=lambda: dpg.configure_item("loadThemeFileSelector", show=True), parent="editThemeWindow")
		dpg.add_button(label="Save Theme", callback=lambda: dpg.configure_item("saveThemeFileSelector", show=True), parent="editThemeWindow")
		myEditVar = """dpg.add_color_edit(self.getThemeColor("[VARGOESHERE]"), source=self.confParser['Theme']["[VARGOESHERE]"], display_type=dpg.mvColorEdit_uint8, alpha_bar=True, alpha_preview=dpg.mvColorEdit_AlphaPreviewHalf, tag="colEdit[VARGOESHERE]", parent='editThemeWindow', label='[VARGOESHERE]', callback=lambda: dpg.add_theme_color(dpg.[VARGOESHERE], dpg.get_value("colEdit[VARGOESHERE]"), parent=dpg.add_theme_component(dpg.mvAll, parent="glob"))- 0 if not dpg.bind_theme("glob") else 0)"""
		#dpg.add_color_edit(self.getThemeColor("[VARGOESHERE]"), source=self.confParser['Theme']["[VARGOESHERE]"], display_type=dpg.mvColorEdit_uint8, alpha_bar=True, alpha_preview=dpg.mvColorEdit_AlphaPreviewHalf,tag="colEdit[VARGOESHERE]", parent='editThemeWindow', label='[VARGOESHERE]', callback=lambda: dpg.add_theme_color(dpg.[VARGOESHERE], dpg.get_value("colEdit[VARGOESHERE]"), parent=dpg.add_theme_component(dpg.mvAll, parent="glob"))- 0 if not dpg.bind_theme("glob") else 0)
		self.parse_color_operation(myEditVar)
	
	def loadAll(self, a=None, sentFileDict=None):
		try:
			#self.confParser = configparser.ConfigParser()
			self.confParser.read(sentFileDict["file_path_name"])
			dpg.delete_item("editThemeWindow")
			self.createTheme()
			self.editTheme()
			#with open("newFile.ini", 'w') as f:			self.confParser.write(f)
		except:
			return

		
	
	def loadTheme(self):
		try:
			with dpg.file_dialog(callback=self.loadAll, directory_selector=False, width=700, height=400, default_path="themes", default_filename="default.ini", show=False, tag="loadThemeFileSelector", cancel_callback=doNothing):
				dpg.add_file_extension(".ini")
			pass
		except:
			return
	
	def saveTheme(self):
		try:
			with dpg.file_dialog(default_path="themes", default_filename=".ini",callback=self.saveAll, directory_selector=False, width=700, height=400, show=False, tag="saveThemeFileSelector", cancel_callback=doNothing):
				dpg.add_file_extension(".ini")
				dpg.add_button(label="save")
		except:
			return
		pass
		return

	def saveAll(self, a, b):
		#print(f"{self}, {a}, {b}")
		if not "file_path_name" in b:
			return
		try:
			#self.confParser['Theme'] = {}
			#print(a, b, c)
			saveConf = "self.confParser['Theme']['[VARGOESHERE]']=str(dpg.get_value('colEdit[VARGOESHERE]'))"
			self.parse_color_operation(saveConf)
			with open(b["file_path_name"], 'w') as f: self.confParser.write(f,True)
		except:
			return

	def getThemeColor(self, themeCol):
		itm = self.confParser['Theme'][themeCol]
		#print(type(itm))
		itm = itm.removeprefix('[')
		itm = itm.removesuffix(']')
		thrIntAsStr = itm
		#print(thrIntAsStr)
		a = thrIntAsStr.split(',')
		#print(a[0])
		#print((int(float(a[0].strip())),int(float(a[1].strip())),int(float(a[2].strip())),int(float(a[3].strip()))))
		return (int(float(a[0].strip())),int(float(a[1].strip())),int(float(a[2].strip())),int(float(a[3].strip())))
		#c = dict()
		#i = 0
		#for a in thrIntAsStr.split(','):
		#	b = a.split(' * ')
		#	#print(b)
		#	c[str(i)] = float(b[0].strip()) * float(b[1].strip())
		#	i += 1
		##print(c)
		#self.confParser['Theme'][themeCol] = f'{int(c["0"])},{int(c["1"])},{int(c["2"])},{int(c["3"])}'
		#return (int(c["0"]),int(c["1"]),int(c["2"]),int(c["3"]))
		#self.confParser['Theme']['mvThemeCol_ModalWindowDimBg']=dpg.get_value('colEditmvThemeCol_ModalWindowDimBg')


	def parse_color_operation(self, operation: str):
		"""Write operation like this:

		Args:
			operation (str): 'dpg.add_theme_color(dpg.[VARGOESHERE],		self.getThemeColor("[VARGOESHERE]"),			category=dpg.mvThemeCat_Core)'
		"""
		for c in self.theColors:
			newVar = operation
			newVar = newVar.replace("[VARGOESHERE]", c)
			try:
				#print(newVar)
				exec(newVar)
			except:
				#exec(f"#print(dpg.{c})")
				pass##print(newVar)
		return


	def all_saved_colors(self):
		self.theColors = [
			"mvThemeCol_Text",
			"mvThemeCol_TextSelectedBg",
			"mvThemeCol_TextDisabled",
			"mvThemeCol_TabActive",
			"mvThemeCol_TabUnfocused",
			"mvThemeCol_TabUnfocusedActive",
			"mvThemeCol_TabHovered",
			"mvThemeCol_Tab",
			"mvThemeCol_Button",
			"mvThemeCol_ButtonHovered",
			"mvThemeCol_ButtonActive",
			"mvThemeCol_WindowBg",
			"mvThemeCol_ChildBg",
			"mvThemeCol_PopupBg",
			"mvThemeCol_FrameBg",
			"mvThemeCol_FrameBgHovered",
			"mvThemeCol_FrameBgActive",
			"mvThemeCol_TitleBg",
			"mvThemeCol_TitleBgActive",
			"mvThemeCol_TitleBgCollapsed",
			"mvThemeCol_MenuBarBg",
			"mvThemeCol_DockingEmptyBg",
			"mvThemeCol_ScrollbarBg",
			"mvThemeCol_ResizeGripActive",
			"mvThemeCol_ScrollbarGrab",
			"mvThemeCol_ScrollbarGrabHovered",
			"mvThemeCol_ScrollbarGrabActive",
			"mvThemeCol_Border",
			"mvThemeCol_BorderShadow",
			"mvThemeCol_SliderGrabActive",
			"mvThemeCol_DockingPreview",
			"mvThemeCol_Header",
			"mvThemeCol_PlotLines",
			"mvThemeCol_HeaderHovered",
			"mvThemeCol_PlotLinesHovered",
			"mvThemeCol_HeaderActive",
			"mvThemeCol_PlotHistogram",
			"mvThemeCol_Separator",
			"mvThemeCol_PlotHistogramHovered",
			"mvThemeCol_SeparatorHovered",
			"mvThemeCol_TableHeaderBg",
			"mvThemeCol_SeparatorActive",
			"mvThemeCol_TableBorderStrong",
			"mvThemeCol_ResizeGrip",
			"mvThemeCol_TableBorderLight",
			"mvThemeCol_ResizeGripHovered",
			"mvThemeCol_TableRowBg",
			"mvThemeCol_TableRowBgAlt",
			"mvThemeCol_DragDropTarget",
			"mvThemeCol_NavHighlight",
			"mvThemeCol_NavWindowingHighlight",
			"mvThemeCol_CheckMark",
			"mvThemeCol_NavWindowingDimBg",
			"mvThemeCol_SliderGrab",
			"mvThemeCol_ModalWindowDimBg",
		]

def doNothing(*args):
		return

if __name__=="__main__":
	dpg.create_context()
	dpg.create_viewport(title='Custom Title', width=1200, height=800)
	with dpg.theme() as global_theme:
		with dpg.theme_component(dpg.mvAll) as gtc:
			dpg.add_theme_color(dpg.mvThemeCol_Text,					(0,0,0,255),						category=dpg.mvThemeCat_Core, tag="globcolor")
	dpg.bind_theme(global_theme)
	#dpg.add_theme(tag="glob")
	def bind():dpg.add_theme_color(dpg.mvThemeCol_Text, dpg.get_value("ve"), parent=dpg.add_theme_component(dpg.mvAll, parent="glob"));dpg.bind_theme("glob")
	with dpg.window(tag="main", show=False):
		dpg.add_color_edit(parent="main",tag='ve', callback=bind)
		dpg.add_text(dpg.get_item_info("main"), wrap=0)
		dpg.add_text(dpg.get_app_configuration(), wrap=0)
	with dpg.value_registry():
		dpg.add_color_value(source="ve", tag="vete")
	with dpg.window(tag="main2"):
		with dpg.child_window():
			dpg.add_text("This is text")
			dpg.add_button(tag="This is a button", label="THIS IS A BUTTON")
			dpg.add_checkbox(label="Check Box")
			with dpg.child_window(autosize_x=True, autosize_y=True):
				with dpg.tab_bar():
					with dpg.tab(label="THIS IS A TAB"):
						with dpg.tree_node(label="THIS IS A TREE NODE"):
							randListOfStuff = ['THIS', 'IS', 'A', 'LIST']
							dpg.add_combo(randListOfStuff)
							dpg.add_listbox(randListOfStuff)

	with dpg.viewport_menu_bar():
		with dpg.menu(label="Tools"):
			dpg.add_menu_item(label="Show About", 			callback=lambda:dpg.show_tool(dpg.mvTool_About))
			dpg.add_menu_item(label="Show Metrics", 		callback=lambda:dpg.show_tool(dpg.mvTool_Metrics))
			dpg.add_menu_item(label="Show Documentation", 	callback=lambda:dpg.show_tool(dpg.mvTool_Doc))
			dpg.add_menu_item(label="Show Debug", 			callback=lambda:dpg.show_tool(dpg.mvTool_Debug))
			dpg.add_menu_item(label="Show Style Editor", 	callback=lambda:dpg.show_tool(dpg.mvTool_Style))
			dpg.add_menu_item(label="Show Font Manager", 	callback=lambda:dpg.show_tool(dpg.mvTool_Font))
			dpg.add_menu_item(label="Show Item Registry", 	callback=lambda:dpg.show_tool(dpg.mvTool_ItemRegistry))
		EditThemePlugin()
	dpg.set_primary_window("main2", True)
	dpg.setup_dearpygui()

	dpg.show_viewport()
	dpg.start_dearpygui()
	dpg.destroy_context()