import matplotlib.font_manager

# システムで見つかったすべてのフォントのリストを取得
font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

print("Matplotlibが認識しているフォント名の中から 'Noto Sans CJK' を含むものを探します:")
for font_path in font_list:
    try:
        font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
#        if 'nS cjk jp' in font_name.lower() or 'noto sans cjk jp' in font_name.lower(): # 検索文字列を調整
        print(f"  Font Name: {font_name}, Path: {font_path}")
    except RuntimeError:
        # 一部のフォントでエラーが発生する可能性があるためスキップ
        pass
    except Exception as e:
        print(f"Error processing {font_path}: {e}")

# より直接的にFontManagerのリストを確認
print("\nMatplotlibのFontManagerが管理するフォントリストから 'Noto Sans CJK JP' を探します:")
available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
for name in available_fonts:
    if 'nS cjk jp' in name.lower() or 'noto sans cjk jp' in name.lower(): # 検索文字列を調整
         print(f"  Available Font Name: {name}")