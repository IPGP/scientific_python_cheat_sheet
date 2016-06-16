convert page1.png page2.png -page a4 -level 50%,93% -resize 2480x3506\
                            -units PixelsPerInch -density 300x300 high_contrast.pdf

convert page1.png page2.png -page a4 -resize 2480x3506\
                            -units PixelsPerInch -density 300x300 low_contrast.pdf
