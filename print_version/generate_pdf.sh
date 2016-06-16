convert page1.png page2.png -page a4 -resize 2480x3508 -units PixelsPerInch \
        -density 300x300 -level 50%,93% -colors 64 -repage a4 high_contrast.pdf

convert page1.png page2.png -page a4 -resize 2480x3508 -units PixelsPerInch \
        -density 300x300 -colors 64 -repage a4 low_contrast.pdf
